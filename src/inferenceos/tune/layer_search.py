"""Choose how many model layers go on the graphics card (pure, no model loading).

llama.cpp puts the **last** N layers on the card (``-ngl N``). For a model of
``n_layers`` layers, count N means layers ``n_layers - N ... n_layers - 1`` are on the
card, so the first card layer is ``n_layers - N``.

Nothing here loads a model or reads a file: the caller injects ``measure`` (one test of
one count) and ``should_stop`` (the time limit). The search only needs the counts, the
optional per-layer kinds and those two callables.

Design (tested against recorded studies):

1. Group the counts (1 ... ``top`` - 1) by the kind of their first card layer, or put
   them all in one group when no kinds are known. Count 0 and count ``top`` are always
   measured and are single-member groups.
2. Measure each group's points at positions 0, ``spacing``, 2*``spacing``, ... and the
   last position, plus counts 0 and ``top``.
3. Fill in: for each group, lift a lone slow point to its neighbours' level, then split
   any gap between two measured neighbours whose level is close to the best.
4. Race: re-measure the best few counts and drop the clearly worse ones. Stop only once
   the rest measure near-equal **and** the noise is small enough to trust that (a single
   test wobbles by a few percent, so "they measure equal" alone is not enough). The
   chosen count is the smallest one within the tight ``tie_margin`` of the best.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from inferenceos.models.gguf import read_gguf

__all__ = [
    "LayerSearchResult",
    "LayerSearchSettings",
    "choose_winner",
    "full_layer_kinds_from_gguf",
    "group_counts",
    "layer_count_from_gguf",
    "layer_kinds_from_gguf",
    "lift_lone_dips",
    "race_candidates",
    "run_race",
    "score",
    "search_layers",
    "trimmed_mean",
]


@dataclass(frozen=True)
class LayerSearchSettings:
    """Tuning for the layer search. The defaults are the tested design."""

    spacing: int = 3
    keep: float = 0.09
    race_window: float = 0.06
    race_cap: int = 4
    race_min_rounds: int = 3
    race_max_rounds: int = 8
    race_z: float = 2.5
    race_floor: float = 0.015
    race_stop_spread: float = 0.015
    unstable_spread: float = 0.10
    unstable_min_readings: int = 3
    tie_margin: float = 0.005


@dataclass
class LayerSearchResult:
    """What the search decided, and how."""

    chosen: int | None
    readings: dict[int, list[float]]
    tests: int
    complete: bool
    not_measured: list[int]
    failed: list[int]
    reason: str


def trimmed_mean(runs: Sequence[float]) -> float:
    """Drop runs below 85% of the median, then average what is left."""
    values = [float(value) for value in runs]
    if not values:
        raise ValueError("trimmed_mean needs at least one run")
    threshold = 0.85 * statistics.median(values)
    kept = [value for value in values if value >= threshold]
    return statistics.fmean(kept if kept else values)


def score(readings: Sequence[float], settings: LayerSearchSettings | None = None) -> float:
    """The value of a count: its median, or the median of its lower half when jumpy."""
    policy = settings or LayerSearchSettings()
    values = [float(value) for value in readings]
    if not values:
        raise ValueError("score needs at least one reading")
    if len(values) >= policy.unstable_min_readings:
        high = max(values)
        low = min(values)
        if high > 0 and (high - low) / high > policy.unstable_spread:
            ordered = sorted(values)
            lower = ordered[: max(1, len(ordered) // 2)]
            return statistics.median(lower)
    return statistics.median(values)


class _Session:
    """Tracks readings, failures and the test count while measuring counts."""

    def __init__(
        self,
        measure: Callable[[int], float | None],
        should_stop: Callable[[], bool],
        settings: LayerSearchSettings,
    ) -> None:
        self._measure = measure
        self._should_stop = should_stop
        self.settings = settings
        self.readings: dict[int, list[float]] = {}
        self.failed: set[int] = set()
        self.tests = 0
        self.stopped = False

    def measure_once(self, count: int) -> bool:
        """Measure ``count`` once. True when a reading was added."""
        if count in self.failed:
            return False
        if self._should_stop():
            self.stopped = True
            return False
        self.tests += 1
        value = self._measure(count)
        if value is None:
            self.failed.add(count)
            return False
        self.readings.setdefault(count, []).append(float(value))
        return True

    def score(self, count: int) -> float:
        return score(self.readings[count], self.settings)

    def best_count(self) -> int | None:
        measured = list(self.readings)
        if not measured:
            return None
        best = max(self.score(count) for count in measured)
        qualified = [
            count
            for count in measured
            if self.score(count) >= best * (1 - self.settings.tie_margin)
        ]
        return min(qualified)


def _first_positions(length: int, spacing: int) -> list[int]:
    if length <= 0:
        return []
    step = max(1, spacing)
    positions: set[int] = set()
    index = 0
    while index < length:
        positions.add(index)
        index += step
    positions.add(length - 1)
    return sorted(positions)


def lift_lone_dips(values: Sequence[float]) -> list[float]:
    """Raise a lone slow interior value to its neighbours' level."""
    lifted = [float(value) for value in values]
    for index in range(1, len(lifted) - 1):
        lifted[index] = max(lifted[index], min(lifted[index - 1], lifted[index + 1]))
    return lifted


def group_counts(
    top: int, n_layers: int, layer_kinds: Sequence[str] | None
) -> list[list[int]]:
    groups: list[list[int]] = [[0]]
    if top != 0:
        groups.append([top])
    if layer_kinds is None:
        middle = list(range(1, top))
        if middle:
            groups.append(middle)
    else:
        by_kind: dict[str, list[int]] = {}
        for count in range(1, top):
            index = n_layers - count
            kind = layer_kinds[index] if 0 <= index < len(layer_kinds) else "unknown"
            by_kind.setdefault(kind, []).append(count)
        for kind in sorted(by_kind):
            groups.append(by_kind[kind])
    return groups


def _measure_order(top: int, points: Sequence[int]) -> list[int]:
    unique = sorted(set(points))
    order: list[int] = []
    if 0 in unique:
        order.append(0)
    if top != 0 and top in unique:
        order.append(top)
    order.extend(count for count in unique if count != 0 and count != top)
    return order


def race_candidates(
    measured: Sequence[int],
    score_of: Callable[[int], float],
    settings: LayerSearchSettings,
) -> list[int]:
    """Step 4 candidate list: the best few counts within the race window."""
    if not measured:
        return []
    best = max(score_of(count) for count in measured)
    qualified = [
        count
        for count in measured
        if score_of(count) >= best * (1 - settings.race_window)
    ]
    qualified.sort(key=lambda count: (-score_of(count), count))
    return qualified[: settings.race_cap]


def run_race(
    candidates: Sequence[int],
    *,
    readings: MutableMapping[int, list[float]],
    measure_once: Callable[[int], bool],
    score_of: Callable[[int], float],
    settings: LayerSearchSettings | None = None,
) -> list[int]:
    """Step 4: re-measure the candidates and drop clearly worse ones."""
    policy = settings or LayerSearchSettings()
    remaining = list(candidates)
    if not remaining:
        return []
    rounds = 0
    while rounds < policy.race_max_rounds:
        if len(remaining) <= 1 and rounds >= policy.race_min_rounds:
            break
        offset = rounds % len(remaining)
        order = remaining[offset:] + remaining[:offset]
        for count in order:
            measure_once(count)
        rounds += 1

        ratios: list[float] = []
        for count in remaining:
            values = readings.get(count, [])
            if not values:
                continue
            median = statistics.median(values)
            if median <= 0:
                continue
            ratios.extend(abs(value / median - 1.0) for value in values)
        sd = max(0.01, 1.4826 * statistics.median(ratios)) if ratios else 0.01

        best = max(score_of(count) for count in remaining)
        kept: list[int] = []
        for count in remaining:
            sample = max(1, len(readings.get(count, [])))
            tolerance = max(policy.race_floor, policy.race_z * sd * math.sqrt(2 / sample))
            if score_of(count) >= best * (1 - tolerance):
                kept.append(count)
        if not kept:
            kept = [max(remaining, key=score_of)]
        remaining = kept

        best = max(score_of(count) for count in remaining)
        if rounds >= policy.race_min_rounds:
            sample_sizes = [len(readings.get(count, [])) for count in remaining]
            n_min = min(sample_sizes) if sample_sizes else 1
            confident = (
                policy.race_z * sd * math.sqrt(2 / max(1, n_min))
                <= 2 * policy.race_stop_spread
            )
            near_equal = all(
                score_of(count) >= best * (1 - policy.race_stop_spread) for count in remaining
            )
            if confident and near_equal:
                break
    return remaining


def choose_winner(
    candidates: Sequence[int],
    score_of: Callable[[int], float],
    settings: LayerSearchSettings | None = None,
) -> int | None:
    """The smallest candidate within ``tie_margin`` of the best remaining score."""
    policy = settings or LayerSearchSettings()
    if not candidates:
        return None
    best = max(score_of(count) for count in candidates)
    qualified = [
        count
        for count in candidates
        if score_of(count) >= best * (1 - policy.tie_margin)
    ]
    return min(qualified)


def search_layers(
    *,
    top: int,
    n_layers: int,
    layer_kinds: Sequence[str] | None,
    measure: Callable[[int], float | None],
    should_stop: Callable[[], bool],
    settings: LayerSearchSettings | None = None,
) -> LayerSearchResult:
    """Run the four-step layer search and return the chosen count."""
    policy = settings or LayerSearchSettings()
    session = _Session(measure, should_stop, policy)
    groups = group_counts(top, n_layers, layer_kinds)

    # Step 2: first tests.
    first_points: list[int] = []
    for group in groups:
        first_points.extend(group[position] for position in _first_positions(len(group), policy.spacing))
    for count in _measure_order(top, first_points):
        session.measure_once(count)
        if session.stopped:
            break

    # Step 3: fill in.
    while not session.stopped and session.readings:
        changed = False
        best = max(session.score(count) for count in session.readings)
        for group in groups:
            measured_positions = [
                position
                for position, count in enumerate(group)
                if count in session.readings
            ]
            if len(measured_positions) < 2:
                continue
            values = lift_lone_dips(
                [session.score(group[position]) for position in measured_positions]
            )
            for index in range(len(measured_positions) - 1):
                left = measured_positions[index]
                right = measured_positions[index + 1]
                if right - left <= 1:
                    continue
                if max(values[index], values[index + 1]) < best * (1 - policy.keep):
                    continue
                midpoint = group[(left + right) // 2]
                if midpoint in session.failed:
                    continue
                tests_before = session.tests
                session.measure_once(midpoint)
                if session.tests != tests_before:
                    changed = True
                if session.stopped:
                    break
            if session.stopped:
                break
        if not changed:
            break

    # Step 4: race.
    chosen: int | None = None
    if not session.stopped and session.readings:
        candidates = race_candidates(sorted(session.readings), session.score, policy)
        remaining = run_race(
            candidates,
            readings=session.readings,
            measure_once=session.measure_once,
            score_of=session.score,
            settings=policy,
        )
        chosen = choose_winner(remaining, session.score, policy)

    if session.stopped:
        chosen = session.best_count()
        not_measured = [
            count
            for count in range(0, top + 1)
            if count not in session.readings and count not in session.failed
        ]
        reason = (
            "the time limit was reached, so the search was cut short and is not sure; "
            f"chose the best count measured so far ({chosen})"
        )
        return LayerSearchResult(
            chosen=chosen,
            readings=dict(session.readings),
            tests=session.tests,
            complete=False,
            not_measured=not_measured,
            failed=sorted(session.failed),
            reason=reason,
        )

    if chosen is None:
        chosen = session.best_count()
    reason = (
        f"chose {chosen} layer(s) after {session.tests} test(s)"
        if chosen is not None
        else "no count could be measured"
    )
    return LayerSearchResult(
        chosen=chosen,
        readings=dict(session.readings),
        tests=session.tests,
        complete=True,
        not_measured=[],
        failed=sorted(session.failed),
        reason=reason,
    )


def layer_count_from_gguf(path: Path | str) -> int | None:
    """The number of ``blk.<i>.`` blocks in a GGUF file, or ``None``."""
    gguf = read_gguf(path)
    indices: set[int] = set()
    for tensor in gguf.tensors:
        name = tensor.name
        if not name.startswith("blk."):
            continue
        rest = name[len("blk.") :]
        dot = rest.find(".")
        if dot > 0 and rest[:dot].isdigit():
            indices.add(int(rest[:dot]))
    return max(indices) + 1 if indices else None


def full_layer_kinds_from_gguf(path: Path | str) -> list[str] | None:
    """Per-layer kind from tensor names, without collapsing equal kinds.

    For each ``blk.<i>.`` block the kind is the sorted, ``|``-joined set of tensor-name
    suffixes after ``blk.<i>.`` with ``.weight`` and ``.bias`` removed.
    """
    gguf = read_gguf(path)
    blocks: dict[int, list[str]] = {}
    for tensor in gguf.tensors:
        name = tensor.name
        if not name.startswith("blk."):
            continue
        rest = name[len("blk.") :]
        dot = rest.find(".")
        if dot <= 0 or not rest[:dot].isdigit():
            continue
        suffix = rest[dot + 1 :]
        for tail in (".weight", ".bias"):
            if suffix.endswith(tail):
                suffix = suffix[: -len(tail)]
                break
        blocks.setdefault(int(rest[:dot]), []).append(suffix)
    if not blocks:
        return None
    kinds: list[str] = []
    for index in range(max(blocks) + 1):
        parts = blocks.get(index)
        kinds.append("|".join(sorted(set(parts))) if parts else "")
    return kinds


def layer_kinds_from_gguf(path: Path | str) -> list[str] | None:
    """Per-layer kind, or ``None`` when every layer has the same kind."""
    kinds = full_layer_kinds_from_gguf(path)
    if kinds is None:
        return None
    return None if len(set(kinds)) == 1 else kinds
