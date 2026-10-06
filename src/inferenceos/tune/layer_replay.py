"""Replay the layer search against recorded studies (pure logic, no measuring).

Builds "curves" (count -> pass -> run speeds) from the per-model JSON files written by
``scripts/check_layer_range.py``, makes deterministic variants of them, simulates a
noisy test of a count, and compares the ``smart`` search with a plain "every count"
baseline. Everything is seeded so a run repeats exactly.

The "truth" of a count is the safe value the search aims at: if its passes disagree by
more than 10% (it reads very differently depending on what ran before) the truth is the
lowest pass, otherwise the average of the passes.
"""

from __future__ import annotations

import json
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from inferenceos.tune.placement import card_kinds, max_count
from inferenceos.tune.layer_search import (
    LayerSearchSettings,
    choose_winner,
    full_layer_kinds_from_gguf,
    layer_count_from_gguf,
    layer_kinds_from_gguf,
    race_candidates,
    run_race,
    score,
    search_layers,
    trimmed_mean,
)

__all__ = [
    "Curve",
    "CurveResult",
    "DesignMetrics",
    "ReplayResult",
    "best_truth",
    "build_curves",
    "curve_truth",
    "full_layer_kinds_from_gguf",
    "is_jumpy",
    "render_report",
    "run_replay",
    "write_outputs",
]

PassRuns = dict[int, list[float]]
Counts = dict[int, PassRuns]
RunsPerTest = 5
NoiseSigma = 0.01
TruthTolerance = 0.03


@dataclass(frozen=True)
class Curve:
    """One measured curve (a model, or a variant of it)."""

    label: str
    variant: str
    counts: Counts
    top: int
    n_layers: int
    layer_kinds: list[str] | None


@dataclass(frozen=True)
class DesignMetrics:
    avg_tests: float
    within3: float
    worst_gap: float
    consistency: float
    repeats: int


@dataclass(frozen=True)
class CurveResult:
    label: str
    variant: str
    top: int
    n_layers: int
    smart: DesignMetrics
    every: DesignMetrics


@dataclass
class ReplayResult:
    study_folder: str
    repeats: int
    seed: int
    quick: bool
    curves: list[CurveResult] = field(default_factory=list)
    totals: dict[str, dict[str, float]] = field(default_factory=dict)
    passes: list[tuple[str, bool, str]] = field(default_factory=list)
    layer_kinds: dict[str, tuple[int, str] | None] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Building curves                                                             #
# --------------------------------------------------------------------------- #
def real_curve_from_report(report: Mapping[str, object]) -> Counts:
    """count -> pass -> run speeds, from successful runs only."""
    runs = report.get("runs")
    counts: Counts = {}
    if not isinstance(runs, list):
        return counts
    for run in runs:
        if not isinstance(run, dict) or not run.get("ok"):
            continue
        speeds = run.get("writing_speed_runs")
        count = run.get("layers")
        pass_no = run.get("pass")
        if not isinstance(speeds, list) or not speeds:
            continue
        if not isinstance(count, int) or not isinstance(pass_no, int):
            continue
        counts.setdefault(count, {})[pass_no] = [float(value) for value in speeds]
    return counts


def _pass_values(counts: Counts, count: int) -> list[float]:
    passes = counts.get(count)
    if not passes:
        return []
    return [trimmed_mean(runs) for runs in passes.values() if runs]


def is_jumpy(counts: Counts, count: int) -> bool:
    """True when a count's passes disagree by more than 10%.

    Such a count reads very differently depending on what ran before, so it is judged
    by its lowest pass rather than by an average that never actually happens.
    """
    values = _pass_values(counts, count)
    if len(values) < 2:
        return False
    high = max(values)
    return high > 0 and (high - min(values)) / high > 0.10


def curve_truth(counts: Counts, count: int) -> float | None:
    """The safe value of a count: its lowest pass when jumpy, else the average."""
    values = _pass_values(counts, count)
    if not values:
        return None
    if is_jumpy(counts, count):
        return min(values)
    return statistics.fmean(values)


def best_truth(counts: Counts) -> float | None:
    best: float | None = None
    for count in counts:
        value = curve_truth(counts, count)
        if value is not None and (best is None or value > best):
            best = value
    return best


def _reversed(counts: Counts, top: int) -> Counts:
    return {top - count: {p: list(runs) for p, runs in passes.items()} for count, passes in counts.items()}


def _cut(counts: Counts, cut: int) -> Counts:
    return {count: {p: list(runs) for p, runs in passes.items()} for count, passes in counts.items() if count <= cut}


def _doubled(counts: Counts) -> Counts:
    result: Counts = {}
    for count in sorted(counts):
        result[2 * count] = {p: list(runs) for p, runs in counts[count].items()}
        source = count
        neighbour = counts.get(count + 1)
        if neighbour is not None:
            here = curve_truth(counts, count)
            there = curve_truth(counts, count + 1)
            if there is not None and (here is None or there > here):
                source = count + 1
        result[2 * count + 1] = {p: list(runs) for p, runs in counts[source].items()}
    return result


def _hill(counts: Counts, top: int, width: int, centre: int) -> Counts:
    best = best_truth(counts) or 0.0
    half = width // 2
    result: Counts = {count: {p: list(runs) for p, runs in passes.items()} for count, passes in counts.items()}
    for count in range(max(0, centre - half), min(top, centre + half) + 1):
        if count not in counts:
            continue
        here = curve_truth(counts, count)
        if not here:
            continue
        target = best * 1.05 * (1 - 0.02 * abs(count - centre))
        factor = target / here
        result[count] = {
            p: [value * factor for value in runs] for p, runs in counts[count].items()
        }
    return result


def _model_kinds(path: object) -> tuple[list[str] | None, int | None]:
    if not isinstance(path, str):
        return None, None
    file = Path(path)
    if not file.is_file():
        return None, None
    try:
        blocks = full_layer_kinds_from_gguf(file)
        n_layers = layer_count_from_gguf(file)
    except Exception:  # noqa: BLE001 - a bad file just means no kinds
        return None, None
    if n_layers is None:
        return None, None
    # The kind of the piece each count adds, output step included (tune/placement.py);
    # the count that puts everything on the card is one more than the blocks.
    return card_kinds(blocks, n_layers), max_count(n_layers)


def build_curves(
    label: str, report: Mapping[str, object], rng: random.Random, quick: bool = False
) -> list[Curve]:
    """Build the real curve and its variants for one model report."""
    counts = real_curve_from_report(report)
    if not counts:
        return []
    top = max(counts)
    kinds, file_layers = _model_kinds(report.get("model"))
    n_layers = file_layers if file_layers is not None else top
    curves = [Curve(label, "real", counts, top, n_layers, kinds)]
    if quick:
        return curves

    curves.append(Curve(label, "reversed", _reversed(counts, top), top, top, None))
    for fraction in (0.5, 0.7, 0.85):
        cut = int(top * fraction)
        if cut >= 4:
            curves.append(
                Curve(label, f"cut-{fraction:g}", _cut(counts, cut), cut, n_layers, kinds)
            )
    doubled = _doubled(counts)
    curves.append(Curve(label, "doubled", doubled, max(doubled) if doubled else top, top, None))
    if top >= 4:
        for width in (3, 5, 7):
            half = width // 2
            candidates = [
                centre
                for centre in range(2, top - 1)
                if not any(
                    is_jumpy(counts, count)
                    for count in range(max(0, centre - half), min(top, centre + half) + 1)
                )
            ]
            if not candidates:
                continue
            centre = rng.choice(candidates)
            curves.append(Curve(label, f"hill-{width}", _hill(counts, top, width, centre), top, top, None))
    return curves


# --------------------------------------------------------------------------- #
# Simulated measuring                                                         #
# --------------------------------------------------------------------------- #
def simulate_test(counts: Counts, count: int, rng: random.Random) -> float | None:
    """One noisy test of ``count``: a random pass, five runs drawn, trimmed mean."""
    passes = counts.get(count)
    if not passes:
        return None
    pass_runs = list(passes.values())
    chosen = pass_runs[rng.randrange(len(pass_runs))]
    if not chosen:
        return None
    sampled = [chosen[rng.randrange(len(chosen))] for _ in range(RunsPerTest)]
    return trimmed_mean(sampled) * (1 + rng.gauss(0.0, NoiseSigma))


def _design_rng(seed: int, label: str, variant: str, design: str) -> random.Random:
    return random.Random(f"{seed}:{label}:{variant}:{design}")


def _run_smart_once(
    curve: Curve, rng: random.Random, settings: LayerSearchSettings
) -> tuple[int | None, int]:
    def measure(count: int) -> float | None:
        return simulate_test(curve.counts, count, rng)

    result = search_layers(
        top=curve.top,
        n_layers=curve.n_layers,
        layer_kinds=curve.layer_kinds,
        measure=measure,
        should_stop=lambda: False,
        settings=settings,
    )
    return result.chosen, result.tests


def _run_every_once(
    curve: Curve, rng: random.Random, settings: LayerSearchSettings
) -> tuple[int | None, int]:
    readings: dict[int, list[float]] = {}
    tests = 0

    def measure_once(count: int) -> bool:
        nonlocal tests
        tests += 1
        value = simulate_test(curve.counts, count, rng)
        if value is None:
            return False
        readings.setdefault(count, []).append(value)
        return True

    for count in range(0, curve.top + 1):
        measure_once(count)

    def score_of(count: int) -> float:
        return score(readings[count], settings)

    candidates = race_candidates(sorted(readings), score_of, settings)
    remaining = run_race(
        candidates,
        readings=readings,
        measure_once=measure_once,
        score_of=score_of,
        settings=settings,
    )
    return choose_winner(remaining, score_of, settings), tests


# --------------------------------------------------------------------------- #
# Scoring designs                                                             #
# --------------------------------------------------------------------------- #
def _consistency(truths: Sequence[float | None]) -> float:
    if len(truths) < 2:
        return 0.0
    ok = 0
    for left, right in zip(truths, truths[1:]):
        if left is None or right is None:
            continue
        high = max(left, right)
        if high > 0 and abs(left - right) / high <= 0.01:
            ok += 1
    return ok / (len(truths) - 1)


def _evaluate(
    curve: Curve,
    design: str,
    repeats: int,
    seed: int,
    settings: LayerSearchSettings,
) -> DesignMetrics:
    rng = _design_rng(seed, curve.label, curve.variant, design)
    picks: list[int | None] = []
    test_counts: list[int] = []
    for _ in range(repeats):
        if design == "smart":
            chosen, tests = _run_smart_once(curve, rng, settings)
        else:
            chosen, tests = _run_every_once(curve, rng, settings)
        picks.append(chosen)
        test_counts.append(tests)

    best = best_truth(curve.counts) or 0.0
    truths = [curve_truth(curve.counts, pick) if pick is not None else None for pick in picks]
    if best > 0:
        within3 = sum(1 for value in truths if value is not None and value >= best * (1 - TruthTolerance)) / repeats
        worst_gap = max((best - value) / best if value is not None else 1.0 for value in truths)
    else:
        within3 = 0.0
        worst_gap = 1.0
    return DesignMetrics(
        avg_tests=statistics.fmean(test_counts) if test_counts else 0.0,
        within3=within3,
        worst_gap=worst_gap,
        consistency=_consistency(truths),
        repeats=repeats,
    )


def _average(results: Sequence[CurveResult], attribute: str) -> dict[str, float]:
    if not results:
        return {"avg_tests": 0.0, "within3": 0.0, "worst_gap": 0.0, "consistency": 0.0}
    return {
        "avg_tests": statistics.fmean(getattr(result, attribute).avg_tests for result in results),
        "within3": statistics.fmean(getattr(result, attribute).within3 for result in results),
        "worst_gap": statistics.fmean(getattr(result, attribute).worst_gap for result in results),
        "consistency": statistics.fmean(getattr(result, attribute).consistency for result in results),
    }


def _layer_kinds_summary(path: object) -> tuple[int, str] | None:
    if not isinstance(path, str):
        return None
    file = Path(path)
    if not file.is_file():
        return None
    try:
        kinds = full_layer_kinds_from_gguf(file)
    except Exception:  # noqa: BLE001
        return None
    if not kinds:
        return None
    return len(set(kinds)), _describe_kinds(kinds)


def _describe_kinds(kinds: Sequence[str]) -> str:
    ranges: list[str] = []
    start = 0
    for index in range(1, len(kinds) + 1):
        if index == len(kinds) or kinds[index] != kinds[start]:
            end = index - 1
            label = f"{start}-{end}" if end > start else f"{start}"
            ranges.append(f"{label}: {kinds[start]}")
            start = index
    return "; ".join(ranges)


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def run_replay(
    study_folder: Path | str,
    *,
    repeats: int = 300,
    seed: int = 1,
    quick: bool = False,
) -> ReplayResult:
    """Build curves from the study folder and compare the two designs."""
    folder = Path(study_folder)
    if quick:
        repeats = min(repeats, 50)
    settings = LayerSearchSettings()
    curves: list[Curve] = []
    kinds: dict[str, tuple[int, str] | None] = {}
    for path in sorted(folder.glob("*.json")):
        if path.name in {"study.json", "replay.json"}:
            continue
        report = _load_json(path)
        if report is None or not isinstance(report.get("runs"), list):
            continue
        label = path.stem
        kinds[label] = _layer_kinds_summary(report.get("model"))
        rng = random.Random(f"{seed}:{label}:variants")
        curves.extend(build_curves(label, report, rng, quick))

    results: list[CurveResult] = []
    for curve in curves:
        results.append(
            CurveResult(
                label=curve.label,
                variant=curve.variant,
                top=curve.top,
                n_layers=curve.n_layers,
                smart=_evaluate(curve, "smart", repeats, seed, settings),
                every=_evaluate(curve, "every", repeats, seed, settings),
            )
        )

    totals = {"smart": _average(results, "smart"), "every": _average(results, "every")}
    passes = _pass_marks(results)
    return ReplayResult(
        study_folder=str(folder),
        repeats=repeats,
        seed=seed,
        quick=quick,
        curves=results,
        totals=totals,
        passes=passes,
        layer_kinds=kinds,
    )


def _is_core_variant(variant: str) -> bool:
    """Real, reversed and cut curves (the realistic shapes, without the hills)."""
    return variant == "real" or variant == "reversed" or variant.startswith("cut-")


def _miss_rate(
    results: Sequence[CurveResult], design: str, predicate: Callable[[str], bool]
) -> float:
    total = 0
    missed = 0.0
    for result in results:
        if not predicate(result.variant):
            continue
        metrics = result.smart if design == "smart" else result.every
        total += metrics.repeats
        missed += (1 - metrics.within3) * metrics.repeats
    return missed / total if total else 0.0


def _pass_marks(results: Sequence[CurveResult]) -> list[tuple[str, bool, str]]:
    if not results:
        return [("(no curves)", False, "no curves were found")]
    smart_miss = _miss_rate(results, "smart", lambda _variant: True)
    every_miss = _miss_rate(results, "every", lambda _variant: True)
    a_ok = smart_miss <= every_miss + 0.005
    a = (
        f"(a) smart misses {smart_miss:.2%} vs 'every count' {every_miss:.2%} "
        f"(<= +0.5pp): {'ok' if a_ok else 'too high'}"
    )

    smart_core = _miss_rate(results, "smart", _is_core_variant)
    b_ok = smart_core <= 0.015
    b = (
        f"(b) on real+reversed+cut, smart misses {smart_core:.2%} "
        f"(<= 1.5%): {'ok' if b_ok else 'too high'}"
    )

    worst = max(
        (result.every.within3 - result.smart.within3, result.label, result.variant)
        for result in results
    )
    c_ok = worst[0] <= 0.03
    c = (
        f"(c) worst curve gap vs 'every count' is {worst[0]:.2%} on "
        f"{worst[1]}/{worst[2]} (<= 3pp): {'ok' if c_ok else 'too high'}"
    )

    smart_avg = statistics.fmean(result.smart.avg_tests for result in results)
    every_avg = statistics.fmean(result.every.avg_tests for result in results)
    d_ok = smart_avg < every_avg
    d = (
        f"(d) smart used {smart_avg:.1f} tests vs {every_avg:.1f} for 'every count': "
        f"{'fewer' if d_ok else 'not fewer'}"
    )
    return [("a", a_ok, a), ("b", b_ok, b), ("c", c_ok, c), ("d", d_ok, d)]


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #
def render_report(result: ReplayResult) -> str:
    lines: list[str] = []
    lines.append("Layer search replay")
    lines.append(f"study   : {result.study_folder}")
    lines.append(f"repeats : {result.repeats}")
    lines.append(f"seed    : {result.seed}")
    lines.append(f"quick   : {result.quick}")
    lines.append(f"curves  : {len(result.curves)}")
    lines.append("truth   : jumpy counts (passes differ by >10%) use their lowest pass")
    lines.append("hills   : placed only where no count in the hill is jumpy")
    lines.append("")
    header = f"{'label':18} {'variant':10} {'design':7} {'tests':>7} {'within3':>8} {'worst':>8} {'consist':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for curve in result.curves:
        for design, metrics in (("smart", curve.smart), ("every", curve.every)):
            lines.append(
                f"{curve.label:18} {curve.variant:10} {design:7} "
                f"{metrics.avg_tests:>7.1f} {metrics.within3:>7.2%} "
                f"{metrics.worst_gap:>8.2%} {metrics.consistency:>7.2%}"
            )
    lines.append("")
    lines.append("Averages over all curves")
    for design in ("smart", "every"):
        total = result.totals[design]
        lines.append(
            f"  {design:7} tests {total['avg_tests']:.1f}  within3 {total['within3']:.2%}  "
            f"worst {total['worst_gap']:.2%}  consistency {total['consistency']:.2%}"
        )
    smart_core = _miss_rate(result.curves, "smart", _is_core_variant)
    every_core = _miss_rate(result.curves, "every", _is_core_variant)
    lines.append("")
    lines.append(
        f"Miss on real + reversed + cut: smart {smart_core:.2%} "
        f"vs 'every count' {every_core:.2%}"
    )
    lines.append("")
    lines.append("Pass marks")
    for _name, ok, detail in result.passes:
        lines.append(f"  {'PASS' if ok else 'FAIL'} {detail}")
    lines.append("")
    lines.append("Layer kinds")
    for label, summary in result.layer_kinds.items():
        if summary is None:
            lines.append(f"  {label:18} (model file missing or unreadable)")
        else:
            count, description = summary
            lines.append(f"  {label:18} {count} kind(s): {description}")
    return "\n".join(lines) + "\n"


def write_outputs(result: ReplayResult, json_path: Path, txt_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_folder": result.study_folder,
        "repeats": result.repeats,
        "seed": result.seed,
        "quick": result.quick,
        "curves": [
            {
                "label": curve.label,
                "variant": curve.variant,
                "top": curve.top,
                "n_layers": curve.n_layers,
                "smart": asdict(curve.smart),
                "every": asdict(curve.every),
            }
            for curve in result.curves
        ],
        "totals": result.totals,
        "passes": [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in result.passes],
        "layer_kinds": {
            label: ({"kinds": value[0], "layers": value[1]} if value is not None else None)
            for label, value in result.layer_kinds.items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    txt_path.write_text(render_report(result), encoding="utf-8")
