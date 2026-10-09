"""Boundary-aware paging bounds (reviewer, after job 121).

Pages of intervals wholly inside an answer are mandatory; pages of intervals crossing
an answer boundary are optional (none to all of them may belong to the answer);
intervals wholly outside contribute nothing. Full read brackets and the clock slack
are kept; nothing is placed at a midpoint or spread evenly. Total pages and the peak
rate (1 s window, or the whole answer if shorter) are reported as lower and upper
bounds; a single value only where the bounds coincide. The existing bounds-aware
paging guard decides: pass only if every permitted value passes, breach only if every
permitted value breaches, otherwise unsettled. Missing coverage, unusable clocks,
counter resets and sampling stalls stay unknown.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from inferenceos.telemetry.attribution import (
    CLOCK_SLACK_S,
    PAGING_SPEC,
    RawReading,
    intervals_for,
    paging_verdict,
)
from inferenceos.tune.fair import TurnReading, count_guard_verdict
from tests.test_monitor_attribution import RATE32, T0, A, B, readings

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "job121_w25_reconstructed.json"
METRIC = "hard_page_faults_per_sec"


def _blocks(start: tuple[float, float], cand: tuple[float, float], n: int = 4) -> list[TurnReading]:
    def side(b: tuple[float, float]) -> tuple[dict[str, float | None], dict[str, tuple[float, float]]]:
        return ({METRIC: b[0]}, {}) if b[0] == b[1] else ({METRIC: None}, {METRIC: b})

    (sv, sb), (cv, cb) = side(start), side(cand)
    return [TurnReading(turn=k, order="s,c", start=sv, candidate=cv, improvements={},
                        start_bounds=sb, candidate_bounds=cb) for k in range(1, n + 1)]


def _bounds(v: Any) -> tuple[float, float]:
    return float(v.lower_bound_rate), float(v.upper_bound_rate)


# ------------------------------------------------------------- job 121, window w25
def _w25() -> tuple[list[RawReading], float, float, dict[str, Any]]:
    """Job 121's quiet window w25, RECONSTRUCTED from the saved verdict's rounded
    nominal interval times (the job saved no raw readings): each reading at its
    nominal time with a re-created +-1 ms bracket. Not an exact replay."""
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    start = 1_791_449_566.3587067
    times = [iv["from_s"] for iv in fx["intervals"]] + [fx["intervals"][-1]["to_s"]]
    raw = 9_000_000
    out = []
    for k, t in enumerate(times):
        if k > 0:
            raw += fx["intervals"][k - 1]["count"]
        at = start + t
        out.append(RawReading(spec=PAGING_SPEC, read_before=at - 0.001, read_after=at + 0.001, raw=raw, base=0,
                              ts100=int(round(at * 1e7)), counter_type=RATE32, calculated=None,
                              arrival_wall=at + 0.005, arrival_perf=0.0))
    return out, start, start + fx["answer_s"], fx


def test_job121_w25_is_bounded_0_to_78() -> None:
    rs, a, b, fx = _w25()
    v = paging_verdict(rs, a, b)
    print(f"\nw25 (reconstructed): was {fx['original_verdict']['status']} ('{fx['original_verdict']['reason'][:40]}...'); "
          f"now {v.evidence}: pages {v.pages_lower}..{v.pages_upper}, rate {v.lower_bound_rate}..{v.upper_bound_rate}/s")
    assert fx["original_verdict"]["status"] == "unknown" and "RECONSTRUCTED" in fx["label"]
    assert v.status == "measured" and v.evidence == "bounded"
    assert (v.pages_lower, v.pages_upper, v.pages) == (0, 78, None)
    assert _bounds(v) == (0.0, pytest.approx(78.0)) and v.value is None


def test_job121_w25_passes_the_guard_against_a_zero_baseline(tmp_path: Path) -> None:
    """Every permitted value (0 to 78/s) is within the 100/s tolerance and below the
    severe level: the existing guard passes, through compare_fairly too."""
    from tests.test_paging_guard_bounds import BoundedHarness, _compare, _faster

    rs, a, b, _ = _w25()
    w25 = _bounds(paging_verdict(rs, a, b))
    verdict = count_guard_verdict(_blocks((0.0, 0.0), w25), METRIC)
    assert (verdict.state, verdict.typical, verdict.episodes) == ("pass", "pass", "pass")
    fair = _compare(tmp_path, _faster, harness=BoundedHarness(_faster, {False: (0.0, 0.0), True: w25}))
    print(f"\nw25-like candidate vs zero baseline: decision {fair.decision}, kept {fair.kept}, "
          f"breaches {fair.guard_breaches}, undecided {fair.undecided_guards}")
    assert fair.kept and not fair.guard_breaches and not fair.undecided_guards


def test_a_large_ambiguous_burst_leaves_the_comparison_undecided(tmp_path: Path) -> None:
    from tests.test_paging_guard_bounds import BoundedHarness, _compare, _faster

    v = paging_verdict(readings([(A - 0.01, 2048)]), A, B)
    assert v.evidence == "bounded" and (v.pages_lower, v.pages_upper) == (0, 2048)
    fair = _compare(tmp_path, _faster, harness=BoundedHarness(_faster, {False: (0.0, 0.0), True: _bounds(v)}))
    print(f"\n2,048 edge pages ({_bounds(v)}): decision {fair.decision}, kept {fair.kept}, undecided {fair.undecided_guards}")
    assert fair.decision == "unsure" and not fair.kept and fair.undecided_guards == [METRIC]


# ---------------------------------------------------------- what can share a window
def test_interior_and_boundary_pages_that_could_share_one_second() -> None:
    v = paging_verdict(readings([(A + 0.01, 60), (A + 0.35, 60)]), A, B)
    assert (v.pages_lower, v.pages_upper) == (60, 120)
    assert _bounds(v) == (pytest.approx(60.0), pytest.approx(120.0))
    assert count_guard_verdict(_blocks((0.0, 0.0), _bounds(v)), METRIC).state == "unknown"


def test_interior_and_boundary_pages_too_far_apart_to_share_a_second() -> None:
    """Boundary pages can only have fallen in the answer's first ~0.1 s, the interior
    ones 1.6 s later: no window holds both, so the rate is exact (60/s) while the count
    stays bounded (60..120)."""
    v = paging_verdict(readings([(A + 0.01, 60), (A + 1.65, 60)]), A, B)
    assert (v.pages_lower, v.pages_upper, v.pages) == (60, 120, None)
    assert _bounds(v) == (pytest.approx(60.0), pytest.approx(60.0)) and v.value == pytest.approx(60.0)
    assert v.evidence == "bounded"  # the count is not exact


def test_both_answer_boundaries() -> None:
    v = paging_verdict(readings([(A + 0.01, 50), (B - 0.01, 50)]), A, B)
    assert (v.pages_lower, v.pages_upper) == (0, 100)
    assert _bounds(v) == (0.0, pytest.approx(50.0))  # 2 s apart: never in one window


def test_answer_shorter_than_one_second_is_its_own_window() -> None:
    a, b = A, A + 0.5
    v = paging_verdict(readings([(A + 0.01, 30), (b - 0.01, 30)]), a, b)
    assert (v.pages_lower, v.pages_upper) == (0, 60)
    assert _bounds(v) == (0.0, pytest.approx(60 / 0.5))


# ------------------------------------------------------------------ guard decisions
@pytest.mark.parametrize(("start", "cand", "state"), [
    ((0.0, 78.0), (0.0, 78.0), "pass"),      # both uncertain: +78/s at most
    ((0.0, 78.0), (150.0, 150.0), "unknown"),  # +72..+150/s straddles the tolerance
    ((0.0, 78.0), (300.0, 300.0), "breach"),  # +222/s at least
])
def test_uncertainty_on_both_sides(start: tuple[float, float], cand: tuple[float, float], state: str) -> None:
    assert count_guard_verdict(_blocks(start, cand), METRIC).state == state


def test_a_possible_severe_episode_at_an_edge_is_unsettled() -> None:
    v = paging_verdict(readings([(A - 0.01, 1500)]), A, B)
    verdict = count_guard_verdict(_blocks((0.0, 0.0), _bounds(v)), METRIC)
    print(f"\n1,500 edge pages in every block: typical {verdict.typical}, episodes {verdict.episodes} "
          f"(excess {verdict.excess_low}..{verdict.excess_high})")
    assert (verdict.typical, verdict.episodes, verdict.state) == ("unknown", "unknown", "unknown")
    assert any("possibly a severe episode" in w for w in verdict.words)


# ---------------------------------------------------------- still unknown, never zero
def test_missing_coverage_and_bad_timing_stay_unknown() -> None:
    assert paging_verdict(readings(start=A + 0.5), A, B).status == "unknown"  # nothing before
    assert paging_verdict(readings([(A - 0.01, 5)], stop=B - 0.5), A, B).status == "pending"  # nothing after (yet)
    reset = readings([(A - 0.01, 5)], tweak={20: {"raw": 1}})
    assert paging_verdict(reset, A, B).status == "unknown"


def test_a_long_gap_at_an_edge_stays_unknown_judged_on_its_full_span() -> None:
    """A paged interval crossing the answer's start spans 0.4 s, although only ~0.1 s of
    it lies inside the answer: the gap is judged on the original, unclipped interval."""
    rs = [r for r in readings([(A - 0.15, 90)]) if not (A - 0.35 < r.read_before < A - 0.05)]
    span = [iv for iv in intervals_for(rs, PAGING_SPEC) if iv.place(A, B) == "boundary" and iv.count]
    assert span and span[0].outer_end - span[0].outer_start > 0.25 and span[0].outer_end - A < 0.25
    v = paging_verdict(rs, A, B)
    assert v.status == "unknown" and "rate not resolved" in v.reason and v.value is None


def test_a_long_gap_without_pages_still_proves_zero() -> None:
    rs = [r for r in readings() if not (A - 0.35 < r.read_before < A - 0.05)]
    v = paging_verdict(rs, A, B)
    assert v.evidence == "exact" and v.pages == 0 and v.value == 0.0


# --------------------------------------------------------------- adversarial check
def _placements(lo: float, hi: float, step: float) -> list[list[tuple[float, float]]]:
    """Where one interval's pages may lie: all at one grid point, or split in half
    between two grid points (fractions of the count, at times within the span)."""
    grid = [lo + k * step for k in range(int((hi - lo) / step) + 1)]
    return [[(t, 1.0)] for t in grid] + [[(t, 0.5), (u, 0.5)] for t, u in itertools.combinations(grid, 2)]


@pytest.mark.parametrize(("a_off", "length", "events"), [
    (0.0, 0.6, [(-0.05, 40), (0.25, 60), (0.65, 30)]),   # both edges + interior, sub-second answer
    (0.0, 2.0, [(-0.05, 80), (0.85, 50), (1.95, 70)]),   # 2 s answer: interior within reach of each edge
])
def test_every_placement_of_the_pages_stays_within_the_bounds(a_off: float, length: float,
                                                              events: Sequence[tuple[float, int]]) -> None:
    """Enumerates placements of each paged interval's pages anywhere in its full
    possible span (read brackets and clock slack included; a superset of what the
    counter allows): the answer's true count and peak always lie within the bounds."""
    a = A + a_off
    b = a + length
    rs = readings([(a + t, p) for t, p in events], tweak={k: {"read_after": T0 + k * 0.1 + 0.01}
                                                          for k in range(61)})
    v = paging_verdict(rs, a, b)
    assert v.status == "measured"
    paged = [iv for iv in intervals_for(rs, PAGING_SPEC) if iv.count and iv.place(a, b) != "outside"]
    options = [_placements(iv.outer_start, iv.outer_end, 0.02) for iv in paged]
    window = min(1.0, b - a)  # as the monitor computes it
    checked, peak_max, peak_min = 0, 0.0, float("inf")
    for choice in itertools.product(*options):
        pts = [(t, frac * iv.count) for iv, spot in zip(paged, choice) for t, frac in spot]  # type: ignore[operator]
        mine = sorted((t, c) for t, c in pts if a <= t <= b)
        pages = sum(c for _, c in mine)
        peak = max((sum(c for t, c in mine if s <= t <= s + window) for s, _ in mine), default=0.0) / window
        assert (v.pages_lower or 0) - 1e-9 <= pages <= (v.pages_upper or 0) + 1e-9
        assert (v.lower_bound_rate or 0) - 1e-6 <= peak <= (v.upper_bound_rate or 0) + 1e-6, (choice, v)
        peak_max, peak_min = max(peak_max, peak), min(peak_min, peak)
        checked += 1
    print(f"\n{length:g} s answer: {checked:,} placements; true peak {peak_min:.1f}..{peak_max:.1f}/s within "
          f"bounds {v.lower_bound_rate:.1f}..{v.upper_bound_rate:.1f}/s; pages {v.pages_lower}..{v.pages_upper}")
    assert checked > 1000 and CLOCK_SLACK_S == 0.02
