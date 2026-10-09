"""Attribution rules of the measurement-scoped monitor (deterministic; no sampler).

Readings are synthesized like the real sampler's: a paging counter read every 0.1 s,
each read bracketed by the clock, with a cumulative raw page count. Pages are injected
at chosen times relative to an answer. Checks the selection AND the resulting paging
guard decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from inferenceos.telemetry.attribution import (
    CLOCK_SLACK_S,
    PAGING_SPEC,
    RawReading,
    intervals_for,
    paging_verdict,
)

RATE32 = 272696320
T0 = 1_800_000_000.0  # an arbitrary wall-clock origin
A, B = T0 + 2.0, T0 + 4.0  # the answer: 2 s long


def readings(events: Sequence[tuple[float, int]] = (), *, start: float = T0, stop: float = T0 + 6.0,
             step: float = 0.1, lag: float = 0.01, counter_type: int = RATE32,
             tweak: dict[int, dict[str, float]] | None = None) -> list[RawReading]:
    """One reading every ``step`` s; ``events`` are (time, pages) added to the count."""
    out: list[RawReading] = []
    k, t = 0, start
    while t <= stop + 1e-9:
        raw = 4_621_826 + sum(p for at, p in events if at < t)
        fields = {"read_before": t, "read_after": t + 0.002, "arrival_wall": t + 0.002 + lag}
        fields.update((tweak or {}).get(k, {}))
        out.append(RawReading(spec=PAGING_SPEC, read_before=fields["read_before"], read_after=fields["read_after"],
                              raw=int(fields.get("raw", raw)), base=0, ts100=int(round(t * 1e7)),
                              counter_type=counter_type, calculated=None,
                              arrival_wall=fields["arrival_wall"], arrival_perf=0.0))
        k += 1
        t = start + k * step
    return out


def test_quiet_answer_is_measured_zero() -> None:
    v = paging_verdict(readings(), A, B)
    assert (v.status, v.value, v.pages) == ("measured", 0.0, 0)
    assert 2 <= v.boundary <= 4 and v.inside >= 16  # readings at the edges: slack makes both neighbours edges


def test_paging_during_the_answer_is_detected_with_its_count() -> None:
    v = paging_verdict(readings([(A + 0.55, 50)]), A, B)
    assert v.status == "measured" and v.pages == 50
    # Peak over spans of at least 1 s: 50 pages within about 1 s.
    assert 45 <= (v.value or 0) <= 52


def test_paging_just_before_the_answer_is_not_attributed() -> None:
    v = paging_verdict(readings([(A - 0.15, 300)]), A, B)  # an interval wholly before A
    assert (v.status, v.pages, v.value) == ("measured", 0, 0.0)


def test_paging_just_after_the_answer_is_not_attributed() -> None:
    v = paging_verdict(readings([(B + 0.15, 300)]), A, B)
    assert (v.status, v.pages, v.value) == ("measured", 0, 0.0)


@pytest.mark.parametrize("edge", [A + 0.01, B - 0.01, A - 0.01, B + 0.01])
def test_paging_in_an_edge_interval_is_bounded_not_zero(edge: float) -> None:
    """An interval overlapping the answer's start or end with pages in it: the counter
    cannot tell whether they fell inside, so anywhere from none to all of them belong
    to the answer (reviewer, after job 121; never zero, never split evenly). Both
    answer edges; the rate's upper bound is on the 1 s window, not the overlap."""
    v = paging_verdict(readings([(edge, 227)]), A, B)
    assert v.status == "measured" and v.evidence == "bounded" and v.value is None and v.pages is None
    assert (v.pages_lower, v.pages_upper, v.boundary_pages) == (0, 227, 227)
    assert (v.lower_bound_rate, v.upper_bound_rate) == (0.0, pytest.approx(227.0))
    assert any(iv["count"] == 227 and iv["place"] == "boundary" for iv in v.intervals)


def test_zero_page_edge_intervals_prove_zero() -> None:
    v = paging_verdict(readings([(A - 0.5, 10), (B + 0.5, 10)]), A, B)
    assert v.status == "measured" and v.pages == 0


def test_short_answer_inside_one_interval() -> None:
    a, b = A + 0.03, A + 0.07  # shorter than one reading interval: only an edge interval overlaps
    assert paging_verdict(readings(), a, b).status == "measured"
    v = paging_verdict(readings([(A + 0.05, 5)]), a, b)
    # The 40 ms answer is its own peak window: 0 to 5 pages in 0.04 s.
    assert v.evidence == "bounded" and (v.pages_lower, v.pages_upper) == (0, 5)
    assert (v.lower_bound_rate, v.upper_bound_rate) == (0.0, pytest.approx(5 / 0.04))


def test_short_answer_spanning_a_few_intervals_reports_its_whole_length_rate() -> None:
    a, b = A, A + 0.5  # shorter than the 1 s peak span: the whole answer is the span
    v = paging_verdict(readings([(A + 0.25, 10)]), a, b)
    assert v.status == "measured" and v.pages == 10
    assert 18 <= (v.value or 0) <= 26  # about 10 pages over about 0.5 s


def test_delayed_arrival_within_the_limit_is_still_usable() -> None:
    v = paging_verdict(readings([(A + 0.55, 50)], lag=1.5), A, B)
    assert v.status == "measured" and v.pages == 50


def test_arrival_far_too_late_is_unknown() -> None:
    v = paging_verdict(readings(lag=3.0), A, B)
    assert v.status == "unknown"


def test_no_reading_before_the_answer_is_unknown() -> None:
    v = paging_verdict(readings(start=A + 0.05), A, B)
    assert v.status == "unknown" and "before the answer" in v.reason


def test_no_reading_after_the_answer_is_pending_then_unknown() -> None:
    v = paging_verdict(readings(stop=B - 0.05), A, B)
    assert v.status == "pending"  # the monitor waits (bounded) and then calls it unknown


def test_a_stall_keeps_the_exact_count_but_not_the_rate() -> None:
    """A stall leaves a long interval: the cumulative count still holds every page, but
    the rate cannot be placed within a 1 s window, so it is unknown (reviewer, monitor
    round 2: the earlier version reported such a case as measured and diluted)."""
    rs = [r for r in readings([(A + 1.0, 40)]) if not (A + 0.6 < r.read_before < A + 1.6)]
    v = paging_verdict(rs, A, B)
    assert v.status == "unknown" and v.pages == 40 and "rate not resolved" in v.reason


def test_a_stall_with_no_pages_still_proves_zero() -> None:
    rs = [r for r in readings() if not (A + 0.6 < r.read_before < A + 1.6)]
    v = paging_verdict(rs, A, B)
    assert (v.status, v.value, v.pages) == ("measured", 0.0, 0)


def test_clock_stepped_back_is_unknown() -> None:
    k = int(round((A + 1.0 - T0) / 0.1))
    rs = readings(tweak={k: {"arrival_wall": A + 1.0 - 0.5}})  # arrived before it was read
    v = paging_verdict(rs, A, B)
    assert v.status == "unknown" and "clock" in v.reason


def test_reversed_read_bracket_is_unknown() -> None:
    k = int(round((A + 1.0 - T0) / 0.1))
    rs = readings(tweak={k: {"read_before": A + 1.01, "read_after": A + 1.0}})
    assert paging_verdict(rs, A, B).status == "unknown"


def test_counter_reset_is_unknown() -> None:
    k = int(round((A + 1.0 - T0) / 0.1))
    rs = readings(tweak={k: {"raw": 10}})
    v = paging_verdict(rs, A, B)
    assert v.status == "unknown" and "backwards" in v.reason


def test_a_counter_without_count_semantics_is_not_interpreted() -> None:
    v = paging_verdict(readings(counter_type=65536), A, B)
    assert v.status == "unknown" and "no count semantics" in v.reason


def test_clock_slack_widens_the_boundary() -> None:
    # A reading 5 ms before A cannot be certainly before A with a 20 ms slack: the
    # interval it closes is an edge interval, not outside.
    rs = readings(start=A - 0.105, step=0.1)
    ivs = [iv for iv in intervals_for(rs, PAGING_SPEC) if iv.place(A, B) == "boundary"]
    assert ivs and CLOCK_SLACK_S == 0.02


# ---------------------------------------------------------------- guard decisions


def _block(start_paging: float | None, candidate_paging: float | None, turn: int) -> object:
    from inferenceos.tune.fair import TurnReading

    return TurnReading(turn=turn, order="start,candidate",
                       start={"hard_page_faults_per_sec": start_paging},
                       candidate={"hard_page_faults_per_sec": candidate_paging}, improvements={})


def test_guard_sees_detected_paging_and_breaches() -> None:
    from inferenceos.tune.fair import guard_change

    quiet = paging_verdict(readings(), A, B)
    paged = paging_verdict(readings([(A + 0.55, 300)]), A, B)
    blocks = [_block(quiet.value, paged.value, n) for n in range(1, 5)]
    change = guard_change(blocks, "hard_page_faults_per_sec")  # type: ignore[arg-type]
    assert change is not None and change < -1.0  # beyond the 100/s tolerance: a breach


def test_guard_never_reads_ambiguous_paging_as_clear() -> None:
    from inferenceos.tune.fair import guard_change

    quiet = paging_verdict(readings(), A, B)
    ambiguous = paging_verdict(readings([(A + 0.01, 300)]), A, B)
    assert ambiguous.value is None
    blocks = [_block(quiet.value, ambiguous.value, n) for n in range(1, 5)]
    # Not measured -> no guard value at all (compare_fairly then counts the guard as
    # unavailable and the decision cannot be "better" / "not worse").
    assert guard_change(blocks, "hard_page_faults_per_sec") is None  # type: ignore[arg-type]


# ------------------------------------------------- reviewer, monitor round 2, item 1


def _guard(value: float | None) -> float | None:
    from inferenceos.tune.fair import guard_change

    blocks = [_block(0.0, value, n) for n in range(1, 5)]
    return guard_change(blocks, "hard_page_faults_per_sec")  # type: ignore[arg-type]


def test_reviewer_case_a_burst_inside_a_long_gap_is_not_diluted() -> None:
    end = A + 6.0
    rs = [r for r in readings([(A + 2.0, 300)], stop=A + 8.0) if not (A + 0.55 < r.read_before < A + 4.45)]
    v = paging_verdict(rs, A, end)
    print(f"\n4 s gap, 300 pages: status {v.status}, pages {v.pages}, value {v.value}, bounds "
          f"{v.lower_bound_rate}..{v.upper_bound_rate}, guard change {_guard(v.value)}")
    assert v.status == "unknown" and v.value is None and v.pages == 300  # count kept, rate not claimed
    # Anywhere from spread out to one burst. Spread out, the 300 pages still put at least
    # 300 / 5 = 60 in some 1 s window (the gap spans just over 4 s: five windows cover it).
    assert (v.lower_bound_rate, v.upper_bound_rate) == (60.0, 300.0)
    assert _guard(v.value) is None  # no guard value: the check cannot pass on it


def test_irregular_cadence_still_establishes_the_rate() -> None:
    """Readings 0.05-0.25 s apart (all within the resolution limit): a 300-page burst is
    measured at full height and breaches the guard."""
    steps = [0.05, 0.25, 0.12, 0.2, 0.08, 0.15, 0.22, 0.06]
    rs: list[RawReading] = []
    t, k = T0, 0
    while t <= T0 + 6.0:
        raw = 4_621_826 + (300 if t > A + 1.03 else 0)
        rs.append(RawReading(spec=PAGING_SPEC, read_before=t, read_after=t + 0.002, raw=raw, base=0,
                             ts100=int(round(t * 1e7)), counter_type=RATE32, calculated=None,
                             arrival_wall=t + 0.01, arrival_perf=0.0))
        t += steps[k % len(steps)]
        k += 1
    v = paging_verdict(rs, A, B)
    print(f"\nirregular cadence: status {v.status}, value {v.value}, guard change {_guard(v.value)}")
    assert v.status == "measured" and v.pages == 300 and v.value == pytest.approx(300.0)
    assert (_guard(v.value) or 0) <= -1.0  # breach


@pytest.mark.parametrize(("pages", "breach"), [(110, True), (101, True), (99, False), (90, False)])
def test_bursts_near_the_threshold_keep_their_guard_outcome(pages: int, breach: bool) -> None:
    v = paging_verdict(readings([(A + 1.03, pages)]), A, B)  # inside one 0.1 s interval
    change = _guard(v.value)
    print(f"\n{pages} pages in one 0.1 s interval: value {v.value}, guard change {change}")
    assert v.status == "measured" and v.value == pytest.approx(float(pages))
    assert change is not None and ((change < -1.0) is breach)


def test_a_paged_interval_just_over_the_limit_is_unknown() -> None:
    rs = [r for r in readings([(A + 1.1, 150)]) if not (A + 1.0 < r.read_before < A + 1.3)]  # a 0.3 s interval
    v = paging_verdict(rs, A, B)
    # Its full possible span: 0.3 s + the 2 ms read bracket + 20 ms clock slack each side.
    assert v.status == "unknown" and "0.34 s interval" in v.reason and v.pages == 150


def test_continuous_paging_is_bounded_not_passed_off_as_exact() -> None:
    """Reviewer, monitor round 3: bounds are reported as bounds. 20 pages in each of ten
    0.1 s intervals (true peak 200/s): the timing allows 9 to 10 intervals' pages in one
    second, so 180-200/s -- the truth is inside, and no single value is claimed."""
    events = [(A + 0.5 + 0.1 * i + 0.05, 20) for i in range(10)]  # 20 pages every 0.1 s for 1 s: 200/s
    v = paging_verdict(readings(events), A, B)
    print(f"\ncontinuous 200/s: value {v.value}, bounds {v.lower_bound_rate}..{v.upper_bound_rate}, exact {v.exact}")
    assert v.status == "measured" and v.pages == 200 and not v.exact and v.value is None
    assert (v.lower_bound_rate, v.upper_bound_rate) == (pytest.approx(180.0), pytest.approx(200.0))


# ------------------------------------------- reviewer, monitor round 3, item 2
def _wide(events: Sequence[tuple[float, int]]) -> list[RawReading]:
    """Readings every 0.1 s whose read brackets are 60 ms wide (a slow NextSample()):
    each read happened somewhere in [t, t + 0.06]."""
    n = int(round(6.0 / 0.1)) + 1
    tweak = {k: {"read_after": T0 + k * 0.1 + 0.06, "arrival_wall": T0 + k * 0.1 + 0.07} for k in range(n)}
    return readings(events, tweak=tweak)


def _bounded_block(start: tuple[float, float], cand: tuple[float, float], turn: int) -> Any:
    from inferenceos.tune.fair import TurnReading

    def side(b: tuple[float, float]) -> tuple[dict[str, float | None], dict[str, tuple[float, float]]]:
        exact = b[0] == b[1]
        return {"hard_page_faults_per_sec": b[0] if exact else None}, ({} if exact else {"hard_page_faults_per_sec": b})

    (sv, sb), (cv, cb) = side(start), side(cand)
    return TurnReading(turn=turn, order="start,candidate", start=sv, candidate=cv, improvements={},
                       start_bounds=sb, candidate_bounds=cb)


@pytest.mark.parametrize(("pages", "guard"), [(40, "pass"), (100, "unknown"), (150, "breach")])
def test_bounds_use_the_full_read_brackets_not_midpoints(pages: int, guard: str) -> None:
    """Two bursts in intervals whose midpoints are 1.0 s apart; the 60 ms read brackets
    and the clock slack let them be 0.90 s apart, i.e. inside one second. The midpoint
    version reported lower = upper = one burst (exact); the true peak may be two."""
    from inferenceos.tune.fair import count_guard_verdict

    v = paging_verdict(_wide([(A + 0.25, pages), (A + 1.35, pages)]), A, B)
    blocks = [_bounded_block((0.0, 0.0), (v.lower_bound_rate or 0.0, v.upper_bound_rate or 0.0), n)
              for n in range(1, 5)]
    verdict = count_guard_verdict(blocks, "hard_page_faults_per_sec")
    print(f"\n2 x {pages} pages, midpoints 1.0 s apart: status {v.status}, bounds "
          f"{v.lower_bound_rate}..{v.upper_bound_rate}, exact {v.exact}, value {v.value} | guard vs 0: {verdict.state}")
    assert v.status == "measured" and v.pages == 2 * pages and not v.exact and v.value is None
    assert (v.lower_bound_rate, v.upper_bound_rate) == (pytest.approx(pages), pytest.approx(2 * pages))
    assert verdict.state == guard


def _random_readings(rng: Any, events: Sequence[tuple[float, int]]) -> list[RawReading]:
    """Random cadence and read brackets; the true read moment lies inside each bracket."""
    rs: list[RawReading] = []
    t = T0
    while t <= T0 + 6.0:
        width_before, width_after = rng.uniform(0, 0.03), rng.uniform(0, 0.03)
        raw = 4_621_826 + sum(p for at, p in events if at < t)  # the read happened at t
        rs.append(RawReading(spec=PAGING_SPEC, read_before=t - width_before, read_after=t + width_after,
                             raw=raw, base=0, ts100=int(round(t * 1e7)), counter_type=RATE32,
                             calculated=None, arrival_wall=t + width_after + 0.005, arrival_perf=0.0))
        t += rng.uniform(0.04, 0.12)
    return rs


def _true_peak(events: Sequence[tuple[float, int]], a: float, b: float) -> float:
    """The answer's true peak rate: pages that really fell during [a, b], over the
    densest closed window (1 s, or the whole answer if shorter)."""
    window = min(1.0, b - a)
    mine = sorted((t, p) for t, p in events if a <= t <= b)
    if not mine:
        return 0.0
    return max(sum(p for t, p in mine if s <= t <= s + window) for s, _ in mine) / window


@pytest.mark.parametrize("length", [2.0, 0.6])
def test_bounds_contain_every_peak_the_readings_allow(length: float) -> None:
    """Randomised: random cadence and read brackets, bursts at random true times inside,
    across and outside both answer edges, for a 2 s and a sub-second answer. Whenever
    the verdict is measured (exact or bounded), the answer's true peak and true page
    count lie within its bounds."""
    import random

    rng = random.Random(20261008 + int(length * 10))
    checked = bounded = 0
    a, b = A, A + length
    for _ in range(500):
        events = [(a + rng.uniform(-0.4, length + 0.4), rng.randint(1, 80)) for _ in range(rng.randint(1, 6))]
        v = paging_verdict(_random_readings(rng, events), a, b)
        if v.status != "measured":
            continue
        pages = sum(p for t, p in events if a <= t <= b)
        assert (v.pages_lower or 0) <= pages <= (v.pages_upper or 0), (events, v)
        peak = _true_peak(events, a, b)
        assert (v.lower_bound_rate or 0) - 1e-9 <= peak <= (v.upper_bound_rate or 0) + 1e-9, (events, v)
        checked += 1
        bounded += v.evidence == "bounded"
    print(f"\nrandomised containment ({length:g} s answers): {checked} measured cases ({bounded} bounded), "
          "every true count and peak within its bounds")
    assert checked > 100 and bounded > 30
