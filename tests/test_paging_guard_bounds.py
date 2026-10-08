"""The paging guard over readings known only within bounds (reviewer, monitor round 3).

The experimental scoped monitor can bound a peak paging rate without pinning it. The
guard's rules and thresholds are unchanged (100/s typical tolerance on the median
per-block difference; severe episodes at 1,000/s, counted against the new setup at an
excess of 2), but they are evaluated over every value the bounds support: a pass needs
the most adverse values to pass, a breach needs the most favourable values to breach,
and anything else is unknown -- not a pass. Exact readings decide exactly as before.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inferenceos.contracts.observation import RunMetrics
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.telemetry.attribution import paging_verdict
from inferenceos.tune.fair import (
    COUNT_GUARDS,
    EPISODE_EXCESS,
    TurnReading,
    count_guard_verdict,
    guard_change,
    guard_episodes,
    measurement_bounds,
    measurement_value,
)
from tests import test_monitor_attribution as attr
from tests.experiment_helpers import SampleResult, ScriptedHarness
from tests.test_tune_fair import _compare

METRIC = "hard_page_faults_per_sec"


def _side(b: tuple[float, float] | None) -> tuple[dict[str, float | None], dict[str, tuple[float, float]]]:
    if b is None:
        return {METRIC: None}, {}
    if b[0] == b[1]:
        return {METRIC: b[0]}, {}
    return {METRIC: None}, {METRIC: b}


def _blocks(start: tuple[float, float], cand: tuple[float, float], n: int = 4) -> list[TurnReading]:
    (sv, sb), (cv, cb) = _side(start), _side(cand)
    return [TurnReading(turn=k, order="start,candidate", start=sv, candidate=cv, improvements={},
                        start_bounds=sb, candidate_bounds=cb) for k in range(1, n + 1)]


def _measurement(*runs: RunMetrics) -> Any:
    return SimpleNamespace(run_metrics=list(runs), metrics={})


def _bounded_run(low: float, high: float) -> RunMetrics:
    exact = low == high
    return RunMetrics(hard_page_faults_per_sec=low if exact else None,
                      hard_page_faults_per_sec_lower=low, hard_page_faults_per_sec_upper=high)


# ------------------------------------------------------------------ the reproduced case
def test_reviewer_case_baseline_uncertainty_is_not_hidden() -> None:
    """Starting setup: two 100-page bursts 1.08 s apart (true peak 100/s) that the
    readings can only bound to 100-200/s; candidate: one 260-page burst (exact). The
    old guard compared the two upper bounds (200 vs 260: +60/s, a pass); the true
    regression is +160/s. Now the guard is unknown -- neither pass nor breach."""
    A, B = attr.A, attr.B
    start = paging_verdict(attr.readings([(A + 0.51, 100), (A + 1.59, 100)]), A, B)
    cand = paging_verdict(attr.readings([(A + 1.0, 260)]), A, B)
    start_m = _measurement(*[_bounded_run(start.lower_bound_rate or 0, start.upper_bound_rate or 0)] * 3)
    cand_m = _measurement(*[_bounded_run(cand.lower_bound_rate or 0, cand.upper_bound_rate or 0)] * 3)
    s_b, c_b = measurement_bounds(start_m, METRIC), measurement_bounds(cand_m, METRIC)
    assert s_b is not None and c_b is not None
    blocks = _blocks(s_b, c_b)
    verdict = count_guard_verdict(blocks, METRIC)
    print(f"\nstart bounds {s_b} (value {measurement_value(start_m, METRIC)}), candidate {c_b} "
          f"(value {measurement_value(cand_m, METRIC)}) | middle change {verdict.middle_low:+.0f}..{verdict.middle_high:+.0f}/s "
          f"| guard {verdict.state} | guard_change {guard_change(blocks, METRIC)}")
    assert s_b == (100.0, 200.0) and c_b == (260.0, 260.0)
    assert measurement_value(start_m, METRIC) is None  # no upper bound passed off as a reading
    assert measurement_value(cand_m, METRIC) == 260.0  # exact stays exact
    assert (verdict.middle_low, verdict.middle_high) == (60.0, 160.0)
    assert verdict.state == "unknown" and guard_change(blocks, METRIC) is None


@pytest.mark.parametrize(("start", "cand", "state"), [
    ((100.0, 200.0), (260.0, 260.0), "unknown"),   # the reproduced case
    ((100.0, 120.0), (260.0, 260.0), "breach"),    # +140/s at least: established
    ((100.0, 200.0), (150.0, 180.0), "pass"),      # +80/s at most: passes adversely
    ((0.0, 0.0), (90.0, 110.0), "unknown"),        # straddles the tolerance against zero
    ((0.0, 0.0), (101.0, 300.0), "breach"),
    ((50.0, 60.0), (55.0, 150.0), "pass"),         # +100/s at most: the limit itself passes, as before
])
def test_typical_rule_over_bounds(start: tuple[float, float], cand: tuple[float, float], state: str) -> None:
    verdict = count_guard_verdict(_blocks(start, cand), METRIC)
    assert verdict.typical == state and verdict.state == state
    change = guard_change(_blocks(start, cand), METRIC)
    if state == "unknown":
        assert change is None
    else:
        assert change is not None and (change < -1.0) is (state == "breach")


def test_severe_episode_rule_over_bounds() -> None:
    level = COUNT_GUARDS[METRIC][1]

    def blocks(starts: list[tuple[float, float]], cands: list[tuple[float, float]]) -> list[TurnReading]:
        out = []
        for k, (s, c) in enumerate(zip(starts, cands), 1):
            (sv, sb), (cv, cb) = _side(s), _side(c)
            out.append(TurnReading(turn=k, order="s,c", start=sv, candidate=cv, improvements={},
                                   start_bounds=sb, candidate_bounds=cb))
        return out

    quiet, maybe, sure = (0.0, 0.0), (level - 100, level + 100), (level + 50, level + 50)
    # Two certain new-setup episodes; the current setup's two readings may or may not be
    # episodes: the excess is 0..2 -> unknown (not a pass).
    v = count_guard_verdict(blocks([maybe, maybe, quiet, quiet], [sure, sure, quiet, quiet]), METRIC)
    assert (v.episodes, v.excess_low, v.excess_high) == ("unknown", 0, 2)
    assert v.state == "unknown" and any("possibly a severe episode" in w for w in v.words)
    # Two certain new-setup episodes and none possible in the current one: breach.
    v = count_guard_verdict(blocks([quiet] * 4, [sure, sure, quiet, quiet]), METRIC)
    assert v.episodes == "breach" and guard_episodes(blocks([quiet] * 4, [sure, sure, quiet, quiet]), METRIC)[0]
    # One possible new-setup episode at most: fewer than EPISODE_EXCESS whatever the truth.
    v = count_guard_verdict(blocks([quiet] * 4, [maybe, quiet, quiet, quiet]), METRIC)
    assert v.episodes == "pass" and v.excess_high == 1 < EPISODE_EXCESS
    # Possible episodes in the new setup that the bounds cannot settle: unknown.
    v = count_guard_verdict(blocks([quiet] * 4, [maybe, maybe, quiet, quiet]), METRIC)
    assert (v.episodes, v.excess_low, v.excess_high) == ("unknown", 0, 2)


def test_exact_readings_decide_exactly_as_before() -> None:
    """Against the previous implementation (copied here) on random exact readings."""
    tolerance, level = COUNT_GUARDS[METRIC]

    def old_change(pairs: list[tuple[float, float]]) -> float:
        return -float(statistics.median(n - s for s, n in pairs)) / tolerance

    def old_against(pairs: list[tuple[float, float]]) -> bool:
        return sum(n >= level for _, n in pairs) - sum(s >= level for s, _ in pairs) >= EPISODE_EXCESS

    rng = random.Random(7)
    for _ in range(500):
        pairs = [(rng.choice([0.0, rng.uniform(0, 300), rng.uniform(800, 1500)]),
                  rng.choice([0.0, rng.uniform(0, 300), rng.uniform(800, 1500)])) for _ in range(rng.randint(1, 8))]
        blocks = [TurnReading(turn=k, order="s,c", start={METRIC: s}, candidate={METRIC: n}, improvements={})
                  for k, (s, n) in enumerate(pairs, 1)]
        assert guard_change(blocks, METRIC) == pytest.approx(old_change(pairs))
        assert guard_episodes(blocks, METRIC)[0] is old_against(pairs)
        assert count_guard_verdict(blocks, METRIC).state != "unknown"


def test_measurement_bounds_are_the_medians_of_the_run_bounds() -> None:
    m = _measurement(_bounded_run(0, 0), _bounded_run(100, 200), _bounded_run(260, 260), RunMetrics())
    assert measurement_bounds(m, METRIC) == (100.0, 200.0) and measurement_value(m, METRIC) is None
    exact = _measurement(RunMetrics(hard_page_faults_per_sec=3.0), RunMetrics(hard_page_faults_per_sec=9.0),
                         RunMetrics(hard_page_faults_per_sec=5.0))
    assert measurement_bounds(exact, METRIC) == (5.0, 5.0) and measurement_value(exact, METRIC) == 5.0


# ---------------------------------------------------------- through compare_fairly
class BoundedHarness(ScriptedHarness):
    """Gives every run of each setup a paging reading known only within bounds."""

    def __init__(self, responder: Any, bounds: dict[bool, tuple[float, float]]) -> None:
        super().__init__(responder)
        self.bounds = bounds  # keyed by "is the candidate"

    def run_once(self, task: Any, **kwargs: Any) -> Any:
        run = super().run_once(task, **kwargs)
        low, high = self.bounds[kwargs["options"].threads == 5]
        metrics = run.observation.metrics.model_copy(update={
            "hard_page_faults_per_sec": low if low == high else None,
            "hard_page_faults_per_sec_lower": low, "hard_page_faults_per_sec_upper": high})
        return run.model_copy(update={"observation": run.observation.model_copy(update={"metrics": metrics})})


def _faster(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    return SampleResult(decode_tps=24.0 if options.threads == 5 else 20.0, total_latency_s=2.0)


@pytest.mark.parametrize(("start", "cand", "expect"), [
    ((100.0, 200.0), (260.0, 260.0), "unsettled"),
    ((100.0, 120.0), (260.0, 260.0), "breach"),
    ((100.0, 200.0), (150.0, 180.0), "clear"),
])
def test_compare_fairly_uses_both_bounds(tmp_path: Path, start: tuple[float, float],
                                         cand: tuple[float, float], expect: str) -> None:
    harness = BoundedHarness(_faster, {False: start, True: cand})
    fair = _compare(tmp_path, _faster, harness=harness)
    record = fair.record()
    print(f"\nstart {start} vs new {cand}: decision {fair.decision}, kept {fair.kept}, breaches {fair.guard_breaches}, "
          f"undecided {fair.undecided_guards}, blockers {fair.acceptance_blockers()}")
    if expect == "unsettled":
        assert fair.undecided_guards == [METRIC] and not fair.guard_breaches
        assert fair.decision == "unsure" and not fair.kept and not fair.not_worse
        assert any("not settled by its bounds" in note for note in fair.notes)
        assert record["turns"][0]["start_bounds"] == {METRIC: start}  # bounds shown as bounds
    elif expect == "breach":
        assert METRIC in fair.guard_breaches and not fair.kept and not fair.undecided_guards
    else:
        assert not fair.guard_breaches and not fair.undecided_guards and fair.kept
