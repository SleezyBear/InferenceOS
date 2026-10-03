"""Fair comparison in turns (tune.fair): drift-proof, noise-aware, goal-scored."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferenceos.config import Settings
from inferenceos.contracts.goal import KeepVerdict
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.optimization.keep import ModeRules
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_model
from inferenceos.tune.fair import (
    CLOCK_METRIC,
    TurnReading,
    compare_fairly,
    drift_balanced,
    score_turns,
    split_by_clock,
)
from inferenceos.tune.measure import DisturbanceWatch, judge_pair, measure_steady
from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy

START = RuntimeConfiguration(context_length=2048, gpu_offload_layers=8)
NEW = START.model_copy(update={"threads": 5})


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "reports_dir": tmp_path / "reports",
        "logs_dir": tmp_path / "logs",
    }
    base.update(overrides)
    return Settings(**base)


def _watch() -> DisturbanceWatch:
    return DisturbanceWatch(logical_cores=12, limit_percent=25.0, max_repeats=0)


def _drifting(gain: float, *, drift_per_call: float = 0.004) -> Any:
    """Speed falls a little with every run (the laptop warms up); NEW is ``gain`` faster
    than START measured at the same moment."""

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        base = 20.0 * (1.0 - drift_per_call * call)
        speed = base * (1.0 + gain) if options.threads == 5 else base
        return SampleResult(decode_tps=speed, total_latency_s=2.0)

    return responder


def _compare(tmp_path: Path, responder: Any, *, effort: str = "balanced", turns: int = 3) -> Any:
    settings = _settings(tmp_path, optimization_mode=effort)
    return compare_fairly(
        harness=ScriptedHarness(responder),
        model=sample_model(),
        workload=get_workload("short-short"),
        start=START,
        candidate=NEW,
        policy=build_test_policy(),
        settings=settings,
        rules=ModeRules.from_settings(settings),
        watch=_watch(),
        card_id="runtime.reduce_threads",
        turns=turns,
        phase="test",
    )


def test_a_real_gain_under_drift_is_kept_where_a_single_early_reading_hides_it(
    tmp_path: Path,
) -> None:
    responder = _drifting(0.06)
    # The old way: one early reading of the start, the candidate measured much later.
    settings = _settings(tmp_path, optimization_mode="balanced")
    harness = ScriptedHarness(responder)
    policy = build_test_policy()
    early = measure_steady(
        harness=harness, model=sample_model(), workload=get_workload("short-short"),
        config=START, policy=policy, phase="early", watch=_watch(),
    )
    for _ in range(12):  # the search goes on for a while; the laptop drifts
        measure_steady(
            harness=harness, model=sample_model(), workload=get_workload("short-short"),
            config=START, policy=policy, phase="filler", watch=_watch(),
        )
    late = measure_steady(
        harness=harness, model=sample_model(), workload=get_workload("short-short"),
        config=NEW, policy=policy, phase="late", watch=_watch(),
    )
    old = judge_pair(
        before=early, after=late, workload=get_workload("short-short"), model=sample_model(),
        policy=policy, settings=settings, rules=ModeRules.from_settings(settings),
        card_id="runtime.reduce_threads",
    )
    assert not old.technically_better  # the stale reading hides the real 6% gain

    fair = _compare(tmp_path, responder)
    assert fair.kept
    assert fair.gain is not None and 0.05 <= fair.gain <= 0.07
    assert all(item.improvements["decode_tokens_per_sec"] > 0 for item in fair.turns)


def test_turns_alternate_which_setup_goes_first(tmp_path: Path) -> None:
    fair = _compare(tmp_path, _drifting(0.06), turns=3)
    assert [item.order for item in fair.turns] == [
        "start,candidate",
        "candidate,start",
        "start,candidate",
    ]
    assert fair.as_confirmation().order[:4] == ["start", "candidate", "candidate", "start"]


def test_no_real_difference_is_not_kept(tmp_path: Path) -> None:
    fair = _compare(tmp_path, _drifting(0.0))
    assert not fair.kept
    assert fair.per_measure.get("decode_tokens_per_sec", 0.0) == 0.0


def test_a_small_gain_follows_the_effort_mode(tmp_path: Path) -> None:
    # 3% is real (every turn agrees) but below Balanced's 5% smallest worthwhile gain;
    # Performance first (1.5%) keeps it.
    balanced = _compare(tmp_path, _drifting(0.03), effort="balanced")
    assert not balanced.kept
    assert balanced.judgement is not None
    assert balanced.judgement.verdict is KeepVerdict.NOT_WORTH_IT
    fast = _compare(tmp_path, _drifting(0.03), effort="performance_first")
    assert fast.kept


def test_a_guarded_measure_that_gets_much_worse_is_refused(tmp_path: Path) -> None:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if options.threads == 5:
            return SampleResult(decode_tps=24.0, total_latency_s=2.0, rss_bytes=1_800_000_000)
        return SampleResult(decode_tps=20.0, total_latency_s=2.0, rss_bytes=1_300_000_000)

    fair = _compare(tmp_path, responder)
    assert not fair.kept
    assert "peak_rss_bytes" in fair.guard_breaches


def test_a_quality_failure_is_never_kept(tmp_path: Path) -> None:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if options.threads == 5:
            return SampleResult(decode_tps=30.0, text="zzz qqq www completely different")
        return SampleResult(decode_tps=20.0)

    fair = _compare(tmp_path, responder)
    assert not fair.quality_passed
    assert not fair.kept


def test_turns_that_disagree_count_as_no_change() -> None:
    settings = Settings()
    rules = ModeRules.from_settings(settings)
    turns = [
        TurnReading(1, "start,candidate", {}, {}, {"decode_tokens_per_sec": 0.08}),
        TurnReading(2, "candidate,start", {}, {}, {"decode_tokens_per_sec": -0.06}),
        TurnReading(3, "start,candidate", {}, {}, {"decode_tokens_per_sec": 0.01}),
    ]
    per_measure, raw, agreeing, breaches, assessment = score_turns(
        turns, rules=rules, noise_floor=0.01
    )
    assert abs(raw["decode_tokens_per_sec"] - (-0.0075)) < 1e-9
    assert agreeing["decode_tokens_per_sec"] is False
    assert per_measure["decode_tokens_per_sec"] == 0.0
    assert not breaches


def test_every_turn_is_recorded_for_the_report(tmp_path: Path) -> None:
    fair = _compare(tmp_path, _drifting(0.06))
    record = fair.record()
    turns = record["turns"]
    assert isinstance(turns, list) and len(turns) == 3
    first = turns[0]
    assert isinstance(first, dict)
    assert first["start"]["decode_tokens_per_sec"] is not None
    assert first["candidate"]["decode_tokens_per_sec"] is not None
    assert record["kept"] is True


def test_steady_drift_cancels_out() -> None:
    # A 6% gain seen through a drift that biases odd and even turns by 2% each way.
    estimate, agree = drift_balanced([0.04, 0.08, 0.04])
    assert abs(estimate - 0.06) < 1e-9 and agree
    estimate, agree = drift_balanced([0.05])
    assert estimate == 0.05 and agree


def test_a_turn_with_unusual_processor_speed_is_left_out() -> None:
    # The new setup always runs ~30 points lower (the setting itself does that): fine.
    # In turn 2 the current setup ran far below its usual speed: heat, left out.
    one = TurnReading(1, "start,candidate", {CLOCK_METRIC: 98.0}, {CLOCK_METRIC: 68.0}, {})
    hot = TurnReading(2, "candidate,start", {CLOCK_METRIC: 70.0}, {CLOCK_METRIC: 67.0}, {})
    three = TurnReading(3, "start,candidate", {CLOCK_METRIC: 97.0}, {CLOCK_METRIC: 69.0}, {})
    unknown = TurnReading(4, "candidate,start", {}, {}, {})
    kept, left_out = split_by_clock([one, hot, three, unknown], tolerance=8.0)
    assert [item.turn for item in kept] == [1, 3, 4]
    assert [item.turn for item in left_out] == [2]


def test_a_gain_that_only_comes_from_throttling_is_not_kept(tmp_path: Path) -> None:
    """In turn 2 the laptop throttled while the current setup was measured, so the new
    setup looks 18% faster in that turn only. That turn is left out; the other turns
    show no difference, so nothing is kept. Without the check the 18% would leak into
    the result (the neighbour averages would both be positive)."""

    def throttled(call: int) -> bool:
        return 10 <= call <= 12  # the current setup's measurement in turn 2

    class _ClockHarness(ScriptedHarness):
        def run_once(self, task: Any, **kwargs: Any) -> Any:
            run = super().run_once(task, **kwargs)
            metrics = run.observation.metrics.model_copy(
                update={"cpu_performance_percent": 70.0 if throttled(len(self.calls)) else 99.0}
            )
            return run.model_copy(
                update={"observation": run.observation.model_copy(update={"metrics": metrics})}
            )

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(decode_tps=17.0 if throttled(call) else 20.0, total_latency_s=2.0)

    settings = _settings(tmp_path, optimization_mode="performance_first")
    policy = build_test_policy(measured_repetitions=2, max_repetitions=2, min_valid_samples=2)
    fair = compare_fairly(
        harness=_ClockHarness(responder),
        model=sample_model(),
        workload=get_workload("short-short"),
        start=START,
        candidate=NEW,
        policy=policy,
        settings=settings,
        rules=ModeRules.from_settings(settings),
        watch=_watch(),
        card_id="runtime.reduce_threads",
        turns=3,
        phase="test",
    )
    assert fair.uneven_turns == [2]
    assert any("left out" in note for note in fair.notes)
    assert fair.turns[1].improvements["decode_tokens_per_sec"] > 0.15
    assert not fair.kept


def test_steady_warming_and_a_setting_that_changes_speed_are_not_flagged() -> None:
    """The 9B prompt batch check (Oct 2026): speeds drift down over the run and the new
    setting runs lower, but the gap stays typical. All four turns must count."""
    readings = [(117.3, 105.3), (108.8, 102.5), (106.5, 80.9), (103.0, 85.6)]
    turns = [
        TurnReading(index + 1, "x", {CLOCK_METRIC: start}, {CLOCK_METRIC: new}, {})
        for index, (start, new) in enumerate(readings)
    ]
    kept, left_out = split_by_clock(turns, tolerance=8.0)
    assert left_out == [] and len(kept) == 4


def test_without_throttling_the_heat_check_never_leaves_anything_out() -> None:
    """Guarantee: when no setup runs more than the tolerance below its own best speed,
    no turn is ever left out, whatever the readings and the gaps between setups."""
    import random

    rng = random.Random(7)
    for _ in range(2000):
        turns_count = rng.randint(2, 6)
        start_base = rng.uniform(60.0, 120.0)
        new_base = rng.uniform(60.0, 120.0)  # a setting may change the speed by itself
        turns = [
            TurnReading(
                index + 1,
                "x",
                {CLOCK_METRIC: start_base + rng.uniform(0.0, 8.0)},
                {CLOCK_METRIC: new_base + rng.uniform(0.0, 8.0)},
                {},
            )
            for index in range(turns_count)
        ]
        kept, left_out = split_by_clock(turns, tolerance=8.0)
        assert left_out == [] and len(kept) == turns_count


def _clock_compare(tmp_path: Path, throttle_calls: set[int], gain: float) -> Any:
    """START and NEW in 3 turns; the measurements whose calls are in
    ``throttle_calls`` run throttled (70% speed reading, 15% slower)."""

    class _ClockHarness(ScriptedHarness):
        def run_once(self, task: Any, **kwargs: Any) -> Any:
            run = super().run_once(task, **kwargs)
            hot = len(self.calls) in throttle_calls
            metrics = run.observation.metrics.model_copy(
                update={"cpu_performance_percent": 70.0 if hot else 99.0}
            )
            return run.model_copy(
                update={"observation": run.observation.model_copy(update={"metrics": metrics})}
            )

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        speed = 20.0 * (1.0 + gain) if options.threads == 5 else 20.0
        if call in throttle_calls:
            speed *= 0.85
        return SampleResult(decode_tps=speed, total_latency_s=2.0)

    settings = _settings(tmp_path, optimization_mode="performance_first")
    policy = build_test_policy(measured_repetitions=2, max_repetitions=2, min_valid_samples=2)
    return compare_fairly(
        harness=_ClockHarness(responder),
        model=sample_model(),
        workload=get_workload("short-short"),
        start=START,
        candidate=NEW,
        policy=policy,
        settings=settings,
        rules=ModeRules.from_settings(settings),
        watch=_watch(),
        card_id="runtime.reduce_threads",
        turns=3,
        phase="test",
    )


def test_a_cool_run_reports_no_heat_effect(tmp_path: Path) -> None:
    fair = _clock_compare(tmp_path, set(), gain=0.06)
    assert fair.kept
    assert fair.uneven_turns == [] and fair.heat_effect == ""
    assert not fair.heat_changed_decision


def test_a_decision_changed_by_heat_is_reported(tmp_path: Path) -> None:
    # The current setup throttles in turn 2 (calls 10-12): counting that turn the new
    # setup looks better; without it there is no gain. The user must be told.
    fair = _clock_compare(tmp_path, {10, 11, 12}, gain=0.0)
    assert fair.uneven_turns == [2]
    assert not fair.kept
    assert fair.heat_changed_decision
    assert fair.verdict_with_all_turns == "kept"
    assert fair.heat_effect.startswith("REJECTED BECAUSE OF HEAT")
    assert fair.record()["heat_changed_decision"] is True


def test_heat_that_does_not_change_the_decision_is_still_noted(tmp_path: Path) -> None:
    # A real 10% gain; the new setup throttles in turn 2 (calls 7-9), which only
    # hides part of the gain. Still kept, and the note says the answer is the same.
    fair = _clock_compare(tmp_path, {7, 8, 9}, gain=0.10)
    assert fair.uneven_turns == [2]
    assert fair.kept and not fair.heat_changed_decision
    assert "decision is the same either way" in fair.heat_effect
