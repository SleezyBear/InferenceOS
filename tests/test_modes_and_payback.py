"""Step 4: effort mode everywhere, goal mode, and the payback check."""

from __future__ import annotations

from typing import Any

import pytest

from inferenceos.config import Settings
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    AnomalyClass,
    EvidenceStrength,
    GoalMode,
    OptimizationMode,
    OptimizationOutcome,
    PerformanceState,
    Situation,
)
from inferenceos.contracts.experiment import ExperimentComparison, MeasurementDelta
from inferenceos.contracts.goal import KeepVerdict
from inferenceos.contracts.optimization import BenefitEstimate, OptimizationCandidate
from inferenceos.contracts.runtime import RuntimeOpportunity
from inferenceos.loop.knowledge import NEGATIVE_OUTCOMES
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.goals import (
    assess_comparison,
    build_goal_preference,
)
from inferenceos.optimization.keep import ModeRules, judge_keep
from inferenceos.optimization.payback import payback_check, time_saved_s
from inferenceos.optimization.preference import build_optimization_preference
from inferenceos.runtime.economics import evaluate_intervention_economics
from inferenceos.runtime.policy import build_runtime_policy
from tests.experiment_helpers import SampleResult
from tests.knowledge_helpers import base_result, make_service
from tests.loop_helpers import (
    make_loop,
    make_test_experiment_policy,
    patched_paging_report,
    report_for_config,
)
from tests.optimization_helpers import make_candidate, synthetic_context


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _delta(metric: str, relative: float, *, higher: bool, real: bool = True) -> MeasurementDelta:
    improved = (relative > 0) == higher if relative != 0 else False
    return MeasurementDelta(
        metric=metric,
        direction="higher_is_better" if higher else "lower_is_better",
        baseline_mean=10.0,
        candidate_mean=10.0 * (1.0 + relative),
        relative_delta=relative,
        improved=real and improved,
        regressed=real and not improved and relative != 0,
        significant=real,
    )


def _comparison(**deltas: MeasurementDelta) -> ExperimentComparison:
    return ExperimentComparison(deltas=dict(deltas))


def _rules(**settings: Any) -> ModeRules:
    return ModeRules.from_settings(Settings(**settings))


# --------------------------------------------------------------------------- #
# Effort mode                                                                  #
# --------------------------------------------------------------------------- #
def test_effort_mode_sets_before_loading_search_time() -> None:
    limits = {
        "balanced": 400.0,
        "performance_first": 600.0,
        "minimize_cost": 200.0,
    }
    for name, seconds in limits.items():
        effort = build_optimization_preference(Settings(effort_mode=name))
        assert effort.before_loading_search_time_s == seconds


def test_effort_mode_is_the_new_name_and_the_old_setting_still_works() -> None:
    assert (
        build_optimization_preference(Settings(optimization_mode="performance_first")).mode
        is OptimizationMode.PERFORMANCE_FIRST
    )
    both = Settings(optimization_mode="performance_first", effort_mode="minimize-cost")
    assert build_optimization_preference(both).mode is OptimizationMode.MINIMIZE_OPTIMIZATION_COST


def test_before_loading_time_setting_overrides_mode() -> None:
    effort = build_optimization_preference(
        Settings(optimization_before_loading_search_time_s=123.0)
    )
    assert effort.before_loading_search_time_s == 123.0


# --------------------------------------------------------------------------- #
# Goal mode                                                                    #
# --------------------------------------------------------------------------- #
def test_goal_modes() -> None:
    balanced = build_goal_preference(Settings())
    assert balanced.mode is GoalMode.BALANCED
    assert set(balanced.first) == {"decode_tokens_per_sec", "ttft_s"}
    assert "peak_rss_bytes" in balanced.guarded and "hard_page_faults_per_sec" in balanced.guarded

    fastest = build_goal_preference(Settings(goal_mode="fastest_answers"))
    assert set(fastest.first) == {"decode_tokens_per_sec", "ttft_s"}
    assert fastest.guarded == ["peak_rss_bytes"]

    custom = build_goal_preference(
        Settings(goal_mode="custom", goal_custom_measures="paging, writing_speed")
    )
    assert custom.first["hard_page_faults_per_sec"] > custom.first["decode_tokens_per_sec"]
    assert "hard_page_faults_per_sec" not in custom.guarded


@pytest.mark.parametrize(
    "settings",
    [
        {"goal_mode": "quiet_and_cool"},
        {"goal_mode": "custom"},
        {"goal_mode": "custom", "goal_custom_measures": "writing_speed,fan_noise"},
        {"goal_mode": "custom", "goal_custom_measures": "memory,memory"},
        {"goal_guard_limit_percent": 150.0},
    ],
)
def test_invalid_goal_settings_are_refused(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        build_goal_preference(Settings(**settings))


def test_one_first_measure_improving_counts_in_full() -> None:
    goal = build_goal_preference(Settings())
    comparison = _comparison(
        decode_tokens_per_sec=_delta("decode_tokens_per_sec", 0.08, higher=True),
        ttft_s=_delta("ttft_s", 0.0, higher=False),
    )
    assert assess_comparison(comparison, goal).gain == pytest.approx(0.08)


def test_first_measures_are_weighed_against_each_other_in_a_trade_off() -> None:
    goal = build_goal_preference(Settings())
    comparison = _comparison(
        decode_tokens_per_sec=_delta("decode_tokens_per_sec", 0.08, higher=True),
        ttft_s=_delta("ttft_s", 0.08, higher=False),  # first word 8% slower
    )
    assert assess_comparison(comparison, goal).gain == pytest.approx(0.0)


def test_change_inside_the_noise_counts_as_zero() -> None:
    goal = build_goal_preference(Settings())
    comparison = _comparison(
        decode_tokens_per_sec=_delta("decode_tokens_per_sec", 0.04, higher=True, real=False)
    )
    assert assess_comparison(comparison, goal).gain == 0.0


def test_guarded_measure_worse_beyond_limit_is_a_breach() -> None:
    goal = build_goal_preference(Settings())
    comparison = _comparison(
        decode_tokens_per_sec=_delta("decode_tokens_per_sec", 0.20, higher=True),
        peak_rss_bytes=_delta("peak_rss_bytes", 0.30, higher=False),  # 30% more memory
    )
    assessment = assess_comparison(comparison, goal)
    assert assessment.guard_breaches == ["peak_rss_bytes"]
    # Fastest answers guards memory too; a small increase is within the 10% limit.
    small = _comparison(
        decode_tokens_per_sec=_delta("decode_tokens_per_sec", 0.20, higher=True),
        peak_rss_bytes=_delta("peak_rss_bytes", 0.05, higher=False),
    )
    assert assess_comparison(small, goal).guard_breaches == []


# --------------------------------------------------------------------------- #
# Payback check                                                                #
# --------------------------------------------------------------------------- #
def test_time_saved_formula() -> None:
    # 25% faster answering saves 20% of the answering time.
    assert time_saved_s(0.25, 3600.0) == pytest.approx(720.0)
    assert time_saved_s(0.0, 3600.0) == 0.0


def test_payback_uses_the_answering_hours_settings() -> None:
    settings = Settings()
    before = payback_check(
        gain=0.17, cost_terms={"search": 180.0}, situation=Situation.BEFORE_LOADING,
        settings=settings,
    )
    assert before.answering_s == 36000.0 and before.pays_back is True
    during = payback_check(
        gain=0.17, cost_terms={"search": 180.0}, situation=Situation.WHILE_RUNNING,
        settings=settings,
    )
    assert during.answering_s == 3600.0 and during.pays_back is True
    tiny = payback_check(
        gain=0.01, cost_terms={"search": 180.0}, situation=Situation.WHILE_RUNNING,
        settings=settings,
    )
    assert tiny.pays_back is False
    unknown = payback_check(
        gain=None, cost_terms={}, situation=Situation.WHILE_RUNNING, settings=settings
    )
    assert unknown.pays_back is None


def test_payback_hours_are_settings() -> None:
    stricter = Settings(payback_answering_hours_while_running=0.25)
    check = payback_check(
        gain=0.05, cost_terms={"restart": 60.0}, situation=Situation.WHILE_RUNNING,
        settings=stricter,
    )
    assert check.answering_s == 900.0 and check.pays_back is False


# --------------------------------------------------------------------------- #
# The shared keep rule                                                         #
# --------------------------------------------------------------------------- #
def _judge(comparison: ExperimentComparison, **kwargs: Any) -> Any:
    rules = _rules(**kwargs.pop("settings", {}))
    return judge_keep(
        comparison,
        technically_better=kwargs.pop("technically_better", True),
        effort=rules.effort,
        goal=rules.goal,
        situation=kwargs.pop("situation", Situation.BEFORE_LOADING),
        settings=rules.settings,
        remaining_cost_terms=kwargs.pop("remaining", {}),
        performance_state=PerformanceState.NORMAL,
    )


def _speed(gain: float) -> ExperimentComparison:
    return _comparison(
        decode_tokens_per_sec=_delta("decode_tokens_per_sec", gain, higher=True)
    )


def test_keep_rule_verdicts() -> None:
    assert _judge(_speed(0.17)).verdict is KeepVerdict.KEPT
    # Below balanced's 5%: measured better, not worth it.
    assert _judge(_speed(0.03)).verdict is KeepVerdict.NOT_WORTH_IT
    # Performance first accepts 3%.
    assert (
        _judge(_speed(0.03), settings={"effort_mode": "performance_first"}).verdict
        is KeepVerdict.KEPT
    )
    # Does not pay back while running when a long restart is still to come.
    verdict = _judge(
        _speed(0.06), situation=Situation.WHILE_RUNNING, remaining={"restart": 600.0}
    )
    assert verdict.verdict is KeepVerdict.NOT_WORTH_IT
    assert verdict.payback is not None and verdict.payback.pays_back is False
    # The experiment itself did not show a safe improvement: never rescued.
    assert _judge(_speed(0.5), technically_better=False).verdict is KeepVerdict.NOT_BETTER
    # No gain on the goal's measures (for example a memory-only change under balanced).
    memory_only = _comparison(peak_rss_bytes=_delta("peak_rss_bytes", -0.3, higher=False))
    assert _judge(memory_only).verdict is KeepVerdict.NOT_BETTER
    # ... but a custom goal that ranks memory first keeps it.
    assert (
        _judge(memory_only, settings={"goal_mode": "custom", "goal_custom_measures": "memory"}).verdict
        is KeepVerdict.KEPT
    )


def test_keep_rule_refuses_a_guard_breach() -> None:
    comparison = _comparison(
        decode_tokens_per_sec=_delta("decode_tokens_per_sec", 0.30, higher=True),
        hard_page_faults_per_sec=_delta("hard_page_faults_per_sec", 0.50, higher=False),
    )
    judgement = _judge(comparison)
    assert judgement.verdict is KeepVerdict.NOT_BETTER
    assert "paging" in judgement.reasons[-1]


# --------------------------------------------------------------------------- #
# The watcher: the payback check replaces the fixed 60-second guess             #
# --------------------------------------------------------------------------- #
def _opportunity() -> RuntimeOpportunity:
    return RuntimeOpportunity(
        opportunity_id="opp",
        anomaly=AnomalyClass.PERSISTENT,
        condition_key="k",
        kind="gpu_underutilization",
        persistence_samples=3,
        deviation_score=0.9,
        confidence=0.85,
        strength=EvidenceStrength.MEDIUM,
        meaningful=True,
    )


def _layer_candidate(gain: float) -> OptimizationCandidate:
    ctx = synthetic_context(BottleneckKind.GPU_UNDERUTILIZATION)
    candidate = make_candidate(
        ctx, action_id="runtime.increase_gpu_offload_layers", cost=20.0, overhead=0.0
    )
    return candidate.model_copy(
        update={
            "benefit": BenefitEstimate(
                metric="decode_tokens_per_sec",
                direction="increase",
                unit="tokens/sec",
                known=True,
                magnitude_low=gain,
                magnitude_high=gain,
                confidence=0.85,
            )
        }
    )


def _pre_test(gain: float, *, rules: ModeRules | None, sunk_search_s: float) -> Any:
    candidate = _layer_candidate(gain)
    return evaluate_intervention_economics(
        _opportunity(),
        candidate=candidate,
        action=build_default_registry().get(candidate.action_id),
        policy=build_runtime_policy(Settings()),
        measured_overhead_s=sunk_search_s,
        optimization_preference=build_optimization_preference(Settings()),
        performance_state=PerformanceState.NORMAL,
        mode_rules=rules,
    )


def test_layer_gain_after_a_long_search_is_worth_testing_in_balanced() -> None:
    # The laptop case: a 17% layer gain found by a 3-minute search, default settings.
    old = _pre_test(0.17, rules=None, sunk_search_s=180.0)
    assert old.beneficial is False  # the old fixed guess rejected it
    new = _pre_test(0.17, rules=_rules(), sunk_search_s=180.0)
    assert new.beneficial is True
    assert new.keep_verdict == KeepVerdict.KEPT.value
    assert new.payback is not None and new.payback.pays_back is True
    # Time already spent searching is never charged again.
    assert "search" not in new.payback.cost_terms
    assert new.payback.situation is Situation.WHILE_RUNNING


def test_small_layer_gain_is_not_worth_testing_in_balanced() -> None:
    economics = _pre_test(0.02, rules=_rules(), sunk_search_s=180.0)
    assert economics.beneficial is False
    assert economics.keep_verdict == KeepVerdict.NOT_WORTH_IT.value


def test_remembered_test_time_replaces_the_estimate() -> None:
    candidate = _layer_candidate(0.17)
    economics = evaluate_intervention_economics(
        _opportunity(),
        candidate=candidate,
        action=build_default_registry().get(candidate.action_id),
        policy=build_runtime_policy(Settings()),
        optimization_preference=build_optimization_preference(Settings()),
        performance_state=PerformanceState.NORMAL,
        mode_rules=_rules(),
        remembered_test_cost_s=42.0,
    )
    assert economics.payback is not None
    assert economics.payback.cost_terms["check_run"] == 42.0


# --------------------------------------------------------------------------- #
# The before-loading loop                                                      #
# --------------------------------------------------------------------------- #
def _mmap_responder(*, rss: int, decode: float) -> Any:
    def respond(task: Any, options: Any, call: int) -> SampleResult:
        if options.use_mmap:
            return SampleResult(rss_bytes=rss, hard_faults_per_sec=1.0, decode_tps=decode)
        return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)

    return respond


def _run_loop(responder: Any, rules: ModeRules | None, *, min_effect: float = 0.05) -> Any:
    report = patched_paging_report()
    stack = make_loop(
        responder=responder,
        report=report,
        observer=lambda config: report_for_config(report, config),
        mode_rules=rules,
        experiment_policy=make_test_experiment_policy(
            min_effect=min_effect, noise_tolerance=min(0.10, min_effect)
        ),
    )
    return stack, stack.loop.run(report, report_path="<step4>")


def test_loop_keeps_a_real_speed_gain_under_balanced() -> None:
    stack, result = _run_loop(_mmap_responder(rss=900_000_000, decode=26.0), _rules())
    kept = [i for i in result.run.iterations if i.retained]
    assert kept and kept[0].selected_action_id == "runtime.enable_mmap"
    assert any("keep rule" in note and "kept" in note for note in kept[0].notes)


def test_loop_drops_a_memory_only_gain_under_balanced_but_keeps_it_when_memory_is_ranked() -> None:
    responder = _mmap_responder(rss=900_000_000, decode=20.0)
    # Without mode rules (older behaviour) the memory-only gain is kept.
    _, old = _run_loop(responder, None)
    assert old.changed is True
    # Balanced puts speed and first word first: memory alone is not "better".
    stack, balanced = _run_loop(responder, _rules())
    assert balanced.changed is False
    # A custom goal that ranks memory first keeps it.
    _, custom = _run_loop(responder, _rules(goal_mode="custom", goal_custom_measures="memory"))
    assert custom.changed is True


def test_loop_records_worked_not_worth_it_and_it_is_not_a_bad_result() -> None:
    # 3% faster: a real improvement (the experiment counts changes from 1% here), but
    # below balanced's 5% smallest worthwhile gain.
    stack, result = _run_loop(
        _mmap_responder(rss=1_300_000_000, decode=20.6), _rules(), min_effect=0.01
    )
    assert result.changed is False
    records = stack.service.history("runtime.enable_mmap")
    assert records
    assert records[-1].outcome is OptimizationOutcome.NOT_WORTH_IT
    assert records[-1].keep is False
    assert OptimizationOutcome.NOT_WORTH_IT not in NEGATIVE_OUTCOMES


def test_loop_records_a_gate_drop_truthfully() -> None:
    # 3% faster, but the experiment only counts changes from 5% as real, so the loop's
    # own gate drops it. The saved result must not claim it was kept.
    stack, result = _run_loop(_mmap_responder(rss=1_300_000_000, decode=20.6), _rules())
    assert result.changed is False
    for record in stack.service.history("runtime.enable_mmap"):
        assert not (record.keep and record.outcome is OptimizationOutcome.IMPROVED)


# --------------------------------------------------------------------------- #
# Saved results                                                                #
# --------------------------------------------------------------------------- #
def test_remembered_test_cost_is_the_median_of_this_machines_tests() -> None:
    service = make_service()
    assert service.remembered_test_cost_s("runtime.enable_mmap") is None
    template = base_result()
    for index, seconds in enumerate((30.0, 90.0, 60.0)):
        request = template.request.model_copy(  # type: ignore[union-attr]
            update={"experiment_id": f"exp-cost-{index}"}
        )
        overhead = template.overhead.model_copy(update={"total_s": seconds})  # type: ignore[union-attr]
        service.persist_result(
            template.model_copy(
                update={
                    "experiment_id": f"exp-cost-{index}",
                    "request": request,
                    "overhead": overhead,
                }
            )
        )
    assert service.remembered_test_cost_s("runtime.enable_mmap") == 60.0


# --------------------------------------------------------------------------- #
# The laptop case end to end (scripted): default settings, balanced effort     #
# --------------------------------------------------------------------------- #
def test_watcher_finds_and_keeps_a_layer_gain_with_default_settings() -> None:
    from inferenceos.contracts.enums import ApprovalOutcome
    from inferenceos.runtime.approval import ScriptedApprovalProvider
    from inferenceos.runtime.monitor import RuntimeMonitor
    from inferenceos.runtime.policy import build_runtime_budget
    from inferenceos.runtime.verification import ControlledGpuUnderutilizationDetector
    from inferenceos.selftest import sample_hardware
    from tests.experiment_helpers import ScriptedHarness, build_test_policy
    from tests.test_optimization_policy import HealthyGpuSampler, _FakeClock, _gpu_workload

    settings = Settings()  # defaults: balanced effort, balanced goal, 1 h while running
    clock = _FakeClock()

    def slow_real_gain(task: Any, options: Any, call: int) -> SampleResult:
        # Every measurement takes 20 s of (fake) time, so the layer search itself
        # takes longer than the watcher's 300 s interruption budget, as on the laptop.
        clock.t += 20.0
        layers = getattr(options, "gpu_offload_layers", 0) or 0
        if layers <= 8:
            return SampleResult(decode_tps=20.0, total_latency_s=2.0)
        return SampleResult(decode_tps=23.4, total_latency_s=1.8)  # +17%

    workload = _gpu_workload()
    policy = build_runtime_policy(
        settings, budget=build_runtime_budget(settings, max_duration_s=5000.0)
    )
    assert policy.budget.max_cumulative_intervention_s == 300.0
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=HealthyGpuSampler(),  # type: ignore[arg-type]
        harness=ScriptedHarness(slow_real_gain),
        policy=policy,
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVED] * 2),
        detector=ControlledGpuUnderutilizationDetector(),  # type: ignore[arg-type]
        hardware=sample_hardware(),
        experiment_policy=build_test_policy(min_effect=0.01, noise_tolerance=0.01),
        settings=settings,
        clock=clock,
        sleep=clock.sleep,
    )
    run = monitor.run().run
    assert run.gpu_searches and run.gpu_searches[0].selection.selected_layers is not None
    assert run.overhead.gpu_search_s > 300.0 or run.budget.cumulative_intervention_s > 300.0
    assert run.interventions, "the gain must reach the apply-and-check step"
    kept = run.interventions[0]
    assert kept.retained is True
    assert kept.retention_economics is not None
    assert kept.retention_economics.keep_verdict == KeepVerdict.KEPT.value
    assert workload.configuration().gpu_offload_layers == run.gpu_searches[0].selection.selected_layers
