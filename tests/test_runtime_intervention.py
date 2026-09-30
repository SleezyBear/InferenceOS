"""Phase 10 runtime approval / intervention / quality / resume tests."""

from __future__ import annotations

from inferenceos.contracts.enums import (
    ApprovalOutcome,
    ExperimentDecisionKind,
    QualityStatus,
    RuntimeState,
    RuntimeTerminationReason,
)
from inferenceos.contracts.runtime import RuntimeBudget, RuntimePolicy
from inferenceos.experiment.measurement import MeasurementHarness
from inferenceos.runtime.approval import (
    ApprovalProvider,
    ScriptedApprovalProvider,
    TimeoutApprovalProvider,
)
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.telemetry import RuntimeTelemetrySampler
from inferenceos.selftest import sample_hardware
from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy
from tests.runtime_helpers import ScriptedSampler, ScriptedWorkload, baseline_and_pressured, make_runtime_policy


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


def _budget(duration: float = 12.0) -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=duration,
        telemetry_interval_s=0.5,
        observation_window_size=40,
        baseline_samples=4,
        max_deep_analyses=2,
        max_interventions=2,
        max_cumulative_intervention_s=600.0,
        max_optimizer_overhead_s=120.0,
        approval_timeout_s=10.0,
        max_alerts_per_condition=2,
    )


def _improving_responder(task, options, call):  # type: ignore[no-untyped-def]
    if options.use_mmap:
        return SampleResult(
            rss_bytes=900_000_000,
            hard_faults_per_sec=1.0,
            decode_tps=28.0,
            text="A key value cache stores attention state.",
        )
    return SampleResult(
        rss_bytes=1_300_000_000,
        hard_faults_per_sec=5.0,
        decode_tps=20.0,
        text="A key value cache stores attention state.",
    )


def _quality_regression_responder(task, options, call):  # type: ignore[no-untyped-def]
    if options.use_mmap:
        return SampleResult(
            rss_bytes=900_000_000,
            hard_faults_per_sec=1.0,
            decode_tps=30.0,
            text="Completely unrelated output about a totally different subject entirely.",
        )
    return SampleResult(
        rss_bytes=1_300_000_000,
        hard_faults_per_sec=5.0,
        decode_tps=20.0,
        text="A key value cache stores attention state.",
    )


def _monitor(
    workload: ScriptedWorkload,
    policy: RuntimePolicy,
    approvals: ApprovalProvider,
    harness: MeasurementHarness,
    clock: FakeClock,
    sampler: RuntimeTelemetrySampler | None = None,
) -> RuntimeMonitor:
    return RuntimeMonitor(
        workload=workload,
        sampler=sampler or ScriptedSampler(baseline_and_pressured()),
        harness=harness,
        policy=policy,
        approval_provider=approvals,
        hardware=sample_hardware(),
        experiment_policy=build_test_policy(),
        clock=clock,
        sleep=clock.sleep,
    )


def test_approved_intervention_is_checkpointed_validated_and_resumed() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    pressured = baseline_and_pressured()

    def metrics(sequence: int, condition: str) -> dict[str, float]:
        # The condition persists until the optimization is retained, then recovers.
        if condition == "baseline" or workload.configuration().use_mmap is True:
            return pressured(sequence, "baseline")
        return pressured(sequence, condition)

    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED])
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget(30.0)), approvals,
        ScriptedHarness(_improving_responder), clock,
        sampler=ScriptedSampler(metrics),
    )
    result = monitor.run()
    run = result.run
    assert run.interventions, "an approved intervention should have executed"
    intervention = run.interventions[0]
    assert intervention.retained is True
    assert intervention.decision is ExperimentDecisionKind.IMPROVED
    assert intervention.quality_status is QualityStatus.OK
    assert intervention.resume_verified is True
    assert run.checkpoints and run.checkpoints[0].verified is True
    assert workload.pause_calls >= 1
    assert workload.resume_calls >= 1
    assert run.baselines and run.baselines[-1].source == "post_intervention"
    assert workload.configuration().use_mmap is True


def test_quality_regression_prevents_retention_and_restores() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED])
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget()), approvals,
        ScriptedHarness(_quality_regression_responder), clock,
    )
    result = monitor.run()
    run = result.run
    assert run.interventions
    intervention = run.interventions[0]
    assert intervention.retained is False
    assert intervention.quality_status is QualityStatus.QUALITY_REGRESSION
    assert intervention.resume_verified is True
    # Baseline configuration restored (not the degraded candidate).
    assert workload.configuration().use_mmap is False


def test_approval_timeout_does_not_authorize_intervention() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget()), TimeoutApprovalProvider(),
        ScriptedHarness(_improving_responder), clock,
    )
    result = monitor.run()
    assert result.run.interventions == []
    assert any(a.outcome is ApprovalOutcome.TIMEOUT for a in result.run.approvals)
    assert result.run.cooldowns


def test_stale_approval_cannot_authorize_changed_proposal() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED], mismatch_proposal=True)
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget()), approvals,
        ScriptedHarness(_improving_responder), clock,
    )
    result = monitor.run()
    assert result.run.interventions == []
    assert any(a.outcome is ApprovalOutcome.STALE for a in result.run.approvals)


def test_stale_context_approval_rejected() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED], mismatch_context=True)
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget()), approvals,
        ScriptedHarness(_improving_responder), clock,
    )
    result = monitor.run()
    assert result.run.interventions == []
    assert any(a.outcome is ApprovalOutcome.STALE for a in result.run.approvals)


def test_deceptive_resume_is_a_safety_terminal() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload(deceptive_resume=True)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED])
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget(30.0)), approvals,
        ScriptedHarness(_improving_responder), clock,
    )
    result = monitor.run()
    assert result.final_state is RuntimeState.TERMINAL_SAFE
    assert result.run.interventions
    assert result.run.interventions[0].resume_verified is False


def test_declined_approval_records_and_cooldowns() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    approvals = ScriptedApprovalProvider([ApprovalOutcome.REJECTED])
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget()), approvals,
        ScriptedHarness(_improving_responder), clock,
    )
    result = monitor.run()
    assert result.run.interventions == []
    assert any(a.outcome is ApprovalOutcome.REJECTED for a in result.run.approvals)
    assert result.run.cooldowns
    assert workload.snapshot().running is True
