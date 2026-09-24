"""Phase 10 runtime monitor engine tests (deterministic)."""

from __future__ import annotations

from typing import Callable

from inferenceos.contracts.enums import ApprovalOutcome, RuntimeState, RuntimeTerminationReason
from inferenceos.contracts.runtime import RuntimeBudget, RuntimePolicy
from inferenceos.runtime.approval import (
    ApprovalProvider,
    AutoRejectApprovalProvider,
    ScriptedApprovalProvider,
)
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.notification import CollectingNotificationSink, NotificationSink
from inferenceos.runtime.telemetry import RuntimeTelemetrySampler
from inferenceos.selftest import sample_hardware
from tests.runtime_helpers import ScriptedSampler, ScriptedWorkload, baseline_and_pressured, make_runtime_policy


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


def _healthy_metrics(sequence: int, condition: str) -> dict[str, float]:
    return {
        "memory_pressure": 0.5,
        "hard_page_faults_per_sec": 0.5,
        "available_ram_bytes": float(8 * 1024**3),
        "total_ram_bytes": float(16 * 1024**3),
        "system_cpu_percent": 25.0,
        "decode_tokens_per_sec": 24.0,
        "total_latency_s": 1.4,
        "peak_rss_bytes": 1_300_000_000.0,
    }


def _small_budget(duration: float = 6.0) -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=duration,
        telemetry_interval_s=0.5,
        observation_window_size=40,
        baseline_samples=4,
        max_deep_analyses=2,
        max_interventions=2,
        max_cumulative_intervention_s=300.0,
        max_optimizer_overhead_s=120.0,
        approval_timeout_s=10.0,
        max_alerts_per_condition=2,
    )


def _monitor(
    workload: ScriptedWorkload,
    sampler: RuntimeTelemetrySampler,
    policy: RuntimePolicy,
    approvals: ApprovalProvider,
    sink: NotificationSink | None = None,
    *,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
) -> RuntimeMonitor:
    return RuntimeMonitor(
        workload=workload,
        sampler=sampler,
        policy=policy,
        approval_provider=approvals,
        notification_sink=sink or CollectingNotificationSink(),
        hardware=sample_hardware(),
        clock=clock,
        sleep=sleep,
    )


def test_healthy_workload_no_intervention() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    policy = make_runtime_policy(budget=_small_budget(6.0))
    sink = CollectingNotificationSink()
    monitor = _monitor(
        workload,
        ScriptedSampler(_healthy_metrics),
        policy,
        AutoRejectApprovalProvider(),
        sink,
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    assert result.termination.reason is RuntimeTerminationReason.DURATION_REACHED
    assert result.run.interventions == []
    assert result.run.recommendations == []
    assert result.run.observations_collected >= 4
    assert result.run.termination.clean is True
    assert sink.notifications == []
    assert workload.snapshot().running is True


def test_transient_anomaly_does_not_trigger_intervention() -> None:
    clock = FakeClock()

    def metrics(sequence: int, condition: str) -> dict[str, float]:
        values = dict(_healthy_metrics(sequence, condition))
        # A single short-lived spike (observation 6) then recovery.
        if condition != "baseline" and sequence == 6:
            values["hard_page_faults_per_sec"] = 30000.0
            values["memory_pressure"] = 0.9
        return values

    workload = ScriptedWorkload()
    policy = make_runtime_policy(budget=_small_budget(8.0))
    monitor = _monitor(
        workload, ScriptedSampler(metrics), policy, AutoRejectApprovalProvider(),
        clock=clock, sleep=clock.sleep,
    )
    result = monitor.run()
    assert result.run.diagnosis_events == 0
    assert result.run.recommendations == []
    assert result.run.interventions == []
    assert result.termination.reason is RuntimeTerminationReason.DURATION_REACHED


def test_persistent_opportunity_runs_diagnosis_and_recommends() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    policy = make_runtime_policy(budget=_small_budget(12.0))
    sink = CollectingNotificationSink()
    monitor = _monitor(
        workload,
        ScriptedSampler(baseline_and_pressured()),
        policy,
        ScriptedApprovalProvider([ApprovalOutcome.REJECTED, ApprovalOutcome.REJECTED]),
        sink,
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    run = result.run
    assert run.diagnosis_events >= 1
    assert run.opportunities
    assert run.recommendations
    assert run.interventions == []  # rejected -> never intervened
    assert any(a.outcome is ApprovalOutcome.REJECTED for a in run.approvals)
    assert run.cooldowns  # anti-thrashing cooldown applied
    assert sink.notifications  # structured notification delivered


def test_repeated_detection_suppressed_by_cooldown() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    # Long cooldown relative to the run so the same condition is not re-alerted.
    policy = make_runtime_policy(
        budget=_small_budget(20.0), intervention_cooldown_s=600.0
    )
    monitor = _monitor(
        workload,
        ScriptedSampler(baseline_and_pressured()),
        policy,
        ScriptedApprovalProvider([ApprovalOutcome.REJECTED]),
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    assert len(result.run.recommendations) == 1
    assert len(result.run.approvals) == 1


def test_workload_config_change_is_detected() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    policy = make_runtime_policy(budget=_small_budget(10.0))

    def metrics(sequence: int, condition: str) -> dict[str, float]:
        # Shift configuration after the baseline phase to simulate a workload change.
        from inferenceos.contracts.experiment import RuntimeConfiguration

        if condition != "baseline" and sequence == 6:
            workload.set_configuration(RuntimeConfiguration(context_length=4096, gpu_offload_layers=99))
        return _healthy_metrics(sequence, condition)

    monitor = _monitor(
        workload, ScriptedSampler(metrics), policy, AutoRejectApprovalProvider(),
        clock=clock, sleep=clock.sleep,
    )
    result = monitor.run()
    assert result.run.diagnosis_events >= 1
    assert any(o.context_changed for o in result.run.opportunities)


def test_external_termination_stops_cleanly() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()

    def metrics(sequence: int, condition: str) -> dict[str, float]:
        if sequence == 7:
            workload.terminate_externally()
        return _healthy_metrics(sequence, condition)

    policy = make_runtime_policy(budget=_small_budget(20.0))
    monitor = _monitor(
        workload, ScriptedSampler(metrics), policy, AutoRejectApprovalProvider(),
        clock=clock, sleep=clock.sleep,
    )
    result = monitor.run()
    assert result.termination.reason is RuntimeTerminationReason.EXTERNAL_TERMINATION
