"""Focused tests for insufficient-evidence opportunity suppression/aggregation.

A condition repeatedly observed but diagnosed non-actionable must not create an
unbounded stream of identical opportunity records; it is aggregated with audit counters
and only re-triggered when the evidence materially changes, telemetry availability
changes, the bounded cooldown expires, or the condition clears and reappears.
"""

from __future__ import annotations

from typing import Any

from inferenceos.contracts.runtime import RuntimeBudget, RuntimeMonitorResult
from inferenceos.runtime.approval import AutoRejectApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.notification import CollectingNotificationSink
from inferenceos.runtime.verification import ScriptedRuntimeSampler, ScriptedWorkload
from inferenceos.selftest import sample_hardware
from tests.runtime_helpers import make_runtime_policy


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


def _budget(duration: float) -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=duration,
        telemetry_interval_s=0.5,
        observation_window_size=40,
        baseline_samples=4,
        max_interventions=2,
        max_optimizer_overhead_s=120.0,
    )


def _metrics(decode: float, latency: float, gpu: float | None = None) -> dict[str, float]:
    values = {
        "memory_pressure": 0.5,
        "hard_page_faults_per_sec": 0.5,
        "available_ram_bytes": float(8 * 1024**3),
        "total_ram_bytes": float(16 * 1024**3),
        "system_cpu_percent": 25.0,
        "decode_tokens_per_sec": decode,
        "total_latency_s": latency,
        "peak_rss_bytes": 1_300_000_000.0,
    }
    if gpu is not None:
        values["gpu_utilization_percent"] = gpu
    return values


def _run(metrics_for: Any, duration: float) -> RuntimeMonitorResult:
    clock = _Clock()
    policy = make_runtime_policy(
        budget=_budget(duration),
        persistence_samples=3,
        diagnosis_cooldown_s=0.0,
    )
    monitor = RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=ScriptedRuntimeSampler(metrics_for),
        policy=policy,
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
        clock=clock,
        sleep=clock.sleep,
    )
    return monitor.run()


def _constant_degraded(sequence: int, condition: str) -> dict[str, float]:
    if condition == "baseline":
        return _metrics(24.0, 1.4)
    return _metrics(9.0, 1.4)


def test_insufficient_evidence_opportunities_are_suppressed() -> None:
    result = _run(_constant_degraded, 30.0)
    run = result.run
    # Many samples would previously have appended many opportunities.
    assert run.observations_collected > 20
    assert len(run.opportunities) <= 2
    assert len(run.opportunity_assessments) == 1
    assessment = run.opportunity_assessments[0]
    assert assessment.status == "insufficient_evidence"
    assert assessment.metrics == ["decode_tokens_per_sec"]
    assert assessment.suppressed_count > 0
    assert assessment.first_seen <= assessment.last_seen
    assert "insufficient" in assessment.diagnosis_status.lower()


def test_suppression_does_not_create_candidate_or_action() -> None:
    result = _run(_constant_degraded, 30.0)
    run = result.run
    assert run.interventions == []
    assert run.approvals == []
    assert run.recommendations == []
    assert run.opportunity_assessments[0].suppressed_count > 0


def test_material_evidence_change_retriggers() -> None:
    def metrics_for(sequence: int, condition: str) -> dict[str, float]:
        if condition == "baseline":
            return _metrics(24.0, 1.4)
        # First a modest drop, then a materially larger drop (score delta > 0.15).
        return _metrics(16.0, 1.4) if sequence <= 14 else _metrics(9.0, 1.4)

    result = _run(metrics_for, 40.0)
    assert result.run.diagnosis_events >= 2
    assert len(result.run.opportunities) >= 2


def test_new_telemetry_availability_retriggers() -> None:
    def metrics_for(sequence: int, condition: str) -> dict[str, float]:
        if condition == "baseline":
            return _metrics(24.0, 1.4)
        # GPU telemetry becomes available mid-run: a genuinely new evidence signature.
        if sequence >= 14:
            return _metrics(9.0, 1.4, gpu=12.0)
        return _metrics(9.0, 1.4)

    result = _run(metrics_for, 40.0)
    assert result.run.diagnosis_events >= 2


def test_condition_clears_and_reappears_retriggers() -> None:
    def metrics_for(sequence: int, condition: str) -> dict[str, float]:
        if condition == "baseline":
            return _metrics(24.0, 1.4)
        if 10 <= sequence <= 22:
            return _metrics(24.0, 1.4)  # condition clears
        return _metrics(9.0, 1.4)

    result = _run(metrics_for, 50.0)
    assert result.run.diagnosis_events >= 2


def test_cooldown_and_suppression_remain_safe() -> None:
    result = _run(_constant_degraded, 30.0)
    run = result.run
    # Suppression must never enable an unattended action without approval.
    assert run.interventions == []
    assert all(a.outcome.value in {"rejected", "not_required"} for a in run.approvals)
