"""Focused tests for workload-aware baseline establishment.

The previous effectiveness run established its baseline in ~4 s using only host metrics
(``memory_pressure``, ``available_ram_bytes``) because the first real generation was not
yet complete. These tests prove the runtime monitor now waits for real workload
observations (latency/TTFT/decode) before the baseline becomes eligible, bounded by a
deterministic max wait, and never fabricates metrics.
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


class _FakeClock:
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
        max_optimizer_overhead_s=120.0,
    )


def _host_metrics(pressure: float = 0.5) -> dict[str, float]:
    return {
        "memory_pressure": pressure,
        "hard_page_faults_per_sec": 0.5,
        "available_ram_bytes": float(8 * 1024**3),
        "total_ram_bytes": float(16 * 1024**3),
        "system_cpu_percent": 25.0,
    }


def _workload_metrics() -> dict[str, float]:
    return {
        "decode_tokens_per_sec": 24.0,
        "total_latency_s": 1.4,
        "ttft_s": 0.25,
        "peak_rss_bytes": 1_300_000_000.0,
    }


def _run(metrics_for: Any, policy: Any) -> RuntimeMonitorResult:
    clock = _FakeClock()
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


def test_baseline_waits_for_real_workload_observations() -> None:
    def metrics_for(count: int, condition: str) -> dict[str, float]:
        values = _host_metrics()
        # Simulate a slow workload: no performance metrics until the 10th sample.
        if count >= 10:
            values.update(_workload_metrics())
        return values

    result = _run(
        metrics_for,
        make_runtime_policy(
            budget=_budget(30.0),
            baseline_workload_samples=2,
            baseline_max_wait_s=25.0,
        ),
    )
    baseline = result.run.baseline
    assert baseline is not None
    # The baseline became eligible only after workload observations arrived.
    assert baseline.observation_count >= 10
    assert "decode_tokens_per_sec" in baseline.metrics
    assert baseline.metrics["decode_tokens_per_sec"].count >= 2
    assert "total_latency_s" in baseline.metrics


def test_baseline_legacy_host_only_when_not_requested() -> None:
    def metrics_for(count: int, condition: str) -> dict[str, float]:
        values = _host_metrics()
        if count >= 10:
            values.update(_workload_metrics())
        return values

    result = _run(
        metrics_for,
        make_runtime_policy(budget=_budget(30.0), baseline_workload_samples=0),
    )
    baseline = result.run.baseline
    assert baseline is not None
    # Legacy behaviour: baseline formed at the host sample count, before workload data.
    assert baseline.observation_count == 4
    assert "decode_tokens_per_sec" not in baseline.metrics


def test_baseline_max_wait_falls_back_and_is_recorded() -> None:
    def metrics_for(count: int, condition: str) -> dict[str, float]:
        return _host_metrics()  # never any workload-performance metric

    result = _run(
        metrics_for,
        make_runtime_policy(
            budget=_budget(10.0),
            baseline_workload_samples=2,
            baseline_max_wait_s=1.5,
        ),
    )
    baseline = result.run.baseline
    assert baseline is not None
    assert "decode_tokens_per_sec" not in baseline.metrics
    assert any("baseline wait ended" in error for error in result.run.errors)


def test_ttft_is_watched_in_the_baseline() -> None:
    def metrics_for(count: int, condition: str) -> dict[str, float]:
        values = _host_metrics()
        values.update(_workload_metrics())
        return values

    result = _run(
        metrics_for,
        make_runtime_policy(budget=_budget(6.0), baseline_workload_samples=2),
    )
    baseline = result.run.baseline
    assert baseline is not None
    assert "ttft_s" in baseline.metrics
    assert baseline.metrics["ttft_s"].mean == 0.25


def test_diagnosis_only_after_workload_aware_baseline() -> None:
    def metrics_for(count: int, condition: str) -> dict[str, float]:
        if condition == "baseline":
            values = _host_metrics(0.5)
            values.update(_workload_metrics())
            return values
        values = _host_metrics(0.9)
        values["hard_page_faults_per_sec"] = 20000.0
        values.update(
            {
                "decode_tokens_per_sec": 9.0,
                "total_latency_s": 3.0,
                "ttft_s": 1.0,
                "peak_rss_bytes": 1_700_000_000.0,
            }
        )
        return values

    result = _run(
        metrics_for,
        make_runtime_policy(
            budget=_budget(12.0),
            baseline_workload_samples=2,
            persistence_samples=3,
            diagnosis_cooldown_s=0.0,
        ),
    )
    run = result.run
    assert run.baseline is not None
    assert "decode_tokens_per_sec" in run.baseline.metrics
    # A real opportunity is only produced after the workload-aware baseline exists.
    assert run.opportunities
    assert run.diagnosis_events >= 1
