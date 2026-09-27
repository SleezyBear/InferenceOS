"""Focused tests for target-bound monitor lifetime and adaptive monitoring modes.

These tests prove the architectural correction: the monitor's lifetime is tied to the
target runtime (not to accumulated monitoring overhead), excessive overhead reduces
sampling intensity instead of ending monitoring, and mode transitions are recorded.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from inferenceos.contracts.enums import MonitoringMode, RuntimeTerminationReason
from inferenceos.contracts.runtime import RuntimeBudget, RuntimeMonitorResult
from inferenceos.runtime.approval import AutoRejectApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.notification import CollectingNotificationSink
from inferenceos.runtime.verification import (
    ScriptedRuntimeSampler,
    ScriptedWorkload,
    baseline_and_pressured_metrics,
)
from inferenceos.selftest import sample_hardware
from tests.runtime_helpers import make_runtime_policy


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


MetricsFor = Callable[[int, str], dict[str, float]]


class _CostedSampler(ScriptedRuntimeSampler):
    """Scripted sampler that can charge (fake) sampling time to model overhead."""

    def __init__(self, metrics_for: MetricsFor, clock: _FakeClock, cost_fn: Callable[[int], float]) -> None:
        super().__init__(metrics_for)
        self._clock = clock
        self._cost_fn = cost_fn

    def sample(self, workload: Any, *, sequence: int, condition: str) -> Any:
        observation = super().sample(workload, sequence=sequence, condition=condition)
        cost = self._cost_fn(self.count)
        if cost > 0:
            self._clock.sleep(cost)
        return observation


def _host_metrics() -> dict[str, float]:
    return {
        "memory_pressure": 0.5,
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


def _healthy(sequence: int, condition: str) -> dict[str, float]:
    values = _host_metrics()
    values.update(_workload_metrics())
    return values


def _budget(duration: float) -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=duration,
        telemetry_interval_s=0.5,
        observation_window_size=40,
        baseline_samples=4,
        max_interventions=2,
        max_cumulative_intervention_s=300.0,
        max_optimizer_overhead_s=120.0,
    )


def _monitor(sampler: Any, policy: Any, workload: Any, clock: _FakeClock) -> RuntimeMonitor:
    return RuntimeMonitor(
        workload=workload,
        sampler=sampler,
        policy=policy,
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
        clock=clock,
        sleep=clock.sleep,
    )


def _modes(result: RuntimeMonitorResult) -> list[str]:
    return [event.mode.value for event in result.run.monitoring_modes]


# --------------------------------------------------------------------------- #
# Lifecycle                                                                     #
# --------------------------------------------------------------------------- #
def test_target_alive_keeps_monitor_alive_for_full_session() -> None:
    clock = _FakeClock()
    workload = ScriptedWorkload()
    monitor = _monitor(ScriptedRuntimeSampler(_healthy), make_runtime_policy(budget=_budget(6.0)), workload, clock)
    result = monitor.run()
    assert workload.start_calls >= 1  # target started
    assert result.termination.reason is RuntimeTerminationReason.DURATION_REACHED
    assert result.run.observations_collected >= 4


def test_target_exit_terminates_monitor_without_orphan() -> None:
    clock = _FakeClock()
    workload = ScriptedWorkload()

    def metrics(sequence: int, condition: str) -> dict[str, float]:
        if sequence == 8:
            workload.stop()
        return _healthy(sequence, condition)

    before = threading.active_count()
    monitor = _monitor(ScriptedRuntimeSampler(metrics), make_runtime_policy(budget=_budget(30.0)), workload, clock)
    result = monitor.run()
    assert result.termination.reason is RuntimeTerminationReason.TARGET_EXITED
    assert result.termination.clean is True
    # The monitor is synchronous: no background monitor thread survives the run.
    assert threading.active_count() <= before + 1


class _UnrecoverableWorkload(ScriptedWorkload):
    def snapshot(self) -> Any:
        return super().snapshot().model_copy(update={"unrecoverable": True})


def test_unrecoverable_target_terminates_safely() -> None:
    clock = _FakeClock()
    monitor = _monitor(
        ScriptedRuntimeSampler(_healthy),
        make_runtime_policy(budget=_budget(10.0)),
        _UnrecoverableWorkload(),
        clock,
    )
    result = monitor.run()
    assert result.termination.reason is RuntimeTerminationReason.SAFETY_TERMINAL
    assert result.termination.safety is True


def test_external_termination_still_terminates() -> None:
    clock = _FakeClock()
    workload = ScriptedWorkload()

    def metrics(sequence: int, condition: str) -> dict[str, float]:
        if sequence == 6:
            workload.terminate_externally()
        return _healthy(sequence, condition)

    monitor = _monitor(ScriptedRuntimeSampler(metrics), make_runtime_policy(budget=_budget(20.0)), workload, clock)
    result = monitor.run()
    assert result.termination.reason is RuntimeTerminationReason.EXTERNAL_TERMINATION


# --------------------------------------------------------------------------- #
# Monitoring modes                                                              #
# --------------------------------------------------------------------------- #
def test_excessive_overhead_enters_low_cost_and_does_not_terminate() -> None:
    clock = _FakeClock()
    policy = make_runtime_policy(
        budget=_budget(400.0),
        monitor_mode_overhead_fraction_high=0.10,
        monitor_mode_overhead_fraction_low=0.02,
        monitor_mode_active_window_s=1.0,
        monitor_mode_low_cost_interval_s=4.0,
        monitor_mode_low_cost_scan_interval_s=60.0,
    )
    sampler = _CostedSampler(_healthy, clock, lambda n: 30.0 if n <= 6 else 0.0)
    monitor = _monitor(sampler, policy, ScriptedWorkload(), clock)
    result = monitor.run()
    modes = _modes(result)
    assert MonitoringMode.LOW_COST.value in modes
    # High accumulated overhead must NOT be a termination reason.
    assert result.termination.reason is RuntimeTerminationReason.DURATION_REACHED
    assert result.run.overhead.optimizer_overhead_s > 120.0
    # Once sampling becomes cheap again, intensity recovers (LOW_COST -> NORMAL/ACTIVE).
    low_index = modes.index(MonitoringMode.LOW_COST.value)
    assert any(m in (MonitoringMode.NORMAL.value, MonitoringMode.ACTIVE.value) for m in modes[low_index + 1 :])
    event = next(e for e in result.run.monitoring_modes if e.mode is MonitoringMode.LOW_COST)
    assert event.sampling_interval_s == 4.0
    assert event.expensive_scan_interval_s == 60.0
    assert event.reason


def test_low_cost_retains_essential_observations() -> None:
    clock = _FakeClock()
    policy = make_runtime_policy(
        budget=_budget(400.0),
        monitor_mode_overhead_fraction_high=0.10,
        monitor_mode_low_cost_interval_s=4.0,
    )
    sampler = _CostedSampler(_healthy, clock, lambda n: 10.0 if n <= 20 else 0.0)
    monitor = _monitor(sampler, policy, ScriptedWorkload(), clock)
    result = monitor.run()
    # Cheap workload/host observations continue throughout, including in LOW_COST: the
    # last window and the resource-impact summary still carry decode throughput.
    assert result.run.observations_collected > 20
    assert result.run.monitoring_mode_seconds.get(MonitoringMode.LOW_COST.value, 0.0) > 0.0
    assert result.run.resource_impact.decode_tps_mean is not None


def test_sustained_inactivity_enters_dormant() -> None:
    clock = _FakeClock()

    def host_only(sequence: int, condition: str) -> dict[str, float]:
        return _host_metrics()

    policy = make_runtime_policy(
        budget=_budget(8.0),
        monitor_mode_dormant_after_s=2.0,
        monitor_mode_active_window_s=0.0,
    )
    monitor = _monitor(ScriptedRuntimeSampler(host_only), policy, ScriptedWorkload(), clock)
    result = monitor.run()
    assert MonitoringMode.DORMANT.value in _modes(result)
    assert result.termination.reason is RuntimeTerminationReason.DURATION_REACHED
    events = result.run.monitoring_modes
    dormant = next(e for e in events if e.mode is MonitoringMode.DORMANT)
    assert "no target workload activity" in dormant.reason


def test_workload_resumption_escalates_from_dormant() -> None:
    clock = _FakeClock()

    def metrics(sequence: int, condition: str) -> dict[str, float]:
        values = _host_metrics()
        if sequence >= 8:  # workload resumes after an idle period
            values.update(_workload_metrics())
        return values

    policy = make_runtime_policy(
        budget=_budget(40.0),
        monitor_mode_dormant_after_s=1.0,
        monitor_mode_active_window_s=10.0,
    )
    monitor = _monitor(ScriptedRuntimeSampler(metrics), policy, ScriptedWorkload(), clock)
    result = monitor.run()
    modes = _modes(result)
    assert MonitoringMode.DORMANT.value in modes
    dormant_index = modes.index(MonitoringMode.DORMANT.value)
    assert MonitoringMode.ACTIVE.value in modes[dormant_index + 1 :]


# --------------------------------------------------------------------------- #
# Economics                                                                     #
# --------------------------------------------------------------------------- #
def test_intervention_budget_does_not_end_monitoring() -> None:
    clock = _FakeClock()
    policy = make_runtime_policy(
        budget=_budget(12.0).model_copy(update={"max_interventions": 0}),
        diagnosis_cooldown_s=0.0,
    )
    monitor = _monitor(ScriptedRuntimeSampler(baseline_and_pressured_metrics()), policy, ScriptedWorkload(), clock)
    result = monitor.run()
    assert result.run.interventions == []
    # The monitor keeps observing until its explicit session bound rather than ending
    # when the intervention budget is exhausted.
    assert result.termination.reason is RuntimeTerminationReason.DURATION_REACHED


def test_optimizer_overhead_accounting_is_consistent() -> None:
    clock = _FakeClock()
    policy = make_runtime_policy(budget=_budget(6.0))
    sampler = _CostedSampler(_healthy, clock, lambda n: 0.25)
    monitor = _monitor(sampler, policy, ScriptedWorkload(), clock)
    result = monitor.run()
    overhead = result.run.overhead
    assert overhead.detection_s > 0.0
    assert overhead.optimizer_overhead_s >= overhead.detection_s
    assert overhead.optimizer_overhead_s == overhead.total_s or overhead.total_s >= overhead.optimizer_overhead_s


# --------------------------------------------------------------------------- #
# Workload observations remain first-class                                      #
# --------------------------------------------------------------------------- #
def test_workload_metrics_remain_in_observations() -> None:
    clock = _FakeClock()
    monitor = _monitor(ScriptedRuntimeSampler(_healthy), make_runtime_policy(budget=_budget(6.0)), ScriptedWorkload(), clock)
    result = monitor.run()
    assert result.run.baseline is not None
    assert "decode_tokens_per_sec" in result.run.baseline.metrics
    assert "ttft_s" in result.run.baseline.metrics
    assert "total_latency_s" in result.run.baseline.metrics
