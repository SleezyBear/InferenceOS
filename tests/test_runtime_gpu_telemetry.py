"""Focused tests for mode-controlled runtime GPU telemetry (reuses the Windows sampler).

The runtime sampler must expose GPU state (via the existing GPU Engine counter sampler),
respect monitoring-mode cadence, represent unavailable/stale telemetry explicitly, and
never terminate sampling because GPU collection is expensive or fails.
"""

from __future__ import annotations

from typing import Any, cast

from inferenceos.contracts.enums import MonitoringMode
from inferenceos.contracts.runtime import RuntimeObservation
from inferenceos.contracts.system_state import SystemState
from inferenceos.native.base import CounterSample
from inferenceos.runtime import telemetry as runtime_telemetry
from inferenceos.runtime.telemetry import SystemTelemetrySampler
from inferenceos.runtime.verification import ScriptedRuntimeSampler, ScriptedWorkload
from inferenceos.runtime.workload import ActiveWorkload
from inferenceos.telemetry.system import GPU_ADAPTER_MEMORY_COUNTER, GPU_ENGINE_COUNTER


class _FakePlatform:
    def vulkan_enumerate(self) -> list[Any]:
        return []


class _FakeGpuCounterSampler:
    """Deterministic stand-in for the Windows GPU Engine counter sampler."""

    fail_start = False
    instances: list["_FakeGpuCounterSampler"] = []

    def __init__(self, paths: list[str], interval_s: float = 0.25) -> None:
        self.paths = list(paths)
        self.interval_s = interval_s
        self.started = False
        self.stopped = False
        self._samples: list[CounterSample] = []
        _FakeGpuCounterSampler.instances.append(self)

    def start(self) -> None:
        if _FakeGpuCounterSampler.fail_start:
            raise OSError("no powershell available")
        self.started = True
        self.emit(42.0)

    def wait_until_ready(self, timeout_s: float = 5.0) -> bool:
        return True

    def snapshot(self) -> list[CounterSample]:
        return list(self._samples)

    def stop(self) -> list[CounterSample]:
        self.stopped = True
        return list(self._samples)

    def emit(self, utilization: float, memory: int = 123456) -> None:
        self._samples.append(
            CounterSample(
                path=GPU_ENGINE_COUNTER,
                instance="luid_0x00000000_0x00000001_phys_0_eng_0_engtype_3d",
                value=utilization,
            )
        )
        self._samples.append(
            CounterSample(
                path=GPU_ADAPTER_MEMORY_COUNTER,
                instance="luid_0x00000000_0x00000001_phys_0",
                value=float(memory),
            )
        )


class _Workload:
    def snapshot(self) -> Any:
        from inferenceos.contracts.enums import WorkloadState
        from inferenceos.contracts.runtime import WorkloadSnapshot

        return WorkloadSnapshot(state=WorkloadState.RUNNING, running=True, requests_served=3)

    def configuration(self) -> Any:
        from inferenceos.contracts.experiment import RuntimeConfiguration

        return RuntimeConfiguration(context_length=2048, gpu_offload_layers=0)

    def descriptor(self) -> Any:
        from inferenceos.contracts.runtime import WorkloadDescriptor

        return WorkloadDescriptor(
            workload_id="w", model_ref="m", model_path="m.gguf", backend_name="llama.cpp"
        )

    def last_metrics(self) -> dict[str, float]:
        return {}

    def target_pid(self) -> int | None:
        return None


def _as_workload() -> ActiveWorkload:
    return cast(ActiveWorkload, _Workload())


def _patch(monkeypatch: Any, sampler: SystemTelemetrySampler) -> None:
    monkeypatch.setattr(runtime_telemetry, "CounterSampler", _FakeGpuCounterSampler)
    monkeypatch.setattr(runtime_telemetry, "get_platform", lambda: _FakePlatform())
    monkeypatch.setattr(sampler, "_top_processes", lambda: [])
    monkeypatch.setattr(sampler, "_process_state", lambda pid: None)


def _reset_fakes() -> None:
    _FakeGpuCounterSampler.fail_start = False
    _FakeGpuCounterSampler.instances = []


def test_gpu_sampler_feeds_runtime_telemetry(monkeypatch: Any) -> None:
    _reset_fakes()
    sampler = SystemTelemetrySampler(
        include_gpu=True, gpu_scan_interval_s=5.0, process_scan_interval_s=1000.0
    )
    _patch(monkeypatch, sampler)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    state = observation.system_state
    assert state is not None
    assert state.gpu_telemetry_available is True
    assert state.gpu_source == "windows_gpu_engine_counters"
    assert state.gpu_observed_at is not None
    assert state.gpu_engines and state.gpu_engines[0].utilization_percent == 42.0
    assert state.gpu_memory and state.gpu_memory[0].memory_used_bytes == 123456
    # Diagnosis-facing metric is present only when GPU telemetry is available.
    assert observation.metrics["gpu_utilization_percent"] == 42.0
    assert observation.metrics["gpu_memory_used_bytes"] == 123456.0
    assert observation.requests_served == 3
    sampler.close()
    assert _FakeGpuCounterSampler.instances[-1].stopped is True


def test_gpu_unavailable_is_explicit_not_zero(monkeypatch: Any) -> None:
    _reset_fakes()
    _FakeGpuCounterSampler.fail_start = True
    sampler = SystemTelemetrySampler(
        include_gpu=True, gpu_scan_interval_s=5.0, process_scan_interval_s=1000.0
    )
    _patch(monkeypatch, sampler)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    state = observation.system_state
    assert state is not None
    assert state.gpu_telemetry_available is False
    assert state.gpu_engines == [] and state.gpu_memory == []
    assert state.gpu_observed_at is None
    assert state.gpu_source is not None and state.gpu_source.startswith("unavailable")
    assert "gpu_utilization_percent" not in observation.metrics


def test_gpu_disabled_returns_unavailable(monkeypatch: Any) -> None:
    _reset_fakes()
    sampler = SystemTelemetrySampler(include_gpu=False, process_scan_interval_s=1000.0)
    _patch(monkeypatch, sampler)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    assert observation.system_state is not None
    assert observation.system_state.gpu_telemetry_available is False
    assert "gpu_utilization_percent" not in observation.metrics


def test_gpu_stale_value_not_repolled_between_cadence(monkeypatch: Any) -> None:
    _reset_fakes()
    sampler = SystemTelemetrySampler(
        include_gpu=True, gpu_scan_interval_s=5.0, process_scan_interval_s=1000.0
    )
    _patch(monkeypatch, sampler)
    first = sampler._gpu_state(now=0.0)
    assert first[2] is True
    observed_at_first = first[3]
    # Within the cadence: no re-poll; timestamp/value unchanged (explicitly stale).
    cached = sampler._gpu_state(now=1.0)
    assert cached[3] == observed_at_first
    assert cached[0][0].utilization_percent == 42.0
    # After the cadence a fresh reading replaces the cached one.
    _FakeGpuCounterSampler.instances[-1].emit(77.0)
    refreshed = sampler._gpu_state(now=6.0)
    assert refreshed[0][0].utilization_percent == 77.0
    assert sampler._gpu_last_poll == 6.0
    sampler.close()


def test_mode_gpu_cadence_values() -> None:
    sampler = SystemTelemetrySampler(
        include_gpu=True, gpu_scan_interval_s=5.0, process_scan_interval_s=1000.0
    )
    monitor = _monitor_with(sampler)
    assert monitor._mode_gpu_scan_interval_for(MonitoringMode.ACTIVE) == 5.0
    assert monitor._mode_gpu_scan_interval_for(MonitoringMode.NORMAL) == 5.0
    assert monitor._mode_gpu_scan_interval_for(MonitoringMode.LOW_COST) == 20.0
    assert monitor._mode_gpu_scan_interval_for(MonitoringMode.DORMANT) == 60.0


def test_mode_transition_records_gpu_cadence() -> None:
    # Low-cost transition event should carry the reduced GPU cadence.
    from inferenceos.contracts.runtime import RuntimeBudget
    from inferenceos.runtime.approval import AutoRejectApprovalProvider
    from inferenceos.runtime.monitor import RuntimeMonitor
    from inferenceos.runtime.notification import CollectingNotificationSink
    from inferenceos.selftest import sample_hardware
    from tests.runtime_helpers import make_runtime_policy

    class _Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            return self.t

        def sleep(self, seconds: float) -> None:
            self.t += max(seconds, 0.0)

    class _CostSampler(ScriptedRuntimeSampler):
        def __init__(self, clock: _Clock) -> None:
            super().__init__(_healthy)
            self._clock = clock

        def sample(self, workload: Any, *, sequence: int, condition: str) -> Any:
            observation = super().sample(workload, sequence=sequence, condition=condition)
            if self.count <= 6:
                self._clock.sleep(30.0)
            return observation

    clock = _Clock()
    workload = ScriptedWorkload()
    policy = make_runtime_policy(
        budget=RuntimeBudget(
            max_duration_s=400.0,
            telemetry_interval_s=0.5,
            observation_window_size=40,
            baseline_samples=4,
            max_interventions=2,
            max_optimizer_overhead_s=120.0,
        ),
        monitor_mode_overhead_fraction_high=0.10,
        monitor_mode_gpu_scan_interval_s=5.0,
    )
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=_CostSampler(clock),
        policy=policy,
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    low_cost = [
        e for e in result.run.monitoring_modes if e.mode is MonitoringMode.LOW_COST
    ]
    assert low_cost, "expected a LOW_COST transition"
    assert low_cost[0].data.get("gpu_scan_interval_s") == 20.0
    assert result.termination.reason.value != "monitoring_budget_exhausted"


def test_gpu_failure_does_not_terminate_monitor(monkeypatch: Any) -> None:
    _reset_fakes()
    _FakeGpuCounterSampler.fail_start = True
    from inferenceos.contracts.runtime import RuntimeBudget
    from inferenceos.runtime.approval import AutoRejectApprovalProvider
    from inferenceos.runtime.monitor import RuntimeMonitor
    from inferenceos.runtime.notification import CollectingNotificationSink
    from inferenceos.runtime.verification import ScriptedWorkload
    from inferenceos.selftest import sample_hardware
    from tests.runtime_helpers import make_runtime_policy

    class _Clock:
        def __init__(self) -> None:
            self.t = 0.0

        def __call__(self) -> float:
            return self.t

        def sleep(self, seconds: float) -> None:
            self.t += max(seconds, 0.0)

    sampler = SystemTelemetrySampler(
        include_gpu=True, gpu_scan_interval_s=5.0, process_scan_interval_s=1000.0
    )
    monkeypatch.setattr(runtime_telemetry, "CounterSampler", _FakeGpuCounterSampler)
    monkeypatch.setattr(runtime_telemetry, "get_platform", lambda: _FakePlatform())
    monkeypatch.setattr(sampler, "_top_processes", lambda: [])
    monkeypatch.setattr(sampler, "_process_state", lambda pid: None)
    clock = _Clock()
    monitor = RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=sampler,
        policy=make_runtime_policy(
            budget=RuntimeBudget(
                max_duration_s=4.0,
                telemetry_interval_s=0.5,
                observation_window_size=40,
                baseline_samples=4,
                max_optimizer_overhead_s=120.0,
            )
        ),
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    assert result.termination.reason.value == "duration_reached"


def test_runtime_report_carries_gpu_metric() -> None:
    from datetime import datetime, timezone

    from inferenceos.contracts.enums import WorkloadState
    from inferenceos.runtime.approval import AutoRejectApprovalProvider
    from inferenceos.runtime.monitor import RuntimeMonitor
    from inferenceos.runtime.notification import CollectingNotificationSink
    from inferenceos.runtime.verification import ScriptedWorkload
    from inferenceos.selftest import sample_hardware
    from tests.runtime_helpers import make_runtime_policy

    workload = ScriptedWorkload()
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=ScriptedRuntimeSampler(_healthy),
        policy=make_runtime_policy(),
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
    )
    observation = RuntimeObservation(
        observation_id="obs-gpu",
        captured_at=datetime.now(timezone.utc),
        workload_state=WorkloadState.RUNNING,
        metrics={"gpu_utilization_percent": 42.0, "available_ram_bytes": 8_000.0},
        system_state=SystemState(
            captured_at=datetime.now(timezone.utc),
            total_ram_bytes=16_000,
            available_ram_bytes=8_000,
            memory_pressure=0.5,
            cpu_percent_total=10.0,
            hard_page_faults_per_sec=0.0,
        ),
    )
    run = monitor._to_run(
        observation,
        condition="baseline",
        model=workload.model_profile(),
        workload=workload.workload_profile(),
        hardware=sample_hardware(),
    )
    assert run.observation.metrics.gpu_utilization_percent == 42.0


def _healthy(sequence: int, condition: str) -> dict[str, float]:
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


def _monitor_with(sampler: Any) -> Any:
    from inferenceos.runtime.approval import AutoRejectApprovalProvider
    from inferenceos.runtime.monitor import RuntimeMonitor
    from inferenceos.runtime.notification import CollectingNotificationSink
    from inferenceos.runtime.verification import ScriptedWorkload
    from inferenceos.selftest import sample_hardware
    from tests.runtime_helpers import make_runtime_policy

    return RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=sampler,
        policy=make_runtime_policy(monitor_mode_gpu_scan_interval_s=5.0),
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
    )
