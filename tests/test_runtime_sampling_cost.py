"""Focused tests for runtime sampling cost and its instrumentation.

The full-process-table scan (``psutil`` ``memory_info`` for every process) dominated the
runtime sampling path (~0.24 s for ~200 processes). These tests prove the scan is cached
and reused between samples, the target process is still refreshed every sample, and the
per-stage sampling cost is instrumented for reporting.
"""

from __future__ import annotations

from typing import Any, cast

from inferenceos.contracts.enums import PowerState, WorkloadState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.runtime import WorkloadDescriptor, WorkloadSnapshot
from inferenceos.contracts.system_state import ProcessState
from inferenceos.runtime import telemetry as runtime_telemetry
from inferenceos.runtime.telemetry import SystemTelemetrySampler
from inferenceos.runtime.workload import ActiveWorkload
from inferenceos.telemetry.system import PowerPerformance


class _FakeWorkload:
    def __init__(self, *, target_pid: int | None = None) -> None:
        self._target_pid = target_pid

    def snapshot(self) -> WorkloadSnapshot:
        return WorkloadSnapshot(state=WorkloadState.RUNNING, running=True)

    def configuration(self) -> RuntimeConfiguration:
        return RuntimeConfiguration(context_length=2048, gpu_offload_layers=0)

    def descriptor(self) -> WorkloadDescriptor:
        return WorkloadDescriptor(
            workload_id="probe",
            model_ref="model",
            model_path="model.gguf",
            backend_name="llama.cpp (resident)",
        )

    def last_metrics(self) -> dict[str, float]:
        return {}

    def target_pid(self) -> int | None:
        return self._target_pid


def _make_workload(*, target_pid: int | None = None) -> ActiveWorkload:
    return cast(ActiveWorkload, _FakeWorkload(target_pid=target_pid))


def _process(pid: int, rss: int = 10) -> ProcessState:
    return ProcessState(pid=pid, name=f"p{pid}", cpu_percent=0.0, rss_bytes=rss, vms_bytes=1)


def _install_fakes(monkeypatch: Any, sampler: SystemTelemetrySampler) -> dict[str, int]:
    calls = {"top": 0, "target": 0}

    def fake_top() -> list[ProcessState]:
        calls["top"] += 1
        return [_process(1), _process(2)]

    def fake_target(pid: int) -> ProcessState:
        calls["target"] += 1
        return _process(pid, rss=99)

    monkeypatch.setattr(sampler, "_top_processes", fake_top)
    monkeypatch.setattr(sampler, "_process_state", fake_target)
    monkeypatch.setattr(
        runtime_telemetry,
        "observe_power_performance",
        lambda: PowerPerformance(power_state=PowerState.AC),
    )
    return calls


def test_process_scan_is_cached_between_samples(monkeypatch: Any) -> None:
    sampler = SystemTelemetrySampler(process_scan_interval_s=1000.0)
    calls = _install_fakes(monkeypatch, sampler)
    workload = _make_workload(target_pid=42)
    for sequence in range(6):
        sampler.sample(workload, sequence=sequence, condition="observed")
    # One full process-table scan for the whole window; target refreshed every sample.
    assert calls["top"] == 1
    assert calls["target"] == 6


def test_process_scan_refreshes_after_interval(monkeypatch: Any) -> None:
    sampler = SystemTelemetrySampler(process_scan_interval_s=0.0)
    calls = _install_fakes(monkeypatch, sampler)
    workload = _make_workload(target_pid=7)
    for sequence in range(4):
        sampler.sample(workload, sequence=sequence, condition="observed")
    assert calls["top"] == 4
    assert calls["target"] == 4


class _FakeProc:
    def __init__(self, pid: int, cpu_holder: dict[str, float]) -> None:
        self._pid = pid
        self._cpu = cpu_holder

    def oneshot(self) -> Any:
        import contextlib

        return contextlib.nullcontext()

    def memory_info(self) -> Any:
        import types

        return types.SimpleNamespace(rss=1024, vms=2048)

    def name(self) -> str:
        return "llama-server.exe"

    def cpu_times(self) -> Any:
        import types

        return types.SimpleNamespace(user=self._cpu["value"], system=0.0)


def test_target_process_cpu_is_measured_from_the_target(monkeypatch: Any) -> None:
    import time as _time

    holder = {"value": 0.0}
    monkeypatch.setattr(
        runtime_telemetry.psutil, "Process", lambda pid: _FakeProc(pid, holder)
    )
    sampler = SystemTelemetrySampler(process_scan_interval_s=1000.0)
    first = sampler._process_state(4242)
    assert first is not None
    # First observation cannot know the rate; it is not fabricated as nonzero.
    assert sampler._target_cpu_percent is None
    assert first.cpu_percent == 0.0
    holder["value"] = 0.06
    _time.sleep(0.06)
    second = sampler._process_state(4242)
    assert second is not None
    assert sampler._target_cpu_percent is not None
    assert sampler._target_cpu_percent > 0.0
    assert second.cpu_percent == sampler._target_cpu_percent


def test_process_cpu_metric_absent_until_measured(monkeypatch: Any) -> None:
    sampler = SystemTelemetrySampler(process_scan_interval_s=1000.0)
    _install_fakes(monkeypatch, sampler)
    workload = _make_workload(target_pid=42)
    # ``_install_fakes`` replaces ``_process_state`` with a stub that does not set the
    # CPU rate, so the metric must be absent rather than a fabricated zero.
    observation = sampler.sample(workload, sequence=0, condition="observed")
    assert "process_cpu_percent" not in observation.metrics


def test_process_cpu_metric_present_when_measured(monkeypatch: Any) -> None:
    sampler = SystemTelemetrySampler(process_scan_interval_s=1000.0)
    monkeypatch.setattr(sampler, "_top_processes", lambda: [])
    monkeypatch.setattr(runtime_telemetry, "observe_power_performance", lambda: PowerPerformance())
    sampler._target_cpu_percent = 512.0
    observation = sampler.sample(_make_workload(target_pid=42), sequence=0, condition="observed")
    assert observation.metrics["process_cpu_percent"] == 512.0


def test_target_process_is_merged_fresh_into_process_list(monkeypatch: Any) -> None:
    sampler = SystemTelemetrySampler(process_scan_interval_s=1000.0)
    _install_fakes(monkeypatch, sampler)
    workload = _make_workload(target_pid=42)
    observation = sampler.sample(workload, sequence=0, condition="observed")
    state = observation.system_state
    assert state is not None and state.target_process is not None
    assert state.target_process.pid == 42
    assert state.processes[0].pid == 42


def test_sampling_breakdown_instrumented(monkeypatch: Any) -> None:
    sampler = SystemTelemetrySampler(process_scan_interval_s=1000.0)
    _install_fakes(monkeypatch, sampler)
    sampler.sample(_make_workload(target_pid=1), sequence=0, condition="observed")
    assert {"memory", "cpu", "processes", "power"} <= set(sampler.last_breakdown)
    assert all(value >= 0.0 for value in sampler.last_breakdown.values())


def test_monitor_reports_sampling_breakdown() -> None:
    from inferenceos.runtime.approval import AutoRejectApprovalProvider
    from inferenceos.runtime.monitor import RuntimeMonitor
    from inferenceos.runtime.notification import CollectingNotificationSink
    from inferenceos.selftest import sample_hardware
    from inferenceos.runtime.verification import ScriptedRuntimeSampler, ScriptedWorkload
    from tests.runtime_helpers import make_runtime_policy

    clock = _FakeClock()
    sampler = ScriptedRuntimeSampler(_healthy_metrics)
    sampler.last_breakdown = {"memory": 0.002, "cpu": 0.0001, "processes": 0.01, "power": 0.0002}  # type: ignore[attr-defined]
    monitor = RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=sampler,
        policy=make_runtime_policy(
            budget=_budget(4.0), baseline_workload_samples=0
        ),
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    breakdown = result.run.resource_impact.sampling_breakdown
    assert breakdown.get("processes", 0.0) > 0.0
    assert breakdown.get("memory", 0.0) > 0.0


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


def _budget(duration: float) -> Any:
    from inferenceos.contracts.runtime import RuntimeBudget

    return RuntimeBudget(
        max_duration_s=duration,
        telemetry_interval_s=0.5,
        observation_window_size=40,
        baseline_samples=4,
        max_optimizer_overhead_s=120.0,
    )


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)
