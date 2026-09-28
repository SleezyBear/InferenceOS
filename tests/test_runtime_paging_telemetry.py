"""Runtime memory/paging observability.

Proves the runtime monitor now observes *real* Windows paging and commit telemetry:
a measured value is exposed, an unavailable counter is explicit (never a fabricated
0.0), a genuine measured zero is distinguishable from unavailable, physical and commit
pressure are distinct, reporting status is accurate, and the deterministic paging
diagnosis is reachable from real runtime evidence without false positives.

Everything is reused from the existing Phase 3 counter sampler and
``GlobalMemoryStatusEx`` path; no second memory-monitoring system is introduced.
"""

from __future__ import annotations

import types
from datetime import datetime, timezone
from typing import Any, cast

from inferenceos.contracts.enums import BackendType, BottleneckKind, MetricQuality, PowerState
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.runtime import RuntimeObservation
from inferenceos.contracts.system_state import SystemState
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.native.base import CounterSample, GlobalMemory
from inferenceos.runtime import telemetry as runtime_telemetry
from inferenceos.runtime.telemetry import SystemTelemetrySampler
from inferenceos.runtime.window import ObservationBuffer
from inferenceos.runtime.workload import ActiveWorkload
from inferenceos.telemetry.system import HARD_FAULT_COUNTER, PowerPerformance
from tests.diagnosis_fixtures import make_report, make_run

# --------------------------------------------------------------------------- #
# Deterministic stand-ins                                                      #
# --------------------------------------------------------------------------- #
class _FakeCounterSampler:
    """Stand-in for the Windows counter sampler that can emit the pages-input counter."""

    fail_start = False
    emit_nothing = False
    fault_value = 0.0
    instances: list["_FakeCounterSampler"] = []

    def __init__(self, paths: list[str], interval_s: float = 0.25) -> None:
        self.paths = list(paths)
        self.interval_s = interval_s
        self.stopped = False
        self._samples: list[CounterSample] = []
        _FakeCounterSampler.instances.append(self)

    def _is_paging(self) -> bool:
        return any("pages input" in path.lower() for path in self.paths)

    def start(self) -> None:
        if _FakeCounterSampler.fail_start:
            raise OSError("no powershell available")
        if not _FakeCounterSampler.emit_nothing and self._is_paging():
            self.emit_faults(_FakeCounterSampler.fault_value)

    def wait_until_ready(self, timeout_s: float = 5.0) -> bool:
        return True

    def snapshot(self) -> list[CounterSample]:
        return list(self._samples)

    def stop(self) -> list[CounterSample]:
        self.stopped = True
        return list(self._samples)

    def emit_faults(self, value: float) -> None:
        self._samples.append(CounterSample(path=HARD_FAULT_COUNTER, instance=None, value=value))


class _FakePlatform:
    def __init__(self, memory: GlobalMemory | None = None) -> None:
        self._memory = memory

    def global_memory_status(self) -> GlobalMemory | None:
        return self._memory

    def vulkan_enumerate(self) -> list[Any]:
        return []


class _FakePsutil:
    def __init__(self, total: int, available: int) -> None:
        self._total = total
        self._available = available

    def virtual_memory(self) -> Any:
        return types.SimpleNamespace(total=self._total, available=self._available)

    def cpu_percent(self, interval: float | None = None) -> float:
        return 12.0


class _Workload:
    def snapshot(self) -> Any:
        from inferenceos.contracts.enums import WorkloadState
        from inferenceos.contracts.runtime import WorkloadSnapshot

        return WorkloadSnapshot(state=WorkloadState.RUNNING, running=True, requests_served=1)

    def configuration(self) -> Any:
        from inferenceos.contracts.experiment import RuntimeConfiguration

        return RuntimeConfiguration(context_length=2048, gpu_offload_layers=0)

    def descriptor(self) -> Any:
        from inferenceos.contracts.experiment import RuntimeConfiguration
        from inferenceos.contracts.runtime import WorkloadDescriptor

        return WorkloadDescriptor(
            workload_id="w",
            model_ref="m",
            model_path="m.gguf",
            backend_name="llama.cpp",
            configuration=RuntimeConfiguration(context_length=2048, gpu_offload_layers=0),
        )

    def last_metrics(self) -> dict[str, float]:
        return {}

    def target_pid(self) -> int | None:
        return None


def _as_workload() -> ActiveWorkload:
    return cast(ActiveWorkload, _Workload())


def _reset() -> None:
    _FakeCounterSampler.fail_start = False
    _FakeCounterSampler.emit_nothing = False
    _FakeCounterSampler.fault_value = 0.0
    _FakeCounterSampler.instances = []


def _make_sampler(
    monkeypatch: Any,
    *,
    memory: GlobalMemory | None = None,
    total: int = 16_000,
    available: int = 8_000,
) -> SystemTelemetrySampler:
    sampler = SystemTelemetrySampler(include_paging=True, paging_scan_interval_s=5.0)
    monkeypatch.setattr(runtime_telemetry, "CounterSampler", _FakeCounterSampler)
    monkeypatch.setattr(runtime_telemetry, "get_platform", lambda: _FakePlatform(memory))
    monkeypatch.setattr(runtime_telemetry, "psutil", _FakePsutil(total, available))
    monkeypatch.setattr(
        runtime_telemetry,
        "observe_power_performance",
        lambda: PowerPerformance(power_state=PowerState.AC),
    )
    monkeypatch.setattr(sampler, "_top_processes", lambda: [])
    monkeypatch.setattr(sampler, "_process_state", lambda pid: None)
    return sampler


# --------------------------------------------------------------------------- #
# 1 / 3  Real paging value and genuine zero                                    #
# --------------------------------------------------------------------------- #
def test_real_paging_value_is_exposed(monkeypatch: Any) -> None:
    _reset()
    _FakeCounterSampler.fault_value = 12.5
    sampler = _make_sampler(monkeypatch)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    state = observation.system_state
    assert state is not None
    assert state.paging_telemetry_available is True
    assert state.paging_source == "windows_memory_pages_input_counter"
    assert state.hard_page_faults_per_sec == 12.5
    assert observation.metrics["hard_page_faults_per_sec"] == 12.5
    sampler.close()
    assert _FakeCounterSampler.instances[-1].stopped is True


def test_genuine_zero_paging_is_measured_not_unavailable(monkeypatch: Any) -> None:
    _reset()
    _FakeCounterSampler.fault_value = 0.0
    sampler = _make_sampler(monkeypatch)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    state = observation.system_state
    assert state is not None
    # A genuine zero is available/measured and distinguishable from unavailable.
    assert state.paging_telemetry_available is True
    assert state.hard_page_faults_per_sec == 0.0
    assert observation.metrics["hard_page_faults_per_sec"] == 0.0


# --------------------------------------------------------------------------- #
# 2  Paging counter unavailable is explicit (never a fabricated zero)          #
# --------------------------------------------------------------------------- #
def test_paging_sampler_failure_is_unavailable(monkeypatch: Any) -> None:
    _reset()
    _FakeCounterSampler.fail_start = True
    sampler = _make_sampler(monkeypatch)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    state = observation.system_state
    assert state is not None
    assert state.paging_telemetry_available is False
    assert state.paging_source is not None and state.paging_source.startswith("unavailable")
    assert "hard_page_faults_per_sec" not in observation.metrics
    assert state.hard_page_faults_per_sec == 0.0  # field fallback, NOT a measured value


def test_paging_counter_without_reading_is_unavailable(monkeypatch: Any) -> None:
    _reset()
    _FakeCounterSampler.emit_nothing = True
    sampler = _make_sampler(monkeypatch)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    assert observation.system_state is not None
    assert observation.system_state.paging_telemetry_available is False
    assert "hard_page_faults_per_sec" not in observation.metrics


# --------------------------------------------------------------------------- #
# 4 / 5 / 6  Commit usage, commit pressure, and physical/commit distinction    #
# --------------------------------------------------------------------------- #
def _memory(total_pf: int, avail_pf: int, total_phys: int, avail_phys: int) -> GlobalMemory:
    return GlobalMemory(
        total_physical_bytes=total_phys,
        available_physical_bytes=avail_phys,
        total_pagefile_bytes=total_pf,
        available_pagefile_bytes=avail_pf,
        total_virtual_bytes=0,
        available_virtual_bytes=0,
        memory_load_percent=0,
    )


def test_runtime_exposes_commit_usage_and_limit(monkeypatch: Any) -> None:
    _reset()
    memory = _memory(total_pf=20_000, avail_pf=2_000, total_phys=16_000, avail_phys=8_000)
    sampler = _make_sampler(monkeypatch, memory=memory)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    state = observation.system_state
    assert state is not None
    assert state.committed_bytes == 18_000
    assert state.commit_limit_bytes == 20_000
    assert observation.metrics["committed_bytes"] == 18_000.0
    assert observation.metrics["commit_limit_bytes"] == 20_000.0


def test_commit_pressure_calculation(monkeypatch: Any) -> None:
    _reset()
    memory = _memory(total_pf=20_000, avail_pf=2_000, total_phys=16_000, avail_phys=8_000)
    sampler = _make_sampler(monkeypatch, memory=memory)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    assert observation.system_state is not None
    assert observation.system_state.commit_pressure == 0.9
    assert observation.metrics["commit_pressure"] == 0.9


def test_physical_and_commit_pressure_are_distinct(monkeypatch: Any) -> None:
    _reset()
    memory = _memory(total_pf=20_000, avail_pf=2_000, total_phys=16_000, avail_phys=8_000)
    sampler = _make_sampler(monkeypatch, memory=memory)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    state = observation.system_state
    assert state is not None
    # Physical utilization is 0.5, commit pressure is 0.9; they must not be conflated.
    assert state.physical_memory_pressure == 0.5
    assert state.commit_pressure == 0.9
    # The primary pressure signal preserves Phase 3 semantics (commit when available).
    assert state.memory_pressure == 0.9
    assert observation.metrics["physical_memory_pressure"] == 0.5
    assert observation.metrics["memory_pressure"] == 0.9


def test_commit_unavailable_falls_back_to_physical_pressure(monkeypatch: Any) -> None:
    _reset()
    sampler = _make_sampler(monkeypatch, memory=None, total=16_000, available=4_000)
    observation = sampler.sample(_as_workload(), sequence=0, condition="observed")
    state = observation.system_state
    assert state is not None
    assert state.commit_pressure is None
    assert state.committed_bytes is None
    assert state.physical_memory_pressure == 0.75
    assert state.memory_pressure == 0.75  # physical fallback, not fabricated commit


# --------------------------------------------------------------------------- #
# 7  Reporting status is accurate                                              #
# --------------------------------------------------------------------------- #
def _runtime_observation(*, faults: float | None) -> RuntimeObservation:
    from inferenceos.contracts.enums import WorkloadState

    state = SystemState(
        captured_at=datetime.now(timezone.utc),
        total_ram_bytes=16_000,
        available_ram_bytes=8_000,
        memory_pressure=0.4,
        cpu_percent_total=10.0,
        hard_page_faults_per_sec=faults if faults is not None else 0.0,
        paging_telemetry_available=faults is not None,
    )
    metrics = {"available_ram_bytes": 8_000.0}
    if faults is not None:
        metrics["hard_page_faults_per_sec"] = faults
    return RuntimeObservation(
        observation_id="obs",
        captured_at=datetime.now(timezone.utc),
        workload_state=WorkloadState.RUNNING,
        metrics=metrics,
        system_state=state,
    )


def _monitor() -> Any:
    from inferenceos.runtime.approval import AutoRejectApprovalProvider
    from inferenceos.runtime.monitor import RuntimeMonitor
    from inferenceos.runtime.notification import CollectingNotificationSink
    from inferenceos.selftest import sample_hardware
    from tests.runtime_helpers import make_runtime_policy

    workload = _Workload()
    return RuntimeMonitor(
        workload=cast(ActiveWorkload, workload),
        sampler=cast(Any, object()),  # never used by _to_run
        policy=make_runtime_policy(),
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
    )


def test_report_marks_paging_measured_when_available() -> None:
    monitor = _monitor()
    run = monitor._to_run(
        _runtime_observation(faults=3.5),
        condition="observed",
        model=_model(),
        workload=_workload_profile(),
        hardware=_hardware(),
    )
    assert run.observation.metrics.hard_page_faults_per_sec == 3.5
    assert run.metric_quality["hard_page_faults_per_sec"] is MetricQuality.MEASURED


def test_report_marks_paging_unavailable_when_absent() -> None:
    monitor = _monitor()
    run = monitor._to_run(
        _runtime_observation(faults=None),
        condition="observed",
        model=_model(),
        workload=_workload_profile(),
        hardware=_hardware(),
    )
    assert run.observation.metrics.hard_page_faults_per_sec is None
    assert run.metric_quality["hard_page_faults_per_sec"] is MetricQuality.UNAVAILABLE


def _model() -> Any:
    from inferenceos.selftest import sample_model

    return sample_model()


def _workload_profile() -> Any:
    from inferenceos.contracts.enums import WorkloadKind
    from inferenceos.contracts.workload import WorkloadProfile

    return WorkloadProfile(
        workload_id="w",
        kind=WorkloadKind.SHORT,
        description="t",
        prompt_set_ref="w@test",
        prompts=["p"],
        seed=1,
        expected_context_length=2048,
    )


def _hardware() -> Any:
    from inferenceos.selftest import sample_hardware

    return sample_hardware()


# --------------------------------------------------------------------------- #
# 8 / 9  Diagnosis reachability and false-positive avoidance                    #
# --------------------------------------------------------------------------- #
def _healthy(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "decode_tokens_per_sec": 24.0,
        "prompt_tokens_per_sec": 90.0,
        "total_latency_s": 1.4,
        "prefill_latency_s": 0.45,
        "decode_latency_s": 0.9,
        "model_load_time_s": 2.0,
        "process_cpu_percent": 120.0,
        "system_cpu_percent": 25.0,
        "gpu_utilization_percent": 70.0,
        "peak_rss_bytes": 1_300_000_000,
        "available_ram_bytes": 8 * 1024**3,
        "total_ram_bytes": 16 * 1024**3,
        "hard_page_faults_per_sec": 0.0,
        "prompt_tokens": 40,
        "generated_tokens": 24,
    }
    base.update(overrides)
    return base


def _detect(runs: list[Any]) -> Any:
    return BottleneckDetector().detect(make_report(runs), source_report_path="<test>")


def test_runtime_paging_diagnosis_becomes_reachable() -> None:
    runs = [
        make_run(condition="control", metrics=_healthy(hard_page_faults_per_sec=0.0), memory_pressure=0.4)
        for _ in range(2)
    ]
    runs += [
        make_run(
            condition="observed",
            metrics=_healthy(
                hard_page_faults_per_sec=18000.0,
                decode_tokens_per_sec=8.0,
                total_latency_s=2.6,
            ),
            memory_pressure=0.9,
        )
        for _ in range(2)
    ]
    diag = _detect(runs)
    assert diag.primary is not None and diag.primary.kind is BottleneckKind.PAGING
    assert any("hard_page_faults" in item.metric for item in diag.primary.supporting_evidence)


def test_no_paging_diagnosis_when_paging_is_genuinely_zero() -> None:
    runs = [
        make_run(condition="control", metrics=_healthy(hard_page_faults_per_sec=0.0), memory_pressure=0.4)
        for _ in range(2)
    ]
    runs += [
        make_run(
            condition="observed",
            metrics=_healthy(hard_page_faults_per_sec=0.0),
            memory_pressure=0.42,
        )
        for _ in range(2)
    ]
    diag = _detect(runs)
    assert diag.primary is None or diag.primary.kind is not BottleneckKind.PAGING


# --------------------------------------------------------------------------- #
# 8b  Temporal qualification: paging from a genuine-zero baseline              #
# --------------------------------------------------------------------------- #
def _zero_baseline_buffer() -> ObservationBuffer:
    from inferenceos.contracts.experiment import RuntimeConfiguration

    buffer = ObservationBuffer(capacity=20, baseline_count=3)
    for index in range(3):
        buffer.append(
            RuntimeObservation(
                observation_id=f"b{index}",
                sequence=index,
                captured_at=datetime.now(timezone.utc),
                metrics={"hard_page_faults_per_sec": 0.0},
                condition="baseline",
                config_key="c1",
                workload_ref="short-short",
                model_ref="m",
            )
        )
    buffer.establish_baseline(
        workload_id="short-short",
        model_ref="m",
        backend_name="llama.cpp",
        configuration=RuntimeConfiguration(context_length=2048),
        context_length=2048,
    )
    return buffer


def test_paging_rise_from_zero_baseline_is_temporally_detected() -> None:
    buffer = _zero_baseline_buffer()
    for index in range(3):
        buffer.append(
            RuntimeObservation(
                observation_id=f"o{index}",
                sequence=10 + index,
                captured_at=datetime.now(timezone.utc),
                metrics={"hard_page_faults_per_sec": 500.0},
                condition="observed",
                config_key="c1",
                workload_ref="short-short",
                model_ref="m",
            )
        )
    window = buffer.qualify(threshold=0.25, hysteresis=0.125, persistence=3)
    assert window.anomaly.value in ("persistent", "repeated")
    fault_deviation = next(d for d in window.deviations if d.metric == "hard_page_faults_per_sec")
    assert fault_deviation.exceeds_threshold is True


# --------------------------------------------------------------------------- #
# 10  Existing diagnosis behaviour unchanged                                   #
# --------------------------------------------------------------------------- #
def test_cpu_diagnosis_behaviour_unchanged() -> None:
    runs = [
        make_run(condition="gpu", metrics=_healthy(), backend=BackendType.VULKAN, gpu_offload_layers=99)
        for _ in range(2)
    ]
    runs += [
        make_run(
            condition="cpu",
            metrics=_healthy(gpu_utilization_percent=5.0, process_cpu_percent=520.0, decode_tokens_per_sec=6.0),
            backend=BackendType.CPU,
            gpu_offload_layers=0,
        )
        for _ in range(2)
    ]
    diag = _detect(runs)
    assert diag.primary is not None and diag.primary.kind is BottleneckKind.CPU


def test_gpu_underutilization_behaviour_unchanged() -> None:
    runs = [
        make_run(condition="control", metrics=_healthy(), backend=BackendType.VULKAN, gpu_offload_layers=8)
        for _ in range(2)
    ]
    runs += [
        make_run(
            condition="contended",
            metrics=_healthy(
                gpu_utilization_percent=5.0,
                process_cpu_percent=520.0,
                decode_tokens_per_sec=6.0,
            ),
            backend=BackendType.VULKAN,
            gpu_offload_layers=8,
        )
        for _ in range(2)
    ]
    diag = _detect(runs)
    assert diag.primary is not None and diag.primary.kind is BottleneckKind.GPU_UNDERUTILIZATION


def test_system_state_fields_default_safely() -> None:
    state = SystemState(
        captured_at=datetime.now(timezone.utc),
        total_ram_bytes=1,
        available_ram_bytes=1,
        memory_pressure=0.0,
        cpu_percent_total=0.0,
        hard_page_faults_per_sec=0.0,
    )
    assert state.physical_memory_pressure is None
    assert state.commit_pressure is None
    assert state.paging_telemetry_available is False
    assert state.paging_source is None
