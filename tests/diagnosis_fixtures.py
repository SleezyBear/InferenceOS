"""Synthetic profiling-report fixtures for deterministic Phase 4 diagnosis tests.

These build :class:`BaselineReport` objects directly (no inference), modelling the
plan's causal fixtures (model-dominates / external-process-dominates / no-pressure /
ambiguous) plus stage-specific and negative cases.
"""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.contracts.enums import BackendType, MeasurementMode, WorkloadKind
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunIdentity,
)
from inferenceos.contracts.system_state import (
    GPUEngineUtilization,
    GPUMemoryState,
    ProcessState,
    SystemState,
)
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.profiling.runner import COLD_WARM_SEMANTICS
from inferenceos.selftest import sample_hardware, sample_model

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _opt_int(value: object) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def make_run(
    *,
    model_name: str = "synthetic-model",
    workload_id: str = "short-short",
    mode: MeasurementMode = MeasurementMode.WARM,
    condition: str = "",
    metrics: dict[str, object] | None = None,
    memory_pressure: float | None = None,
    processes: list[ProcessState] | None = None,
    gpu_offload_layers: int = 99,
    gpu_offload_total: int = 99,
    backend: BackendType = BackendType.VULKAN,
    workload_kind: WorkloadKind = WorkloadKind.SHORT,
    gpu_memory_used_bytes: int | None = None,
    gpu_telemetry_available: bool = False,
    gpu_observed_at: datetime | None = None,
    gpu_source: str | None = None,
) -> ProfilingRun:
    values = dict(metrics or {})
    if gpu_memory_used_bytes is not None:
        values.setdefault("gpu_memory_used_bytes", gpu_memory_used_bytes)
    metrics_model = RunMetrics.model_validate(values)
    if gpu_telemetry_available:
        gpu_source = gpu_source or "windows_gpu_engine_counters"
        gpu_observed_at = gpu_observed_at or _NOW
    total_ram = _opt_int(values.get("total_ram_bytes")) or 16 * 1024**3
    available_ram = _opt_int(values.get("available_ram_bytes")) or 8 * 1024**3
    pressure = memory_pressure
    if pressure is None:
        committed = values.get("committed_bytes")
        limit = values.get("commit_limit_bytes")
        if committed is not None and limit:
            pressure = float(committed) / float(limit)  # type: ignore[arg-type]
        else:
            pressure = 0.0
    gpu_memory = (
        [GPUMemoryState(device_index=0, memory_used_bytes=gpu_memory_used_bytes)]
        if gpu_memory_used_bytes is not None
        else []
    )
    state = SystemState(
        captured_at=_NOW,
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        committed_bytes=_opt_int(values.get("committed_bytes")),
        commit_limit_bytes=_opt_int(values.get("commit_limit_bytes")),
        memory_pressure=max(0.0, min(1.0, pressure)),
        cpu_percent_total=float(values.get("system_cpu_percent", 0.0)),  # type: ignore[arg-type]
        hard_page_faults_per_sec=0.0,
        processes=processes or [],
        gpu_memory=gpu_memory,
        gpu_telemetry_available=gpu_telemetry_available,
        gpu_observed_at=gpu_observed_at,
        gpu_source=gpu_source,
        target_process=None,
    )
    hardware = sample_hardware()
    model = sample_model().model_copy(update={"name": model_name})
    workload = WorkloadProfile(
        workload_id=workload_id,
        kind=workload_kind,
        description="synthetic",
        prompt_set_ref=f"{workload_id}@test",
        prompts=["synthetic prompt"],
        seed=1,
        expected_context_length=2048,
    )
    config = ProfilingConfig(
        backend_name="synthetic",
        backend_type=backend,
        gpu_layers_offloaded=gpu_offload_layers,
        gpu_layers_total=gpu_offload_total,
        context_length=2048,
        n_predict=_opt_int(values.get("generated_tokens")) or 32,
        seed=1,
        temperature=0.0,
        extra=({"condition": condition} if condition else {}),
    )
    observation = Observation(
        observation_id=f"obs-{model_name}-{workload_id}-{condition}-{mode.value}",
        captured_at=_NOW,
        mode=mode,
        backend=backend,
        device="synthetic-gpu" if backend is BackendType.VULKAN else None,
        model_ref=model_name,
        workload_ref=workload_id,
        metrics=metrics_model,
        system_state=state,
    )
    return ProfilingRun(
        identity=RunIdentity(
            run_id=f"run-{model_name}-{workload_id}-{condition}-{mode.value}",
            profiler_version="test",
            mode=mode,
            started_at=_NOW,
            finished_at=_NOW,
        ),
        context=RunContext(
            hardware=hardware, model=model, workload=workload, config=config
        ),
        observation=observation,
        metric_quality={},
    )


def make_report(runs: list[ProfilingRun], notes: str | None = None) -> BaselineReport:
    workloads = {run.context.workload.workload_id: run.context.workload for run in runs}
    return BaselineReport(
        profiler_version="test",
        suite_version="test",
        generated_at=_NOW,
        hardware=sample_hardware(),
        backend=runs[0].context.config if runs else ProfilingConfig(),
        workloads=list(workloads.values()),
        runs=runs,
        groups=[],
        cold_warm_semantics=COLD_WARM_SEMANTICS,
        resident_warm_supported=False,
        notes=notes,
    )
