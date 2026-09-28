"""Baseline profiler: real repeated measurements with full run context.

Measurement only. For each (model, workload, mode) it runs the real backend through
the :class:`BackendAdapter`, samples real system telemetry over the inference
interval, and persists a self-contained :class:`ProfilingRun`. It does not detect
bottlenecks or choose configurations.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from inferenceos import __version__
from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.backends.llama_cpp_toolchain import backend_type_from_name
from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    BackendType,
    MeasurementMode,
    MetricQuality,
    ObservationSource,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunGroupStats,
    RunIdentity,
)
from inferenceos.contracts.system_state import (
    GPUEngineUtilization,
    GPUMemoryState,
    SystemState,
)
from inferenceos.hardware.profiler import detect_hardware_profile
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendCapabilities,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.native import get_platform
from inferenceos.native.base import CounterSample, GlobalMemory
from inferenceos.profiling.statistics import summarize
from inferenceos.profiling.workloads import (
    BENCHMARK_SUITE_VERSION,
    BenchmarkWorkload,
    choose_context_length,
    workload_profiles,
)
from inferenceos.telemetry.counters import (
    CounterSampler,
    gpu_adapter_memory_peaks,
    gpu_engine_peaks,
    max_value,
)
from inferenceos.telemetry.system import (
    GPU_ADAPTER_MEMORY_COUNTER,
    GPU_ENGINE_COUNTER,
    HARD_FAULT_COUNTER,
    compute_memory_pressure,
    observe_power_performance,
)

try:
    import psutil
except ImportError:  # pragma: no cover - hard dependency
    psutil = None  # type: ignore[assignment]

PROFILER_VERSION = f"{__version__}+phase3"

COUNTER_PATHS: tuple[str, ...] = (
    HARD_FAULT_COUNTER,
    GPU_ENGINE_COUNTER,
    GPU_ADAPTER_MEMORY_COUNTER,
)

COLD_WARM_SEMANTICS = (
    "Under the current process-per-generation llama.cpp adapter, every measured run "
    "is a fresh llama-cli process that loads weights into memory. COLD labels a run "
    "with no preceding warm-up for that model/workload in this session (the model "
    "file is typically not yet in the OS page cache). WARM labels measured runs "
    "after an explicit warm-up run, so the model file is resident in the OS page "
    "cache; WARM does NOT mean a resident model process or reused KV/weights. "
    "Resident-process warm inference is not supported by this backend adapter."
)

LIMITATIONS: tuple[str, ...] = (
    "COLD runs are performed once per model/workload: the OS page cache cannot be "
    "evicted without elevation, so repetition of COLD runs would not be genuinely "
    "cold.",
    "GPU engine/adapter-memory and hard-page-fault telemetry are sampled over the "
    "inference interval; GPU total VRAM is not exposed by Windows counters and is "
    "reported as unavailable.",
    "system_cpu_percent is measured over the whole interval and therefore includes "
    "the profiler's own counter-sampling process overhead.",
    "TTFT is derived as prefill + one decode step (llama.cpp exposes no first-token "
    "timestamp); it is labelled DERIVED.",
)

_AGGREGATED_METRICS: tuple[str, ...] = (
    "model_load_time_s",
    "ttft_s",
    "prefill_latency_s",
    "decode_latency_s",
    "prompt_tokens_per_sec",
    "decode_tokens_per_sec",
    "total_latency_s",
    "process_wall_s",
    "prompt_tokens",
    "generated_tokens",
    "total_tokens",
    "peak_rss_bytes",
    "process_cpu_percent",
    "system_cpu_percent",
    "gpu_utilization_percent",
    "gpu_memory_used_bytes",
    "hard_page_faults_per_sec",
)


@dataclass
class ProfilerOptions:
    gpu_offload_layers: int = 99
    telemetry_sampling: bool = True
    log_verbosity: int = 4
    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    use_mmap: bool | None = None
    flash_attention: bool | None = None
    kv_cache_type: str | None = None
    device: str | None = None


@dataclass(frozen=True)
class ProfileTask:
    model: ModelProfile
    workload: BenchmarkWorkload
    context_length: int


class BaselineProfiler:
    """Runs real repeated measurements and assembles a baseline report."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        adapter: BackendAdapter | None = None,
        hardware: HardwareProfile | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._adapter: BackendAdapter = adapter or LlamaCppAdapter(settings=self._settings)
        self._hardware = hardware
        self._capabilities: BackendCapabilities | None = None

    @property
    def hardware(self) -> HardwareProfile:
        if self._hardware is None:
            self._hardware = detect_hardware_profile(self._settings)
        return self._hardware

    def capabilities(self) -> BackendCapabilities:
        if self._capabilities is None:
            self._capabilities = self._adapter.capabilities()
        return self._capabilities

    def make_tasks(
        self,
        models: Sequence[ModelProfile],
        workload_ids: Sequence[str] | None,
        context_override: int | None = None,
    ) -> list[ProfileTask]:
        from inferenceos.profiling.workloads import select_workloads

        tasks: list[ProfileTask] = []
        for model in models:
            workloads = select_workloads(workload_ids, model)
            context = choose_context_length(model, context_override)
            for workload in workloads:
                if workload.min_context_tokens <= context:
                    tasks.append(ProfileTask(model=model, workload=workload, context_length=context))
        return tasks

    def run_once(
        self,
        task: ProfileTask,
        *,
        mode: MeasurementMode,
        sequence: int,
        repetition: int,
        warmup: bool,
        options: ProfilerOptions,
    ) -> ProfilingRun:
        if psutil is None:  # pragma: no cover - hard dependency
            raise RuntimeError("psutil is required for profiling")
        adapter = self._adapter
        model = task.model
        workload = task.workload
        requested = _requested_config(task, options)
        adapter.load(
            model,
            RuntimeParams(
                gpu_offload_layers=options.gpu_offload_layers,
                context_length=task.context_length,
                threads=options.threads or 0,
                batch_size=options.batch_size,
                ubatch_size=options.ubatch_size,
                use_mmap=options.use_mmap,
                flash_attention=options.flash_attention,
                kv_cache_type=options.kv_cache_type,
                device=options.device,
                log_verbosity=options.log_verbosity,
                ignore_eos=task.workload.ignore_eos,
            ),
        )

        sampler: CounterSampler | None = None
        if options.telemetry_sampling:
            sampler = CounterSampler(COUNTER_PATHS, interval_s=0.25)
            sampler.start()
            sampler.wait_for_first_sample(timeout_s=6.0)

        psutil.cpu_percent(interval=None)
        memory_before = get_platform().global_memory_status()
        started = datetime.now(timezone.utc)
        started_perf = time.perf_counter()
        try:
            result = adapter.generate(
                GenerationRequest(
                    prompt=workload.prompt,
                    n_predict=workload.n_predict,
                    seed=workload.seed,
                    temperature=workload.temperature,
                )
            )
        finally:
            wall_s = time.perf_counter() - started_perf
            samples = sampler.stop() if sampler is not None else []
        finished = datetime.now(timezone.utc)
        telemetry = adapter.telemetry()
        device_info = adapter.last_device
        system_cpu = float(psutil.cpu_percent(interval=None))
        memory_after = get_platform().global_memory_status()
        adapter.unload()

        metrics = _build_metrics(
            result=result,
            samples=samples,
            telemetry_sampling=options.telemetry_sampling,
            system_cpu=system_cpu,
            memory_after=memory_after,
            wall_s=wall_s,
        )
        system_state = _build_system_state(
            samples=samples,
            captured_at=finished,
            system_cpu=system_cpu,
            memory_after=memory_after,
            hard_faults=metrics.hard_page_faults_per_sec,
        )
        quality = _metric_quality(metrics)
        backend_type = (
            backend_type_from_name(device_info.backend)
            if device_info is not None
            else requested.backend_type
        )
        config = requested.model_copy(
            update={
                "backend_type": backend_type,
                "backend_name": self.capabilities().name,
                "backend_version": self.capabilities().version,
                "device": telemetry.device,
                "gpu_layers_offloaded": telemetry.gpu_layers_offloaded,
                "gpu_layers_total": telemetry.gpu_layers_total,
            }
        )

        run_id = f"run-{sequence:04d}-{uuid.uuid4().hex[:8]}"
        workload_profile = workload.to_workload_profile()
        observation = Observation(
            observation_id=run_id,
            captured_at=finished,
            source=ObservationSource.HARNESS,
            mode=mode,
            backend=backend_type,
            device=telemetry.device,
            model_ref=model.name,
            workload_ref=workload_profile.workload_id,
            metrics=metrics,
            system_state=system_state,
            generated_text=result.text or None,
        )
        notes = (
            "warm-up run (excluded from aggregation)"
            if warmup
            else f"repetition {repetition} of measured {mode.value} runs"
        )
        if not options.telemetry_sampling:
            notes += "; external telemetry sampling disabled"
        _ = memory_before  # retained for symmetry/future interval-pressure diagnostics
        return ProfilingRun(
            identity=RunIdentity(
                run_id=run_id,
                profiler_version=PROFILER_VERSION,
                sequence=sequence,
                repetition=repetition,
                mode=mode,
                warmup=warmup,
                started_at=started,
                finished_at=finished,
            ),
            context=RunContext(
                hardware=self.hardware,
                model=model,
                workload=workload_profile,
                config=config,
            ),
            observation=observation,
            metric_quality=quality,
            notes=notes,
        )

    def profile(
        self,
        tasks: Sequence[ProfileTask],
        *,
        modes: Sequence[MeasurementMode],
        repetitions: int = 3,
        options: ProfilerOptions | None = None,
        suite_version: str = BENCHMARK_SUITE_VERSION,
    ) -> BaselineReport:
        options = options or ProfilerOptions()
        runs: list[ProfilingRun] = []
        sequence = 0
        for task in tasks:
            for mode in modes:
                if mode is MeasurementMode.COLD:
                    sequence += 1
                    runs.append(
                        self.run_once(
                            task,
                            mode=mode,
                            sequence=sequence,
                            repetition=0,
                            warmup=False,
                            options=options,
                        )
                    )
                else:
                    sequence += 1
                    runs.append(
                        self.run_once(
                            task,
                            mode=mode,
                            sequence=sequence,
                            repetition=0,
                            warmup=True,
                            options=options,
                        )
                    )
                    for repetition in range(max(repetitions, 1)):
                        sequence += 1
                        runs.append(
                            self.run_once(
                                task,
                                mode=mode,
                                sequence=sequence,
                                repetition=repetition,
                                warmup=False,
                                options=options,
                            )
                        )
        groups = _aggregate(runs)
        backend = runs[0].context.config if runs else ProfilingConfig()
        return BaselineReport(
            profiler_version=PROFILER_VERSION,
            suite_version=suite_version,
            generated_at=datetime.now(timezone.utc),
            hardware=self.hardware,
            backend=backend,
            workloads=workload_profiles([task.workload for task in tasks]),
            runs=runs,
            groups=groups,
            cold_warm_semantics=COLD_WARM_SEMANTICS,
            resident_warm_supported=False,
            limitations=list(LIMITATIONS),
        )


def _requested_config(task: ProfileTask, options: ProfilerOptions) -> ProfilingConfig:
    return ProfilingConfig(
        backend_name="llama.cpp",
        backend_type=(
            BackendType.VULKAN if options.gpu_offload_layers > 0 else BackendType.CPU
        ),
        device_selector=options.device,
        gpu_layers_offloaded=options.gpu_offload_layers,
        threads=options.threads,
        batch_size=options.batch_size,
        ubatch_size=options.ubatch_size,
        use_mmap=options.use_mmap,
        flash_attention=options.flash_attention,
        kv_cache_type=options.kv_cache_type,
        log_verbosity=options.log_verbosity,
        ignore_eos=task.workload.ignore_eos,
        context_length=task.context_length,
        n_predict=task.workload.n_predict,
        seed=task.workload.seed,
        temperature=task.workload.temperature,
        telemetry_sampling=options.telemetry_sampling,
    )


def _build_metrics(
    *,
    result: GenerationResult,
    samples: Sequence[CounterSample],
    telemetry_sampling: bool,
    system_cpu: float,
    memory_after: GlobalMemory | None,
    wall_s: float,
) -> RunMetrics:
    engine_peaks = gpu_engine_peaks(samples)
    gpu_util = _peak_utilization(engine_peaks)
    gpu_mem = _peak_adapter_memory(samples)
    hard_faults = max_value(samples, "pages input") if telemetry_sampling else None
    total_tokens = (
        result.prompt_tokens + result.generated_tokens
        if result.prompt_tokens is not None and result.generated_tokens is not None
        else None
    )
    memory = memory_after
    return RunMetrics(
        model_load_time_s=result.model_load_s,
        ttft_s=result.ttft_s,
        prefill_latency_s=result.prefill_s,
        decode_latency_s=result.decode_s,
        prompt_tokens_per_sec=result.prompt_tokens_per_sec,
        decode_tokens_per_sec=result.decode_tokens_per_sec,
        total_latency_s=result.total_s,
        process_wall_s=result.process_wall_s if result.process_wall_s is not None else wall_s,
        prompt_tokens=result.prompt_tokens,
        generated_tokens=result.generated_tokens,
        total_tokens=total_tokens,
        peak_rss_bytes=result.peak_rss_bytes,
        final_rss_bytes=result.final_rss_bytes,
        process_cpu_percent=result.peak_process_cpu_percent,
        system_cpu_percent=system_cpu,
        cpu_seconds=result.cpu_seconds,
        cpu_seconds_per_token=_per_token(result.cpu_seconds, result.generated_tokens),
        context_switches_voluntary=result.context_switches_voluntary,
        context_switches_involuntary=result.context_switches_involuntary,
        context_switches_total=result.context_switches_total,
        context_switches_per_token=_per_token(
            result.context_switches_total, result.generated_tokens
        ),
        gpu_utilization_percent=gpu_util,
        gpu_memory_used_bytes=gpu_mem,
        gpu_memory_total_bytes=None,
        hard_page_faults=None,
        hard_page_faults_per_sec=hard_faults,
        total_ram_bytes=memory.total_physical_bytes if memory is not None else None,
        available_ram_bytes=memory.available_physical_bytes if memory is not None else None,
        committed_bytes=(
            max(memory.total_pagefile_bytes - memory.available_pagefile_bytes, 0)
            if memory is not None
            else None
        ),
        commit_limit_bytes=memory.total_pagefile_bytes if memory is not None else None,
    )


def _per_token(value: float | int | None, tokens: int | None) -> float | None:
    """Normalize a per-generation quantity by generated tokens.

    Returns ``None`` (unknown) when the value is unavailable or the token count is
    missing/zero, so a ratio is never fabricated.
    """
    if value is None or tokens is None or tokens <= 0:
        return None
    return float(value) / float(tokens)


def _peak_utilization(engine_peaks: Sequence[tuple[str, str, float]]) -> float | None:
    if not engine_peaks:
        return None
    compute = [value for _luid, engine, value in engine_peaks if engine == "compute"]
    if compute:
        return min(max(compute), 100.0)
    return min(max(value for _luid, _engine, value in engine_peaks), 100.0)


def _peak_adapter_memory(samples: Sequence[CounterSample]) -> int | None:
    peaks = gpu_adapter_memory_peaks(samples)
    return max((value for _luid, value in peaks), default=None)


def _build_system_state(
    *,
    samples: Sequence[CounterSample],
    captured_at: datetime,
    system_cpu: float,
    memory_after: GlobalMemory | None,
    hard_faults: float | None,
) -> SystemState:
    engine_peaks = gpu_engine_peaks(samples)
    mem_peaks = gpu_adapter_memory_peaks(samples)
    luids = sorted({luid for luid, _e, _v in engine_peaks} | {luid for luid, _v in mem_peaks})
    index = {luid: position for position, luid in enumerate(luids)}
    engines = [
        GPUEngineUtilization(
            device_index=index[luid],
            engine=engine,
            utilization_percent=min(value, 100.0),
        )
        for luid, engine, value in engine_peaks
    ]
    memory = [
        GPUMemoryState(device_index=index[luid], memory_used_bytes=value, memory_total_bytes=None)
        for luid, value in mem_peaks
    ]
    if memory_after is not None:
        total = memory_after.total_physical_bytes
        available = memory_after.available_physical_bytes
        committed = max(memory_after.total_pagefile_bytes - memory_after.available_pagefile_bytes, 0)
        commit_limit = memory_after.total_pagefile_bytes
    else:  # pragma: no cover - Windows API available in practice
        total = 0
        available = 0
        committed = None
        commit_limit = None
    power = observe_power_performance()
    return SystemState(
        captured_at=captured_at,
        total_ram_bytes=total,
        available_ram_bytes=available,
        committed_bytes=committed,
        commit_limit_bytes=commit_limit,
        memory_pressure=compute_memory_pressure(total, available, committed, commit_limit),
        cpu_percent_total=system_cpu,
        hard_page_faults_per_sec=hard_faults if hard_faults is not None else 0.0,
        processes=[],
        gpu_engines=engines,
        gpu_memory=memory,
        target_process=None,
        power_state=power.power_state,
        cpu_current_mhz=power.current_mhz,
        cpu_max_mhz=power.max_mhz,
        cpu_limit_mhz=power.limit_mhz,
        cpu_performance_ratio=power.ratio,
    )


def _metric_quality(metrics: RunMetrics) -> dict[str, MetricQuality]:
    quality: dict[str, MetricQuality] = {}
    for name in _AGGREGATED_METRICS:
        value = getattr(metrics, name)
        if value is None:
            quality[name] = MetricQuality.UNAVAILABLE
        elif name in {"ttft_s", "total_tokens"}:
            quality[name] = MetricQuality.DERIVED
        else:
            quality[name] = MetricQuality.MEASURED
    quality["gpu_memory_total_bytes"] = MetricQuality.UNAVAILABLE
    quality["hard_page_faults"] = MetricQuality.UNAVAILABLE
    return quality


def _metric_value(metrics: RunMetrics, name: str) -> float | None:
    value = getattr(metrics, name)
    if value is None:
        return None
    return float(value)


def _aggregate(runs: Sequence[ProfilingRun]) -> list[RunGroupStats]:
    grouped: dict[tuple[str, str, MeasurementMode], list[ProfilingRun]] = {}
    for run in runs:
        if run.identity.warmup:
            continue
        key = (
            run.context.model.name,
            run.context.workload.workload_id,
            run.identity.mode,
        )
        grouped.setdefault(key, []).append(run)
    groups: list[RunGroupStats] = []
    for (model_ref, workload_id, mode), members in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2].value)
    ):
        metrics = {
            name: summarize(
                name, [_metric_value(member.observation.metrics, name) for member in members]
            )
            for name in _AGGREGATED_METRICS
        }
        groups.append(
            RunGroupStats(
                model_ref=model_ref,
                workload_id=workload_id,
                mode=mode,
                repetitions=len(members),
                metrics=metrics,
            )
        )
    return groups
