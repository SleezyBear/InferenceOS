"""Real measurement orchestration over the existing Phase 3 harness.

Phase 7 does not build a second benchmark system: it drives the same
:class:`BaselineProfiler` (same model, workload, telemetry, and metric
definitions) with a controlled runtime configuration and a bounded repetition
policy. Model performance metrics come from llama.cpp's own counters; wall-clock
orchestration time is tracked separately as optimizer overhead.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Protocol

from inferenceos.contracts.enums import MeasurementMode, MetricQuality
from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentPolicy,
    RuntimeConfiguration,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import MetricStats, ProfilingRun
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.statistics import summarize
from inferenceos.profiling.workloads import BenchmarkWorkload, choose_context_length

AGGREGATED_METRICS: tuple[str, ...] = (
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
    "peak_rss_bytes",
    "process_cpu_percent",
    "system_cpu_percent",
    # CPU efficiency / contention (lower is better); evaluated by the comparison.
    "cpu_seconds",
    "cpu_seconds_per_token",
    "context_switches_voluntary",
    "context_switches_involuntary",
    "context_switches_total",
    "context_switches_per_token",
    "gpu_utilization_percent",
    "gpu_memory_used_bytes",
    "hard_page_faults_per_sec",
    "available_ram_bytes",
    "total_ram_bytes",
    "committed_bytes",
    "commit_limit_bytes",
)


class MeasurementHarness(Protocol):
    """Structural interface for the Phase 3 harness (one real measured run)."""

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
        ...


#: Signs in a failure message that the setup does not fit in memory. Retrying the
#: same setup can only fail again (and every retry is a slow model reload), so the
#: measurement stops at the first such failure.
_OUT_OF_MEMORY_MARKERS = (
    "out of memory",
    "outofdevicememory",
    "outofhostmemory",
    "failed to allocate",
    "unable to allocate",
    "cannot allocate",
)


def is_out_of_memory(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _OUT_OF_MEMORY_MARKERS)


def options_for(config: RuntimeConfiguration) -> ProfilerOptions:
    return ProfilerOptions(
        gpu_offload_layers=(
            config.gpu_offload_layers if config.gpu_offload_layers is not None else 99
        ),
        telemetry_sampling=True,
        threads=config.threads,
        threads_batch=config.threads_batch,
        engine_switches=config.engine_switches,
        batch_size=config.batch_size,
        ubatch_size=config.ubatch_size,
        use_mmap=config.use_mmap,
        flash_attention=config.flash_attention,
        kv_cache_type=config.kv_cache_type,
        device=config.device,
    )


def measure_configuration(
    *,
    profiler: MeasurementHarness,
    model: ModelProfile,
    workload: BenchmarkWorkload,
    config: RuntimeConfiguration,
    policy: ExperimentPolicy,
    phase: str,
    seed: int = 1234,
    repetitions: int | None = None,
    deadline: float | None = None,
    max_runs: int | None = None,
) -> ExperimentMeasurement:
    context_length = config.context_length or choose_context_length(model)
    task = ProfileTask(model=model, workload=workload, context_length=context_length)
    options = options_for(config)
    runs: list[ProfilingRun] = []
    notes: list[str] = []
    invalid = 0
    sequence = 0
    budget = max_runs if max_runs is not None else policy.max_total_runs
    now = time.perf_counter()
    deadline = deadline if deadline is not None else now + policy.time_limit_s

    warmup_done = 0
    out_of_memory = False
    for _ in range(min(policy.warmup_repetitions, max(budget, 0))):
        if time.perf_counter() >= deadline:
            notes.append("time limit reached before warm-up completed")
            break
        sequence += 1
        warmup_done += 1
        try:
            profiler.run_once(
                task,
                mode=MeasurementMode.WARM,
                sequence=sequence,
                repetition=0,
                warmup=True,
                options=options,
            )
        except Exception as exc:  # noqa: BLE001 - a failed warm-up is not fatal
            invalid += 1
            notes.append(f"warm-up run failed: {type(exc).__name__}: {exc}")
            if is_out_of_memory(exc):
                out_of_memory = True
                notes.append("out of memory: this setup does not fit; not retried")
                break

    remaining = max(budget - warmup_done, 0)
    target = repetitions if repetitions is not None else policy.measured_repetitions
    target = max(0, min(target, policy.max_repetitions, remaining))
    repetition = 0
    if out_of_memory:
        target = 0
    while repetition < target:
        if time.perf_counter() >= deadline:
            notes.append("time limit reached before the requested repetitions")
            break
        sequence += 1
        try:
            run = profiler.run_once(
                task,
                mode=MeasurementMode.WARM,
                sequence=sequence,
                repetition=repetition,
                warmup=False,
                options=options,
            )
            runs.append(run)
        except Exception as exc:  # noqa: BLE001 - invalid runs are recorded, not hidden
            invalid += 1
            notes.append(f"measured run failed: {type(exc).__name__}: {exc}")
            if is_out_of_memory(exc):
                notes.append("out of memory: this setup does not fit; not retried")
                break
        repetition += 1
        if len(runs) >= policy.min_valid_samples and _stable(runs, policy):
            notes.append("early stop: observed variance is within the noise band")
            break

    metrics = {
        name: summarize(name, [_metric(run, name) for run in runs])
        for name in AGGREGATED_METRICS
    }
    metric_quality: dict[str, MetricQuality] = {}
    for run in runs:
        for name, quality in run.metric_quality.items():
            metric_quality.setdefault(name, quality)

    return ExperimentMeasurement(
        measurement_id=f"meas-{uuid.uuid4().hex[:10]}",
        created_at=datetime.now(timezone.utc),
        phase=phase,
        configuration=config,
        repetitions=len(runs),
        valid_runs=len(runs),
        invalid_runs=invalid,
        warmup_runs=warmup_done,
        total_runs=warmup_done + repetition,
        run_ids=[run.identity.run_id for run in runs],
        metrics=metrics,
        run_metrics=[run.observation.metrics for run in runs],
        generated_texts=[run.observation.generated_text or "" for run in runs],
        metric_quality=metric_quality,
        runs=runs,
        notes=notes,
    )


def _metric(run: ProfilingRun, name: str) -> float | None:
    value = getattr(run.observation.metrics, name, None)
    return float(value) if value is not None else None


def _stable(runs: list[ProfilingRun], policy: ExperimentPolicy) -> bool:
    """True when the primary throughput metric is already tightly measured."""
    values = [
        value
        for value in (_metric(run, "decode_tokens_per_sec") for run in runs)
        if value is not None
    ]
    if len(values) < policy.min_valid_samples:
        return False
    stats: MetricStats = summarize("decode_tokens_per_sec", values)
    if stats.cv is None:
        return True
    return stats.cv <= policy.noise_tolerance / 2.0
