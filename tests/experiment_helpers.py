"""Shared fixtures for Phase 7 experiment tests."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from inferenceos.contracts.enums import (
    BackendType,
    MeasurementMode,
    ObservationSource,
)
from inferenceos.contracts.experiment import (
    ExperimentPolicy,
    ExperimentRequest,
    RuntimeConfiguration,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.optimization import CandidatePlan, OptimizationCandidate
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunIdentity,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import BenchmarkWorkload, get_workload
from inferenceos.selftest import sample_hardware, sample_model

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass
class SampleResult:
    decode_tps: float = 20.0
    prompt_tps: float = 80.0
    total_latency_s: float = 2.0
    rss_bytes: int = 1_300_000_000
    hard_faults_per_sec: float | None = 0.5
    generated_tokens: int = 32
    load_s: float = 1.0
    cpu_percent: float = 120.0
    text: str = "A key value cache stores attention state."
    fail: bool = False
    #: What the engine "printed" about its setup (``ProfilingConfig.observed``).
    observed: dict[str, object] | None = None


Responder = Callable[[ProfileTask, ProfilerOptions, int], SampleResult]


class ScriptedHarness:
    """A Phase 3 harness substitute: deterministic synthetic real-run records."""

    def __init__(self, responder: Responder) -> None:
        self._responder = responder
        self.calls: list[tuple[int | None, bool | None, int | None]] = []

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
        self.calls.append((task.context_length, options.use_mmap, options.gpu_offload_layers))
        sample = self._responder(task, options, len(self.calls))
        if sample.fail:
            raise RuntimeError("scripted inference failure")
        return make_run(
            task,
            options,
            sample,
            mode=mode,
            warmup=warmup,
            sequence=sequence,
            repetition=repetition,
        )


def make_run(
    task: ProfileTask,
    options: ProfilerOptions,
    sample: SampleResult,
    *,
    mode: MeasurementMode,
    warmup: bool,
    sequence: int,
    repetition: int,
) -> ProfilingRun:
    model = task.model
    workload = task.workload.to_workload_profile()
    offloaded = options.gpu_offload_layers or 0
    config = ProfilingConfig(
        backend_name="llama.cpp",
        backend_type=BackendType.CPU if offloaded == 0 else BackendType.VULKAN,
        gpu_layers_offloaded=offloaded,
        gpu_layers_total=offloaded,
        threads=options.threads,
        use_mmap=options.use_mmap,
        context_length=task.context_length,
        n_predict=task.workload.n_predict,
        seed=task.workload.seed,
        temperature=task.workload.temperature,
        observed=dict(sample.observed or {}),
    )
    metrics = RunMetrics(
        model_load_time_s=sample.load_s,
        ttft_s=sample.total_latency_s * 0.1,
        prefill_latency_s=sample.total_latency_s * 0.3,
        decode_latency_s=sample.total_latency_s * 0.7,
        prompt_tokens_per_sec=sample.prompt_tps,
        decode_tokens_per_sec=sample.decode_tps,
        total_latency_s=sample.total_latency_s,
        generated_tokens=sample.generated_tokens,
        prompt_tokens=max(len(task.workload.prompt) // 5, 1),
        peak_rss_bytes=sample.rss_bytes,
        process_cpu_percent=sample.cpu_percent,
        system_cpu_percent=30.0,
        hard_page_faults_per_sec=sample.hard_faults_per_sec,
        available_ram_bytes=8 * 1024**3,
        total_ram_bytes=16 * 1024**3,
    )
    run_id = f"run-{sequence:04d}-{uuid.uuid4().hex[:6]}"
    observation = Observation(
        observation_id=run_id,
        captured_at=_NOW,
        source=ObservationSource.HARNESS,
        mode=mode,
        backend=config.backend_type,
        model_ref=model.name,
        workload_ref=workload.workload_id,
        metrics=metrics,
        generated_text=sample.text,
    )
    return ProfilingRun(
        identity=RunIdentity(
            run_id=run_id,
            profiler_version="test",
            sequence=sequence,
            repetition=repetition,
            mode=mode,
            warmup=warmup,
            started_at=_NOW,
            finished_at=_NOW,
        ),
        context=RunContext(
            hardware=sample_hardware(), model=model, workload=workload, config=config
        ),
        observation=observation,
        metric_quality={},
    )


def build_test_policy(**overrides: object) -> ExperimentPolicy:
    base: dict[str, object] = {
        "warmup_repetitions": 1,
        "measured_repetitions": 2,
        "min_valid_samples": 2,
        "max_repetitions": 3,
        "time_limit_s": 300.0,
        "min_effect": 0.05,
        "noise_tolerance": 0.10,
    }
    base.update(overrides)
    return ExperimentPolicy.model_validate(base)


def make_request(
    *,
    candidate: OptimizationCandidate | None = None,
    policy: ExperimentPolicy | None = None,
    baseline: RuntimeConfiguration | None = None,
    experiment: RuntimeConfiguration | None = None,
    workload: BenchmarkWorkload | None = None,
    model: ModelProfile | None = None,
) -> ExperimentRequest:
    model = model or sample_model()
    workload = workload or get_workload("short-short")
    baseline = baseline or RuntimeConfiguration(
        context_length=2048, gpu_offload_layers=99, use_mmap=False
    )
    experiment = experiment or baseline.model_copy(update={"use_mmap": True})
    return ExperimentRequest(
        experiment_id="exp-test",
        candidate_id=candidate.candidate_id if candidate else "cand-test",
        action_id=candidate.action_id if candidate else "runtime.enable_mmap",
        parameters=dict(candidate.parameters) if candidate else {"use_mmap": True},
        source_diagnosis_id="diag-test",
        diagnosed_bottleneck="paging",
        bottleneck_metric="peak_rss_bytes",
        bottleneck_direction="decrease",
        model=model,
        hardware=sample_hardware(),
        workload=workload.to_workload_profile(),
        backend=ProfilingConfig(backend_name="llama.cpp", context_length=2048),
        baseline_configuration=baseline,
        experiment_configuration=experiment,
        policy=policy or build_test_policy(),
        seed=1234,
    )


def paging_candidate_and_plan() -> tuple[
    OptimizationCandidate, CandidatePlan, BaselineReport
]:
    """A real eligible Phase 6 candidate generated from a synthetic paging report."""
    from tests.reasoning_helpers import paging_bundle

    context, diagnosis, report = paging_bundle()
    plan = generate_candidates(
        diagnosis, report=report, registry=build_default_registry()
    )
    candidate = next(
        item
        for item in plan.candidates
        if item.candidate_id in plan.selected_ids and item.action_id == "runtime.enable_mmap"
    )
    return candidate, plan, report
