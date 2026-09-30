"""Controlled fixtures for the autonomous-loop safety scenarios.

The real scenarios (healthy no-op and the full real optimization lifecycle) use the
real harness and the real llama.cpp/Vulkan backend. The deterministic, fault-injection
and adversarial scenarios need a *control* over measurement outcomes that the real
machine cannot provide on demand, so this module ships a scripted measurement harness
and synthetic reports explicitly labelled as verification-only.

Nothing here is used by ``inferenceos optimize`` or by any production path; it is the
loop analogue of the existing ``reasoning.providers.fake`` test double.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from inferenceos.contracts.enums import (
    BackendType,
    MeasurementMode,
    ObservationSource,
    RestorationStatus,
    WorkloadKind,
)
from inferenceos.contracts.experiment import RestorationResult, RuntimeConfiguration
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunIdentity,
)
from inferenceos.contracts.system_state import SystemState
from inferenceos.contracts.workload import ContextRequirement, WorkloadProfile
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.profiling.runner import COLD_WARM_SEMANTICS, ProfilerOptions, ProfileTask
from inferenceos.selftest import sample_hardware, sample_model

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass
class SampleRun:
    decode_tps: float = 20.0
    prompt_tps: float = 80.0
    total_latency_s: float = 2.0
    rss_bytes: int = 1_300_000_000
    hard_faults_per_sec: float = 0.5
    generated_tokens: int = 32
    load_s: float = 1.0
    cpu_percent: float = 120.0
    text: str = "A key value cache stores attention state."
    fail: bool = False


Responder = Callable[[ProfileTask, ProfilerOptions, int], SampleRun]


class ScriptedHarness:
    """A verification-only measurement harness producing deterministic real-run records."""

    def __init__(self, responder: Responder) -> None:
        self._responder = responder
        self.calls = 0

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
        self.calls += 1
        sample = self._responder(task, options, self.calls)
        if sample.fail:
            raise RuntimeError("scripted inference failure")
        return _make_run(task, options, sample, mode=mode, warmup=warmup, sequence=sequence)


class FaultyRestoreExecutor(RuntimeConfigurationExecutor):
    """Executor whose restore always fails (controlled restoration-failure seam)."""

    def restore(self, expected: RuntimeConfiguration) -> RestorationResult:
        return RestorationResult(
            attempted=True,
            status=RestorationStatus.FAILED,
            expected_configuration=expected,
            restored_configuration=self.current(),
            verified=False,
            reason="injected restoration failure (loop verification)",
            restored_at=datetime.now(timezone.utc),
        )


def no_improvement_responder(
    task: ProfileTask, options: ProfilerOptions, call: int
) -> SampleRun:
    return SampleRun(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def context_improvement_responder(
    task: ProfileTask, options: ProfilerOptions, call: int
) -> SampleRun:
    if (task.context_length or 0) <= 1024:
        return SampleRun(rss_bytes=900_000_000, hard_faults_per_sec=1.0, decode_tps=28.0)
    return SampleRun(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def quality_regression_responder(
    task: ProfileTask, options: ProfilerOptions, call: int
) -> SampleRun:
    candidate = (task.context_length or 0) <= 1024 or bool(options.use_mmap)
    if candidate:
        return SampleRun(
            rss_bytes=900_000_000,
            hard_faults_per_sec=1.0,
            decode_tps=30.0,
            text="!!!! 1234 5678",
        )
    return SampleRun(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def failing_responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleRun:
    return SampleRun(fail=True)


def make_paging_report(
    *, use_mmap: bool = False, context_length: int = 2048
) -> BaselineReport:
    runs = [
        _report_run("control", hard_faults=0.5, memory_pressure=0.6, use_mmap=use_mmap, context_length=context_length),
        _report_run("control", hard_faults=0.6, memory_pressure=0.6, use_mmap=use_mmap, context_length=context_length),
        _report_run("pressured", hard_faults=17000.0, memory_pressure=0.84, total_latency=2.5, use_mmap=use_mmap, context_length=context_length),
        _report_run("pressured", hard_faults=18000.0, memory_pressure=0.85, total_latency=2.6, use_mmap=use_mmap, context_length=context_length),
    ]
    return _assemble(runs, notes="controlled paging report")


def make_insufficient_report() -> BaselineReport:
    """Elevated pressure without corroboration: a below-threshold candidate signal."""
    runs = [
        _report_run("control", hard_faults=0.1, memory_pressure=0.6),
        _report_run("control", hard_faults=0.1, memory_pressure=0.6),
        _report_run("pressured", hard_faults=0.1, memory_pressure=0.85),
        _report_run("pressured", hard_faults=0.1, memory_pressure=0.86),
    ]
    return _assemble(runs, notes="controlled insufficient-evidence report")


def make_healthy_report() -> BaselineReport:
    runs = [
        _report_run("healthy", hard_faults=0.5, memory_pressure=0.5),
        _report_run("healthy", hard_faults=0.5, memory_pressure=0.5),
    ]
    return _assemble(runs, notes="controlled healthy report")


def report_for_config(base: BaselineReport, config: RuntimeConfiguration) -> BaselineReport:
    runs = []
    for run in base.runs:
        existing = run.context.config
        updated = existing.model_copy(
            update={
                "use_mmap": config.use_mmap,
                "context_length": config.context_length,
                "gpu_layers_offloaded": (
                    config.gpu_offload_layers
                    if config.gpu_offload_layers is not None
                    else existing.gpu_layers_offloaded
                ),
                "threads": config.threads if config.threads is not None else existing.threads,
            }
        )
        runs.append(run.model_copy(update={"context": run.context.model_copy(update={"config": updated})}))
    return _assemble(runs, notes="controlled re-observation")


def _report_run(
    condition: str,
    *,
    hard_faults: float,
    memory_pressure: float,
    total_latency: float = 1.4,
    use_mmap: bool = False,
    context_length: int = 2048,
    model_name: str = "verify-model",
    workload_id: str = "short-short",
    hard_required_context_tokens: int = 512,
) -> ProfilingRun:
    model = sample_model().model_copy(update={"name": model_name})
    workload = WorkloadProfile(
        workload_id=workload_id,
        kind=WorkloadKind.SHORT,
        description="verify",
        prompt_set_ref=f"{workload_id}@verify",
        prompts=["verify prompt"],
        seed=1,
        expected_context_length=context_length,
        context_requirement=ContextRequirement(
            hard_required_context_tokens=hard_required_context_tokens
        ),
    )
    metrics = RunMetrics(
        decode_tokens_per_sec=20.0 if hard_faults > 1.0 else 24.0,
        prompt_tokens_per_sec=90.0,
        total_latency_s=total_latency,
        prefill_latency_s=total_latency * 0.3,
        decode_latency_s=total_latency * 0.7,
        model_load_time_s=2.0,
        process_cpu_percent=120.0,
        system_cpu_percent=25.0,
        gpu_utilization_percent=70.0,
        peak_rss_bytes=1_300_000_000,
        available_ram_bytes=8 * 1024**3,
        total_ram_bytes=16 * 1024**3,
        hard_page_faults_per_sec=hard_faults,
        prompt_tokens=40,
        generated_tokens=24,
    )
    state = SystemState(
        captured_at=_NOW,
        total_ram_bytes=16 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        memory_pressure=memory_pressure,
        cpu_percent_total=25.0,
        hard_page_faults_per_sec=0.0,
        processes=[],
        target_process=None,
    )
    config = ProfilingConfig(
        backend_name="synthetic",
        backend_type=BackendType.VULKAN,
        gpu_layers_offloaded=99,
        gpu_layers_total=99,
        context_length=context_length,
        use_mmap=use_mmap,
        n_predict=24,
        seed=1,
        temperature=0.0,
        extra={"condition": condition},
    )
    observation = Observation(
        observation_id=f"obs-{condition}-{hard_faults}",
        captured_at=_NOW,
        mode=MeasurementMode.WARM,
        backend=BackendType.VULKAN,
        model_ref=model_name,
        workload_ref=workload_id,
        metrics=metrics,
        system_state=state,
        generated_text="A key value cache stores attention state.",
    )
    return ProfilingRun(
        identity=RunIdentity(
            run_id=f"run-{condition}-{hard_faults}",
            profiler_version="verify",
            mode=MeasurementMode.WARM,
            started_at=_NOW,
            finished_at=_NOW,
        ),
        context=RunContext(hardware=sample_hardware(), model=model, workload=workload, config=config),
        observation=observation,
        metric_quality={},
    )


def _assemble(runs: list[ProfilingRun], *, notes: str) -> BaselineReport:
    workloads = {run.context.workload.workload_id: run.context.workload for run in runs}
    return BaselineReport(
        profiler_version="verify",
        suite_version="verify",
        generated_at=_NOW,
        hardware=sample_hardware(),
        backend=runs[0].context.config,
        workloads=list(workloads.values()),
        runs=runs,
        groups=[],
        cold_warm_semantics=COLD_WARM_SEMANTICS,
        resident_warm_supported=False,
        notes=notes,
    )


def _make_run(
    task: ProfileTask,
    options: ProfilerOptions,
    sample: SampleRun,
    *,
    mode: MeasurementMode,
    warmup: bool,
    sequence: int,
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
        prompt_tokens=40,
        peak_rss_bytes=sample.rss_bytes,
        process_cpu_percent=sample.cpu_percent,
        system_cpu_percent=30.0,
        hard_page_faults_per_sec=sample.hard_faults_per_sec,
        available_ram_bytes=8 * 1024**3,
        total_ram_bytes=16 * 1024**3,
    )
    run_id = f"run-{sequence:04d}"
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
            profiler_version="verify",
            sequence=sequence,
            repetition=0,
            mode=mode,
            warmup=warmup,
            started_at=_NOW,
            finished_at=_NOW,
        ),
        context=RunContext(hardware=sample_hardware(), model=model, workload=workload, config=config),
        observation=observation,
        metric_quality={},
    )


__all__ = [
    "FaultyRestoreExecutor",
    "SampleRun",
    "ScriptedHarness",
    "context_improvement_responder",
    "failing_responder",
    "make_healthy_report",
    "make_insufficient_report",
    "make_paging_report",
    "no_improvement_responder",
    "quality_regression_responder",
    "report_for_config",
]
