"""Controlled fixtures for ``verify-phase10`` (verification-only).

The real acceptance scenarios use the real Phase 3/10 telemetry and the real
llama.cpp/Vulkan backend through :class:`ControlledLlamaWorkload`. The controlled,
fault-injection and adversarial scenarios need deterministic measurement outcomes the
real machine cannot produce on demand, so this module ships scripted workload/sampler
and static-harness doubles explicitly labelled as verification-only.

Nothing here is used by ``inferenceos monitor`` or any production path; it is the
Phase 10 analogue of ``reasoning.providers.fake``.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    BackendType,
    MeasurementMode,
    ObservationSource,
    WorkloadState,
)
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunIdentity,
)
from inferenceos.contracts.runtime import RuntimeObservation, WorkloadDescriptor, WorkloadSnapshot
from inferenceos.contracts.system_state import SystemState
from inferenceos.contracts.workload import ContextRequirement, WorkloadProfile
from inferenceos.native import get_platform
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.runtime.telemetry import RuntimeTelemetrySampler
from inferenceos.runtime.workload import ActiveWorkload, config_key
from inferenceos.selftest import sample_hardware, sample_model, sample_workload

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

_MEMORY_HOLDER_CODE = r"""
import sys, time
size = int(sys.argv[1])
buf = bytearray(size)
for i in range(0, size, 4096):
    buf[i] = 1
print('READY', flush=True)
time.sleep(600)
"""


# --------------------------------------------------------------------------- #
# Real induced-pressure helper                                                 #
# --------------------------------------------------------------------------- #
def start_memory_holder(
    *, leave_bytes: int, cap_bytes: int
) -> tuple[subprocess.Popen[str] | None, int]:
    """Launch a real memory-holding process (verification-only pressure source)."""
    memory = get_platform().global_memory_status()
    if memory is None:
        return None, 0
    hold = min(max(memory.available_physical_bytes - leave_bytes, 0), cap_bytes)
    if hold < 1024**3:
        return None, 0
    try:
        proc: subprocess.Popen[str] = subprocess.Popen(
            [sys.executable, "-c", _MEMORY_HOLDER_CODE, str(hold)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:  # pragma: no cover - defensive
        return None, 0
    if proc.stdout is not None:
        proc.stdout.readline()
    return proc, hold


def stop_memory_holder(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        proc.kill()


# --------------------------------------------------------------------------- #
# Scripted workload / sampler                                                  #
# --------------------------------------------------------------------------- #
class ScriptedWorkload(ActiveWorkload):
    """A controllable active workload for deterministic runtime scenarios."""

    def __init__(
        self,
        *,
        model: ModelProfile | None = None,
        workload: WorkloadProfile | None = None,
        configuration: RuntimeConfiguration | None = None,
        context_requirement: ContextRequirement | None = None,
        generating: bool = False,
        requests_served: int = 0,
        resume_ok: bool = True,
        resume_verifies: bool = True,
        deceptive_resume: bool = False,
        terminated: bool = False,
        checkpoint_supported: bool = True,
        pause_supported: bool = True,
    ) -> None:
        from inferenceos.profiling.workloads import get_workload

        self._model = model or sample_model()
        base_workload = workload or get_workload("short-short").to_workload_profile()
        if context_requirement is not None and base_workload.context_requirement is None:
            base_workload = base_workload.model_copy(
                update={"context_requirement": context_requirement}
            )
        elif workload is None and base_workload.context_requirement is None:
            # The default scripted scenario declares a hard context requirement so the
            # controlled context-reduction intervention path stays exercisable. A
            # workload passed explicitly keeps its own (possibly unknown) requirement.
            base_workload = base_workload.model_copy(
                update={
                    "context_requirement": ContextRequirement(
                        hard_required_context_tokens=512
                    )
                }
            )
        self._workload = base_workload
        self._configuration = configuration or RuntimeConfiguration(
            context_length=2048, gpu_offload_layers=99
        )
        self._running = True
        self._paused = False
        self._generating = generating
        self._requests = requests_served
        self._resume_ok = resume_ok
        self._resume_verifies = resume_verifies
        self._deceptive_resume = deceptive_resume
        self._terminated = terminated
        self._checkpoint_supported = checkpoint_supported
        self._pause_supported = pause_supported
        self.start_calls = 0
        self.pause_calls = 0
        self.resume_calls = 0
        self.set_calls: list[RuntimeConfiguration] = []

    def descriptor(self) -> WorkloadDescriptor:
        return WorkloadDescriptor(
            workload_id=self._workload.workload_id,
            workload_kind=self._workload.kind,
            model_ref=self._model.name,
            model_path=self._model.path,
            backend_name="llama.cpp",
            configuration=self._configuration,
            context_length=self._configuration.context_length,
            prompts_ref=self._workload.prompt_set_ref,
            checkpoint_supported=self._checkpoint_supported,
            pause_supported=self._pause_supported,
            resume_supported=True,
            limitations=["scripted verification workload (verify-phase10)"],
        )

    def snapshot(self) -> WorkloadSnapshot:
        if self._terminated:
            state = WorkloadState.TERMINATED_EXTERNALLY
        elif not self._running:
            state = WorkloadState.STOPPED
        elif self._paused:
            state = WorkloadState.PAUSED
        else:
            state = WorkloadState.RUNNING
        return WorkloadSnapshot(
            state=state,
            running=self._running and not self._terminated,
            generating=self._generating,
            terminated_externally=self._terminated,
            safe_to_pause=not self._generating,
            can_resume=self._running and not self._terminated,
            requests_served=self._requests,
        )

    def configuration(self) -> RuntimeConfiguration:
        return self._configuration

    def set_configuration(self, configuration: RuntimeConfiguration) -> None:
        self._configuration = configuration
        self.set_calls.append(configuration)

    def pause(self) -> bool:
        self.pause_calls += 1
        if not self._pause_supported:
            return False
        self._paused = True
        self._generating = False
        return True

    def resume(self) -> bool:
        self.resume_calls += 1
        if not self._resume_ok:
            return False
        if self._deceptive_resume:
            return True
        self._paused = False
        self._running = True
        self._requests += 1
        return True

    def start(self) -> None:
        self.start_calls += 1
        self._running = True

    def stop(self) -> None:
        self._running = False

    def terminate_externally(self) -> None:
        self._terminated = True
        self._running = False

    def model_profile(self) -> ModelProfile:
        return self._model

    def workload_profile(self) -> WorkloadProfile:
        return self._workload

    def verify_resume(self, *, previous_requests: int, timeout_s: float) -> bool:
        if self._deceptive_resume:
            return False
        return self._resume_verifies


def _system_state(memory_pressure: float, faults: float = 0.5) -> SystemState:
    return SystemState(
        captured_at=_NOW,
        total_ram_bytes=16 * 1024**3,
        available_ram_bytes=int((1.0 - memory_pressure) * 16 * 1024**3),
        memory_pressure=memory_pressure,
        cpu_percent_total=30.0,
        hard_page_faults_per_sec=faults,
        processes=[],
        target_process=None,
    )


class ScriptedRuntimeSampler(RuntimeTelemetrySampler):
    """Deterministic runtime sampler driven by a metrics function."""

    def __init__(self, metrics_for: "MetricsFor") -> None:
        self._metrics_for = metrics_for
        self.count = 0

    def sample(
        self, workload: ActiveWorkload, *, sequence: int, condition: str
    ) -> RuntimeObservation:
        self.count += 1
        metrics = dict(self._metrics_for(self.count, condition))
        pressure = metrics.get("memory_pressure", 0.5)
        faults = metrics.get("hard_page_faults_per_sec", 0.5)
        configuration = workload.configuration()
        snapshot = workload.snapshot()
        descriptor = workload.descriptor()
        return RuntimeObservation(
            observation_id=f"rtobs-{uuid.uuid4().hex[:8]}",
            sequence=sequence,
            captured_at=datetime.now(timezone.utc),
            workload_state=snapshot.state,
            generating=snapshot.generating,
            context_length=configuration.context_length,
            config_key=config_key(configuration),
            workload_ref=descriptor.workload_id,
            model_ref=descriptor.model_ref,
            metrics=metrics,
            system_state=_system_state(pressure, faults),
            condition=condition,
        )


MetricsFor = Callable[[int, str], dict[str, float]]


def healthy_metrics(sequence: int, condition: str) -> dict[str, float]:
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


def baseline_and_pressured_metrics(
    *,
    baseline_faults: float = 0.5,
    pressured_faults: float = 18000.0,
    baseline_pressure: float = 0.5,
    pressured_pressure: float = 0.85,
) -> MetricsFor:
    def metrics_for(sequence: int, condition: str) -> dict[str, float]:
        if condition == "baseline":
            return {
                "memory_pressure": baseline_pressure,
                "hard_page_faults_per_sec": baseline_faults,
                "available_ram_bytes": float(8 * 1024**3),
                "total_ram_bytes": float(16 * 1024**3),
                "system_cpu_percent": 25.0,
                "decode_tokens_per_sec": 24.0,
                "total_latency_s": 1.4,
                "peak_rss_bytes": 1_300_000_000.0,
            }
        return {
            "memory_pressure": pressured_pressure,
            "hard_page_faults_per_sec": pressured_faults,
            "available_ram_bytes": float(4 * 1024**3),
            "total_ram_bytes": float(16 * 1024**3),
            "system_cpu_percent": 55.0,
            "decode_tokens_per_sec": 9.0,
            "total_latency_s": 3.0,
            "peak_rss_bytes": 1_700_000_000.0,
        }

    return metrics_for


# --------------------------------------------------------------------------- #
# Static (scripted) measurement harness                                        #
# --------------------------------------------------------------------------- #
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


HarnessResponder = Callable[[ProfileTask, ProfilerOptions, int], SampleRun]


class StaticHarness:
    """A verification-only Phase 3-harness substitute returning scripted real-run records."""

    def __init__(self, responder: HarnessResponder) -> None:
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
        run_id = f"run-{sequence:04d}-{uuid.uuid4().hex[:6]}"
        return ProfilingRun(
            identity=RunIdentity(
                run_id=run_id,
                profiler_version="verify-phase10",
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
            observation=Observation(
                observation_id=run_id,
                captured_at=_NOW,
                source=ObservationSource.HARNESS,
                mode=mode,
                backend=config.backend_type,
                model_ref=model.name,
                workload_ref=workload.workload_id,
                metrics=metrics,
                generated_text=sample.text,
            ),
            metric_quality={},
        )


def improving_responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleRun:
    if (task.context_length or 0) <= 1024:
        return SampleRun(rss_bytes=900_000_000, hard_faults_per_sec=1.0, decode_tps=28.0)
    return SampleRun(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def no_improvement_responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleRun:
    return SampleRun(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def quality_regression_responder(
    task: ProfileTask, options: ProfilerOptions, call: int
) -> SampleRun:
    if (task.context_length or 0) <= 1024:
        return SampleRun(
            rss_bytes=900_000_000,
            hard_faults_per_sec=1.0,
            decode_tps=30.0,
            text="Completely unrelated output about a totally different subject entirely.",
        )
    return SampleRun(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def make_verification_workload(
    settings: Settings,
    *,
    model: ModelProfile | None = None,
    configuration: RuntimeConfiguration | None = None,
) -> ScriptedWorkload:
    _ = settings
    return ScriptedWorkload(
        model=model or sample_model(),
        workload=sample_workload(),
        configuration=configuration,
    )


__all__ = [
    "ControlledFootprintDetector",
    "ControlledGpuUnderutilizationDetector",
    "HarnessResponder",
    "MetricsFor",
    "SampleRun",
    "ScriptedRuntimeSampler",
    "ScriptedWorkload",
    "StaticHarness",
    "baseline_and_pressured_metrics",
    "healthy_metrics",
    "improving_responder",
    "make_verification_workload",
    "no_improvement_responder",
    "quality_regression_responder",
    "start_memory_holder",
    "stop_memory_holder",
]


class ControlledGpuUnderutilizationDetector:
    """Verification-only: surface a GPU-underutilization opportunity from real metrics.

    The bottleneck kind is forced so a bounded validation can exercise the runtime
    GPU-offload search, but every evidence value is read from the **real** runtime
    report (real GPU utilization, real process CPU, real configured offload). When the
    real detector already finds a meaningful opportunity, or the report has no usable
    partial-offload run, the real diagnosis is returned unchanged.
    """

    def __init__(self, *, force: bool = True, confidence: float = 0.75) -> None:
        from inferenceos.diagnosis.detector import BottleneckDetector

        self._real = BottleneckDetector()
        self._force = force
        self._confidence = confidence

    def detect(
        self, report: BaselineReport, *, source_report_path: str = "<runtime-monitor>"
    ) -> BottleneckDiagnosis:
        diagnosis = self._real.detect(report, source_report_path=source_report_path)
        if not self._force:
            return diagnosis
        # Bounded validation: exercise the runtime GPU-offload search deterministically
        # whenever the report carries a usable partial-offload run; otherwise fall back
        # to the real diagnosis. No metric is fabricated either way.
        return self._controlled_gpu(report, source_report_path) or diagnosis

    def _controlled_gpu(
        self, report: BaselineReport, source_report_path: str
    ) -> BottleneckDiagnosis | None:
        from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
        from inferenceos.contracts.diagnosis import BottleneckDiagnosis, DiagnosisContext
        from inferenceos.contracts.enums import (
            BackendType,
            BottleneckKind,
            EvidencePolarity,
            EvidenceStrength,
            InferenceStage,
        )
        from inferenceos.diagnosis.groups import condition_of

        usable = [
            run
            for run in report.runs
            if run.context.config.backend_type is BackendType.VULKAN
            and run.context.config.gpu_layers_offloaded
            and run.context.config.gpu_layers_total
            and run.context.config.gpu_layers_offloaded
            <= run.context.config.gpu_layers_total
            and run.observation.metrics.gpu_utilization_percent is not None
            and run.observation.metrics.process_cpu_percent is not None
        ]
        if not usable:
            return None
        # Ground the controlled opportunity in the *most recent* real observation so the
        # resource envelope uses fresh telemetry (never the stale baseline window).
        chosen = max(usable, key=lambda run: run.observation.captured_at)
        metrics = chosen.observation.metrics
        config = chosen.context.config
        gpu = float(metrics.gpu_utilization_percent or 0.0)
        cpu = float(metrics.process_cpu_percent or 0.0)
        model_ref = chosen.context.model.name
        workload_id = chosen.context.workload.workload_id
        condition = condition_of(chosen)
        subject = f"{model_ref}/{workload_id}/{chosen.identity.mode.value}"
        if condition:
            subject = f"{subject}[{condition}]"
        evidence = [
            EvidenceItem(
                metric="gpu_utilization_percent",
                observed_value=gpu,
                reference_value=40.0,
                description="real measured GPU utilization while offload is configured",
                polarity=EvidencePolarity.SUPPORTING,
                stage=InferenceStage.DECODE,
                strength=EvidenceStrength.MEDIUM,
                source="phase10.controlled.gpu_underutilization",
                evidence_id="E1",
            ),
            EvidenceItem(
                metric="process_cpu_percent",
                observed_value=cpu,
                reference_value=float(report.hardware.cpu.physical_cores * 100 * 0.7),
                description="real measured target process CPU during the opportunity",
                polarity=EvidencePolarity.SUPPORTING,
                stage=InferenceStage.DECODE,
                strength=EvidenceStrength.MEDIUM,
                source="phase10.controlled.gpu_underutilization",
                evidence_id="E2",
            ),
        ]
        primary = Bottleneck(
            bottleneck_id=f"bn-{uuid.uuid4().hex[:10]}-gpu",
            kind=BottleneckKind.GPU_UNDERUTILIZATION,
            confidence=self._confidence,
            is_meaningful=True,
            summary=(
                "controlled GPU-underutilization opportunity grounded in real measured "
                f"utilization ({gpu:.1f}%) with configured offload "
                f"{config.gpu_layers_offloaded}/{config.gpu_layers_total}"
            ),
            evidence=evidence,
            contributing_factors=["offload configured", "real measured utilization"],
            recommended_surface=[],
            source="phase10.controlled.gpu_underutilization",
            stage=InferenceStage.DECODE,
            subject=subject,
            evidence_strength=EvidenceStrength.MEDIUM,
            supporting_evidence=evidence,
            rationale=["real telemetry supports a resource-aware GPU-offload search"],
            detector_version="phase10.controlled",
        )
        return BottleneckDiagnosis(
            diagnosis_id=f"diag-{uuid.uuid4().hex[:10]}-gpu",
            created_at=datetime.now(timezone.utc),
            context=DiagnosisContext(
                source_report_path=source_report_path,
                detector_version="phase10.controlled",
                hardware=report.hardware,
                backend=config,
                model_refs=[model_ref],
                workload_ids=[workload_id],
                conditions=["controlled_gpu_underutilization_grounded_in_real_metrics"],
            ),
            meaningful_opportunity=True,
            insufficient_evidence=False,
            escalation_required=False,
            confidence=self._confidence,
            evidence_strength=EvidenceStrength.MEDIUM,
            primary=primary,
            supporting_evidence=evidence,
            summary="controlled runtime GPU-underutilization opportunity for Phase 10",
        )


class ControlledFootprintDetector:
    """Wraps the real detector and (optionally) forces a *controlled* opportunity.

    This is a verification-only double. The forced ``MODEL_FOOTPRINT`` diagnosis is
    grounded in the **real** measured ``peak_rss_bytes`` from the runtime report that
    the monitor itself built from live telemetry; it is explicitly labelled
    ``controlled``. When ``force`` is False (or the real detector already finds a
    meaningful opportunity) the real detector's diagnosis is returned unchanged, so the
    no-op scenario exercises the genuine detector.
    """

    def __init__(self, *, force: bool = True, confidence: float = 0.85) -> None:
        from inferenceos.diagnosis.detector import BottleneckDetector

        self._real = BottleneckDetector()
        self._force = force
        self._confidence = confidence

    def detect(
        self, report: BaselineReport, *, source_report_path: str = "<runtime-monitor>"
    ) -> BottleneckDiagnosis:
        diagnosis = self._real.detect(report, source_report_path=source_report_path)
        if not self._force or diagnosis.meaningful_opportunity:
            return diagnosis
        return self._controlled_footprint(report, source_report_path)

    def _controlled_footprint(
        self, report: BaselineReport, source_report_path: str
    ) -> BottleneckDiagnosis:
        from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
        from inferenceos.contracts.diagnosis import BottleneckDiagnosis, DiagnosisContext
        from inferenceos.contracts.enums import (
            BottleneckKind,
            EvidencePolarity,
            EvidenceStrength,
            InferenceStage,
        )

        peak = next(
            (
                float(run.observation.metrics.peak_rss_bytes)
                for run in report.runs
                if run.observation.metrics.peak_rss_bytes is not None
            ),
            0.0,
        )
        model_ref = report.runs[0].context.model.name if report.runs else "unknown"
        workload_id = (
            report.runs[0].context.workload.workload_id if report.runs else "unknown"
        )
        evidence = EvidenceItem(
            metric="peak_rss_bytes",
            observed_value=peak,
            description="real measured resident target RSS from this runtime session",
            polarity=EvidencePolarity.SUPPORTING,
            stage=InferenceStage.END_TO_END,
            strength=EvidenceStrength.HIGH,
            source="phase10r.controlled",
            evidence_id="E1",
        )
        primary = Bottleneck(
            bottleneck_id=f"bn-{uuid.uuid4().hex[:10]}-footprint",
            kind=BottleneckKind.MODEL_FOOTPRINT,
            confidence=self._confidence,
            is_meaningful=True,
            summary="controlled model-footprint opportunity grounded in live RSS",
            evidence=[evidence],
            contributing_factors=["resident model/KV footprint under active use"],
            recommended_surface=[],
            source="phase10r.controlled",
            stage=InferenceStage.END_TO_END,
            subject=f"{model_ref}/{workload_id}",
            evidence_strength=EvidenceStrength.HIGH,
            supporting_evidence=[evidence],
            rationale=["live measured resident RSS justifies context right-sizing"],
            detector_version="phase10r.controlled",
        )
        return BottleneckDiagnosis(
            diagnosis_id=f"diag-{uuid.uuid4().hex[:10]}-controlled",
            created_at=datetime.now(timezone.utc),
            context=DiagnosisContext(
                source_report_path=source_report_path,
                detector_version="phase10r.controlled",
                hardware=report.hardware,
                backend=report.backend,
                model_refs=[model_ref],
                workload_ids=[workload_id],
                conditions=["controlled_model_footprint_grounded_in_live_rss"],
            ),
            meaningful_opportunity=True,
            insufficient_evidence=False,
            escalation_required=False,
            confidence=self._confidence,
            evidence_strength=EvidenceStrength.HIGH,
            primary=primary,
            supporting_evidence=[evidence],
            summary="controlled runtime model-footprint opportunity for Phase 10R",
        )
