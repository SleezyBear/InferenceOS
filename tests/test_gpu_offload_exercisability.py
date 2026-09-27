"""GPU-offload exercisability: the existing policy path from runtime evidence to an
`increase_gpu_offload_layers` candidate.

These tests exercise the real diagnosis → candidate-generation pipeline for CPU-only,
partial-offload and full-offload configurations. They do not change any threshold; they
assert the corrected `offload_configured` applicability predicate (offload configured
with headroom on Vulkan) plus the unchanged CPU-saturation / GPU-utilization gates.
"""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    BackendType,
    BottleneckKind,
    CandidateEligibility,
    WorkloadState,
)
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.optimization import CandidatePlan
from inferenceos.contracts.profiling import BaselineReport, ProfilingConfig, ProfilingRun
from inferenceos.contracts.runtime import RuntimeObservation
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.diagnosis.groups import extract_groups
from inferenceos.optimization.generator import generate_candidates
from inferenceos.remediation import (
    _gpu_underutilization_diagnosis,
    _gpu_underutilization_report,
)
from inferenceos.runtime.approval import AutoRejectApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.notification import CollectingNotificationSink
from inferenceos.runtime.verification import ScriptedRuntimeSampler, ScriptedWorkload
from inferenceos.selftest import sample_hardware, sample_model
from tests.diagnosis_fixtures import make_report, make_run
from tests.runtime_helpers import make_runtime_policy

INCREASE = "runtime.increase_gpu_offload_layers"


def _report(
    *,
    offloaded: int,
    total: int,
    backend: BackendType,
    gpu: float | None = 5.0,
    cpu: float | None = 520.0,
    count: int = 2,
) -> BaselineReport:
    metrics: dict[str, object] = {
        "decode_tokens_per_sec": 6.0,
        "total_latency_s": 3.0,
        "available_ram_bytes": 8 * 1024**3,
        "total_ram_bytes": 16 * 1024**3,
        "gpu_memory_used_bytes": 700 * 1024**2,
    }
    if gpu is not None:
        metrics["gpu_utilization_percent"] = gpu
    if cpu is not None:
        metrics["process_cpu_percent"] = cpu
    observed = datetime.now(timezone.utc)
    runs = [
        make_run(
            model_name="qwen-ab",
            workload_id="short-short",
            condition="partial",
            gpu_offload_layers=offloaded,
            gpu_offload_total=total,
            backend=backend,
            metrics=metrics,
            gpu_telemetry_available=backend is BackendType.VULKAN,
            gpu_observed_at=observed,
        )
        for _ in range(count)
    ]
    return make_report(runs)


def _diagnose_and_generate(report: BaselineReport) -> tuple[BottleneckDiagnosis, CandidatePlan]:
    diagnosis = BottleneckDetector().detect(report)
    plan = generate_candidates(diagnosis, report=report)
    return diagnosis, plan


def _has_gpu_underutilization(diagnosis: BottleneckDiagnosis) -> bool:
    return any(b.kind is BottleneckKind.GPU_UNDERUTILIZATION for b in diagnosis.candidates) or (
        diagnosis.primary is not None
        and diagnosis.primary.kind is BottleneckKind.GPU_UNDERUTILIZATION
    )


# --------------------------------------------------------------------------- #
# Applicability predicate                                                       #
# --------------------------------------------------------------------------- #
def test_offload_configured_semantics() -> None:
    full = extract_groups(
        _report(offloaded=28, total=28, backend=BackendType.VULKAN)
    )
    partial = extract_groups(
        _report(offloaded=14, total=28, backend=BackendType.VULKAN)
    )
    cpu_only = extract_groups(
        _report(offloaded=0, total=28, backend=BackendType.CPU)
    )
    assert full[0].offload_configured is True
    assert partial[0].offload_configured is True  # configured with headroom
    assert cpu_only[0].offload_configured is False


# --------------------------------------------------------------------------- #
# A. CPU-only: the GPU rule must not misfire                                     #
# --------------------------------------------------------------------------- #
def test_a_cpu_only_does_not_diagnose_gpu_underutilization() -> None:
    report = _report(offloaded=0, total=28, backend=BackendType.CPU)
    diagnosis, plan = _diagnose_and_generate(report)
    assert not _has_gpu_underutilization(diagnosis)
    assert all(c.action_id != INCREASE for c in plan.candidates)


# --------------------------------------------------------------------------- #
# B. Partial Vulkan offload -> GPU_UNDERUTILIZATION -> eligible increase          #
# --------------------------------------------------------------------------- #
def test_b_partial_offload_generates_eligible_increase_candidate() -> None:
    report = _report(offloaded=14, total=28, backend=BackendType.VULKAN)
    diagnosis, plan = _diagnose_and_generate(report)
    assert diagnosis.meaningful_opportunity is True
    assert diagnosis.primary is not None
    assert diagnosis.primary.kind is BottleneckKind.GPU_UNDERUTILIZATION
    increase = [c for c in plan.candidates if c.action_id == INCREASE]
    assert increase, "expected an increase_gpu_offload_layers candidate"
    assert increase[0].eligibility is CandidateEligibility.ELIGIBLE
    assert increase[0].executable is True


# --------------------------------------------------------------------------- #
# C. Full Vulkan offload -> rule fires but increase has no headroom (no-op)      #
# --------------------------------------------------------------------------- #
def test_c_full_offload_cannot_increase() -> None:
    report = _report(offloaded=28, total=28, backend=BackendType.VULKAN)
    diagnosis, plan = _diagnose_and_generate(report)
    assert _has_gpu_underutilization(diagnosis)
    increase = [c for c in plan.candidates if c.action_id == INCREASE]
    assert all(c.eligibility is not CandidateEligibility.ELIGIBLE for c in increase)
    assert all(not c.executable for c in increase)


# --------------------------------------------------------------------------- #
# D. Missing CPU evidence -> no fabricated saturation                            #
# --------------------------------------------------------------------------- #
def test_d_missing_cpu_evidence_no_gpu_diagnosis() -> None:
    report = _report(offloaded=14, total=28, backend=BackendType.VULKAN, cpu=None)
    diagnosis, plan = _diagnose_and_generate(report)
    assert not _has_gpu_underutilization(diagnosis)
    assert all(c.action_id != INCREASE for c in plan.candidates)


# --------------------------------------------------------------------------- #
# E. Missing GPU telemetry -> no fabricated zero utilization                     #
# --------------------------------------------------------------------------- #
def test_e_missing_gpu_telemetry_no_gpu_diagnosis() -> None:
    report = _report(offloaded=14, total=28, backend=BackendType.VULKAN, gpu=None)
    diagnosis, plan = _diagnose_and_generate(report)
    assert not _has_gpu_underutilization(diagnosis)
    assert all(c.action_id != INCREASE for c in plan.candidates)


# --------------------------------------------------------------------------- #
# F. Candidate target is strictly above the current offload                      #
# --------------------------------------------------------------------------- #
def test_f_candidate_target_above_current() -> None:
    report = _report(offloaded=14, total=28, backend=BackendType.VULKAN)
    _, plan = _diagnose_and_generate(report)
    increase = next(c for c in plan.candidates if c.action_id == INCREASE)
    target = increase.parameters["gpu_offload_layers"]
    assert isinstance(target, int)
    assert target > 14
    assert target <= 28
    # The resource-aware planner proposes a bounded probe, never the full layer total.
    assert target < 28


# --------------------------------------------------------------------------- #
# Regression guards (previous verification, still true)                          #
# --------------------------------------------------------------------------- #
def test_ngl0_is_not_diagnosed_as_gpu_underutilization() -> None:
    report = _report(offloaded=0, total=28, backend=BackendType.CPU)
    diagnosis = BottleneckDetector().detect(report)
    assert not _has_gpu_underutilization(diagnosis)


def test_forced_ngl0_gpu_diagnosis_is_a_noop_for_increase() -> None:
    diagnosis = _gpu_underutilization_diagnosis(Settings())
    context = diagnosis.context.model_copy(
        update={
            "backend": ProfilingConfig(
                backend_name="llama.cpp (resident)",
                backend_type=BackendType.CPU,
                gpu_layers_offloaded=0,
                gpu_layers_total=0,
                context_length=4096,
            )
        }
    )
    forced = diagnosis.model_copy(update={"context": context})
    plan = generate_candidates(forced)
    increase = [c for c in plan.candidates if c.action_id == INCREASE]
    assert all(c.eligibility is not CandidateEligibility.ELIGIBLE for c in increase)


def test_controlled_diagnosis_generates_increase_candidate_when_gates_satisfied() -> None:
    settings = Settings()
    diagnosis = _gpu_underutilization_diagnosis(settings)
    report = _gpu_underutilization_report(settings)
    plan = generate_candidates(diagnosis, report=report)
    increase = [c for c in plan.candidates if c.action_id == INCREASE]
    assert increase, "the increase action should be mapped for GPU_UNDERUTILIZATION"
    assert increase[0].eligibility is CandidateEligibility.ELIGIBLE
    target = increase[0].parameters["gpu_offload_layers"]
    assert isinstance(target, int)
    # Resource-aware: a bounded probe within the feasible range, not the layer total.
    assert 14 < target < 28


# --------------------------------------------------------------------------- #
# Runtime report representation of partial offload                              #
# --------------------------------------------------------------------------- #
class _PartialWorkload(ScriptedWorkload):
    def __init__(self, model: ModelProfile, configuration: RuntimeConfiguration) -> None:
        super().__init__(configuration=configuration)
        self._model = model

    def model_profile(self) -> ModelProfile:
        return self._model


def _to_run_for(offload: int) -> ProfilingRun:
    model = sample_model().model_copy(update={"n_layers": 28})
    configuration = RuntimeConfiguration(context_length=4096, gpu_offload_layers=offload)
    workload = _PartialWorkload(model, configuration)
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=ScriptedRuntimeSampler(lambda sequence, condition: {}),
        policy=make_runtime_policy(),
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
    )
    observation = RuntimeObservation(
        observation_id="obs-offload",
        captured_at=datetime.now(timezone.utc),
        workload_state=WorkloadState.RUNNING,
        context_length=4096,
        metrics={
            "process_cpu_percent": 520.0,
            "gpu_utilization_percent": 5.0,
            "decode_tokens_per_sec": 6.0,
        },
    )
    return monitor._to_run(
        observation,
        condition="partial",
        model=model,
        workload=workload.workload_profile(),
        hardware=sample_hardware(),
    )


def test_to_run_represents_partial_offload_truthfully() -> None:
    run = _to_run_for(14)
    config = run.context.config
    assert config.gpu_layers_offloaded == 14
    assert config.gpu_layers_total == 28
    assert config.backend_type is BackendType.VULKAN
    assert run.observation.metrics.process_cpu_percent == 520.0


def test_to_run_represents_cpu_only_and_full_offload() -> None:
    cpu_only = _to_run_for(0).context.config
    assert cpu_only.gpu_layers_offloaded == 0
    assert cpu_only.gpu_layers_total == 28
    assert cpu_only.backend_type is BackendType.CPU
    full = _to_run_for(99).context.config
    assert full.gpu_layers_offloaded == 28  # clamped to the model total, not 99
    assert full.gpu_layers_total == 28
    assert full.backend_type is BackendType.VULKAN


def test_to_run_peak_rss_is_at_least_target_current_rss() -> None:
    from inferenceos.contracts.system_state import ProcessState, SystemState

    model = sample_model().model_copy(update={"n_layers": 32})
    configuration = RuntimeConfiguration(context_length=4096, gpu_offload_layers=8)
    workload = _PartialWorkload(model, configuration)
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=ScriptedRuntimeSampler(lambda sequence, condition: {}),
        policy=make_runtime_policy(),
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
    )
    state = SystemState(
        captured_at=datetime.now(timezone.utc),
        total_ram_bytes=16 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        memory_pressure=0.5,
        cpu_percent_total=10.0,
        hard_page_faults_per_sec=0.0,
        target_process=ProcessState(
            pid=1, name="llama-server.exe", cpu_percent=500.0, rss_bytes=2000, vms_bytes=3000
        ),
    )
    observation = RuntimeObservation(
        observation_id="obs-peak",
        captured_at=datetime.now(timezone.utc),
        workload_state=WorkloadState.RUNNING,
        context_length=4096,
        metrics={"peak_rss_bytes": 1500.0},
        system_state=state,
    )
    run = monitor._to_run(
        observation,
        condition="partial",
        model=model,
        workload=workload.workload_profile(),
        hardware=sample_hardware(),
    )
    # The reported peak is never below the observed current target RSS (a true lower
    # bound), so the target cannot be misread as an external memory consumer.
    assert run.observation.metrics.peak_rss_bytes == 2000

