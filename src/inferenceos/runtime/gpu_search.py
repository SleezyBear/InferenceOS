"""Resident-workload integration for the resource-aware GPU-offload search.

The runtime monitor already owns a **resident llama-server** through
:class:`inferenceos.runtime.resident_workload.ResidentActiveWorkload`, whose
``set_configuration`` performs a *controlled reload* (stop + start with the new
``-ngl``) while preserving process ownership, identity, API-key handling and cleanup.

This module lets the generic search run against that live target:

* :class:`ResidentMeasurementAdapter` is a :class:`BackendAdapter` shim whose ``load``
  routes through ``workload.set_configuration`` (controlled reload) and whose ``unload``
  is a deliberate no-op, so a candidate measurement never tears the target down. It is
  wrapped in the existing :class:`BaselineProfiler`, which therefore builds all real
  ``ProfilingRun`` metrics/telemetry exactly as the CLI harness does.
* :func:`run_workload_gpu_offload_search` builds the resource envelope from the runtime
  diagnosis context, constructs the evaluator, reuses the existing quality-validation
  machinery, and delegates to the generic :func:`run_gpu_offload_search`.

Nothing here applies a retained change: the monitor restores the known-good
configuration after the search and only the existing approval/experiment path may keep
a selected value.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable, Protocol

from inferenceos.backends.llama_cpp_toolchain import DeviceInfo
from inferenceos.config import Settings
from inferenceos.contracts.experiment import (
    ExperimentPolicy,
    QualityValidationResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.optimization import OptimizationCandidate
from inferenceos.contracts.search import GpuOffloadResourceEnvelope, GpuOffloadSearchResult
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.experiment.measurement import MeasurementHarness
from inferenceos.experiment.quality import validate_quality
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendCapabilities,
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.search.evaluation import (
    PointMeasurement,
    RuntimeConfigurationPointEvaluator,
)
from inferenceos.optimization.search.resources import envelope_from_context
from inferenceos.optimization.search.runner import run_gpu_offload_search
from inferenceos.profiling.runner import BaselineProfiler
from inferenceos.profiling.workloads import BenchmarkWorkload
from inferenceos.runtime.workload import ActiveWorkload

INCREASE_GPU_OFFLOAD_ACTION = "runtime.increase_gpu_offload_layers"

QualityEvaluator = Callable[
    [PointMeasurement, PointMeasurement], QualityValidationResult | None
]


class _ResidentAdapterLike(Protocol):
    """The subset of the resident adapter the measurement shim depends on."""

    def capabilities(self) -> BackendCapabilities: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def set_runtime_params(self, params: RuntimeParams) -> None: ...
    def telemetry(self) -> BackendTelemetry: ...
    @property
    def is_loaded(self) -> bool: ...
    @property
    def last_device(self) -> DeviceInfo | None: ...


def configuration_from_params(
    params: RuntimeParams, *, fallback: RuntimeConfiguration
) -> RuntimeConfiguration:
    """Map backend :class:`RuntimeParams` onto a runtime configuration.

    Fields the backend did not provide fall back to the current workload configuration
    so a candidate never silently drops the context size or model identity.
    """
    return RuntimeConfiguration(
        threads=params.threads if params.threads is not None else fallback.threads,
        batch_size=params.batch_size if params.batch_size is not None else fallback.batch_size,
        ubatch_size=(
            params.ubatch_size if params.ubatch_size is not None else fallback.ubatch_size
        ),
        gpu_offload_layers=(
            params.gpu_offload_layers
            if params.gpu_offload_layers is not None
            else fallback.gpu_offload_layers
        ),
        use_mmap=params.use_mmap if params.use_mmap is not None else fallback.use_mmap,
        context_length=(
            params.context_length if params.context_length is not None else fallback.context_length
        ),
        flash_attention=(
            params.flash_attention
            if params.flash_attention is not None
            else fallback.flash_attention
        ),
        kv_cache_type=(
            params.kv_cache_type if params.kv_cache_type is not None else fallback.kv_cache_type
        ),
        device=params.device if params.device is not None else fallback.device,
    )


class ResidentMeasurementAdapter(BackendAdapter):
    """Measure the live resident target without tearing it down.

    ``load`` applies the requested configuration through the workload's controlled
    reload; ``unload`` is intentionally inert so the resident server (and its
    ownership/JOB-object/API-key containment) survives every candidate. The monitor
    restores the known-good configuration after the search.
    """

    def __init__(self, workload: ActiveWorkload) -> None:
        self._workload = workload
        self._adapter: _ResidentAdapterLike = getattr(workload, "adapter")
        self._model = workload.model_profile()
        self._params = RuntimeParams()

    def capabilities(self) -> BackendCapabilities:
        return self._adapter.capabilities()

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        self._model = model
        self._params = params
        configuration = configuration_from_params(
            params, fallback=self._workload.configuration()
        )
        # Controlled reload owned by the resident workload (preserves identity/cleanup).
        self._workload.set_configuration(configuration)

    def unload(self) -> None:
        # Deliberately inert: the resident target stays alive and owned by the workload.
        return

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self._adapter.generate(request)

    def set_runtime_params(self, params: RuntimeParams) -> None:
        self._adapter.set_runtime_params(params)

    def telemetry(self) -> BackendTelemetry:
        return self._adapter.telemetry()

    @property
    def last_device(self) -> DeviceInfo | None:
        return self._adapter.last_device

    @property
    def is_loaded(self) -> bool:
        return bool(self._adapter.is_loaded)


def _is_resident_workload(workload: ActiveWorkload) -> bool:
    from inferenceos.runtime.resident_workload import ResidentActiveWorkload

    return isinstance(workload, ResidentActiveWorkload)


def build_search_harness(
    workload: ActiveWorkload,
    *,
    settings: Settings,
    hardware: HardwareProfile | None,
    injected: MeasurementHarness | None,
) -> MeasurementHarness:
    """Return the measurement harness the search should use.

    An explicitly injected harness always wins (tests / verification). For a resident
    workload the harness wraps the live adapter so candidates are measured against the
    *actual* resident server; otherwise the default harness is used.
    """
    if injected is not None:
        return injected
    if _is_resident_workload(workload) and getattr(workload, "adapter", None) is not None:
        return BaselineProfiler(
            settings=settings,
            adapter=ResidentMeasurementAdapter(workload),
            hardware=hardware,
        )
    return BaselineProfiler(settings=settings, hardware=hardware)


def benchmark_from_profile(profile: WorkloadProfile) -> BenchmarkWorkload:
    """Synthesize a real benchmark workload from a runtime workload profile.

    The runtime workload is never replaced: this only supplies the prompt/shape the
    measurement harness needs, and it preserves the workload identity/kind so a
    candidate is measured on the same work the resident target is actually serving.
    """
    prompt = profile.prompts[0] if profile.prompts else "In one sentence, respond."
    return BenchmarkWorkload(
        workload_id=profile.workload_id,
        kind=profile.kind,
        description=profile.description or "runtime workload",
        prompt=prompt,
        n_predict=32,
        seed=profile.seed,
        temperature=0.0,
        min_context_tokens=profile.expected_context_length or 64,
        ignore_eos=False,
    )


def make_quality_evaluator(
    *,
    workload: ActiveWorkload,
    settings: Settings,
    experiment_policy: ExperimentPolicy,
    action_id: str = INCREASE_GPU_OFFLOAD_ACTION,
) -> QualityEvaluator:
    benchmark = benchmark_from_profile(workload.workload_profile())
    model = workload.model_profile()

    def evaluate(
        baseline_point: PointMeasurement, candidate_point: PointMeasurement
    ) -> QualityValidationResult | None:
        if baseline_point.measurement is None or candidate_point.measurement is None:
            return None
        return validate_quality(
            baseline=baseline_point.measurement,
            candidate=candidate_point.measurement,
            workload=benchmark,
            policy=experiment_policy,
            settings=settings,
            action_id=action_id,
            diagnosed_bottleneck="gpu_underutilization",
            model=model,
        )

    return evaluate


def run_workload_gpu_offload_search(
    *,
    workload: ActiveWorkload,
    context: OptimizationContext,
    settings: Settings,
    optimization_policy: OptimizationPolicy,
    experiment_policy: ExperimentPolicy,
    harness: MeasurementHarness,
    base_configuration: RuntimeConfiguration,
    seed: int = 1234,
    paging_per_sec_ceiling: float | None = None,
    deadline: float | None = None,
) -> GpuOffloadSearchResult:
    """Run the resource-aware search against the active workload."""
    envelope: GpuOffloadResourceEnvelope = envelope_from_context(
        context, optimization_policy.search, now=datetime.now(timezone.utc)
    )
    benchmark = benchmark_from_profile(workload.workload_profile())
    context_length = (
        base_configuration.context_length
        or envelope.context_length
        or (context.config.context_length if context.config else None)
        or 2048
    )
    evaluator = RuntimeConfigurationPointEvaluator(
        profiler=harness,
        model=workload.model_profile(),
        workload=benchmark,
        experiment_policy=experiment_policy,
        context_length=int(context_length),
        envelope=envelope,
        base_configuration=base_configuration,
        seed=seed,
        paging_per_sec_ceiling=paging_per_sec_ceiling,
        deadline=deadline,
    )
    return run_gpu_offload_search(
        envelope=envelope,
        evaluator=evaluator,
        search_policy=optimization_policy.search,
        experiment_policy=experiment_policy,
        model_ref=workload.model_profile().name,
        quality_evaluator=make_quality_evaluator(
            workload=workload, settings=settings, experiment_policy=experiment_policy
        ),
    )


def selected_gpu_candidate(
    candidate: OptimizationCandidate,
    target: int,
    *,
    measured_relative: float | None = None,
    confidence: float | None = None,
) -> OptimizationCandidate:
    """Return the candidate bound to the empirically selected layer count.

    The selected value was chosen on measured *decode throughput*, so the candidate's
    bottleneck objective is re-declared to that metric; otherwise the Phase 7 decision
    would key on ``gpu_utilization_percent`` alone and could not recognise a real
    workload improvement (the task requires the complete objective, not GPU
    utilization alone). When the search measured a relative improvement it is carried
    as a known benefit estimate so the runtime economics reflect the real evidence.
    """
    from inferenceos.contracts.optimization import BenefitEstimate

    if measured_relative is not None and measured_relative > 0.0:
        benefit = BenefitEstimate(
            metric="decode_tokens_per_sec",
            direction="increase",
            unit="tokens/sec",
            known=True,
            magnitude_low=float(measured_relative),
            magnitude_high=float(measured_relative),
            confidence=confidence if confidence is not None else candidate.benefit.confidence,
            evidence_basis=list(candidate.benefit.evidence_basis),
            notes=(
                "measured relative decode-throughput improvement selected by the "
                "resource-aware GPU-offload search; Phase 7 independently validates it"
            ),
        )
    else:
        benefit = BenefitEstimate(
            metric="decode_tokens_per_sec",
            direction="increase",
            unit="tokens/sec",
            known=False,
            confidence=confidence if confidence is not None else candidate.benefit.confidence,
            evidence_basis=list(candidate.benefit.evidence_basis),
            notes=(
                "empirically selected by the resource-aware GPU-offload search; Phase 7 "
                "independently validates the retained effect"
            ),
        )
    return candidate.model_copy(
        update={
            "parameters": {"gpu_offload_layers": int(target)},
            "benefit": benefit,
        }
    )


def no_change_search_result(
    *,
    envelope: GpuOffloadResourceEnvelope,
    model_ref: str | None,
    reason: str,
    current_layers: int | None,
) -> GpuOffloadSearchResult:
    """A NO_CHANGE search record for cases where measurement cannot begin."""
    from inferenceos.contracts.enums import SearchSelectionKind
    from inferenceos.contracts.search import GpuOffloadSelection

    baseline = int(current_layers) if current_layers is not None else 0
    return GpuOffloadSearchResult(
        search_id=f"search-{uuid.uuid4().hex[:12]}",
        model_ref=model_ref,
        context_length=envelope.context_length,
        resource=envelope,
        sampled_layers=[baseline] if current_layers is not None else [],
        requested_layers=[],
        points=[],
        selection=GpuOffloadSelection(
            kind=SearchSelectionKind.NO_CHANGE,
            baseline_layers=baseline,
            reason=reason,
            comparisons=[],
        ),
        environment_restored=True,
        environments_mutated=False,
        limitations=list(envelope.reasons),
        summary=reason,
    )


__all__ = [
    "INCREASE_GPU_OFFLOAD_ACTION",
    "QualityEvaluator",
    "ResidentMeasurementAdapter",
    "benchmark_from_profile",
    "build_search_harness",
    "configuration_from_params",
    "make_quality_evaluator",
    "no_change_search_result",
    "run_workload_gpu_offload_search",
]
