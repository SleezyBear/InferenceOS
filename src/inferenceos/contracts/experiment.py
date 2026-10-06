"""Experiment contracts (Phase 7 real before/after + quality validation).

Phase 0 defined the minimal ``ExperimentResult``. Phase 7 extends it additively into
a complete, auditable experiment lifecycle: request, baseline snapshot, real
measurements, quality validation, statistical comparison, decision, application and
restoration results, lifecycle events, and separately-measured optimizer overhead.

A Phase 7 experiment applies exactly one Phase 6 candidate's *runtime configuration*
through a controlled executor, measures real inference before and after, validates
output quality, decides, and restores the protected baseline. It never persists
deployments or learns from history (Phase 8+).
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.quality import (
    AdaptiveQualityPolicy,
    QualityCaseResult,
    QualitySuiteResult,
)
from inferenceos.contracts.enums import (
    BackendType,
    Decision,
    EvidenceStrength,
    ExperimentDecisionKind,
    ExperimentStage,
    MetricQuality,
    QualityStatus,
    RestorationStatus,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.observation import RunMetrics
from inferenceos.contracts.profiling import MetricStats, ProfilingConfig, ProfilingRun
from inferenceos.contracts.workload import WorkloadProfile

EXPERIMENT_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Runtime configuration + policy                                               #
# --------------------------------------------------------------------------- #
class RuntimeConfiguration(ContractBase):
    """The target model's runtime configuration that an experiment mutates.

    This is the inference-level surface only (no OS/process/system-wide state).
    """

    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    gpu_offload_layers: int | None = None
    use_mmap: bool | None = None
    context_length: int | None = None
    flash_attention: bool | None = None
    # Threads used to read the question (prompt), when different from ``threads``
    # (llama.cpp ``-tb``; unset = the same as ``threads``).
    threads_batch: int | None = None
    kv_cache_type: str | None = None
    device: str | None = None
    # Engine switches (environment variables the engine reads at start), canonical
    # "NAME=VALUE;NAME=VALUE" sorted by name; None = none (``tune/engine_switches.py``).
    engine_switches: str | None = None

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()


class ExperimentPolicy(ContractBase):
    """Explicit, inspectable Phase 7 experiment policy (no magic constants)."""

    warmup_repetitions: int = Field(default=1, ge=0)
    measured_repetitions: int = Field(default=3, ge=1)
    max_repetitions: int = Field(default=5, ge=1)
    min_valid_samples: int = Field(default=2, ge=1)
    max_total_runs: int = Field(default=12, ge=1)
    time_limit_s: float = Field(default=900.0, gt=0.0)
    run_timeout_s: float = Field(default=300.0, gt=0.0)
    min_effect: float = Field(default=0.05, ge=0.0)
    noise_tolerance: float = Field(default=0.10, ge=0.0)
    quality_similarity_min: float = Field(default=0.60, ge=0.0, le=1.0)
    require_quality_validation: bool = True
    restore_baseline: bool = True
    abort_on_restoration_failure: bool = True
    isolation: str = "single_target_model_runtime"

    def as_dict(self) -> dict[str, object]:
        return {
            "warmup_repetitions": self.warmup_repetitions,
            "measured_repetitions": self.measured_repetitions,
            "max_repetitions": self.max_repetitions,
            "min_valid_samples": self.min_valid_samples,
            "max_total_runs": self.max_total_runs,
            "time_limit_s": self.time_limit_s,
            "run_timeout_s": self.run_timeout_s,
            "min_effect": self.min_effect,
            "noise_tolerance": self.noise_tolerance,
            "quality_similarity_min": self.quality_similarity_min,
            "require_quality_validation": self.require_quality_validation,
            "restore_baseline": self.restore_baseline,
            "abort_on_restoration_failure": self.abort_on_restoration_failure,
            "isolation": self.isolation,
        }


# --------------------------------------------------------------------------- #
# Lifecycle / request                                                          #
# --------------------------------------------------------------------------- #
class ExperimentEvent(ContractBase):
    stage: ExperimentStage
    timestamp: datetime = Field(default_factory=_utcnow)
    detail: str = ""
    data: dict[str, object] = Field(default_factory=dict)


class ExperimentRequest(ContractBase):
    experiment_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    run_id: str | None = None
    candidate_id: str
    action_id: str
    parameters: dict[str, object] = Field(default_factory=dict)
    source_diagnosis_id: str | None = None
    source_plan_id: str | None = None
    source_bottleneck_id: str | None = None
    diagnosed_bottleneck: str | None = None
    diagnosis_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    diagnosis_evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    bottleneck_metric: str | None = None
    bottleneck_direction: str | None = None

    model: ModelProfile
    hardware: HardwareProfile
    workload: WorkloadProfile
    backend: ProfilingConfig
    baseline_configuration: RuntimeConfiguration
    experiment_configuration: RuntimeConfiguration
    policy: ExperimentPolicy = Field(default_factory=ExperimentPolicy)

    seed: int = 1234
    time_limit_s: float = 900.0
    reproducibility: dict[str, object] = Field(default_factory=dict)
    source_report_path: str | None = None
    source_report_hash: str | None = None


# --------------------------------------------------------------------------- #
# Baseline / measurement                                                       #
# --------------------------------------------------------------------------- #
class BaselineSnapshot(ContractBase):
    snapshot_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    configuration: RuntimeConfiguration
    model_ref: str
    model_path: str
    model_file_bytes: int | None = None
    model_content_hash: str | None = None
    backend_name: str | None = None
    backend_version: str | None = None
    backend_type: BackendType | None = None
    device: str | None = None
    hardware_hostname: str | None = None
    workload_id: str
    workload_prompt_hash: str | None = None
    seed: int = 1234
    baseline_run_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricStats] = Field(default_factory=dict)
    quality_reference_text: str | None = None
    quality_reference_hash: str | None = None
    provenance: str = "phase3-harness"


class ExperimentMeasurement(ContractBase):
    measurement_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    phase: str
    configuration: RuntimeConfiguration
    repetitions: int = Field(default=0, ge=0)
    valid_runs: int = Field(default=0, ge=0)
    invalid_runs: int = Field(default=0, ge=0)
    warmup_runs: int = Field(default=0, ge=0)
    total_runs: int = Field(default=0, ge=0)
    run_ids: list[str] = Field(default_factory=list)
    metrics: dict[str, MetricStats] = Field(default_factory=dict)
    run_metrics: list[RunMetrics] = Field(default_factory=list)
    generated_texts: list[str] = Field(default_factory=list)
    metric_quality: dict[str, MetricQuality] = Field(default_factory=dict)
    runs: list[ProfilingRun] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Quality                                                                      #
# --------------------------------------------------------------------------- #
class QualityCheck(ContractBase):
    check_id: str
    description: str
    passed: bool
    detail: str = ""
    hard: bool = True


class QualityValidationResult(ContractBase):
    status: QualityStatus
    validated: bool
    execution_ok: bool
    structural_ok: bool
    reference_available: bool = False
    deterministic_match: bool | None = None
    similarity: float | None = None
    unacceptable_output: bool = False
    quality_regression: bool = False
    checks: list[QualityCheck] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    baseline_text_hash: str | None = None
    candidate_text_hash: str | None = None

    # Phase 7R: versioned suite + adaptive tolerance evidence.
    suite_id: str | None = None
    suite_version: str | None = None
    suite: QualitySuiteResult | None = None
    policy: AdaptiveQualityPolicy | None = None
    cases: list[QualityCaseResult] = Field(default_factory=list)
    aggregate_score: float | None = None
    tolerance: float | None = None
    tolerance_basis: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Comparison + decision                                                        #
# --------------------------------------------------------------------------- #
class MeasurementDelta(ContractBase):
    metric: str
    role: str = "other"
    direction: str = "higher_is_better"
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    absolute_delta: float | None = None
    relative_delta: float | None = None
    baseline_stdev: float | None = None
    baseline_cv: float | None = None
    candidate_cv: float | None = None
    sample_count: int = Field(default=0, ge=0)
    noise_band: float | None = None
    improved: bool | None = None
    regressed: bool | None = None
    significant: bool = False


class ExperimentComparison(ContractBase):
    deltas: dict[str, MeasurementDelta] = Field(default_factory=dict)
    bottleneck_metric: str | None = None
    bottleneck_direction: str | None = None
    bottleneck_improved: bool = False
    bottleneck_regressed: bool = False
    bottleneck_persistent: bool = True
    objective_improved: bool = False
    objective_regressed: bool = False
    resource_regressed: bool = False
    quality_change: str = "unknown"
    sample_count: int = Field(default=0, ge=0)
    decision_thresholds: dict[str, float] = Field(default_factory=dict)
    uncertainty: str = ""
    rationale: list[str] = Field(default_factory=list)


class ExperimentDecision(ContractBase):
    kind: ExperimentDecisionKind
    keep: bool
    reason: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    regression: bool = False
    quality_regression: bool = False
    bottleneck_improved: bool = False
    restoration_status: RestorationStatus = RestorationStatus.PENDING
    decided_at: datetime = Field(default_factory=_utcnow)


# --------------------------------------------------------------------------- #
# Application / restoration / overhead                                         #
# --------------------------------------------------------------------------- #
class ExperimentChange(ContractBase):
    parameter: str
    prior_value: object = None
    applied_value: object = None


class ApplicationResult(ContractBase):
    applied: bool
    action_id: str
    supported: bool
    changes: list[ExperimentChange] = Field(default_factory=list)
    reason: str = ""
    applied_at: datetime | None = None


class RestorationResult(ContractBase):
    attempted: bool
    status: RestorationStatus
    expected_configuration: RuntimeConfiguration | None = None
    restored_configuration: RuntimeConfiguration | None = None
    verified: bool = False
    changes: list[ExperimentChange] = Field(default_factory=list)
    reason: str = ""
    restored_at: datetime | None = None
    # Phase 7R: independent, real post-restoration verification.
    verification: "RestorationVerification | None" = None


class RestorationVerification(ContractBase):
    """Independent verification that the target really is back at baseline.

    Populated from a real generation/measurement against the restored target, not
    from Python-object equality.
    """

    attempted: bool
    running: bool
    configuration_match: bool
    probe_match: bool
    verified: bool
    baseline_probe_hash: str | None = None
    restored_probe_hash: str | None = None
    baseline_latency_s: float | None = None
    restored_latency_s: float | None = None
    latency_ratio: float | None = None
    within_tolerance: bool = False
    reason: str = ""
    measured_at: datetime | None = None
    notes: list[str] = Field(default_factory=list)


class ExperimentProvenance(ContractBase):
    """Binds an experiment/result to the exact evidence that produced it."""

    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    source_report_path: str | None = None
    source_report_hash: str | None = None
    diagnosis_id: str | None = None
    plan_id: str | None = None
    candidate_id: str | None = None
    action_id: str | None = None
    parameters_hash: str | None = None
    model_path: str | None = None
    model_content_hash: str | None = None


class ExperimentOverhead(ContractBase):
    """Optimizer/experiment cost, kept separate from measured workload metrics."""

    baseline_setup_s: float = Field(default=0.0, ge=0.0)
    application_s: float = Field(default=0.0, ge=0.0)
    measurement_s: float = Field(default=0.0, ge=0.0)
    validation_s: float = Field(default=0.0, ge=0.0)
    restoration_s: float = Field(default=0.0, ge=0.0)
    total_s: float = Field(default=0.0, ge=0.0)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Result (extends the Phase 0 minimal shape additively)                        #
# --------------------------------------------------------------------------- #
class ExperimentResult(ContractBase):
    schema_version: str = EXPERIMENT_SCHEMA_VERSION
    experiment_id: str
    run_id: str | None = None
    provenance: ExperimentProvenance | None = None
    candidate_id: str
    # Phase 0 minimal fields (kept for the audit schema / selftest).
    baseline_observation_id: str | None = None
    result_observation_id: str | None = None
    performance_delta: dict[str, float] = Field(default_factory=dict)
    quality_before: list["QualityScore"] = Field(default_factory=list)
    quality_after: list["QualityScore"] = Field(default_factory=list)
    tolerance_band: float | None = None
    decision: Decision = Decision.PENDING
    notes: str | None = None

    # Phase 7 lifecycle + evidence.
    created_at: datetime = Field(default_factory=_utcnow)
    action_id: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)
    status: ExperimentDecisionKind | None = None
    request: ExperimentRequest | None = None
    baseline_snapshot: BaselineSnapshot | None = None
    baseline_measurement: ExperimentMeasurement | None = None
    candidate_measurement: ExperimentMeasurement | None = None
    application: ApplicationResult | None = None
    quality: QualityValidationResult | None = None
    comparison: ExperimentComparison | None = None
    decision_detail: ExperimentDecision | None = None
    restoration: RestorationResult | None = None
    events: list[ExperimentEvent] = Field(default_factory=list)
    overhead: ExperimentOverhead | None = None
    errors: list[str] = Field(default_factory=list)
    executed: bool = False
    environment_restored: bool = False
    reproducibility: dict[str, object] = Field(default_factory=dict)
    summary: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "ExperimentResult":
        return cls.model_validate_json(data)


class QualityScore(ContractBase):
    metric: str
    value: float
    higher_is_better: bool = True
    tolerance: float | None = None


ExperimentResult.model_rebuild()
