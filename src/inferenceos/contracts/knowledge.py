"""Persistent knowledge & rollback contracts (Phase 8).

Phase 7 produces authoritative ``ExperimentResult`` objects. Phase 8 turns each one
into an immutable, contextual :class:`OptimizationRecord` (plus a :class:`RollbackRecord`
when the experiment mutated configuration) and stores it durably. Historical evidence
is *conditional*: :func:`evaluate_applicability` classifies how well a stored record
matches the current context rather than treating a past success as universal truth.

Nothing here is executable. Stored action ids are validated against the live registry,
stored numbers are evidence (not authority), and rollback carries prior *values* only.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import (
    ApplicabilityClass,
    DimensionStatus,
    ExperimentDecisionKind,
    KnowledgeEventType,
    KnowledgeHealthStatus,
    KnowledgeWriteStatus,
    OptimizationOutcome,
    QualityStatus,
    RestorationStatus,
    RevalidationTrigger,
)
from inferenceos.contracts.experiment import (
    ExperimentChange,
    RestorationVerification,
    RuntimeConfiguration,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import MetricStats, ProfilingConfig
from inferenceos.contracts.workload import WorkloadProfile

KNOWLEDGE_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Context fingerprinting                                                       #
# --------------------------------------------------------------------------- #
class ContextFingerprint(ContractBase):
    """Compact, reproducible context identity, separated from transient conditions.

    ``identity_hash`` covers only stable identity dimensions (hardware capabilities,
    backend, model, workload, runtime configuration, diagnosis). ``conditions_hash``
    covers transient operating conditions (memory pressure, available RAM, system CPU,
    page faults) so irrelevant telemetry never invalidates identity.
    """

    identity_hash: str
    conditions_hash: str
    hardware_key: str
    cpu_key: str
    gpu_key: str
    os_key: str
    backend_key: str
    model_key: str
    quantization: str = "unknown"
    model_size_bytes: int | None = None
    model_content_hash: str | None = None
    workload_key: str
    runtime_key: str
    diagnosis: str | None = None
    conditions: dict[str, float] = Field(default_factory=dict)


class ApplicabilityContext(ContractBase):
    """The full context used for historical matching, with provenance preserved."""

    fingerprint: ContextFingerprint
    hardware: HardwareProfile
    model: ModelProfile | None = None
    workload: WorkloadProfile | None = None
    backend: ProfilingConfig = Field(default_factory=ProfilingConfig)
    runtime_configuration: RuntimeConfiguration = Field(default_factory=RuntimeConfiguration)
    diagnosed_bottleneck: str | None = None


# --------------------------------------------------------------------------- #
# Persistent records                                                           #
# --------------------------------------------------------------------------- #
class OptimizationRecord(ContractBase):
    """An immutable, contextual record of one completed optimization experiment.

    It is derived explicitly from a Phase 7 ``ExperimentResult`` — it never replaces
    the authoritative experiment contract.
    """

    schema_version: str = KNOWLEDGE_SCHEMA_VERSION
    record_id: str
    experiment_id: str
    created_at: datetime = Field(default_factory=_utcnow)

    candidate_id: str
    action_id: str
    action_registered: bool = True
    parameters: dict[str, object] = Field(default_factory=dict)
    diagnosis_id: str | None = None
    diagnosed_bottleneck: str | None = None

    outcome: OptimizationOutcome
    decision: ExperimentDecisionKind | None = None
    keep: bool = False
    regression: bool = False
    quality_regression: bool = False
    bottleneck_improved: bool = False
    restoration_status: RestorationStatus = RestorationStatus.PENDING
    environment_restored: bool = False

    quality_status: QualityStatus | None = None
    quality_validated: bool = False
    quality_similarity: float | None = None

    context: ApplicabilityContext
    baseline_metrics: dict[str, MetricStats] = Field(default_factory=dict)
    candidate_metrics: dict[str, MetricStats] = Field(default_factory=dict)
    measured_benefit: dict[str, float] = Field(default_factory=dict)
    measured_cost: dict[str, float] = Field(default_factory=dict)
    experiment_overhead_s: float | None = None
    evidence: list[str] = Field(default_factory=list)
    uncertainty: str = ""

    rollback_id: str | None = None
    provenance: str = "phase7-experiment"
    summary: str = ""
    integrity_hash: str | None = None


class RollbackVerification(ContractBase):
    """Integrity/consistency checks a stored rollback must pass before any reuse."""

    verified: bool
    action_registered: bool
    parameters_match: bool
    expected_matches_previous: bool
    context_compatible: bool | None = None
    reasons: list[str] = Field(default_factory=list)


class RollbackRecord(ContractBase):
    """The prior configuration needed to re-apply a known-good state.

    It stores *values*, not commands. Applying it is a later phase's decision; Phase 8
    only records and integrity-checks it.
    """

    schema_version: str = KNOWLEDGE_SCHEMA_VERSION
    rollback_id: str
    experiment_id: str
    candidate_id: str
    action_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    target_scope: str = "target_model_runtime"

    previous_state: RuntimeConfiguration
    resulting_state: RuntimeConfiguration
    expected_configuration: RuntimeConfiguration
    changes: list[ExperimentChange] = Field(default_factory=list)
    restore_parameters: dict[str, object] = Field(default_factory=dict)

    experiment_restored: bool = False
    restoration_status: RestorationStatus = RestorationStatus.PENDING
    context_identity: str | None = None
    verification: RollbackVerification
    integrity_hash: str | None = None
    provenance: str = "phase7-experiment"

    # Phase 8R: identity sufficient to reject an incompatible/stale rollback before
    # execution, plus the Phase 7R restoration evidence (how restoration was checked).
    model_key: str | None = None
    model_content_hash: str | None = None
    backend_key: str | None = None
    hardware_key: str | None = None
    workload_key: str | None = None
    previous_runtime_key: str | None = None
    resulting_runtime_key: str | None = None
    restoration_verification: RestorationVerification | None = None


class KnowledgeEvent(ContractBase):
    """Append-only event; historical records are never silently overwritten."""

    event_id: str
    record_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    event_type: KnowledgeEventType
    detail: str = ""
    data: dict[str, object] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Queries / results                                                            #
# --------------------------------------------------------------------------- #
class KnowledgeQuery(ContractBase):
    action_id: str | None = None
    candidate_id: str | None = None
    model_ref: str | None = None
    workload_id: str | None = None
    diagnosis: str | None = None
    outcome: OptimizationOutcome | None = None
    limit: int = Field(default=50, ge=1)


class DimensionMatch(ContractBase):
    dimension: str
    status: DimensionStatus
    historical: str | None = None
    current: str | None = None
    detail: str = ""


class ApplicabilityResult(ContractBase):
    """Explainable applicability verdict for one historical record vs. current context."""

    classification: ApplicabilityClass
    record_id: str
    experiment_id: str
    action_id: str
    record_outcome: OptimizationOutcome
    score: float | None = None
    components: dict[str, float] = Field(default_factory=dict)
    matches: list[DimensionMatch] = Field(default_factory=list)
    differences: list[DimensionMatch] = Field(default_factory=list)
    unknown: list[DimensionMatch] = Field(default_factory=list)
    conflicts: list[DimensionMatch] = Field(default_factory=list)
    revalidation_triggers: list[RevalidationTrigger] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    rationale: str = ""


class KnowledgeWriteResult(ContractBase):
    """Persistence outcome, distinguishing persisted / duplicate / failure."""

    status: KnowledgeWriteStatus
    experiment_id: str
    record_id: str | None = None
    duplicate: bool = False
    detail: str = ""
    latency_s: float = Field(default=0.0, ge=0.0)


class KnowledgeHealth(ContractBase):
    status: KnowledgeHealthStatus
    schema_version: str
    expected_schema_version: str
    record_count: int = Field(default=0, ge=0)
    rollback_count: int = Field(default=0, ge=0)
    event_count: int = Field(default=0, ge=0)
    corrupted_records: list[str] = Field(default_factory=list)
    db_size_bytes: int | None = None
    detail: str = ""


class StorageStats(ContractBase):
    db_path: str
    db_size_bytes: int | None = None
    optimization_records: int = Field(default=0, ge=0)
    rollback_records: int = Field(default=0, ge=0)
    knowledge_events: int = Field(default=0, ge=0)
    schema_version: str = KNOWLEDGE_SCHEMA_VERSION
    write_seconds: float = Field(default=0.0, ge=0.0)
    read_seconds: float = Field(default=0.0, ge=0.0)
    applicability_seconds: float = Field(default=0.0, ge=0.0)
