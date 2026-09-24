"""Phase 9R fresh-run provenance, deployment state, and run contracts.

A fresh autonomous run must be able to bind every run-critical artifact to its own
run identity, prove that it established its own hardware/model/baseline/diagnosis
state, and expose the final retained runtime state as a structured deployment
representation. These contracts are additive to the Phase 9 loop contracts.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import MeasurementMode, StrEnum
from inferenceos.contracts.experiment import RuntimeConfiguration

FRESH_RUN_SCHEMA_VERSION = "9r.1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactKind(StrEnum):
    CHARACTERIZATION = "characterization"
    MODEL_INTROSPECTION = "model_introspection"
    BASELINE = "baseline"
    DIAGNOSIS = "diagnosis"
    REASONING = "reasoning"
    CANDIDATE_PLAN = "candidate_plan"
    EXPERIMENT = "experiment"
    QUALITY = "quality"
    RESTORATION = "restoration"
    ROLLBACK = "rollback"
    KNOWLEDGE = "knowledge"


class DeploymentStatus(StrEnum):
    NO_CHANGE = "no_change"
    RETAINED = "retained"
    CANDIDATE_REJECTED = "candidate_rejected"
    NO_CANDIDATE = "no_candidate"
    SAFETY_STOP = "safety_stop"


class RunArtifactRef(ContractBase):
    """A provenance reference to one run-critical artifact."""

    artifact_id: str
    kind: ArtifactKind
    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    content_hash: str | None = None
    path: str | None = None
    # Whether this artifact was intentionally reused rather than freshly produced.
    reused: bool = False
    reuse_basis: str | None = None
    notes: list[str] = Field(default_factory=list)


class FreshRunProvenance(ContractBase):
    """Binds every run-critical artifact to the current run identity."""

    schema_version: str = FRESH_RUN_SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    hardware_hash: str | None = None
    model_hash: str | None = None
    backend_key: str | None = None
    baseline_artifact_id: str | None = None
    diagnosis_id: str | None = None
    candidate_plan_id: str | None = None
    experiment_ids: list[str] = Field(default_factory=list)
    knowledge_record_ids: list[str] = Field(default_factory=list)
    reasoning_outcome_ids: list[str] = Field(default_factory=list)
    artifacts: list[RunArtifactRef] = Field(default_factory=list)
    stale_artifacts: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def all_bound_to_run(self) -> bool:
        return all(artifact.run_id == self.run_id for artifact in self.artifacts)


class DeploymentState(ContractBase):
    """Structured representation of the final optimized state for other components.

    It distinguishes the *retained/recommended* state from merely proposed candidate
    state, and binds it to model/backend/runtime identity and provenance.
    """

    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    status: DeploymentStatus
    model_ref: str
    model_path: str
    model_content_hash: str | None = None
    backend_name: str | None = None
    backend_version: str | None = None
    baseline_configuration: RuntimeConfiguration
    final_configuration: RuntimeConfiguration
    retained: bool = False
    retained_candidate_ids: list[str] = Field(default_factory=list)
    source_experiment_id: str | None = None
    provenance_hash: str | None = None
    notes: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class FreshRunPolicy(ContractBase):
    """Explicit policy for how a fresh run establishes and controls its own state."""

    workload_ids: list[str] = Field(default_factory=lambda: ["short-short"])
    modes: list[MeasurementMode] = Field(default_factory=lambda: [MeasurementMode.WARM])
    baseline_repetitions: int = Field(default=2, ge=1)
    context_override: int | None = None
    require_characterization: bool = True
    require_introspection: bool = True
    require_fresh_baseline: bool = True
    controlled_optimization_allowed: bool = False
    controlled_diagnosis_label: str = "controlled"


class FreshRunResult(ContractBase):
    """Top-level result of one fresh autonomous run with full provenance."""

    schema_version: str = FRESH_RUN_SCHEMA_VERSION
    run_id: str
    policy: FreshRunPolicy
    hardware_summary: dict[str, object] = Field(default_factory=dict)
    model_summary: dict[str, object] = Field(default_factory=dict)
    baseline_report_path: str | None = None
    baseline_artifact_id: str | None = None
    diagnosis_id: str | None = None
    diagnosis_controlled: bool = False
    provenance: FreshRunProvenance
    deployment: DeploymentState
    characterization_s: float = Field(default=0.0, ge=0.0)
    introspection_s: float = Field(default=0.0, ge=0.0)
    baseline_s: float = Field(default=0.0, ge=0.0)
    stale_artifact_checks: list[str] = Field(default_factory=list)
    summary: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "FreshRunResult":
        return cls.model_validate_json(data)
