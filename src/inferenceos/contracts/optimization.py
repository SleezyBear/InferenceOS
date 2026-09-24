"""Phase 6 optimization contracts: candidate actions, candidates and selection.

These contracts describe *what could be done* and *why a candidate is or is not
worth considering* — never an executed command. A candidate references a registered
:class:`CandidateAction` by id; it never carries a shell/PowerShell command, an
executable invocation, a registry edit, or any other free-form executable.

The plan produced here is consumed by Phase 7, which owns experiment execution.
Phase 6 mutates nothing: it inspects context and ranks plausible interventions.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    ApprovalRequirement,
    CandidateEligibility,
    CandidatePlanOutcome,
    EvidenceStrength,
    ExternalActionSafetyClass,
    OptimizationScope,
    OptimizationSurface,
    ParameterType,
    PreconditionKind,
    PrivilegeRequirement,
    Reversibility,
    RiskLevel,
    WorkloadKind,
)
from inferenceos.contracts.hardware import HardwareProfile

OPTIMIZATION_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Registered action definition                                                 #
# --------------------------------------------------------------------------- #
class ParameterSpec(ContractBase):
    """Bounded parameter schema for a registered action.

    Actions may only bind parameters declared here, within these bounds; the
    optimizer never invents a parameter or a value.
    """

    name: str
    type: ParameterType
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: list[str] = Field(default_factory=list)
    unit: str | None = None


class CandidateAction(ContractBase):
    """A registered, bounded optimization intervention definition.

    Metadata is deliberately explicit so a later executor (Phase 7) can consume the
    action without the optimizer constructing an arbitrary command.
    """

    action_id: str
    name: str
    description: str
    surface: OptimizationSurface
    subsystem: str
    supported_bottlenecks: list[BottleneckKind]
    causal_rationale: str

    parameters: list[ParameterSpec] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    expected_metrics: list[str] = Field(default_factory=list)
    expected_direction: str = "decrease"

    reversibility: Reversibility
    risk_level: RiskLevel
    scope: OptimizationScope
    privilege: PrivilegeRequirement
    requires_human_approval: bool = False

    execution_cost_estimate: float = Field(default=1.0, ge=0.0)
    optimizer_overhead_estimate: float = Field(default=0.01, ge=0.0)
    experimentation_supported: bool = True
    enabled: bool = True
    version: str = "1.0"

    # Phase 11 external-action metadata (additive; safe defaults keep prior phases
    # recommendation-only and non-executable from the external layer).
    external_safety_class: ExternalActionSafetyClass = (
        ExternalActionSafetyClass.RECOMMENDATION_ONLY
    )
    implemented: bool = False
    phase11_allowed: bool = False
    verification_supported: bool = False

    # Minimum diagnosis strength required before this action may be considered.
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    min_evidence_strength: EvidenceStrength = EvidenceStrength.NONE


# --------------------------------------------------------------------------- #
# Concrete candidate                                                           #
# --------------------------------------------------------------------------- #
class CandidatePrecondition(ContractBase):
    """One explicitly evaluated precondition of a concrete candidate."""

    check_id: str
    kind: PreconditionKind
    description: str
    satisfied: bool | None
    detail: str = ""
    required: bool = True


class BenefitEstimate(ContractBase):
    """Uncertainty-aware expected benefit — never a fabricated measurement.

    ``known`` is False when the system cannot responsibly estimate a magnitude; a
    later phase measures the real effect. ``magnitude_*`` are estimates, explicitly
    distinct from measured results.
    """

    metric: str
    direction: str = "decrease"
    unit: str = "value"
    known: bool = False
    magnitude_low: float | None = None
    magnitude_high: float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_basis: list[str] = Field(default_factory=list)
    notes: str = ""


class CostEstimate(ContractBase):
    """First-class cost: intervention, reconfiguration, reload, experiment, optimizer."""

    execution_cost_s: float = Field(default=0.0, ge=0.0)
    reconfiguration_cost_s: float = Field(default=0.0, ge=0.0)
    model_reload_cost_s: float = Field(default=0.0, ge=0.0)
    experiment_cost_s: float = Field(default=0.0, ge=0.0)
    reasoning_cost_s: float = Field(default=0.0, ge=0.0)
    optimizer_overhead_s: float = Field(default=0.0, ge=0.0)
    total_cost_s: float = Field(default=0.0, ge=0.0)
    notes: list[str] = Field(default_factory=list)


class RiskAssessment(ContractBase):
    risk_level: RiskLevel
    reversibility: Reversibility
    scope: OptimizationScope
    privilege: PrivilegeRequirement
    failure_consequences: str = ""
    rollback_available: bool = False
    requires_approval: bool = False
    gating_reasons: list[str] = Field(default_factory=list)


class RankingFactors(ContractBase):
    """Decomposed ranking: every contributing factor is preserved and explained."""

    benefit_score: float = Field(ge=0.0, le=1.0)
    uncertainty_score: float = Field(ge=0.0, le=1.0)
    cost_score: float = Field(ge=0.0, le=1.0)
    overhead_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    reversibility_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)
    causal_strength_score: float = Field(ge=0.0, le=1.0)
    scope_score: float = Field(ge=0.0, le=1.0)
    approval_score: float = Field(ge=0.0, le=1.0)
    precondition_score: float = Field(ge=0.0, le=1.0)
    expected_value: float
    composite_score: float
    weights: dict[str, float] = Field(default_factory=dict)
    explanation: list[str] = Field(default_factory=list)


class OptimizationCandidate(ContractBase):
    """A concrete candidate generated for the current real context.

    It carries the action identity, concrete bound parameters, the source diagnosis
    and evidence references, hardware/model/workload context, preconditions,
    benefit/cost/risk, approval and eligibility. It is inert: nothing here is
    executed.
    """

    candidate_id: str
    action_id: str
    action_name: str
    surface: OptimizationSurface
    subsystem: str
    parameters: dict[str, object] = Field(default_factory=dict)

    source_diagnosis_id: str
    source_bottleneck_id: str | None = None
    diagnosis_kind: BottleneckKind
    diagnosis_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    diagnosis_strength: EvidenceStrength = EvidenceStrength.NONE
    causal_rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)

    hardware: HardwareProfile
    model_ref: str | None = None
    model_architecture: str | None = None
    workload_id: str | None = None
    workload_kind: WorkloadKind = WorkloadKind.UNKNOWN
    observed_metrics: dict[str, float] = Field(default_factory=dict)

    preconditions: list[CandidatePrecondition] = Field(default_factory=list)
    benefit: BenefitEstimate
    cost: CostEstimate
    risk: RiskAssessment
    approval: ApprovalRequirement = ApprovalRequirement.NOT_REQUIRED
    eligibility: CandidateEligibility = CandidateEligibility.ELIGIBLE
    rejection_reason: str | None = None
    ranking: RankingFactors | None = None
    generated_by: str = "deterministic"
    executable: bool = False


# --------------------------------------------------------------------------- #
# Reasoning-assisted proposal (deterministically validated)                    #
# --------------------------------------------------------------------------- #
class ReasoningProposalOutcome(ContractBase):
    """Audit record of an optional reasoning proposal pass.

    The model may only propose registered action ids with schema-valid parameters
    and existing evidence ids; anything else is rejected here. The deterministic
    engine remains authoritative and provides the fallback.
    """

    used: bool = False
    provider: str | None = None
    model: str | None = None
    status: str | None = None
    validation_ok: bool = False
    proposed_action_ids: list[str] = Field(default_factory=list)
    accepted_action_ids: list[str] = Field(default_factory=list)
    rejected_action_ids: list[str] = Field(default_factory=list)
    rejections: list[str] = Field(default_factory=list)
    failure: str | None = None
    latency_s: float | None = None
    raw_output: str | None = None


# --------------------------------------------------------------------------- #
# Phase-level selection / plan                                                 #
# --------------------------------------------------------------------------- #
class OptimizerOverhead(ContractBase):
    """Measured cost of running the optimizer itself (never assumed free)."""

    generation_latency_s: float = Field(default=0.0, ge=0.0)
    reasoning_latency_s: float | None = None
    actions_considered: int = Field(default=0, ge=0)
    candidates_generated: int = Field(default=0, ge=0)
    eligible: int = Field(default=0, ge=0)
    gated: int = Field(default=0, ge=0)
    rejected: int = Field(default=0, ge=0)
    insufficient_evidence: int = Field(default=0, ge=0)
    peak_optimizer_rss_bytes: int | None = None
    notes: list[str] = Field(default_factory=list)


class CandidatePlan(ContractBase):
    """Auditable Phase 6 result; carries candidates but executes nothing."""

    schema_version: str = OPTIMIZATION_SCHEMA_VERSION
    plan_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    source_diagnosis_id: str

    source_diagnosis: BottleneckDiagnosis
    source_outcome_id: str | None = None
    effective_bottleneck: BottleneckKind | None = None
    effective_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    effective_basis: str = "deterministic"

    outcome: CandidatePlanOutcome
    terminal_reason: str

    candidates: list[OptimizationCandidate] = Field(default_factory=list)
    eligible_ids: list[str] = Field(default_factory=list)
    gated_ids: list[str] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)
    insufficient_evidence_ids: list[str] = Field(default_factory=list)
    ranking: list[str] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)

    reasoning: ReasoningProposalOutcome | None = None
    overhead: OptimizerOverhead = Field(default_factory=OptimizerOverhead)
    policy: dict[str, object] = Field(default_factory=dict)
    reproducibility: dict[str, object] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)

    # Phase boundary proof: candidate generation is non-mutating.
    environment_mutated: bool = False
    executed: bool = False

    summary: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "CandidatePlan":
        return cls.model_validate_json(data)
