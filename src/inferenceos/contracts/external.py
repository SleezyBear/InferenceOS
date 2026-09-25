"""Phase 11 external-process recommendation contracts (staged, minimal).

InferenceOS may **observe** and **recommend** actions concerning external processes
and system resources, but it must not silently control them. These contracts keep the
stages strictly separate:

    observation -> attribution -> recommendation -> provenance -> user decision

Phase 11R is strictly **recommendation-only**. There is no execution stage: a
recommendation — even an accepted one — terminates at a recorded user decision and
never autonomously mutates an external process.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import (
    ApprovalOutcome,
    AttributionEvidenceCategory,
    EvidenceStrength,
    ExternalActionSafetyClass,
    ExternalProcessAccess,
    ExternalRecommendationClass,
    ExternalRecommendationStatus,
    ExternalResource,
    ExternalRunTerminationReason,
    ExternalScope,
)

EXTERNAL_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Process identity / observation                                               #
# --------------------------------------------------------------------------- #
class ExternalProcessIdentity(ContractBase):
    """A contextual process identity — never PID alone.

    PIDs are recycled; ``identity_key`` and ``pid_reuse_guard`` incorporate the
    process start time and executable identity so a stale approval cannot be applied
    to a different process that reuses the same PID. The identity is explicitly marked
    as potentially stale.
    """

    pid: int = Field(ge=0)
    name: str
    executable_path: str | None = None
    executable_identity: str | None = None
    owner: str | None = None
    create_time_epoch: float | None = None
    create_time_iso: str | None = None
    identity_key: str
    pid_reuse_guard: str
    access: ExternalProcessAccess = ExternalProcessAccess.READABLE
    is_target: bool = False
    potentially_stale: bool = False
    notes: list[str] = Field(default_factory=list)


class ExternalProcessObservation(ContractBase):
    """Read-only observed state of one external process (minimal, non-invasive)."""

    observation_id: str
    identity: ExternalProcessIdentity
    captured_at: datetime = Field(default_factory=_utcnow)
    cpu_percent: float | None = None
    cpu_time_s: float | None = None
    rss_bytes: int | None = None
    private_bytes: int | None = None
    page_faults_per_sec: float | None = None
    io_read_bytes: int | None = None
    io_write_bytes: int | None = None
    num_threads: int | None = None
    measurable: list[str] = Field(default_factory=list)
    access: ExternalProcessAccess = ExternalProcessAccess.READABLE
    inaccessible_reason: str | None = None


class ExternalObservationReport(ContractBase):
    """A bounded, read-only snapshot of external processes plus system totals."""

    report_id: str
    captured_at: datetime = Field(default_factory=_utcnow)
    scope: ExternalScope = ExternalScope.EXTERNAL_PROCESS
    target_pid: int | None = None
    own_pid: int | None = None
    logical_cores: int | None = None
    total_ram_bytes: int | None = None
    available_ram_bytes: int | None = None
    memory_pressure: float | None = None
    cpu_percent_total: float | None = None
    processes: list[ExternalProcessObservation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    privacy_notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Attribution                                                                  #
# --------------------------------------------------------------------------- #
class ExternalResourceImpact(ContractBase):
    """Proportional comparison (never an absolute-size ranking)."""

    resource: ExternalResource
    unit: str = "value"
    process_contribution: float | None = None
    target_contribution: float | None = None
    system_total: float | None = None
    share_of_system: float | None = None
    relative_significance: float | None = None
    dominant: bool = False
    within_threshold: bool = False
    explanation: str = ""


class AttributionEvidence(ContractBase):
    category: AttributionEvidenceCategory
    metric: str
    value: float | None = None
    reference: float | None = None
    description: str = ""


class AttributionAssessment(ContractBase):
    """Deterministic estimate of how one external process relates to the target.

    Correlation is never represented as causation: ``causal`` is only true when
    controlled-comparison or convergent pressure + target-degradation evidence exists.
    """

    assessment_id: str
    identity: ExternalProcessIdentity
    resource: ExternalResource
    evidence: list[AttributionEvidence] = Field(default_factory=list)
    categories: list[AttributionEvidenceCategory] = Field(default_factory=list)
    impacts: list[ExternalResourceImpact] = Field(default_factory=list)
    causal: bool = False
    correlation_only: bool = True
    target_degraded: bool = False
    controlled_comparison: bool = False
    historical: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    strength: EvidenceStrength = EvidenceStrength.NONE
    sufficient_for_recommendation: bool = False
    rationale: list[str] = Field(default_factory=list)
    insufficient_reason: str | None = None


class ExternalControlledComparison(ContractBase):
    """A safe observational comparison (process active vs naturally inactive)."""

    comparison_id: str
    identity_key: str
    metric: str
    active_mean: float | None = None
    inactive_mean: float | None = None
    relative_change: float | None = None
    improved_when_inactive: bool = False
    safe: bool = True
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Recommendation / safety / approval / action                                  #
# --------------------------------------------------------------------------- #
class ExternalProvenance(ContractBase):
    """Traceability for an external recommendation (never a free-floating opinion).

    Every recommendation is bound to the runtime/optimization session, the diagnosis
    and the concrete telemetry observation that produced it. A recommendation whose
    provenance no longer matches the current session is stale, not current.
    """

    optimization_session_id: str = ""
    run_id: str = ""
    diagnosis_id: str | None = None
    observation_report_id: str | None = None
    process_observation_id: str | None = None
    attribution_id: str | None = None
    target_pid: int | None = None
    target_identity_key: str | None = None
    context_identity: str = ""
    generator: str = "external.rule_engine.v11r"
    generated_at: datetime = Field(default_factory=_utcnow)


class ExternalUserDecision(ContractBase):
    """A recorded human decision about a recommendation.

    ``accepted`` means the user agrees with the recommendation. It does **not**
    authorize InferenceOS to manipulate the external process. ``autonomous_execution``
    is always false in Phase 11R and exists to make that boundary auditable.
    """

    decision_id: str
    recommendation_id: str
    run_id: str = ""
    decision: ExternalRecommendationStatus = ExternalRecommendationStatus.PRESENTED
    decided_at: datetime = Field(default_factory=_utcnow)
    decided_by: str | None = None
    context_identity: str = ""
    process_identity_key: str | None = None
    autonomous_execution: bool = False
    detail: str = ""


class ExternalRecommendation(ContractBase):
    recommendation_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    classification: ExternalRecommendationClass
    scope: ExternalScope
    resource: ExternalResource
    identity: ExternalProcessIdentity | None = None
    neutral_description: str = ""
    why_it_matters: str = ""
    evidence_summary: list[str] = Field(default_factory=list)
    attribution_id: str | None = None
    proposed_action_id: str | None = None
    proposed_parameters: dict[str, object] = Field(default_factory=dict)
    expected_benefit: str = ""
    expected_benefit_available: bool = False
    expected_cost_s: float = Field(default=0.0, ge=0.0)
    expected_disruption_s: float = Field(default=0.0, ge=0.0)
    requires_approval: bool = True
    requires_user_approval: bool = True
    supported: bool = False
    reversible: bool = True
    verification_possible: bool = False
    safety_class: ExternalActionSafetyClass = ExternalActionSafetyClass.RECOMMENDATION_ONLY
    context_identity: str = ""
    expires_at: datetime | None = None
    limitations: list[str] = Field(default_factory=list)

    # Phase 11R structured context / provenance / lifecycle.
    run_id: str = ""
    session_id: str = ""
    target_identity: ExternalProcessIdentity | None = None
    bottleneck: str | None = None
    observed_metrics: dict[str, float] = Field(default_factory=dict)
    observed_at: datetime | None = None
    rationale: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: str = ""
    disruption_risk: str = ""
    provenance: ExternalProvenance | None = None
    status: ExternalRecommendationStatus = ExternalRecommendationStatus.GENERATED
    stale_reason: str | None = None


class ExternalActionSafety(ContractBase):
    action_id: str
    recommendation_id: str
    identity_key: str | None = None
    allowed: bool = False
    requires_approval: bool = True
    elevation_required: bool = False
    implemented: bool = False
    phase11_allowed: bool = False
    reversible: bool = False
    verification_supported: bool = False
    safety_class: ExternalActionSafetyClass = ExternalActionSafetyClass.RECOMMENDATION_ONLY
    reasons: list[str] = Field(default_factory=list)


class ExternalApproval(ContractBase):
    approval_id: str
    recommendation_id: str
    action_id: str | None = None
    outcome: ApprovalOutcome = ApprovalOutcome.PENDING
    process_identity_key: str | None = None
    context_identity: str = ""
    requested_at: datetime = Field(default_factory=_utcnow)
    decided_at: datetime | None = None
    expires_at: datetime | None = None
    decided_by: str | None = None
    detail: str = ""


class ExternalActionCheckpoint(ContractBase):
    """Capture only what the supported action can actually restore."""

    checkpoint_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    identity: ExternalProcessIdentity
    action_id: str
    prior_state: dict[str, object] = Field(default_factory=dict)
    configurable_fields: list[str] = Field(default_factory=list)
    reversible: bool = True
    restore_supported: bool = True
    target_workload_state: dict[str, object] = Field(default_factory=dict)
    approval_id: str | None = None
    verified: bool = False
    integrity: str | None = None
    limitations: list[str] = Field(default_factory=list)


class ExternalActionResult(ContractBase):
    action_result_id: str
    action_id: str
    recommendation_id: str | None = None
    approval_id: str | None = None
    checkpoint_id: str | None = None
    attempted: bool = False
    executed: bool = False
    identity_before: ExternalProcessIdentity | None = None
    identity_after: ExternalProcessIdentity | None = None
    changes: dict[str, object] = Field(default_factory=dict)
    reason: str = ""
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    duration_s: float = Field(default=0.0, ge=0.0)


class ExternalVerification(ContractBase):
    verification_id: str
    action_result_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    external_state_changed: bool = False
    external_state_restored: bool = False
    target_state_observed: bool = False
    target_healthy: bool = True
    materially_helped: bool | None = None
    verified: bool = False
    failure_reason: str | None = None
    measurement: dict[str, float] = Field(default_factory=dict)
    restoration_attempted: bool = False
    restoration_verified: bool = False
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Knowledge / economics / policy / run                                         #
# --------------------------------------------------------------------------- #
class ExternalKnowledgeRecord(ContractBase):
    record_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    recommendation_id: str
    identity_key: str
    executable_identity: str | None = None
    resource: ExternalResource
    classification: ExternalRecommendationClass
    action_id: str | None = None
    outcome: str = ""
    approved: bool = False
    executed: bool = False
    verified: bool = False
    restored: bool = False
    context_identity: str = ""
    evidence_summary: list[str] = Field(default_factory=list)
    detail: str = ""
    integrity_hash: str | None = None


class ExternalEconomics(ContractBase):
    expected_benefit_score: float = Field(default=0.0, ge=0.0)
    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_score: float = Field(default=0.0, ge=0.0)
    cost_score: float = Field(default=0.0, ge=0.0)
    net_value: float = 0.0
    beneficial: bool = False
    requires_approval: bool = True
    costs: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)


class ExternalCostModel(ContractBase):
    observation_cost_s: float = Field(default=0.1, ge=0.0)
    attribution_cost_s: float = Field(default=0.05, ge=0.0)
    knowledge_query_cost_s: float = Field(default=0.01, ge=0.0)
    checkpoint_cost_s: float = Field(default=0.05, ge=0.0)
    approval_cost_s: float = Field(default=0.0, ge=0.0)
    action_cost_s: float = Field(default=1.0, ge=0.0)
    verification_cost_s: float = Field(default=2.0, ge=0.0)
    disruption_cost_s: float = Field(default=5.0, ge=0.0)
    risk_penalty_s: float = Field(default=10.0, ge=0.0)
    benefit_scale_s: float = Field(default=120.0, gt=0.0)
    weight_benefit: float = Field(default=1.0, ge=0.0)
    weight_cost: float = Field(default=1.0, ge=0.0)
    min_net_value: float = Field(default=0.0)

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()


class ExternalBudget(ContractBase):
    max_duration_s: float = Field(default=300.0, gt=0.0)
    max_processes: int = Field(default=40, ge=1)
    max_recommendations: int = Field(default=4, ge=1)
    max_actions: int = Field(default=1, ge=0)
    max_cumulative_action_s: float = Field(default=60.0, gt=0.0)

    observations_used: int = Field(default=0, ge=0)
    recommendations_used: int = Field(default=0, ge=0)
    actions_used: int = Field(default=0, ge=0)
    cumulative_action_s: float = Field(default=0.0, ge=0.0)

    def exhaustion(self, *, elapsed_s: float) -> ExternalRunTerminationReason | None:
        if elapsed_s >= self.max_duration_s:
            return ExternalRunTerminationReason.DURATION_REACHED
        if self.recommendations_used >= self.max_recommendations:
            return ExternalRunTerminationReason.RECOMMENDATION_LIMIT_REACHED
        return None

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()


class ExternalPolicy(ContractBase):
    """Conservative Phase 11 policy; default is observe -> recommend -> ask user."""

    budget: ExternalBudget = Field(default_factory=ExternalBudget)
    observation_interval_s: float = Field(default=1.0, gt=0.0)
    scope: ExternalScope = ExternalScope.EXTERNAL_PROCESS
    memory_pressure_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    memory_dominance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    cpu_dominance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_relative_significance: float = Field(default=0.5, ge=0.0, le=1.0)
    min_process_rss_bytes: int = Field(default=256 * 1024 * 1024, ge=0)
    min_process_cpu_percent: float = Field(default=25.0, ge=0.0)
    require_target_degradation_for_recommendation: bool = True
    require_approval_for_actions: bool = True
    restore_after_action: bool = True
    elevation_never_automatic: bool = True
    allow_unsupported_actions: bool = False
    cooldown_s: float = Field(default=300.0, ge=0.0)
    approval_timeout_s: float = Field(default=60.0, gt=0.0)
    max_alerts_per_identity: int = Field(default=2, ge=1)
    cost_model: ExternalCostModel = Field(default_factory=ExternalCostModel)

    def as_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget.as_dict(),
            "observation_interval_s": self.observation_interval_s,
            "scope": self.scope.value,
            "memory_pressure_threshold": self.memory_pressure_threshold,
            "memory_dominance_threshold": self.memory_dominance_threshold,
            "cpu_dominance_threshold": self.cpu_dominance_threshold,
            "min_relative_significance": self.min_relative_significance,
            "min_process_rss_bytes": self.min_process_rss_bytes,
            "min_process_cpu_percent": self.min_process_cpu_percent,
            "require_target_degradation_for_recommendation": (
                self.require_target_degradation_for_recommendation
            ),
            "require_approval_for_actions": self.require_approval_for_actions,
            "restore_after_action": self.restore_after_action,
            "elevation_never_automatic": self.elevation_never_automatic,
            "allow_unsupported_actions": self.allow_unsupported_actions,
            "cooldown_s": self.cooldown_s,
            "approval_timeout_s": self.approval_timeout_s,
            "max_alerts_per_identity": self.max_alerts_per_identity,
            "cost_model": self.cost_model.as_dict(),
        }


class ExternalCooldown(ContractBase):
    """Anti-nagging cooldown keyed by external process identity."""

    cooldown_id: str
    key: str
    identity_key: str
    reason: str
    trigger: str
    started_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime


class ExternalTermination(ContractBase):
    reason: ExternalRunTerminationReason
    detail: str = ""
    clean: bool = True
    safety: bool = False


class ExternalOverhead(ContractBase):
    observation_s: float = Field(default=0.0, ge=0.0)
    attribution_s: float = Field(default=0.0, ge=0.0)
    knowledge_query_s: float = Field(default=0.0, ge=0.0)
    recommendation_s: float = Field(default=0.0, ge=0.0)
    approval_s: float = Field(default=0.0, ge=0.0)
    checkpoint_s: float = Field(default=0.0, ge=0.0)
    action_s: float = Field(default=0.0, ge=0.0)
    verification_s: float = Field(default=0.0, ge=0.0)
    total_s: float = Field(default=0.0, ge=0.0)
    notes: list[str] = Field(default_factory=list)


class ExternalRecommendationRun(ContractBase):
    """The complete machine-readable audit record of an external recommendation run."""

    schema_version: str = EXTERNAL_SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None

    scope: ExternalScope = ExternalScope.EXTERNAL_PROCESS
    policy: ExternalPolicy = Field(default_factory=ExternalPolicy)
    budget: ExternalBudget = Field(default_factory=ExternalBudget)
    target_pid: int | None = None
    target_identity: ExternalProcessIdentity | None = None
    session_id: str = ""
    diagnosis_id: str | None = None
    diagnosed_bottleneck: str | None = None
    no_recommendation_reason: str | None = None

    observation_reports: list[ExternalObservationReport] = Field(default_factory=list)
    assessments: list[AttributionAssessment] = Field(default_factory=list)
    controlled_comparisons: list[ExternalControlledComparison] = Field(default_factory=list)
    recommendations: list[ExternalRecommendation] = Field(default_factory=list)
    user_decisions: list[ExternalUserDecision] = Field(default_factory=list)
    safety_checks: list[ExternalActionSafety] = Field(default_factory=list)
    approvals: list[ExternalApproval] = Field(default_factory=list)
    checkpoints: list[ExternalActionCheckpoint] = Field(default_factory=list)
    action_results: list[ExternalActionResult] = Field(default_factory=list)
    verifications: list[ExternalVerification] = Field(default_factory=list)
    knowledge_queries: list[ExternalKnowledgeRecord] = Field(default_factory=list)
    knowledge_writes: list[ExternalKnowledgeRecord] = Field(default_factory=list)
    cooldowns: list[ExternalCooldown] = Field(default_factory=list)

    termination: ExternalTermination
    overhead: ExternalOverhead = Field(default_factory=ExternalOverhead)
    errors: list[str] = Field(default_factory=list)
    summary: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "ExternalRecommendationRun":
        return cls.model_validate_json(data)


class ExternalAdvisor(Protocol):
    """Runtime integration hook (implemented by the Phase 11 engine).

    The runtime monitor passes the target identity and whether an internal
    optimization already exists; the advisor returns a structured external
    recommendation run. It must never execute an external action without approval.
    """

    def advise(
        self,
        *,
        target_pid: int | None,
        target_identity: ExternalProcessIdentity | None,
        target_degraded: bool,
        target_metrics: dict[str, float],
        context_identity: str,
        diagnosed_bottleneck: str | None = None,
        approval_available: bool = True,
    ) -> ExternalRecommendationRun:
        ...
