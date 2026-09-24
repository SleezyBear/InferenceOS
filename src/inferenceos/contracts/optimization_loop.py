"""Phase 9 pre-deployment autonomous optimization-loop contracts.

The loop orchestrates the existing phases (observe -> diagnose -> deep-diagnose ->
generate -> knowledge-query -> select -> checkpoint -> experiment -> validate ->
decide -> keep/revert -> persist -> continue/terminate). It is an orchestrator, not a
second experiment implementation: it reuses the Phase 4 detector, the Phase 5
reasoning lifecycle, the Phase 6 generator, the Phase 7 engine, and the Phase 8
knowledge store.

Everything here is inert data. Nothing in these contracts executes a command; the
only mutation surface remains the registered Phase 7 runtime-configuration executor.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import (
    CandidatePlanOutcome,
    CheckpointStatus,
    Decision,
    EvidenceStrength,
    ExperimentDecisionKind,
    KnowledgeAvailability,
    KnowledgeHealthStatus,
    KnowledgeSelectionEffect,
    KnowledgeWriteStatus,
    LoopDecisionKind,
    LoopIterationOutcome,
    LoopState,
    LoopTerminationReason,
    RevalidationTrigger,
    RestorationStatus,
)
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.knowledge import KnowledgeHealth, StorageStats
from inferenceos.contracts.optimization import CandidatePlan

LOOP_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Budget / policy                                                              #
# --------------------------------------------------------------------------- #
class OptimizationBudget(ContractBase):
    """Explicit, configurable resource budget for an autonomous run.

    The ``*_used`` fields are the consumed portion; the loop updates them by
    ``model_copy`` so the final record shows exactly what was spent.
    """

    max_iterations: int = Field(default=4, ge=1)
    max_experiments: int = Field(default=4, ge=1)
    max_wall_clock_s: float = Field(default=1800.0, gt=0.0)
    max_optimizer_overhead_s: float = Field(default=300.0, gt=0.0)
    max_deep_diagnoses: int = Field(default=2, ge=0)

    iterations_used: int = Field(default=0, ge=0)
    experiments_used: int = Field(default=0, ge=0)
    deep_diagnoses_used: int = Field(default=0, ge=0)
    wall_clock_used_s: float = Field(default=0.0, ge=0.0)
    optimizer_overhead_used_s: float = Field(default=0.0, ge=0.0)

    def exhaustion(
        self, *, wall_clock_s: float, optimizer_overhead_s: float
    ) -> LoopTerminationReason | None:
        """Return the first exhausted budget, or ``None`` when work may continue.

        Checked in a fixed, deterministic order so termination is reproducible.
        """
        if wall_clock_s >= self.max_wall_clock_s:
            return LoopTerminationReason.TIME_BUDGET_EXHAUSTED
        if optimizer_overhead_s >= self.max_optimizer_overhead_s:
            return LoopTerminationReason.OVERHEAD_BUDGET_EXHAUSTED
        if self.iterations_used >= self.max_iterations:
            return LoopTerminationReason.OPTIMIZATION_BUDGET_EXHAUSTED
        if self.experiments_used >= self.max_experiments:
            return LoopTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED
        return None

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()


class LoopPolicy(ContractBase):
    """Inspectable Phase 9 policy (budget ceilings + safety/decision gates)."""

    budget: OptimizationBudget = Field(default_factory=OptimizationBudget)
    allow_deep_diagnosis: bool = True
    require_reasoning_when_escalated: bool = False
    require_knowledge: bool = False
    knowledge_lookup_limit: int = Field(default=20, ge=1)
    skip_directly_negative: bool = True
    allow_revalidation: bool = True
    max_candidates_per_iteration: int = Field(default=3, ge=1)
    autonomous_only: bool = True
    stop_on_quality_regression: bool = True
    stop_on_restoration_failure: bool = True
    reobserve_after_retention: bool = True
    knowledge_positive_bonus: float = Field(default=0.05, ge=0.0)
    knowledge_negative_penalty: float = Field(default=0.15, ge=0.0)
    min_experiment_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget.as_dict(),
            "allow_deep_diagnosis": self.allow_deep_diagnosis,
            "require_reasoning_when_escalated": self.require_reasoning_when_escalated,
            "require_knowledge": self.require_knowledge,
            "knowledge_lookup_limit": self.knowledge_lookup_limit,
            "skip_directly_negative": self.skip_directly_negative,
            "allow_revalidation": self.allow_revalidation,
            "max_candidates_per_iteration": self.max_candidates_per_iteration,
            "autonomous_only": self.autonomous_only,
            "stop_on_quality_regression": self.stop_on_quality_regression,
            "stop_on_restoration_failure": self.stop_on_restoration_failure,
            "reobserve_after_retention": self.reobserve_after_retention,
            "knowledge_positive_bonus": self.knowledge_positive_bonus,
            "knowledge_negative_penalty": self.knowledge_negative_penalty,
            "min_experiment_confidence": self.min_experiment_confidence,
        }


# --------------------------------------------------------------------------- #
# Checkpoint                                                                   #
# --------------------------------------------------------------------------- #
class SafetyCheckpoint(ContractBase):
    """The protected optimization-controlled state captured before any mutation.

    Phase 9 does not claim to roll back the whole operating system. This checkpoint
    covers exactly the target model's runtime configuration that InferenceOS mutates;
    the Phase 7 executor remains authoritative for restoration.
    """

    checkpoint_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    status: CheckpointStatus = CheckpointStatus.CAPTURED
    runtime_configuration: RuntimeConfiguration
    model_ref: str | None = None
    model_path: str | None = None
    hardware_hostname: str | None = None
    workload_id: str | None = None
    diagnosis_id: str | None = None
    context_identity: str | None = None
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    target_scope: str = "target_model_runtime"
    restorable: bool = True
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Audit events                                                                 #
# --------------------------------------------------------------------------- #
class LoopStageEvent(ContractBase):
    """One explicit, timestamped state transition (or notable action) of the loop."""

    stage: LoopState
    timestamp: datetime = Field(default_factory=_utcnow)
    iteration: int = Field(default=0, ge=0)
    detail: str = ""
    data: dict[str, object] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Knowledge-aware selection                                                     #
# --------------------------------------------------------------------------- #
class KnowledgeEvidence(ContractBase):
    """Explainable historical evidence consulted for one candidate."""

    candidate_id: str
    action_id: str
    effect: KnowledgeSelectionEffect
    availability: KnowledgeAvailability = KnowledgeAvailability.OK
    records_considered: int = Field(default=0, ge=0)
    classifications: dict[str, int] = Field(default_factory=dict)
    outcomes: dict[str, int] = Field(default_factory=dict)
    applicable_positive: int = Field(default=0, ge=0)
    applicable_negative: int = Field(default=0, ge=0)
    requires_revalidation: bool = False
    revalidation_triggers: list[RevalidationTrigger] = Field(default_factory=list)
    best_score: float | None = None
    record_ids: list[str] = Field(default_factory=list)
    detail: str = ""


class LoopDecision(ContractBase):
    """The orchestrator's per-iteration decision (proceed / reject / terminate)."""

    kind: LoopDecisionKind
    reason: str
    selected_candidate_id: str | None = None
    action_id: str | None = None
    params: dict[str, object] = Field(default_factory=dict)
    ranked_candidate_ids: list[str] = Field(default_factory=list)
    termination_reason: LoopTerminationReason | None = None
    knowledge: list[KnowledgeEvidence] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Iteration / run                                                              #
# --------------------------------------------------------------------------- #
class LoopIteration(ContractBase):
    """A complete, auditable optimization iteration."""

    iteration: int = Field(ge=1)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    state_events: list[LoopStageEvent] = Field(default_factory=list)

    diagnosis_id: str | None = None
    diagnosis_kind: str | None = None
    diagnosis_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    diagnosis_strength: EvidenceStrength = EvidenceStrength.NONE
    deep_diagnosis_used: bool = False
    deep_diagnosis_reason: str | None = None
    deep_diagnosis_outcome_id: str | None = None

    plan_id: str | None = None
    plan_outcome: CandidatePlanOutcome | None = None
    plan_terminal_reason: str = ""
    plan: CandidatePlan | None = None
    candidates_considered: int = Field(default=0, ge=0)
    eligible_candidate_ids: list[str] = Field(default_factory=list)
    rejected_candidates: dict[str, str] = Field(default_factory=dict)

    decision: LoopDecision | None = None
    selected_candidate_id: str | None = None
    selected_action_id: str | None = None
    selected_parameters: dict[str, object] = Field(default_factory=dict)
    knowledge: list[KnowledgeEvidence] = Field(default_factory=list)

    checkpoint: SafetyCheckpoint | None = None
    experiment_id: str | None = None
    experiment_status: ExperimentDecisionKind | None = None
    experiment_decision: Decision = Decision.PENDING
    retained: bool = False
    restoration_status: RestorationStatus = RestorationStatus.NOT_REQUIRED
    restoration_verified: bool = False
    persist_status: KnowledgeWriteStatus | None = None

    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    candidate_metrics: dict[str, float] = Field(default_factory=dict)
    outcome: LoopIterationOutcome = LoopIterationOutcome.NO_CANDIDATE
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    overhead_s: float = Field(default=0.0, ge=0.0)


class LoopOverhead(ContractBase):
    """Measured optimizer overhead, kept separate from experiment measurement time."""

    observation_s: float = Field(default=0.0, ge=0.0)
    diagnosis_s: float = Field(default=0.0, ge=0.0)
    deep_reasoning_s: float = Field(default=0.0, ge=0.0)
    candidate_generation_s: float = Field(default=0.0, ge=0.0)
    knowledge_query_s: float = Field(default=0.0, ge=0.0)
    selection_s: float = Field(default=0.0, ge=0.0)
    checkpoint_s: float = Field(default=0.0, ge=0.0)
    experiment_s: float = Field(default=0.0, ge=0.0)
    persistence_s: float = Field(default=0.0, ge=0.0)
    orchestration_s: float = Field(default=0.0, ge=0.0)
    optimizer_overhead_s: float = Field(default=0.0, ge=0.0)
    total_s: float = Field(default=0.0, ge=0.0)
    notes: list[str] = Field(default_factory=list)


class LoopTermination(ContractBase):
    """Explicit terminal state of a run (never hidden behind a generic exception)."""

    reason: LoopTerminationReason
    detail: str
    clean: bool = True
    safety: bool = False
    at_iteration: int = Field(default=0, ge=0)


class LoopContext(ContractBase):
    """The observed initial context of a run (identity preserved for audit)."""

    report_source: str | None = None
    hardware_hostname: str | None = None
    model_ref: str | None = None
    model_architecture: str | None = None
    workload_id: str | None = None
    backend_name: str | None = None
    context_identity: str | None = None
    conditions: dict[str, float] = Field(default_factory=dict)


class OptimizationRun(ContractBase):
    """The complete machine-readable audit record of one autonomous run."""

    schema_version: str = LOOP_SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None

    context: LoopContext = Field(default_factory=LoopContext)
    initial_configuration: RuntimeConfiguration
    final_configuration: RuntimeConfiguration
    policy: LoopPolicy = Field(default_factory=LoopPolicy)
    budget: OptimizationBudget = Field(default_factory=OptimizationBudget)

    iterations: list[LoopIteration] = Field(default_factory=list)
    state_events: list[LoopStageEvent] = Field(default_factory=list)
    termination: LoopTermination
    retained_candidate_ids: list[str] = Field(default_factory=list)
    changed: bool = False

    knowledge_health: KnowledgeHealthStatus = KnowledgeHealthStatus.OK
    knowledge_available: bool = True
    reasoning_available: bool = False
    knowledge: KnowledgeHealth | None = None
    storage: StorageStats | None = None

    overhead: LoopOverhead = Field(default_factory=LoopOverhead)
    errors: list[str] = Field(default_factory=list)
    summary: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "OptimizationRun":
        return cls.model_validate_json(data)


class LoopResult(ContractBase):
    """The top-level result returned by an autonomous run."""

    schema_version: str = LOOP_SCHEMA_VERSION
    run: OptimizationRun
    termination: LoopTermination
    initial_configuration: RuntimeConfiguration
    final_configuration: RuntimeConfiguration
    changed: bool = False
    iterations_performed: int = Field(default=0, ge=0)
    report_path: str | None = None
    summary: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "LoopResult":
        return cls.model_validate_json(data)
