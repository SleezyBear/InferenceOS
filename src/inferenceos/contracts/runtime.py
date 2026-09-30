"""Phase 10 runtime adaptive monitoring contracts (conservative mode).

The runtime monitor observes an *active* workload, temporally qualifies deviations,
decides whether diagnosis/intervention is worth the disruption, recommends deep
analysis rather than forcing it, requires explicit user approval for disruptive
active-runtime changes, checkpoints before pausing, and returns the workload safely to
its protected state.

Everything here is inert data. The only mutation surface remains the registered,
Phase 7-compatible runtime configuration of the workload InferenceOS controls.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.goal import PaybackCheck
from inferenceos.contracts.accounting import NetBenefitAssessment
from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import (
    AnomalyClass,
    ApprovalOutcome,
    BottleneckKind,
    CheckpointStatus,
    EvidenceStrength,
    ExperimentDecisionKind,
    InterventionClass,
    KnowledgeSelectionEffect,
    KnowledgeWriteStatus,
    MonitoringMode,
    QualityStatus,
    RestorationStatus,
    RuntimeState,
    RuntimeTerminationReason,
    WorkloadKind,
    WorkloadState,
)
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.external import ExternalRecommendationRun
from inferenceos.contracts.lifecycle import ResidentTargetRecovery
from inferenceos.contracts.preference import (
    OptimizationPreference,
    PerformanceStateAssessment,
    ResumeVerification,
)
from inferenceos.contracts.profiling import MetricStats
from inferenceos.contracts.search import GpuOffloadSearchResult
from inferenceos.contracts.system_state import SystemState

RUNTIME_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Policy / budget                                                              #
# --------------------------------------------------------------------------- #
class RuntimeBudget(ContractBase):
    """Explicit, configurable bounds for a bounded runtime monitoring run."""

    max_duration_s: float = Field(default=900.0, gt=0.0)
    telemetry_interval_s: float = Field(default=0.5, gt=0.0)
    observation_window_size: int = Field(default=120, ge=8)
    baseline_samples: int = Field(default=8, ge=2)
    max_deep_analyses: int = Field(default=1, ge=0)
    max_interventions: int = Field(default=2, ge=0)
    max_cumulative_intervention_s: float = Field(default=300.0, gt=0.0)
    max_optimizer_overhead_s: float = Field(default=120.0, gt=0.0)
    approval_timeout_s: float = Field(default=60.0, gt=0.0)
    max_alerts_per_condition: int = Field(default=2, ge=1)

    observations_used: int = Field(default=0, ge=0)
    diagnoses_used: int = Field(default=0, ge=0)
    deep_analyses_used: int = Field(default=0, ge=0)
    interventions_used: int = Field(default=0, ge=0)
    cumulative_intervention_s: float = Field(default=0.0, ge=0.0)
    optimizer_overhead_used_s: float = Field(default=0.0, ge=0.0)

    def exhaustion(self, *, elapsed_s: float) -> RuntimeTerminationReason | None:
        """First exhausted bound, or ``None`` when monitoring may continue."""
        if elapsed_s >= self.max_duration_s:
            return RuntimeTerminationReason.DURATION_REACHED
        if self.optimizer_overhead_used_s >= self.max_optimizer_overhead_s:
            return RuntimeTerminationReason.MONITORING_BUDGET_EXHAUSTED
        if self.interventions_used >= self.max_interventions:
            return RuntimeTerminationReason.INTERVENTION_LIMIT_REACHED
        return None

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()


class InterventionCostModel(ContractBase):
    """Normalized cost inputs for the intervention-economics decision (seconds)."""

    detection_cost_s: float = Field(default=0.02, ge=0.0)
    diagnosis_cost_s: float = Field(default=0.05, ge=0.0)
    reasoning_load_cost_s: float = Field(default=8.0, ge=0.0)
    reasoning_run_cost_s: float = Field(default=6.0, ge=0.0)
    checkpoint_cost_s: float = Field(default=0.05, ge=0.0)
    pause_cost_s: float = Field(default=0.5, ge=0.0)
    experiment_cost_s: float = Field(default=30.0, ge=0.0)
    reload_cost_s: float = Field(default=2.0, ge=0.0)
    disruption_cost_s: float = Field(default=1.0, ge=0.0)
    benefit_scale_s: float = Field(default=60.0, gt=0.0)
    weight_benefit: float = Field(default=1.0, ge=0.0)
    weight_cost: float = Field(default=1.0, ge=0.0)
    min_net_value: float = Field(default=0.05)
    deep_analysis_evidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()


class RuntimePolicy(ContractBase):
    """Conservative active-runtime policy (inspectable; no magic constants)."""

    budget: RuntimeBudget = Field(default_factory=RuntimeBudget)
    persistence_samples: int = Field(default=3, ge=2)
    condition_lookback_samples: int = Field(default=6, ge=2)
    # Workload-aware baseline eligibility. ``baseline_workload_samples`` is the number of
    # baseline observations that must carry real workload-performance metrics
    # (latency/TTFT/decode throughput) before a baseline may be established; ``0`` keeps
    # the legacy host-only baseline. ``baseline_max_wait_s`` bounds that wait (``0`` means
    # no explicit cap beyond the run duration) so a stalled workload cannot hang
    # monitoring. This never fabricates metrics: if none arrive, the baseline is host-only.
    baseline_workload_samples: int = Field(default=0, ge=0)
    baseline_max_wait_s: float = Field(default=0.0, ge=0.0)
    # Adaptive monitoring intensity. A value of 0 disables the corresponding behaviour
    # (so directly-constructed policies keep the legacy single-intensity behaviour);
    # ``build_runtime_policy`` enables them from settings for real runs. These only
    # change *sampling intensity*; they never terminate the monitor.
    monitor_mode_overhead_fraction_high: float = Field(default=0.0, ge=0.0)
    monitor_mode_overhead_fraction_low: float = Field(default=0.0, ge=0.0)
    monitor_mode_active_window_s: float = Field(default=0.0, ge=0.0)
    monitor_mode_dormant_after_s: float = Field(default=0.0, ge=0.0)
    monitor_mode_low_cost_interval_s: float = Field(default=0.0, ge=0.0)
    monitor_mode_dormant_interval_s: float = Field(default=0.0, ge=0.0)
    monitor_mode_expensive_scan_interval_s: float = Field(default=0.0, ge=0.0)
    monitor_mode_low_cost_scan_interval_s: float = Field(default=0.0, ge=0.0)
    monitor_mode_dormant_scan_interval_s: float = Field(default=0.0, ge=0.0)
    # GPU telemetry cadence (mode-controlled). ``0`` disables runtime GPU sampling.
    monitor_mode_gpu_scan_interval_s: float = Field(default=0.0, ge=0.0)
    # Repeated-opportunity suppression: a condition diagnosed non-actionable is
    # aggregated/suppressed until it materially changes (by ``score_delta``), telemetry
    # availability changes, the bounded cooldown expires, or it clears and reappears.
    opportunity_resuppress_score_delta: float = Field(default=0.15, ge=0.0)
    opportunity_resuppress_cooldown_s: float = Field(default=120.0, ge=0.0)
    deviation_threshold: float = Field(default=0.25, gt=0.0)
    hysteresis_ratio: float = Field(default=0.5, gt=0.0)
    baseline_drift_tolerance: float = Field(default=0.15, ge=0.0)
    diagnosis_cooldown_s: float = Field(default=30.0, ge=0.0)
    intervention_cooldown_s: float = Field(default=120.0, ge=0.0)
    min_evidence_strength: EvidenceStrength = EvidenceStrength.LOW
    require_approval_for_disruptive: bool = True
    allow_non_disruptive_auto_adaptation: bool = False
    max_alerts_per_condition: int = Field(default=2, ge=1)
    max_observation_processes: int = Field(default=6, ge=1)
    cost_model: InterventionCostModel = Field(default_factory=InterventionCostModel)

    # Phase 8 knowledge reuse (kept explicit so the runtime layer reuses the same
    # explainable applicability/selection semantics as Phase 9).
    knowledge_lookup_limit: int = Field(default=20, ge=1)
    skip_directly_negative: bool = True
    allow_revalidation: bool = True
    knowledge_positive_bonus: float = Field(default=0.05, ge=0.0)
    knowledge_negative_penalty: float = Field(default=0.15, ge=0.0)

    def as_dict(self) -> dict[str, object]:
        return {
            "budget": self.budget.as_dict(),
            "persistence_samples": self.persistence_samples,
            "condition_lookback_samples": self.condition_lookback_samples,
            "baseline_workload_samples": self.baseline_workload_samples,
            "baseline_max_wait_s": self.baseline_max_wait_s,
            "monitor_mode_overhead_fraction_high": self.monitor_mode_overhead_fraction_high,
            "monitor_mode_overhead_fraction_low": self.monitor_mode_overhead_fraction_low,
            "monitor_mode_active_window_s": self.monitor_mode_active_window_s,
            "monitor_mode_dormant_after_s": self.monitor_mode_dormant_after_s,
            "monitor_mode_low_cost_interval_s": self.monitor_mode_low_cost_interval_s,
            "monitor_mode_dormant_interval_s": self.monitor_mode_dormant_interval_s,
            "monitor_mode_expensive_scan_interval_s": self.monitor_mode_expensive_scan_interval_s,
            "monitor_mode_low_cost_scan_interval_s": self.monitor_mode_low_cost_scan_interval_s,
            "monitor_mode_dormant_scan_interval_s": self.monitor_mode_dormant_scan_interval_s,
            "monitor_mode_gpu_scan_interval_s": self.monitor_mode_gpu_scan_interval_s,
            "opportunity_resuppress_score_delta": self.opportunity_resuppress_score_delta,
            "opportunity_resuppress_cooldown_s": self.opportunity_resuppress_cooldown_s,
            "deviation_threshold": self.deviation_threshold,
            "hysteresis_ratio": self.hysteresis_ratio,
            "baseline_drift_tolerance": self.baseline_drift_tolerance,
            "diagnosis_cooldown_s": self.diagnosis_cooldown_s,
            "intervention_cooldown_s": self.intervention_cooldown_s,
            "min_evidence_strength": self.min_evidence_strength.value,
            "require_approval_for_disruptive": self.require_approval_for_disruptive,
            "allow_non_disruptive_auto_adaptation": self.allow_non_disruptive_auto_adaptation,
            "max_alerts_per_condition": self.max_alerts_per_condition,
            "max_observation_processes": self.max_observation_processes,
            "cost_model": self.cost_model.as_dict(),
            "knowledge_lookup_limit": self.knowledge_lookup_limit,
            "skip_directly_negative": self.skip_directly_negative,
            "allow_revalidation": self.allow_revalidation,
            "knowledge_positive_bonus": self.knowledge_positive_bonus,
            "knowledge_negative_penalty": self.knowledge_negative_penalty,
        }


# --------------------------------------------------------------------------- #
# Workload abstraction                                                         #
# --------------------------------------------------------------------------- #
class WorkloadDescriptor(ContractBase):
    """Identity and capabilities of the controlled active workload."""

    workload_id: str
    workload_kind: WorkloadKind = WorkloadKind.UNKNOWN
    model_ref: str
    model_path: str
    backend_name: str
    configuration: RuntimeConfiguration = Field(default_factory=RuntimeConfiguration)
    context_length: int | None = None
    prompts_ref: str | None = None
    owned: bool = True
    controllable: bool = True
    pause_supported: bool = True
    checkpoint_supported: bool = True
    resume_supported: bool = True
    limitations: list[str] = Field(default_factory=list)


class WorkloadSnapshot(ContractBase):
    """A cheap point-in-time view of workload continuity/safety."""

    captured_at: datetime = Field(default_factory=_utcnow)
    state: WorkloadState = WorkloadState.UNKNOWN
    running: bool = False
    generating: bool = False
    terminated_externally: bool = False
    safe_to_pause: bool = False
    can_resume: bool = False
    requests_served: int = Field(default=0, ge=0)
    in_flight_request_id: str | None = None
    last_error: str | None = None
    # B2: the resident target was lost and bounded recovery failed; the workload is
    # degraded/unrecoverable and will deterministically refuse further requests.
    unrecoverable: bool = False


# --------------------------------------------------------------------------- #
# Observations / temporal qualification                                        #
# --------------------------------------------------------------------------- #
class RuntimeObservation(ContractBase):
    """One bounded runtime telemetry sample (never persisted to long-term knowledge)."""

    observation_id: str
    sequence: int = Field(default=0, ge=0)
    captured_at: datetime = Field(default_factory=_utcnow)
    workload_state: WorkloadState = WorkloadState.UNKNOWN
    generating: bool = False
    context_length: int | None = None
    config_key: str = ""
    # The *requested* thread configuration and the observed *effective* generation thread
    # count at sample time. They are kept distinct; ``effective_threads`` is backend
    # evidence and is never inferred from CPU topology. ``None`` means unset/unknown.
    requested_threads: int | None = Field(default=None, ge=0)
    effective_threads: int | None = Field(default=None, ge=1)
    # The *effective* weight-load mode observed from backend evidence at sample time
    # (``True`` = mmap, ``False`` = non-mmap, ``None`` = unknown); never inferred from
    # the configured value.
    effective_mmap: bool | None = None
    workload_ref: str | None = None
    model_ref: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    system_state: SystemState | None = None
    run_id: str | None = None
    # Monotonic count of completed workload generations/requests observed at sample time
    # (from the workload's existing ``requests_served``). Used to build workload-baseline
    # statistics from *distinct* completed generations instead of repeatedly sampling the
    # same completed generation. ``None`` when the workload does not expose it.
    requests_served: int | None = Field(default=None, ge=0)
    condition: str = "observed"


class MetricDeviation(ContractBase):
    """Explainable deviation of one metric's recent window vs. the baseline."""

    metric: str
    direction: str = "lower_is_better"
    baseline_mean: float | None = None
    recent_mean: float | None = None
    absolute_delta: float | None = None
    relative_delta: float | None = None
    deviation_score: float = Field(default=0.0, ge=0.0)
    baseline_cv: float | None = None
    exceeds_threshold: bool = False


class ObservationWindow(ContractBase):
    """Bounded-window temporal qualification verdict."""

    capacity: int = Field(ge=1)
    total_observed: int = Field(default=0, ge=0)
    baseline_count: int = Field(default=0, ge=0)
    recent_count: int = Field(default=0, ge=0)
    window_seconds: float = Field(default=0.0, ge=0.0)
    anomaly: AnomalyClass = AnomalyClass.NONE
    persistence_samples: int = Field(default=0, ge=0)
    context_changed: bool = False
    condition_key: str | None = None
    deviations: list[MetricDeviation] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


class RuntimeBaseline(ContractBase):
    """A workload baseline, kept in bounded history so drift is explicit."""

    baseline_id: str
    established_at: datetime = Field(default_factory=_utcnow)
    workload_id: str
    model_ref: str
    backend_name: str
    configuration: RuntimeConfiguration = Field(default_factory=RuntimeConfiguration)
    context_length: int | None = None
    metrics: dict[str, MetricStats] = Field(default_factory=dict)
    observation_count: int = Field(default=0, ge=0)
    source: str = "initial"
    supersedes: str | None = None


# --------------------------------------------------------------------------- #
# Opportunity / economics / recommendation                                     #
# --------------------------------------------------------------------------- #
class RuntimeOpportunity(ContractBase):
    """A temporally-qualified suspected opportunity (not yet an intervention)."""

    opportunity_id: str
    detected_at: datetime = Field(default_factory=_utcnow)
    anomaly: AnomalyClass
    condition_key: str
    kind: str
    persistence_samples: int = Field(default=0, ge=0)
    deviation_score: float = Field(default=0.0, ge=0.0)
    context_changed: bool = False
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)

    diagnosis_id: str | None = None
    diagnosis_kind: BottleneckKind | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    strength: EvidenceStrength = EvidenceStrength.NONE
    meaningful: bool = False
    plan_id: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)


class InterventionEconomics(ContractBase):
    """Explicit "is this worth interrupting the workload?" decision."""

    expected_benefit_score: float = Field(default=0.0, ge=0.0)
    evidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    persistence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_score: float = Field(default=0.0, ge=0.0)
    net_value: float = 0.0
    beneficial: bool = False
    requires_approval: bool = True
    intervention_class: InterventionClass = InterventionClass.NONE
    costs: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)
    # User-controlled economics (separate from the technical candidate decision): which
    # policy/state applied and whether the measured improvement cleared the required
    # worthwhile-improvement threshold.
    optimization_mode: str | None = None
    performance_state: str | None = None
    required_improvement: float | None = None
    measured_improvement: float | None = None
    improvement_justified: bool = True
    # Step 4: goal mode and payback check (``None`` when the old rules were used).
    goal_mode: str | None = None
    goal_gain: float | None = None
    keep_verdict: str | None = None
    payback: PaybackCheck | None = None


class RuntimeRecommendation(ContractBase):
    """A structured deep-analysis / adaptation proposal shown to the user."""

    proposal_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    opportunity_id: str
    condition: str
    why_it_matters: str
    analysis_summary: str
    proposed_action_id: str | None = None
    proposed_parameters: dict[str, object] = Field(default_factory=dict)
    expected_benefit: str = ""
    expected_cost_s: float = Field(default=0.0, ge=0.0)
    expected_disruption_s: float = Field(default=0.0, ge=0.0)
    pause_required: bool = True
    checkpoint_planned: bool = True
    reversible: bool = True
    expected_learning: str = ""
    context_identity: str = ""
    expires_at: datetime | None = None
    requires_approval: bool = True
    intervention_class: InterventionClass = InterventionClass.DISRUPTIVE_ADAPTATION


class RuntimeNotification(ContractBase):
    """Structured notification payload sent through the notification abstraction."""

    notification_id: str
    proposal_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    level: str = "info"
    title: str = ""
    summary: str = ""
    recommendation: RuntimeRecommendation


class ApprovalRecord(ContractBase):
    """Auditable approval decision, bound to a specific proposal + context."""

    approval_id: str
    proposal_id: str
    outcome: ApprovalOutcome
    requested_at: datetime = Field(default_factory=_utcnow)
    decided_at: datetime | None = None
    context_identity: str = ""
    decided_by: str | None = None
    detail: str = ""


# --------------------------------------------------------------------------- #
# Checkpoint / intervention / cooldown / state events                          #
# --------------------------------------------------------------------------- #
class RuntimeCheckpoint(ContractBase):
    """Protected workload state captured before a pause/intervention.

    Under the current process-per-generation llama.cpp backend this covers the
    workload's runtime configuration + identity + resume information. It is *not* an
    OS/process memory snapshot and does not claim to be one.
    """

    checkpoint_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    status: CheckpointStatus = CheckpointStatus.CAPTURED
    configuration: RuntimeConfiguration
    workload_id: str
    model_ref: str
    backend_name: str
    context_identity: str | None = None
    reason: str = ""
    resume_supported: bool = True
    resume_token: str | None = None
    integrity: str | None = None
    verified: bool = False
    previous_baseline_id: str | None = None
    limitations: list[str] = Field(default_factory=list)


class RuntimeStateEvent(ContractBase):
    state: RuntimeState
    timestamp: datetime = Field(default_factory=_utcnow)
    observation: int = Field(default=0, ge=0)
    detail: str = ""
    data: dict[str, object] = Field(default_factory=dict)


class MonitoringModeEvent(ContractBase):
    """One adaptive monitoring-intensity transition (separate from the lifecycle state).

    Recorded so a run can show the current mode, why it changed, the resulting sampling
    interval/cadence, and the measured monitoring-overhead fraction at the transition.
    """

    mode: MonitoringMode
    timestamp: datetime = Field(default_factory=_utcnow)
    observation: int = Field(default=0, ge=0)
    reason: str = ""
    sampling_interval_s: float = Field(default=0.0, ge=0.0)
    expensive_scan_interval_s: float = Field(default=0.0, ge=0.0)
    overhead_fraction: float | None = None
    data: dict[str, object] = Field(default_factory=dict)


class OpportunityAssessmentStatus:
    """Status of a diagnosed condition that produced no actionable opportunity."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_ACTIONABLE_SIGNAL = "no_actionable_signal"


class OpportunityAssessment(ContractBase):
    """Aggregated record of a condition that was diagnosed non-actionable.

    Prevents repeated identical opportunity records while the underlying condition is
    materially unchanged, while preserving auditability. A new assessment (and therefore
    a new opportunity) is emitted when the condition materially changes, telemetry
    availability changes, the bounded cooldown expires, or the condition clears and
    reappears.
    """

    condition_key: str
    metrics: list[str] = Field(default_factory=list)
    status: str = OpportunityAssessmentStatus.INSUFFICIENT_EVIDENCE
    first_seen: datetime = Field(default_factory=_utcnow)
    last_seen: datetime = Field(default_factory=_utcnow)
    suppressed_count: int = Field(default=0, ge=0)
    diagnosis_id: str | None = None
    diagnosis_status: str = ""
    last_score: float = Field(default=0.0, ge=0.0)
    expires_at: datetime | None = None


class RuntimeIntervention(ContractBase):
    """One approved controlled intervention (deep analysis and/or experiment)."""

    intervention_id: str
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    proposal_id: str | None = None
    approval_outcome: ApprovalOutcome = ApprovalOutcome.NOT_REQUIRED
    checkpoint_id: str | None = None
    candidate_id: str | None = None
    action_id: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)
    experiment_id: str | None = None
    decision: ExperimentDecisionKind | None = None
    retained: bool = False
    quality_status: QualityStatus | None = None
    restoration_status: RestorationStatus = RestorationStatus.NOT_REQUIRED
    validation_ok: bool = False
    resume_verified: bool = False
    duration_s: float = Field(default=0.0, ge=0.0)
    knowledge_write_status: KnowledgeWriteStatus | None = None
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    # Phase 12C: the post-measurement economic retention decision (technical improvement
    # is necessary but not sufficient for retention).
    retention_economics: InterventionEconomics | None = None

    # Phase 10R: real request-lifecycle / in-flight safety evidence.
    opportunity_detected_at: datetime | None = None
    in_flight_at_start: bool = False
    in_flight_request_id: str | None = None
    in_flight_completed_at: datetime | None = None
    deferred_for_in_flight: bool = False
    intervention_applied_at: datetime | None = None
    resumed_at: datetime | None = None
    target_pid: int | None = None
    target_identity_ok: bool | None = None


class RuntimeRequestRecord(ContractBase):
    """One real inference request handled by an active workload (real timestamps)."""

    request_id: str
    submitted_at: datetime = Field(default_factory=_utcnow)
    first_token_at: datetime | None = None
    completed_at: datetime | None = None
    latency_s: float | None = None
    generated_tokens: int | None = None
    decode_tokens_per_sec: float | None = None
    success: bool = True
    error: str | None = None


class RuntimeSessionProvenance(ContractBase):
    """Identity binding for one runtime monitoring session.

    It records the monitor process and the *distinct* resident target so a stale or
    self-referential arrangement cannot silently drive a runtime decision.
    """

    run_id: str
    session_id: str
    started_at: datetime = Field(default_factory=_utcnow)
    monitor_pid: int
    monitor_create_time: float | None = None
    target_pid: int | None = None
    target_create_time: float | None = None
    target_alive: bool | None = None
    distinct_processes: bool | None = None
    model_ref: str | None = None
    model_path: str | None = None
    backend_name: str | None = None
    hardware_hash: str | None = None
    notes: list[str] = Field(default_factory=list)



class RuntimeCooldown(ContractBase):
    cooldown_id: str
    key: str
    condition_key: str
    reason: str
    trigger: str
    started_at: datetime = Field(default_factory=_utcnow)
    expires_at: datetime


class RuntimeKnowledgeQuery(ContractBase):
    candidate_id: str
    action_id: str
    effect: KnowledgeSelectionEffect
    records_considered: int = Field(default=0, ge=0)
    detail: str = ""


# --------------------------------------------------------------------------- #
# Run / result                                                                 #
# --------------------------------------------------------------------------- #
class RuntimeOverhead(ContractBase):
    """Measured monitoring and intervention overhead (kept separate from workload)."""

    detection_s: float = Field(default=0.0, ge=0.0)
    diagnosis_s: float = Field(default=0.0, ge=0.0)
    reasoning_s: float = Field(default=0.0, ge=0.0)
    checkpoint_s: float = Field(default=0.0, ge=0.0)
    pause_s: float = Field(default=0.0, ge=0.0)
    experiment_s: float = Field(default=0.0, ge=0.0)
    validation_s: float = Field(default=0.0, ge=0.0)
    resume_s: float = Field(default=0.0, ge=0.0)
    notification_s: float = Field(default=0.0, ge=0.0)
    # Phase 10 search integration: wall time spent running the resource-aware
    # multi-candidate GPU-offload search (a target-interruption/measurement cost,
    # accounted separately from monitoring overhead).
    gpu_search_s: float = Field(default=0.0, ge=0.0)
    monitoring_wall_s: float = Field(default=0.0, ge=0.0)
    optimizer_overhead_s: float = Field(default=0.0, ge=0.0)
    total_s: float = Field(default=0.0, ge=0.0)
    sample_count: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)


class ResourceImpact(ContractBase):
    """Whether observing the workload materially distorted it."""

    sampling_overhead_s: float = Field(default=0.0, ge=0.0)
    sampling_cpu_percent: float | None = None
    sampling_rss_bytes: int | None = None
    # Instrumentation of the sampling path itself (per-stage accumulated seconds), so a
    # run can report where sampling cost actually goes rather than only the aggregate.
    sampling_breakdown: dict[str, float] = Field(default_factory=dict)
    decode_tps_mean: float | None = None
    decode_tps_cv: float | None = None
    throughput_within_tolerance: bool = True
    notes: list[str] = Field(default_factory=list)


class RuntimeTermination(ContractBase):
    reason: RuntimeTerminationReason
    detail: str
    clean: bool = True
    safety: bool = False
    at_observation: int = Field(default=0, ge=0)


class RuntimeMonitorRun(ContractBase):
    """The complete machine-readable audit record of one runtime monitoring run."""

    schema_version: str = RUNTIME_SCHEMA_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None

    workload: WorkloadDescriptor
    policy: RuntimePolicy = Field(default_factory=RuntimePolicy)
    budget: RuntimeBudget = Field(default_factory=RuntimeBudget)

    baseline: RuntimeBaseline | None = None
    baselines: list[RuntimeBaseline] = Field(default_factory=list)
    states: list[RuntimeStateEvent] = Field(default_factory=list)
    window: ObservationWindow | None = None
    opportunities: list[RuntimeOpportunity] = Field(default_factory=list)
    recommendations: list[RuntimeRecommendation] = Field(default_factory=list)
    notifications: list[RuntimeNotification] = Field(default_factory=list)
    approvals: list[ApprovalRecord] = Field(default_factory=list)
    checkpoints: list[RuntimeCheckpoint] = Field(default_factory=list)
    interventions: list[RuntimeIntervention] = Field(default_factory=list)
    cooldowns: list[RuntimeCooldown] = Field(default_factory=list)
    knowledge_queries: list[RuntimeKnowledgeQuery] = Field(default_factory=list)
    # Phase 11: external attribution/recommendation runs recorded alongside the
    # runtime run (external processes are never controlled without approval).
    external_runs: list[ExternalRecommendationRun] = Field(default_factory=list)
    # B2: explicit bounded resident-target loss/recovery lifecycles.
    target_recoveries: list[ResidentTargetRecovery] = Field(default_factory=list)
    # Phase 10 search integration: the full resource-aware GPU-offload search records
    # (feasible range, sampled candidates, per-candidate measurements, selected value
    # or NO_CHANGE, and rationale) that were run during this monitoring session.
    gpu_searches: list[GpuOffloadSearchResult] = Field(default_factory=list)
    # H4: consolidated optimizer overhead vs target benefit accounting.
    net_benefit: NetBenefitAssessment | None = None
    # User-controlled optimization policy that was active for this run, the measured
    # performance state it resolved against, and the bounded resume-verification records.
    optimization_preference: OptimizationPreference | None = None
    performance_state: PerformanceStateAssessment | None = None
    resume_verifications: list[ResumeVerification] = Field(default_factory=list)
    # Adaptive monitoring intensity: transition history and time spent per mode. This
    # distinguishes "cheap because sampling is efficient" from "cheap because monitoring
    # stopped" (the monitor's lifetime is tied to the target, not to this mode).
    monitoring_modes: list[MonitoringModeEvent] = Field(default_factory=list)
    final_monitoring_mode: MonitoringMode = MonitoringMode.NORMAL
    monitoring_mode_seconds: dict[str, float] = Field(default_factory=dict)
    # Aggregated non-actionable conditions (suppression/auditability of repeated
    # opportunities that did not change the decision).
    opportunity_assessments: list[OpportunityAssessment] = Field(default_factory=list)

    observations_collected: int = Field(default=0, ge=0)
    diagnosis_events: int = Field(default=0, ge=0)
    deep_analysis_events: int = Field(default=0, ge=0)
    # Requested thread configuration vs the observed *effective* value (backend evidence);
    # both are recorded so the relationship is explicit. ``None`` means unset/unknown.
    requested_threads: int | None = Field(default=None, ge=0)
    effective_threads: int | None = Field(default=None, ge=1)
    final_state: RuntimeState = RuntimeState.TERMINAL_SAFE

    termination: RuntimeTermination
    overhead: RuntimeOverhead = Field(default_factory=RuntimeOverhead)
    resource_impact: ResourceImpact = Field(default_factory=ResourceImpact)
    session: RuntimeSessionProvenance | None = None
    errors: list[str] = Field(default_factory=list)
    summary: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "RuntimeMonitorRun":
        return cls.model_validate_json(data)


class RuntimeMonitorResult(ContractBase):
    schema_version: str = RUNTIME_SCHEMA_VERSION
    run: RuntimeMonitorRun
    termination: RuntimeTermination
    final_state: RuntimeState
    observations_collected: int = Field(default=0, ge=0)
    report_path: str | None = None
    summary: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "RuntimeMonitorResult":
        return cls.model_validate_json(data)
