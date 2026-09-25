"""Phase 5R resident backend & deep-diagnosis lifecycle contracts.

These contracts make the resident target lifecycle observable and auditable:

* process/resource evidence that the target really is a resident OS process;
* measured load/unload/reload timings;
* a real streaming TTFT measurement (not derived from counters);
* a timestamped lifecycle transition log;
* the entry-economics and feasibility decisions, including explicit refusals;
* the queue/reject behaviour for requests arriving during the diagnosis window;
* the deterministic-vs-LLM corroboration assessment.

They are additive: nothing here replaces the Phase 5 reasoning contracts.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.reasoning import (
    CorroborationAssessment,
    DeepDiagnosisOutcome,
    EscalationDecision,
)

PHASE5R_SCHEMA_VERSION = "5r.1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Explicit string constants (kept as a plain enum-like set for JSON stability).
class ResidentState:
    RESIDENT_TARGET = "resident_target"
    ENTRY_DECISION = "entry_decision"
    CHECKPOINT = "checkpoint"
    UNLOAD_TARGET = "unload_target"
    LOAD_REASONER = "load_reasoner"
    DEEP_DIAGNOSIS = "deep_diagnosis"
    UNLOAD_REASONER = "unload_reasoner"
    RELOAD_TARGET = "reload_target"
    VERIFY_TARGET = "verify_target"
    RESUME = "resume"
    REFUSED = "refused"
    FAILED = "failed"


class TargetHealth:
    """Point-in-time health of the resident target (B2).

    ``HEALTHY`` means the adapter reports a live process whose identity matches the
    captured identity. ``LOST`` means the process exited or its identity no longer
    matches (PID reuse). ``UNRECOVERABLE`` is terminal for the operation: the bounded
    recovery attempt has already been consumed and failed.
    """

    HEALTHY = "healthy"
    LOST = "lost"
    UNRECOVERABLE = "unrecoverable"


class TargetLossReason:
    """Why the resident target was considered lost (B2)."""

    PROCESS_EXITED = "process_exited"
    IDENTITY_CHANGED = "identity_changed"
    GENERATION_FAILED = "generation_failed"
    NOT_READY = "not_ready"


class ResidentRecoveryState:
    """Explicit, distinguishable lifecycle events for a bounded target recovery."""

    TARGET_LOST = "target_lost"
    TARGET_RECOVERED = "target_recovered"
    TARGET_UNRECOVERABLE = "target_unrecoverable"


class RequestPolicy:
    QUEUE = "queue"
    REJECT = "reject"


class ProcessIdentity(ContractBase):
    """OS process identity that is sufficient to detect PID reuse."""

    pid: int
    create_time: float
    name: str | None = None
    executable: str | None = None
    # A hash of the command line (never the command line itself) because the
    # server command line is not expected to contain secrets, but hashing avoids
    # leaking long paths into logs by accident.
    cmdline_hash: str | None = None


class ResourceSnapshot(ContractBase):
    """A real measurement of a process / the system at a point in time."""

    captured_at: datetime = Field(default_factory=_utcnow)
    pid: int | None = None
    exists: bool = False
    rss_bytes: int | None = None
    # Authoritative OS peak working-set when the platform exposes it (Windows
    # ``peak_wset``); ``None`` where unavailable rather than a guessed value.
    peak_rss_bytes: int | None = None
    vms_bytes: int | None = None
    system_available_bytes: int | None = None
    system_total_bytes: int | None = None
    gpu_memory_bytes: int | None = None
    gpu_memory_available: bool = False
    notes: list[str] = Field(default_factory=list)


class StopMethod:
    JOB_OBJECT = "job_object"
    PROCESS_TREE = "process_tree"
    ALREADY_EXITED = "already_exited"
    NONE = "none"


class ResidentLoadMeasurement(ContractBase):
    model_ref: str
    model_path: str
    pid: int
    identity: ProcessIdentity | None = None
    host: str
    port: int
    ready: bool
    startup_s: float
    ready_s: float | None = None
    before: ResourceSnapshot
    after: ResourceSnapshot
    resident_verified: bool = False
    notes: list[str] = Field(default_factory=list)


class ResidentUnloadMeasurement(ContractBase):
    pid: int | None = None
    requested_s: float = 0.0
    exited: bool = False
    exit_code: int | None = None
    stop_method: str = StopMethod.NONE
    before: ResourceSnapshot
    after: ResourceSnapshot
    unload_s: float = 0.0
    cleanup_complete: bool = False
    notes: list[str] = Field(default_factory=list)


class ResidentGenerationMeasurement(ContractBase):
    """One real, streamed generation with a measured (not derived) TTFT."""

    submitted_at: datetime
    first_token_at: datetime | None = None
    completed_at: datetime
    ttft_s: float | None = None
    total_s: float
    generated_tokens: int | None = None
    prompt_tokens: int | None = None
    decode_tokens_per_sec: float | None = None
    # True peak RSS during the generation interval (sampled + OS peak working set),
    # distinct from the final RSS observed when generation completed.
    peak_rss_bytes: int | None = None
    final_rss_bytes: int | None = None
    rss_sample_count: int = 0
    streamed: bool = False
    text: str = ""
    cache_prompt: bool = False
    notes: list[str] = Field(default_factory=list)


class LifecycleTransition(ContractBase):
    """A single timestamped transition in the deep-diagnosis state machine."""

    sequence: int
    state: str
    event: str
    wall_clock: datetime = Field(default_factory=_utcnow)
    monotonic_s: float
    duration_s: float | None = None
    success: bool = True
    detail: str = ""
    error: str | None = None


class LifecycleCostBreakdown(ContractBase):
    target_unload_s: float = 0.0
    reasoner_load_s: float = 0.0
    diagnosis_s: float = 0.0
    reasoner_unload_s: float = 0.0
    target_reload_s: float = 0.0
    interruption_cost_s: float = 0.0
    optimizer_overhead_s: float = 0.0
    total_s: float = 0.0


class EntryEconomics(ContractBase):
    """Transparent normalized-utility decision about entering deep diagnosis.

    Units are explicitly *seconds-equivalent utility* — not money. The model is
    deliberately simple and every component is recorded so a human can audit it.
    """

    enter: bool
    benefit_utility: float
    cost_utility: float
    net_utility: float
    deterministic_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    competing_families: int = Field(default=0, ge=0)
    benefits: dict[str, float] = Field(default_factory=dict)
    costs: dict[str, float] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


class FeasibilityAssessment(ContractBase):
    feasible: bool
    available_bytes: int | None = None
    target_resident_bytes: int | None = None
    reasoner_required_bytes: int | None = None
    projected_free_bytes: int | None = None
    headroom_bytes: int | None = None
    reasoner_footprint_basis: str = "unknown"
    reasons: list[str] = Field(default_factory=list)


class DiagnosisRequestRecord(ContractBase):
    """A request that arrived while the target was unloaded for diagnosis."""

    submitted_at: datetime = Field(default_factory=_utcnow)
    policy: str = RequestPolicy.REJECT
    accepted: bool = False
    queued: bool = False
    rejected: bool = False
    drained: bool = False
    prompt_hash: str | None = None
    response_detail: str = ""


class PostReloadVerification(ContractBase):
    new_identity: bool = False
    pid_before: int | None = None
    pid_after: int | None = None
    configuration_match: bool = False
    generation_ok: bool = False
    equivalence: str = "unknown"
    baseline_text_hash: str | None = None
    reloaded_text_hash: str | None = None
    invariants: dict[str, object] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ResidentRecoveryEvent(ContractBase):
    """One timestamped event in the bounded resident-target recovery lifecycle."""

    sequence: int
    state: str  # one of ``ResidentRecoveryState``
    at: datetime = Field(default_factory=_utcnow)
    detail: str = ""
    old_pid: int | None = None
    old_identity: ProcessIdentity | None = None
    new_pid: int | None = None
    new_identity: ProcessIdentity | None = None
    data: dict[str, object] = Field(default_factory=dict)


class ResidentTargetRecovery(ContractBase):
    """The complete evidence record for one bounded resident-target recovery.

    A recovery is only ever attempted when the target is actually lost, at most once
    per failure event. Every verification result is recorded so a recovered target is
    never accepted on liveness alone, and an unrecoverable outcome is explicit.
    """

    recovery_id: str
    reason: str  # one of ``TargetLossReason``
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    attempted: bool = False
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=1, ge=1)
    old_pid: int | None = None
    old_identity: ProcessIdentity | None = None
    new_pid: int | None = None
    new_identity: ProcessIdentity | None = None
    identity_verified: bool = False
    readiness_verified: bool = False
    configuration_verified: bool = False
    model_verified: bool = False
    backend_verified: bool = False
    recovered: bool = False
    unrecoverable: bool = False
    failure_reason: str | None = None
    cleanup_complete: bool = False
    orphan_free: bool = True
    events: list[ResidentRecoveryEvent] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "ResidentTargetRecovery":
        return cls.model_validate_json(data)


class LifecyclePolicy(ContractBase):
    """Explicit Phase 5R policy (normalized-utility economics + feasibility)."""

    benefit_scale_s: float = Field(default=300.0, ge=0.0)
    benefit_uncertainty_weight: float = Field(default=1.0, ge=0.0)
    benefit_competing_weight: float = Field(default=0.5, ge=0.0)
    cost_weight: float = Field(default=1.0, ge=0.0)
    min_net_utility: float = 0.0
    diagnosis_estimate_s: float = Field(default=20.0, ge=0.0)
    reasoner_load_estimate_s: float = Field(default=8.0, ge=0.0)
    reasoner_unload_estimate_s: float = Field(default=1.0, ge=0.0)
    feasibility_headroom_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    reasoner_reclaim_fraction: float = Field(default=0.9, ge=0.0, le=1.0)
    corroboration_ceiling: float = Field(default=0.25, ge=0.0, le=1.0)
    transition_log_limit: int = Field(default=200, ge=1)

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()


class RuntimeParamChange(ContractBase):
    parameter: str
    prior_value: object = None
    applied_value: object = None


class RuntimeParamApplication(ContractBase):
    """Outcome of applying runtime parameters to a loaded resident backend."""

    changed: bool
    requires_reload: bool = False
    reloaded: bool = False
    verified: bool = False
    changes: list[RuntimeParamChange] = Field(default_factory=list)
    reason: str = ""
    error: str | None = None


class ResidentDeepDiagnosisOutcome(ContractBase):
    """The complete evidence record for one resident deep-diagnosis attempt."""

    schema_version: str = PHASE5R_SCHEMA_VERSION
    outcome_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    source_diagnosis_id: str | None = None
    entered: bool = False
    refused_reason: str | None = None
    escalation: EscalationDecision
    entry_economics: EntryEconomics
    feasibility: FeasibilityAssessment | None = None
    target_load: ResidentLoadMeasurement | None = None
    target_unload: ResidentUnloadMeasurement | None = None
    reasoner_load: ResidentLoadMeasurement | None = None
    reasoner_unload: ResidentUnloadMeasurement | None = None
    target_reload: ResidentLoadMeasurement | None = None
    diagnosis: DeepDiagnosisOutcome | None = None
    post_reload: PostReloadVerification | None = None
    requests: list[DiagnosisRequestRecord] = Field(default_factory=list)
    transitions: list[LifecycleTransition] = Field(default_factory=list)
    cost: LifecycleCostBreakdown | None = None
    corroboration: CorroborationAssessment | None = None
    errors: list[str] = Field(default_factory=list)
    summary: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "ResidentDeepDiagnosisOutcome":
        return cls.model_validate_json(data)


# Apply explicit state labels so JSON carries stable strings.
ResidentDeepDiagnosisOutcome.model_rebuild()
