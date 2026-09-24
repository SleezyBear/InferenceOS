"""Enumerations shared across InferenceOS contracts."""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String-valued enum that serializes/compares as its value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class GPUVendor(StrEnum):
    AMD = "amd"
    NVIDIA = "nvidia"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


class GPUDeviceType(StrEnum):
    DISCRETE = "discrete"
    INTEGRATED = "integrated"
    VIRTUAL = "virtual"
    CPU = "cpu"
    UNKNOWN = "unknown"


class BackendType(StrEnum):
    VULKAN = "vulkan"
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    OPENCL = "opencl"
    SYCL = "sycl"
    CPU = "cpu"


class QuantizationType(StrEnum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    Q8_0 = "q8_0"
    Q6_K = "q6_k"
    Q5_K_M = "q5_k_m"
    Q5_K_S = "q5_k_s"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"
    Q4_K_M = "q4_k_m"
    Q4_K_S = "q4_k_s"
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q3_K_M = "q3_k_m"
    Q3_K_S = "q3_k_s"
    Q3_K_L = "q3_k_l"
    Q2_K = "q2_k"
    IQ4_XS = "iq4_xs"
    IQ4_NL = "iq4_nl"
    IQ3_M = "iq3_m"
    IQ2_M = "iq2_m"
    UNKNOWN = "unknown"


class AttentionType(StrEnum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"
    UNKNOWN = "unknown"


class StorageMediaType(StrEnum):
    SSD = "ssd"
    HDD = "hdd"
    UNKNOWN = "unknown"


class WorkloadKind(StrEnum):
    SHORT = "short"
    LONG = "long"
    CODE = "code"
    PROSE = "prose"
    CHAT = "chat"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class MeasurementMode(StrEnum):
    COLD = "cold"
    WARM = "warm"


class ObservationSource(StrEnum):
    HARNESS = "harness"
    CONTINUOUS_MONITOR = "continuous_monitor"
    DIAGNOSTIC = "diagnostic"
    MANUAL = "manual"


class MetricQuality(StrEnum):
    """Availability/provenance of a single measurement.

    ``MEASURED``   — observed directly from the real system/backend.
    ``DERIVED``    — computed deterministically from other measured values.
    ``UNAVAILABLE`` — the subsystem could not provide it (explicitly ``None``).
    ``NOT_APPLICABLE`` — the metric does not apply to this run.
    """

    MEASURED = "measured"
    DERIVED = "derived"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class BottleneckKind(StrEnum):
    """Extensible deterministic bottleneck vocabulary (Phase 4).

    ``NONE`` and ``INSUFFICIENT_EVIDENCE`` are first-class terminal outcomes: a
    resource being heavily used is not itself a bottleneck.
    """

    CPU = "cpu"
    GPU_UNDERUTILIZATION = "gpu_underutilization"
    GPU_MEMORY_LIMIT = "gpu_memory_limit"
    COMPUTE_BOUND = "compute_bound"
    MEMORY_PRESSURE = "memory_pressure"
    MODEL_FOOTPRINT = "model_footprint"
    PAGING = "paging"
    STORAGE = "storage"
    IO_BOUND = "io_bound"
    LATENCY = "latency"
    EXTERNAL_PROCESS = "external_process"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NONE = "none"


class InferenceStage(StrEnum):
    """Which phase of an inference run a finding applies to."""

    LOAD = "load"
    PREFILL = "prefill"
    TTFT = "ttft"
    DECODE = "decode"
    END_TO_END = "end_to_end"


class EvidenceStrength(StrEnum):
    """Explainable, evidence-derived confidence category (not a probability)."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidencePolarity(StrEnum):
    """Whether an evidence item supports or contradicts a candidate finding."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"


class CandidateKind(StrEnum):
    THREADS = "threads"
    BATCH_SIZE = "batch_size"
    UBATCH_SIZE = "ubatch_size"
    GPU_OFFLOAD_LAYERS = "gpu_offload_layers"
    MMAP = "mmap"
    CONTEXT_LENGTH = "context_length"
    KV_CACHE_DTYPE = "kv_cache_dtype"
    FLASH_ATTENTION = "flash_attention"


class CandidateStatus(StrEnum):
    PROPOSED = "proposed"
    DISCARDED = "discarded"
    SELECTED = "selected"
    APPLIED = "applied"
    KEPT = "kept"
    REVERTED = "reverted"


class Decision(StrEnum):
    KEEP = "keep"
    REVERT = "revert"
    NO_OP = "no_op"
    PENDING = "pending"


class ReasoningEngineKind(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    STUB = "stub"


class ReasoningTaskType(StrEnum):
    """What the reasoning engine is being asked to do."""

    ROOT_CAUSE_DIAGNOSIS = "root_cause_diagnosis"
    AMBIGUITY_RESOLUTION = "ambiguity_resolution"


class ReasoningStatus(StrEnum):
    OK = "ok"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Phase 6 — candidate generation & controlled optimization                      #
# --------------------------------------------------------------------------- #
class OptimizationSurface(StrEnum):
    """The kind of optimization surface an action operates on."""

    RUNTIME_PARAMS = "runtime_params"
    CONTEXT_MANAGEMENT = "context_management"
    GPU_OFFLOAD = "gpu_offload"
    MEMORY_MANAGEMENT = "memory_management"
    EXTERNAL_PROCESS = "external_process"
    SYSTEM_CONFIGURATION = "system_configuration"
    MODEL_REPRESENTATION = "model_representation"


class ParameterType(StrEnum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    STRING = "string"


class RiskLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Reversibility(StrEnum):
    """Conceptual reversibility of an intervention (Phase 8 owns rollback execution)."""

    REVERSIBLE = "reversible"
    REVERSIBLE_WITH_RESTART = "reversible_with_restart"
    NON_REVERSIBLE = "non_reversible"


class OptimizationScope(StrEnum):
    """How wide the blast radius of an intervention is."""

    TARGET_MODEL_RUNTIME = "target_model_runtime"
    TARGET_INFERENCE_PROCESS = "target_inference_process"
    CURRENT_WORKLOAD = "current_workload"
    USER_PROCESS = "user_process"
    EXTERNAL_PROCESS = "external_process"
    SYSTEM_WIDE = "system_wide"


class PrivilegeRequirement(StrEnum):
    STANDARD_USER = "standard_user"
    ELEVATION_REQUIRED = "elevation_required"
    UNKNOWN = "unknown"


class ApprovalRequirement(StrEnum):
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"


class CandidateEligibility(StrEnum):
    ELIGIBLE = "eligible"
    GATED = "gated"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class PreconditionKind(StrEnum):
    """Category of a precondition; drives deterministic eligibility policy."""

    CAPABILITY = "capability"
    APPLICABILITY = "applicability"
    EVIDENCE = "evidence"
    PARAMETER = "parameter"
    SAFETY = "safety"
    PRIVILEGE = "privilege"


class CandidatePlanOutcome(StrEnum):
    """Terminal outcome of one candidate-generation run."""

    CANDIDATES_GENERATED = "candidates_generated"
    NO_OPPORTUNITY = "no_opportunity"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_VIABLE_CANDIDATES = "no_viable_candidates"


# --------------------------------------------------------------------------- #
# Phase 7 — experiment execution & quality validation                         #
# --------------------------------------------------------------------------- #
class ExperimentStage(StrEnum):
    """Observable experiment-lifecycle transitions (audit trail)."""

    REQUESTED = "experiment_requested"
    PRECONDITIONS_REVALIDATED = "preconditions_revalidated"
    BASELINE_CAPTURED = "baseline_captured"
    CANDIDATE_APPLIED = "candidate_applied"
    MEASUREMENT_STARTED = "measurement_started"
    MEASUREMENT_COMPLETED = "measurement_completed"
    QUALITY_VALIDATED = "quality_validation_completed"
    COMPARISON_COMPLETED = "comparison_completed"
    DECISION_MADE = "decision_made"
    RESTORATION_STARTED = "restoration_started"
    RESTORATION_COMPLETED = "restoration_completed"
    FAILED = "experiment_failed"


class ExperimentDecisionKind(StrEnum):
    """Terminal classification of an experiment (never forced into pass/fail)."""

    IMPROVED = "improved"
    NO_MEANINGFUL_IMPROVEMENT = "no_meaningful_improvement"
    REGRESSION = "regression"
    QUALITY_REGRESSION = "quality_regression"
    INVALID = "experiment_invalid"
    PRECONDITION_FAILURE = "precondition_failure"
    APPLICATION_FAILURE = "application_failure"
    MEASUREMENT_FAILURE = "measurement_failure"
    RESTORATION_FAILURE = "restoration_failure"
    UNSAFE = "unsafe_to_continue"
    REJECTED = "candidate_rejected"


class QualityStatus(StrEnum):
    """Outcome of output quality/correctness validation."""

    OK = "ok"
    ACCEPTABLE_VARIATION = "acceptable_variation"
    QUALITY_REGRESSION = "quality_regression"
    EXECUTION_FAILURE = "execution_failure"
    STRUCTURAL_FAILURE = "structural_failure"
    UNKNOWN = "unknown"


class RestorationStatus(StrEnum):
    """Whether the protected baseline configuration was restored."""

    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    RESTORED = "restored"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Phase 8 — persistent knowledge & rollback                                    #
# --------------------------------------------------------------------------- #
class OptimizationOutcome(StrEnum):
    """Contextual outcome of a persisted optimization event (never a Boolean)."""

    IMPROVED = "improved"
    NO_MEANINGFUL_IMPROVEMENT = "no_meaningful_improvement"
    REGRESSION = "regression"
    QUALITY_REGRESSION = "quality_regression"
    INVALID = "invalid"
    FAILED = "failed"
    REVERTED = "reverted"
    BLOCKED = "blocked"


class ApplicabilityClass(StrEnum):
    """Explainable classification of whether historical evidence applies now."""

    DIRECTLY_APPLICABLE = "directly_applicable"
    CONDITIONALLY_APPLICABLE = "conditionally_applicable"
    REQUIRES_REVALIDATION = "requires_revalidation"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class DimensionStatus(StrEnum):
    """Per-dimension comparison of a historical context against the current one."""

    MATCH = "match"
    DIFFER = "differ"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class RevalidationTrigger(StrEnum):
    """Explicit reason historical evidence must be re-validated before reuse."""

    HARDWARE_CHANGE = "hardware_change"
    BACKEND_CHANGE = "backend_change"
    MODEL_CHANGE = "model_change"
    MODEL_VERSION_CHANGE = "model_version_change"
    QUANTIZATION_CHANGE = "quantization_change"
    WORKLOAD_CHANGE = "workload_change"
    RUNTIME_CHANGE = "runtime_change"
    SYSTEM_CONDITIONS_CHANGE = "system_conditions_change"
    DIAGNOSIS_CHANGE = "diagnosis_change"
    STALE = "stale"
    ACTION_UNREGISTERED = "action_unregistered"
    NONE = "none"


class KnowledgeWriteStatus(StrEnum):
    """Outcome of a persistence write (distinguishes persisted/duplicate/failure)."""

    PERSISTED = "persisted"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class KnowledgeEventType(StrEnum):
    """Append-only knowledge lifecycle events."""

    RECORDED = "recorded"
    REVALIDATION_MARKED = "revalidation_marked"
    ROLLBACK_LINKED = "rollback_linked"
    REVERTED = "reverted"
    CORRECTION = "correction"


class KnowledgeHealthStatus(StrEnum):
    """Health of the persistent knowledge store."""

    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    CORRUPT = "corrupt"


# --------------------------------------------------------------------------- #
# Phase 9 — pre-deployment autonomous optimization loop                        #
# --------------------------------------------------------------------------- #
class LoopState(StrEnum):
    """Explicit, auditable transitions of the pre-deployment optimization loop."""

    INITIALIZED = "initialized"
    OBSERVING = "observing"
    DIAGNOSING = "diagnosing"
    DEEP_DIAGNOSING = "deep_diagnosing"
    GENERATING = "generating"
    KNOWLEDGE_QUERY = "knowledge_query"
    SELECTING = "selecting"
    CHECKPOINTING = "checkpointing"
    EXPERIMENTING = "experimenting"
    DECIDING = "deciding"
    RESTORING = "restoring"
    PERSISTING = "persisting"
    RE_OBSERVING = "re_observing"
    TERMINATED = "terminated"


class LoopTerminationReason(StrEnum):
    """Why the autonomous loop stopped. A clean no-op is a success."""

    NO_MEANINGFUL_OPPORTUNITY = "no_meaningful_opportunity"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_ELIGIBLE_CANDIDATES = "no_eligible_candidates"
    ALL_CANDIDATES_REJECTED = "all_candidates_rejected"
    OPTIMIZATION_BUDGET_EXHAUSTED = "optimization_budget_exhausted"
    EXPERIMENT_BUDGET_EXHAUSTED = "experiment_budget_exhausted"
    TIME_BUDGET_EXHAUSTED = "time_budget_exhausted"
    OVERHEAD_BUDGET_EXHAUSTED = "overhead_budget_exhausted"
    DEEP_DIAGNOSIS_BUDGET_EXHAUSTED = "deep_diagnosis_budget_exhausted"
    QUALITY_SAFETY_STOP = "quality_safety_stop"
    RESTORATION_FAILURE = "restoration_failure"
    REASONING_UNAVAILABLE = "reasoning_unavailable"
    EXECUTION_FAILURE = "execution_failure"
    KNOWLEDGE_UNAVAILABLE = "knowledge_unavailable"
    OPERATOR_REQUIRED = "operator_required"
    COMPLETED = "completed"


class LoopIterationOutcome(StrEnum):
    """Terminal classification of one optimization iteration."""

    CANDIDATE_KEPT = "candidate_kept"
    CANDIDATE_REVERTED = "candidate_reverted"
    CANDIDATE_REJECTED = "candidate_rejected"
    NO_CANDIDATE = "no_candidate"
    EXECUTION_ERROR = "execution_error"
    RESTORATION_FAILURE = "restoration_failure"
    SAFETY_STOP = "safety_stop"


class LoopDecisionKind(StrEnum):
    """What the orchestrator decides to do next in an iteration."""

    PROCEED = "proceed"
    REJECT = "reject"
    TERMINATE = "terminate"


class KnowledgeSelectionEffect(StrEnum):
    """Explainable influence of historical evidence on candidate handling."""

    PREFERRED = "preferred"
    NEUTRAL = "neutral"
    REVALIDATION_REQUIRED = "requires_revalidation"
    DEPRIORITIZED = "deprioritized"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


class KnowledgeAvailability(StrEnum):
    """Whether the durable knowledge store could be consulted."""

    OK = "ok"
    UNAVAILABLE = "unavailable"


class CheckpointStatus(StrEnum):
    """Lifecycle of the protected pre-experiment checkpoint."""

    CAPTURED = "captured"
    RESTORED = "restored"
    FAILED = "failed"
    NOT_REQUIRED = "not_required"
