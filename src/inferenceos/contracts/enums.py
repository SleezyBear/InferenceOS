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
