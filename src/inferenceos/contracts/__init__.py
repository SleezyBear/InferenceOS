"""Core data contracts (shared vocabulary) for InferenceOS.

Contract objects are immutable (frozen) pydantic models with ``extra="forbid"`` so
that every field participating in a decision is explicit and validated, and so that
records can be safely persisted and reconstructed (audit requirement, Phase 8).
"""

from __future__ import annotations

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.candidate import Candidate
from inferenceos.contracts.diagnosis import (
    DIAGNOSIS_SCHEMA_VERSION,
    BottleneckDiagnosis,
    DiagnosisContext,
)
from inferenceos.contracts.enums import (
    AttentionType,
    BackendType,
    BottleneckKind,
    CandidateKind,
    CandidateStatus,
    Decision,
    EvidencePolarity,
    EvidenceStrength,
    GPUDeviceType,
    GPUVendor,
    InferenceStage,
    MeasurementMode,
    MetricQuality,
    ObservationSource,
    QuantizationType,
    ReasoningEngineKind,
    ReasoningStatus,
    ReasoningTaskType,
    StorageMediaType,
    WorkloadKind,
)
from inferenceos.contracts.experiment import ExperimentResult, QualityScore
from inferenceos.contracts.hardware import (
    BackendAvailability,
    BackendDevice,
    CacheProfile,
    CPUProfile,
    GPUProfile,
    HardwareProfile,
    MemoryModule,
    MemoryProfile,
    OSProfile,
    StorageDevice,
)
from inferenceos.contracts.model import KVEstimate, ModelFootprint, ModelProfile
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.profiling import (
    PROFILING_SCHEMA_VERSION,
    BaselineReport,
    MetricStats,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunGroupStats,
    RunIdentity,
)
from inferenceos.contracts.reasoning import (
    REASONING_SCHEMA_VERSION,
    DeepDiagnosisOutcome,
    EscalationDecision,
    ReasoningBackendContext,
    ReasoningDiagnosisContext,
    ReasoningEvidence,
    ReasoningLifecycle,
    ReasoningMetric,
    ReasoningModelContext,
    ReasoningRequest,
    ReasoningResponse,
    ReasoningResult,
    ReasoningWorkloadContext,
)
from inferenceos.contracts.system_state import (
    GPUEngineUtilization,
    GPUMemoryState,
    ProcessState,
    SystemState,
)
from inferenceos.contracts.workload import WorkloadProfile

__all__ = [
    "AttentionType",
    "BackendAvailability",
    "BackendDevice",
    "BackendType",
    "BaselineReport",
    "Bottleneck",
    "BottleneckDiagnosis",
    "BottleneckKind",
    "CPUProfile",
    "CacheProfile",
    "Candidate",
    "CandidateKind",
    "CandidateStatus",
    "ContractBase",
    "DIAGNOSIS_SCHEMA_VERSION",
    "Decision",
    "DeepDiagnosisOutcome",
    "DiagnosisContext",
    "EscalationDecision",
    "EvidenceItem",
    "EvidencePolarity",
    "EvidenceStrength",
    "ExperimentResult",
    "GPUDeviceType",
    "GPUEngineUtilization",
    "GPUMemoryState",
    "GPUProfile",
    "GPUVendor",
    "HardwareProfile",
    "InferenceStage",
    "KVEstimate",
    "MeasurementMode",
    "MemoryModule",
    "MemoryProfile",
    "MetricQuality",
    "MetricStats",
    "ModelFootprint",
    "ModelProfile",
    "OSProfile",
    "Observation",
    "ObservationSource",
    "PROFILING_SCHEMA_VERSION",
    "ProcessState",
    "ProfilingConfig",
    "ProfilingRun",
    "QualityScore",
    "QuantizationType",
    "REASONING_SCHEMA_VERSION",
    "ReasoningBackendContext",
    "ReasoningDiagnosisContext",
    "ReasoningEngineKind",
    "ReasoningEvidence",
    "ReasoningLifecycle",
    "ReasoningMetric",
    "ReasoningModelContext",
    "ReasoningRequest",
    "ReasoningResponse",
    "ReasoningResult",
    "ReasoningStatus",
    "ReasoningTaskType",
    "ReasoningWorkloadContext",
    "RunContext",
    "RunGroupStats",
    "RunIdentity",
    "RunMetrics",
    "StorageDevice",
    "StorageMediaType",
    "SystemState",
    "WorkloadKind",
    "WorkloadProfile",
]
