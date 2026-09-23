"""Core data contracts (shared vocabulary) for InferenceOS.

Contract objects are immutable (frozen) pydantic models with ``extra="forbid"`` so
that every field participating in a decision is explicit and validated, and so that
records can be safely persisted and reconstructed (audit requirement, Phase 8).
"""

from __future__ import annotations

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.candidate import Candidate
from inferenceos.contracts.enums import (
    AttentionType,
    BackendType,
    BottleneckKind,
    CandidateKind,
    CandidateStatus,
    Decision,
    GPUDeviceType,
    GPUVendor,
    MeasurementMode,
    MetricQuality,
    ObservationSource,
    QuantizationType,
    ReasoningEngineKind,
    ReasoningStatus,
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
from inferenceos.contracts.reasoning import ReasoningRequest, ReasoningResponse
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
    "BottleneckKind",
    "CPUProfile",
    "CacheProfile",
    "Candidate",
    "CandidateKind",
    "CandidateStatus",
    "ContractBase",
    "Decision",
    "EvidenceItem",
    "ExperimentResult",
    "GPUDeviceType",
    "GPUEngineUtilization",
    "GPUMemoryState",
    "GPUProfile",
    "GPUVendor",
    "HardwareProfile",
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
    "ReasoningEngineKind",
    "ReasoningRequest",
    "ReasoningResponse",
    "ReasoningStatus",
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
