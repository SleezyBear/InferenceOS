"""Reasoning contracts (Phase 5 deep-diagnosis mode).

Phase 0 defined the abstract ``ReasoningEngine`` and the minimal
``ReasoningRequest``/``ReasoningResponse`` round-trip. Phase 5 adds a *structured*
diagnosis context, a strict LLM-result schema, and a deep-diagnosis outcome that
records the deterministic escalation decision, the (validated) reasoning result,
the lifecycle cost, and the safe fallback.

The reasoning result is always expressed with the shared bottleneck vocabulary and
references evidence by stable id — the model never supplies raw measurements.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.bottleneck import Bottleneck
from inferenceos.contracts.candidate import Candidate
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    BackendType,
    BottleneckKind,
    EvidencePolarity,
    EvidenceStrength,
    InferenceStage,
    MeasurementMode,
    MetricQuality,
    ReasoningEngineKind,
    ReasoningStatus,
    ReasoningTaskType,
    WorkloadKind,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.observation import Observation
from inferenceos.contracts.workload import WorkloadProfile

REASONING_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Structured deep-diagnosis context                                            #
# --------------------------------------------------------------------------- #
class ReasoningModelContext(ContractBase):
    model_ref: str
    architecture: str
    quantization: str = "unknown"
    format: str = "unknown"
    n_layers: int | None = None
    n_attention_heads: int | None = None
    n_kv_heads: int | None = None
    context_length: int | None = None
    parameter_count: int | None = None


class ReasoningBackendContext(ContractBase):
    backend_name: str | None = None
    backend_version: str | None = None
    backend_type: BackendType | None = None
    device: str | None = None
    gpu_layers_offloaded: int | None = None
    gpu_layers_total: int | None = None
    offload_configured: bool = False


class ReasoningWorkloadContext(ContractBase):
    workload_id: str
    kind: WorkloadKind = WorkloadKind.UNKNOWN
    objective: str = ""
    mode: MeasurementMode = MeasurementMode.WARM
    stage: InferenceStage = InferenceStage.END_TO_END
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    summary: str = ""


class ReasoningMetric(ContractBase):
    metric: str
    value: float | None = None
    unit: str
    quality: MetricQuality = MetricQuality.MEASURED
    description: str | None = None


class ReasoningEvidence(ContractBase):
    """A single grounded observation the model may reference by ``evidence_id``."""

    evidence_id: str
    metric: str
    observed_value: float
    reference_value: float | None = None
    polarity: EvidencePolarity = EvidencePolarity.NEUTRAL
    stage: InferenceStage | None = None
    source: str | None = None
    description: str


class ReasoningDiagnosisContext(ContractBase):
    source_diagnosis_id: str | None = None
    source_report_path: str | None = None
    hardware: HardwareProfile
    model: ReasoningModelContext
    backend: ReasoningBackendContext
    workload: ReasoningWorkloadContext
    metrics: list[ReasoningMetric] = Field(default_factory=list)
    evidence: list[ReasoningEvidence] = Field(default_factory=list)
    deterministic: BottleneckDiagnosis
    unavailable_metrics: list[str] = Field(default_factory=list)
    variance: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Strict LLM result                                                            #
# --------------------------------------------------------------------------- #
class ReasoningResult(ContractBase):
    """Schema the local model must produce (validated, evidence-grounded)."""

    schema_version: str = REASONING_SCHEMA_VERSION
    root_cause_class: BottleneckKind
    affected_stage: InferenceStage | None = None
    contributing_factors: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    uncertainty: str = ""
    further_evidence_required: bool = False
    summary: str = ""
    reasoning: str = ""


# --------------------------------------------------------------------------- #
# Request / response                                                           #
# --------------------------------------------------------------------------- #
class ReasoningRequest(ContractBase):
    request_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    task_type: ReasoningTaskType = ReasoningTaskType.ROOT_CAUSE_DIAGNOSIS
    reason: str
    objective: str
    # Phase 0 identity fields (kept; optional).
    hardware: HardwareProfile | None = None
    model: ModelProfile | None = None
    workload: WorkloadProfile | None = None
    observations: list[Observation] = Field(default_factory=list)
    deterministic_findings: list[Bottleneck] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    output_schema_hint: str | None = None
    # Phase 5 deep-diagnosis additions.
    provider: str | None = None
    model_ref: str | None = None
    escalation_reason: str | None = None
    context: ReasoningDiagnosisContext | None = None


class ReasoningResponse(ContractBase):
    request_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    status: ReasoningStatus = ReasoningStatus.OK
    engine: ReasoningEngineKind = ReasoningEngineKind.STUB
    provider: str | None = None
    model_used: str | None = None
    model_version: str | None = None
    conclusion: str = ""
    bottleneck: Bottleneck | None = None
    recommended_candidates: list[Candidate] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    duration_s: float | None = None
    # Phase 5 deep-diagnosis additions.
    task_type: ReasoningTaskType = ReasoningTaskType.ROOT_CAUSE_DIAGNOSIS
    stage: InferenceStage | None = None
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    supporting_evidence_refs: list[str] = Field(default_factory=list)
    contradicting_evidence_refs: list[str] = Field(default_factory=list)
    uncertainty: str | None = None
    further_evidence_required: bool = False
    result: ReasoningResult | None = None
    validation_ok: bool = False
    failure: str | None = None
    raw_output: str | None = None
    escalation_reason: str | None = None


# --------------------------------------------------------------------------- #
# Deep-diagnosis outcome / lifecycle                                           #
# --------------------------------------------------------------------------- #
class EscalationDecision(ContractBase):
    """Deterministic (inspectable) decision about whether to invoke the LLM."""

    should_escalate: bool
    reason_code: str
    rationale: str
    deterministic_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate_signals: int = Field(default=0, ge=0)
    competing_families: int = Field(default=0, ge=0)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class ReasoningLifecycle(ContractBase):
    """Observable resource/time cost of the (non-resident) reasoning mode."""

    reasoning_performed: bool
    provider: str | None = None
    model: str | None = None
    target_state: str = "unknown"
    target_released: bool = False
    reasoner_load_s: float | None = None
    reasoning_latency_s: float | None = None
    reasoner_cleanup_ok: bool = True
    target_resume_verified: bool | None = None
    peak_reasoner_rss_bytes: int | None = None
    notes: list[str] = Field(default_factory=list)


class DeepDiagnosisOutcome(ContractBase):
    schema_version: str = REASONING_SCHEMA_VERSION
    outcome_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    source_diagnosis_id: str | None = None
    source_report_path: str | None = None
    escalation: EscalationDecision
    deterministic: BottleneckDiagnosis
    reasoning: ReasoningResponse | None = None
    lifecycle: ReasoningLifecycle
    fallback_used: bool = False
    final_root_cause: BottleneckKind | None = None
    final_stage: InferenceStage | None = None
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    final_evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    uncertainty: str | None = None
    limitations: list[str] = Field(default_factory=list)
    summary: str = ""

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "DeepDiagnosisOutcome":
        return cls.model_validate_json(data)
