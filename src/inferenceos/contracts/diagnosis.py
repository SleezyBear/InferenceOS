"""Deterministic diagnosis contracts (Phase 4).

A :class:`BottleneckDiagnosis` is the durable output of the deterministic detector:
it carries the source-report identity, the hardware/model/backend/workload context,
the meaningful-opportunity decision, the primary finding (if any), candidate and
observation-only findings, and the full evidence/uncertainty basis.

It contains no optimization recommendations — later phases decide actions.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.enums import EvidenceStrength
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.profiling import ProfilingConfig
from inferenceos.contracts.validity import MeasurementValidityAssessment

DIAGNOSIS_SCHEMA_VERSION = "1.0"


class DiagnosisContext(ContractBase):
    source_report_path: str
    source_report_generated_at: datetime | None = None
    profiler_version: str | None = None
    detector_version: str
    hardware: HardwareProfile
    backend: ProfilingConfig
    model_refs: list[str] = Field(default_factory=list)
    workload_ids: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)


class BottleneckDiagnosis(ContractBase):
    diagnosis_id: str
    created_at: datetime
    schema_version: str = DIAGNOSIS_SCHEMA_VERSION
    context: DiagnosisContext

    meaningful_opportunity: bool
    insufficient_evidence: bool
    escalation_required: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE

    primary: Bottleneck | None = None
    candidates: list[Bottleneck] = Field(default_factory=list)
    observations: list[Bottleneck] = Field(default_factory=list)

    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = Field(default_factory=list)
    unavailable_measurements: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    no_opportunity_reason: str | None = None
    summary: str = ""

    # B1: whether the underlying measurements can be trusted as an optimization
    # signal (power/thermal/frequency confound assessment).
    measurement_validity: MeasurementValidityAssessment | None = None

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "BottleneckDiagnosis":
        return cls.model_validate_json(data)
