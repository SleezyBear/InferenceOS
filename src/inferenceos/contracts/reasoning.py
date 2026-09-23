"""Reasoning contracts (Phase 5 deep-diagnosis mode).

A ``ReasoningRequest`` is emitted when deterministic confidence is below threshold;
a ``ReasoningResponse`` must be structured (schema-valid), never free text.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.bottleneck import Bottleneck
from inferenceos.contracts.candidate import Candidate
from inferenceos.contracts.enums import ReasoningEngineKind, ReasoningStatus
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.observation import Observation
from inferenceos.contracts.workload import WorkloadProfile


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ReasoningRequest(ContractBase):
    request_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    reason: str
    objective: str
    hardware: HardwareProfile | None = None
    model: ModelProfile | None = None
    workload: WorkloadProfile | None = None
    observations: list[Observation] = Field(default_factory=list)
    deterministic_findings: list[Bottleneck] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    output_schema_hint: str | None = None


class ReasoningResponse(ContractBase):
    request_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    status: ReasoningStatus = ReasoningStatus.OK
    engine: ReasoningEngineKind = ReasoningEngineKind.STUB
    model_used: str | None = None
    conclusion: str = ""
    bottleneck: Bottleneck | None = None
    recommended_candidates: list[Candidate] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    duration_s: float | None = None
