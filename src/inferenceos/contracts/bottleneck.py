"""Bottleneck contracts (Phase 4 deterministic detection, Phase 5 reasoning)."""

from __future__ import annotations

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import BottleneckKind, CandidateKind


class EvidenceItem(ContractBase):
    metric: str
    observed_value: float
    reference_value: float | None = None
    ratio: float | None = None
    description: str


class Bottleneck(ContractBase):
    bottleneck_id: str
    kind: BottleneckKind
    confidence: float = Field(ge=0.0, le=1.0)
    is_meaningful: bool
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    contributing_factors: list[str] = Field(default_factory=list)
    recommended_surface: list[CandidateKind] = Field(default_factory=list)
    source: str = "unknown"
