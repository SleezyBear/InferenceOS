"""Bottleneck contracts (Phase 4 deterministic detection, Phase 5 reasoning).

The contract deliberately separates raw observation from derived interpretation:
each :class:`EvidenceItem` records a metric, its observed value, an optional
reference, and whether it supports or contradicts the finding. A :class:`Bottleneck`
carries both supporting and contradicting evidence plus the deterministic rationale,
so a human or a later reasoning component can inspect the basis of the conclusion.
"""

from __future__ import annotations

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import (
    BottleneckKind,
    CandidateKind,
    EvidencePolarity,
    EvidenceStrength,
    InferenceStage,
)


class EvidenceItem(ContractBase):
    metric: str
    observed_value: float
    reference_value: float | None = None
    ratio: float | None = None
    description: str
    polarity: EvidencePolarity = EvidencePolarity.NEUTRAL
    stage: InferenceStage | None = None
    strength: EvidenceStrength = EvidenceStrength.NONE
    source: str | None = None


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

    # Phase 4 deterministic-diagnosis additions (all optional/defaulted).
    stage: InferenceStage | None = None
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: list[EvidenceItem] = Field(default_factory=list)
    unavailable_measurements: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    detector_version: str = "unknown"
