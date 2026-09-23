"""Candidate intervention contracts (Phase 6 generation, Phase 7 experiments)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import CandidateKind, CandidateStatus


class Candidate(ContractBase):
    candidate_id: str
    kind: CandidateKind
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_benefit: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    cost: float = Field(ge=0.0)
    expected_value: float
    status: CandidateStatus = CandidateStatus.PROPOSED
    rationale: str = ""
    source_bottleneck_id: str | None = None
