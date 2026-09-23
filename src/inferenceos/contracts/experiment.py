"""Experiment result contracts (Phase 7 real before/after + quality validation)."""

from __future__ import annotations

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import Decision


class QualityScore(ContractBase):
    metric: str
    value: float
    higher_is_better: bool = True
    tolerance: float | None = None


class ExperimentResult(ContractBase):
    experiment_id: str
    candidate_id: str
    baseline_observation_id: str | None = None
    result_observation_id: str | None = None
    performance_delta: dict[str, float] = Field(default_factory=dict)
    quality_before: list[QualityScore] = Field(default_factory=list)
    quality_after: list[QualityScore] = Field(default_factory=list)
    tolerance_band: float | None = None
    decision: Decision = Decision.PENDING
    notes: str | None = None
