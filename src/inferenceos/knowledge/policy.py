"""Explicit, inspectable Phase 8 knowledge/applicability/retention policy."""

from __future__ import annotations

from pydantic import Field

from inferenceos.config import Settings
from inferenceos.contracts.base import ContractBase


class MatchWeights(ContractBase):
    """Per-dimension contribution to the explainable applicability score."""

    hardware: float = 0.20
    backend: float = 0.15
    model: float = 0.25
    quantization: float = 0.10
    workload: float = 0.10
    runtime: float = 0.10
    diagnosis: float = 0.05
    os: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "hardware": self.hardware,
            "backend": self.backend,
            "model": self.model,
            "quantization": self.quantization,
            "workload": self.workload,
            "runtime": self.runtime,
            "diagnosis": self.diagnosis,
            "os": self.os,
        }


class KnowledgePolicy(ContractBase):
    lookup_limit: int = Field(default=50, ge=1)
    verify_integrity: bool = True
    retention_days: float = Field(default=0.0, ge=0.0)  # 0 = keep forever
    revalidation_max_age_days: float = Field(default=30.0, ge=0.0)
    condition_tolerance: float = Field(default=0.25, ge=0.0)
    not_applicable_critical_threshold: int = Field(default=2, ge=1)
    db_path: str = ""
    weights: MatchWeights = Field(default_factory=MatchWeights)

    def as_dict(self) -> dict[str, object]:
        return {
            "lookup_limit": self.lookup_limit,
            "verify_integrity": self.verify_integrity,
            "retention_days": self.retention_days,
            "revalidation_max_age_days": self.revalidation_max_age_days,
            "condition_tolerance": self.condition_tolerance,
            "not_applicable_critical_threshold": self.not_applicable_critical_threshold,
            "weights": self.weights.as_dict(),
        }


def build_knowledge_policy(settings: Settings) -> KnowledgePolicy:
    return KnowledgePolicy(
        lookup_limit=settings.phase8_lookup_limit,
        verify_integrity=settings.phase8_verify_integrity,
        retention_days=settings.phase8_retention_days,
        revalidation_max_age_days=settings.phase8_revalidation_max_age_days,
        condition_tolerance=settings.phase8_condition_tolerance,
        not_applicable_critical_threshold=settings.phase8_not_applicable_critical_threshold,
        db_path=str(settings.phase8_db_path),
    )
