"""Explicit, inspectable Phase 6 optimization policy.

Safety-critical behaviour (autonomous risk ceiling, approval policy, evidence
thresholds, optimizer-overhead ceiling) lives here as named, validated fields rather
than as magic constants inside the generator. Ranking weights are configurable and
decomposed so a human can see why candidate A outranked candidate B.
"""

from __future__ import annotations

from pydantic import Field

from inferenceos.config import Settings
from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import EvidenceStrength, RiskLevel
from inferenceos.optimization.search.policy import SearchPolicy, build_search_policy

APPROVAL_POLICY = "gated_never_auto_approved"


class RankingWeights(ContractBase):
    benefit: float = 0.30
    uncertainty: float = 0.08
    cost: float = 0.15
    overhead: float = 0.05
    risk: float = 0.15
    reversibility: float = 0.07
    confidence: float = 0.08
    causal_strength: float = 0.06
    scope: float = 0.03
    approval: float = 0.02
    precondition: float = 0.01

    def as_dict(self) -> dict[str, float]:
        return {
            "benefit": self.benefit,
            "uncertainty": self.uncertainty,
            "cost": self.cost,
            "overhead": self.overhead,
            "risk": self.risk,
            "reversibility": self.reversibility,
            "confidence": self.confidence,
            "causal_strength": self.causal_strength,
            "scope": self.scope,
            "approval": self.approval,
            "precondition": self.precondition,
        }


class OptimizationPolicy(ContractBase):
    max_candidates: int = Field(default=8, ge=1)
    min_diagnosis_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    min_evidence_strength: EvidenceStrength = EvidenceStrength.LOW
    context_reduction_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    min_context_length: int = Field(default=512, ge=1)
    autonomous_max_risk: RiskLevel = RiskLevel.MEDIUM
    allow_reasoning: bool = True
    reasoning_only_when_insufficient: bool = True
    max_optimizer_overhead_s: float = Field(default=120.0, gt=0.0)
    experiment_repetitions: int = Field(default=4, ge=1)
    approval_policy: str = APPROVAL_POLICY
    weights: RankingWeights = Field(default_factory=RankingWeights)
    search: SearchPolicy = Field(default_factory=SearchPolicy)

    def as_dict(self) -> dict[str, object]:
        return {
            "max_candidates": self.max_candidates,
            "min_diagnosis_confidence": self.min_diagnosis_confidence,
            "min_evidence_strength": self.min_evidence_strength.value,
            "context_reduction_fraction": self.context_reduction_fraction,
            "min_context_length": self.min_context_length,
            "autonomous_max_risk": self.autonomous_max_risk.value,
            "allow_reasoning": self.allow_reasoning,
            "reasoning_only_when_insufficient": self.reasoning_only_when_insufficient,
            "max_optimizer_overhead_s": self.max_optimizer_overhead_s,
            "experiment_repetitions": self.experiment_repetitions,
            "approval_policy": self.approval_policy,
            "weights": self.weights.as_dict(),
            "search": self.search.as_dict(),
        }


def build_policy(settings: Settings) -> OptimizationPolicy:
    return OptimizationPolicy(
        max_candidates=settings.phase6_max_candidates,
        min_diagnosis_confidence=settings.phase6_min_diagnosis_confidence,
        min_evidence_strength=EvidenceStrength(settings.phase6_min_evidence_strength),
        context_reduction_fraction=settings.phase6_context_reduction_fraction,
        min_context_length=settings.phase6_min_context_length,
        autonomous_max_risk=RiskLevel(settings.phase6_autonomous_max_risk),
        allow_reasoning=settings.phase6_allow_reasoning,
        max_optimizer_overhead_s=settings.phase6_max_optimizer_overhead_s,
        experiment_repetitions=settings.phase6_experiment_repetitions,
        weights=RankingWeights(
            benefit=settings.phase6_weight_benefit,
            uncertainty=settings.phase6_weight_uncertainty,
            cost=settings.phase6_weight_cost,
            overhead=settings.phase6_weight_overhead,
            risk=settings.phase6_weight_risk,
            reversibility=settings.phase6_weight_reversibility,
            confidence=settings.phase6_weight_confidence,
            causal_strength=settings.phase6_weight_causal_strength,
            scope=settings.phase6_weight_scope,
            approval=settings.phase6_weight_approval,
            precondition=settings.phase6_weight_precondition,
        ),
        search=build_search_policy(settings),
    )
