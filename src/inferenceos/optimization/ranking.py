"""Transparent, multi-factor candidate ranking.

Ranking is not "largest expected speedup wins". Every contributing factor — benefit,
benefit uncertainty, execution cost, optimizer overhead, risk, reversibility,
diagnosis confidence, causal strength, scope, approval burden, precondition certainty —
is preserved and explained, so a human can inspect why candidate A outranked B.
"""

from __future__ import annotations

from inferenceos.contracts.enums import RiskLevel
from inferenceos.contracts.optimization import OptimizationCandidate, RankingFactors
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.estimate import benefit_score
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.safety import (
    REVERSIBILITY_SCORE,
    risk_rank,
    scope_score,
    strength_rank,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def rank_candidate(
    candidate: OptimizationCandidate,
    ctx: OptimizationContext,
    policy: OptimizationPolicy,
) -> RankingFactors:
    benefit = _clamp(benefit_score(candidate.benefit, ctx))
    uncertainty = _clamp(candidate.benefit.confidence)
    cost = _clamp(1.0 / (1.0 + candidate.cost.total_cost_s))
    overhead = _clamp(1.0 / (1.0 + candidate.cost.optimizer_overhead_s * 10.0))
    risk = _clamp(1.0 - risk_rank(candidate.risk.risk_level) / risk_rank(RiskLevel.CRITICAL))
    reversibility = _clamp(REVERSIBILITY_SCORE.get(candidate.risk.reversibility, 0.0))
    confidence = _clamp(candidate.diagnosis_confidence)
    causal_strength = _clamp(strength_rank(candidate.diagnosis_strength) / 3.0)
    scope = _clamp(scope_score(candidate.risk.scope))
    approval = 0.2 if candidate.risk.requires_approval else 1.0
    precondition = _precondition_score(candidate)

    expected_value = (benefit * confidence) / max(candidate.cost.total_cost_s, 0.1)

    weights = policy.weights.as_dict()
    factors = {
        "benefit": benefit,
        "uncertainty": uncertainty,
        "cost": cost,
        "overhead": overhead,
        "risk": risk,
        "reversibility": reversibility,
        "confidence": confidence,
        "causal_strength": causal_strength,
        "scope": scope,
        "approval": approval,
        "precondition": precondition,
    }
    composite = sum(weights[name] * score for name, score in factors.items())

    explanation = [
        f"benefit={benefit:.3f} (known={candidate.benefit.known}, "
        f"metric={candidate.benefit.metric}, {candidate.benefit.direction})",
        f"uncertainty={uncertainty:.3f} (benefit-estimate confidence)",
        f"cost={cost:.3f} (total estimated {candidate.cost.total_cost_s:.2f}s)",
        f"overhead={overhead:.3f} (optimizer {candidate.cost.optimizer_overhead_s:.3f}s)",
        f"risk={risk:.3f} ({candidate.risk.risk_level.value})",
        f"reversibility={reversibility:.3f} ({candidate.risk.reversibility.value})",
        f"confidence={confidence:.3f} (diagnosis)",
        f"causal_strength={causal_strength:.3f} ({candidate.diagnosis_strength.value})",
        f"scope={scope:.3f} ({candidate.risk.scope.value})",
        f"approval={approval:.3f} (requires_approval={candidate.risk.requires_approval})",
        f"precondition={precondition:.3f} (satisfied required preconditions)",
        f"expected_value=(benefit*confidence)/cost={expected_value:.4f}",
        f"composite={composite:.4f} over weights {weights}",
    ]
    return RankingFactors(
        benefit_score=benefit,
        uncertainty_score=uncertainty,
        cost_score=cost,
        overhead_score=overhead,
        risk_score=risk,
        reversibility_score=reversibility,
        confidence_score=confidence,
        causal_strength_score=causal_strength,
        scope_score=scope,
        approval_score=approval,
        precondition_score=precondition,
        expected_value=expected_value,
        composite_score=composite,
        weights=weights,
        explanation=explanation,
    )


def _precondition_score(candidate: OptimizationCandidate) -> float:
    required = [item for item in candidate.preconditions if item.required]
    if not required:
        return 1.0
    total = 0.0
    for item in required:
        if item.satisfied is True:
            total += 1.0
        elif item.satisfied is None:
            total += 0.5
    return _clamp(total / len(required))
