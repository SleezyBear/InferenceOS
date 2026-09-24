"""Ranking accounts for benefit, cost, risk and optimizer overhead (not speedup alone)."""

from __future__ import annotations

from inferenceos.contracts.enums import BottleneckKind, RiskLevel
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.ranking import rank_candidate
from tests.optimization_helpers import make_candidate, synthetic_context


def _policy() -> OptimizationPolicy:
    return OptimizationPolicy()


def test_ranking_components_are_decomposed() -> None:
    ctx = synthetic_context(BottleneckKind.PAGING)
    factors = rank_candidate(make_candidate(ctx), ctx, _policy())
    assert factors.benefit_score > 0.0
    assert factors.cost_score > 0.0
    assert factors.risk_score > 0.0
    assert factors.overhead_score > 0.0
    assert factors.weights
    assert factors.explanation
    assert any("cost=" in line for line in factors.explanation)
    assert any("risk=" in line for line in factors.explanation)
    assert any("overhead=" in line for line in factors.explanation)


def test_higher_cost_ranks_lower() -> None:
    ctx = synthetic_context(BottleneckKind.PAGING)
    policy = _policy()
    cheap = rank_candidate(make_candidate(ctx, cost=1.0), ctx, policy)
    expensive = rank_candidate(make_candidate(ctx, cost=100.0, overhead=1.0), ctx, policy)
    assert cheap.composite_score > expensive.composite_score
    assert cheap.cost_score > expensive.cost_score
    assert cheap.overhead_score > expensive.overhead_score


def test_higher_risk_ranks_lower() -> None:
    ctx = synthetic_context(BottleneckKind.PAGING)
    policy = _policy()
    low = rank_candidate(make_candidate(ctx, risk_level=RiskLevel.LOW), ctx, policy)
    high = rank_candidate(make_candidate(ctx, risk_level=RiskLevel.HIGH), ctx, policy)
    assert low.composite_score > high.composite_score
    assert low.risk_score > high.risk_score


def test_known_benefit_ranks_above_unknown_benefit() -> None:
    ctx = synthetic_context(BottleneckKind.PAGING)
    policy = _policy()
    known = rank_candidate(make_candidate(ctx, known=True, magnitude=8 * 1024**3), ctx, policy)
    unknown = rank_candidate(make_candidate(ctx, known=False), ctx, policy)
    assert known.composite_score > unknown.composite_score
    assert known.benefit_score > unknown.benefit_score


def test_expected_value_is_benefit_over_cost() -> None:
    ctx = synthetic_context(BottleneckKind.PAGING)
    policy = _policy()
    candidate = make_candidate(ctx, cost=4.0)
    factors = rank_candidate(candidate, ctx, policy)
    expected = (factors.benefit_score * factors.confidence_score) / max(
        candidate.cost.total_cost_s, 0.1
    )
    assert abs(factors.expected_value - expected) < 1e-9
