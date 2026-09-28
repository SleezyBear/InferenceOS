"""Phase 12C economics tests: pre-experiment permission, candidate-specific cost,
post-measurement retention, and GPU compatibility.

The architecture distinguishes:
* pre-experiment **permission** (cheap/safe/evidence-backed enough to attempt), which
  never fabricates a performance magnitude; and
* post-measurement **retention** economics, which consumes the actual Phase 7 measured
  throughput improvement and the mode's minimum worthwhile improvement.
"""

from __future__ import annotations

from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    AnomalyClass,
    BottleneckKind,
    EvidenceStrength,
    OptimizationMode,
    PerformanceState,
)
from inferenceos.contracts.experiment import ExperimentComparison, ExperimentResult, MeasurementDelta
from inferenceos.contracts.optimization import BenefitEstimate, OptimizationCandidate
from inferenceos.contracts.preference import OptimizationPreference
from inferenceos.contracts.runtime import (
    InterventionEconomics,
    RuntimeOpportunity,
    RuntimePolicy,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.preference import build_optimization_preference
from inferenceos.runtime.economics import (
    evaluate_intervention_economics,
    evaluate_retention_economics,
    measured_throughput_improvement,
)
from inferenceos.runtime.policy import build_runtime_policy
from inferenceos.selftest import sample_experiment_result
from tests.optimization_helpers import make_candidate, synthetic_context


def _opportunity(deviation: float = 0.9, confidence: float = 0.85) -> RuntimeOpportunity:
    return RuntimeOpportunity(
        opportunity_id="opp",
        anomaly=AnomalyClass.PERSISTENT,
        condition_key="k",
        kind="cpu",
        persistence_samples=3,
        deviation_score=deviation,
        confidence=confidence,
        strength=EvidenceStrength.MEDIUM,
        meaningful=True,
    )


def _preference(mode: str) -> OptimizationPreference:
    return build_optimization_preference(Settings(optimization_mode=mode))


def _policy() -> RuntimePolicy:
    return build_runtime_policy(Settings())


def _unknown_candidate(cost: float) -> OptimizationCandidate:
    ctx = synthetic_context(BottleneckKind.CPU)
    return make_candidate(ctx, action_id="runtime.reduce_threads", known=False, cost=cost, overhead=0.0)


def _known_throughput_candidate(improvement: float, *, confidence: float = 0.85, cost: float = 5.0) -> OptimizationCandidate:
    ctx = synthetic_context(BottleneckKind.GPU_UNDERUTILIZATION)
    candidate = make_candidate(
        ctx, action_id="runtime.increase_gpu_offload_layers", cost=cost, overhead=0.0
    )
    return candidate.model_copy(
        update={
            "benefit": BenefitEstimate(
                metric="decode_tokens_per_sec",
                direction="increase",
                unit="tokens/sec",
                known=True,
                magnitude_low=improvement,
                magnitude_high=improvement,
                confidence=confidence,
            )
        }
    )


def _pre(candidate: OptimizationCandidate, mode: str, *, overhead_s: float = 0.0) -> InterventionEconomics:
    return evaluate_intervention_economics(
        _opportunity(),
        candidate=candidate,
        action=build_default_registry().get(candidate.action_id),
        policy=_policy(),
        measured_overhead_s=overhead_s,
        optimization_preference=_preference(mode),
        performance_state=PerformanceState.NORMAL,
    )


# --------------------------------------------------------------------------- #
# A. PRE-ECONOMICS (permission)                                                 #
# --------------------------------------------------------------------------- #
def test_unknown_benefit_cheap_candidate_is_permitted() -> None:
    candidate = _unknown_candidate(cost=3.0)
    economics = _pre(candidate, "balanced")
    assert economics.beneficial is True
    assert economics.measured_improvement is None


def test_unknown_benefit_expensive_candidate_is_rejected() -> None:
    candidate = _unknown_candidate(cost=500.0)
    economics = _pre(candidate, "balanced")
    assert economics.beneficial is False


def test_minimize_is_more_conservative_than_balanced() -> None:
    candidate = _unknown_candidate(cost=45.0)  # between MINIMIZE (30) and BALANCED (60)
    assert _pre(candidate, "minimize_optimization_cost").beneficial is False
    assert _pre(candidate, "balanced").beneficial is True


def test_performance_first_is_more_permissive_than_balanced() -> None:
    candidate = _unknown_candidate(cost=90.0)  # between BALANCED (60) and PERFORMANCE_FIRST (120)
    assert _pre(candidate, "balanced").beneficial is False
    assert _pre(candidate, "performance_first").beneficial is True


def test_pre_gate_does_not_fabricate_a_benefit() -> None:
    candidate = _unknown_candidate(cost=3.0)
    economics = _pre(candidate, "balanced")
    assert candidate.benefit.known is False
    assert economics.measured_improvement is None
    assert candidate.benefit.magnitude_low is None
    assert candidate.benefit.magnitude_high is None


# --------------------------------------------------------------------------- #
# B. CANDIDATE-SPECIFIC COST                                                    #
# --------------------------------------------------------------------------- #
def test_candidate_specific_cost_is_used() -> None:
    economics = _pre(_unknown_candidate(cost=12.0), "balanced")
    # The candidate's CostEstimate components are used, not the generic 30 s constant.
    assert economics.costs["execution"] == 12.0
    assert economics.costs.get("experiment", 0.0) == 0.0
    assert economics.costs["disruption"] == _policy().cost_model.disruption_cost_s


def test_cost_is_not_double_counted() -> None:
    economics = _pre(_unknown_candidate(cost=3.0), "balanced", overhead_s=5.0)
    assert economics.costs["disruption"] == _policy().cost_model.disruption_cost_s + 5.0
    assert sum(economics.costs.values()) == economics.cost_score * _policy().cost_model.benefit_scale_s


def test_gpu_search_overhead_is_accounted() -> None:
    cheap = _pre(_known_throughput_candidate(0.3), "balanced", overhead_s=0.0)
    expensive = _pre(_known_throughput_candidate(0.3), "balanced", overhead_s=200.0)
    assert expensive.costs["disruption"] == _policy().cost_model.disruption_cost_s + 200.0
    assert expensive.cost_score > cheap.cost_score


# --------------------------------------------------------------------------- #
# C. POST-ECONOMICS (retention)                                                 #
# --------------------------------------------------------------------------- #
def _result_with_throughput(improvement: float, *, improved: bool = True) -> ExperimentResult:
    result = sample_experiment_result()
    delta = MeasurementDelta(
        metric="decode_tokens_per_sec",
        role="objective",
        direction="higher_is_better",
        baseline_mean=10.0,
        candidate_mean=10.0 * (1.0 + improvement),
        absolute_delta=10.0 * improvement,
        relative_delta=improvement,
        improved=improved,
        significant=improved,
    )
    comparison = ExperimentComparison(
        deltas={"decode_tokens_per_sec": delta},
        bottleneck_metric="process_cpu_percent",
        bottleneck_improved=improved,
    )
    return result.model_copy(update={"comparison": comparison})


def _retention(improvement: float, mode: str = "balanced", *, improved: bool = True) -> InterventionEconomics:
    candidate = _unknown_candidate(cost=3.0)
    return evaluate_retention_economics(
        result=_result_with_throughput(improvement, improved=improved),
        candidate=candidate,
        policy=_policy(),
        optimization_preference=_preference(mode),
        performance_state=PerformanceState.NORMAL,
    )


def test_measured_large_improvement_is_retained() -> None:
    economics = _retention(0.552)
    assert measured_throughput_improvement(_result_with_throughput(0.552)) == 0.552
    assert economics.beneficial is True
    assert economics.measured_improvement == 0.552


def test_measured_improvement_below_mode_minimum_is_not_retained() -> None:
    # BALANCED requires 5%; a measured +2% is technically fine but not worth retaining.
    assert _retention(0.02).beneficial is False
    # PERFORMANCE_FIRST requires 1.5%, so the same +2% is retained.
    assert _retention(0.02, mode="performance_first").beneficial is True


def test_technically_failed_result_cannot_be_rescued_by_economics() -> None:
    economics = _retention(0.552, improved=False)
    assert economics.measured_improvement is None
    assert economics.beneficial is False


# --------------------------------------------------------------------------- #
# D. GPU COMPATIBILITY                                                          #
# --------------------------------------------------------------------------- #
def test_known_gpu_benefit_pre_gate_and_retention_remain_intact() -> None:
    candidate = _known_throughput_candidate(0.30, confidence=0.85, cost=5.0)
    pre = _pre(candidate, "performance_first")
    assert pre.beneficial is True
    assert pre.measured_improvement == 0.30
    retention = evaluate_retention_economics(
        result=_result_with_throughput(0.30),
        candidate=candidate,
        policy=_policy(),
        optimization_preference=_preference("performance_first"),
        performance_state=PerformanceState.NORMAL,
    )
    assert retention.beneficial is True


def test_known_gpu_benefit_still_respects_mode_minimum() -> None:
    candidate = _known_throughput_candidate(0.03, confidence=0.85, cost=5.0)
    assert _pre(candidate, "balanced").beneficial is False  # 5% required
    assert _pre(candidate, "performance_first").beneficial is True  # 1.5% required


# --------------------------------------------------------------------------- #
# E. TWO-STAGE LIFECYCLE                                                        #
# --------------------------------------------------------------------------- #
def test_cheap_unknown_permitted_then_retained_on_measured_evidence() -> None:
    candidate = _unknown_candidate(cost=3.0)
    # 1. Pre-experiment: permitted to attempt (unknown benefit, cheap).
    pre = _pre(candidate, "balanced")
    assert pre.beneficial is True and pre.measured_improvement is None
    # 2. Experiment measured +55.2%; post-economics retains only because it is measured.
    retention = _retention(0.552)
    assert retention.beneficial is True and retention.measured_improvement == 0.552
    # 3. A technically-valid but economically trivial result would restore the baseline.
    assert _retention(0.005).beneficial is False

