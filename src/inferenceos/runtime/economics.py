"""Explicit intervention economics for conservative runtime adaptation.

The runtime monitor never decides "bottleneck confidence > X -> intervene". It asks
whether the expected benefit is worth the *disruption* of interrupting the user's
active workload: reasoning load/run, checkpoint, pause, experiment, reload, and
expected disruption are first-class costs. All quantities are normalized/relative;
no fake monetary value is invented.
"""

from __future__ import annotations

from inferenceos.contracts.enums import (
    InterventionClass,
    PrivilegeRequirement,
    Reversibility,
    RiskLevel,
)
from inferenceos.contracts.optimization import CandidateAction, OptimizationCandidate
from inferenceos.contracts.runtime import InterventionEconomics, RuntimeOpportunity, RuntimePolicy
from inferenceos.optimization.safety import strength_rank


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def classify_action(action: CandidateAction, policy: RuntimePolicy) -> InterventionClass:
    """Classify how disruptive an action is under active-runtime policy."""
    broad = action.scope.value in {"external_process", "system_wide"}
    elevated = action.privilege is PrivilegeRequirement.ELEVATION_REQUIRED
    if broad or elevated:
        return InterventionClass.EXTERNAL_SYSTEM_WIDE
    disruptive = (
        action.requires_human_approval
        or action.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        or action.reversibility is not Reversibility.REVERSIBLE
    )
    if disruptive:
        return InterventionClass.DISRUPTIVE_ADAPTATION
    return InterventionClass.NON_DISRUPTIVE_ADAPTATION


def evaluate_intervention_economics(
    opportunity: RuntimeOpportunity,
    *,
    candidate: OptimizationCandidate | None,
    action: CandidateAction | None,
    policy: RuntimePolicy,
    reasoning_required: bool = False,
    measured_overhead_s: float = 0.0,
) -> InterventionEconomics:
    cost = policy.cost_model
    persistence_required = max(policy.persistence_samples, 1)
    persistence_score = _clamp(opportunity.persistence_samples / persistence_required)

    strength_score = strength_rank(opportunity.strength) / 3.0
    evidence_score = _clamp(
        0.5 * opportunity.confidence
        + 0.3 * strength_score
        + 0.2 * persistence_score
    )

    deviation = _clamp(opportunity.deviation_score)
    candidate_confidence = (
        candidate.benefit.confidence if candidate is not None and candidate.benefit.known else 0.0
    )
    expected_benefit_score = _clamp(
        0.6 * deviation + 0.4 * candidate_confidence
        if candidate is not None
        else 0.5 * deviation
    )

    cost_terms: dict[str, float] = {
        "detection": cost.detection_cost_s,
        "diagnosis": cost.diagnosis_cost_s,
        "checkpoint": cost.checkpoint_cost_s,
        "pause": cost.pause_cost_s,
        "experiment": cost.experiment_cost_s,
        "disruption": cost.disruption_cost_s + measured_overhead_s,
    }
    if reasoning_required:
        cost_terms["reasoning"] = cost.reasoning_load_cost_s + cost.reasoning_run_cost_s
    if action is not None and candidate is not None:
        cost_terms["reload"] = cost.reload_cost_s
    total_cost_s = sum(cost_terms.values())
    cost_score = total_cost_s / cost.benefit_scale_s

    net_value = (
        cost.weight_benefit * expected_benefit_score * evidence_score
        - cost.weight_cost * cost_score
    )

    intervention_class = (
        classify_action(action, policy)
        if action is not None
        else (InterventionClass.DISRUPTIVE_ADAPTATION if reasoning_required else InterventionClass.NONE)
    )
    if intervention_class is InterventionClass.EXTERNAL_SYSTEM_WIDE:
        requires_approval = True
    elif intervention_class is InterventionClass.DISRUPTIVE_ADAPTATION:
        requires_approval = policy.require_approval_for_disruptive
    elif intervention_class is InterventionClass.NON_DISRUPTIVE_ADAPTATION:
        requires_approval = not policy.allow_non_disruptive_auto_adaptation
    else:
        requires_approval = False

    beneficial = net_value >= cost.min_net_value and evidence_score > 0.0

    rationale = [
        f"evidence_score={evidence_score:.3f} (confidence={opportunity.confidence:.2f}, "
        f"strength={opportunity.strength.value}, persistence={opportunity.persistence_samples})",
        f"expected_benefit_score={expected_benefit_score:.3f} "
        f"(deviation={deviation:.3f}, candidate_confidence={candidate_confidence:.2f})",
        f"cost_score={cost_score:.3f} (total {total_cost_s:.2f}s over "
        f"benefit_scale {cost.benefit_scale_s:.0f}s)",
        f"net_value={net_value:.4f} (weights benefit={cost.weight_benefit}, cost={cost.weight_cost})",
        f"intervention_class={intervention_class.value} requires_approval={requires_approval}",
        f"beneficial={beneficial} (min_net_value={cost.min_net_value})",
    ]

    return InterventionEconomics(
        expected_benefit_score=expected_benefit_score,
        evidence_score=evidence_score,
        persistence_score=persistence_score,
        cost_score=cost_score,
        net_value=net_value,
        beneficial=beneficial,
        requires_approval=requires_approval,
        intervention_class=intervention_class,
        costs=cost_terms,
        rationale=rationale,
    )
