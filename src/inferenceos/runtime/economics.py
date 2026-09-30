"""Explicit intervention economics for conservative runtime adaptation.

The runtime monitor asks two distinct economic questions:

* **Pre-experiment permission** (:func:`evaluate_intervention_economics`): "is this
  candidate cheap/safe/reasonable enough to *attempt*?" It never fabricates a
  performance magnitude. When a candidate's benefit was already measured before
  approval (e.g. the GPU-offload search) the existing value-vs-cost gate applies; when
  the benefit is genuinely unknown until Phase 7 measures it, the gate is an
  affordability/evidence check bounded by the mode's
  ``maximum_experiment_cost_s``.
* **Post-experiment retention** (:func:`evaluate_retention_economics`): "now that the
  intervention was measured, is *keeping* it economically worthwhile?" It consumes the
  actual Phase 7 comparison, applies the mode's minimum worthwhile improvement, and is
  the only place where a measured benefit drives retention economics.

Costs use the candidate-specific :class:`CostEstimate` when available (the estimated
apply/reload cost plus the bounded experiment cost), with the GPU search's *measured*
overhead added once as disruption. All quantities are normalized/relative; no fake
monetary value is invented.
"""

from __future__ import annotations

from inferenceos.contracts.enums import (
    ExperimentDecisionKind,
    InterventionClass,
    OptimizationSurface,
    PerformanceState,
    PrivilegeRequirement,
    Reversibility,
    RiskLevel,
    Situation,
)
from inferenceos.contracts.experiment import ExperimentResult
from inferenceos.contracts.optimization import CandidateAction, OptimizationCandidate
from inferenceos.contracts.preference import OptimizationPreference
from inferenceos.contracts.runtime import (
    InterventionCostModel,
    InterventionEconomics,
    RuntimeOpportunity,
    RuntimePolicy,
)
from inferenceos.experiment.comparison import measured_memory_objective
from inferenceos.optimization.goals import assess_known_gains
from inferenceos.optimization.keep import ModeRules, judge_assessment, judge_keep
from inferenceos.optimization.preference import candidate_measured_improvement
from inferenceos.optimization.safety import strength_rank

#: Throughput metrics that are directly comparable as a relative performance value.
_PERFORMANCE_METRICS: tuple[str, ...] = ("decode_tokens_per_sec", "prompt_tokens_per_sec")


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


def _intervention_cost_terms(
    *,
    candidate: OptimizationCandidate | None,
    action: CandidateAction | None,
    cost: InterventionCostModel,
    reasoning_required: bool,
    measured_overhead_s: float,
) -> dict[str, float]:
    """Cost components for the intervention, without double counting.

    When a candidate carries a real :class:`CostEstimate` its apply/reload/experiment
    components are authoritative; the GPU search's measured overhead enters once as
    ``disruption``. Otherwise the generic policy defaults are used (unchanged legacy
    behavior for candidate-less calls).
    """
    disruption = cost.disruption_cost_s + measured_overhead_s
    if candidate is not None and candidate.cost.total_cost_s > 0.0:
        estimate = candidate.cost
        return {
            "execution": estimate.execution_cost_s,
            "reconfiguration": estimate.reconfiguration_cost_s,
            "reload": estimate.model_reload_cost_s,
            "experiment": estimate.experiment_cost_s,
            "reasoning": estimate.reasoning_cost_s,
            "optimizer_overhead": estimate.optimizer_overhead_s,
            "disruption": disruption,
        }
    terms: dict[str, float] = {
        "detection": cost.detection_cost_s,
        "diagnosis": cost.diagnosis_cost_s,
        "checkpoint": cost.checkpoint_cost_s,
        "pause": cost.pause_cost_s,
        "experiment": cost.experiment_cost_s,
        "disruption": disruption,
    }
    if reasoning_required:
        terms["reasoning"] = cost.reasoning_load_cost_s + cost.reasoning_run_cost_s
    if action is not None and candidate is not None:
        terms["reload"] = cost.reload_cost_s
    return terms


def evaluate_intervention_economics(
    opportunity: RuntimeOpportunity,
    *,
    candidate: OptimizationCandidate | None,
    action: CandidateAction | None,
    policy: RuntimePolicy,
    reasoning_required: bool = False,
    measured_overhead_s: float = 0.0,
    optimization_preference: OptimizationPreference | None = None,
    performance_state: PerformanceState = PerformanceState.UNKNOWN,
    mode_rules: ModeRules | None = None,
    remembered_test_cost_s: float | None = None,
) -> InterventionEconomics:
    """Pre-test permission ("is this worth testing now?").

    With ``mode_rules`` (step 4) a gain already measured before the test (for example
    by the layer search) is judged by the shared keep rule: the goal's measures, the
    effort mode's smallest worthwhile gain, and the payback check against the costs
    *still to come* (applying it, the restart pause, the check-run). Time already spent
    searching is never charged again. ``remembered_test_cost_s`` (the test time measured
    on this machine before) replaces the estimated test cost when known. Without
    ``mode_rules`` the older value-vs-cost rule applies unchanged.
    """
    if mode_rules is not None and optimization_preference is None:
        optimization_preference = mode_rules.effort
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

    cost_terms = _intervention_cost_terms(
        candidate=candidate,
        action=action,
        cost=cost,
        reasoning_required=reasoning_required,
        measured_overhead_s=measured_overhead_s,
    )
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

    # A *known* (already measured) comparable improvement drives the existing value-vs-cost
    # gate and the mode's minimum-improvement requirement. An *unknown* benefit must not
    # be fabricated: the pre-gate instead checks whether the experiment is affordable and
    # evidence-backed, and worth is decided post-measurement.
    measured_improvement = candidate_measured_improvement(candidate)
    required_improvement: float | None = None
    improvement_justified = True
    if optimization_preference is not None:
        required_improvement = optimization_preference.minimum_improvement_for(performance_state)
        if measured_improvement is not None:
            improvement_justified = measured_improvement >= required_improvement

    permission_reason: str
    if measured_improvement is not None:
        permission = net_value >= cost.min_net_value and evidence_score > 0.0
        permission_reason = "known pre-measured benefit: value-vs-cost gate"
    elif optimization_preference is not None:
        budget = float(optimization_preference.maximum_experiment_cost_s)
        permission = evidence_score > 0.0 and total_cost_s <= budget
        permission_reason = (
            f"unknown benefit: affordability/evidence permission "
            f"(estimated {total_cost_s:.2f}s <= mode budget {budget:.2f}s)"
        )
    else:
        permission = net_value >= cost.min_net_value and evidence_score > 0.0
        permission_reason = "no policy supplied: legacy value-vs-cost gate"

    payback = None
    keep_verdict: str | None = None
    goal_gain: float | None = None
    if mode_rules is not None and optimization_preference is not None:
        forward = _forward_cost_terms(candidate, remembered_test_cost_s)
        gains = (
            {candidate.benefit.metric: measured_improvement}
            if measured_improvement is not None and candidate is not None
            else {}
        )
        assessment = assess_known_gains(gains, mode_rules.goal)
        goal_gain = assessment.gain
        if assessment.gain is not None:
            judgement = judge_assessment(
                assessment,
                technically_better=True,
                effort=optimization_preference,
                goal=mode_rules.goal,
                situation=Situation.WHILE_RUNNING,
                settings=mode_rules.settings,
                remaining_cost_terms=forward,
                performance_state=performance_state,
            )
            payback = judgement.payback
            keep_verdict = judgement.verdict.value
            permission = judgement.keep and evidence_score > 0.0
            improvement_justified = assessment.gain >= (required_improvement or 0.0)
            permission_reason = "step 4 keep rule on the gain measured before the test: " + (
                judgement.reasons[-1] if judgement.reasons else ""
            )
        else:
            forward_total = sum(forward.values())
            budget = float(optimization_preference.maximum_experiment_cost_s)
            permission = evidence_score > 0.0 and forward_total <= budget
            permission_reason = (
                "gain unknown until tested: affordability within the effort mode's "
                f"test budget (still to spend {forward_total:.2f}s <= {budget:.2f}s)"
            )

    beneficial = permission and improvement_justified

    rationale = [
        f"evidence_score={evidence_score:.3f} (confidence={opportunity.confidence:.2f}, "
        f"strength={opportunity.strength.value}, persistence={opportunity.persistence_samples})",
        f"expected_benefit_score={expected_benefit_score:.3f} "
        f"(deviation={deviation:.3f}, candidate_confidence={candidate_confidence:.2f})",
        f"cost_score={cost_score:.3f} (total {total_cost_s:.2f}s over "
        f"benefit_scale {cost.benefit_scale_s:.0f}s; components {cost_terms})",
        f"net_value={net_value:.4f} (weights benefit={cost.weight_benefit}, cost={cost.weight_cost})",
        f"intervention_class={intervention_class.value} requires_approval={requires_approval}",
        f"beneficial={beneficial} ({permission_reason}; min_net_value={cost.min_net_value})",
    ]
    if optimization_preference is not None:
        measured_text = (
            "unknown" if measured_improvement is None else f"{measured_improvement:+.2%}"
        )
        required_text = (
            "n/a" if required_improvement is None else f"{required_improvement:.2%}"
        )
        rationale.append(
            f"optimization_mode={optimization_preference.mode.value} "
            f"performance_state={performance_state.value} "
            f"measured_improvement={measured_text} required_improvement={required_text} "
            f"improvement_justified={improvement_justified}"
        )

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
        optimization_mode=(
            optimization_preference.mode.value
            if optimization_preference is not None
            else None
        ),
        performance_state=performance_state.value,
        required_improvement=required_improvement,
        measured_improvement=measured_improvement,
        improvement_justified=improvement_justified,
        goal_mode=mode_rules.goal.mode.value if mode_rules is not None else None,
        goal_gain=goal_gain,
        keep_verdict=keep_verdict,
        payback=payback,
    )


def _forward_cost_terms(
    candidate: OptimizationCandidate | None, remembered_test_cost_s: float | None
) -> dict[str, float]:
    """Costs still to come for testing and keeping a candidate (nothing already spent).

    The reload term is the restart pause the user sees. The test cost is the one
    measured on this machine before, when known, instead of the estimate.
    """
    if candidate is None:
        return {}
    estimate = candidate.cost
    experiment = (
        float(remembered_test_cost_s)
        if remembered_test_cost_s is not None
        else estimate.experiment_cost_s
    )
    return {
        "apply": estimate.execution_cost_s + estimate.reconfiguration_cost_s,
        "restart_pause": estimate.model_reload_cost_s,
        "check_run": experiment,
        "reasoning": estimate.reasoning_cost_s,
        "optimizer": estimate.optimizer_overhead_s,
    }


def measured_throughput_improvement(result: ExperimentResult) -> float | None:
    """The best *measured* comparable throughput improvement from a Phase 7 result.

    Only throughput metrics marked statistically ``improved`` by the existing comparison
    are used; this is real measured evidence, not an estimate. ``None`` when no
    comparable measured improvement exists.
    """
    comparison = result.comparison
    if comparison is None:
        return None
    best: float | None = None
    for metric in _PERFORMANCE_METRICS:
        delta = comparison.deltas.get(metric)
        if delta is None or delta.relative_delta is None or not delta.improved:
            continue
        value = float(delta.relative_delta)
        best = value if best is None else max(best, value)
    return best


def measured_memory_improvement(result: ExperimentResult) -> float | None:
    """The best *measured* memory-objective reduction from a Phase 7 result.

    Only memory metrics marked statistically ``improved`` by the existing comparison are
    used (``peak_rss_bytes``, ``committed_bytes``, ``gpu_memory_used_bytes``); the
    returned magnitude (e.g. ``0.1``) is measured resource evidence, never a claim about
    physical residency or performance. ``None`` when no memory improvement exists.
    """
    comparison = result.comparison
    if comparison is None:
        return None
    return measured_memory_objective(comparison)


def _is_memory_surface(candidate: OptimizationCandidate | None) -> bool:
    return candidate is not None and candidate.surface is OptimizationSurface.MEMORY_MANAGEMENT


def evaluate_retention_economics(
    *,
    result: ExperimentResult,
    candidate: OptimizationCandidate | None,
    policy: RuntimePolicy,
    optimization_preference: OptimizationPreference | None = None,
    performance_state: PerformanceState = PerformanceState.UNKNOWN,
    mode_rules: ModeRules | None = None,
) -> InterventionEconomics:
    """Post-measurement economic retention decision.

    With ``mode_rules`` (step 4) this is the shared keep rule: the goal's measures,
    the effort mode's smallest worthwhile gain, the guarded measures, and the payback
    check against costs still to come (none: the change is already in place, and the
    test time is already spent). Without ``mode_rules`` the older rule below applies.

    Technical improvement is necessary but not sufficient: this consumes the *actual*
    Phase 7 measured outcome and the mode's minimum worthwhile improvement, and confirms
    the retained change's candidate-specific cost is within the mode's affordability
    budget.

    The relevant objective depends on the candidate surface. Performance-surfaced
    changes (and everything that is not a memory-management surface) are judged on
    measured throughput. A **memory-management** candidate (e.g. mmap) may instead be
    retained on a measured **memory-objective reduction** — memory efficiency is a
    legitimate optimization objective even when throughput does not improve. It never
    fabricates a benefit and never converts a memory saving into a performance claim.
    """
    if mode_rules is not None:
        return _retention_by_keep_rule(
            result=result,
            candidate=candidate,
            policy=policy,
            effort=optimization_preference or mode_rules.effort,
            performance_state=performance_state,
            mode_rules=mode_rules,
        )
    cost = policy.cost_model
    measured_perf = measured_throughput_improvement(result)
    measured_mem = measured_memory_improvement(result) if _is_memory_surface(candidate) else None
    if measured_perf is not None:
        measured: float | None = measured_perf
        objective = "performance"
    elif measured_mem is not None:
        measured = measured_mem
        objective = "memory"
    else:
        measured = None
        objective = "performance"
    required = (
        optimization_preference.minimum_improvement_for(performance_state)
        if optimization_preference is not None
        else 0.0
    )
    change_cost_s = candidate.cost.total_cost_s if candidate is not None else 0.0
    budget = (
        float(optimization_preference.maximum_experiment_cost_s)
        if optimization_preference is not None
        else float("inf")
    )
    cost_score = change_cost_s / cost.benefit_scale_s
    within_budget = change_cost_s <= budget
    improvement_justified = measured is not None and measured >= required
    beneficial = bool(improvement_justified and measured is not None and measured > 0.0 and within_budget)

    rationale = [
        f"objective={objective}",
        f"measured_throughput_improvement={'unknown' if measured_perf is None else f'{measured_perf:+.2%}'}",
        f"measured_memory_improvement={'unknown' if measured_mem is None else f'{measured_mem:.2%}'}",
        f"measured{'(' + objective + ')'}={'unknown' if measured is None else f'{measured:.2%}'}",
        f"required_improvement={required:.2%} improvement_justified={improvement_justified}",
        f"retained_change_cost={change_cost_s:.2f}s within_mode_budget={within_budget} "
        f"(budget {budget:.2f}s)" if optimization_preference is not None
        else f"retained_change_cost={change_cost_s:.2f}s",
        f"beneficial={beneficial}",
        f"optimization_mode={optimization_preference.mode.value if optimization_preference else None} "
        f"performance_state={performance_state.value}",
    ]
    return InterventionEconomics(
        expected_benefit_score=_clamp(measured) if measured is not None else 0.0,
        evidence_score=1.0 if measured is not None else 0.0,
        persistence_score=1.0,
        cost_score=cost_score,
        net_value=(measured - required) if measured is not None else 0.0,
        beneficial=beneficial,
        requires_approval=False,
        intervention_class=InterventionClass.NONE,
        costs={"retained_change": change_cost_s},
        rationale=rationale,
        optimization_mode=(
            optimization_preference.mode.value if optimization_preference is not None else None
        ),
        performance_state=performance_state.value,
        required_improvement=required,
        measured_improvement=measured,
        improvement_justified=improvement_justified,
    )


def _retention_by_keep_rule(
    *,
    result: ExperimentResult,
    candidate: OptimizationCandidate | None,
    policy: RuntimePolicy,
    effort: OptimizationPreference,
    performance_state: PerformanceState,
    mode_rules: ModeRules,
) -> InterventionEconomics:
    decision = result.decision_detail
    technically_better = bool(
        result.status is ExperimentDecisionKind.IMPROVED
        and (decision is None or decision.keep)
    )
    judgement = judge_keep(
        result.comparison,
        technically_better=technically_better,
        effort=effort,
        goal=mode_rules.goal,
        situation=Situation.WHILE_RUNNING,
        settings=mode_rules.settings,
        remaining_cost_terms={},
        performance_state=performance_state,
    )
    gain = judgement.goal.gain
    change_cost_s = candidate.cost.total_cost_s if candidate is not None else 0.0
    return InterventionEconomics(
        expected_benefit_score=_clamp(gain) if gain is not None else 0.0,
        evidence_score=1.0 if gain is not None else 0.0,
        persistence_score=1.0,
        cost_score=change_cost_s / policy.cost_model.benefit_scale_s,
        net_value=(gain - judgement.required_gain) if gain is not None else 0.0,
        beneficial=judgement.keep,
        requires_approval=False,
        intervention_class=InterventionClass.NONE,
        costs={"retained_change": change_cost_s},
        rationale=[*judgement.reasons, f"beneficial={judgement.keep}"],
        optimization_mode=effort.mode.value,
        performance_state=performance_state.value,
        required_improvement=judgement.required_gain,
        measured_improvement=gain,
        improvement_justified=gain is not None and gain >= judgement.required_gain,
        goal_mode=mode_rules.goal.mode.value,
        goal_gain=gain,
        keep_verdict=judgement.verdict.value,
        payback=judgement.payback,
    )
