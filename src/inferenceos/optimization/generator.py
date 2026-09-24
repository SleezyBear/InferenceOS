"""Phase 6 candidate generation orchestrator.

Consumes a validated Phase 4/5 diagnosis and produces a bounded, deduplicated,
ranked, auditable :class:`CandidatePlan`. The generator only inspects context and
computes eligibility/benefit/cost/risk; it never executes an intervention and never
mutates the environment.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import psutil

from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    ApprovalRequirement,
    BottleneckKind,
    CandidateEligibility,
    CandidatePlanOutcome,
    OptimizationSurface,
    WorkloadKind,
)
from inferenceos.contracts.optimization import (
    CandidateAction,
    CandidatePlan,
    OptimizationCandidate,
    OptimizerOverhead,
    ReasoningProposalOutcome,
    RiskAssessment,
)
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.reasoning import DeepDiagnosisOutcome
from inferenceos.optimization.actions import ACTION_REGISTRY_VERSION, build_default_registry
from inferenceos.optimization.context import (
    build_optimization_context,
    OptimizationContext,
)
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.preconditions import evaluate_eligibility
from inferenceos.optimization.ranking import rank_candidate
from inferenceos.optimization.reasoning import propose_actions
from inferenceos.optimization.registry import ActionRegistry, UnknownActionError
from inferenceos.optimization.strategies import PreparedAction, prepare
from inferenceos.reasoning.providers.base import ReasoningProvider


def generate_candidates(
    diagnosis: BottleneckDiagnosis,
    *,
    report: BaselineReport | None = None,
    outcome: DeepDiagnosisOutcome | None = None,
    registry: ActionRegistry | None = None,
    policy: OptimizationPolicy | None = None,
    provider: ReasoningProvider | None = None,
    enable_reasoning: bool = False,
) -> CandidatePlan:
    started = time.perf_counter()
    registry = registry or build_default_registry()
    policy = policy or OptimizationPolicy()
    ctx = build_optimization_context(
        diagnosis, report=report, outcome=outcome, policy=policy
    )

    candidates: list[OptimizationCandidate] = []
    reasoning_outcome: ReasoningProposalOutcome | None = None
    effective_kind = ctx.effective_kind
    deterministic_used = effective_kind is not None

    if effective_kind is not None:
        for action in registry.for_bottleneck(effective_kind):
            prepared = prepare(action, ctx, policy)
            if prepared is None:
                continue
            candidates.append(_build_candidate(action, prepared, ctx, policy, "deterministic"))

    reasoning_needed = (
        not deterministic_used
        and policy.allow_reasoning
        and enable_reasoning
        and provider is not None
        and verdict_insufficient(diagnosis, outcome)
    )
    if reasoning_needed and provider is not None:
        proposals, reasoning_outcome = propose_actions(ctx, registry, provider)
        for proposal in proposals:
            try:
                action = registry.get(str(proposal["action_id"]))
            except UnknownActionError:  # pragma: no cover - parse already validates
                continue
            prepared = prepare(action, ctx, policy)
            if prepared is None:
                continue
            prepared = _apply_proposal(prepared, proposal)
            evidence_refs = [
                item for item in proposal["evidence_ids"] if isinstance(item, str)
            ]
            candidates.append(
                _build_candidate(
                    action,
                    prepared,
                    ctx,
                    policy,
                    "reasoning",
                    evidence_refs=evidence_refs,
                )
            )

    candidates = _deduplicate(candidates)
    candidates = _resolve_conflicts(candidates)

    for index, candidate in enumerate(candidates):
        if candidate.eligibility is CandidateEligibility.ELIGIBLE:
            candidates[index] = candidate.model_copy(
                update={"ranking": rank_candidate(candidate, ctx, policy)}
            )
    eligible = [c for c in candidates if c.eligibility is CandidateEligibility.ELIGIBLE]
    eligible.sort(key=lambda c: (c.ranking.composite_score if c.ranking else 0.0), reverse=True)

    selected = eligible[: policy.max_candidates]
    plan_outcome, terminal_reason = _terminal_outcome(ctx, candidates, eligible)
    latency = time.perf_counter() - started
    overhead = _overhead(candidates, reasoning_outcome, latency)

    return CandidatePlan(
        plan_id=f"plan-{uuid.uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc),
        source_diagnosis_id=diagnosis.diagnosis_id,
        source_diagnosis=diagnosis,
        source_outcome_id=outcome.outcome_id if outcome is not None else None,
        effective_bottleneck=ctx.effective_kind,
        effective_confidence=ctx.effective_confidence,
        effective_basis=ctx.effective_basis,
        outcome=plan_outcome,
        terminal_reason=terminal_reason,
        candidates=candidates,
        eligible_ids=[c.candidate_id for c in candidates if c.eligibility is CandidateEligibility.ELIGIBLE],
        gated_ids=[c.candidate_id for c in candidates if c.eligibility is CandidateEligibility.GATED],
        rejected_ids=[c.candidate_id for c in candidates if c.eligibility is CandidateEligibility.REJECTED],
        insufficient_evidence_ids=[
            c.candidate_id
            for c in candidates
            if c.eligibility is CandidateEligibility.INSUFFICIENT_EVIDENCE
        ],
        ranking=[c.candidate_id for c in eligible],
        selected_ids=[c.candidate_id for c in selected],
        reasoning=reasoning_outcome,
        overhead=overhead,
        policy=policy.as_dict(),
        reproducibility={
            "registry_version": ACTION_REGISTRY_VERSION,
            "registry_actions": registry.ids(),
            "detector_version": diagnosis.context.detector_version,
            "source_report_path": diagnosis.context.source_report_path,
            "policy": policy.as_dict(),
        },
        limitations=_limitations(ctx, reasoning_outcome, policy, overhead),
        environment_mutated=False,
        executed=False,
        summary=_summary(plan_outcome, ctx, candidates, selected),
    )


def verdict_insufficient(
    diagnosis: BottleneckDiagnosis, outcome: DeepDiagnosisOutcome | None
) -> bool:
    return bool(diagnosis.insufficient_evidence or not diagnosis.meaningful_opportunity)


def _build_candidate(
    action: CandidateAction,
    prepared: PreparedAction,
    ctx: OptimizationContext,
    policy: OptimizationPolicy,
    generated_by: str,
    *,
    evidence_refs: list[str] | None = None,
) -> OptimizationCandidate:
    eligibility, approval, reason = evaluate_eligibility(action, prepared.preconditions, policy)
    gating_reasons = [
        f"{item.check_id}: {item.detail or item.description}"
        for item in prepared.preconditions
        if item.required and item.satisfied is not True
    ]
    risk = RiskAssessment(
        risk_level=action.risk_level,
        reversibility=action.reversibility,
        scope=action.scope,
        privilege=action.privilege,
        failure_consequences=prepared.failure_consequences,
        rollback_available=prepared.rollback_available,
        requires_approval=approval is ApprovalRequirement.REQUIRED,
        gating_reasons=gating_reasons,
    )
    executable = (
        eligibility is CandidateEligibility.ELIGIBLE
        and approval is ApprovalRequirement.NOT_REQUIRED
    )
    source_bottleneck = ctx.source_bottleneck
    diagnosis_kind = ctx.effective_kind or BottleneckKind.INSUFFICIENT_EVIDENCE
    return OptimizationCandidate(
        candidate_id=f"cand-{uuid.uuid4().hex[:10]}",
        action_id=action.action_id,
        action_name=action.name,
        surface=action.surface,
        subsystem=action.subsystem,
        parameters=dict(prepared.parameters),
        source_diagnosis_id=ctx.diagnosis.diagnosis_id,
        source_bottleneck_id=(
            source_bottleneck.bottleneck_id if source_bottleneck is not None else None
        ),
        diagnosis_kind=diagnosis_kind,
        diagnosis_confidence=ctx.effective_confidence,
        diagnosis_strength=ctx.effective_strength,
        causal_rationale=prepared.causal_rationale,
        evidence_refs=(
            evidence_refs if evidence_refs is not None else list(ctx.evidence_ids)
        ),
        hardware=ctx.hardware,
        model_ref=ctx.model.name if ctx.model is not None else None,
        model_architecture=ctx.model.architecture if ctx.model is not None else None,
        workload_id=ctx.workload.workload_id if ctx.workload is not None else None,
        workload_kind=ctx.workload.kind if ctx.workload is not None else WorkloadKind.UNKNOWN,
        observed_metrics=dict(ctx.observed_metrics),
        preconditions=list(prepared.preconditions),
        benefit=prepared.benefit,
        cost=prepared.cost,
        risk=risk,
        approval=approval,
        eligibility=eligibility,
        rejection_reason=reason,
        generated_by=generated_by,
        executable=executable,
    )


def _apply_proposal(
    prepared: PreparedAction, proposal: dict[str, object]
) -> PreparedAction:
    params = proposal.get("parameters")
    if not isinstance(params, dict) or not params:
        return prepared
    return PreparedAction(
        parameters={**prepared.parameters, **params},
        preconditions=prepared.preconditions,
        benefit=prepared.benefit,
        cost=prepared.cost,
        causal_rationale=prepared.causal_rationale,
        failure_consequences=prepared.failure_consequences,
        rollback_available=prepared.rollback_available,
    )


def _canonical(value: object) -> str:
    if isinstance(value, bool):
        return f"b:{value}"
    if isinstance(value, (int, float)):
        return f"n:{float(value)}"
    return f"s:{value}"


def _identity(candidate: OptimizationCandidate) -> tuple[str, str]:
    parameter_key = json.dumps(
        {key: _canonical(value) for key, value in sorted(candidate.parameters.items())},
        sort_keys=True,
    )
    return candidate.action_id, parameter_key


def _deduplicate(
    candidates: list[OptimizationCandidate],
) -> list[OptimizationCandidate]:
    seen: set[tuple[str, str]] = set()
    unique: list[OptimizationCandidate] = []
    for candidate in candidates:
        key = _identity(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _resolve_conflicts(
    candidates: list[OptimizationCandidate],
) -> list[OptimizationCandidate]:
    """Deterministic handling of mutually incompatible parameterizations.

    A candidate conflicts with another when it binds the same action (or the same
    surface + parameter) to a different value. The better candidate (eligible first,
    then higher benefit) is kept; the other is rejected with an explicit reason.
    """
    best: dict[str, OptimizationCandidate] = {}
    for candidate in candidates:
        for key in _conflict_keys(candidate):
            current = best.get(key)
            if current is None or _preference(candidate) > _preference(current):
                best[key] = candidate

    winners = {id(candidate) for candidate in best.values()}
    resolved: list[OptimizationCandidate] = []
    for candidate in candidates:
        if id(candidate) in winners:
            resolved.append(candidate)
            continue
        conflict = sorted(_conflict_keys(candidate))
        resolved.append(
            candidate.model_copy(
                update={
                    "eligibility": CandidateEligibility.REJECTED,
                    "executable": False,
                    "rejection_reason": (
                        "conflicts with a preferred candidate for "
                        + ", ".join(conflict)
                    ),
                    "ranking": None,
                }
            )
        )
    return resolved


def _conflict_keys(candidate: OptimizationCandidate) -> list[str]:
    if candidate.surface is OptimizationSurface.EXTERNAL_PROCESS:
        return [candidate.action_id]
    keys = [candidate.action_id]
    for name in candidate.parameters:
        keys.append(f"{candidate.surface.value}:{name}")
    return keys


def _preference(candidate: OptimizationCandidate) -> tuple[int, float]:
    eligible = 1 if candidate.eligibility is CandidateEligibility.ELIGIBLE else 0
    benefit = candidate.benefit.magnitude_high or 0.0
    return eligible, benefit


def _terminal_outcome(
    ctx: OptimizationContext,
    candidates: list[OptimizationCandidate],
    eligible: list[OptimizationCandidate],
) -> tuple[CandidatePlanOutcome, str]:
    if ctx.effective_kind is None:
        if ctx.diagnosis.insufficient_evidence or not ctx.diagnosis.meaningful_opportunity:
            if ctx.diagnosis.insufficient_evidence:
                reason = (
                    "evidence is insufficient to establish a meaningful optimization "
                    f"opportunity: {ctx.diagnosis.no_opportunity_reason or ctx.diagnosis.summary}"
                )
                return CandidatePlanOutcome.INSUFFICIENT_EVIDENCE, reason
            reason = (
                "no meaningful optimization opportunity exists; resource utilization "
                f"alone is not an opportunity: {ctx.diagnosis.no_opportunity_reason or ctx.diagnosis.summary}"
            )
            return CandidatePlanOutcome.NO_OPPORTUNITY, reason
        return (
            CandidatePlanOutcome.INSUFFICIENT_EVIDENCE,
            "no effective diagnosis is available to map to a registered action",
        )
    if not candidates:
        return (
            CandidatePlanOutcome.NO_VIABLE_CANDIDATES,
            f"no registered action addresses the diagnosed {ctx.effective_kind.value} bottleneck",
        )
    if not eligible:
        return (
            CandidatePlanOutcome.NO_VIABLE_CANDIDATES,
            "candidates were generated but none passed the preconditions/safety gates",
        )
    return (
        CandidatePlanOutcome.CANDIDATES_GENERATED,
        f"{len(eligible)} eligible candidate(s) generated for the diagnosed "
        f"{ctx.effective_kind.value} bottleneck; none executed",
    )


def _overhead(
    candidates: list[OptimizationCandidate],
    reasoning: ReasoningProposalOutcome | None,
    latency: float,
) -> OptimizerOverhead:
    rss: int | None = None
    try:
        rss = psutil.Process().memory_info().rss
    except (psutil.Error, OSError):  # pragma: no cover - defensive
        rss = None
    return OptimizerOverhead(
        generation_latency_s=latency,
        reasoning_latency_s=reasoning.latency_s if reasoning is not None else None,
        actions_considered=len(candidates),
        candidates_generated=len(candidates),
        eligible=sum(1 for c in candidates if c.eligibility is CandidateEligibility.ELIGIBLE),
        gated=sum(1 for c in candidates if c.eligibility is CandidateEligibility.GATED),
        rejected=sum(1 for c in candidates if c.eligibility is CandidateEligibility.REJECTED),
        insufficient_evidence=sum(
            1 for c in candidates if c.eligibility is CandidateEligibility.INSUFFICIENT_EVIDENCE
        ),
        peak_optimizer_rss_bytes=rss,
        notes=["candidate generation is read-only; no intervention was executed"],
    )


def _limitations(
    ctx: OptimizationContext,
    reasoning: ReasoningProposalOutcome | None,
    policy: OptimizationPolicy,
    overhead: OptimizerOverhead,
) -> list[str]:
    limitations: list[str] = list(ctx.diagnosis.limitations)
    if ctx.effective_kind is None:
        limitations.append(
            "no effective bottleneck was established, so deterministic mapping produced "
            "no actionable candidates"
        )
    if reasoning is not None and reasoning.status == "failed":
        limitations.append(f"reasoning proposal unavailable: {reasoning.failure}")
    if overhead.generation_latency_s > policy.max_optimizer_overhead_s:
        limitations.append(
            f"optimizer overhead {overhead.generation_latency_s:.3f}s exceeded the "
            f"configured ceiling {policy.max_optimizer_overhead_s:.3f}s"
        )
    limitations.append(
        "benefit magnitudes that are not evidence-derivable are marked unknown and must "
        "be measured in Phase 7"
    )
    return limitations


def _summary(
    outcome: CandidatePlanOutcome,
    ctx: OptimizationContext,
    candidates: list[OptimizationCandidate],
    selected: list[OptimizationCandidate],
) -> str:
    if outcome is CandidatePlanOutcome.NO_OPPORTUNITY:
        return "No meaningful optimization opportunity; no candidates generated."
    if outcome is CandidatePlanOutcome.INSUFFICIENT_EVIDENCE:
        return "Evidence insufficient to justify candidates; no executable candidate produced."
    if outcome is CandidatePlanOutcome.NO_VIABLE_CANDIDATES:
        return (
            f"{len(candidates)} candidate(s) considered for the diagnosed "
            f"{ctx.effective_kind.value if ctx.effective_kind else 'unknown'} bottleneck; "
            "none cleared the safety/eligibility gates."
        )
    return (
        f"{len(selected)} of {len(candidates)} candidate(s) selected for Phase 7 "
        f"consideration (not executed)."
    )
