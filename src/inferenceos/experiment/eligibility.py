"""Immediately-before-execution candidate revalidation.

A Phase 6 candidate that was eligible during generation may have become invalid,
stale, or unsafe. Phase 7 never trusts stored candidate data: it resolves the action
through the registry, re-runs the preconditions against the current real context,
re-checks safety/privilege/scope, verifies the environment still matches the
candidate's assumptions, and verifies the executor has a real safe surface.

If any check fails, the candidate is not executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from inferenceos.contracts.enums import (
    ApprovalRequirement,
    CandidateEligibility,
    ExperimentDecisionKind,
    PreconditionKind,
)
from inferenceos.contracts.optimization import (
    CandidateAction,
    CandidatePlan,
    CandidatePrecondition,
    OptimizationCandidate,
)
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.preconditions import evaluate_eligibility, make
from inferenceos.optimization.registry import ActionRegistry, UnknownActionError
from inferenceos.optimization.strategies import prepare
from inferenceos.experiment.executor import ActionExecutor, validate_parameters

DEFAULT_TTL_S = 24 * 3600.0


@dataclass(frozen=True)
class RevalidationResult:
    ok: bool
    reason: str
    kind: ExperimentDecisionKind | None = None
    action: CandidateAction | None = None
    checks: list[CandidatePrecondition] = field(default_factory=list)


def revalidate_candidate(
    candidate: OptimizationCandidate,
    *,
    plan: CandidatePlan | None,
    context: OptimizationContext,
    registry: ActionRegistry,
    policy: OptimizationPolicy,
    executor: ActionExecutor,
    ttl_s: float = DEFAULT_TTL_S,
    now: datetime | None = None,
) -> RevalidationResult:
    now = now or datetime.now(timezone.utc)
    checks: list[CandidatePrecondition] = []

    # 1. The candidate must belong to the supplied (or a known) plan.
    in_plan = plan is None or candidate.candidate_id in {
        item.candidate_id for item in plan.candidates
    }
    checks.append(
        make(
            "candidate_in_plan",
            PreconditionKind.SAFETY,
            "candidate is present in the source plan",
            in_plan,
            f"plan={plan.plan_id}" if plan is not None else "no plan supplied",
        )
    )
    if not in_plan:
        return _fail("candidate is not part of the source plan", checks, ExperimentDecisionKind.UNSAFE)

    # 2. The action must be registered (no invented actions survive).
    action: CandidateAction | None
    try:
        action = registry.get(candidate.action_id)
    except UnknownActionError:
        action = None
    checks.append(
        make(
            "action_registered",
            PreconditionKind.SAFETY,
            "action id is registered",
            action is not None,
            candidate.action_id,
        )
    )
    if action is None:
        return _fail(
            f"unregistered action {candidate.action_id!r} rejected before execution",
            checks,
            ExperimentDecisionKind.UNSAFE,
        )

    # 3. Safety: eligible, executable, no approval required, covered by the executor.
    safe = (
        candidate.eligibility is CandidateEligibility.ELIGIBLE
        and candidate.executable
        and candidate.approval is ApprovalRequirement.NOT_REQUIRED
    )
    checks.append(
        make(
            "safety_eligible",
            PreconditionKind.SAFETY,
            "candidate is eligible and not approval-required",
            safe,
            f"eligibility={candidate.eligibility.value} approval={candidate.approval.value}",
        )
    )
    if not safe:
        return _fail(
            "candidate is not autonomously executable (gated/rejected/approval-required)",
            checks,
            ExperimentDecisionKind.UNSAFE,
        )

    supported = executor.supports(action.action_id)
    checks.append(
        make(
            "executor_supports",
            PreconditionKind.CAPABILITY,
            "executor has a real, safe surface for this action",
            supported,
            executor.name,
        )
    )
    if not supported:
        return _fail(
            f"action {action.action_id!r} has no safe runtime surface; not executed",
            checks,
            ExperimentDecisionKind.PRECONDITION_FAILURE,
            action,
        )

    # 4. Parameters must validate against the registry schema.
    params_ok, params_reason = validate_parameters(action, candidate.parameters)
    checks.append(
        make(
            "parameters_valid",
            PreconditionKind.PARAMETER,
            "candidate parameters validate against the registered schema",
            params_ok,
            params_reason or "ok",
        )
    )
    if not params_ok:
        return _fail(params_reason, checks, ExperimentDecisionKind.PRECONDITION_FAILURE, action)

    # 5. Re-run the Phase 6 preconditions against the current real context.
    prepared = prepare(action, context, policy)
    if prepared is None:
        return _fail(
            "action has no preparation strategy", checks, ExperimentDecisionKind.PRECONDITION_FAILURE, action
        )
    eligibility, approval, reason = evaluate_eligibility(action, prepared.preconditions, policy)
    checks.extend(prepared.preconditions)
    fresh_ok = eligibility is CandidateEligibility.ELIGIBLE and approval is ApprovalRequirement.NOT_REQUIRED
    if not fresh_ok:
        return _fail(
            f"preconditions no longer hold: {reason}",
            checks,
            ExperimentDecisionKind.PRECONDITION_FAILURE,
            action,
        )

    # 6. The environment must still match the candidate's assumptions.
    environment_ok, environment_reason = _environment_matches(candidate, context)
    checks.append(
        make(
            "environment_matches",
            PreconditionKind.APPLICABILITY,
            "target environment still matches the candidate's assumptions",
            environment_ok,
            environment_reason,
        )
    )
    if not environment_ok:
        return _fail(environment_reason, checks, ExperimentDecisionKind.PRECONDITION_FAILURE, action)

    # 7. Not expired.
    expired = False
    if plan is not None and ttl_s > 0:
        age = (now - plan.created_at).total_seconds()
        expired = age > ttl_s
    checks.append(
        make(
            "candidate_fresh",
            PreconditionKind.APPLICABILITY,
            "candidate has not expired",
            not expired,
            f"ttl_s={ttl_s:g}",
        )
    )
    if expired:
        return _fail("candidate expired", checks, ExperimentDecisionKind.PRECONDITION_FAILURE, action)

    return RevalidationResult(ok=True, reason="revalidated", action=action, checks=checks)


def _environment_matches(
    candidate: OptimizationCandidate, context: OptimizationContext
) -> tuple[bool, str]:
    if context.model is not None and candidate.model_ref and candidate.model_ref != context.model.name:
        return False, f"model changed: candidate={candidate.model_ref} current={context.model.name}"
    if (
        context.workload is not None
        and candidate.workload_id
        and candidate.workload_id != context.workload.workload_id
    ):
        return False, (
            f"workload changed: candidate={candidate.workload_id} "
            f"current={context.workload.workload_id}"
        )
    if candidate.hardware.cpu.name != context.hardware.cpu.name:
        return False, "hardware CPU identity changed"
    if candidate.hardware.cpu.physical_cores != context.hardware.cpu.physical_cores:
        return False, "hardware core count changed"
    if context.effective_kind is None or context.effective_kind != candidate.diagnosis_kind:
        return False, (
            "effective bottleneck changed: "
            f"candidate={candidate.diagnosis_kind.value} "
            f"current={context.effective_kind.value if context.effective_kind else None}"
        )
    return True, "model/workload/hardware/bottleneck assumptions hold"


def _fail(
    reason: str,
    checks: list[CandidatePrecondition],
    kind: ExperimentDecisionKind,
    action: CandidateAction | None = None,
) -> RevalidationResult:
    return RevalidationResult(ok=False, reason=reason, kind=kind, action=action, checks=checks)
