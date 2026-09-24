"""Explicit preconditions and the deterministic eligibility policy.

Preconditions are evaluated against the real Phase 1-5 context. A condition that
cannot be established is ``None`` — it never silently becomes eligible. Hard safety
and privilege gates dominate: they can only gate or reject, never approve.
"""

from __future__ import annotations

from inferenceos.contracts.enums import (
    ApprovalRequirement,
    CandidateEligibility,
    PreconditionKind,
    PrivilegeRequirement,
    Reversibility,
    RiskLevel,
)
from inferenceos.contracts.optimization import CandidateAction, CandidatePrecondition
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.safety import requires_approval, risk_rank, strength_rank


def make(
    check_id: str,
    kind: PreconditionKind,
    description: str,
    satisfied: bool | None,
    detail: str = "",
    *,
    required: bool = True,
) -> CandidatePrecondition:
    return CandidatePrecondition(
        check_id=check_id,
        kind=kind,
        description=description,
        satisfied=satisfied,
        detail=detail,
        required=required,
    )


def check_evidence_support(
    ctx: OptimizationContext, action: CandidateAction, policy: OptimizationPolicy
) -> CandidatePrecondition:
    if ctx.effective_kind is None:
        return make(
            "evidence_supported",
            PreconditionKind.EVIDENCE,
            "a validated diagnosis establishes this bottleneck",
            None,
            "no effective diagnosis; evidence cannot establish relevance",
        )
    if ctx.effective_kind not in action.supported_bottlenecks:
        return make(
            "evidence_supported",
            PreconditionKind.EVIDENCE,
            "action addresses the diagnosed bottleneck",
            False,
            f"effective bottleneck {ctx.effective_kind.value} is not in the action's "
            "supported set",
        )
    if strength_rank(ctx.effective_strength) < strength_rank(action.min_evidence_strength):
        return make(
            "evidence_supported",
            PreconditionKind.EVIDENCE,
            "diagnosis evidence strength meets the action's minimum",
            None,
            f"strength {ctx.effective_strength.value} below required "
            f"{action.min_evidence_strength.value}",
        )
    if ctx.effective_confidence < action.min_confidence:
        return make(
            "evidence_supported",
            PreconditionKind.EVIDENCE,
            "diagnosis confidence meets the action's minimum",
            None,
            f"confidence {ctx.effective_confidence:.2f} below required "
            f"{action.min_confidence:.2f}",
        )
    return make(
        "evidence_supported",
        PreconditionKind.EVIDENCE,
        "diagnosis evidence supports this action",
        True,
        f"{ctx.effective_kind.value} at confidence {ctx.effective_confidence:.2f} "
        f"({ctx.effective_strength.value})",
    )


def evaluate_eligibility(
    action: CandidateAction,
    preconditions: list[CandidatePrecondition],
    policy: OptimizationPolicy,
) -> tuple[CandidateEligibility, ApprovalRequirement, str | None]:
    """Deterministic eligibility/approval decision. Safety dominates."""
    if not action.enabled:
        return CandidateEligibility.REJECTED, ApprovalRequirement.NOT_REQUIRED, "action disabled"

    required = [item for item in preconditions if item.required]

    for item in required:
        if item.satisfied is False:
            if item.kind is PreconditionKind.EVIDENCE:
                return (
                    CandidateEligibility.INSUFFICIENT_EVIDENCE,
                    ApprovalRequirement.NOT_REQUIRED,
                    f"insufficient evidence: {item.description}",
                )
            if item.kind is PreconditionKind.PRIVILEGE and action.requires_human_approval:
                return (
                    CandidateEligibility.GATED,
                    ApprovalRequirement.REQUIRED,
                    f"approval required: {item.description}",
                )
            return (
                CandidateEligibility.REJECTED,
                ApprovalRequirement.NOT_REQUIRED,
                f"precondition failed: {item.description}",
            )

    # Non-reversible / critical work is never autonomously selectable.
    if action.reversibility is Reversibility.NON_REVERSIBLE:
        return (
            CandidateEligibility.REJECTED,
            ApprovalRequirement.NOT_REQUIRED,
            "non-reversible action is not autonomously selectable",
        )
    if action.risk_level is RiskLevel.CRITICAL:
        return (
            CandidateEligibility.REJECTED,
            ApprovalRequirement.NOT_REQUIRED,
            "critical-risk action is not autonomously selectable",
        )

    # External/system-wide/elevated or explicitly approval-required -> gated.
    if requires_approval(
        scope=action.scope, privilege=action.privilege, flag=action.requires_human_approval
    ):
        return (
            CandidateEligibility.GATED,
            ApprovalRequirement.REQUIRED,
            "external/system-wide/elevated action requires human approval",
        )

    for item in required:
        if item.satisfied is None:
            if item.kind is PreconditionKind.EVIDENCE:
                return (
                    CandidateEligibility.INSUFFICIENT_EVIDENCE,
                    ApprovalRequirement.NOT_REQUIRED,
                    f"insufficient evidence: {item.description}",
                )
            if item.kind is PreconditionKind.PRIVILEGE:
                return (
                    CandidateEligibility.GATED,
                    ApprovalRequirement.REQUIRED,
                    f"privilege not established: {item.description}",
                )
            return (
                CandidateEligibility.REJECTED,
                ApprovalRequirement.NOT_REQUIRED,
                f"precondition not established: {item.description}",
            )

    if risk_rank(action.risk_level) > risk_rank(policy.autonomous_max_risk):
        return (
            CandidateEligibility.GATED,
            ApprovalRequirement.REQUIRED,
            "risk above the autonomous ceiling",
        )

    return CandidateEligibility.ELIGIBLE, ApprovalRequirement.NOT_REQUIRED, None


def privilege_precondition(
    privilege: PrivilegeRequirement, available: bool | None
) -> CandidatePrecondition:
    if privilege is PrivilegeRequirement.STANDARD_USER:
        satisfied: bool | None = True
        detail = "standard-user privilege"
    elif privilege is PrivilegeRequirement.ELEVATION_REQUIRED:
        satisfied = available
        detail = "administrator elevation required"
    else:
        satisfied = available
        detail = "privilege availability unknown"
    return make(
        "privilege_available",
        PreconditionKind.PRIVILEGE,
        "required privilege is available",
        satisfied,
        detail,
    )
