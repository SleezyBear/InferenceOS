"""External action safety evaluation (Phase 11R: recommendation-only).

Phase 11R deliberately contains **no executor**. InferenceOS observes external
processes and recommends human actions, but it never terminates, suspends, reprioritises,
changes affinity of, or otherwise mutates an unrelated external process. This module
evaluates a *proposed* external action solely to prove it is not executable; the result
is always ``allowed=False`` with an explicit reason.
"""

from __future__ import annotations

from inferenceos.contracts.enums import (
    ExternalActionSafetyClass,
    OptimizationScope,
    PrivilegeRequirement,
)
from inferenceos.contracts.external import (
    ExternalActionSafety,
    ExternalPolicy,
    ExternalProcessIdentity,
)
from inferenceos.optimization.registry import ActionRegistry, UnknownActionError

# Phase 11R safety boundary: no external-process action is ever authorized for
# autonomous execution by InferenceOS. Even an accepted recommendation is only a
# recorded user decision.
PHASE11R_RECOMMENDATION_ONLY_REASON = (
    "Phase 11R is recommendation-only: InferenceOS never autonomously mutates an "
    "external process; the user decides"
)


def evaluate_external_action_safety(
    action_id: str | None,
    *,
    registry: ActionRegistry,
    policy: ExternalPolicy,
    identity: ExternalProcessIdentity | None,
    elevation_available: bool = False,
) -> ExternalActionSafety:
    """Evaluate an external action for safety.

    The result can never authorize an external-process mutation in Phase 11R. This is
    enforced structurally (no external action is registered as implemented) *and*
    defensively (``allowed`` is always false) so a registry change can never silently
    reintroduce autonomous external-process control here.
    """

    reasons: list[str] = [PHASE11R_RECOMMENDATION_ONLY_REASON]
    if action_id is None:
        return ExternalActionSafety(
            action_id="",
            recommendation_id="",
            identity_key=identity.identity_key if identity else None,
            allowed=False,
            requires_approval=True,
            reasons=reasons + ["no registered action proposed"],
        )
    try:
        action = registry.get(action_id)
    except UnknownActionError:
        return ExternalActionSafety(
            action_id=action_id,
            recommendation_id="",
            identity_key=identity.identity_key if identity else None,
            allowed=False,
            reasons=reasons + [f"action {action_id!r} is not registered"],
        )

    if action.scope is not OptimizationScope.EXTERNAL_PROCESS:
        reasons.append("action scope is not an external process (out of Phase 11)")
    elevation_required = action.privilege is PrivilegeRequirement.ELEVATION_REQUIRED
    if elevation_required:
        reasons.append("action requires elevation, which is never automatic")
        if policy.elevation_never_automatic and not elevation_available:
            reasons.append("elevation is not available and will not be requested silently")
    if action.implemented:
        reasons.append(
            "action is marked implemented, which Phase 11R forbids for external actions"
        )
    else:
        reasons.append("action is recommendation-only / not implemented in Phase 11")
    if action.external_safety_class is ExternalActionSafetyClass.UNSUPPORTED:
        reasons.append("action is classified unsupported")

    return ExternalActionSafety(
        action_id=action.action_id,
        recommendation_id="",
        identity_key=identity.identity_key if identity else None,
        allowed=False,
        requires_approval=action.requires_human_approval or policy.require_approval_for_actions,
        elevation_required=elevation_required,
        implemented=False,
        phase11_allowed=action.phase11_allowed,
        reversible=action.reversibility.value == "reversible",
        verification_supported=action.verification_supported,
        safety_class=action.external_safety_class,
        reasons=reasons,
    )
