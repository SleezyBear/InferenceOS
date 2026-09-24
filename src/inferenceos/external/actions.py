"""External action safety evaluation and the minimal supported action executor.

Only actions explicitly registered, implemented, Phase 11 allowed, approval-required,
and independently verifiable can be executed — currently only
``external.reduce_priority`` (reversible, verifiable, standard-user). Everything else
is recommendation-only. Elevation is never automatic.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from inferenceos.contracts.enums import (
    ExternalActionSafetyClass,
    OptimizationScope,
    PrivilegeRequirement,
)
from inferenceos.contracts.external import (
    ExternalActionCheckpoint,
    ExternalActionResult,
    ExternalActionSafety,
    ExternalPolicy,
    ExternalProcessIdentity,
)
from inferenceos.external.identity import verify_identity
from inferenceos.optimization.registry import ActionRegistry, UnknownActionError

try:  # pragma: no cover - psutil is a hard dependency in practice
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

_PRIORITY_CLASSES: dict[str, int] = {
    "idle": getattr(psutil, "IDLE_PRIORITY_CLASS", 64) if psutil is not None else 64,
    "below_normal": getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 16384)
    if psutil is not None
    else 16384,
}


def evaluate_external_action_safety(
    action_id: str | None,
    *,
    registry: ActionRegistry,
    policy: ExternalPolicy,
    identity: ExternalProcessIdentity | None,
    elevation_available: bool = False,
) -> ExternalActionSafety:
    reasons: list[str] = []
    if action_id is None:
        return ExternalActionSafety(
            action_id="",
            recommendation_id="",
            identity_key=identity.identity_key if identity else None,
            allowed=False,
            requires_approval=True,
            reasons=["no registered action proposed"],
        )
    try:
        action = registry.get(action_id)
    except UnknownActionError:
        return ExternalActionSafety(
            action_id=action_id,
            recommendation_id="",
            identity_key=identity.identity_key if identity else None,
            allowed=False,
            reasons=[f"action {action_id!r} is not registered"],
        )

    if action.scope is not OptimizationScope.EXTERNAL_PROCESS:
        reasons.append("action scope is not an external process (out of Phase 11)")
    elevation_required = action.privilege is PrivilegeRequirement.ELEVATION_REQUIRED
    if elevation_required:
        reasons.append("action requires elevation, which is never automatic")
        if policy.elevation_never_automatic and not elevation_available:
            reasons.append("elevation is not available and will not be requested silently")
    implemented = bool(action.implemented and action.phase11_allowed)
    if not implemented:
        reasons.append("action is recommendation-only / not implemented in Phase 11")
    if not action.verification_supported:
        reasons.append("action does not support independent verification")
    if action.external_safety_class is ExternalActionSafetyClass.UNSUPPORTED:
        reasons.append("action is classified unsupported")

    allowed = (
        implemented
        and action.scope is OptimizationScope.EXTERNAL_PROCESS
        and not elevation_required
        and action.verification_supported
        and identity is not None
        and not identity.potentially_stale
    )
    if allowed:
        reasons.append("action is registered, implemented, verifiable, and standard-user")
    return ExternalActionSafety(
        action_id=action.action_id,
        recommendation_id="",
        identity_key=identity.identity_key if identity else None,
        allowed=allowed,
        requires_approval=action.requires_human_approval or policy.require_approval_for_actions,
        elevation_required=elevation_required,
        implemented=implemented,
        phase11_allowed=action.phase11_allowed,
        reversible=action.reversibility.value == "reversible",
        verification_supported=action.verification_supported,
        safety_class=action.external_safety_class,
        reasons=reasons,
    )


def capture_external_state(
    identity: ExternalProcessIdentity, action_id: str
) -> dict[str, object]:
    """Capture only the fields the supported action can actually restore."""
    if psutil is None:  # pragma: no cover - hard dependency
        return {}
    if action_id != "external.reduce_priority":
        return {}
    try:
        process = psutil.Process(identity.pid)
        if not verify_identity(identity):
            return {}
        return {"nice": int(process.nice()), "pid": identity.pid, "guard": identity.pid_reuse_guard}
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return {}


class ExternalActionExecutor:
    """Executes only the minimal supported external action, after revalidation."""

    def __init__(self, *, own_user: str | None = None) -> None:
        self._own_user = own_user

    def execute(
        self,
        *,
        action_id: str,
        identity: ExternalProcessIdentity,
        parameters: dict[str, object],
        checkpoint: ExternalActionCheckpoint | None,
    ) -> ExternalActionResult:
        started = time.perf_counter()
        result = ExternalActionResult(
            action_result_id=f"extact-{uuid.uuid4().hex[:10]}",
            action_id=action_id,
            checkpoint_id=checkpoint.checkpoint_id if checkpoint else None,
            attempted=True,
            identity_before=identity,
            started_at=datetime.now(timezone.utc),
        )
        if action_id != "external.reduce_priority":
            return result.model_copy(
                update={
                    "executed": False,
                    "reason": f"action {action_id!r} has no safe external executor",
                    "finished_at": datetime.now(timezone.utc),
                    "duration_s": time.perf_counter() - started,
                }
            )
        if psutil is None:  # pragma: no cover - hard dependency
            return result.model_copy(
                update={"executed": False, "reason": "psutil unavailable"}
            )

        # 1. Re-resolve and revalidate the process identity immediately before acting.
        if not verify_identity(identity):
            return result.model_copy(
                update={
                    "executed": False,
                    "reason": "process identity changed or disappeared; recommendation is stale",
                    "finished_at": datetime.now(timezone.utc),
                    "duration_s": time.perf_counter() - started,
                }
            )
        try:
            process = psutil.Process(identity.pid)
            owner = process.username()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as exc:
            return result.model_copy(
                update={
                    "executed": False,
                    "reason": f"process not accessible: {type(exc).__name__}",
                    "finished_at": datetime.now(timezone.utc),
                    "duration_s": time.perf_counter() - started,
                }
            )
        own_user = self._own_user or _current_user()
        if own_user is not None and owner is not None and owner != own_user:
            return result.model_copy(
                update={
                    "executed": False,
                    "reason": "process is owned by another user; elevation is not automatic",
                    "finished_at": datetime.now(timezone.utc),
                    "duration_s": time.perf_counter() - started,
                }
            )
        target = str(parameters.get("priority_class") or "below_normal")
        if target not in _PRIORITY_CLASSES:
            return result.model_copy(
                update={"executed": False, "reason": f"unknown priority class {target!r}"}
            )
        try:
            process.nice(_PRIORITY_CLASSES[target])
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError) as exc:
            return result.model_copy(
                update={
                    "executed": False,
                    "reason": f"could not change priority: {type(exc).__name__}",
                    "finished_at": datetime.now(timezone.utc),
                    "duration_s": time.perf_counter() - started,
                }
            )
        try:
            observed = int(process.nice())
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            observed = -1
        from inferenceos.external.identity import resolve_identity

        identity_after = resolve_identity(identity.pid)
        prior_nice = checkpoint.prior_state.get("nice") if checkpoint is not None else None
        changes: dict[str, object] = {
            "priority_class": target,
            "nice_before": prior_nice,
            "nice_after": observed,
        }
        return result.model_copy(
            update={
                "executed": True,
                "identity_after": identity_after,
                "changes": changes,
                "reason": "priority class reduced",
                "finished_at": datetime.now(timezone.utc),
                "duration_s": time.perf_counter() - started,
            }
        )

    def restore(
        self,
        *,
        action_id: str,
        identity: ExternalProcessIdentity,
        checkpoint: ExternalActionCheckpoint | None,
    ) -> bool:
        if psutil is None or checkpoint is None:  # pragma: no cover - hard dependency
            return False
        prior = checkpoint.prior_state.get("nice")
        if not isinstance(prior, int):
            return False
        if not verify_identity(identity):
            return False
        try:
            process = psutil.Process(identity.pid)
            process.nice(prior)
            return int(process.nice()) == prior
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
            return False


def _current_user() -> str | None:
    if psutil is None:  # pragma: no cover - hard dependency
        return None
    try:
        return psutil.Process().username()
    except (psutil.AccessDenied, OSError):  # pragma: no cover - defensive
        return None
