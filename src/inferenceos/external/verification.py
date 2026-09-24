"""Independent verification of an external action outcome.

Verification failure is never reported as success. If the supported action requires
restoration and restoration cannot be verified, the result is marked unverified and a
failure reason is surfaced.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from inferenceos.contracts.external import (
    ExternalActionCheckpoint,
    ExternalActionResult,
    ExternalProcessIdentity,
    ExternalVerification,
)


def _relative_improvement(metric: str, before: float, after: float) -> float | None:
    if before == 0.0:
        return None
    if metric in {"decode_tokens_per_sec", "prompt_tokens_per_sec", "gpu_utilization_percent"}:
        return (after - before) / abs(before)
    return (before - after) / abs(before)


def verify_external_action(
    *,
    action_id: str,
    identity: ExternalProcessIdentity,
    action_result: ExternalActionResult,
    checkpoint: ExternalActionCheckpoint | None,
    executor: object,
    restore_after: bool,
    target_healthy: bool = True,
    before_metrics: dict[str, float] | None = None,
    after_metrics: dict[str, float] | None = None,
    improvement_metric: str = "decode_tokens_per_sec",
    improvement_threshold: float = 0.0,
) -> ExternalVerification:
    from inferenceos.external.actions import ExternalActionExecutor

    notes: list[str] = []
    external_state_changed = bool(
        action_result.executed
        and action_result.changes.get("nice_after") is not None
        and action_result.changes.get("nice_after") != -1
    )
    if not action_result.executed:
        notes.append(f"action not executed: {action_result.reason}")

    materially_helped: bool | None = None
    target_state_observed = bool(before_metrics and after_metrics)
    if target_state_observed and before_metrics is not None and after_metrics is not None:
        before = before_metrics.get(improvement_metric)
        after = after_metrics.get(improvement_metric)
        if before is not None and after is not None:
            relative = _relative_improvement(improvement_metric, before, after)
            materially_helped = relative is not None and relative > improvement_threshold
            notes.append(
                f"target {improvement_metric}: {before:.4g} -> {after:.4g} "
                f"(relative {relative})"
            )

    restoration_attempted = False
    restoration_verified = False
    if restore_after and action_result.executed:
        if checkpoint is None or not checkpoint.restore_supported:
            notes.append("restoration requested but no restorable checkpoint exists")
        elif isinstance(executor, ExternalActionExecutor):
            restoration_attempted = True
            restoration_verified = executor.restore(
                action_id=action_id, identity=identity, checkpoint=checkpoint
            )
            if not restoration_verified:
                notes.append("restoration could not be verified")

    failure_reason: str | None = None
    if not action_result.executed:
        failure_reason = action_result.reason or "action not executed"
    elif not external_state_changed:
        failure_reason = "external state change could not be verified"
    elif not target_healthy:
        failure_reason = "target workload did not remain healthy after the action"
    elif restoration_attempted and not restoration_verified:
        failure_reason = "restoration could not be verified"

    verified = failure_reason is None
    return ExternalVerification(
        verification_id=f"extver-{uuid.uuid4().hex[:10]}",
        action_result_id=action_result.action_result_id,
        created_at=datetime.now(timezone.utc),
        external_state_changed=external_state_changed,
        external_state_restored=restoration_verified,
        target_state_observed=target_state_observed,
        target_healthy=target_healthy,
        materially_helped=materially_helped,
        verified=verified,
        failure_reason=failure_reason,
        measurement={
            key: value
            for key, value in (after_metrics or {}).items()
            if isinstance(value, (int, float))
        },
        restoration_attempted=restoration_attempted,
        restoration_verified=restoration_verified,
        notes=notes,
    )
