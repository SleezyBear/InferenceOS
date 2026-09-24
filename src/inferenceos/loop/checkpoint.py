"""Protected baseline checkpoint capture/restore/verify.

Phase 9 never claims to roll back the whole machine. The checkpoint covers exactly the
optimization-controlled state InferenceOS mutates — the target model's runtime
configuration — and the Phase 7 :class:`ActionExecutor` remains authoritative for
restoring it.
"""

from __future__ import annotations

import uuid

from inferenceos.contracts.enums import RestorationStatus
from inferenceos.contracts.experiment import RestorationResult, RuntimeConfiguration
from inferenceos.contracts.optimization_loop import SafetyCheckpoint
from inferenceos.experiment.executor import ActionExecutor


def capture_checkpoint(
    *,
    configuration: RuntimeConfiguration,
    model_ref: str | None = None,
    model_path: str | None = None,
    hardware_hostname: str | None = None,
    workload_id: str | None = None,
    diagnosis_id: str | None = None,
    context_identity: str | None = None,
    baseline_metrics: dict[str, float] | None = None,
    notes: list[str] | None = None,
) -> SafetyCheckpoint:
    """Capture the protected configuration before any mutating experiment."""
    return SafetyCheckpoint(
        checkpoint_id=f"ckpt-{uuid.uuid4().hex[:10]}",
        runtime_configuration=configuration,
        model_ref=model_ref,
        model_path=model_path,
        hardware_hostname=hardware_hostname,
        workload_id=workload_id,
        diagnosis_id=diagnosis_id,
        context_identity=context_identity,
        baseline_metrics=dict(baseline_metrics or {}),
        notes=list(notes or []),
    )


def verify_checkpoint(executor: ActionExecutor, checkpoint: SafetyCheckpoint) -> bool:
    """Whether the executor currently holds exactly the protected configuration."""
    return executor.current() == checkpoint.runtime_configuration


def restore_checkpoint(
    executor: ActionExecutor, checkpoint: SafetyCheckpoint
) -> RestorationResult:
    """Restore the protected configuration through the Phase 7 executor."""
    try:
        result = executor.restore(checkpoint.runtime_configuration)
    except Exception as exc:  # noqa: BLE001 - never leave a mutation uncontrolled
        from datetime import datetime, timezone

        return RestorationResult(
            attempted=True,
            status=RestorationStatus.FAILED,
            expected_configuration=checkpoint.runtime_configuration,
            restored_configuration=executor.current(),
            verified=False,
            reason=f"restoration raised: {type(exc).__name__}: {exc}",
            restored_at=datetime.now(timezone.utc),
        )
    return result
