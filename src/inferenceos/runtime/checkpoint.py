"""Runtime checkpoint capture / verification / restore.

The checkpoint covers the controlled workload's runtime configuration + identity +
resume information. Under the process-per-generation llama.cpp backend it is explicitly
*not* an OS/process memory snapshot; the limitations are recorded on the checkpoint.

A checkpoint is only reported as present/verified after it has actually been
established and re-checked; a failed checkpoint never interrupts the workload.
"""

from __future__ import annotations

import hashlib
import json
import uuid

from inferenceos.contracts.enums import CheckpointStatus
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.runtime import RuntimeCheckpoint
from inferenceos.runtime.workload import ActiveWorkload


def _integrity(configuration: RuntimeConfiguration) -> str:
    material = json.dumps(configuration.as_dict(), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def capture_runtime_checkpoint(
    workload: ActiveWorkload,
    *,
    reason: str,
    context_identity: str | None,
    previous_baseline_id: str | None = None,
) -> RuntimeCheckpoint:
    descriptor = workload.descriptor()
    snapshot = workload.snapshot()
    configuration = workload.configuration()
    if not descriptor.checkpoint_supported:
        return RuntimeCheckpoint(
            checkpoint_id=f"rtckpt-{uuid.uuid4().hex[:10]}",
            status=CheckpointStatus.FAILED,
            configuration=configuration,
            workload_id=descriptor.workload_id,
            model_ref=descriptor.model_ref,
            backend_name=descriptor.backend_name,
            context_identity=context_identity,
            reason=reason,
            resume_supported=False,
            integrity=_integrity(configuration),
            verified=False,
            previous_baseline_id=previous_baseline_id,
            limitations=list(descriptor.limitations)
            + ["checkpoint is not supported by this workload"],
        )
    checkpoint = RuntimeCheckpoint(
        checkpoint_id=f"rtckpt-{uuid.uuid4().hex[:10]}",
        status=CheckpointStatus.CAPTURED,
        configuration=configuration,
        workload_id=descriptor.workload_id,
        model_ref=descriptor.model_ref,
        backend_name=descriptor.backend_name,
        context_identity=context_identity,
        reason=reason,
        resume_supported=descriptor.resume_supported and snapshot.can_resume,
        resume_token=f"resume-{uuid.uuid4().hex[:10]}" if descriptor.resume_supported else None,
        integrity=_integrity(configuration),
        verified=False,
        previous_baseline_id=previous_baseline_id,
        limitations=list(descriptor.limitations),
    )
    verified = verify_runtime_checkpoint(workload, checkpoint)
    status = CheckpointStatus.CAPTURED if verified else CheckpointStatus.FAILED
    return checkpoint.model_copy(update={"verified": verified, "status": status})


def verify_runtime_checkpoint(workload: ActiveWorkload, checkpoint: RuntimeCheckpoint) -> bool:
    current = workload.configuration()
    snapshot = workload.snapshot()
    return (
        checkpoint.integrity == _integrity(current)
        and current == checkpoint.configuration
        and snapshot.can_resume
        and not snapshot.terminated_externally
    )


def restore_runtime_checkpoint(workload: ActiveWorkload, checkpoint: RuntimeCheckpoint) -> bool:
    workload.set_configuration(checkpoint.configuration)
    return verify_runtime_checkpoint(workload, checkpoint)
