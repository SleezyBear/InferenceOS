"""External action checkpoint: capture only what the action can actually restore.

The checkpoint does **not** claim to snapshot arbitrary process memory/state; it
records only the observable, configurable fields the supported action can restore.
"""

from __future__ import annotations

import hashlib
import json
import uuid

from inferenceos.contracts.external import ExternalActionCheckpoint, ExternalProcessIdentity
from inferenceos.external.actions import capture_external_state
from inferenceos.external.identity import verify_identity


def _integrity(prior_state: dict[str, object]) -> str:
    material = json.dumps(prior_state, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def capture_external_checkpoint(
    identity: ExternalProcessIdentity,
    action_id: str,
    *,
    target_workload_state: dict[str, float] | None = None,
    approval_id: str | None = None,
) -> ExternalActionCheckpoint:
    prior_state = capture_external_state(identity, action_id)
    reversible = action_id == "external.reduce_priority"
    verified = bool(prior_state) and verify_identity(identity)
    return ExternalActionCheckpoint(
        checkpoint_id=f"extckpt-{uuid.uuid4().hex[:10]}",
        identity=identity,
        action_id=action_id,
        prior_state=prior_state,
        configurable_fields=sorted(prior_state.keys()),
        reversible=reversible,
        restore_supported=bool(prior_state),
        target_workload_state=dict(target_workload_state or {}),
        approval_id=approval_id,
        verified=verified,
        integrity=_integrity(prior_state) if prior_state else None,
        limitations=[
            "checkpoint covers only observable configurable fields",
            "no process memory or arbitrary application state is captured",
            "if the process exits before restore, restoration is not possible",
        ],
    )
