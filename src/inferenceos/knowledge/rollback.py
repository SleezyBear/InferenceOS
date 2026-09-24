"""Rollback record construction and integrity verification.

Phase 7 already restores the baseline after every experiment. Phase 8 records the
prior/resulting configuration and the exact change so a later phase *could* re-apply
the known prior state. Nothing here executes a rollback; it stores values and checks
that a stored rollback is internally consistent and usable.
"""

from __future__ import annotations

import hashlib
import json

from inferenceos.contracts.enums import RestorationStatus
from inferenceos.contracts.experiment import ExperimentResult, RuntimeConfiguration
from inferenceos.contracts.knowledge import (
    ApplicabilityContext,
    RollbackRecord,
    RollbackVerification,
)
from inferenceos.knowledge.fingerprint import build_context_from_request
from inferenceos.optimization.registry import ActionRegistry


def _integrity_hash(record: RollbackRecord) -> str:
    material = {
        "rollback_id": record.rollback_id,
        "experiment_id": record.experiment_id,
        "action_id": record.action_id,
        "restore_parameters": record.restore_parameters,
        "expected": record.expected_configuration.model_dump(),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def build_rollback_record(
    result: ExperimentResult,
    *,
    registry: ActionRegistry,
    rollback_id: str | None = None,
) -> RollbackRecord | None:
    """Build a rollback record from a Phase 7 result, or ``None`` if nothing changed."""
    request = result.request
    if request is None or result.application is None or not result.application.applied:
        return None
    restoration = result.restoration
    previous_state = (
        restoration.expected_configuration
        if restoration is not None and restoration.expected_configuration is not None
        else request.baseline_configuration
    )
    resulting_state = request.experiment_configuration
    action = registry.get_optional(request.action_id)
    target_scope = action.scope.value if action is not None else "target_model_runtime"
    changes = list(result.application.changes)
    restore_parameters = {change.parameter: change.prior_value for change in changes}
    restoration_status = (
        restoration.status if restoration is not None else RestorationStatus.PENDING
    )
    baseline_metrics = (
        result.baseline_snapshot.metrics if result.baseline_snapshot is not None else None
    )
    context = build_context_from_request(request, baseline_metrics=baseline_metrics)
    record = RollbackRecord(
        rollback_id=rollback_id or f"rb-{request.experiment_id}",
        experiment_id=request.experiment_id,
        candidate_id=request.candidate_id,
        action_id=request.action_id,
        target_scope=target_scope,
        previous_state=previous_state,
        resulting_state=resulting_state,
        expected_configuration=previous_state,
        changes=changes,
        restore_parameters=restore_parameters,
        experiment_restored=bool(result.environment_restored),
        restoration_status=restoration_status,
        context_identity=context.fingerprint.identity_hash,
        verification=RollbackVerification(
            verified=False,
            action_registered=False,
            parameters_match=False,
            expected_matches_previous=False,
            context_compatible=None,
            reasons=[],
        ),
    )
    verification = verify_rollback(record, registry=registry)
    record = record.model_copy(update={"verification": verification})
    return record.model_copy(update={"integrity_hash": _integrity_hash(record)})


def verify_rollback(
    record: RollbackRecord,
    *,
    registry: ActionRegistry,
    current_context: ApplicabilityContext | None = None,
) -> RollbackVerification:
    reasons: list[str] = []
    action = registry.get_optional(record.action_id)
    action_registered = action is not None
    if not action_registered:
        reasons.append(f"action {record.action_id!r} is not registered")

    declared = {spec.name for spec in action.parameters} if action is not None else set()
    runtime_fields = set(RuntimeConfiguration.model_fields)
    parameters_match = action_registered and all(
        change.parameter in declared and change.parameter in runtime_fields
        for change in record.changes
    )
    if not parameters_match and action_registered:
        missing = sorted(
            {
                change.parameter
                for change in record.changes
                if change.parameter not in declared
                and change.parameter in runtime_fields
            }
        )
        if missing:
            reasons.append(f"parameters not declared by action: {missing}")
        else:
            reasons.append("rollback parameters are not runtime configuration fields")

    expected_matches_previous = all(
        getattr(record.previous_state, change.parameter, None) == change.prior_value
        for change in record.changes
        if change.parameter in runtime_fields
    )
    if not expected_matches_previous:
        reasons.append("previous_state does not match the recorded prior values")

    context_compatible: bool | None = None
    if current_context is not None and record.context_identity is not None:
        context_compatible = (
            record.context_identity == current_context.fingerprint.identity_hash
        )
        if not context_compatible:
            reasons.append("rollback context differs from the current context")

    verified = (
        action_registered
        and parameters_match
        and expected_matches_previous
        and context_compatible is not False
    )
    if verified and not reasons:
        reasons.append("rollback record consistent and integrity metadata present")
    return RollbackVerification(
        verified=verified,
        action_registered=action_registered,
        parameters_match=parameters_match,
        expected_matches_previous=expected_matches_previous,
        context_compatible=context_compatible,
        reasons=reasons,
    )
