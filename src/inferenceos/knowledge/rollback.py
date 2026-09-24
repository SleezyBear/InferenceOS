"""Rollback record construction, integrity verification, and real applicability.

Phase 7R produces a real ``ExperimentResult`` (with a ``RestorationVerification``).
This module turns it into a self-contained :class:`RollbackRecord` that carries:

* **what changed** (``changes``),
* **what must be restored** (``previous_state`` / ``restore_parameters``),
* **how restoration was verified** (``restoration_verification``),
* the **identity** needed to reject an incompatible/stale record before execution.

Phase 8R adds ``classify_rollback_applicability`` and public integrity sealing so the
real executor (``knowledge.rollback_executor``) can decide whether to apply a record.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from inferenceos.contracts.enums import (
    RestorationStatus,
    RollbackApplicabilityClass,
)
from inferenceos.contracts.experiment import ExperimentResult, RuntimeConfiguration
from inferenceos.contracts.knowledge import (
    ApplicabilityContext,
    RollbackRecord,
    RollbackVerification,
)
from inferenceos.contracts.rollback import RollbackApplicability
from inferenceos.knowledge.fingerprint import (
    build_context_from_request,
    runtime_key,
)
from inferenceos.knowledge.policy import KnowledgePolicy
from inferenceos.optimization.registry import ActionRegistry


def _integrity_material(record: RollbackRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "rollback_id": record.rollback_id,
        "experiment_id": record.experiment_id,
        "action_id": record.action_id,
        "target_scope": record.target_scope,
        "previous_state": record.previous_state.model_dump(),
        "resulting_state": record.resulting_state.model_dump(),
        "expected_configuration": record.expected_configuration.model_dump(),
        "restore_parameters": record.restore_parameters,
        "changes": [change.model_dump() for change in record.changes],
        "model_key": record.model_key,
        "backend_key": record.backend_key,
        "hardware_key": record.hardware_key,
        "workload_key": record.workload_key,
    }


def rollback_integrity_hash(record: RollbackRecord) -> str:
    encoded = json.dumps(
        _integrity_material(record), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def seal_rollback_record(record: RollbackRecord) -> RollbackRecord:
    """Return a copy with a correct integrity hash for its current fields."""
    return record.model_copy(update={"integrity_hash": rollback_integrity_hash(record)})


def verify_rollback_integrity(record: RollbackRecord) -> bool:
    if not record.integrity_hash:
        return False
    return record.integrity_hash == rollback_integrity_hash(record)


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
    fingerprint = context.fingerprint
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
        context_identity=fingerprint.identity_hash,
        verification=RollbackVerification(
            verified=False,
            action_registered=False,
            parameters_match=False,
            expected_matches_previous=False,
            context_compatible=None,
            reasons=[],
        ),
        model_key=fingerprint.model_key,
        model_content_hash=fingerprint.model_content_hash,
        backend_key=fingerprint.backend_key,
        hardware_key=fingerprint.hardware_key,
        workload_key=fingerprint.workload_key,
        previous_runtime_key=runtime_key(previous_state),
        resulting_runtime_key=runtime_key(resulting_state),
        restoration_verification=(
            restoration.verification if restoration is not None else None
        ),
    )
    record = seal_rollback_record(record)
    verification = verify_rollback(record, registry=registry, verify_integrity=True)
    record = record.model_copy(update={"verification": verification})
    return seal_rollback_record(record)


def verify_rollback(
    record: RollbackRecord,
    *,
    registry: ActionRegistry,
    current_context: ApplicabilityContext | None = None,
    verify_integrity: bool = False,
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

    integrity_ok: bool | None = None
    if verify_integrity:
        integrity_ok = verify_rollback_integrity(record)
        if not integrity_ok:
            reasons.append("rollback integrity hash does not match its fields")

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
        and integrity_ok is not False
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


def _changes_at_prior(configuration: RuntimeConfiguration, record: RollbackRecord) -> bool:
    return all(
        getattr(configuration, change.parameter, None) == change.prior_value
        for change in record.changes
        if change.parameter in RuntimeConfiguration.model_fields
    )


def _changes_at_result(configuration: RuntimeConfiguration, record: RollbackRecord) -> bool:
    return all(
        getattr(configuration, change.parameter, None) == change.applied_value
        for change in record.changes
        if change.parameter in RuntimeConfiguration.model_fields
    )


def classify_rollback_applicability(
    record: RollbackRecord,
    current: ApplicabilityContext,
    policy: KnowledgePolicy,
    *,
    registry: ActionRegistry,
    now: datetime | None = None,
) -> RollbackApplicability:
    """Classify whether a persisted rollback may be executed against the live target."""
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []

    integrity_ok = verify_rollback_integrity(record)
    if not integrity_ok:
        reasons.append("integrity hash mismatch: the rollback record was tampered with")
        return RollbackApplicability(
            classification=RollbackApplicabilityClass.INTEGRITY_FAILURE,
            rollback_id=record.rollback_id,
            experiment_id=record.experiment_id,
            integrity_ok=False,
            reasons=reasons,
            checked_at=now,
        )

    if registry.get_optional(record.action_id) is None:
        reasons.append(f"action {record.action_id!r} is not registered")
        return RollbackApplicability(
            classification=RollbackApplicabilityClass.INCOMPATIBLE_TARGET,
            rollback_id=record.rollback_id,
            experiment_id=record.experiment_id,
            integrity_ok=True,
            reasons=reasons,
            checked_at=now,
        )

    current_fp = current.fingerprint
    if current.model is None or current.hardware is None:
        reasons.append("current target/model identity is unavailable")
        return RollbackApplicability(
            classification=RollbackApplicabilityClass.INSUFFICIENT_EVIDENCE,
            rollback_id=record.rollback_id,
            experiment_id=record.experiment_id,
            integrity_ok=True,
            reasons=reasons,
            checked_at=now,
        )

    model_match = (
        record.model_key is None or record.model_key == current_fp.model_key
    )
    model_conflict = bool(
        record.model_key
        and record.model_key == current_fp.model_key
        and record.model_content_hash
        and current_fp.model_content_hash
        and record.model_content_hash != current_fp.model_content_hash
    )
    backend_match = (
        record.backend_key is None or record.backend_key == current_fp.backend_key
    )
    hardware_match = (
        record.hardware_key is None or record.hardware_key == current_fp.hardware_key
    )
    current_runtime = current.runtime_configuration
    already_restored = _changes_at_prior(current_runtime, record)
    in_resulting_state = _changes_at_result(current_runtime, record)
    stale = False
    if policy.revalidation_max_age_days > 0:
        age_days = (now - record.created_at).total_seconds() / 86400.0
        stale = age_days > policy.revalidation_max_age_days

    if already_restored:
        reasons.append("target already holds the rollback's previous configuration")
        classification = RollbackApplicabilityClass.ALREADY_RESTORED
    elif not model_match or model_conflict:
        reasons.append("model identity changed since the rollback was recorded")
        classification = RollbackApplicabilityClass.INCOMPATIBLE_MODEL
    elif not backend_match:
        reasons.append("backend identity changed since the rollback was recorded")
        classification = RollbackApplicabilityClass.INCOMPATIBLE_BACKEND
    elif not hardware_match:
        reasons.append("hardware identity changed since the rollback was recorded")
        classification = RollbackApplicabilityClass.INCOMPATIBLE_TARGET
    elif stale:
        reasons.append("rollback record is older than the revalidation window")
        classification = RollbackApplicabilityClass.STALE
    elif not in_resulting_state:
        reasons.append(
            "target configuration is neither the rollback's resulting nor previous state"
        )
        classification = RollbackApplicabilityClass.INCOMPATIBLE_TARGET
    else:
        reasons.append("target is in the rollback's resulting state; rollback applies")
        classification = RollbackApplicabilityClass.APPLICABLE

    return RollbackApplicability(
        classification=classification,
        rollback_id=record.rollback_id,
        experiment_id=record.experiment_id,
        integrity_ok=True,
        model_match=model_match and not model_conflict,
        backend_match=backend_match,
        hardware_match=hardware_match,
        target_in_resulting_state=in_resulting_state,
        already_restored=already_restored,
        stale=stale,
        reasons=reasons,
        checked_at=now,
    )
