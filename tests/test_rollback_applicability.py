"""Real rollback applicability classification."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from inferenceos.contracts.enums import RollbackApplicabilityClass
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.knowledge import ApplicabilityContext, RollbackRecord
from inferenceos.contracts.profiling import ProfilingConfig
from inferenceos.contracts.rollback import RollbackApplicability
from inferenceos.knowledge import (
    build_rollback_record,
    classify_rollback_applicability,
    seal_rollback_record,
)
from inferenceos.optimization.actions import build_default_registry
from tests.knowledge_helpers import altered_model, base_result, current_context, make_service


def _rollback() -> RollbackRecord:
    record = build_rollback_record(base_result(), registry=build_default_registry())
    assert record is not None
    return record


def _classify(record: RollbackRecord, context: ApplicabilityContext) -> RollbackApplicability:
    return classify_rollback_applicability(
        record, context, make_service().policy, registry=build_default_registry()
    )


def test_applicable_when_target_in_resulting_state() -> None:
    record = _rollback()
    service = make_service()
    context = current_context(
        service.build_record(base_result()), runtime=record.resulting_state
    )
    result = _classify(record, context)
    assert result.classification is RollbackApplicabilityClass.APPLICABLE
    assert result.executable is True


def test_already_restored_when_target_in_previous_state() -> None:
    record = _rollback()
    service = make_service()
    context = current_context(
        service.build_record(base_result()), runtime=record.previous_state
    )
    result = _classify(record, context)
    assert result.classification is RollbackApplicabilityClass.ALREADY_RESTORED
    assert result.already_restored is True


def test_incompatible_model() -> None:
    record = _rollback()
    service = make_service()
    context = current_context(
        service.build_record(base_result()),
        model=altered_model(),
        runtime=record.resulting_state,
    )
    result = _classify(record, context)
    assert result.classification is RollbackApplicabilityClass.INCOMPATIBLE_MODEL


def test_incompatible_backend() -> None:
    record = _rollback()
    service = make_service()
    context = current_context(
        service.build_record(base_result()),
        backend=ProfilingConfig(backend_name="other-backend", backend_version="9.9"),
        runtime=record.resulting_state,
    )
    result = _classify(record, context)
    assert result.classification is RollbackApplicabilityClass.INCOMPATIBLE_BACKEND


def test_incompatible_target_when_runtime_unrelated() -> None:
    from inferenceos.contracts.experiment import RuntimeConfiguration

    record = _rollback()
    service = make_service()
    context = current_context(
        service.build_record(base_result()),
        runtime=RuntimeConfiguration(context_length=1234),
    )
    result = _classify(record, context)
    assert result.classification is RollbackApplicabilityClass.INCOMPATIBLE_TARGET


def test_stale_record() -> None:
    old = datetime.now(timezone.utc) - timedelta(days=90)
    record = seal_rollback_record(_rollback().model_copy(update={"created_at": old}))
    service = make_service()
    context = current_context(
        service.build_record(base_result()), runtime=record.resulting_state
    )
    result = _classify(record, context)
    assert result.classification is RollbackApplicabilityClass.STALE


def test_integrity_failure_takes_precedence() -> None:
    record = _rollback().model_copy(update={"previous_state": record_prev()})
    service = make_service()
    context = current_context(
        service.build_record(base_result()), runtime=record.resulting_state
    )
    result = _classify(record, context)
    assert result.classification is RollbackApplicabilityClass.INTEGRITY_FAILURE
    assert result.executable is False


def record_prev() -> RuntimeConfiguration:
    return RuntimeConfiguration(context_length=1)
