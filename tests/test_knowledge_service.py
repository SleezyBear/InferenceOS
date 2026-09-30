"""Knowledge service: outcome mapping, persistence, history, events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from inferenceos.contracts.enums import (
    ExperimentDecisionKind,
    KnowledgeEventType,
    OptimizationOutcome,
)
from inferenceos.contracts.experiment import ExperimentResult
from inferenceos.knowledge.service import outcome_from_result
from tests.knowledge_helpers import base_result, make_service, with_decision

_EXPECTED = {
    ExperimentDecisionKind.IMPROVED: OptimizationOutcome.IMPROVED,
    ExperimentDecisionKind.NO_MEANINGFUL_IMPROVEMENT: OptimizationOutcome.NO_MEANINGFUL_IMPROVEMENT,
    ExperimentDecisionKind.REGRESSION: OptimizationOutcome.REGRESSION,
    ExperimentDecisionKind.QUALITY_REGRESSION: OptimizationOutcome.QUALITY_REGRESSION,
    ExperimentDecisionKind.INVALID: OptimizationOutcome.INVALID,
    ExperimentDecisionKind.MEASUREMENT_FAILURE: OptimizationOutcome.INVALID,
    ExperimentDecisionKind.PRECONDITION_FAILURE: OptimizationOutcome.BLOCKED,
    ExperimentDecisionKind.APPLICATION_FAILURE: OptimizationOutcome.BLOCKED,
    ExperimentDecisionKind.UNSAFE: OptimizationOutcome.BLOCKED,
    ExperimentDecisionKind.REJECTED: OptimizationOutcome.BLOCKED,
    ExperimentDecisionKind.RESTORATION_FAILURE: OptimizationOutcome.FAILED,
}


def test_outcome_mapping_never_boolean() -> None:
    for kind, expected in _EXPECTED.items():
        assert outcome_from_result(with_decision(base_result(), kind)) is expected


def test_success_and_negative_outcomes_are_all_retained() -> None:
    service = make_service()
    kinds = [
        ExperimentDecisionKind.IMPROVED,
        ExperimentDecisionKind.NO_MEANINGFUL_IMPROVEMENT,
        ExperimentDecisionKind.REGRESSION,
        ExperimentDecisionKind.QUALITY_REGRESSION,
        ExperimentDecisionKind.INVALID,
    ]
    for index, kind in enumerate(kinds):
        result = with_decision(base_result(), kind)
        assert result.request is not None
        request = result.request.model_copy(update={"experiment_id": f"exp-{index}"})
        result = result.model_copy(update={"experiment_id": f"exp-{index}", "request": request})
        assert service.persist_result(result).persisted
    outcomes = {record.outcome.value for record in service.all_records(limit=100)}
    assert {
        "improved",
        "no_meaningful_improvement",
        "regression",
        "quality_regression",
        "invalid",
    } <= outcomes


def test_blocked_result_has_no_rollback() -> None:
    service = make_service()
    result = with_decision(base_result(), ExperimentDecisionKind.UNSAFE)
    result = result.model_copy(
        update={"application": None, "restoration": None, "environment_restored": False}
    )
    outcome = service.persist_result(result)
    assert outcome.persisted
    assert outcome.rollback_record is None
    assert outcome.record is not None and outcome.record.outcome is OptimizationOutcome.BLOCKED


def test_historical_sequence_preserved_oldest_first() -> None:
    service = make_service()
    first = base_result()
    first = first.model_copy(update={"created_at": datetime(2026, 1, 1, tzinfo=timezone.utc)})
    second = base_result()
    assert second.request is not None
    request = second.request.model_copy(update={"experiment_id": "exp-2"})
    second = second.model_copy(
        update={
            "experiment_id": "exp-2",
            "request": request,
            "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        }
    )
    service.persist_result(first)
    service.persist_result(second)
    history = service.history("runtime.enable_mmap")
    assert [record.experiment_id for record in history] == ["exp-1", "exp-2"]


def test_revalidation_and_revert_events_are_append_only() -> None:
    service = make_service()
    result = base_result()
    service.persist_result(result)
    record_id = service.build_record(result).record_id
    service.mark_revalidation_required(record_id, "hardware changed")
    service.mark_reverted(record_id, "workload changed")
    events = service.store.events_for_record(record_id)
    types = [event.event_type for event in events]
    assert KnowledgeEventType.RECORDED in types
    assert KnowledgeEventType.REVALIDATION_MARKED in types
    assert KnowledgeEventType.REVERTED in types


def test_rollback_created_and_linked_for_mutating_experiment() -> None:
    service = make_service()
    result = base_result()
    outcome = service.persist_result(result)
    assert outcome.rollback_record is not None
    assert outcome.rollback is not None and outcome.rollback.status.value == "persisted"
    rollback = service.rollback_for_experiment(result.experiment_id)
    assert rollback is not None
    record_id = service.build_record(result).record_id
    types = [event.event_type for event in service.store.events_for_record(record_id)]
    assert KnowledgeEventType.ROLLBACK_LINKED in types


def test_query_applicable_prioritizes_direct_matches() -> None:
    service = make_service()
    result = base_result()
    service.persist_result(result)
    record = service.build_record(result)
    results = service.query_applicable("runtime.enable_mmap", record.context)
    assert results
    assert results[0].classification.value == "directly_applicable"
