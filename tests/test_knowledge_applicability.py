"""Contextual applicability: exact, partial, materially different, unknown."""

from __future__ import annotations

from inferenceos.contracts.enums import ApplicabilityClass, RevalidationTrigger
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.knowledge import OptimizationRecord
from inferenceos.knowledge import KnowledgeService, evaluate_applicability
from tests.knowledge_helpers import (
    altered_hardware,
    altered_model,
    base_result,
    current_context,
    make_service,
    other_workload,
)


def _record() -> tuple[KnowledgeService, OptimizationRecord]:
    service = make_service()
    return service, service.build_record(base_result())


def test_exact_context_is_directly_applicable() -> None:
    service, record = _record()
    result = evaluate_applicability(record, current_context(record), service.policy)
    assert result.classification is ApplicabilityClass.DIRECTLY_APPLICABLE
    assert result.score == 1.0
    assert result.differences == []


def test_transient_conditions_do_not_invalidate_identity() -> None:
    service, record = _record()
    changed = current_context(record, conditions={"memory_pressure": 0.55, "available_ram_bytes": 7e9})
    result = evaluate_applicability(record, changed, service.policy)
    # A non-memory diagnosis would remain directly applicable; here the diagnosis is a
    # memory-pressure diagnosis, so a materially different condition is conditional.
    assert result.classification in (
        ApplicabilityClass.DIRECTLY_APPLICABLE,
        ApplicabilityClass.CONDITIONALLY_APPLICABLE,
    )


def test_different_workload_is_conditional() -> None:
    service, record = _record()
    result = evaluate_applicability(
        record, current_context(record, workload=other_workload()), service.policy
    )
    assert result.classification is ApplicabilityClass.CONDITIONALLY_APPLICABLE
    assert RevalidationTrigger.WORKLOAD_CHANGE in result.revalidation_triggers


def test_materially_different_hardware_requires_revalidation() -> None:
    service, record = _record()
    result = evaluate_applicability(
        record, current_context(record, hardware=altered_hardware()), service.policy
    )
    assert result.classification is ApplicabilityClass.REQUIRES_REVALIDATION
    assert RevalidationTrigger.HARDWARE_CHANGE in result.revalidation_triggers


def test_different_model_requires_revalidation() -> None:
    service, record = _record()
    result = evaluate_applicability(
        record, current_context(record, model=altered_model()), service.policy
    )
    assert result.classification is ApplicabilityClass.REQUIRES_REVALIDATION
    assert RevalidationTrigger.MODEL_CHANGE in result.revalidation_triggers


def test_changed_runtime_requires_revalidation() -> None:
    service, record = _record()
    runtime = RuntimeConfiguration(context_length=1024, gpu_offload_layers=0)
    result = evaluate_applicability(
        record, current_context(record, runtime=runtime), service.policy
    )
    assert result.classification is ApplicabilityClass.REQUIRES_REVALIDATION
    assert RevalidationTrigger.RUNTIME_CHANGE in result.revalidation_triggers


def test_changed_diagnosis_requires_revalidation() -> None:
    service, record = _record()
    result = evaluate_applicability(
        record, current_context(record, diagnosis="cpu"), service.policy
    )
    assert result.classification is ApplicabilityClass.REQUIRES_REVALIDATION
    assert RevalidationTrigger.DIAGNOSIS_CHANGE in result.revalidation_triggers


def test_multiple_critical_differences_are_not_applicable() -> None:
    service, record = _record()
    result = evaluate_applicability(
        record,
        current_context(record, hardware=altered_hardware(), model=altered_model()),
        service.policy,
    )
    assert result.classification is ApplicabilityClass.NOT_APPLICABLE


def test_missing_context_is_insufficient() -> None:
    service, record = _record()
    result = evaluate_applicability(
        record, current_context(record, include_model=False), service.policy
    )
    assert result.classification is ApplicabilityClass.INSUFFICIENT_CONTEXT


def test_unregistered_action_is_not_applicable() -> None:
    service, record = _record()
    unregistered = record.model_copy(
        update={"action_id": "shell.run_command", "action_registered": False}
    )
    result = evaluate_applicability(unregistered, current_context(record), service.policy)
    assert result.classification is ApplicabilityClass.NOT_APPLICABLE
    assert RevalidationTrigger.ACTION_UNREGISTERED in result.revalidation_triggers


def test_explainability_components_and_reasons_present() -> None:
    service, record = _record()
    result = evaluate_applicability(
        record, current_context(record, hardware=altered_hardware()), service.policy
    )
    assert result.components
    assert result.reasons
    assert result.rationale
    assert result.matches
