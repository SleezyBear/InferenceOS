"""Rollback records: construction, integrity verification, mismatch rejection."""

from __future__ import annotations

from inferenceos.contracts.enums import RestorationStatus
from inferenceos.contracts.experiment import ExperimentChange, RuntimeConfiguration
from inferenceos.knowledge import build_rollback_record, verify_rollback
from inferenceos.optimization.actions import build_default_registry
from tests.knowledge_helpers import base_result, current_context, make_service


def test_build_rollback_from_real_shape() -> None:
    registry = build_default_registry()
    record = build_rollback_record(base_result(), registry=registry)
    assert record is not None
    assert record.action_id == "runtime.enable_mmap"
    assert record.previous_state.use_mmap is False
    assert record.resulting_state.use_mmap is True
    assert record.restore_parameters == {"use_mmap": False}
    assert record.resulting_state.context_length == record.previous_state.context_length
    assert record.verification.verified is True
    assert record.integrity_hash


def test_verify_rollback_with_matching_context() -> None:
    registry = build_default_registry()
    rollback = build_rollback_record(base_result(), registry=registry)
    assert rollback is not None
    service = make_service()
    context = service.build_record(base_result()).context
    verification = verify_rollback(rollback, registry=registry, current_context=context)
    assert verification.verified is True
    assert verification.context_compatible is True


def test_unregistered_rollback_target_rejected() -> None:
    registry = build_default_registry()
    rollback = build_rollback_record(base_result(), registry=registry)
    assert rollback is not None
    tampered = rollback.model_copy(update={"action_id": "shell.run_command"})
    verification = verify_rollback(tampered, registry=registry)
    assert verification.verified is False
    assert verification.action_registered is False


def test_undeclared_parameter_rejected() -> None:
    registry = build_default_registry()
    rollback = build_rollback_record(base_result(), registry=registry)
    assert rollback is not None
    tampered = rollback.model_copy(
        update={
            "changes": [
                ExperimentChange(parameter="cmd", prior_value="x", applied_value="y")
            ]
        }
    )
    verification = verify_rollback(tampered, registry=registry)
    assert verification.verified is False
    assert verification.parameters_match is False


def test_mismatched_previous_state_rejected() -> None:
    registry = build_default_registry()
    rollback = build_rollback_record(base_result(), registry=registry)
    assert rollback is not None
    tampered = rollback.model_copy(
        update={
            "previous_state": RuntimeConfiguration(context_length=1, gpu_offload_layers=0)
        }
    )
    verification = verify_rollback(tampered, registry=registry)
    assert verification.verified is False
    assert verification.expected_matches_previous is False


def test_restoration_failure_is_represented() -> None:
    registry = build_default_registry()
    result = base_result()
    assert result.restoration is not None
    restoration = result.restoration.model_copy(
        update={"status": RestorationStatus.FAILED, "verified": False}
    )
    failed = result.model_copy(
        update={"restoration": restoration, "environment_restored": False}
    )
    record = build_rollback_record(failed, registry=registry)
    assert record is not None
    assert record.restoration_status is RestorationStatus.FAILED
    assert record.experiment_restored is False


def test_no_rollback_when_no_change() -> None:
    registry = build_default_registry()
    result = base_result().model_copy(update={"application": None})
    assert build_rollback_record(result, registry=registry) is None
