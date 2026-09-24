"""Phase 8 knowledge contracts: existence, round-trip, strictness."""

from __future__ import annotations

import pytest

from inferenceos.contracts.enums import ApplicabilityClass, OptimizationOutcome
from inferenceos.contracts.knowledge import (
    ApplicabilityResult,
    OptimizationRecord,
    RollbackRecord,
)
from inferenceos.selftest import (
    sample_optimization_record,
    sample_rollback_record,
    verify_contract_roundtrip,
)

REQUIRED = {
    "ContextFingerprint",
    "ApplicabilityContext",
    "OptimizationRecord",
    "RollbackVerification",
    "RollbackRecord",
    "DimensionMatch",
    "ApplicabilityResult",
    "KnowledgeQuery",
    "KnowledgeWriteResult",
    "KnowledgeEvent",
    "KnowledgeHealth",
    "StorageStats",
}


def test_phase8_contracts_roundtrip() -> None:
    passed = verify_contract_roundtrip()
    assert REQUIRED.issubset(set(passed))


def test_optimization_record_roundtrips() -> None:
    record = sample_optimization_record()
    assert OptimizationRecord.model_validate_json(record.model_dump_json()) == record


def test_rollback_record_roundtrips() -> None:
    record = sample_rollback_record()
    assert RollbackRecord.model_validate_json(record.model_dump_json()) == record


def test_contracts_reject_unknown_fields() -> None:
    record = sample_optimization_record()
    with pytest.raises(ValueError):
        OptimizationRecord.model_validate({**record.model_dump(), "unexpected": 1})
    applicable = ApplicabilityResult(
        classification=ApplicabilityClass.DIRECTLY_APPLICABLE,
        record_id="r",
        experiment_id="e",
        action_id="runtime.right_size_context",
        record_outcome=OptimizationOutcome.IMPROVED,
    )
    with pytest.raises(ValueError):
        ApplicabilityResult.model_validate({**applicable.model_dump(), "unexpected": 1})


def test_optimization_record_preserves_context_and_outcome() -> None:
    record = sample_optimization_record()
    assert record.outcome.value == "improved"
    assert record.context.fingerprint.identity_hash
    assert record.measured_benefit
    assert record.rollback_id == "rb-exp-1"
