"""Audit-record schema tests (persistence + rollback data)."""

from __future__ import annotations

from inferenceos.audit.schema import AUDIT_SCHEMA_VERSION, AuditRecord
from inferenceos.contracts.enums import Decision
from inferenceos.selftest import sample_audit_record


def test_audit_record_json_roundtrip() -> None:
    record = sample_audit_record()
    restored = AuditRecord.from_json(record.to_json())
    assert restored == record
    assert restored.schema_version == AUDIT_SCHEMA_VERSION


def test_audit_record_preserves_rollback_values() -> None:
    record = sample_audit_record()
    change = record.applied_changes[0]
    assert change.parameter == "use_mmap"
    assert change.prior_value is False
    assert change.applied_value is True


def test_audit_record_decision_and_chain_present() -> None:
    record = sample_audit_record()
    assert record.decision is Decision.KEEP
    assert record.baseline is not None
    assert record.diagnosis is not None
    assert record.experiment is not None
    assert record.reasoning_response is not None
