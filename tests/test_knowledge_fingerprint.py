"""Context fingerprinting: stable identity, transient conditions separated."""

from __future__ import annotations

from inferenceos.contracts.knowledge import OptimizationRecord
from inferenceos.knowledge import KnowledgeService, fingerprint_context
from tests.knowledge_helpers import (
    altered_hardware,
    base_result,
    current_context,
    make_service,
)


def _record() -> tuple[KnowledgeService, OptimizationRecord]:
    service = make_service()
    return service, service.build_record(base_result())


def test_fingerprint_is_reproducible() -> None:
    _service, record = _record()
    recomputed = fingerprint_context(record.context)
    assert recomputed.identity_hash == record.context.fingerprint.identity_hash
    assert recomputed.conditions_hash == record.context.fingerprint.conditions_hash


def test_conditions_do_not_change_identity() -> None:
    _service, record = _record()
    first = current_context(record, conditions={"memory_pressure": 0.4, "available_ram_bytes": 8e9})
    second = current_context(record, conditions={"memory_pressure": 0.9, "available_ram_bytes": 2e9})
    assert first.fingerprint.identity_hash == second.fingerprint.identity_hash
    assert first.fingerprint.conditions_hash != second.fingerprint.conditions_hash


def test_material_change_changes_identity() -> None:
    _service, record = _record()
    baseline = current_context(record)
    altered = current_context(record, hardware=altered_hardware())
    assert baseline.fingerprint.identity_hash != altered.fingerprint.identity_hash


def test_runtime_key_ignores_unset_fields() -> None:
    _service, record = _record()
    context = current_context(record)
    assert "context_length" in context.fingerprint.runtime_key
    assert "threads" not in context.fingerprint.runtime_key
