"""Durable SQLite knowledge store: persistence, restart, integrity, failures."""

from __future__ import annotations

import sqlite3

import pytest

from inferenceos.contracts.enums import KnowledgeHealthStatus, KnowledgeWriteStatus
from inferenceos.knowledge import KnowledgeService, SQLiteKnowledgeStore
from inferenceos.knowledge.store import (
    KnowledgeCorruptionError,
    KnowledgeSchemaError,
    KnowledgeUnavailableError,
)
from inferenceos.optimization.actions import build_default_registry
from tests.knowledge_helpers import base_result, make_service


def _reopen(service: KnowledgeService) -> KnowledgeService:
    return KnowledgeService(
        store=SQLiteKnowledgeStore(service.store.db_path),
        registry=build_default_registry(),
        policy=service.policy,
    )


def test_persist_and_readback_across_restart() -> None:
    service = make_service()
    result = base_result()
    outcome = service.persist_result(result)
    assert outcome.optimization.status is KnowledgeWriteStatus.PERSISTED

    reopened = _reopen(service)
    record = reopened.store.get_by_experiment(result.experiment_id)
    assert record is not None
    assert record.experiment_id == result.experiment_id
    assert record.context.fingerprint.identity_hash


def test_duplicate_persist_is_idempotent() -> None:
    service = make_service()
    result = base_result()
    assert service.persist_result(result).optimization.status is KnowledgeWriteStatus.PERSISTED
    before = service.stats().optimization_records
    again = service.persist_result(result)
    assert again.optimization.status is KnowledgeWriteStatus.DUPLICATE
    assert again.optimization.duplicate is True
    assert service.stats().optimization_records == before == 1


def test_conflicting_payload_returns_conflict_not_overwrite() -> None:
    service = make_service()
    result = base_result()
    record = service.build_record(result)
    assert service.store.record_optimization(record).status is KnowledgeWriteStatus.PERSISTED
    conflicting = record.model_copy(update={"candidate_id": "cand-other"})
    outcome = service.store.record_optimization(conflicting)
    assert outcome.status is KnowledgeWriteStatus.CONFLICT
    stored = service.store.get_by_experiment(result.experiment_id)
    assert stored is not None and stored.candidate_id == record.candidate_id


def test_schema_mismatch_detected() -> None:
    service = make_service()
    with pytest.raises(KnowledgeSchemaError):
        SQLiteKnowledgeStore(service.store.db_path, schema_version="999")


def test_corruption_detected_on_read_and_health() -> None:
    service = make_service()
    result = base_result()
    service.persist_result(result)
    record_id = service.build_record(result).record_id
    assert service.store.record_integrity(record_id) is True

    connection = sqlite3.connect(str(service.store.db_path))
    connection.execute(
        "UPDATE optimization_records SET payload_json = payload_json || ' ' "
        "WHERE experiment_id = ?",
        (result.experiment_id,),
    )
    connection.commit()
    connection.close()

    health = service.store.health()
    assert health.status is KnowledgeHealthStatus.CORRUPT
    assert record_id in health.corrupted_records
    with pytest.raises(KnowledgeCorruptionError):
        service.store.get_by_experiment(result.experiment_id)
    assert service.store.record_integrity(record_id) is False


def test_write_failure_is_returned_not_fabricated() -> None:
    service = make_service()
    record = service.build_record(base_result())
    connection = sqlite3.connect(str(service.store.db_path))
    connection.execute("DROP TABLE optimization_records")
    connection.commit()
    connection.close()
    outcome = service.store.record_optimization(record)
    assert outcome.status is KnowledgeWriteStatus.FAILED
    assert "failed" in outcome.detail.lower()


def test_unavailable_store_path_raises() -> None:
    tmp = make_service().store.db_path.parent
    with pytest.raises(KnowledgeUnavailableError):
        SQLiteKnowledgeStore(tmp)


def test_health_and_stats() -> None:
    service = make_service()
    service.persist_result(base_result())
    health = service.health()
    assert health.status is KnowledgeHealthStatus.OK
    assert health.record_count == 1 and health.rollback_count == 1
    stats = service.stats()
    assert stats.optimization_records == 1
    assert stats.schema_version == "1.0"
