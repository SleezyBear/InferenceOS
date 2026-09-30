"""Durable SQLite knowledge store.

A small local SQLite database (the project's existing durable substrate) holds
immutable optimization records, rollback records, and append-only lifecycle events.
Every write is transactional; every payload carries a checksum so tampering/corruption
is detected on read; ``experiment_id`` uniqueness makes duplicate recording idempotent.

In-memory state is never authoritative: the store opens a fresh connection per
operation, so knowledge survives process restart.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from inferenceos.contracts.enums import (
    KnowledgeEventType,
    KnowledgeHealthStatus,
    KnowledgeWriteStatus,
)
from inferenceos.contracts.knowledge import (
    KNOWLEDGE_SCHEMA_VERSION,
    KnowledgeEvent,
    KnowledgeHealth,
    KnowledgeQuery,
    KnowledgeWriteResult,
    OptimizationRecord,
    RollbackRecord,
    StorageStats,
)

__all__ = [
    "KNOWLEDGE_SCHEMA_VERSION",
    "KnowledgeConflictError",
    "KnowledgeCorruptionError",
    "KnowledgeError",
    "KnowledgeSchemaError",
    "KnowledgeStoreError",
    "KnowledgeUnavailableError",
    "SQLiteKnowledgeStore",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS optimization_records (
    record_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    action_id TEXT NOT NULL,
    diagnosis TEXT,
    model_ref TEXT,
    workload_id TEXT,
    outcome TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    checksum TEXT NOT NULL,
    schema_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rollback_records (
    rollback_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    action_id TEXT NOT NULL,
    target_scope TEXT,
    verified INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    checksum TEXT NOT NULL,
    schema_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_events (
    event_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT,
    data_json TEXT NOT NULL,
    checksum TEXT NOT NULL,
    schema_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_opt_action ON optimization_records(action_id);
CREATE INDEX IF NOT EXISTS idx_opt_model ON optimization_records(model_ref);
CREATE INDEX IF NOT EXISTS idx_opt_diagnosis ON optimization_records(diagnosis);
CREATE INDEX IF NOT EXISTS idx_events_record ON knowledge_events(record_id);
"""


class KnowledgeError(Exception):
    """Base class for persistent-knowledge errors."""


class KnowledgeUnavailableError(KnowledgeError):
    """The store could not be opened/read; historical knowledge is unavailable."""


class KnowledgeStoreError(KnowledgeError):
    """Base class for store write/read failures."""


class KnowledgeSchemaError(KnowledgeStoreError):
    """The on-disk schema version differs from the expected version."""


class KnowledgeCorruptionError(KnowledgeStoreError):
    """A stored record failed its integrity checksum or could not be parsed."""


class KnowledgeConflictError(KnowledgeStoreError):
    """An existing experiment_id has a different payload (integrity conflict)."""


def _checksum(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:64]


class SQLiteKnowledgeStore:
    """Immutable, idempotent, integrity-checked persistence of optimization knowledge."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        schema_version: str = KNOWLEDGE_SCHEMA_VERSION,
        initialize: bool = True,
    ) -> None:
        self._db_path = Path(db_path)
        self._schema_version = schema_version
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if initialize:
            self._initialize()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        try:
            connection = sqlite3.connect(str(self._db_path), timeout=15.0)
        except sqlite3.Error as exc:  # pragma: no cover - environment dependent
            raise KnowledgeUnavailableError(f"cannot open knowledge store: {exc}") from exc
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except sqlite3.Error:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(_SCHEMA)
                row = connection.execute(
                    "SELECT value FROM knowledge_meta WHERE key = 'schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO knowledge_meta(key, value) VALUES ('schema_version', ?)",
                        (self._schema_version,),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO knowledge_meta(key, value) VALUES ('created_at', ?)",
                        (datetime.now(timezone.utc).isoformat(),),
                    )
                elif str(row["value"]) != self._schema_version:
                    raise KnowledgeSchemaError(
                        f"knowledge store schema {row['value']!r} != expected "
                        f"{self._schema_version!r}"
                    )
        except KnowledgeError:
            raise
        except sqlite3.Error as exc:
            raise KnowledgeUnavailableError(
                f"cannot initialize knowledge store at {self._db_path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Writes                                                              #
    # ------------------------------------------------------------------ #
    def record_optimization(self, record: OptimizationRecord) -> KnowledgeWriteResult:
        payload = record.model_dump_json()
        checksum = _checksum(payload)
        model_ref = record.context.model.name if record.context.model is not None else None
        workload_id = (
            record.context.workload.workload_id
            if record.context.workload is not None
            else None
        )
        started = time.perf_counter()
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO optimization_records
                    (record_id, experiment_id, created_at, candidate_id, action_id,
                     diagnosis, model_ref, workload_id, outcome, payload_json, checksum,
                     schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.experiment_id,
                        record.created_at.isoformat(),
                        record.candidate_id,
                        record.action_id,
                        record.diagnosed_bottleneck,
                        model_ref,
                        workload_id,
                        record.outcome.value,
                        payload,
                        checksum,
                        record.schema_version,
                    ),
                )
                if cursor.rowcount == 1:
                    return self._write_result(
                        KnowledgeWriteStatus.PERSISTED,
                        record.experiment_id,
                        record.record_id,
                        False,
                        "record persisted",
                        started,
                    )
                existing = connection.execute(
                    "SELECT record_id, checksum FROM optimization_records "
                    "WHERE experiment_id = ?",
                    (record.experiment_id,),
                ).fetchone()
            if existing is None:  # pragma: no cover - defensive
                return self._write_result(
                    KnowledgeWriteStatus.FAILED,
                    record.experiment_id,
                    None,
                    False,
                    "record not persisted and not found",
                    started,
                )
            if existing["checksum"] == checksum:
                return self._write_result(
                    KnowledgeWriteStatus.DUPLICATE,
                    record.experiment_id,
                    str(existing["record_id"]),
                    True,
                    "identical record already persisted",
                    started,
                )
            return self._write_result(
                KnowledgeWriteStatus.CONFLICT,
                record.experiment_id,
                str(existing["record_id"]),
                True,
                "experiment_id already persisted with a different payload",
                started,
            )
        except KnowledgeError:
            raise
        except sqlite3.Error as exc:
            return self._write_result(
                KnowledgeWriteStatus.FAILED,
                record.experiment_id,
                None,
                False,
                f"write failed: {exc}",
                started,
            )

    def record_rollback(self, record: RollbackRecord) -> KnowledgeWriteResult:
        payload = record.model_dump_json()
        checksum = _checksum(payload)
        started = time.perf_counter()
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO rollback_records
                    (rollback_id, experiment_id, created_at, action_id, target_scope,
                     verified, payload_json, checksum, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.rollback_id,
                        record.experiment_id,
                        record.created_at.isoformat(),
                        record.action_id,
                        record.target_scope,
                        1 if record.verification.verified else 0,
                        payload,
                        checksum,
                        record.schema_version,
                    ),
                )
                if cursor.rowcount == 1:
                    return self._write_result(
                        KnowledgeWriteStatus.PERSISTED,
                        record.experiment_id,
                        record.rollback_id,
                        False,
                        "rollback record persisted",
                        started,
                    )
                existing = connection.execute(
                    "SELECT rollback_id, checksum FROM rollback_records WHERE experiment_id = ?",
                    (record.experiment_id,),
                ).fetchone()
            if existing is not None and existing["checksum"] == checksum:
                return self._write_result(
                    KnowledgeWriteStatus.DUPLICATE,
                    record.experiment_id,
                    str(existing["rollback_id"]),
                    True,
                    "identical rollback already persisted",
                    started,
                )
            return self._write_result(
                KnowledgeWriteStatus.CONFLICT,
                record.experiment_id,
                None if existing is None else str(existing["rollback_id"]),
                True,
                "experiment_id already has a different rollback record",
                started,
            )
        except KnowledgeError:
            raise
        except sqlite3.Error as exc:
            return self._write_result(
                KnowledgeWriteStatus.FAILED,
                record.experiment_id,
                None,
                False,
                f"rollback write failed: {exc}",
                started,
            )

    def append_event(
        self,
        record_id: str,
        event_type: KnowledgeEventType,
        detail: str = "",
        data: dict[str, object] | None = None,
    ) -> KnowledgeEvent:
        event = KnowledgeEvent(
            event_id=f"evt-{uuid.uuid4().hex[:12]}",
            record_id=record_id,
            event_type=event_type,
            detail=detail,
            data=data or {},
        )
        payload = event.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO knowledge_events
                (event_id, record_id, created_at, event_type, detail, data_json,
                 checksum, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.record_id,
                    event.created_at.isoformat(),
                    event.event_type.value,
                    event.detail,
                    payload,
                    _checksum(payload),
                    KNOWLEDGE_SCHEMA_VERSION,
                ),
            )
        return event

    def mark_revalidation_required(self, record_id: str, reason: str) -> KnowledgeEvent:
        return self.append_event(
            record_id, KnowledgeEventType.REVALIDATION_MARKED, reason
        )

    def mark_reverted(self, record_id: str, reason: str) -> KnowledgeEvent:
        return self.append_event(record_id, KnowledgeEventType.REVERTED, reason)

    def link_rollback(self, record_id: str, rollback_id: str) -> KnowledgeEvent:
        return self.append_event(
            record_id,
            KnowledgeEventType.ROLLBACK_LINKED,
            f"rollback={rollback_id}",
            {"rollback_id": rollback_id},
        )

    # ------------------------------------------------------------------ #
    # Reads                                                               #
    # ------------------------------------------------------------------ #
    def get_record(self, record_id: str) -> OptimizationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, checksum FROM optimization_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        return self._parse_record(row) if row is not None else None

    def get_by_experiment(self, experiment_id: str) -> OptimizationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, checksum FROM optimization_records "
                "WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return self._parse_record(row) if row is not None else None

    def find_records(
        self, query: KnowledgeQuery, *, newest_first: bool = True
    ) -> list[OptimizationRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if query.action_id:
            clauses.append("action_id = ?")
            params.append(query.action_id)
        if query.candidate_id:
            clauses.append("candidate_id = ?")
            params.append(query.candidate_id)
        if query.model_ref:
            clauses.append("model_ref = ?")
            params.append(query.model_ref)
        if query.workload_id:
            clauses.append("workload_id = ?")
            params.append(query.workload_id)
        if query.diagnosis:
            clauses.append("diagnosis = ?")
            params.append(query.diagnosis)
        if query.outcome:
            clauses.append("outcome = ?")
            params.append(query.outcome.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = "DESC" if newest_first else "ASC"
        params.append(query.limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT payload_json, checksum FROM optimization_records {where} "
                f"ORDER BY created_at {order} LIMIT ?",
                tuple(params),
            ).fetchall()
        return [self._parse_record(row) for row in rows]

    def history_for_action(self, action_id: str) -> list[OptimizationRecord]:
        return self.find_records(
            KnowledgeQuery(action_id=action_id, limit=1000), newest_first=False
        )

    def all_records(self, *, limit: int = 50) -> list[OptimizationRecord]:
        return self.find_records(KnowledgeQuery(limit=limit), newest_first=True)

    def get_rollback(self, rollback_id: str) -> RollbackRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, checksum FROM rollback_records WHERE rollback_id = ?",
                (rollback_id,),
            ).fetchone()
        return self._parse_rollback(row) if row is not None else None

    def rollback_for_experiment(self, experiment_id: str) -> RollbackRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, checksum FROM rollback_records WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        return self._parse_rollback(row) if row is not None else None

    def all_rollbacks(self, *, limit: int = 50) -> list[RollbackRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json, checksum FROM rollback_records "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._parse_rollback(row) for row in rows]

    def events_for_record(self, record_id: str) -> list[KnowledgeEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT data_json, checksum FROM knowledge_events WHERE record_id = ? "
                "ORDER BY created_at ASC",
                (record_id,),
            ).fetchall()
        events: list[KnowledgeEvent] = []
        for row in rows:
            if _checksum(row["data_json"]) != row["checksum"]:
                raise KnowledgeCorruptionError("knowledge event checksum mismatch")
            events.append(KnowledgeEvent.model_validate_json(row["data_json"]))
        return events

    # ------------------------------------------------------------------ #
    # Integrity / health                                                   #
    # ------------------------------------------------------------------ #
    def record_integrity(self, record_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json, checksum FROM optimization_records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
        if row is None:
            return False
        return _checksum(str(row["payload_json"])) == str(row["checksum"])

    def health(self) -> KnowledgeHealth:
        try:
            with self._connect() as connection:
                version_row = connection.execute(
                    "SELECT value FROM knowledge_meta WHERE key = 'schema_version'"
                ).fetchone()
                version = str(version_row["value"]) if version_row is not None else "unknown"
                opt_rows = connection.execute(
                    "SELECT record_id, payload_json, checksum FROM optimization_records"
                ).fetchall()
                rollback_count = int(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM rollback_records"
                    ).fetchone()["n"]
                )
                event_count = int(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM knowledge_events"
                    ).fetchone()["n"]
                )
        except (sqlite3.Error, KnowledgeError) as exc:
            return KnowledgeHealth(
                status=KnowledgeHealthStatus.UNAVAILABLE,
                schema_version="unknown",
                expected_schema_version=self._schema_version,
                detail=f"knowledge store unavailable: {exc}",
            )
        corrupted: list[str] = []
        for row in opt_rows:
            if _checksum(row["payload_json"]) != row["checksum"]:
                corrupted.append(str(row["record_id"]))
        if version != self._schema_version:
            status = KnowledgeHealthStatus.CORRUPT
            detail = f"schema version {version!r} != expected {self._schema_version!r}"
        elif corrupted:
            status = KnowledgeHealthStatus.CORRUPT
            detail = f"{len(corrupted)} record(s) failed integrity checks"
        else:
            status = KnowledgeHealthStatus.OK
            detail = "knowledge store healthy"
        return KnowledgeHealth(
            status=status,
            schema_version=version,
            expected_schema_version=self._schema_version,
            record_count=len(opt_rows),
            rollback_count=rollback_count,
            event_count=event_count,
            corrupted_records=corrupted,
            db_size_bytes=self._db_size(),
            detail=detail,
        )

    def stats(self) -> StorageStats:
        counts = {"records": 0, "rollbacks": 0, "events": 0}
        version = self._schema_version
        try:
            with self._connect() as connection:
                counts["records"] = int(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM optimization_records"
                    ).fetchone()["n"]
                )
                counts["rollbacks"] = int(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM rollback_records"
                    ).fetchone()["n"]
                )
                counts["events"] = int(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM knowledge_events"
                    ).fetchone()["n"]
                )
                row = connection.execute(
                    "SELECT value FROM knowledge_meta WHERE key = 'schema_version'"
                ).fetchone()
                if row is not None:
                    version = str(row["value"])
        except sqlite3.Error as exc:  # pragma: no cover - environment dependent
            raise KnowledgeUnavailableError(f"cannot read store stats: {exc}") from exc
        return StorageStats(
            db_path=str(self._db_path),
            db_size_bytes=self._db_size(),
            optimization_records=counts["records"],
            rollback_records=counts["rollbacks"],
            knowledge_events=counts["events"],
            schema_version=version,
        )

    def _db_size(self) -> int | None:
        try:
            return self._db_path.stat().st_size if self._db_path.is_file() else 0
        except OSError:  # pragma: no cover - defensive
            return None

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _parse_record(self, row: sqlite3.Row) -> OptimizationRecord:
        if _checksum(row["payload_json"]) != row["checksum"]:
            raise KnowledgeCorruptionError("record checksum mismatch (tampered/corrupt)")
        try:
            return OptimizationRecord.model_validate_json(row["payload_json"])
        except ValueError as exc:
            raise KnowledgeCorruptionError(f"record payload invalid: {exc}") from exc

    def _parse_rollback(self, row: sqlite3.Row) -> RollbackRecord:
        if _checksum(row["payload_json"]) != row["checksum"]:
            raise KnowledgeCorruptionError("rollback checksum mismatch (tampered/corrupt)")
        try:
            return RollbackRecord.model_validate_json(row["payload_json"])
        except ValueError as exc:
            raise KnowledgeCorruptionError(f"rollback payload invalid: {exc}") from exc

    def _write_result(
        self,
        status: KnowledgeWriteStatus,
        experiment_id: str,
        record_id: str | None,
        duplicate: bool,
        detail: str,
        started: float,
    ) -> KnowledgeWriteResult:
        return KnowledgeWriteResult(
            status=status,
            experiment_id=experiment_id,
            record_id=record_id,
            duplicate=duplicate,
            detail=detail,
            latency_s=time.perf_counter() - started,
        )
