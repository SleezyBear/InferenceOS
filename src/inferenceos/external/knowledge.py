"""Durable external recommendation/action knowledge (kept separate from Phase 8).

External records have different semantics from internal optimization records (process
identity, recommendation/approval/action/verification), so they live in their own
schema-versioned SQLite database. Historical records are contextual evidence only and
never bypass current process-identity/safety validation.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from inferenceos.contracts.enums import KnowledgeWriteStatus
from inferenceos.contracts.external import EXTERNAL_SCHEMA_VERSION, ExternalKnowledgeRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS external_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS external_records (
    record_id TEXT PRIMARY KEY,
    identity_key TEXT NOT NULL,
    executable_identity TEXT,
    resource TEXT NOT NULL,
    classification TEXT NOT NULL,
    action_id TEXT,
    outcome TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    checksum TEXT NOT NULL,
    schema_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ext_identity ON external_records(identity_key);
"""


class ExternalKnowledgeError(RuntimeError):
    """Raised when the external knowledge store is unavailable or corrupt."""


def _checksum(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:64]


class ExternalKnowledgeStore:
    """Durable, idempotent, integrity-checked external recommendation knowledge."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        schema_version: str = EXTERNAL_SCHEMA_VERSION,
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

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(str(self._db_path), timeout=15.0)
        except sqlite3.Error as exc:  # pragma: no cover - environment dependent
            raise ExternalKnowledgeError(f"cannot open external knowledge store: {exc}") from exc
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        try:
            connection = self._connect()
            try:
                connection.executescript(_SCHEMA)
                row = connection.execute(
                    "SELECT value FROM external_meta WHERE key = 'schema_version'"
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO external_meta(key, value) VALUES ('schema_version', ?)",
                        (self._schema_version,),
                    )
                elif str(row["value"]) != self._schema_version:
                    raise ExternalKnowledgeError(
                        f"external schema {row['value']!r} != expected {self._schema_version!r}"
                    )
                connection.commit()
            finally:
                connection.close()
        except ExternalKnowledgeError:
            raise
        except sqlite3.Error as exc:  # pragma: no cover - environment dependent
            raise ExternalKnowledgeError(
                f"cannot initialize external knowledge store: {exc}"
            ) from exc

    def record(self, record: ExternalKnowledgeRecord) -> KnowledgeWriteStatus:
        payload = record.model_dump_json()
        checksum = _checksum(payload)
        try:
            connection = self._connect()
            try:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO external_records
                    (record_id, identity_key, executable_identity, resource, classification,
                     action_id, outcome, created_at, payload_json, checksum, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.identity_key,
                        record.executable_identity,
                        record.resource.value,
                        record.classification.value,
                        record.action_id,
                        record.outcome,
                        record.created_at.isoformat(),
                        payload,
                        checksum,
                        self._schema_version,
                    ),
                )
                if cursor.rowcount == 1:
                    connection.commit()
                    return KnowledgeWriteStatus.PERSISTED
                existing = connection.execute(
                    "SELECT checksum FROM external_records WHERE record_id = ?",
                    (record.record_id,),
                ).fetchone()
                connection.commit()
            finally:
                connection.close()
        except sqlite3.Error:  # pragma: no cover - environment dependent
            return KnowledgeWriteStatus.FAILED
        if existing is not None and existing["checksum"] == checksum:
            return KnowledgeWriteStatus.DUPLICATE
        return KnowledgeWriteStatus.CONFLICT

    def find_for_identity(
        self, identity_key: str, *, limit: int = 20
    ) -> list[ExternalKnowledgeRecord]:
        try:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT payload_json, checksum FROM external_records "
                    "WHERE identity_key = ? ORDER BY created_at DESC LIMIT ?",
                    (identity_key, limit),
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:  # pragma: no cover - environment dependent
            raise ExternalKnowledgeError(f"cannot read external knowledge: {exc}") from exc
        return [self._parse(row) for row in rows]

    def all_records(self, *, limit: int = 50) -> list[ExternalKnowledgeRecord]:
        try:
            connection = self._connect()
            try:
                rows = connection.execute(
                    "SELECT payload_json, checksum FROM external_records "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.Error as exc:  # pragma: no cover - environment dependent
            raise ExternalKnowledgeError(f"cannot read external knowledge: {exc}") from exc
        return [self._parse(row) for row in rows]

    def count(self) -> int:
        try:
            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT COUNT(*) AS n FROM external_records"
                ).fetchone()
                return int(row["n"])
            finally:
                connection.close()
        except sqlite3.Error:  # pragma: no cover - environment dependent
            return 0

    def _parse(self, row: sqlite3.Row) -> ExternalKnowledgeRecord:
        if _checksum(str(row["payload_json"])) != str(row["checksum"]):
            raise ExternalKnowledgeError("external record checksum mismatch (tampered/corrupt)")
        return ExternalKnowledgeRecord.model_validate_json(str(row["payload_json"]))
