"""Immutable, transactional tune history, isolated under the configured data folder.

Lookup never creates a database. This store has no path back into acceptance or the
planner. C4-1 intentionally records observations without consuming them.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

from pydantic import ValidationError

from inferenceos.contracts.tune_learning import TuneHistoryRun

VERSION = "tune-history-1"


def _finite(value: object) -> None:
    """Reject invalid nested numbers before Pydantic's JSON mode can turn them null."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite tune-history number")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


class HistoryError(RuntimeError):
    pass


def history_paths(data_dir: Path) -> tuple[Path, ...]:
    db = data_dir / "tune_history.sqlite3"
    return (db, *(db.with_name(db.name + suffix) for suffix in ("-journal", "-wal", "-shm")))


class TuneHistoryStore:
    def __init__(self, data_dir: Path, *, timeout_s: float = 0.25) -> None:
        self.path = history_paths(data_dir)[0]
        self.timeout_s = timeout_s

    @staticmethod
    def _version(db: sqlite3.Connection) -> None:
        rows = db.execute("SELECT version FROM tune_history_meta").fetchall()
        if rows != [(VERSION,)]:
            raise HistoryError("unsupported or corrupt tune-history schema")

    @staticmethod
    def _decode(payload: str, digest: str, run_id: str) -> TuneHistoryRun:
        if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
            raise HistoryError("tune-history checksum mismatch")
        try:
            _finite(json.loads(payload))
            record = TuneHistoryRun.model_validate_json(payload)
        except (ValidationError, ValueError) as exc:
            raise HistoryError("invalid tune-history payload") from exc
        if record.run_id != run_id:
            raise HistoryError("tune-history row identity mismatch")
        return record

    def append(self, record: TuneHistoryRun) -> bool:
        # Revalidate even a model_copy(update=...) supplied by an internal caller.
        _finite(record.model_dump(mode="python"))
        raw = json.dumps(record.model_dump(mode="json"), sort_keys=True, allow_nan=False)
        record = TuneHistoryRun.model_validate_json(raw)
        payload = json.dumps(record.model_dump(mode="json"), sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with closing(sqlite3.connect(self.path, timeout=self.timeout_s)) as db:
                # DDL and insert share one transaction; concurrent initializers serialize.
                db.execute("BEGIN IMMEDIATE")
                try:
                    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if not tables:
                        db.execute("CREATE TABLE tune_history_meta (version TEXT NOT NULL)")
                        db.execute("INSERT INTO tune_history_meta VALUES (?)", (VERSION,))
                        db.execute("CREATE TABLE tune_history_runs (run_id TEXT PRIMARY KEY, payload TEXT NOT NULL, checksum TEXT NOT NULL)")
                    self._version(db)
                    row = db.execute("SELECT payload,checksum FROM tune_history_runs WHERE run_id=?", (record.run_id,)).fetchone()
                    if row is not None:
                        self._decode(row[0], row[1], record.run_id)
                        if row != (payload, digest):
                            raise HistoryError("conflicting tune-history run ID")
                        db.rollback()
                        return False
                    db.execute("INSERT INTO tune_history_runs VALUES (?,?,?)", (record.run_id, payload, digest))
                    db.commit()
                    return True
                except BaseException:
                    db.rollback()
                    raise
        except sqlite3.Error as exc:
            raise HistoryError(f"tune history unavailable: {exc}") from exc

    def get(self, run_id: str) -> TuneHistoryRun | None:
        if not self.path.exists():
            return None
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True,
                                         timeout=self.timeout_s)) as db:
                self._version(db)
                row = db.execute("SELECT payload,checksum FROM tune_history_runs WHERE run_id=?", (run_id,)).fetchone()
                return None if row is None else self._decode(row[0], row[1], run_id)
        except sqlite3.Error as exc:
            raise HistoryError(f"tune history unavailable: {exc}") from exc
