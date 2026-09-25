"""H2 storage-retention and disk-protection contracts.

These records make log/report retention observable and auditable. They are additive
and do not replace any existing contract.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase

STORAGE_SCHEMA_VERSION = "h2.1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactClass:
    """How a generated artifact is treated by retention.

    Only ``OPERATIONAL_LOG`` and ``GENERATED_REPORT`` are ever automatically removed.
    ``DURABLE_STATE`` (SQLite databases, deployment/rollback state, configuration) and
    ``TEMPORARY`` in-flight files are never treated as disposable evidence.
    """

    OPERATIONAL_LOG = "operational_log"
    GENERATED_REPORT = "generated_report"
    DURABLE_STATE = "durable_state"
    TEMPORARY = "temporary"


class DiskSpaceStatus:
    """Conservative, explicit free-space classification."""

    SUFFICIENT = "sufficient"
    LOW = "low"
    CRITICAL = "critical"


class RetentionSweep(ContractBase):
    """One bounded retention pass over InferenceOS-owned disposable artifacts."""

    schema_version: str = STORAGE_SCHEMA_VERSION
    sweep_id: str
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    enabled: bool = True

    triggers: list[str] = Field(default_factory=list)
    report_file_limit: int = Field(default=0, ge=0)
    report_byte_limit: int = Field(default=0, ge=0)
    min_free_bytes: int = Field(default=0, ge=0)
    max_deletions_per_pass: int = Field(default=0, ge=0)

    scanned_files: int = Field(default=0, ge=0)
    scanned_bytes: int = Field(default=0, ge=0)
    disposable_files: int = Field(default=0, ge=0)
    disposable_bytes: int = Field(default=0, ge=0)
    protected_files: int = Field(default=0, ge=0)

    deleted_files: int = Field(default=0, ge=0)
    deleted_bytes: int = Field(default=0, ge=0)
    deleted_temp_files: int = Field(default=0, ge=0)
    deleted_names: list[str] = Field(default_factory=list)
    skipped_files: int = Field(default=0, ge=0)

    free_bytes_before: int | None = None
    free_bytes_after: int | None = None
    disk_status_before: str = DiskSpaceStatus.SUFFICIENT
    disk_status_after: str = DiskSpaceStatus.SUFFICIENT
    bounded: bool = True
    errors: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "RetentionSweep":
        return cls.model_validate_json(data)


class PersistenceResult(ContractBase):
    """Outcome of persisting one disposable report.

    ``persisted=False, degraded=True`` is the explicit, non-fatal disk-pressure result:
    the report was not written, and the caller must not claim otherwise.
    """

    path: str
    persisted: bool = False
    degraded: bool = False
    atomic: bool = False
    reason: str = ""
    disk_status: str = DiskSpaceStatus.SUFFICIENT
    free_bytes: int | None = None
    sweep_id: str | None = None

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)
