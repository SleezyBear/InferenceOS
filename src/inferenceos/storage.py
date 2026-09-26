"""H2 log/report retention and disk-space protection for unattended operation.

This module owns a small, explicit retention policy for the *disposable* artifacts
InferenceOS generates (acceptance/verification reports, per-run benchmark artifacts,
per-phase operational logs). It deliberately does **not** manage durable state.

Artifact classes (see :class:`inferenceos.contracts.storage.ArtifactClass`):

* operational logs     -> rotated by the standard logging handler (bounded size/count);
* generated reports    -> allowlisted, timestamped evidence, removed oldest-first;
* durable state        -> SQLite databases, deployment/rollback/configuration: never
  automatically deleted;
* temporary files      -> in-flight atomic-write temporaries: cleaned only when stale.

Retention is intentionally allowlist-driven: a file is disposable only when its name
matches a known generated-artifact pattern *and* it is a direct, non-symlink child of
the configured reports directory. Anything else is left untouched.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from inferenceos.config import Settings
from inferenceos.contracts.storage import (
    DiskSpaceStatus,
    PersistenceResult,
    RetentionSweep,
)
from inferenceos.logging_config import get_logger

logger = get_logger("storage")

# Reused only when Settings cannot be read for some reason.
_DEFAULT_REPORT_MAX_FILES = 1000
_DEFAULT_REPORT_MAX_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_DELETIONS = 500
_DEFAULT_MIN_FREE_BYTES = 1024 * 1024 * 1024
_DEFAULT_LOG_MAX_BYTES = 8 * 1024 * 1024
_DEFAULT_LOG_BACKUP_COUNT = 3

# A UTC file-stamp as used by every generated InferenceOS report: YYYYMMDDTHHMMSSZ.
_STAMP = r"\d{8}T\d{6}Z"

# Allowlist of disposable generated-report filename patterns. Broad directory deletion
# is never performed; only these exact families are candidates.
_DISPOSABLE_REPORT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        rf"accounting_verification_{_STAMP}\.json",
        rf"hardening_verification_{_STAMP}\.json",
        rf"baseline_{_STAMP}\.json",
        rf"candidates_{_STAMP}\.json",
        rf"deep_diagnosis_{_STAMP}\.json",
        rf"diagnosis_{_STAMP}\.json",
        rf"experiment_{_STAMP}\.json",
        rf"experiment_audit_{_STAMP}\.json",
        rf"hardware_profile_{_STAMP}\.json",
        rf"phase\d+r?_[a-z0-9_]+_{_STAMP}\.json",
        rf"phase\d+r-(?:noop|opt)-{_STAMP}_baseline\.json",
        rf"remediation_verification_{_STAMP}\.json",
        rf"resident_recovery_verification_{_STAMP}\.json",
        rf"storage_verification_{_STAMP}\.json",
        rf"validity_verification_{_STAMP}\.json",
        rf"serve_{_STAMP}\.json",
    )
)
# Atomic-write temporaries created by this module (leading dot + .tmp suffix).
_TEMP_PATTERN = re.compile(r"^\..+\.tmp$")
_STALE_TEMP_AGE_S = 3600.0

DiskUsageFn = Callable[[Path], tuple[int, int, int]]


@dataclass(frozen=True)
class StoragePolicy:
    """Explicit, deterministic retention policy derived from :class:`Settings`."""

    enabled: bool = True
    report_max_files: int = _DEFAULT_REPORT_MAX_FILES
    report_max_bytes: int = _DEFAULT_REPORT_MAX_BYTES
    min_free_bytes: int = _DEFAULT_MIN_FREE_BYTES
    max_deletions_per_pass: int = _DEFAULT_MAX_DELETIONS
    tmp_max_age_s: float = _STALE_TEMP_AGE_S

    @property
    def critical_free_bytes(self) -> int:
        """Below this, disposable report writes fail closed (not persisted)."""
        return max(self.min_free_bytes // 2, 0)

    @classmethod
    def from_settings(cls, settings: object) -> "StoragePolicy":
        return cls(
            enabled=bool(getattr(settings, "retention_enabled", True)),
            report_max_files=max(
                int(getattr(settings, "retention_report_max_files", _DEFAULT_REPORT_MAX_FILES)),
                0,
            ),
            report_max_bytes=max(
                int(getattr(settings, "retention_report_max_bytes", _DEFAULT_REPORT_MAX_BYTES)),
                0,
            ),
            min_free_bytes=max(
                int(getattr(settings, "retention_min_free_bytes", _DEFAULT_MIN_FREE_BYTES)),
                0,
            ),
            max_deletions_per_pass=max(
                int(
                    getattr(
                        settings,
                        "retention_max_deletions_per_pass",
                        _DEFAULT_MAX_DELETIONS,
                    )
                ),
                0,
            ),
        )


# --------------------------------------------------------------------------- #
# Classification / path safety                                                #
# --------------------------------------------------------------------------- #
def is_disposable_report_name(name: str) -> bool:
    """True when ``name`` is a known generated-report filename (not a path)."""
    return any(pattern.fullmatch(name) for pattern in _DISPOSABLE_REPORT_PATTERNS)


def is_temporary_name(name: str) -> bool:
    return _TEMP_PATTERN.fullmatch(name) is not None


def owned_reports_root(settings: object) -> Path:
    try:
        return Path(getattr(settings, "reports_dir")).resolve()
    except OSError:
        return Path(str(getattr(settings, "reports_dir"))).resolve()


def is_safe_disposable(path: Path, root: Path) -> bool:
    """Only a direct, non-symlink, regular child of ``root`` may be deleted.

    Symlinks and paths whose resolved parent escapes ``root`` are refused so a crafted
    name cannot cause deletion outside the owned artifact directory.
    """
    try:
        if path.is_symlink():
            return False
        if not path.is_file():
            return False
        resolved = path.resolve()
    except OSError:
        return False
    if resolved.parent != root:
        return False
    return is_disposable_report_name(path.name)


# --------------------------------------------------------------------------- #
# Disk space                                                                  #
# --------------------------------------------------------------------------- #
def _free_bytes(root: Path, disk_usage: DiskUsageFn) -> int | None:
    try:
        _total, _used, free = disk_usage(root)
        return int(free)
    except OSError:
        return None


def classify_free_space(free_bytes: int | None, policy: StoragePolicy) -> str:
    if free_bytes is None:
        # Unavailable telemetry never blocks a write; it is simply not protective.
        return DiskSpaceStatus.SUFFICIENT
    if free_bytes < policy.critical_free_bytes:
        return DiskSpaceStatus.CRITICAL
    if free_bytes < policy.min_free_bytes:
        return DiskSpaceStatus.LOW
    return DiskSpaceStatus.SUFFICIENT


def disk_status(
    settings: object,
    *,
    policy: StoragePolicy | None = None,
    disk_usage: DiskUsageFn = shutil.disk_usage,
) -> tuple[str, int | None]:
    policy = policy or StoragePolicy.from_settings(settings)
    free = _free_bytes(owned_reports_root(settings), disk_usage)
    return classify_free_space(free, policy), free


# --------------------------------------------------------------------------- #
# Retention                                                                   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Disposable:
    path: Path
    size: int
    mtime: float


def _scan_disposable(root: Path) -> tuple[list[_Disposable], int, int, int, int]:
    """Return (disposable, scanned_files, scanned_bytes, protected, skipped)."""
    disposable: list[_Disposable] = []
    scanned_files = 0
    scanned_bytes = 0
    protected = 0
    skipped = 0
    with os.scandir(root) as entries:
        for entry in entries:
            try:
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    continue
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                skipped += 1
                continue
            scanned_files += 1
            scanned_bytes += int(stat.st_size)
            path = Path(entry.path)
            if is_safe_disposable(path, root):
                disposable.append(
                    _Disposable(path=path, size=int(stat.st_size), mtime=float(stat.st_mtime))
                )
            else:
                protected += 1
    return disposable, scanned_files, scanned_bytes, protected, skipped


def _clean_stale_temporaries(
    root: Path, *, now: float, max_age_s: float
) -> tuple[int, list[str]]:
    deleted = 0
    errors: list[str] = []
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not is_temporary_name(entry.name):
                    continue
                try:
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        continue
                    stat = entry.stat(follow_symlinks=False)
                    if (now - float(stat.st_mtime)) < max_age_s:
                        continue
                    os.unlink(entry.path)
                    deleted += 1
                except OSError as exc:
                    errors.append(f"{entry.name}: {type(exc).__name__}: {exc}")
    except OSError as exc:
        errors.append(f"temp scan: {type(exc).__name__}: {exc}")
    return deleted, errors


def enforce_retention(
    settings: object,
    *,
    policy: StoragePolicy | None = None,
    disk_usage: DiskUsageFn = shutil.disk_usage,
    now: datetime | None = None,
) -> RetentionSweep:
    """Run one bounded retention pass over the owned reports directory.

    Cleanup occurs when **any** limit is exceeded (file count OR byte budget OR free
    disk below the floor). Oldest disposable artifacts are removed first, until every
    applicable limit is satisfied or the per-pass deletion bound is reached. A malformed
    or unrecognized file is never deleted. Individual deletion failures are recorded and
    never abort the sweep.
    """
    policy = policy or StoragePolicy.from_settings(settings)
    now_dt = now or datetime.now(timezone.utc)
    sweep_id = f"rsweep-{uuid.uuid4().hex[:10]}"
    errors: list[str] = []
    notes: list[str] = []

    if not policy.enabled:
        return RetentionSweep(
            sweep_id=sweep_id,
            started_at=now_dt,
            finished_at=now_dt,
            enabled=False,
            notes=["retention disabled"],
        )

    root = owned_reports_root(settings)
    if not root.is_dir():
        return RetentionSweep(
            sweep_id=sweep_id,
            started_at=now_dt,
            finished_at=now_dt,
            notes=["reports directory does not exist"],
        )

    free_before = _free_bytes(root, disk_usage)
    status_before = classify_free_space(free_before, policy)

    try:
        disposable, scanned_files, scanned_bytes, protected, skipped = _scan_disposable(root)
    except OSError as exc:
        return RetentionSweep(
            sweep_id=sweep_id,
            started_at=now_dt,
            finished_at=now_dt,
            free_bytes_before=free_before,
            disk_status_before=status_before,
            errors=[f"scan failed: {type(exc).__name__}: {exc}"],
        )

    disposable_bytes = sum(item.size for item in disposable)
    triggers: list[str] = []
    if len(disposable) > policy.report_max_files:
        triggers.append("count")
    if disposable_bytes > policy.report_max_bytes:
        triggers.append("bytes")
    if free_before is not None and free_before < policy.min_free_bytes:
        triggers.append("disk_free")

    deleted_names: list[str] = []
    deleted_bytes = 0
    remaining = len(disposable)
    remaining_bytes = disposable_bytes
    current_free = free_before

    if triggers:
        for item in sorted(disposable, key=lambda entry: (entry.mtime, entry.path.name)):
            if len(deleted_names) >= policy.max_deletions_per_pass:
                break
            within_count = remaining <= policy.report_max_files
            within_bytes = remaining_bytes <= policy.report_max_bytes
            within_disk = current_free is None or current_free >= policy.min_free_bytes
            if within_count and within_bytes and within_disk:
                break
            try:
                item.path.unlink()
            except OSError as exc:
                errors.append(f"{item.path.name}: {type(exc).__name__}: {exc}")
                continue
            deleted_names.append(item.path.name)
            deleted_bytes += item.size
            remaining -= 1
            remaining_bytes -= item.size
            if current_free is not None:
                current_free += item.size

    deleted_temp, temp_errors = _clean_stale_temporaries(
        root, now=now_dt.timestamp(), max_age_s=policy.tmp_max_age_s
    )
    errors.extend(temp_errors)

    free_after = _free_bytes(root, disk_usage)
    status_after = classify_free_space(free_after, policy)
    bounded = len(deleted_names) >= policy.max_deletions_per_pass
    if bounded:
        notes.append("per-pass deletion bound reached; further limits may remain")
    if skipped:
        notes.append(f"{skipped} file(s) skipped (unreadable metadata)")

    return RetentionSweep(
        sweep_id=sweep_id,
        started_at=now_dt,
        finished_at=datetime.now(timezone.utc),
        enabled=True,
        triggers=triggers,
        report_file_limit=policy.report_max_files,
        report_byte_limit=policy.report_max_bytes,
        min_free_bytes=policy.min_free_bytes,
        max_deletions_per_pass=policy.max_deletions_per_pass,
        scanned_files=scanned_files,
        scanned_bytes=scanned_bytes,
        disposable_files=len(disposable),
        disposable_bytes=disposable_bytes,
        protected_files=protected,
        deleted_files=len(deleted_names),
        deleted_bytes=deleted_bytes,
        deleted_temp_files=deleted_temp,
        deleted_names=deleted_names[:50],
        skipped_files=skipped,
        free_bytes_before=free_before,
        free_bytes_after=free_after,
        disk_status_before=status_before,
        disk_status_after=status_after,
        bounded=bounded,
        errors=errors,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Safe report writing                                                         #
# --------------------------------------------------------------------------- #
def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` atomically (temp file + fsync + atomic rename).

    A crash leaves either the previous state or nothing at the final path; a partially
    written temporary can never be mistaken for a completed report.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "w", encoding=encoding) as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def guarded_write_text(
    path: Path,
    text: str,
    settings: object,
    *,
    policy: StoragePolicy | None = None,
    disk_usage: DiskUsageFn = shutil.disk_usage,
    encoding: str = "utf-8",
) -> PersistenceResult:
    """Persist a disposable report, protecting against disk exhaustion.

    On low space the retention pass is attempted and the space rechecked. If space is
    still critically low the write is refused and returned as ``degraded``; the caller
    must not claim persistence succeeded. The write itself is atomic.
    """
    policy = policy or StoragePolicy.from_settings(settings)
    root = owned_reports_root(settings)
    target = Path(path)
    try:
        resolved = target.resolve()
    except OSError as exc:
        return PersistenceResult(
            path=str(target), degraded=True, reason=f"path could not be resolved: {exc}"
        )
    if resolved.parent != root:
        return PersistenceResult(
            path=str(resolved),
            degraded=True,
            reason="refusing to write outside the owned reports directory",
        )

    status, free = disk_status(settings, policy=policy, disk_usage=disk_usage)
    sweep_id: str | None = None
    if status is not DiskSpaceStatus.SUFFICIENT:
        sweep = enforce_retention(settings, policy=policy, disk_usage=disk_usage)
        sweep_id = sweep.sweep_id
        status, free = disk_status(settings, policy=policy, disk_usage=disk_usage)

    if status is DiskSpaceStatus.CRITICAL:
        return PersistenceResult(
            path=str(resolved),
            persisted=False,
            degraded=True,
            reason="disk space critically low; disposable report not persisted",
            disk_status=status,
            free_bytes=free,
            sweep_id=sweep_id,
        )
    try:
        atomic_write_text(resolved, text, encoding=encoding)
    except OSError as exc:
        return PersistenceResult(
            path=str(resolved),
            persisted=False,
            degraded=True,
            reason=f"write failed: {type(exc).__name__}: {exc}",
            disk_status=status,
            free_bytes=free,
            sweep_id=sweep_id,
        )
    return PersistenceResult(
        path=str(resolved),
        persisted=True,
        atomic=True,
        disk_status=status,
        free_bytes=free,
        sweep_id=sweep_id,
    )


# --------------------------------------------------------------------------- #
# Startup                                                                     #
# --------------------------------------------------------------------------- #
def run_startup_retention(settings: object | None = None) -> RetentionSweep | None:
    """Best-effort retention at process startup. Never raises, never fatal."""
    try:
        resolved = settings if settings is not None else Settings()
        ensure = getattr(resolved, "ensure_directories", None)
        if callable(ensure):
            ensure()
        sweep = enforce_retention(resolved)
        if sweep.deleted_files or sweep.deleted_temp_files:
            logger.info(
                "startup retention removed %d artifact(s) and %d stale temporary(s), "
                "freeing %d bytes",
                sweep.deleted_files,
                sweep.deleted_temp_files,
                sweep.deleted_bytes,
            )
        if sweep.errors:
            logger.warning(
                "startup retention completed with %d non-fatal error(s)", len(sweep.errors)
            )
        return sweep
    except Exception as exc:  # noqa: BLE001 - retention must never prevent startup
        logger.warning("startup retention skipped (non-fatal): %s", exc)
        return None
