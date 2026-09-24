"""Robust, contextual external process identity.

PIDs are ephemeral; this module never treats PID alone as identity. The identity key
incorporates the process start time and a lightweight executable identity (path +
size + mtime, never a full content read) so a recycled PID cannot masquerade as the
original process. Every identity is explicitly marked potentially stale.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.contracts.enums import ExternalProcessAccess
from inferenceos.contracts.external import ExternalProcessIdentity

try:  # pragma: no cover - psutil is a hard dependency in practice
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def _executable_identity(path: str | None) -> str | None:
    if not path:
        return None
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    material = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def build_identity(
    *,
    pid: int,
    name: str,
    executable_path: str | None,
    owner: str | None,
    create_time_epoch: float | None,
    access: ExternalProcessAccess,
    is_target: bool = False,
    notes: list[str] | None = None,
) -> ExternalProcessIdentity:
    executable_identity = _executable_identity(executable_path)
    guard_material = (
        f"{pid}|{create_time_epoch if create_time_epoch is not None else 'unknown'}|"
        f"{executable_identity or executable_path or name}"
    )
    guard = hashlib.sha256(guard_material.encode("utf-8")).hexdigest()[:20]
    identity_key = f"{name}#{guard}"
    create_iso = (
        datetime.fromtimestamp(create_time_epoch, tz=timezone.utc).isoformat()
        if create_time_epoch is not None
        else None
    )
    stale = create_time_epoch is None or executable_identity is None
    combined_notes = list(notes or [])
    if stale:
        combined_notes.append(
            "identity may be stale: start time and/or executable identity unavailable"
        )
    return ExternalProcessIdentity(
        pid=pid,
        name=name,
        executable_path=executable_path,
        executable_identity=executable_identity,
        owner=owner,
        create_time_epoch=create_time_epoch,
        create_time_iso=create_iso,
        identity_key=identity_key,
        pid_reuse_guard=guard,
        access=access,
        is_target=is_target,
        potentially_stale=stale,
        notes=combined_notes,
    )


def identity_from_process(
    process: "psutil.Process", *, is_target: bool = False
) -> ExternalProcessIdentity:
    """Build an identity from an accessible psutil process (never reads cmdline/env)."""
    notes: list[str] = []
    access = ExternalProcessAccess.READABLE
    try:
        pid = int(process.pid)
        name = str(process.name())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return build_identity(
            pid=int(getattr(process, "pid", -1)),
            name="<unavailable>",
            executable_path=None,
            owner=None,
            create_time_epoch=None,
            access=ExternalProcessAccess.DENIED,
            is_target=is_target,
            notes=["process name/pid could not be read"],
        )
    try:
        create_time = float(process.create_time())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        create_time = None
        notes.append("create time unavailable")
    try:
        executable_path = process.exe()
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        executable_path = None
        notes.append("executable path not accessible")
    try:
        owner = process.username()
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        owner = None
        notes.append("owner not accessible")
    if executable_path is None or owner is None or create_time is None:
        access = ExternalProcessAccess.PARTIAL
    return build_identity(
        pid=pid,
        name=name,
        executable_path=executable_path,
        owner=owner,
        create_time_epoch=create_time,
        access=access,
        is_target=is_target,
        notes=notes,
    )


def resolve_identity(pid: int, *, is_target: bool = False) -> ExternalProcessIdentity | None:
    """Resolve the current identity for ``pid`` (``None`` if the process is gone)."""
    if psutil is None:  # pragma: no cover - hard dependency
        return None
    try:
        process = psutil.Process(pid)
        if not process.is_running():
            return None
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return None
    return identity_from_process(process, is_target=is_target)


def verify_identity(identity: ExternalProcessIdentity) -> bool:
    """Whether ``pid`` still refers to the *same* process (PID-reuse guarded)."""
    if psutil is None:  # pragma: no cover - hard dependency
        return False
    try:
        process = psutil.Process(identity.pid)
        if not process.is_running():
            return False
        current = identity_from_process(process)
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False
    if identity.create_time_epoch is not None and current.create_time_epoch is not None:
        if abs(identity.create_time_epoch - current.create_time_epoch) > 0.01:
            return False
    if identity.executable_identity is not None and current.executable_identity is not None:
        if identity.executable_identity != current.executable_identity:
            return False
    # At minimum the PID-reuse guard (start time included when available) must match.
    return current.pid_reuse_guard == identity.pid_reuse_guard
