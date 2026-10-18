"""Read resource-release evidence without acquiring or cleaning up any resource.

The resident backend remains the cleanup owner. Process exit and credential removal
are separate facts; an idle Python handle is not proof that an unconfirmed PID exited.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MISSING = object()


def owner_released(owner: object, measurement: object | None = None) -> bool:
    """Read a wrapper's owner evidence when available, with explicit stop vetoes."""
    try:
        reader = getattr(owner, "release_evidence", None)
        clean = bool(reader().clean) if callable(reader) else release_evidence(owner, stopped=measurement).clean
        return clean and getattr(measurement, "cleanup_complete", None) is not False and getattr(
            measurement, "credential_removed", None) is not False
    except Exception:
        return False


@dataclass(frozen=True)
class ReleaseEvidence:
    process_gone: bool
    credential_removed: bool | None
    credential_files: tuple[str, ...]
    problems: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return self.process_gone and self.credential_removed is not False and not self.problems

    @property
    def reason(self) -> str | None:
        return "; ".join(self.problems) or None


def release_evidence(backend: object, *, stopped: object | None = None) -> ReleaseEvidence:
    """Read the owner's current record, with stop evidence when the caller has it.

    Legacy doubles without retained-PID/credential fields keep their existing
    no-longer-running fallback. Missing optional credential evidence remains unknown;
    explicit failed cleanup and unreadable owner state never become success.
    """
    problems: list[str] = []

    def read(name: str) -> Any:
        try:
            value = getattr(backend, name, _MISSING)
            return value() if callable(value) else value
        except Exception as exc:
            problems.append(f"cleanup evidence {name} unavailable: {type(exc).__name__}")
            return _MISSING

    active = read("is_active")
    if type(active) is not bool:
        active = read("is_running")
    if type(active) is not bool:
        problems.append("the model's active state was not confirmed")
    pending = read("unconfirmed_pid")
    known_pid_owner = pending is not _MISSING
    if known_pid_owner and pending is not None:
        problems.append(f"the model's process {pending} was not confirmed gone")
    measurement = stopped
    if measurement is None and not known_pid_owner:
        measurement = read("last_unload")
    process_gone = active is False and (pending is _MISSING or pending is None)
    if getattr(measurement, "cleanup_complete", None) is False:
        process_gone = False
    if not process_gone and not any("process" in problem for problem in problems):
        problems.append("the model's process was not confirmed gone")

    current_key, pending_keys = read("api_key_file"), read("undeleted_key_files")
    known_keys = current_key is not _MISSING or pending_keys is not _MISSING
    files: list[str] = []
    if current_key is not _MISSING and current_key is not None:
        files.append(str(current_key))
    if pending_keys is not _MISSING and pending_keys is not None:
        if isinstance(pending_keys, (list, tuple)):
            files.extend(str(path) for path in pending_keys)
        else:
            problems.append("the password-file cleanup record was not readable")
    files = list(dict.fromkeys(files))
    credential: bool | None = not files if known_keys else getattr(measurement, "credential_removed", None)
    if stopped is not None and getattr(stopped, "credential_removed", None) is False:
        credential = False
    if credential is False:
        problems.append("the password file could not be deleted"
                        + (f" ({', '.join(files)}); delete it by hand" if files else ""))
    return ReleaseEvidence(process_gone, credential, tuple(files), tuple(problems))
