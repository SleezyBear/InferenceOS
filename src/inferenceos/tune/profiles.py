"""Saved setups: one approved setup per model, context size and power source (step 5).

Saved setups live in one file in the data folder. A setup is found by the model file's
path; before it is trusted the file's size and modification time are compared, and if
they differ the file's contents are checked, so a replaced model is never started with
another model's setup.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferenceos.config import Settings
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.tune import ModelFileIdentity, SavedSetup
from inferenceos.logging_config import get_logger

logger = get_logger("tune.profiles")


def file_sha256(path: Path, *, check: Callable[[], None] | None = None) -> str:
    """SHA-256 of a file, read in 4 MiB chunks. ``check`` (if given) is called before
    every chunk and may raise to abandon the hash (a user stop during saving)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            if check is not None:
                check()
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(
    path: Path, *, with_hash: bool = True, check: Callable[[], None] | None = None
) -> ModelFileIdentity:
    """Path, size, modification time and (optionally) content hash of the model file.
    Not cached: the hash binds the saved setup to the file as it is when saving."""
    resolved = path.resolve()
    stat = resolved.stat()
    return ModelFileIdentity(
        path=str(resolved),
        size_bytes=int(stat.st_size),
        modified_ns=int(stat.st_mtime_ns),
        sha256=file_sha256(resolved, check=check) if with_hash else None,
    )


class StoreBusy(RuntimeError):
    """Another process held the saved-setups lock for too long; nothing was saved."""


#: errno values that mean "the lock is held by someone else" (anything else is a real
#: error and is raised at once). POSIX ``flock(LOCK_NB)``: EWOULDBLOCK / EAGAIN.
#: Windows ``msvcrt.locking(LK_NBLCK)``: EACCES (and EDEADLOCK, which the C runtime
#: uses for a lock it could not take).
_CONTENTION_ERRNOS = frozenset(
    {errno.EAGAIN, errno.EWOULDBLOCK}
    if sys.platform != "win32"
    else {errno.EACCES, getattr(errno, "EDEADLOCK", errno.EDEADLK), errno.EDEADLK}
)


def _lock_contended(exc: OSError) -> bool:
    return isinstance(exc, BlockingIOError) or exc.errno in _CONTENTION_ERRNOS


class _StoreLock:
    """An exclusive, process-level lock on ``<file>.lock`` held from reading the saved
    setups to the atomic replace, so two writers (two tune processes, a tune and a
    manual command) can never lose each other's update.

    Waits at most ``timeout_s`` while the lock is held elsewhere, calling ``check``
    (if given) between attempts -- a user stop raises from it and aborts the wait at
    once. Only "lock is held" errors are retried; any other error (permissions, a bad
    handle, a filesystem error) is raised immediately."""

    def __init__(self, path: Path, timeout_s: float = 60.0, *,
                 check: Callable[[], None] | None = None, poll_s: float = 0.05) -> None:
        self._path = path
        self._timeout = timeout_s
        self._check = check
        self._poll = poll_s
        self._handle: Any = None

    @staticmethod
    def _try_lock(handle: Any) -> None:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __enter__(self) -> _StoreLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._path, "a+b")  # noqa: SIM115 - closed in __exit__ / on failure
        deadline = time.monotonic() + self._timeout
        try:
            while True:
                if self._check is not None:
                    self._check()  # a user stop aborts here, before anything is written
                try:
                    self._try_lock(handle)
                    break
                except OSError as exc:
                    if not _lock_contended(exc):
                        raise
                    if time.monotonic() >= deadline:
                        raise StoreBusy(
                            f"the saved setups are being written by another process ({self._path})"
                        ) from exc
                    time.sleep(self._poll)
        except BaseException:
            handle.close()
            raise
        self._handle = handle
        return self

    def __exit__(self, *exc: object) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def same_model_file(saved: ModelFileIdentity, path: Path) -> tuple[bool, str]:
    """Whether ``path`` is still the model file the setup was saved for."""
    resolved = path.resolve()
    if not resolved.is_file():
        return False, f"model file not found: {resolved}"
    stat = resolved.stat()
    if int(stat.st_size) == saved.size_bytes and int(stat.st_mtime_ns) == saved.modified_ns:
        return True, "model file unchanged (same size and date)"
    if saved.sha256 is None:
        return False, "the model file changed and the saved setup has no content check"
    if int(stat.st_size) != saved.size_bytes:
        return False, "the model file is a different size from the one that was tuned"
    if file_sha256(resolved) == saved.sha256:
        return True, "model file date changed but its contents are the same"
    return False, "the model file's contents changed since it was tuned"


def power_key(power: PowerState) -> PowerState:
    """Setups are kept for plugged in and for battery; unknown counts as plugged in."""
    return PowerState.BATTERY if power is PowerState.BATTERY else PowerState.AC


def power_words(power: PowerState) -> str:
    return "on battery" if power_key(power) is PowerState.BATTERY else "plugged in"


class CommitRefused(Exception):
    """The commit check refused publication; the previous saved setups are unchanged."""


@dataclass
class CommitGuard:
    """Where publication really happens (review 10).

    Saving is prepared first (reading the old file, serializing, writing a temporary file
    next to it); the *commit* is the one atomic replace of the saved-setups file. The
    guard's ``check`` runs under ``lock`` immediately before that replace, and the same
    lock is what a cancellation must take to register -- so a cancellation either comes
    before the check (nothing is published, the previous file is untouched) or after the
    replace (the new setup stays, and ``committed`` has already recorded that). No
    rollback either way.
    """

    check: Callable[[], None]
    lock: threading.Lock = field(default_factory=threading.Lock)
    committed: Callable[[], None] = lambda: None
    #: Lifecycle events, in order: ``hashing`` (the model file's content hash, emitted
    #: by ``Tuner._save``; the hash calls ``check`` between chunks), ``preparing``,
    #: ``writing`` (temporary file open and written, not yet closed), ``written``,
    #: ``committing`` (just before the guarded replace), ``committed``. Used by tests
    #: and the simulator to place a stop exactly.
    event: Callable[[str], None] | None = None

    def emit(self, name: str) -> None:
        if self.event is not None:
            self.event(name)


class SetupStore:
    """The saved setups file (``data/saved_setups.json``)."""

    def __init__(self, settings: Settings, *, path: Path | None = None) -> None:
        self._path = path or settings.data_dir / "saved_setups.json"

    @property
    def path(self) -> Path:
        return self._path

    def all(self) -> list[SavedSetup]:
        if not self._path.is_file():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("saved setups file unreadable (%s): %s", self._path, exc)
            return []
        setups: list[SavedSetup] = []
        for item in raw.get("setups", []) if isinstance(raw, dict) else []:
            try:
                setups.append(SavedSetup.model_validate(item))
            except ValueError as exc:
                logger.warning("skipping an unreadable saved setup: %s", exc)
        return setups

    def _write(self, setups: list[SavedSetup], guard: CommitGuard | None = None) -> None:
        """Prepare a temporary file, then commit it with one atomic replace. Any failure
        or refusal before the replace leaves the previous file exactly as it was."""
        emit = guard.emit if guard is not None else (lambda _name: None)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"setups": [setup.model_dump(mode="json") for setup in setups]}, indent=2)
        handle, name = tempfile.mkstemp(prefix=self._path.name + ".", suffix=".tmp", dir=self._path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(text)
                emit("writing")
                out.flush()
                os.fsync(out.fileno())
            emit("written")
            emit("committing")
            if guard is None:
                os.replace(temporary, self._path)
            else:
                with guard.lock:
                    guard.check()  # raises to refuse: nothing is published
                    os.replace(temporary, self._path)
                    guard.committed()
        except BaseException:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best effort
                pass
            raise
        emit("committed")

    @staticmethod
    def _same_slot(a: SavedSetup, b: SavedSetup) -> bool:
        return (
            a.model.path == b.model.path
            and a.context_length == b.context_length
            and power_key(a.power_source) is power_key(b.power_source)
            and a.program == b.program
        )

    def save(self, setup: SavedSetup, *, guard: CommitGuard | None = None) -> None:
        """Save ``setup``, replacing the old one for the same model, context and power.

        With a ``guard`` the publication point is the guarded atomic replace (see
        :class:`CommitGuard`)."""
        if guard is not None:
            guard.emit("preparing")
        # Single writer: read-modify-replace under the process-level store lock.
        lock = _StoreLock(self._path.with_name(self._path.name + ".lock"),
                          check=guard.check if guard is not None else None)
        with lock:
            kept = [item for item in self.all() if not self._same_slot(item, setup)]
            kept.append(setup)
            self._write(kept, guard)

    def for_model(self, model_path: Path, *, program: str = "llama.cpp") -> list[SavedSetup]:
        resolved = str(model_path.resolve())
        found = [
            item
            for item in self.all()
            if item.model.path == resolved and item.program == program
        ]
        found.sort(key=lambda item: item.created_at, reverse=True)
        return found

    def find(
        self,
        model_path: Path,
        *,
        context_length: int | None,
        power: PowerState,
        program: str = "llama.cpp",
    ) -> SavedSetup | None:
        wanted = power_key(power)
        for item in self.for_model(model_path, program=program):
            if context_length is not None and item.context_length != context_length:
                continue
            if power_key(item.power_source) is wanted:
                return item
        return None


__all__ = [
    "CommitGuard",
    "CommitRefused",
    "SetupStore",
    "StoreBusy",
    "file_sha256",
    "model_identity",
    "power_key",
    "power_words",
    "same_model_file",
]
