"""Saved setups: one approved setup per model, context size and power source (step 5).

Saved setups live in one file in the data folder. A setup is found by the model file's
path; before it is trusted the file's size and modification time are compared, and if
they differ the file's contents are checked, so a replaced model is never started with
another model's setup.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.tune import ModelFileIdentity, SavedSetup
from inferenceos.logging_config import get_logger

logger = get_logger("tune.profiles")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(path: Path, *, with_hash: bool = True) -> ModelFileIdentity:
    resolved = path.resolve()
    stat = resolved.stat()
    return ModelFileIdentity(
        path=str(resolved),
        size_bytes=int(stat.st_size),
        modified_ns=int(stat.st_mtime_ns),
        sha256=file_sha256(resolved) if with_hash else None,
    )


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

    def _write(self, setups: list[SavedSetup]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"setups": [setup.model_dump(mode="json") for setup in setups]}
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self._path)

    @staticmethod
    def _same_slot(a: SavedSetup, b: SavedSetup) -> bool:
        return (
            a.model.path == b.model.path
            and a.context_length == b.context_length
            and power_key(a.power_source) is power_key(b.power_source)
            and a.program == b.program
        )

    def save(self, setup: SavedSetup) -> None:
        """Save ``setup``, replacing the old one for the same model, context and power."""
        kept = [item for item in self.all() if not self._same_slot(item, setup)]
        kept.append(setup)
        self._write(kept)

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
    "SetupStore",
    "file_sha256",
    "model_identity",
    "power_key",
    "power_words",
    "same_model_file",
]
