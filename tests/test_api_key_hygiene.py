"""H3 Windows temporary API-key hygiene tests."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

from inferenceos.backends.resident_server import (
    LlamaServerResidentBackend,
    sweep_stale_key_files,
)
from inferenceos.interfaces.backend_adapter import RuntimeParams
from tests.resident_helpers import synthetic_model


class _FailingProcess:
    """Process factory double whose start() fails before any server exists."""

    def __init__(self, command: list[str]) -> None:
        self.command = command

    def start(self) -> int:
        raise RuntimeError("injected server start failure")

    def stop(self, *, timeout_s: float = 5.0) -> tuple[int | None, str, bool]:
        return None, "none", True


def _key_files(root: Path) -> set[str]:
    return {path.name for path in root.glob("inferenceos-key-*.txt")}


def test_sweep_removes_only_stale_owned_keys(tmp_path: Path) -> None:
    old_key = tmp_path / "inferenceos-key-old.txt"
    fresh_key = tmp_path / "inferenceos-key-fresh.txt"
    unrelated_old = tmp_path / "other-secret.txt"
    unrelated_keylike = tmp_path / "mykey.txt"
    directory = tmp_path / "inferenceos-key-dir.txt"
    for path in (old_key, fresh_key, unrelated_old, unrelated_keylike):
        path.write_text("secret", encoding="utf-8")
    directory.mkdir()
    now = time.time()
    old = now - 10 * 3600
    os.utime(old_key, (old, old))
    os.utime(unrelated_old, (old, old))
    os.utime(unrelated_keylike, (old, old))

    removed = sweep_stale_key_files(temp_dir=tmp_path, max_age_s=3600.0, now=now)

    assert removed == 1
    assert not old_key.exists()
    assert fresh_key.exists()
    assert unrelated_old.exists()
    assert unrelated_keylike.exists()
    assert directory.is_dir()


def test_sweep_does_not_follow_symlinks(tmp_path: Path) -> None:
    external = tmp_path / "external-secret.txt"
    external.write_text("secret", encoding="utf-8")
    link = tmp_path / "inferenceos-key-link.txt"
    try:
        link.symlink_to(external)
    except OSError:
        return  # symlink creation not permitted; nothing to prove here
    old = time.time() - 10 * 3600
    os.utime(external, (old, old))
    sweep_stale_key_files(temp_dir=tmp_path, max_age_s=3600.0)
    assert external.exists()
    assert link.is_symlink()


def test_cleanup_key_file_removes_and_clears(tmp_path: Path) -> None:
    backend = LlamaServerResidentBackend(server=Path("unused"))
    key = tmp_path / "inferenceos-key-cleanup.txt"
    key.write_text("secret", encoding="utf-8")
    backend._key_file = key  # noqa: SLF001 - deliberate unit-level state
    backend._cleanup_key_file()  # noqa: SLF001
    assert not key.exists()
    assert backend._key_file is None


def test_start_failure_leaves_no_key_file(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"\x00")
    model = synthetic_model(str(model_path))
    temp_root = Path(tempfile.gettempdir())
    before = _key_files(temp_root)

    backend = LlamaServerResidentBackend(
        server=Path("unused"), process_factory=_FailingProcess  # type: ignore[arg-type]
    )
    try:
        backend.start(model, RuntimeParams(context_length=512))
    except RuntimeError:
        pass
    else:  # pragma: no cover - the injected failure must occur
        raise AssertionError("expected the injected start failure")

    after = _key_files(temp_root)
    assert backend._key_file is None
    # The failed start must not leave a new InferenceOS-owned key file behind.
    assert after <= before
