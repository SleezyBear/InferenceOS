"""H2 storage retention, path safety, disk protection and atomic-write tests."""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import pytest

from inferenceos import storage
from inferenceos.config import Settings
from inferenceos.contracts.storage import DiskSpaceStatus
from inferenceos.logging_config import setup_logging
from inferenceos.storage import (
    StoragePolicy,
    atomic_write_text,
    enforce_retention,
    guarded_write_text,
    is_disposable_report_name,
    is_safe_disposable,
    run_startup_retention,
)

_STAMP = "20260101T000000Z"


def _settings(root: Path) -> Settings:
    return Settings(reports_dir=root, logs_dir=root)


def _write(path: Path, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * size, encoding="utf-8")


def _age(path: Path, seconds: float) -> None:
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


class _ScriptedDisk:
    def __init__(self, frees: list[int]) -> None:
        self._frees = frees
        self.calls = 0

    def __call__(self, path: Path) -> tuple[int, int, int]:
        free = self._frees[min(self.calls, len(self._frees) - 1)]
        self.calls += 1
        total = 10 * 1024**3
        return total, total - free, free


def test_count_retention_removes_oldest_disposable_and_preserves_others(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    for day, age in ((1, 5000.0), (2, 4000.0), (3, 3000.0), (4, 2000.0), (5, 1000.0)):
        path = root / f"phase10_verification_2026010{day}T000000Z.json"
        _write(path)
        _age(path, age)
    _write(root / "deployment_20260101T000000Z.json")  # durable -> protected
    _write(root / "notes.json")  # unrecognized -> protected

    sweep = enforce_retention(
        _settings(root),
        policy=StoragePolicy(report_max_files=2, report_max_bytes=10**9, min_free_bytes=0),
    )
    remaining = {p.name for p in root.iterdir()}
    assert sweep.deleted_files == 3
    assert sweep.triggers == ["count"]
    assert "phase10_verification_20260104T000000Z.json" in remaining
    assert "phase10_verification_20260105T000000Z.json" in remaining
    assert "deployment_20260101T000000Z.json" in remaining
    assert "notes.json" in remaining
    assert not (root / "phase10_verification_20260101T000000Z.json").exists()


def test_byte_retention_removes_oldest_until_budget_satisfied(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    for day in range(1, 7):
        path = root / f"experiment_2026020{day}T000000Z.json"
        _write(path, size=1000)
        _age(path, 10000.0 - day * 100)
    _write(root / "deployment_state.json")

    sweep = enforce_retention(
        _settings(root),
        policy=StoragePolicy(report_max_files=1000, report_max_bytes=2500, min_free_bytes=0),
    )
    remaining = {p.name for p in root.iterdir()}
    assert "bytes" in sweep.triggers
    assert sweep.deleted_files >= 4
    assert sweep.disposable_bytes - sweep.deleted_bytes <= 2500
    assert "deployment_state.json" in remaining


def test_malformed_and_unknown_files_are_not_deleted(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    for name in ("baseline_not-a-timestamp.json", "garbage.json", "README.txt"):
        path = root / name
        _write(path)
        _age(path, 10**6)
    _write(root / f"baseline_{_STAMP}.json")
    _age(root / f"baseline_{_STAMP}.json", 10**6)

    sweep = enforce_retention(
        _settings(root),
        policy=StoragePolicy(report_max_files=0, report_max_bytes=0, min_free_bytes=0),
    )
    remaining = {p.name for p in root.iterdir()}
    assert sweep.deleted_files == 1
    assert remaining == {"baseline_not-a-timestamp.json", "garbage.json", "README.txt"}


def test_path_outside_owned_directory_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    outside = tmp_path / "outside.json"
    _write(outside)
    policy = StoragePolicy(min_free_bytes=0)

    assert is_safe_disposable(outside, root.resolve()) is False
    result = guarded_write_text(outside, "nope", _settings(root), policy=policy)
    assert result.persisted is False
    assert result.degraded is True
    assert "outside" in result.reason


def test_symlink_escape_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    external = tmp_path / "external.json"
    _write(external)
    link = root / f"baseline_{_STAMP}.json"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("symlink creation is not permitted on this platform")
    _age(link, 10**6)

    sweep = enforce_retention(
        _settings(root),
        policy=StoragePolicy(report_max_files=0, min_free_bytes=0),
    )
    assert sweep.deleted_files == 0
    assert external.is_file()
    assert link.is_symlink()


def test_disk_pressure_triggers_cleanup_and_refuses_write(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    for day in range(1, 5):
        path = root / f"diagnosis_2026040{day}T000000Z.json"
        _write(path, size=500)
    settings = _settings(root)
    policy = StoragePolicy(report_max_files=10**6, report_max_bytes=10**12, min_free_bytes=1000)

    sweep = enforce_retention(settings, policy=policy, disk_usage=_ScriptedDisk([10]))
    assert "disk_free" in sweep.triggers
    assert sweep.deleted_files >= 1

    result = guarded_write_text(
        root / f"candidates_{_STAMP}.json",
        "{}",
        settings,
        policy=policy,
        disk_usage=_ScriptedDisk([10]),
    )
    assert result.persisted is False
    assert result.degraded is True
    assert result.disk_status == DiskSpaceStatus.CRITICAL


def test_disk_pressure_recovered_after_cleanup_allows_write(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    for day in range(1, 5):
        _write(root / f"diagnosis_2026050{day}T000000Z.json", size=500)
    settings = _settings(root)
    policy = StoragePolicy(report_max_files=10**6, report_max_bytes=10**12, min_free_bytes=1000)
    # low, low (free before), high (free after), high (recheck) -> allowed.
    disk = _ScriptedDisk([10, 10, 5000, 5000])
    result = guarded_write_text(
        root / f"hardware_profile_{_STAMP}.json", "{}", settings, policy=policy, disk_usage=disk
    )
    assert result.persisted is True


def test_atomic_write_appears_at_final_path_without_temp(tmp_path: Path) -> None:
    target = tmp_path / "reports" / f"storage_verification_{_STAMP}.json"
    atomic_write_text(target, '{"ok": true}')
    assert target.read_text(encoding="utf-8") == '{"ok": true}'
    assert not [p for p in target.parent.iterdir() if p.name.endswith(".tmp")]


def test_interrupted_atomic_write_leaves_no_final_or_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "reports" / f"validity_verification_{_STAMP}.json"

    def boom(src: object, dst: object) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(storage.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_text(target, "partial")
    assert not target.exists()
    assert not [p for p in target.parent.iterdir() if p.name.endswith(".tmp")]


def test_stale_temporaries_cleaned_fresh_kept(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    stale = root / ".baseline_x.json.123.tmp"
    fresh = root / ".baseline_y.json.456.tmp"
    _write(stale)
    _write(fresh)
    _age(stale, 10**6)

    sweep = enforce_retention(
        _settings(root), policy=StoragePolicy(report_max_files=10**6, min_free_bytes=0)
    )
    assert sweep.deleted_temp_files == 1
    assert not stale.exists()
    assert fresh.exists()


def test_disabled_policy_deletes_nothing(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    _write(root / f"diagnosis_{_STAMP}.json")
    sweep = enforce_retention(
        _settings(root), policy=StoragePolicy(enabled=False, report_max_files=0)
    )
    assert sweep.enabled is False
    assert sweep.deleted_files == 0
    assert (root / f"diagnosis_{_STAMP}.json").exists()


def test_startup_retention_is_non_fatal_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise PermissionError("injected retention failure")

    monkeypatch.setattr(storage, "enforce_retention", boom)
    result = run_startup_retention(_settings(tmp_path))
    assert result is None


def test_startup_retention_captures_individual_deletion_failure(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    for day in range(1, 4):
        path = root / f"experiment_2026060{day}T000000Z.json"
        _write(path)
        _age(path, 10000.0 - day * 100)
    locked = root / "experiment_20260601T000000Z.json"
    try:
        os.chmod(locked, stat.S_IREAD)
        sweep = enforce_retention(
            _settings(root),
            policy=StoragePolicy(report_max_files=1, min_free_bytes=0),
        )
        captured = bool(sweep.errors) or not locked.exists()
        assert captured
        # The sweep still completed and processed the remaining files.
        assert sweep.deleted_files >= 1
    finally:
        try:
            os.chmod(locked, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass


def test_file_logging_rotates_and_stays_bounded(tmp_path: Path) -> None:
    log_file = tmp_path / "phaseX.log"
    try:
        logger = setup_logging("INFO", log_file=log_file, max_bytes=2048, backup_count=2)
        for index in range(200):
            logger.info("line %d %s", index, "z" * 400)
        for handler in logger.handlers:
            handler.flush()
        backups = [p.name for p in tmp_path.iterdir() if p.name.startswith("phaseX.log.")]
        assert len(backups) <= 2
        assert log_file.is_file()
        assert log_file.read_text(encoding="utf-8")
    finally:
        setup_logging("INFO")


def test_disposable_name_classification() -> None:
    assert is_disposable_report_name(f"phase10r_verification_{_STAMP}.json")
    assert is_disposable_report_name(f"storage_verification_{_STAMP}.json")
    assert is_disposable_report_name(f"phase9r-noop-{_STAMP}_baseline.json")
    assert not is_disposable_report_name("deployment_20260101T000000Z.json")
    assert not is_disposable_report_name("knowledge.db")
    assert not is_disposable_report_name("random.json")
    assert not is_disposable_report_name("phase10_verification_notatimestamp.json")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows read-only semantics")
def test_windows_readonly_deletion_failure_recorded(tmp_path: Path) -> None:
    root = tmp_path / "reports"
    root.mkdir()
    path = root / f"diagnosis_{_STAMP}.json"
    _write(path)
    _age(path, 10**5)
    os.chmod(path, stat.S_IREAD)
    try:
        sweep = enforce_retention(
            _settings(root), policy=StoragePolicy(report_max_files=0, min_free_bytes=0)
        )
        assert sweep.errors
    finally:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
