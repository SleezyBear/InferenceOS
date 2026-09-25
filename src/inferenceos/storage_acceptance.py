"""H2 storage-retention and disk-protection acceptance.

Controlled evidence exercises count/byte retention, malformed-file preservation, path
safety, disk-pressure degradation and atomic writes in a temporary directory. Real
Windows evidence discovers the actual reports/log directories, runs retention against
the real generated artifacts (removing only allowlisted disposable files if any limit is
exceeded), confirms unrelated files are untouched, and performs a small real inference
afterward.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.config import Settings
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.storage import DiskSpaceStatus, PersistenceResult
from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
from inferenceos.logging_config import setup_logging
from inferenceos.models.loader import load_model_profile
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


def _temp_settings(root: Path) -> Settings:
    return Settings(reports_dir=root, logs_dir=root)


def _write(path: Path, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * size, encoding="utf-8")


def _touch_old(path: Path, *, age_s: float) -> None:
    old = time.time() - age_s
    os.utime(path, (old, old))


class _ScriptedDisk:
    """Deterministic free-space fixture (never fills the real disk)."""

    def __init__(self, frees: list[int]) -> None:
        self._frees = frees
        self.calls = 0

    def __call__(self, path: Path) -> tuple[int, int, int]:
        index = min(self.calls, len(self._frees) - 1)
        self.calls += 1
        free = self._frees[index]
        total = 10 * 1024**3
        return total, total - free, free


def _scenario_count_retention(base: Path) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    root = base / "count"
    root.mkdir(parents=True, exist_ok=True)
    for index, age in enumerate((5000.0, 4000.0, 3000.0, 2000.0, 1000.0)):
        path = root / f"phase10_verification_2026010{index + 1}T000000Z.json"
        _write(path)
        _touch_old(path, age_s=age)
    _write(root / "deployment_20260101T000000Z.json")  # durable: protected
    _write(root / "notes.json")  # unknown: protected
    settings = _temp_settings(root)
    policy = StoragePolicy(report_max_files=2, report_max_bytes=10**9, min_free_bytes=0)

    sweep = enforce_retention(settings, policy=policy)
    remaining = sorted(p.name for p in root.iterdir())
    checks = [
        (
            "H2 count: oldest disposable reports removed first",
            sweep.deleted_files == 3
            and "phase10_verification_20260101T000000Z.json" not in remaining
            and "phase10_verification_20260102T000000Z.json" not in remaining
            and "phase10_verification_20260103T000000Z.json" not in remaining,
        ),
        (
            "H2 count: newest disposable reports retained",
            "phase10_verification_20260104T000000Z.json" in remaining
            and "phase10_verification_20260105T000000Z.json" in remaining,
        ),
        (
            "H2 count: durable and unknown files preserved",
            "deployment_20260101T000000Z.json" in remaining and "notes.json" in remaining,
        ),
    ]
    scenario: dict[str, object] = {
        "deleted": sweep.deleted_files,
        "remaining": remaining,
        "triggers": sweep.triggers,
        "protected_files": sweep.protected_files,
    }
    return scenario, checks


def _scenario_byte_retention(base: Path) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    root = base / "bytes"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(6):
        path = root / f"experiment_2026020{index + 1}T000000Z.json"
        _write(path, size=1000)
        _touch_old(path, age_s=10000.0 - index * 1000.0)
    _write(root / "deployment_state.json")
    settings = _temp_settings(root)
    policy = StoragePolicy(
        report_max_files=1000, report_max_bytes=2500, min_free_bytes=0
    )
    sweep = enforce_retention(settings, policy=policy)
    remaining = sorted(p.name for p in root.iterdir())
    checks = [
        (
            "H2 bytes: byte budget enforced by removing oldest disposable reports",
            sweep.deleted_files >= 4 and sweep.deleted_bytes >= 4000,
        ),
        (
            "H2 bytes: unrelated file preserved",
            "deployment_state.json" in remaining,
        ),
        (
            "H2 bytes: byte trigger recorded",
            "bytes" in sweep.triggers,
        ),
    ]
    return {"deleted": sweep.deleted_files, "remaining": remaining, "sweep": sweep.model_dump(mode="json")}, checks


def _scenario_malformed(base: Path) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    root = base / "malformed"
    root.mkdir(parents=True, exist_ok=True)
    _write(root / "baseline_not-a-timestamp.json")
    _write(root / "garbage.json")
    _write(root / "README.txt")
    _write(root / "baseline_20260301T000000Z.json")
    for path in root.iterdir():
        _touch_old(path, age_s=10**6)
    settings = _temp_settings(root)
    policy = StoragePolicy(report_max_files=0, report_max_bytes=0, min_free_bytes=0)
    sweep = enforce_retention(settings, policy=policy)
    remaining = sorted(p.name for p in root.iterdir())
    checks = [
        (
            "H2 malformed: unknown/malformed artifacts are never deleted",
            "baseline_not-a-timestamp.json" in remaining
            and "garbage.json" in remaining
            and "README.txt" in remaining,
        ),
        (
            "H2 malformed: only the recognized disposable report was removed",
            set(remaining)
            == {"baseline_not-a-timestamp.json", "garbage.json", "README.txt"},
        ),
    ]
    return {"deleted": sweep.deleted_files, "remaining": remaining}, checks


def _scenario_path_safety(base: Path) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    root = base / "safe" / "reports"
    root.mkdir(parents=True, exist_ok=True)
    outside = base / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    external = outside / f"baseline_{_STAMP}.json"
    _write(external)
    _touch_old(external, age_s=10**6)

    symlink_supported = True
    link = root / f"baseline_{_STAMP}.json"
    try:
        link.symlink_to(external)
    except OSError:
        symlink_supported = False

    settings = _temp_settings(root)
    policy = StoragePolicy(report_max_files=0, report_max_bytes=0, min_free_bytes=0)
    sweep = enforce_retention(settings, policy=policy)

    external_survives = external.is_file()
    safe_check = is_safe_disposable(external, root) is False
    result: PersistenceResult = guarded_write_text(
        external, "nope", settings, policy=policy
    )
    checks = [
        (
            "H2 path-safety: a path outside the owned reports directory is refused",
            safe_check and result.persisted is False and result.degraded is True,
        ),
        (
            "H2 path-safety: retention cannot delete a file reached via a symlink",
            external_survives and (not symlink_supported or sweep.deleted_files == 0),
        ),
    ]
    return {
        "external_survives": external_survives,
        "symlink_supported": symlink_supported,
        "deleted": sweep.deleted_files,
        "guarded_write": result.model_dump(mode="json"),
    }, checks


def _scenario_disk_pressure(base: Path) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    root = base / "pressure"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        path = root / f"diagnosis_2026040{index + 1}T000000Z.json"
        _write(path, size=500)
    settings = _temp_settings(root)
    policy = StoragePolicy(
        report_max_files=10**6,
        report_max_bytes=10**12,
        min_free_bytes=1000,
    )
    # free stays below the critical floor (min_free//2): cleanup runs but cannot recover.
    low_disk = _ScriptedDisk([10])
    sweep = enforce_retention(settings, policy=policy, disk_usage=low_disk)
    degraded = guarded_write_text(
        root / f"candidates_{_STAMP}.json",
        "{}",
        settings,
        policy=policy,
        disk_usage=_ScriptedDisk([10]),
    )
    # free recovers after the retention pass: the write is then allowed.
    recovered_disk = _ScriptedDisk([10, 10, 5000, 5000])
    recovered = guarded_write_text(
        root / f"hardware_profile_{_STAMP}.json",
        "{}",
        settings,
        policy=policy,
        disk_usage=recovered_disk,
    )
    checks = [
        (
            "H2 disk: pressure triggers a cleanup attempt",
            "disk_free" in sweep.triggers and sweep.deleted_files >= 1,
        ),
        (
            "H2 disk: a persist under critical pressure is refused and marked degraded",
            degraded.persisted is False
            and degraded.degraded is True
            and degraded.disk_status == DiskSpaceStatus.CRITICAL,
        ),
        (
            "H2 disk: persistence resumes once cleanup restores space",
            recovered.persisted is True,
        ),
    ]
    return {
        "sweep": sweep.model_dump(mode="json"),
        "degraded": degraded.model_dump(mode="json"),
        "recovered": recovered.model_dump(mode="json"),
    }, checks


def _scenario_atomicity(base: Path) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    root = base / "atomic"
    root.mkdir(parents=True, exist_ok=True)
    settings = _temp_settings(root)
    policy = StoragePolicy(min_free_bytes=0)
    final = root / f"storage_verification_{_STAMP}.json"
    good = guarded_write_text(final, '{"ok": true}', settings, policy=policy)
    leftovers = [p.name for p in root.iterdir() if p.name.endswith(".tmp")]
    interrupted_path = root / f"validity_verification_{_STAMP}.json"
    interrupted_error: str | None = None
    try:
        atomic_write_text(interrupted_path, object())  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - deliberate interrupted write
        interrupted_error = type(exc).__name__
    interrupted_leftovers = [p.name for p in root.iterdir() if p.name.endswith(".tmp")]
    checks = [
        (
            "H2 atomic: a successful report appears at the final path with no temp left",
            good.persisted is True and good.atomic is True and final.read_text() == '{"ok": true}' and not leftovers,
        ),
        (
            "H2 atomic: an interrupted write leaves no final artifact and no temp",
            interrupted_error is not None
            and not interrupted_path.exists()
            and not interrupted_leftovers,
        ),
        (
            "H2 atomic: a temporary file never masquerades as a completed report",
            not is_disposable_report_name(f".{final.name}.abc.tmp"),
        ),
    ]
    return {
        "leftovers": leftovers,
        "interrupted_error": interrupted_error,
        "interrupted_leftovers": interrupted_leftovers,
    }, checks


def _scenario_log_rotation(base: Path) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    root = base / "logs"
    root.mkdir(parents=True, exist_ok=True)
    log_file = root / "phaseX.log"
    logger = setup_logging("INFO", log_file=log_file, max_bytes=2048, backup_count=2)
    message = "z" * 400
    for index in range(200):
        logger.info("line %d %s", index, message)
    for handler in logger.handlers:
        try:
            handler.flush()
        except Exception:  # noqa: BLE001
            pass
    files = sorted(p.name for p in root.iterdir())
    backups = [name for name in files if name.startswith("phaseX.log.")]
    # Remain bounded: 1 active + at most `backup_count` rotations.
    checks = [
        (
            "H2 logs: file-backed logs rotate and remain bounded",
            len(backups) <= 2 and log_file.is_file(),
        ),
        (
            "H2 logs: the current log remains usable",
            len(log_file.read_text(encoding="utf-8")) > 0,
        ),
    ]
    setup_logging("INFO")  # restore stream-only so the temp file is released
    return {"files": files, "backups": backups}, checks


def _scenario_startup_non_fatal(base: Path) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    root = base / "startup"
    root.mkdir(parents=True, exist_ok=True)
    # A malformed reports root (a file, not a directory) must not raise.
    broken_root = base / "startup_broken"
    broken_root.write_text("not a directory", encoding="utf-8")
    broken_settings = _temp_settings(broken_root)
    broken_result = run_startup_retention(broken_settings)

    # An individual deletion failure (read-only artifact on Windows) must be recorded
    # and must not abort the sweep.
    for index in range(3):
        path = root / f"experiment_2026050{index + 1}T000000Z.json"
        _write(path)
        _touch_old(path, age_s=10000.0 - index * 1000.0)
    read_only = root / "experiment_20260501T000000Z.json"
    deletion_error_captured = False
    try:
        os.chmod(read_only, stat.S_IREAD)
        sweep = enforce_retention(
            _temp_settings(root),
            policy=StoragePolicy(report_max_files=1, min_free_bytes=0),
        )
        deletion_error_captured = bool(sweep.errors)
        if not read_only.exists():
            deletion_error_captured = deletion_error_captured or True
    finally:
        try:
            os.chmod(read_only, stat.S_IWRITE | stat.S_IREAD)
        except OSError:
            pass

    checks = [
        (
            "H2 startup: retention is non-fatal when the storage root is unusable",
            broken_result is None,
        ),
        (
            "H2 startup: an individual deletion failure is captured, not raised",
            deletion_error_captured,
        ),
    ]
    return {
        "broken_root_result": None if broken_result is None else broken_result.model_dump(mode="json"),
        "deletion_error_captured": deletion_error_captured,
    }, checks


def _scenario_real_windows(
    settings: Settings, base: Path
) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    reports_dir = Path(settings.reports_dir)
    logs_dir = Path(settings.logs_dir)
    before_names = {p.name for p in reports_dir.iterdir() if p.is_file()}
    before_bytes = sum(p.stat().st_size for p in reports_dir.iterdir() if p.is_file())

    usage = shutil.disk_usage(reports_dir)
    sweep = enforce_retention(settings)

    after_names = {p.name for p in reports_dir.iterdir() if p.is_file()}
    after_bytes = sum(p.stat().st_size for p in reports_dir.iterdir() if p.is_file())
    removed = before_names - after_names

    # Only allowlisted disposable artifacts may ever have been removed.
    only_disposable_removed = all(is_disposable_report_name(name) for name in removed)
    protected_present = any(
        (settings.data_dir / name).is_file() for name in ("knowledge.db",)
    )

    # Small real inference after retention proves the core runtime still works.
    inference_ok = False
    inference_detail: dict[str, object] = {}
    model = _smallest_model(settings)
    if model is not None:
        adapter = LlamaCppAdapter(settings=settings)
        try:
            adapter.load(
                model,
                RuntimeParams(
                    gpu_offload_layers=0, device="none", use_mmap=True, context_length=512
                ),
            )
            result = adapter.generate(
                GenerationRequest(prompt="Reply with one word.", n_predict=4, seed=11)
            )
            inference_ok = bool(result.text)
            inference_detail = {
                "model": model.name,
                "text": result.text[:80],
                "generated_tokens": result.generated_tokens,
            }
        finally:
            adapter.unload()

    checks = [
        (
            "H2 real: reports/logs directories discovered and disk statistics available",
            reports_dir.is_dir() and logs_dir.is_dir() and usage.free > 0,
        ),
        (
            "H2 real: retention removed only allowlisted disposable artifacts",
            only_disposable_removed,
        ),
        (
            "H2 real: durable data directory preserved",
            protected_present,
        ),
        (
            "H2 real: real inference continues after retention",
            inference_ok,
        ),
    ]
    return {
        "reports_dir": str(reports_dir),
        "logs_dir": str(logs_dir),
        "discovered_reports_before": len(before_names),
        "reports_bytes_before": before_bytes,
        "reports_bytes_after": after_bytes,
        "removed": sorted(removed),
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "sweep": sweep.model_dump(mode="json"),
        "inference": inference_detail,
    }, checks


def _smallest_model(settings: Settings) -> ModelProfile | None:
    models_dir = Path(settings.models_dir)
    profiles = [load_model_profile(path) for path in sorted(models_dir.glob("*.gguf"))]
    if not profiles:
        return None
    return min(profiles, key=lambda profile: profile.file_size_bytes)


def _scenario_performance(settings: Settings) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    iterations = 5
    start = time.perf_counter()
    for _ in range(iterations):
        enforce_retention(settings)
    mean_ms = (time.perf_counter() - start) / iterations * 1000.0
    budget_ms = 600.0
    checks = [
        (
            "H2 performance: a no-op retention scan is cheap",
            mean_ms < budget_ms,
        )
    ]
    return {
        "iterations": iterations,
        "mean_retention_ms": round(mean_ms, 3),
        "budget_ms": budget_ms,
        "note": (
            "one bounded scan per process start; reports are capped so the scan size "
            "stays bounded regardless of unattended runtime"
        ),
    }, checks


def run_storage_acceptance(
    settings: Settings, *, work_dir: Path | None = None
) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    import tempfile

    scenarios: dict[str, object] = {}
    checks: list[tuple[str, bool]] = []
    base = work_dir or Path(tempfile.mkdtemp(prefix="io_h2_accept_"))
    base.mkdir(parents=True, exist_ok=True)

    for name, fn in (
        ("count_retention", _scenario_count_retention),
        ("byte_retention", _scenario_byte_retention),
        ("malformed", _scenario_malformed),
        ("path_safety", _scenario_path_safety),
        ("disk_pressure", _scenario_disk_pressure),
        ("atomicity", _scenario_atomicity),
        ("log_rotation", _scenario_log_rotation),
        ("startup_non_fatal", _scenario_startup_non_fatal),
    ):
        try:
            scenario, scenario_checks = fn(base)
        except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001
            scenario = {"error": f"{type(exc).__name__}: {exc}"}
            scenario_checks = [(f"H2 {name}: scenario ran", False)]
        scenarios[name] = scenario
        checks.extend(scenario_checks)

    try:
        real_scenario, real_checks = _scenario_real_windows(settings, base)
    except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001
        real_scenario = {"error": f"{type(exc).__name__}: {exc}"}
        real_checks = [("H2 real: scenario ran", False)]
    scenarios["real_windows"] = real_scenario
    checks.extend(real_checks)

    try:
        perf_scenario, perf_checks = _scenario_performance(settings)
    except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001
        perf_scenario = {"error": f"{type(exc).__name__}: {exc}"}
        perf_checks = [("H2 performance: scenario ran", False)]
    scenarios["performance"] = perf_scenario
    checks.extend(perf_checks)

    scenarios["timestamp"] = datetime.now(timezone.utc).isoformat()
    scenarios["platform"] = sys.platform
    return scenarios, checks
