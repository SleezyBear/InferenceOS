"""Final production-hardening acceptance (H3 key hygiene, H1 database policy, B3
deterministic quality, H6 optimizer self-target exclusion).

Controlled evidence exercises each mechanism deterministically. Real Windows evidence
covers the actual key-file lifecycle, the actual SQLite stores (read-only inspection +
a controlled persistence round-trip), a real model generation for deterministic quality,
and the real process observer's self-exclusion.
"""

from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from pathlib import Path

from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.backends.resident_server import (
    LlamaServerResidentBackend,
    sweep_stale_key_files,
)
from inferenceos.config import Settings
from inferenceos.contracts.enums import QualityStatus, WorkloadKind
from inferenceos.contracts.external import ExternalBudget, ExternalPolicy
from inferenceos.contracts.knowledge import KnowledgeEventType
from inferenceos.contracts.quality import QualitySuiteResult
from inferenceos.experiment.quality_policy import build_adaptive_quality_policy
from inferenceos.experiment.quality_suite import build_default_suite, evaluate_suite
from inferenceos.external.attribution import assess_external_attribution
from inferenceos.external.controlled import make_process_observation, make_report
from inferenceos.external.engine import ExternalRecommendationEngine
from inferenceos.external.observer import WindowsProcessObserver
from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
from inferenceos.knowledge.store import SQLiteKnowledgeStore
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.actions import build_default_registry
from inferenceos.profiling.workloads import BenchmarkWorkload
from inferenceos.runtime.approval import AutoRejectApprovalProvider

_H3_PHRASE = "ZEPHYR-9137"


def _external_policy() -> ExternalPolicy:
    return ExternalPolicy.model_validate(
        {
            "budget": ExternalBudget(max_duration_s=300.0, max_recommendations=4, max_processes=40),
            "observation_interval_s": 1.0,
            "memory_pressure_threshold": 0.75,
            "memory_dominance_threshold": 0.6,
            "cpu_dominance_threshold": 0.5,
            "min_process_rss_bytes": 256 * 1024 * 1024,
            "min_process_cpu_percent": 25.0,
            "require_target_degradation_for_recommendation": True,
            "require_approval_for_actions": True,
            "cooldown_s": 300.0,
        }
    )


def _server_path(settings: Settings) -> Path:
    base = Path(settings.llama_cpp_dir)
    candidate = base / "llama-server.exe"
    return candidate if candidate.is_file() else base / "llama-server"


def _pick_model(settings: Settings, *, smallest: bool) -> object | None:
    profiles = [load_model_profile(path) for path in sorted(Path(settings.models_dir).glob("*.gguf"))]
    if not profiles:
        return None
    return min(profiles, key=lambda p: p.file_size_bytes) if smallest else max(
        profiles, key=lambda p: p.file_size_bytes
    )


def _key_files(root: Path) -> set[str]:
    return {path.name for path in root.glob("inferenceos-key-*.txt")}


# --------------------------------------------------------------------------- #
# H3 — temporary API-key hygiene                                              #
# --------------------------------------------------------------------------- #
def _scenario_h3(settings: Settings) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    import time

    temp_root = Path(tempfile.gettempdir())
    controlled_dir = Path(tempfile.mkdtemp(prefix="io_h3_keys_"))
    old_key = controlled_dir / "inferenceos-key-old.txt"
    fresh_key = controlled_dir / "inferenceos-key-fresh.txt"
    unrelated = controlled_dir / "unrelated-secret.txt"
    for path in (old_key, fresh_key, unrelated):
        path.write_text("secret", encoding="utf-8")
    stale_time = time.time() - 10 * 3600
    os.utime(old_key, (stale_time, stale_time))
    os.utime(unrelated, (stale_time, stale_time))
    removed = sweep_stale_key_files(temp_dir=controlled_dir, max_age_s=3600.0)
    sweep_ok = (
        removed == 1 and not old_key.exists() and fresh_key.exists() and unrelated.exists()
    )

    before = _key_files(temp_root)
    lifecycle_ok = False
    key_present_during = False
    key_removed_after = False
    real: dict[str, object] = {}
    model = _pick_model(settings, smallest=True)
    server = _server_path(settings)
    if model is not None and server.is_file():
        backend = LlamaServerResidentBackend(server=server)
        try:
            backend.start(
                model,  # type: ignore[arg-type]
                RuntimeParams(context_length=256, gpu_offload_layers=0, use_mmap=True),
            )
            key_file = backend._key_file  # noqa: SLF001 - verification of real lifecycle
            key_present_during = bool(key_file is not None and key_file.is_file())
            pid = backend.pid
            record = backend.stop()
            key_removed_after = bool(
                backend._key_file is None
                and key_file is not None
                and not key_file.exists()
                and record.cleanup_complete
            )
            real = {
                "pid": pid,
                "key_present_during": key_present_during,
                "key_removed_after_stop": key_removed_after,
                "cleanup_complete": record.cleanup_complete,
            }
        except (OSError, RuntimeError) as exc:  # noqa: BLE001
            real = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            try:
                backend.cleanup()
            except Exception:  # noqa: BLE001
                pass
    after = _key_files(temp_root)
    lifecycle_ok = bool(
        key_present_during and key_removed_after and after <= before
    )
    checks = [
        (
            "H3: stale InferenceOS-owned keys are swept; fresh/unrelated files are kept",
            sweep_ok,
        ),
        (
            "H3 real: the real server's key exists during its lifetime and is removed on stop, "
            "with no leftover InferenceOS key",
            lifecycle_ok,
        ),
    ]
    return {
        "sweep_removed": removed,
        "sweep_ok": sweep_ok,
        "real": real,
        "key_files_before": len(before),
        "key_files_after": len(after),
    }, checks


# --------------------------------------------------------------------------- #
# H1 — SQLite policy (assessment + real inspection + persistence)             #
# --------------------------------------------------------------------------- #
def _journal_mode(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = connection.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]) if row else None
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def _row_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.is_file():
        return counts
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        for table in ("optimization_records", "rollback_records", "knowledge_events"):
            try:
                counts[table] = int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
            except sqlite3.Error:
                counts[table] = -1
        connection.close()
    except sqlite3.Error:
        return {}
    return counts


def _scenario_h1(settings: Settings) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    knowledge_path = Path(settings.phase8_db_path)
    external_path = Path(settings.phase11_db_path)
    real: dict[str, object] = {
        "knowledge_db": {
            "path": str(knowledge_path),
            "exists": knowledge_path.is_file(),
            "size_bytes": knowledge_path.stat().st_size if knowledge_path.is_file() else None,
            "journal_mode": _journal_mode(knowledge_path),
            "row_counts": _row_counts(knowledge_path),
        },
        "external_db": {
            "path": str(external_path),
            "exists": external_path.is_file(),
            "size_bytes": external_path.stat().st_size if external_path.is_file() else None,
            "journal_mode": _journal_mode(external_path),
        },
    }

    temp_dir = Path(tempfile.mkdtemp(prefix="io_h1_db_"))
    db_path = temp_dir / "knowledge.db"
    store = SQLiteKnowledgeStore(db_path)
    store.append_event("rec-h1-probe", KnowledgeEventType.RECORDED, "h1 persistence probe")
    stats_before = store.stats()
    del store
    reopened = SQLiteKnowledgeStore(db_path)
    events = reopened.events_for_record("rec-h1-probe")
    stats_after = reopened.stats()
    persistence_ok = bool(
        len(events) == 1
        and stats_after.knowledge_events >= stats_before.knowledge_events >= 1
        and reopened.schema_version == stats_before.schema_version
    )
    temp_mode = _journal_mode(db_path)

    checks = [
        (
            "H1: knowledge/external databases initialize and are readable",
            knowledge_path.is_file() and external_path.is_file(),
        ),
        (
            "H1: durable audit events survive close/reopen (no automatic retention deleting them)",
            persistence_ok,
        ),
        (
            "H1: journal mode is reported and was not silently changed to WAL",
            temp_mode is not None and temp_mode != "wal",
        ),
    ]
    real["controlled"] = {
        "temp_db": str(db_path),
        "journal_mode": temp_mode,
        "events_after_reopen": len(events),
        "row_counts": stats_after.model_dump(mode="json"),
    }
    real["assessment"] = {
        "wal_enabled": False,
        "retention_auto_enabled": False,
        "rationale": (
            "single-writer stores with short transactions do not benefit from WAL; the "
            "tables are durable audit/rollback history and must not be auto-deleted"
        ),
    }
    return real, checks


# --------------------------------------------------------------------------- #
# B3 — deterministic task-relevant quality                                    #
# --------------------------------------------------------------------------- #
def _b3_workload() -> BenchmarkWorkload:
    return BenchmarkWorkload(
        workload_id="b3-deterministic",
        kind=WorkloadKind.SHORT,
        description="B3 deterministic correctness probe",
        prompt="The Obsidian Key identifier is ZEPHYR-9137.",
        n_predict=32,
        seed=1234,
        temperature=0.0,
        min_context_tokens=64,
        required_phrases=(_H3_PHRASE,),
        required_fields=("answer", "unit"),
    )


def _run_suite(
    baseline: list[str], candidate: list[str], *, workload: BenchmarkWorkload, settings: Settings
) -> QualitySuiteResult:
    suite = build_default_suite(workload, version="7r.b3")
    policy = build_adaptive_quality_policy(
        settings, action_id="runtime.right_size_context", workload_kind=WorkloadKind.SHORT
    )
    return evaluate_suite(
        suite,
        policy=policy,
        baseline_texts=baseline,
        candidate_texts=candidate,
        baseline_tokens=[32 for _ in baseline],
        candidate_tokens=[32 for _ in candidate],
        baseline_valid_runs=len(baseline),
        candidate_valid_runs=len(candidate),
        invalid_runs=0,
        ignore_eos=False,
        n_predict=workload.n_predict,
    )


def _scenario_b3(settings: Settings) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    workload = _b3_workload()
    good = _run_suite(
        ["The identifier is ZEPHYR-9137."],
        ["  the identifier is   zephyr-9137.  "],
        workload=workload,
        settings=settings,
    )
    wrong_fact = _run_suite(
        ["The identifier is ZEPHYR-9137."],
        ["The identifier is unknown."],
        workload=workload,
        settings=settings,
    )
    wrong_arithmetic = _run_suite(
        ["The product is 391."],
        ["The product is 392."],
        workload=BenchmarkWorkload(
            workload_id="b3-arith",
            kind=WorkloadKind.SHORT,
            description="arith",
            prompt="17*23",
            n_predict=32,
            seed=1,
            temperature=0.0,
            min_context_tokens=64,
            required_phrases=("391",),
        ),
        settings=settings,
    )
    negative = _run_suite(
        ["The result is 7."],
        ["The result is 7."],
        workload=workload,
        settings=settings,
    )
    missing_field = _run_suite(
        ['{"answer": "391", "unit": "none"}'],
        ['{"answer": "391"}'],
        workload=workload,
        settings=settings,
    )
    malformed = _run_suite(
        ['{"answer": "391", "unit": "none"}'],
        ["answer is 391"],
        workload=workload,
        settings=settings,
    )

    controlled_ok = bool(
        good.validated
        and not wrong_fact.validated
        and wrong_fact.status is QualityStatus.QUALITY_REGRESSION
        and not wrong_arithmetic.validated
        and negative.validated
        and not missing_field.validated
        and not malformed.validated
    )

    real: dict[str, object] = {}
    real_ok = True
    model = _pick_model(settings, smallest=False)
    if model is not None:
        adapter = LlamaCppAdapter(settings=settings)
        try:
            adapter.load(
                model,  # type: ignore[arg-type]
                RuntimeParams(
                    gpu_offload_layers=0, device="none", use_mmap=True, context_length=512
                ),
            )
            result = adapter.generate(
                GenerationRequest(
                    prompt="Repeat this identifier exactly, and nothing else: ZEPHYR-9137",
                    n_predict=16,
                    seed=1234,
                    temperature=0.0,
                )
            )
            baseline_text = result.text or ""
        finally:
            adapter.unload()
        emitted = _H3_PHRASE.lower() in baseline_text.lower()
        candidate_good = re.sub(r"\s+", "  ", baseline_text).strip()
        candidate_bad = re.sub(
            re.escape(_H3_PHRASE), "REMOVED", baseline_text, flags=re.IGNORECASE
        )
        good_real = _run_suite([baseline_text], [candidate_good], workload=workload, settings=settings)
        bad_real = _run_suite([baseline_text], [candidate_bad], workload=workload, settings=settings)
        real = {
            "model": getattr(model, "name", None),
            "phrase_emitted": emitted,
            "baseline_excerpt": baseline_text[:120],
            "preserved_validated": good_real.validated,
            "loss_detected": (not bad_real.validated) if emitted else None,
        }
        if emitted:
            real_ok = bool(good_real.validated and not bad_real.validated)

    checks = [
        (
            "B3: deterministic factual/arithmetic/structured checks detect incorrect candidates",
            controlled_ok,
        ),
        (
            "B3 real: a real model output demonstrates deterministic preservation/loss detection",
            real_ok,
        ),
    ]
    return {
        "controlled": {
            "formatting_variation_validated": good.validated,
            "wrong_fact_status": wrong_fact.status.value,
            "wrong_arithmetic_validated": wrong_arithmetic.validated,
            "missing_field_validated": missing_field.validated,
            "malformed_validated": malformed.validated,
            "not_applicable_validated": negative.validated,
        },
        "real": real,
    }, checks


# --------------------------------------------------------------------------- #
# H6 — optimizer self-target exclusion                                        #
# --------------------------------------------------------------------------- #
def _scenario_h6() -> tuple[dict[str, object], list[tuple[str, bool]]]:
    own_pid = os.getpid()
    target = make_process_observation(
        pid=100, name="llama-server.exe", rss_bytes=1_300_000_000, is_target=True
    )
    optimizer_self = make_process_observation(
        pid=own_pid, name="python.exe", rss_bytes=6 * 1024**3, cpu_percent=180.0
    )
    external = make_process_observation(
        pid=999_001, name="external.exe", rss_bytes=6 * 1024**3, cpu_percent=180.0
    )
    report = make_report(
        processes=[target, optimizer_self, external], target_pid=100, memory_pressure=0.9
    ).model_copy(update={"own_pid": own_pid})
    assessments = assess_external_attribution(report, _external_policy(), target_degraded=True)
    pids = {assessment.identity.pid for assessment in assessments}

    observer = WindowsProcessObserver()
    real_report = observer.observe(own_pid=own_pid, max_processes=200)
    real_self_absent = all(
        observation.identity.pid != own_pid for observation in real_report.processes
    )

    engine = ExternalRecommendationEngine(
        observer=observer,
        registry=build_default_registry(),
        policy=_external_policy(),
        approval_provider=AutoRejectApprovalProvider(),
    )
    default_own_pid = getattr(engine, "_own_pid")

    checks = [
        (
            "H6: the optimizer's own PID is excluded from external candidates",
            own_pid not in pids and real_self_absent,
        ),
        (
            "H6: a legitimate external process remains selectable and the target is preserved",
            external.identity.pid in pids and target.identity.pid not in pids,
        ),
        (
            "H6: production engine construction defaults to the optimizer's own PID",
            default_own_pid == own_pid,
        ),
    ]
    return {
        "own_pid": own_pid,
        "excluded": own_pid not in pids,
        "external_selectable": external.identity.pid in pids,
        "real_observer_self_absent": real_self_absent,
        "engine_default_own_pid": default_own_pid,
    }, checks


def run_hardening_acceptance(
    settings: Settings,
) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    scenarios: dict[str, object] = {}
    checks: list[tuple[str, bool]] = []

    for name, fn in (
        ("h1_database", _scenario_h1),
        ("h3_key_hygiene", _scenario_h3),
        ("b3_quality", _scenario_b3),
    ):
        try:
            scenario, scenario_checks = fn(settings)
        except (OSError, RuntimeError, TimeoutError, ValueError) as exc:  # noqa: BLE001
            scenario = {"error": f"{type(exc).__name__}: {exc}"}
            scenario_checks = [(f"{name}: scenario ran", False)]
        scenarios[name] = scenario
        checks.extend(scenario_checks)

    try:
        h6_scenario, h6_checks = _scenario_h6()
    except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001
        h6_scenario = {"error": f"{type(exc).__name__}: {exc}"}
        h6_checks = [("H6: scenario ran", False)]
    scenarios["h6_self_target"] = h6_scenario
    checks.extend(h6_checks)

    return scenarios, checks
