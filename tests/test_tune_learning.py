"""C4-1: real Practical observations and durable, isolated, non-authoritative history."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from typing import Any

import pytest

from inferenceos.contracts.tune_learning import TuneHistoryRun
from inferenceos.tune.learning import capture_identity, persist_completed, workload_identity
from inferenceos.tune.learning_store import HistoryError, TuneHistoryStore, history_paths
from tests.test_budgeted_tune import Clock, _practical, _round, _world, clock  # noqa: F401


def record(name: str = "run-1") -> TuneHistoryRun:
    return TuneHistoryRun(run_id=name, identity={"missing": ["test identity"]}, observations=(),
        comparisons=(), disposition={"saved": False}, source_report_sha256="a" * 64, origin="simulation")


def test_read_missing_history_does_not_create_directory(tmp_path: Path) -> None:
    store = TuneHistoryStore(tmp_path / "absent")
    assert store.get("none") is None
    assert not store.path.parent.exists()


def test_roundtrip_idempotence_and_conflict(tmp_path: Path) -> None:
    store = TuneHistoryStore(tmp_path)
    r = record()
    assert store.append(r) is True
    assert TuneHistoryStore(tmp_path).get(r.run_id) == r
    assert store.append(r) is False
    with pytest.raises(HistoryError, match="conflicting"):
        store.append(r.model_copy(update={"disposition": {"saved": True}}))
    assert store.get(r.run_id) == r


@pytest.mark.parametrize("kind", ["checksum", "version", "row_identity", "garbage"])
def test_corrupt_or_incompatible_store_is_never_empty_healthy_history(tmp_path: Path, kind: str) -> None:
    store = TuneHistoryStore(tmp_path)
    store.append(record())
    if kind == "garbage":
        store.path.write_bytes(b"not sqlite")
    else:
        with sqlite3.connect(store.path) as db:
            if kind == "checksum":
                db.execute("UPDATE tune_history_runs SET checksum='wrong'")
            elif kind == "row_identity":
                db.execute("UPDATE tune_history_runs SET run_id='other'")
            else:
                db.execute("UPDATE tune_history_meta SET version='future'")
    with pytest.raises(HistoryError):
        store.get("other" if kind == "row_identity" else "run-1")


def test_unknown_schema_is_not_migrated_or_overwritten(tmp_path: Path) -> None:
    store = TuneHistoryStore(tmp_path)
    with sqlite3.connect(store.path) as db:
        db.execute("CREATE TABLE unrelated (value TEXT)")
    with pytest.raises(HistoryError):
        store.append(record())
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("unrelated",)]


def test_busy_store_is_bounded_and_keeps_existing_run(tmp_path: Path) -> None:
    store = TuneHistoryStore(tmp_path, timeout_s=0.01)
    store.append(record())
    with sqlite3.connect(store.path) as db:
        db.execute("BEGIN IMMEDIATE")
        with pytest.raises(HistoryError, match="locked"):
            store.append(record("second"))
        db.rollback()
    assert store.get("run-1") == record()
    assert store.get("second") is None


def test_nonfinite_nested_payload_cannot_be_silently_changed_to_null(tmp_path: Path) -> None:
    store = TuneHistoryStore(tmp_path)
    with pytest.raises(ValueError):
        store.append(record().model_copy(update={"identity": {"bad": float("nan")}}))
    assert not store.path.exists()


def test_fresh_processes_preserve_concurrent_distinct_runs(tmp_path: Path) -> None:
    code = """import sys
from pathlib import Path
from inferenceos.tune.learning_store import TuneHistoryStore
from inferenceos.contracts.tune_learning import TuneHistoryRun
s=TuneHistoryStore(Path(sys.argv[1]),timeout_s=3)
r=TuneHistoryRun(run_id=sys.argv[2],identity={},observations=(),comparisons=(),disposition={},source_report_sha256='a'*64)
s.append(r)
assert s.get(sys.argv[2]) == r
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    processes = [subprocess.Popen([sys.executable, "-B", "-c", code, str(tmp_path), f"run-{n}"],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for n in range(4)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, (stdout, stderr)
    assert all(TuneHistoryStore(tmp_path).get(f"run-{n}") is not None for n in range(4))


def test_practical_records_actual_costs_comparisons_and_final_disposition(tmp_path: Path, clock: Clock) -> None:
    tuner, _, driver, _ = _practical(tmp_path, _world(clock, answer_s=5.123456789), 1600, clock=clock)
    result = driver.run()
    collector = tuner._learning
    assert collector.diagnostics == []
    assert collector.observations and collector.comparisons
    budget = _round(result, "time budget")
    measured = [o for o in collector.observations if o.kind == "screening"]
    ops = {o["n"]: o for o in budget["operations"]}
    assert all(round(o.wall_s, 2) == ops[o.operation_id]["wall_s"] for o in measured)
    assert any(o.wall_s != round(o.wall_s, 2) for o in measured)
    assert all(o.request["policy"] is not None for o in measured)
    assert all(o.phases["raw"]["valid_runs"] == o.valid_runs for o in measured)
    assert all(o.request["configuration"]["context_length"] == tuner.context_length for o in measured)
    assert any(o.kind == "final-side" for o in collector.observations)
    assert all(o.status == "incomplete" and o.request["policy"] is None
               for o in collector.observations if o.kind == "final-side")
    path = tmp_path / "report.json"
    path.write_text(result.to_json(), encoding="utf-8")
    original = path.read_bytes()
    assert persist_completed(tuner, result, path, origin="simulation") == "recorded"
    assert persist_completed(tuner, result, path, origin="simulation") == "already recorded"
    saved = TuneHistoryStore(tuner.settings.data_dir).get(result.run_id)
    assert saved and saved.disposition["saved"] == (result.saved_setup is not None)
    assert saved.origin == "simulation" and saved.reusable_for_acceptance is False
    assert saved.started_at == result.started_at
    assert all(c["raw_noise"] is not None for c in saved.comparisons)
    assert saved.source_report_sha256 == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original
    # Reused measurements remain references; they are not duplicated as new costs.
    ids = [o.operation_id for o in saved.observations]
    assert len(set(ids)) == len(ids)
    assert len(saved.comparisons) == len(budget["screenings"])
    before = saved.comparisons[0]["reference"]["configuration"].copy()
    collector.comparisons[0]["reference"]["configuration"]["threads"] = 999
    reloaded = TuneHistoryStore(tuner.settings.data_dir).get(result.run_id)
    assert reloaded is not None
    assert reloaded.comparisons[0]["reference"]["configuration"] == before
    assert {c["path"] for c in saved.comparisons} <= {"layers", "threads", "prompt_batch"}


def test_identity_hashes_content_without_claiming_verified_engine(tmp_path: Path, clock: Clock) -> None:
    tuner, _, _, _ = _practical(tmp_path, _world(clock), 800, clock=clock)
    identity = capture_identity(tuner.model_path, tuner.machine, tuner.context_length, tuner.settings)
    assert identity["model"]["sha256"] == hashlib.sha256(tuner.model_path.read_bytes()).hexdigest()
    assert identity["engine_bundle_verified"] is False and identity["missing"]


def test_changed_workload_under_same_name_has_different_identity() -> None:
    from dataclasses import replace
    from inferenceos.profiling.workloads import get_workload
    w = get_workload("short-short")
    a, b = workload_identity(w), workload_identity(replace(w, prompt="different"))
    assert a["id"] == b["id"] and a["sha256"] != b["sha256"]


@pytest.mark.parametrize("problem", ["unpublished", "mismatched", "blocked", "store_unavailable", "model_changed"])
def test_persistence_failure_cannot_change_tune_result(tmp_path: Path, clock: Clock, problem: str) -> None:
    tuner, _, driver, _ = _practical(tmp_path, _world(clock), 800, clock=clock)
    identity = capture_identity(tuner.model_path, tuner.machine, tuner.context_length, tuner.settings)
    result = driver.run()
    original = result.to_json()
    path = tmp_path / "report.json"
    if problem != "unpublished":
        path.write_text(result.to_json(), encoding="utf-8")
    if problem == "mismatched":
        path.write_text(result.model_copy(update={"run_id": "other"}).to_json(), encoding="utf-8")
    if problem == "blocked":
        tuner._learning.blocked_cleanup = True
    if problem == "store_unavailable":
        store = TuneHistoryStore(tuner.settings.data_dir)
        store.path.mkdir(parents=True)
    if problem == "model_changed":
        tuner.model_path.write_bytes(b"changed model")
    status = persist_completed(tuner, result, path, identity=identity, origin="simulation")
    assert result.to_json() == original
    if problem == "model_changed":
        assert status == "recorded"
        saved = TuneHistoryStore(tuner.settings.data_dir).get(result.run_id)
        assert saved and saved.identity["model"]["sha256"] is None
    else:
        assert status.startswith("not written:")


def test_scratch_persistence_does_not_touch_real_data(tmp_path: Path, clock: Clock) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "sentinel").write_bytes(b"unchanged")
    tuner, _, driver, _ = _practical(tmp_path / "scratch", _world(clock), 800, clock=clock)
    result = driver.run()
    report = tmp_path / "report.json"
    report.write_text(result.to_json(), encoding="utf-8")
    assert persist_completed(tuner, result, report, origin="simulation") == "recorded"
    assert [(p.name, p.read_bytes()) for p in real.iterdir()] == [("sentinel", b"unchanged")]
    assert all(p.resolve().is_relative_to(tuner.settings.data_dir.resolve()) for p in history_paths(tuner.settings.data_dir))


@pytest.mark.parametrize("unavailable", [False, True])
def test_command_flushes_after_stop_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             unavailable: bool) -> None:
    from tests.test_coordinator import _run_tune
    import inferenceos.tune.learning as learning
    tuner, _, calls, report = _run_tune(tmp_path, monkeypatch, RuntimeError("not invoked"))
    # Build a real successful result before entering the mocked command boundary.
    other, _, driver, _ = _practical(tmp_path / "result", _world(None), 800)
    result = driver.run()
    tuner._learning = other._learning
    tuner.run = lambda: result
    if unavailable:
        TuneHistoryStore(tuner.settings.data_dir).path.mkdir(parents=True)
    original = learning.persist_completed
    def checked(*args: Any, **kwargs: Any) -> str:
        assert calls == ["stop"] and report.is_file()
        return original(*args, **kwargs)
    monkeypatch.setattr(learning, "persist_completed", checked)
    assert tuner._run_tune() == 0
    assert calls == ["stop"] and report.read_text(encoding="utf-8") == result.to_json()
    if not unavailable:
        assert TuneHistoryStore(tuner.settings.data_dir).get(result.run_id) is not None


def test_cache_reuse_records_no_cost_but_fresh_repeat_does(tmp_path: Path, clock: Clock) -> None:
    tuner, _, driver, _ = _practical(tmp_path, _world(clock), 800, clock=clock)
    driver.run()
    workload = driver._short
    assert workload is not None
    key = next(k for k in driver._meas if k[1] == workload.workload_id)
    count = len(tuner._learning.observations)
    cached = driver._meas[key]
    assert driver._measure(key[0], workload, "cached") is cached
    assert len(tuner._learning.observations) == count
    driver._measure(key[0], workload, "fresh", fresh=True)
    assert len(tuner._learning.observations) == count + 1


def test_actual_unresolved_cleanup_never_flushes_history(tmp_path: Path) -> None:
    from tests.test_budgeted_tune import BlockingHarness
    tuner, _, driver, _ = _practical(tmp_path, _world(None), 3600)
    tuner.harness = BlockingHarness(_world(None), block_after=12)
    result = driver.run()
    assert result.outcome.startswith("failed:") and result.saved_setup is None
    assert tuner._learning.blocked_cleanup
    report = tmp_path / "report.json"
    report.write_text(result.to_json(), encoding="utf-8")
    assert persist_completed(tuner, result, report) == "not written: monitor cleanup unresolved"
    assert not TuneHistoryStore(tuner.settings.data_dir).path.exists()


def test_identity_interrupt_keeps_existing_command_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_coordinator import _run_tune
    import inferenceos.tune.learning as learning
    tuner, _, calls, report = _run_tune(tmp_path, monkeypatch, RuntimeError("not invoked"))
    tuner.settings.tune_experimental_budget_s = 800
    error = KeyboardInterrupt("identity interrupted")
    def interrupted(*args: Any) -> Any:
        raise error
    monkeypatch.setattr(learning, "capture_identity", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        tuner._run_tune()
    assert caught.value is error and calls == ["stop"]
    assert not report.exists() and not TuneHistoryStore(tuner.settings.data_dir).path.exists()


def test_malformed_observation_is_diagnostic_not_tune_error() -> None:
    from inferenceos.tune.learning import PracticalHistory
    collector = PracticalHistory()
    collector.comparison({"category": "unknown"})
    collector.observe(1, None, None, None, 1)
    assert len(collector.diagnostics) == 2
    assert not collector.comparisons and not collector.observations


def test_nonfinite_read_with_valid_checksum_is_rejected(tmp_path: Path) -> None:
    store = TuneHistoryStore(tmp_path)
    store.append(record())
    payload = json.dumps({**record().model_dump(mode="json"), "identity": {"bad": float("inf")}})
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE tune_history_runs SET payload=?,checksum=?",
                   (payload, hashlib.sha256(payload.encode()).hexdigest()))
    with pytest.raises(HistoryError, match="invalid"):
        store.get("run-1")


def test_cancelled_run_is_persisted_as_cancelled_not_saved(tmp_path: Path) -> None:
    tuner, _, driver, _ = _practical(tmp_path, _world(None), 800)
    driver.request_cancel("scripted before search")
    result = driver.run()
    assert result.outcome.startswith("cancelled") and result.saved_setup is None
    report = tmp_path / "report.json"
    report.write_text(result.to_json(), encoding="utf-8")
    assert persist_completed(tuner, result, report, origin="simulation") == "recorded"
    saved = TuneHistoryStore(tuner.settings.data_dir).get(result.run_id)
    assert saved and saved.disposition["saved"] is False
    assert saved.disposition["outcome"] == result.outcome and not saved.reusable_for_acceptance


def test_automatic_same_second_run_ids_persist_both_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.25)
    ids = []
    for n in range(2):
        tuner, _, driver, _ = _practical(tmp_path / str(n), _world(None), 800)
        tuner.settings.data_dir = tmp_path / "shared"
        driver.request_cancel(f"separate run {n}")
        result = driver.run()
        report = tmp_path / f"report{n}.json"
        report.write_text(result.to_json(), encoding="utf-8")
        assert persist_completed(tuner, result, report, origin="simulation") == "recorded"
        ids.append(result.run_id)
    assert ids[0] != ids[1]
    store = TuneHistoryStore(tmp_path / "shared")
    for run_id in ids:
        saved = store.get(run_id)
        assert saved is not None and saved.run_id == run_id


def test_explicit_run_id_is_preserved(tmp_path: Path) -> None:
    tuner, _, driver, _ = _practical(tmp_path, _world(None), 800)
    tuner.run_id = "owner-provided-id"
    driver.request_cancel("before search")
    assert driver.run().run_id == "owner-provided-id"


def test_failed_measurement_reason_survives_without_primary_report(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    import inferenceos.tune.budgeted as budgeted
    tuner, _, driver, _ = _practical(tmp_path, _world(None), 800)
    def fail(**kwargs: Any) -> Any:
        raise RuntimeError("unique engine failure 83921")
    monkeypatch.setattr(budgeted, "measure_steady", fail)
    result = driver.run()
    report = tmp_path / "report.json"
    report.write_text(result.to_json(), encoding="utf-8")
    assert persist_completed(tuner, result, report, origin="simulation") == "recorded"
    report.unlink()  # this test's temporary report only
    saved = TuneHistoryStore(tuner.settings.data_dir).get(result.run_id)
    assert saved is not None
    failed = next(o for o in saved.observations if o.status == "failed")
    assert failed.failure == {"type": "RuntimeError", "message": "unique engine failure 83921"}
    assert not saved.disposition["saved"] and not failed.acceptance_evidence


def test_unusable_measurement_keeps_notes_and_actual_watch_controls(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.contracts.experiment import ExperimentMeasurement
    from inferenceos.tune.measure import DisturbanceWatch
    import inferenceos.tune.budgeted as budgeted
    tuner, _, driver, _ = _practical(tmp_path, _world(None), 800)
    # These differ from Settings: prove we record the actual object used, not a
    # reconstruction of what today's defaults would have constructed.
    watch = DisturbanceWatch(logical_cores=2, limit_percent=7.0, max_repeats=5)
    monkeypatch.setattr(driver, "_watch", lambda: watch)
    def unusable(**kwargs: Any) -> ExperimentMeasurement:
        assert kwargs["watch"] is watch
        watch.repeats = 3
        return ExperimentMeasurement(measurement_id="unusable", phase="test", configuration=kwargs["config"],
                                     notes=["not used: still disturbed"], valid_runs=0)
    monkeypatch.setattr(budgeted, "measure_steady", unusable)
    result = driver.run()
    report = tmp_path / "report.json"
    report.write_text(result.to_json(), encoding="utf-8")
    assert persist_completed(tuner, result, report, origin="simulation") == "recorded"
    saved = TuneHistoryStore(tuner.settings.data_dir).get(result.run_id)
    assert saved is not None
    obs = saved.observations[0]
    assert obs.status == "incomplete" and obs.measurement_notes == ("not used: still disturbed",)
    assert obs.retries == 3
    assert obs.request["disturbance"] == {"rule_version": "disturbance-1", "logical_cores": 2,
                                          "limit_percent": 7.0, "max_repeats": 5}
    assert obs.request["runtime_controls"]["watchdog_factor"] == tuner.settings.tune_experimental_stuck_factor


def test_changed_watch_rules_produce_different_request_identities(tmp_path: Path) -> None:
    from inferenceos.tune.learning import request_identity
    from inferenceos.contracts.experiment import RuntimeConfiguration
    tuner, _, driver, _ = _practical(tmp_path, _world(None), 800)
    config = RuntimeConfiguration(context_length=tuner.context_length)
    policy = tuner._path_policy(tuner.workload)
    tuner.settings.tune_max_disturbed_repeats = 0
    tuner.settings.tune_disturbance_cpu_percent = 10
    a = request_identity(config, tuner.workload, policy, watch=driver._watch())
    tuner.settings.tune_max_disturbed_repeats = 5
    tuner.settings.tune_disturbance_cpu_percent = 90
    b = request_identity(config, tuner.workload, policy, watch=driver._watch())
    assert a != b and a["policy"] == b["policy"]
    assert a["format_version"] == b["format_version"] == "tune-request-2"


def test_legacy_request_remains_readable_but_cannot_claim_complete_identity(tmp_path: Path) -> None:
    from inferenceos.contracts.tune_learning import MeasurementObservation
    from inferenceos.tune.learning import request_identity_problems
    legacy = {"configuration": {}, "workload": {"id": "short-short"}, "policy": {}}
    r = record().model_copy(update={"observations": (MeasurementObservation(operation_id=1, kind="screening",
        request=legacy, status="completed", wall_s=10),)})
    store = TuneHistoryStore(tmp_path)
    store.append(r)
    saved = store.get(r.run_id)
    assert saved is not None and saved.observations[0].request == legacy
    assert request_identity_problems(saved.observations[0].request) == ["unsupported or legacy request identity"]
    assert "disturbance" not in saved.observations[0].request


def test_missing_watch_never_claims_complete_measurement_identity(tmp_path: Path) -> None:
    from inferenceos.tune.learning import request_identity, request_identity_problems
    from inferenceos.contracts.experiment import RuntimeConfiguration
    tuner, _, _, _ = _practical(tmp_path, _world(None), 800)
    request = request_identity(RuntimeConfiguration(), tuner.workload, tuner._path_policy(tuner.workload))
    assert request["disturbance"] is None
    problems = request_identity_problems(request)
    assert "disturbance controls" in problems and "runtime controls" in problems
    assert "effective per-answer watchdog limits" in problems
