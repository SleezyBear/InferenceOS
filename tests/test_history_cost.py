"""C4-2: compatible cost priors must change forecasts, never acceptance evidence."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inferenceos.contracts.experiment import ExperimentPolicy, RuntimeConfiguration
from inferenceos.contracts.tune_learning import MeasurementObservation, TuneHistoryRun
from inferenceos.profiling.workloads import get_workload
from inferenceos.tune.history_cost import HistoricalCosts, compatible_request, envelope_key
from inferenceos.tune.learning import request_identity, request_identity_problems
from inferenceos.tune.learning_store import TuneHistoryStore
from inferenceos.tune.measure import DisturbanceWatch
from tests.test_budgeted_tune import Clock, _practical, _world, clock  # noqa: F401

NOW = datetime(2026, 10, 14, tzinfo=timezone.utc)


def runtime() -> dict[str, Any]:
    return {"version": "loaded-runtime-1", "modules": {"server.exe": "a" * 64, "engine.dll": "b" * 64},
            "bundle": {"server.exe": "a" * 64, "engine.dll": "b" * 64},
            "executable": "server.exe", "backend": "vulkan", "device": "test GPU"}


def identity() -> dict[str, Any]:
    return {"model": {"sha256": "c" * 64, "size_bytes": 100}, "context_length": 2048,
            "os": "test Windows", "environment_sha256": "d" * 64, "available_ram_bytes": 4 * 1024**3,
            "machine": {"hostname": "test", "hardware_signature": "hardware", "program_version": "build",
                        "power_plan": {"plan_id": "balanced"}, "power_source": "ac",
                        "graphics_drivers": ["driver"], "free_graphics_memory_bytes": 2 * 1024**3,
                        "total_graphics_memory_bytes": 4 * 1024**3}}


def request() -> dict[str, Any]:
    return request_identity(RuntimeConfiguration(context_length=2048), get_workload("short-short"),
        ExperimentPolicy(), watch=DisturbanceWatch(logical_cores=8, limit_percent=25.0, max_repeats=2),
        runtime_controls={"monitor_scoped": False, "watchdog_factor": 3.0, "watchdog_grace_s": 120.0,
                          "resident_request_timeout_s": 600.0, "resident_readiness_timeout_s": 600.0,
                          "measurement_deadline": None},
        watchdog_limits={"warmup_s": 360.0, "answer_s": 150.0}, runtime_identity=runtime())


def history(n: int = 3, cost: float = 20.0) -> tuple[TuneHistoryRun, ...]:
    return tuple(TuneHistoryRun(run_id=f"prior-{i}", started_at=NOW - timedelta(days=i), origin="simulation",
        identity=identity(), observations=(MeasurementObservation(operation_id=1, kind="screening",
            request=request(), status="completed", wall_s=cost + i, valid_runs=20, retries=0,
            phases={"raw": {"invalid_runs": 0}}),), comparisons=(), disposition={"saved": False},
        source_report_sha256="e" * 64) for i in range(n))


def forecaster(runs: tuple[TuneHistoryRun, ...]) -> HistoricalCosts:
    return HistoricalCosts(runs, identity(), origin="simulation", now=NOW)


def test_prior_requires_independent_runs_and_preserves_precision() -> None:
    assert request_identity_problems(request()) == []
    assert forecaster(history(2)).estimate(request()) is None
    assert forecaster((history()[0],) * 3).estimate(request()) is None
    estimate = forecaster(history(cost=20.12345678901)).estimate(request())
    assert estimate is not None
    assert estimate["seconds"] == 22.12345678901 and estimate["run_count"] == 3
    assert estimate["run_ids"] == ["prior-0", "prior-1", "prior-2"]


@pytest.mark.parametrize("path,value", [
    (("model", "sha256"), "f" * 64), (("context_length",), 4096),
    (("machine", "graphics_drivers"), ["new driver"]), (("machine", "power_source"), "battery"),
    (("machine", "power_plan", "plan_id"), "performance"),
    (("machine", "free_graphics_memory_bytes"), 512 * 1024**2), (("available_ram_bytes",), 1024**3),
    (("machine", "hardware_signature"), None), (("machine", "program_version"), "different engine"),
    (("os",), "different OS"), (("environment_sha256",), "f" * 64),
])
def test_changed_identity_cannot_reuse(path: tuple[str, ...], value: Any) -> None:
    changed = identity()
    part = changed
    for key in path[:-1]:
        part = part[key]
    part[path[-1]] = value
    assert HistoricalCosts(history(), changed, origin="simulation", now=NOW).estimate(request()) is None


@pytest.mark.parametrize("path,value", [
    (("context_length",), None), (("context_length",), True),
    (("environment_sha256",), True), (("environment_sha256",), "z" * 64),
    (("model", "sha256"), "z" * 64), (("model", "size_bytes"), 0),
    (("missing",), ["unreadable power condition"]), (("missing",), "unknown"),
    (("machine", "hostname"), True), (("machine", "hardware_signature"), "unknown"),
    (("machine", "graphics_drivers"), "driver"), (("machine", "graphics_drivers"), ["unknown"]),
    (("machine", "total_graphics_memory_bytes"), True),
    (("machine", "free_graphics_memory_bytes"), 5 * 1024**3),
])
def test_matching_malformed_identities_never_become_compatible(path: tuple[str, ...], value: Any) -> None:
    changed = identity()
    part = changed
    for key in path[:-1]:
        part = part[key]
    part[path[-1]] = value
    req = request()
    req["configuration"]["context_length"] = changed["context_length"]
    runs = tuple(r.model_copy(update={"identity": changed, "observations": (
        r.observations[0].model_copy(update={"request": req}),)}) for r in history())
    assert envelope_key(changed) is None
    assert HistoricalCosts(runs, changed, origin="simulation", now=NOW).estimate(req) is None


def test_only_run_level_engine_unknowns_are_superseded_by_observed_runtime() -> None:
    changed = identity()
    changed["missing"] = ["engine bundle/dependency identity", "effective backend/device identity"]
    assert envelope_key(changed) == envelope_key(identity())
    assert forecaster(history()).estimate(request()) is not None


@pytest.mark.parametrize("field,value", [("backend", "unknown"), ("device", " UNKNOWN "),
                                        ("backend", True), ("device", " ")])
def test_unknown_runtime_is_ineligible_even_when_both_requests_match(
        field: str, value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import inferenceos.tune.history_identity as module
    req = request()
    req["runtime_identity"][field] = value
    assert request_identity_problems(req)
    assert not compatible_request(req, deepcopy(req), 20.0)
    monkeypatch.setattr(module.sys, "platform", "win32")
    device = {"backend": "vulkan", "device": "GPU", field: value}
    reader = module.RuntimeIdentityReader(lambda: SimpleNamespace(pid=1),
        lambda: SimpleNamespace(**device), gpu_count=1)
    assert reader() is None  # must reject before touching a process


@pytest.mark.parametrize("path,value", [
    (("configuration", "threads"), 4), (("configuration", "gpu_offload_layers"), 1),
    (("workload", "sha256"), "f" * 64), (("policy", "measured_repetitions"), 999),
    (("disturbance", "max_repeats"), 0), (("runtime_controls", "monitor_scoped"), True),
    (("runtime_identity", "modules", "engine.dll"), "f" * 64),
    (("runtime_identity", "bundle", "engine.dll"), "f" * 64),
    (("runtime_identity", "device"), "another device"), (("runtime_identity", "backend"), "cpu"),
    (("runtime_controls", "resident_request_timeout_s"), 900.0),
    (("runtime_controls", "watchdog_factor"), 4.0),
    (("format_version",), "tune-request-2"), (("missing",), ["unknown"]),
])
def test_changed_request_cannot_reuse(path: tuple[str, ...], value: Any) -> None:
    changed = request()
    part = changed
    for key in path[:-1]:
        part = part[key]
    part[path[-1]] = value
    assert forecaster(history()).estimate(changed) is None


def test_known_nonbinding_watchdog_limits_are_compatible_but_missing_or_binding_are_not() -> None:
    old, new = request(), request()
    new["watchdog_limits"]["answer_s"] = 100.123456789
    assert compatible_request(old, new, 99)
    assert not compatible_request(old, new, 101)
    new.pop("watchdog_limits")
    assert not compatible_request(old, new, 1)


@pytest.mark.parametrize("changes", [
    {"started_at": NOW - timedelta(days=31)}, {"started_at": NOW + timedelta(seconds=1)},
    {"started_at": None}, {"origin": "hardware"}, {"origin": "unknown"},
])
def test_stale_future_unknown_and_simulated_records_are_separate(changes: dict[str, Any]) -> None:
    assert forecaster(tuple(r.model_copy(update=changes) for r in history())).estimate(request()) is None


@pytest.mark.parametrize("changes", [{"status": "failed"}, {"status": "incomplete"},
    {"kind": "final-side"}, {"valid_runs": 0}, {"retries": 1}, {"retries": None},
    {"phases": {}}, {"phases": {"raw": None}}, {"phases": {"raw": {"invalid_runs": 1}}},
    {"failure": {"type": "Error", "message": "failed"}}, {"wall_s": 0.0}])
def test_bad_measurements_do_not_teach_cheap_costs(changes: dict[str, Any]) -> None:
    runs = tuple(r.model_copy(update={"observations": (r.observations[0].model_copy(update=changes),)}) for r in history())
    assert forecaster(runs).estimate(request()) is None


def test_many_samples_in_one_run_do_not_dominate_and_recent_window_is_bounded() -> None:
    runs = list(history(25))
    runs[0] = runs[0].model_copy(update={"observations": tuple(runs[0].observations[0].model_copy(
        update={"operation_id": i + 1}) for i in range(100))})
    estimate = forecaster(tuple(reversed(runs))).estimate(request())
    assert estimate is not None
    assert estimate["run_count"] == 20 and estimate["seconds"] == 34


def test_store_read_only_window_and_corruption(tmp_path: Path) -> None:
    store = TuneHistoryStore(tmp_path / "store")
    assert store.recent() == () and not store.path.exists()
    for r in history():
        store.append(r)
    assert [r.run_id for r in store.recent(limit=2)] == ["prior-2", "prior-1"]
    with pytest.raises(ValueError):
        store.recent(limit=201)


def wired_run(tmp_path: Path, clock: Clock, records: tuple[TuneHistoryRun, ...] = (), *,
              target: float = 1600, **world: Any) -> tuple[Any, Any, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tuner, _, driver, _ = _practical(tmp_path, _world(clock, **world), target, clock=clock)
    tuner.history_identity = identity()
    tuner.history_origin = "simulation"
    tuner.history_records = records
    tuner.history_runtime_reader = runtime
    tuner.history_runtime_valid = lambda observed: True
    return tuner, driver, driver.run()


def test_live_recording_has_actual_watchdog_limits_and_fresh_cost_wins(tmp_path: Path, clock: Clock) -> None:
    tuner, driver, result = wired_run(tmp_path, clock, answer_s=5)
    observations = [o for o in tuner._learning.observations if o.kind == "screening"]
    assert observations and all(not request_identity_problems(o.request) for o in observations)
    assert observations[0].request["watchdog_limits"]["answer_s"] == 150.0
    config = driver._config(driver._start_setup)
    source = next(o for o in observations if o.request["configuration"] == config.model_dump(mode="json")
                  and o.request["policy"] == tuner._path_policy(driver._short).model_dump(mode="json"))
    records = tuple(r.model_copy(update={"observations": (source.model_copy(update={"wall_s": 100.0}),)}) for r in history())
    driver._historical_costs = forecaster(records)
    before = dict(driver._meas)
    cost = driver._request_cost(driver._start_setup, "short")
    assert cost == source.wall_s
    assert driver._forecast_events[-1]["source"] == "current request"
    assert driver._meas == before


def test_runtime_inspection_failure_is_a_fallback_not_a_failed_tune(tmp_path: Path, clock: Clock) -> None:
    tuner, _, driver, _ = _practical(tmp_path, _world(clock), 800, clock=clock)
    def unavailable() -> Any:
        raise OSError("cannot inspect")
    tuner.history_runtime_reader = unavailable
    result = driver.run()
    assert result.outcome == "saved"
    assert any("runtime identity unavailable" in d for d in tuner._learning.diagnostics)


def test_loaded_dependency_and_optional_bundle_changes_invalidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inferenceos.tune.history_identity as module
    server, library, optional = (tmp_path / n for n in ("server.exe", "engine.dll", "optional.dll"))
    for p in (server, library, optional):
        p.write_bytes(p.name.encode())
    process = SimpleNamespace(exe=lambda: str(server),
        memory_maps=lambda: [SimpleNamespace(path=str(p)) for p in (server, library)])
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module, "identity_matches", lambda target: True)
    monkeypatch.setattr(module.psutil, "Process", lambda pid: process)
    reader = module.RuntimeIdentityReader(lambda: SimpleNamespace(pid=123),
        lambda: SimpleNamespace(backend="vulkan", device="GPU"), gpu_count=1)
    original = reader()
    assert original is not None and str(optional.resolve()) in original["bundle"]
    assert str(optional.resolve()) not in original["modules"]
    assert reader.unchanged()
    optional.write_bytes(b"changed engine dependency")
    assert not reader.unchanged()
    assert reader() != original
    assert reader.unchanged()
    assert not reader.unchanged(original)  # old setup's cached identity must not revive
    (tmp_path / "new.dll").write_bytes(b"new optional code")
    assert not reader.unchanged()
    monkeypatch.setattr(module, "identity_matches", lambda target: False)
    assert reader() is None


def test_runtime_identity_timeout_disables_inspection_for_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inferenceos.tune.history_identity as module
    server, lib = tmp_path / "server.exe", tmp_path / "engine.dll"
    server.write_bytes(b"server")
    lib.write_bytes(b"lib")
    process = SimpleNamespace(exe=lambda: str(server),
        memory_maps=lambda: [SimpleNamespace(path=str(p)) for p in (server, lib)])
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module, "identity_matches", lambda target: True)
    monkeypatch.setattr(module.psutil, "Process", lambda pid: process)
    ticks = iter([0.0, 3.0])
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    reader = module.RuntimeIdentityReader(lambda: SimpleNamespace(pid=1),
        lambda: SimpleNamespace(backend="vulkan", device="GPU"), gpu_count=1)
    with pytest.raises(OSError, match="allowance"):
        reader()
    assert reader() is None and not reader.unchanged()


def test_request_completeness_is_validated_not_trusted() -> None:
    for field in ("configuration", "policy", "runtime_controls", "workload", "runtime_identity", "watchdog_limits"):
        for value in (None, {}, {"unexpected": 1}):
            changed = request()
            changed[field] = value
            assert request_identity_problems(changed)
            assert forecaster(history()).estimate(changed) is None


def test_no_records_or_unobserved_setup_keeps_existing_forecast(tmp_path: Path, clock: Clock) -> None:
    tuner, driver, _ = wired_run(tmp_path, clock)
    expected = driver._cost("short")
    assert driver._request_cost(driver._start_setup, "short") == expected
    driver._historical_costs = forecaster(history())
    driver._runtime_identities.clear()
    assert driver._request_cost(driver._start_setup, "short") == expected
    assert not driver._forecast_events


def test_model_replacement_disables_priors(tmp_path: Path) -> None:
    from inferenceos.tune.learning import model_identity_unchanged
    model = tmp_path / "model.gguf"
    model.write_bytes(b"first")
    stat = model.stat()
    captured = {"model": {"path": str(model), "size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns, "sha256": "a" * 64}}
    assert model_identity_unchanged(captured)
    model.write_bytes(b"different content")
    assert not model_identity_unchanged(captured)


def test_command_reads_only_configured_store_and_keeps_lookup_failure_nonfatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_coordinator import _run_tune
    import inferenceos.tune.learning as learning
    tuner, _, calls, report = _run_tune(tmp_path, monkeypatch, RuntimeError("not invoked"))
    other, _, driver, _ = _practical(tmp_path / "result", _world(None), 800)
    result = driver.run()
    tuner._learning = other._learning
    tuner.settings.tune_experimental_budget_s = 800
    monkeypatch.setattr(learning, "capture_identity", lambda *args: identity())
    looked_up = []
    def unavailable(store: Any) -> Any:
        looked_up.append(store.path)
        raise OSError("unavailable")
    monkeypatch.setattr(TuneHistoryStore, "recent", unavailable)
    def run() -> Any:
        assert tuner.history_origin == "hardware"
        assert "unavailable" in tuner.history_diagnostic
        return result
    tuner.run = run
    assert tuner._run_tune() == 0
    assert looked_up == [tuner.settings.data_dir / "tune_history.sqlite3"]
    assert calls == ["stop"] and report.is_file()


def test_complete_tunes_use_persisted_priors_for_real_forecasts(tmp_path: Path) -> None:
    from scripts.evaluate_history_cost import evaluate
    result = evaluate(tmp_path / "paired.json")
    assert result["pairs"] == 15 and result["training_tunes"] == 9
    assert result["forecast_count_with_observed_cost"] > 0
    assert result["history_mean_absolute_error_s"] < result["fallback_mean_absolute_error_s"]
    assert result["target_overruns"] == {"control": 0, "history": 0}
    assert result["saved_outcome_changes"] == 0
    for pair in result["cases"]:
        assert pair["control"]["confirmation_count"] == pair["history"]["confirmation_count"]
    assert (tmp_path / "paired.evidence.json.gz").is_file()


@pytest.mark.parametrize("mode", ["regression", "stuck", "cancel"])
def test_history_cannot_make_bad_final_check_save(tmp_path: Path, clock: Clock,
                                                monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    from inferenceos.tune.pipeline import Tuner
    from scripts.evaluate_history_cost import run_case

    records = []
    for n in range(3):
        t, d, r = run_case(tmp_path / f"train-{n}", clock, (), 15000, "different-costs")
        records.append(history()[0].model_copy(update={"run_id": r.run_id,
            "started_at": datetime.now(timezone.utc), "observations": tuple(t._learning.observations)}))
    # Inject the failure AFTER actual history-enabled exploration, at the existing
    # final-check owner. This does not manufacture a failed report or saved outcome.
    original = Tuner._check_whole_setup
    def final(tuner: Any, *args: Any, **kwargs: Any) -> Any:
        if mode == "cancel":
            driver = tuner.harness._driver
            driver.request_cancel("test stop with history")
        elif mode == "stuck":
            tuner.harness._inner._responder = _world(clock, hang=lambda *a: 10000)
        else:
            tuner.harness._inner._responder = _world(clock, flat=True)
        return original(tuner, *args, **kwargs)
    monkeypatch.setattr(Tuner, "_check_whole_setup", final)
    t, d, result = run_case(tmp_path / "test", clock, tuple(records), 6000, "different-costs")
    assert any(e["source"] == "history" for e in d._forecast_events)
    assert result.saved_setup is None
    assert sum(r.get("round") == "final check" for r in result.rounds) <= 1
