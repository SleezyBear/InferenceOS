"""Regressions for the 15 Oct whole-project review (R1-R5).

Each test fails on the reviewed commit (b7d1e23) and passes after the fix. Temporary
stores and fake engines only; no model or server is run.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.tune import SavedSetup
from inferenceos.selftest import sample_model
from inferenceos.tune.profiles import SetupStore, StoreUnreadable
from inferenceos.tune.start import serve_until_stopped
from tests.test_start import FakeBackend, _tuned


# --------------------------------------------------------------------------- #
# R1: a save never publishes a replacement built from an incomplete read       #
# --------------------------------------------------------------------------- #
def _in_slot(setup: SavedSetup, setup_id: str, factor: int) -> SavedSetup:
    context = setup.context_length * factor
    return setup.model_copy(update={
        "setup_id": setup_id,
        "context_length": context,
        "configuration": setup.configuration.model_copy(update={"context_length": context}),
    })


def _two_setups(tmp_path: Path) -> tuple[SetupStore, SavedSetup]:
    _, tuned, store = _tuned(tmp_path)
    assert tuned.saved_setup is not None
    store.save(_in_slot(tuned.saved_setup, "second", 2))
    assert len(store.all()) == 2
    return store, tuned.saved_setup


def test_r1_a_read_failure_refuses_the_save_and_keeps_the_file(tmp_path: Path) -> None:
    store, first = _two_setups(tmp_path)
    before = store.path.read_bytes()
    original = Path.read_text

    def denied(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == store.path:
            raise PermissionError("injected transient read denial")
        return original(path, *args, **kwargs)

    with patch.object(Path, "read_text", denied), pytest.raises(StoreUnreadable):
        store.save(_in_slot(first, "third", 4))
    assert store.path.read_bytes() == before
    assert sorted(s.setup_id for s in store.all()) == sorted([first.setup_id, "second"])
    assert not list(store.path.parent.glob(store.path.name + ".*.tmp"))


@pytest.mark.parametrize("text", ['{"setups": [', '["not", "a", "store"]', '{"setups": "x"}'])
def test_r1_a_damaged_file_refuses_the_save_and_keeps_its_bytes(tmp_path: Path, text: str) -> None:
    store, first = _two_setups(tmp_path)
    store.path.write_text(text, encoding="utf-8")
    with pytest.raises(StoreUnreadable):
        store.save(_in_slot(first, "third", 4))
    assert store.path.read_text(encoding="utf-8") == text


def test_r1_unreadable_entries_are_carried_over_unchanged(tmp_path: Path) -> None:
    store, first = _two_setups(tmp_path)
    document = json.loads(store.path.read_text(encoding="utf-8"))
    odd = {"written_by": "a newer version", "setup_id": "future"}
    document["setups"].insert(1, odd)
    document["note"] = "kept"
    store.path.write_text(json.dumps(document), encoding="utf-8")

    store.save(_in_slot(first, "third", 4))

    after = json.loads(store.path.read_text(encoding="utf-8"))
    assert after["note"] == "kept"
    assert after["setups"][:3] == document["setups"]  # every earlier entry, byte-for-value
    assert sorted(s.setup_id for s in store.all()) == sorted([first.setup_id, "second", "third"])


def test_r1_the_same_slot_is_still_replaced_and_a_missing_file_is_empty(tmp_path: Path) -> None:
    store, first = _two_setups(tmp_path)
    store.save(first.model_copy(update={"setup_id": "replacement"}))
    assert sorted(s.setup_id for s in store.all()) == ["replacement", "second"]
    fresh = SetupStore(Settings(), path=tmp_path / "new" / "saved.json")
    fresh.save(first)
    assert [s.setup_id for s in fresh.all()] == [first.setup_id]


def test_r1_a_tune_that_cannot_read_the_store_saves_nothing(tmp_path: Path) -> None:
    from tests.test_tune import _tuner, layer_gain_responder

    tuner, _, _, store = _tuner(tmp_path, layer_gain_responder, answers=[True])
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text('{"setups": [', encoding="utf-8")
    with pytest.raises(StoreUnreadable):
        tuner.run()
    assert store.path.read_text(encoding="utf-8") == '{"setups": ['


def test_r1_the_tune_command_reports_the_refused_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_coordinator import _run_tune

    tuner, _, engine_calls, out = _run_tune(tmp_path, monkeypatch, StoreUnreadable("store unreadable"))
    code = tuner._run_tune()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and report["outcome"].startswith("failed: store unreadable")
    assert report["error"]["type"] == "StoreUnreadable" and report["saved_setup"] is None
    assert engine_calls.count("stop") >= 1


# --------------------------------------------------------------------------- #
# R2: start owns the engine for its whole lifetime and reports cleanup truly   #
# --------------------------------------------------------------------------- #
def _setup(tmp_path: Path) -> SavedSetup:
    _, tuned, _ = _tuned(tmp_path)
    assert tuned.saved_setup is not None
    return tuned.saved_setup


def test_r2_a_failure_in_the_startup_report_still_stops_the_engine(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    backend = FakeBackend()
    error = BrokenPipeError("stdout consumer closed")

    def broken(_line: str) -> None:
        raise error

    with pytest.raises(BrokenPipeError) as caught:
        serve_until_stopped(setup=setup, model=sample_model(), backend=backend, say=broken,
                            keep_running=lambda: False)
    assert caught.value is error  # the original error, unchanged
    assert backend.stops == 1 and not backend.running


def test_r2_the_original_error_wins_when_the_stop_also_fails(tmp_path: Path) -> None:
    setup = _setup(tmp_path)

    class StopFails(FakeBackend):
        def stop(self) -> object:
            self.stops += 1
            raise RuntimeError("stop failed too")

    backend = StopFails()
    error = BrokenPipeError("stdout consumer closed")

    def broken(_line: str) -> None:
        raise error

    with pytest.raises(BrokenPipeError) as caught:
        serve_until_stopped(setup=setup, model=sample_model(), backend=backend, say=broken)
    assert caught.value is error and backend.stops == 1


def test_r2_an_incomplete_stop_is_not_reported_clean(tmp_path: Path) -> None:
    setup = _setup(tmp_path)

    class IncompleteStop(FakeBackend):
        def stop(self) -> object:
            self.stops += 1
            return SimpleNamespace(cleanup_complete=False, exited=False)

    backend = IncompleteStop()
    outcome = serve_until_stopped(setup=setup, model=sample_model(), backend=backend,
                                  say=lambda _line: None, keep_running=lambda: False)
    assert outcome.started and not outcome.stopped_cleanly and backend.stops == 1


def test_r2_a_complete_stop_is_reported_clean(tmp_path: Path) -> None:
    setup = _setup(tmp_path)

    class CompleteStop(FakeBackend):
        def stop(self) -> object:
            super().stop()
            return SimpleNamespace(cleanup_complete=True, exited=True)

    outcome = serve_until_stopped(setup=setup, model=sample_model(), backend=CompleteStop(),
                                  say=lambda _line: None, keep_running=lambda: False)
    assert outcome.stopped_cleanly


def test_r2_an_engine_that_died_is_still_released(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    key = tmp_path / "dead-engine-key.txt"
    key.write_text("synthetic credential", encoding="utf-8")

    class Dead(FakeBackend):
        @property
        def api_key_file(self) -> Path | None:
            return key

        def stop(self) -> object:
            key.unlink(missing_ok=True)
            return super().stop()

    backend = Dead(dies_after=0)
    outcome = serve_until_stopped(setup=setup, model=sample_model(), backend=backend,
                                  say=lambda _line: None)
    assert outcome.reason == "the model stopped on its own"
    assert backend.stops == 1 and not key.exists() and outcome.stopped_cleanly


def test_r2_a_failing_stop_falls_back_to_cleanup_and_is_not_clean(tmp_path: Path) -> None:
    setup = _setup(tmp_path)

    class StopFails(FakeBackend):
        cleaned = 0

        def stop(self) -> object:
            self.stops += 1
            raise RuntimeError("no resident server to unload")

        def cleanup(self) -> None:
            self.cleaned += 1
            self.running = False

    backend = StopFails()
    outcome = serve_until_stopped(setup=setup, model=sample_model(), backend=backend,
                                  say=lambda _line: None, keep_running=lambda: False)
    assert not outcome.stopped_cleanly and backend.cleaned == 1


def test_r2_ctrl_c_during_the_startup_report_is_a_normal_stop(tmp_path: Path) -> None:
    setup = _setup(tmp_path)
    backend = FakeBackend()

    def interrupted(_line: str) -> None:
        raise KeyboardInterrupt

    outcome = serve_until_stopped(setup=setup, model=sample_model(), backend=backend, say=interrupted)
    assert outcome.reason == "stopped by you" and outcome.stopped_cleanly and backend.stops == 1


# --------------------------------------------------------------------------- #
# R3: serve --deployment verifies against what the engine showed               #
# --------------------------------------------------------------------------- #
def _serve(tmp_path: Path, requested: RuntimeConfiguration, telemetry: Any) -> Any:
    from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
    from inferenceos.deployment import serve_deployment
    from tests.resident_helpers import FakeResidentBackend
    from tests.test_deployment_state import _profile, _settings, _state

    settings, profile = _settings(tmp_path), _profile(tmp_path)
    backend = FakeResidentBackend()
    backend.telemetry = lambda: telemetry  # type: ignore[method-assign]
    adapter = LlamaCppResidentAdapter(settings=settings, backend=backend)
    with patch("inferenceos.deployment._safe_model_profile", return_value=profile), \
            patch("inferenceos.deployment.cli_version", return_value="fake-version"), \
            patch("inferenceos.deployment.process_control.process_exists", return_value=False):
        return serve_deployment(_state(profile, final_configuration=requested), settings=settings,
                                prompt="hi", adapter=adapter)


def test_r3_an_engine_showing_other_layers_is_a_mismatch(tmp_path: Path) -> None:
    from inferenceos.interfaces.backend_adapter import BackendTelemetry

    evidence = _serve(tmp_path, RuntimeConfiguration(context_length=1024, gpu_offload_layers=2),
                      BackendTelemetry(gpu_layers_offloaded=4, gpu_layers_total=29))
    assert evidence.configuration_matches is False and not evidence.served
    assert evidence.configuration_observed.gpu_offload_layers == 4
    assert any("gpu_offload_layers" in item for item in evidence.configuration_contradictions)


def test_r3_shown_settings_match_and_the_rest_stay_unconfirmed(tmp_path: Path) -> None:
    from inferenceos.interfaces.backend_adapter import BackendTelemetry

    evidence = _serve(tmp_path, RuntimeConfiguration(context_length=1024, gpu_offload_layers=28, threads=6),
                      BackendTelemetry(gpu_layers_offloaded=28, gpu_layers_total=29, effective_threads=6))
    assert evidence.served and evidence.configuration_matches is None
    assert evidence.configuration_unverified == ["context_length"]
    observed = evidence.configuration_observed
    assert observed.gpu_offload_layers == 28 and observed.threads == 6
    assert observed.context_length is None  # never the request echoed back


def test_r3_no_engine_evidence_is_unconfirmed_not_a_match(tmp_path: Path) -> None:
    from inferenceos.interfaces.backend_adapter import BackendTelemetry

    evidence = _serve(tmp_path, RuntimeConfiguration(context_length=1024, gpu_offload_layers=2),
                      BackendTelemetry())
    assert evidence.configuration_matches is None
    assert sorted(evidence.configuration_unverified) == ["context_length", "gpu_offload_layers"]
    assert evidence.configuration_observed.gpu_offload_layers is None


def test_r3_everything_shown_is_a_full_match() -> None:
    from inferenceos.deployment import observed_configuration
    from inferenceos.interfaces.backend_adapter import BackendTelemetry

    requested = RuntimeConfiguration(gpu_offload_layers=4, threads=6, use_mmap=False, flash_attention=True,
                                     kv_cache_type_k="q8_0", kv_cache_type_v="q8_0")
    telemetry = BackendTelemetry(gpu_layers_offloaded=4, gpu_layers_total=29, effective_threads=6,
                                 mmap_effective=False,
                                 observed={"flash_attention": True, "cache_type_k": "q8_0", "cache_type_v": "q8_0"})
    observed, contradictions, unverified = observed_configuration(requested, telemetry)
    assert contradictions == [] and unverified == []
    assert observed.kv_cache_type_k == "q8_0" and observed.flash_attention is True
    telemetry.observed["cache_type_v"] = "f16"
    _, contradictions, _ = observed_configuration(requested, telemetry)
    assert len(contradictions) == 1 and contradictions[0].startswith("kv_cache_type_v (values)")
    # The engine caps a larger request at the model's layer count: not a mismatch.
    _, contradictions, _ = observed_configuration(
        RuntimeConfiguration(gpu_offload_layers=99), BackendTelemetry(gpu_layers_offloaded=29, gpu_layers_total=29))
    assert contradictions == []


# --------------------------------------------------------------------------- #
# R4: generated report names never collide; publication is atomic              #
# --------------------------------------------------------------------------- #
def test_r4_two_runs_in_the_same_second_keep_both_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inferenceos.tune.commands as commands
    from tests.test_coordinator import _run_tune

    _, tuned, _ = _tuned(tmp_path / "tuned")

    class FixedDate(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> "FixedDate":
            return cls(2026, 10, 15, 6, 0, 0, tzinfo=timezone.utc)

    tuner, _, _, _ = _run_tune(tmp_path / "commands", monkeypatch, RuntimeError("unused"))
    monkeypatch.setattr(commands, "datetime", FixedDate)
    monkeypatch.setattr(commands, "read_server_capabilities", lambda *a, **kw: None)
    with contextlib.redirect_stdout(io.StringIO()):
        for index in (1, 2):
            result = tuned.model_copy(update={"run_id": f"distinct-run-{index}"})
            tuner.run = lambda result=result: result
            assert commands.run_tune(tuner.settings, model_path=tuner.model_path,
                                     context_length=tuned.context_length, approve_all=True) == 0
    files = sorted(tuner.settings.reports_dir.glob("tune_*.json"))
    assert len(files) == 2
    assert all(f.name.startswith("tune_20261015T060000Z_") for f in files)
    assert sorted(json.loads(f.read_text(encoding="utf-8"))["run_id"] for f in files) == [
        "distinct-run-1", "distinct-run-2"]
    assert not list(tuner.settings.reports_dir.glob("*.tmp"))


def test_r4_an_explicit_out_path_is_still_replaced(tmp_path: Path) -> None:
    from inferenceos.tune.commands import publish_report

    out = tmp_path / "chosen.json"
    publish_report(out, '{"a": 1}')
    publish_report(out, '{"a": 2}')
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["chosen.json"]


# --------------------------------------------------------------------------- #
# R5: the stale-key sweep never removes a live owner's key                     #
# --------------------------------------------------------------------------- #
def _aged(path: Path, hours: float, now: float) -> Path:
    path.write_text("synthetic-key", encoding="utf-8")
    old = now - hours * 3600
    os.utime(path, (old, old))
    return path


def test_r5_an_old_key_of_a_running_owner_is_kept(tmp_path: Path) -> None:
    from inferenceos.backends.resident_server import key_file_prefix, sweep_stale_key_files

    now = time.time()
    prefix = key_file_prefix()
    assert prefix.startswith(f"inferenceos-key-o{os.getpid()}x")
    active = _aged(tmp_path / f"{prefix}active.txt", 25, now)
    assert sweep_stale_key_files(temp_dir=tmp_path, now=now) == 0
    assert active.exists()


def test_r5_abandoned_keys_are_still_removed(tmp_path: Path) -> None:
    from inferenceos.backends import resident_server as rs

    now = time.time()
    start_ms = rs._process_start_ms(os.getpid())
    assert start_ms is not None
    reused = _aged(tmp_path / f"inferenceos-key-o{os.getpid()}x{start_ms - 60_000}-reused.txt", 25, now)
    legacy = _aged(tmp_path / "inferenceos-key-legacy.txt", 25, now)
    fresh_legacy = _aged(tmp_path / "inferenceos-key-fresh.txt", 1, now)
    with patch.object(rs, "_process_start_ms", side_effect=lambda pid: None if pid == 999_999_999 else start_ms):
        gone = _aged(tmp_path / f"inferenceos-key-o999999999x{start_ms}-gone.txt", 25, now)
        assert rs.sweep_stale_key_files(temp_dir=tmp_path, now=now) == 3
    assert not reused.exists() and not legacy.exists() and not gone.exists()
    assert fresh_legacy.exists()


def test_r5_an_owner_that_cannot_be_checked_keeps_its_key(tmp_path: Path) -> None:
    from inferenceos.backends import resident_server as rs

    now = time.time()
    key = _aged(tmp_path / "inferenceos-key-o1234x5678-unknown.txt", 25, now)
    with patch.object(rs, "_process_start_ms", side_effect=PermissionError("access denied")):
        assert rs.sweep_stale_key_files(temp_dir=tmp_path, now=now) == 0
    assert key.exists()


# --------------------------------------------------------------------------- #
# Second round (reviewer's Windows review of the fixes, 15 Oct)                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", ["0.12345678901234567890123456789", "1e400",
                                   "1e9999999999999999999", "1e-9999999999999999999"])
def test_r1b_a_value_that_would_change_refuses_the_save(tmp_path: Path, value: str) -> None:
    _, tuned, store = _tuned(tmp_path)
    assert tuned.saved_setup is not None
    text = '{"setups": [{"future": ' + value + '}]}'
    store.path.write_text(text, encoding="utf-8")
    with pytest.raises(StoreUnreadable, match="cannot be carried over unchanged"):
        store.save(tuned.saved_setup)
    assert store.path.read_text(encoding="utf-8") == text


def test_r1b_a_repeated_name_refuses_the_save(tmp_path: Path) -> None:
    _, tuned, store = _tuned(tmp_path)
    assert tuned.saved_setup is not None
    text = '{"setups": [{"future": 1, "future": 2}]}'
    store.path.write_text(text, encoding="utf-8")
    with pytest.raises(StoreUnreadable, match="appear twice"):
        store.save(tuned.saved_setup)
    assert store.path.read_text(encoding="utf-8") == text


def test_r1b_ordinary_numbers_in_unknown_entries_are_kept_exactly(tmp_path: Path) -> None:
    from decimal import Decimal

    _, tuned, store = _tuned(tmp_path)
    assert tuned.saved_setup is not None
    text = '{"setups": [{"a": 0.1, "b": 1e5, "c": -0.0, "d": 123456789012345678901234567890, "e": 2.5e-308}]}'
    store.path.write_text(text, encoding="utf-8")
    store.save(tuned.saved_setup)
    before = json.loads(text, parse_float=Decimal)["setups"][0]
    after = json.loads(store.path.read_text(encoding="utf-8"), parse_float=Decimal)["setups"][0]
    assert after == before


def test_r1b_a_held_store_lock_saves_nothing_and_keeps_the_file(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    import functools

    from inferenceos.tune import profiles
    from inferenceos.tune.profiles import StoreBusy

    store, first = _two_setups(tmp_path)
    before = store.path.read_bytes()
    monkeypatch.setattr(profiles, "_StoreLock", functools.partial(profiles._StoreLock, timeout_s=0.2))
    with profiles._StoreLock(store.path.with_name(store.path.name + ".lock")):
        with pytest.raises(StoreBusy):
            store.save(_in_slot(first, "third", 4))
    assert store.path.read_bytes() == before


def test_r1b_the_tune_command_reports_a_held_store_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.tune.profiles import StoreBusy
    from tests.test_coordinator import _run_tune

    tuner, store, engine_calls, out = _run_tune(tmp_path, monkeypatch, StoreBusy("injected lock timeout"))
    with contextlib.redirect_stdout(io.StringIO()):
        code = tuner._run_tune()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and report["outcome"].startswith("failed: injected lock timeout")
    assert report["error"]["type"] == "StoreBusy" and report["saved_setup"] is None
    assert engine_calls.count("stop") >= 1


def _dead_real_backend(key: Path) -> Any:
    from inferenceos.backends import resident_server as rs
    from inferenceos.backends.llama_server_toolchain import LlamaServerProcess

    process = LlamaServerProcess(["not-executed"], guard=SimpleNamespace(close=lambda: None))  # type: ignore[arg-type]
    process._proc = SimpleNamespace(pid=999_999_999, poll=lambda: 0, returncode=0)  # type: ignore[assignment]
    backend = rs.LlamaServerResidentBackend(server=Path("not-executed.exe"))
    backend._process, backend._state, backend._key_file = process, "ready", key
    return backend


def test_r2b_a_password_file_that_cannot_be_deleted_is_not_clean_and_is_retried(tmp_path: Path) -> None:
    from inferenceos.backends import resident_server as rs
    from inferenceos.contracts.lifecycle import ResourceSnapshot

    setup = _setup(tmp_path)
    key = tmp_path / "undeletable-key"
    key.mkdir()  # deleting a folder as a file fails on every platform, like a locked file
    backend = _dead_real_backend(key)
    with patch.object(backend, "start", return_value=None), \
            patch.object(rs, "resource_snapshot", return_value=ResourceSnapshot(exists=False)), \
            patch.object(rs, "wait_process_exit", return_value=True):
        outcome = serve_until_stopped(setup=setup, model=sample_model(), backend=backend, say=lambda _line: None)
    assert not outcome.stopped_cleanly
    assert outcome.cleanup_problem is not None and "password file could not be deleted" in outcome.cleanup_problem
    assert str(key) in outcome.cleanup_problem
    assert backend.last_unload.credential_removed is False
    assert backend.undeleted_key_files == [key]  # kept on record, not forgotten
    key.rmdir()  # the obstacle goes away: the next cleanup removes nothing more and clears the record
    assert backend._cleanup_key_file() is True and backend.undeleted_key_files == []


def test_r2b_an_ordinary_dead_engine_with_a_deletable_key_is_clean(tmp_path: Path) -> None:
    from inferenceos.backends import resident_server as rs
    from inferenceos.contracts.lifecycle import ResourceSnapshot

    setup = _setup(tmp_path)
    key = tmp_path / "key.txt"
    key.write_text("synthetic credential", encoding="utf-8")
    backend = _dead_real_backend(key)
    with patch.object(backend, "start", return_value=None), \
            patch.object(rs, "resource_snapshot", return_value=ResourceSnapshot(exists=False)), \
            patch.object(rs, "wait_process_exit", return_value=True):
        outcome = serve_until_stopped(setup=setup, model=sample_model(), backend=backend, say=lambda _line: None)
    assert outcome.stopped_cleanly and outcome.cleanup_problem is None and not key.exists()
    assert backend.last_unload.credential_removed is True


def test_r2b_an_error_reading_the_address_still_stops_the_engine(tmp_path: Path) -> None:
    setup = _setup(tmp_path)

    class NoAddress(FakeBackend):
        @property
        def base_url(self) -> str | None:
            raise RuntimeError("address unavailable")

    backend = NoAddress()
    with pytest.raises(RuntimeError, match="address unavailable"):
        serve_until_stopped(setup=setup, model=sample_model(), backend=backend, say=lambda _line: None)
    assert backend.stops == 1


def test_r3b_a_one_sided_request_does_not_require_the_other_side(tmp_path: Path) -> None:
    from inferenceos.interfaces.backend_adapter import BackendTelemetry

    evidence = _serve(tmp_path, RuntimeConfiguration(context_length=1024, kv_cache_type_k="q8_0"),
                      BackendTelemetry(observed={"cache_type_k": "q8_0", "cache_type_v": "q8_0"}))
    assert evidence.served and evidence.configuration_contradictions == []
    assert evidence.configuration_unverified == ["context_length"]
    assert evidence.configuration_observed.kv_cache_type_k == "q8_0"
    assert evidence.configuration_observed.kv_cache_type_v == "q8_0"  # shown, though not requested


def test_r3b_a_shown_mismatch_wins_over_a_missing_side(tmp_path: Path) -> None:
    from inferenceos.interfaces.backend_adapter import BackendTelemetry

    evidence = _serve(tmp_path, RuntimeConfiguration(context_length=1024, kv_cache_type="q8_0"),
                      BackendTelemetry(observed={"cache_type_k": "f16"}))
    assert not evidence.served and evidence.configuration_matches is False
    assert len(evidence.configuration_contradictions) == 1
    assert "keys" in evidence.configuration_contradictions[0]
    assert evidence.configuration_observed.kv_cache_type_k == "f16"
    assert "kv_cache_type" in evidence.configuration_unverified  # the values side stays unknown


def test_r5b_a_new_key_with_an_unreadable_owner_start_is_still_protected(tmp_path: Path) -> None:
    from inferenceos.backends import resident_server as rs

    with patch.object(rs, "_process_start_ms", side_effect=PermissionError("injected owner query failure")):
        prefix = rs.key_file_prefix()
    assert prefix == f"inferenceos-key-o{os.getpid()}xu-"
    now = time.time()
    live = _aged(tmp_path / f"{prefix}new-active.txt", 25, now)
    gone = _aged(tmp_path / "inferenceos-key-o999999999xu-gone.txt", 25, now)
    real = rs._process_start_ms
    with patch.object(rs, "_process_start_ms", side_effect=lambda pid: None if pid == 999_999_999 else real(pid)):
        assert rs.sweep_stale_key_files(temp_dir=tmp_path, now=now) == 1
    assert live.exists() and not gone.exists()
