"""Development capture and audit gates, without launching any engine or model."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import psutil
import pytest

from inferenceos.backends.llama_server_toolchain import LlamaServerProcess
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.tune import reuse_evidence as evidence
from inferenceos.tune.prompt_reuse import Reading, SWA_FULL_WARNING, evaluate
from inferenceos.tune.prompt_reuse_sim import SimulatedReuseEngine
from inferenceos.tune.reuse_correctness import DEFAULT, SWA_FULL, build_pairs, answer_oracle, correctness_summary

ROOT = Path(__file__).resolve().parents[1]


def script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_file_identity_detects_a_file_changed_while_open(tmp_path: Path) -> None:
    path = tmp_path / "engine.dll"
    path.write_bytes(b"before")
    digest = evidence.file_identity(path)
    assert digest["sha256"] == hashlib.sha256(b"before").hexdigest()
    original = hashlib.file_digest

    def mutate(handle: Any, algorithm: str) -> Any:
        result = original(handle, algorithm)
        path.write_bytes(b"after-longer")
        return result

    with patch.object(hashlib, "file_digest", side_effect=mutate), pytest.raises(OSError, match="changed"):
        evidence.file_identity(path)


def test_path_and_descriptor_ctime_may_differ_but_each_must_remain_stable(tmp_path: Path) -> None:
    path = tmp_path / "stable.dll"
    path.write_bytes(b"stable")
    original = evidence.os.fstat
    def descriptor(fd: int) -> Any:
        stat = original(fd)
        return SimpleNamespace(st_dev=stat.st_dev, st_ino=stat.st_ino, st_size=stat.st_size,
                               st_mtime_ns=stat.st_mtime_ns, st_ctime_ns=stat.st_ctime_ns + 999)
    with patch.object(evidence.os, "fstat", side_effect=descriptor):
        assert evidence.file_identity(path)["sha256"] == hashlib.sha256(b"stable").hexdigest()


def test_image_hashes_the_actual_mapped_libraries_not_model_files(tmp_path: Path) -> None:
    paths = [tmp_path / n for n in ("engine.exe", "real.dll", "lib.so.1", "model.gguf")]
    for path in paths:
        path.write_bytes(path.name.encode())
    owner = SimpleNamespace(create_time=lambda: 1.5, exe=lambda: str(paths[0]),
                            memory_maps=lambda **_: [SimpleNamespace(path=str(p)) for p in paths])
    result = evidence.process_image(23, process=lambda _: owner)
    assert result["status"] == "observed"
    assert {Path(f["path"]).name for f in result["files"]} == {"engine.exe", "real.dll", "lib.so.1"}
    replacement = SimpleNamespace(create_time=lambda: 2.5, exe=owner.exe)
    with patch.object(evidence, "file_identity", wraps=evidence.file_identity):
        factory = iter([owner, replacement])
        result = evidence.process_image(23, process=lambda _: next(factory))
    assert result["status"] == "incomplete" and "identity changed" in str(result["problems"])


@pytest.mark.parametrize("fault", ["denied", "unreadable", "deleted", "exited", "absent"])
def test_incomplete_image_observations_never_attest(tmp_path: Path, fault: str) -> None:
    exe = tmp_path / "engine.exe"
    exe.write_bytes(b"launcher")

    def maps(**_: Any) -> list[Any]:
        if fault == "denied":
            raise psutil.AccessDenied(23)
        return [SimpleNamespace(path=str(tmp_path / "missing.dll") + (" (deleted)" if fault == "deleted" else ""))]

    owner = SimpleNamespace(create_time=lambda: 1.5, exe=lambda: str(exe), memory_maps=maps)
    def factory(pid: int) -> Any:
        if fault == "exited":
            raise psutil.NoSuchProcess(pid)
        return owner
    result = evidence.process_image(None if fault == "absent" else 23, process=factory)
    assert result["status"] == "incomplete" and result["problems"]
    if fault == "denied":
        assert result["files"][0]["sha256"] == hashlib.sha256(b"launcher").hexdigest()


def test_startup_capture_reports_loss_and_stream_failure() -> None:
    owner = LlamaServerProcess(["never-executed.exe"])
    owner._startup_limit = 2
    reader = owner._spawn_reader(io.StringIO("a\nb\nc\n"), "stdout")
    reader.join(2)
    first = owner.startup_snapshot()
    assert first["text"] == "a\nb" and first["truncated"] and first["dropped_lines"] == 1

    class BadStream:
        def __iter__(self) -> Any:
            raise OSError("closed")
    reader = owner._spawn_reader(BadStream(), "stderr")
    reader.join(2)
    assert owner.startup_snapshot()["reader_errors"] == ["stderr"]
    first["reader_errors"].append("external mutation")
    assert owner.startup_snapshot()["reader_errors"] == ["stderr"]


def test_startup_snapshot_is_bound_to_backend_generation() -> None:
    from inferenceos.backends.resident_server import LlamaServerResidentBackend
    backend = LlamaServerResidentBackend(server=Path("not-executed.exe"))
    def snapshot() -> dict[str, Any]:
        backend._epoch += 1
        return {"text": "old process"}
    backend._process = SimpleNamespace(startup_snapshot=snapshot)  # type: ignore[assignment]
    assert backend.startup_log_snapshot()["generation_current"] is False


def engine_with_capture(enabled: bool = True) -> tuple[Any, Any]:
    backend = SimpleNamespace(pid=23, identity=SimpleNamespace(pid=23, create_time=1.5),
                              start=lambda *_: None, is_active=lambda: False,
                              unconfirmed_pid=None, undeleted_key_files=[], api_key_file=None,
                              startup_log_snapshot=lambda: {"available": True, "text": "model loaded",
                                                            "generation_current": True})
    engine = script("evaluate_prompt_reuse").RealReuseEngine(
        model=object(), configuration=RuntimeConfiguration(context_length=4096),
        backend_factory=lambda: backend, capture_load_evidence=enabled, run_id="run")
    return engine, backend


def test_capture_is_opt_in_and_does_not_create_another_cleanup_owner() -> None:
    engine, backend = engine_with_capture(False)
    with patch.object(evidence, "process_image", side_effect=AssertionError("must not query")):
        engine.start(DEFAULT)
        assert engine.stop().clean
    assert engine.load_evidence == []
    engine, backend = engine_with_capture()
    with patch.object(evidence, "process_image", return_value={"create_time_s": 1.5}):
        engine.start(DEFAULT)
        assert engine.load_evidence_id == "run/load-000001"
        assert engine._params.log_verbosity == 4
        engine.stop()
    load = engine.load_evidence[0]
    assert [s["phase"] for s in load["snapshots"]] == ["ready", "before_stop"]
    assert load["state"] == "released" and load["capture_seconds"] >= 0
    assert engine.load_evidence_id is None


@pytest.mark.parametrize("error", [OSError("diagnostic denied"), KeyboardInterrupt()])
def test_capture_failure_still_stops_through_the_existing_owner(error: BaseException) -> None:
    engine, _ = engine_with_capture()
    with patch.object(evidence, "process_image", return_value={"create_time_s": 1.5}):
        engine.start(DEFAULT)
    with patch.object(evidence, "process_image", side_effect=error):
        if isinstance(error, KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt) as raised:
                engine.stop()
            assert raised.value is error
        else:
            assert engine.stop().clean
    assert not engine._loaded and engine.held is None
    assert engine.load_evidence[0]["state"] == "released"
    if isinstance(error, OSError):
        assert engine.load_evidence[0]["capture_errors"]


def test_start_error_is_not_masked_by_an_ordinary_capture_error() -> None:
    engine, backend = engine_with_capture()
    original = RuntimeError("original startup failure")
    def fail(*_: Any) -> None:
        raise original
    backend.start = fail
    with patch.object(evidence, "process_image", side_effect=OSError("denied")):
        with pytest.raises(RuntimeError) as raised:
            engine.start(DEFAULT)
        assert raised.value is original
        assert engine.stop().clean
    assert engine.load_evidence[0]["capture_errors"]


def complete_record() -> dict[str, Any]:
    """Synthetic observations only: used to attack audit logic, never a real artifact."""
    from inferenceos.tune.prompt_reuse import TrialResult
    audit = script("audit_prompt_reuse_evidence")
    pairs = tuple(p for p in build_pairs() if p.regression)
    config = RuntimeConfiguration(context_length=4096).model_dump(mode="json")
    sim = SimulatedReuseEngine(oracle=answer_oracle(pairs), sliding_window=True)
    result = evaluate(sim, cases=[p.case for p in pairs], variants=(DEFAULT, SWA_FULL), blocks=3,
                      context_length=4096, configuration=config, run_id="fixture-only", power=lambda: "ac",
                      power_check=True, cancellation=False)
    result.update(simulated=False, model_file_unchanged=True, model_file_identity={"sha256": audit.MODEL_SHA256},
                  model_preflight={"expected_sha256": audit.MODEL_SHA256, "pin_verified": True,
                                   "saved_setup_required": True, "saved_setup_verified": True},
                  study={"version": "reuse-correctness-8"}, preregistration={"sha256": "a" * 64},
                  data_folder_unchanged=True,
                  data_folder_evidence={"version": "file-fingerprint-1", "root": "fixture", "before": {},
                                        "after": {}, "before_complete": True, "after_complete": True})
    loads = []
    for index, trial in enumerate(result["trials"]):
        identity = f"fixture-only/load-{index + 1:06d}"
        for key in ("prime", "measured"):
            trial[key]["load_evidence_id"] = identity
        full = trial["variant_id"] == SWA_FULL.variant_id
        log = "creating non-SWA KV cache, size = 4096 cells\ncreating SWA KV cache, size = "
        log += "4096 cells\n" + SWA_FULL_WARNING + "\n" if full else "1024 cells\n"
        log += "srv    load_model: model loaded"
        from inferenceos.tune.prompt_reuse import swa_cache_evidence
        trial["swa_cache"] = swa_cache_evidence(log, requested_full=full)
        files = [{"path": name, "sha256": hashlib.sha256(name.encode()).hexdigest()}
                 for name in ["llama-server.exe", *sorted(audit.REQUIRED_LIBRARIES)]]
        snapshot = {"process_current": True,
                    "image": {"status": "observed", "pid": 23 + index, "create_time_s": 1.5,
                              "executable": "llama-server.exe", "files": files, "problems": []},
                    "startup": {"available": True, "truncated": False, "reader_errors": [], "text": log,
                                "lines": 4, "line_limit": 5000, "dropped_lines": 0,
                                "text_sha256": hashlib.sha256(log.encode()).hexdigest()}}
        loads.append({"load_id": identity, "variant": asdict(SWA_FULL if full else DEFAULT),
                      "requested_params": {**config, "swa_full": full, "cache_reuse": None, "log_verbosity": 4},
                      "state": "released", "cleanup": trial["cleanup"], "capture_errors": [],
                      "snapshots": [{**deepcopy(snapshot), "phase": phase} for phase in ("ready", "before_stop")]})
    result["load_evidence"] = {"version": evidence.EVIDENCE_VERSION, "simulated": False, "loads": loads}
    result["correctness"] = correctness_summary([TrialResult(**t) for t in result["trials"]], pairs)
    return cast(dict[str, Any], json.loads(json.dumps(result)))


def test_complete_audit_does_not_imply_a_mitigation() -> None:
    result = script("audit_prompt_reuse_evidence").audit(complete_record(), "a" * 64)
    assert result["observations_complete"], result["problems"]
    assert result["g3_interpretable"]  # neither arm wrong, not proof of a mitigation
    assert "wrong under neither" in str(result["correctness"]["swa_full_comparison"])


@pytest.mark.parametrize("fault", ["duplicate", "missing", "wrong_id", "swapped", "variant", "params", "pid",
                                  "digest", "truncated", "map", "map_malformed", "library", "bytes",
                                  "simulation", "stop", "unknown_state", "raw_state", "rules", "model", "power",
                                  "reused_process"])
def test_audit_rejects_incomplete_or_mismatched_observations(fault: str) -> None:
    record = complete_record()
    loads = record["load_evidence"]["loads"]
    snap = loads[0]["snapshots"][1]
    if fault == "duplicate": loads.append(deepcopy(loads[0]))
    if fault == "missing": loads.pop()
    if fault == "wrong_id": record["trials"][0]["measured"]["load_evidence_id"] = "old/load-000001"
    if fault == "swapped": loads[0], loads[1] = loads[1], loads[0]
    if fault == "variant": loads[0]["variant"]["swa_full"] = True
    if fault == "params": del loads[0]["requested_params"]["threads"]
    if fault == "pid": snap["image"]["create_time_s"] += 1
    if fault == "digest": snap["startup"]["text"] += "changed"
    if fault == "truncated": snap["startup"]["truncated"] = True
    if fault == "map": record["data_folder_evidence"]["after"]["changed"] = {}
    if fault == "map_malformed": record["data_folder_evidence"]["before_complete"] = False
    if fault == "library": snap["image"]["files"].pop()
    if fault == "bytes": snap["image"]["files"][0]["sha256"] = "c" * 64
    if fault == "simulation": record["simulated"] = True
    if fault == "stop": loads[0]["state"] = "unreleased"
    if fault == "unknown_state": record["correctness"]["swa_full_comparison"]["interpretable"] = False
    if fault == "raw_state": snap["startup"]["text"] += "\n" + SWA_FULL_WARNING
    if fault == "rules": record["preregistration"]["sha256"] = "b" * 64
    if fault == "model": record["model_file_identity"]["sha256"] = "b" * 64
    if fault == "power": record["trials"][0]["power"][1] = "unknown"
    if fault == "reused_process":
        for snapshot in loads[1]["snapshots"]:
            snapshot["image"].update(pid=loads[0]["snapshots"][0]["image"]["pid"])
    result = script("audit_prompt_reuse_evidence").audit(record, "a" * 64)
    assert not result["observations_complete"] and not result["g3_interpretable"], fault


def test_audit_cli_exclusive_and_protects_source_and_rules(tmp_path: Path) -> None:
    audit = script("audit_prompt_reuse_evidence")
    source, rules, out = (tmp_path / name for name in ("input.json", "rules.txt", "audit.json"))
    engine_files = {name: hashlib.sha256(name.encode()).hexdigest()
                    for name in audit.REQUIRED_LIBRARIES | {"llama-server.exe"}}
    rules.write_text(json.dumps({"protocol": "g3-evidence-2", "study_version": "reuse-correctness-8",
                                "model_sha256": audit.MODEL_SHA256, "require_saved_setup": True,
                                "engine_files": engine_files}), encoding="utf-8")
    digest = hashlib.sha256(rules.read_bytes()).hexdigest()
    record = complete_record()
    record["preregistration"]["sha256"] = digest
    source.write_text(json.dumps(record), encoding="utf-8")
    before = source.read_bytes()
    args = [str(source), "--rules", str(rules), "--rules-sha256", digest, "--out"]
    assert audit.main([*args, str(out)]) == 0
    original = out.read_bytes()
    assert audit.main([*args, str(out)]) == 4
    assert out.read_bytes() == original
    for target in (source, rules):
        assert audit.main([*args, str(target)]) == 4
    assert source.read_bytes() == before


def test_retained_fingerprint_maps_and_baseline_precede_model_work(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _real_mode
    out = tmp_path / "out.json"
    class CheckBaseline(SimulatedReuseEngine):
        def start(self, variant: Any) -> None:
            baseline = json.loads(out.read_text())["data_folder_evidence"]
            assert baseline["before_complete"] is True and baseline["after_complete"] is False
            super().start(variant)
    code, saved = _real_mode(tmp_path, out, engine=CheckBaseline())
    record = json.loads(out.read_text())
    maps = record["data_folder_evidence"]
    assert code == 0 and maps["before_complete"] and maps["after_complete"]
    assert maps["before"] == maps["after"]
    assert maps["before"]["saved_setups.json"]["sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()


@pytest.mark.parametrize("stage", ["before", "after"])
@pytest.mark.parametrize("error", [OSError("cannot fingerprint"), KeyboardInterrupt()])
def test_fingerprint_failure_keeps_explicit_completeness_and_prior_map(
    tmp_path: Path, stage: str, error: BaseException,
) -> None:
    from tests.test_prompt_reuse import _real_mode
    out = tmp_path / "out.json"
    original = evidence.file_identity
    calls = 0
    def fail(path: Path) -> dict[str, Any]:
        nonlocal calls
        if path.name == "saved_setups.json":
            calls += 1
            if calls == (1 if stage == "before" else 2):
                raise error
        return original(path)
    sim = SimulatedReuseEngine()
    with patch.object(evidence, "file_identity", side_effect=fail):
        if isinstance(error, KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                _real_mode(tmp_path, out, engine=sim)
        else:
            assert _real_mode(tmp_path, out, engine=sim)[0] == 3
    record = json.loads(out.read_text())
    maps = record["data_folder_evidence"]
    assert maps["after_complete"] is False and maps["after"] is None
    assert maps["before_complete"] is (stage == "after")
    if stage == "before":
        assert sim.starts == 0
    else:
        assert maps["before"]["saved_setups.json"]["sha256"]
    assert record["status"] != "done"


def test_changed_data_keeps_both_maps_and_refuses_success(tmp_path: Path) -> None:
    from tests.test_prompt_reuse import _real_mode
    class ChangeData(SimulatedReuseEngine):
        def start(self, variant: Any) -> None:
            (tmp_path / "data" / "saved_setups.json").write_bytes(b"changed by test double")
            super().start(variant)
    out = tmp_path / "out.json"
    assert _real_mode(tmp_path, out, engine=ChangeData())[0] == 6
    record = json.loads(out.read_text())
    maps = record["data_folder_evidence"]
    assert maps["before_complete"] and maps["after_complete"] and maps["before"] != maps["after"]
    assert record["data_folder_unchanged"] is False


def test_readings_from_both_apis_reference_the_captured_load() -> None:
    from inferenceos.interfaces.backend_adapter import GenerationResult
    engine, backend = engine_with_capture()
    backend.generate = lambda _: GenerationResult(text="57")
    backend.last_response_counters = {}
    backend.chat_request = lambda *_a, **_k: {"choices": [{"message": {"content": "57"}, "finish_reason": "stop"}]}
    with patch.object(evidence, "process_image", return_value={"create_time_s": 1.5}):
        engine.start(DEFAULT)
        native = engine.generate("question", cache_prompt=False, n_predict=24, seed=42)
        chat = engine.chat([], cache_prompt=None, max_tokens=24, seed=42)
        assert native.load_evidence_id == chat.load_evidence_id == "run/load-000001"
        engine.stop()


def test_missing_cache_logs_remain_unknown_despite_all_other_observations() -> None:
    from inferenceos.tune.prompt_reuse import TrialResult, swa_cache_evidence
    record = complete_record()
    for load, trial in zip(record["load_evidence"]["loads"], record["trials"]):
        full = trial["variant_id"] == SWA_FULL.variant_id
        text = SWA_FULL_WARNING if full else "no cache lines retained"
        for snapshot in load["snapshots"]:
            snapshot["startup"].update(text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())
        trial["swa_cache"] = swa_cache_evidence(text, requested_full=full)
    pairs = tuple(p for p in build_pairs() if p.regression)
    record["correctness"] = correctness_summary([TrialResult(**t) for t in record["trials"]], pairs)
    result = script("audit_prompt_reuse_evidence").audit(record, "a" * 64)
    assert result["observations_complete"], result["problems"]
    assert result["g3_interpretable"] is False


def test_fingerprint_rejects_population_change(tmp_path: Path) -> None:
    runner = script("evaluate_prompt_reuse")
    (tmp_path / "original").write_bytes(b"x")
    def hash_and_add(path: Path) -> dict[str, Any]:
        (tmp_path / "new").write_bytes(b"y")
        return {"size_bytes": 1, "modified_ns": 1, "sha256": "a" * 64}
    with patch.object(evidence, "file_identity", side_effect=hash_and_add), pytest.raises(OSError, match="population"):
        runner._fingerprint(tmp_path)


def test_fingerprint_cannot_silently_skip_an_unreadable_directory(tmp_path: Path) -> None:
    runner = script("evaluate_prompt_reuse")
    with patch.object(runner.os, "scandir", side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError):
            runner._fingerprint(tmp_path)


@pytest.mark.parametrize("kind", ["symlink", "reparse"])
def test_fingerprint_refuses_a_link_instead_of_claiming_its_target_unchanged(tmp_path: Path, kind: str) -> None:
    runner = script("evaluate_prompt_reuse")
    link = SimpleNamespace(path=str(tmp_path / "outside"), is_symlink=lambda: kind == "symlink",
                           stat=lambda **_: SimpleNamespace(st_file_attributes=1024))
    from contextlib import nullcontext
    with patch.object(runner.os, "scandir", return_value=nullcontext([link])):
        with pytest.raises(OSError, match="link/junction"):
            runner._fingerprint(tmp_path)


def preflight_case(tmp_path: Path, fault: str = "") -> tuple[int, dict[str, Any], int, int]:
    """Production setup store and GGUF parser; only engine/power are replaced."""
    from inferenceos.config import Settings
    from inferenceos.contracts.enums import PowerState
    from inferenceos.contracts.tune import SavedSetup, ModelFileIdentity, MachineCheck
    from inferenceos.models.loader import load_model_profile
    from tests.test_gguf import _write_gguf, _kv_string, _kv_u32

    data = tmp_path / "data"
    data.mkdir()
    model = tmp_path / "metadata_only.gguf"
    _write_gguf(model, 3, [_kv_string("general.architecture", "qwen2"), _kv_u32("qwen2.block_count", 2),
                          _kv_u32("qwen2.context_length", 4096), _kv_u32("qwen2.embedding_length", 16),
                          _kv_u32("qwen2.attention.head_count", 2), _kv_u32("qwen2.attention.head_count_kv", 1)], [])
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    config = RuntimeConfiguration(context_length=4096)
    st = model.stat()
    saved = SavedSetup(setup_id="test-only", model_name="metadata fixture",
                       model=ModelFileIdentity(path=str(model.resolve()), size_bytes=st.st_size,
                                               modified_ns=st.st_mtime_ns, sha256=digest),
                       context_length=4096, power_source=PowerState.AC, machine=MachineCheck(),
                       configuration=config, starting_configuration=config)
    updates: dict[str, Any] = {}
    if fault == "saved_hash": updates["sha256"] = "b" * 64
    if fault == "saved_missing_hash": updates["sha256"] = None
    if fault == "saved_size": updates["size_bytes"] = saved.model.size_bytes + 1
    if fault == "saved_date": updates["modified_ns"] = 0
    saved = saved.model_copy(update={"model": saved.model.model_copy(update=updates)})
    store = data / "saved_setups.json"
    store.write_text(json.dumps({"setups": [] if fault == "no_saved" else [saved.model_dump(mode="json")]}))
    original_store = store.read_bytes()
    rules = tmp_path / "rules.json"
    pin = "a" * 64 if fault == "pin" else digest
    frozen: dict[str, Any] = {"protocol": "g3-evidence-2", "study_version": "reuse-correctness-8",
                              "model_sha256": pin, "require_saved_setup": True}
    if fault == "pin_missing": del frozen["model_sha256"]
    if fault == "pin_malformed": frozen["model_sha256"] = "not-a-sha"
    if fault == "saved_not_required": frozen["require_saved_setup"] = False
    if fault == "rules_version": frozen["study_version"] = "older (not reuse-correctness-8)"
    rules.write_text(json.dumps([frozen] if fault == "rules_array" else frozen))
    rules_digest = hashlib.sha256(rules.read_bytes()).hexdigest()
    pairs = tuple(p for p in build_pairs() if p.regression)
    sim = SimulatedReuseEngine(oracle=answer_oracle(pairs), sliding_window=True)
    runner = script("evaluate_prompt_reuse")
    out = tmp_path / "result.json"
    settings = Settings().model_copy(update={"data_dir": data})
    architectures: list[str] = []
    def factory(**kwargs: Any) -> Any:
        architectures.append(kwargs["model"].architecture)
        return sim
    def parse(path: Path) -> Any:
        profile = load_model_profile(path)
        if fault == "changed_during_metadata":
            path.write_bytes(path.read_bytes() + b"changed after parsing")
        return profile
    args = ["--study", "correctness", "--model", str(model), "--context", "4096", "--blocks", "3",
            "--cases", "correctness/regression-pump-b-unit-7", "--swa-full-variant", "--capture-load-evidence",
            "--preregistration", str(rules), "--preregistration-sha256", rules_digest, "--out", str(out)]
    if fault in ("no_saved", "saved_hash"):
        args += ["--layers", "3"]  # Cannot bypass a stale or required saved setup.
    with patch("inferenceos.config.Settings", return_value=settings), \
            patch.object(runner, "RealReuseEngine", side_effect=factory) as made, \
            patch("inferenceos.models.loader.load_model_profile", side_effect=parse), \
            patch.object(runner, "power_state", return_value="ac"):
        code = runner.main(args)
    assert store.read_bytes() == original_store
    record = json.loads(out.read_bytes()) if out.exists() else {}
    if code == 0:
        assert architectures == ["qwen2"]  # The positive control used the real metadata parser.
    return code, record, made.call_count, sim.starts


@pytest.mark.parametrize("fault", ["pin", "saved_hash", "saved_missing_hash", "saved_size", "no_saved",
                                  "pin_missing", "pin_malformed", "saved_not_required", "changed_during_metadata",
                                  "rules_version", "rules_array"])
def test_frozen_preflight_refuses_before_engine_construction(tmp_path: Path, fault: str) -> None:
    code, record, constructions, starts = preflight_case(tmp_path, fault)
    assert code in (4, 6) and constructions == starts == 0
    assert record.get("status") != "done"
    if record:
        assert record["data_folder_unchanged"] is True


@pytest.mark.parametrize("fault", ["", "saved_date"])
def test_frozen_preflight_accepts_only_verified_model_and_saved_bytes(tmp_path: Path, fault: str) -> None:
    code, record, constructions, starts = preflight_case(tmp_path, fault)
    assert code == 0 and record["status"] == "done" and constructions == 1 and starts == 12
    check = record["model_preflight"]
    assert check["pin_verified"] and check["saved_setup_verified"]
    assert check["expected_sha256"] == record["model_file_identity"]["sha256"]


def test_a_model_changed_after_preflight_cannot_construct_a_backend(tmp_path: Path) -> None:
    from inferenceos.models.identity import model_file_signature
    from unittest.mock import Mock
    model = tmp_path / "never_loaded.gguf"
    model.write_bytes(b"before")
    generation = model_file_signature(model)
    factory = Mock()
    engine = script("evaluate_prompt_reuse").RealReuseEngine(
        model=object(), configuration=RuntimeConfiguration(context_length=4096),
        backend_factory=factory, model_generation=generation)
    model.write_bytes(b"replacement bytes")
    with pytest.raises(RuntimeError, match="model file changed"):
        engine.start(DEFAULT)
    factory.assert_not_called()
    assert engine.stop().clean
