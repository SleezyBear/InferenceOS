"""Production runner/auditor boundaries, using synthetic files and fake engines only."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import pytest

from inferenceos.config import Settings
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.tune import MachineCheck, ModelFileIdentity, SavedSetup
from inferenceos.experiment.restoration import runtime_params_from
from inferenceos.interfaces.backend_adapter import LookupDecodingParams
from inferenceos.tune.lookup_audit import audit
from inferenceos.tune.lookup_decoding import CONTROL, LOOKUP, VERSION, PROTOCOL, build_cases, digest, evaluate, summarize
from inferenceos.tune.lookup_sim import LookupSimulator
from tests.test_gguf import _write_gguf, _kv_string, _kv_u32

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def runner() -> Any:
    return importlib.import_module("evaluate_lookup_decoding")


def rules_for(models: dict[str, Any]) -> dict[str, Any]:
    for value in models.values():
        value.setdefault("configuration_sha256", digest(RuntimeConfiguration(context_length=4096).model_dump(mode="json")))
    return {"protocol": PROTOCOL, "version": VERSION, "parameters": runner().parameters(),
            "require_saved_setup": True, "models": models, "model_order": list(models)}


def synthetic_record() -> tuple[dict[str, Any], dict[str, Any], str]:
    """In-memory fake evidence, explicitly never a hardware result."""
    config = RuntimeConfiguration(context_length=4096)
    configuration = config.model_dump(mode="json")
    sha = digest(configuration)
    record = evaluate(LookupSimulator(), config_sha=sha, power=lambda: "ac", checkpoint=lambda r: None)
    rules = rules_for({"models/a.gguf": {"sha256": "a" * 64}, "models/b.gguf": {"sha256": "b" * 64}})
    rules_sha = "c" * 64
    required = ["llama-server.exe", "llama-server-impl.dll", "llama.dll", "ggml-base.dll", "ggml-vulkan.dll"]
    files = [{"path": "C:/synthetic-engine/" + name, "sha256": "e" * 64} for name in required]
    rules["engine_files"] = {"tools/" + name: "e" * 64 for name in required}
    fingerprint = {"only-file": {"size": 3, "mtime_ns": 4, "sha256": "f" * 64}}
    record.update(protocol=PROTOCOL, run_id="synthetic-test-only", simulated=False,
                  configuration=configuration, configuration_sha256=sha, parameters=runner().parameters(),
                  preregistration={"sha256": rules_sha}, model="C:/fixture/models/a.gguf",
                  model_file_identity={"sha256": "a" * 64}, model_file_unchanged=True, frozen_inputs_unchanged=True,
                  model_preflight={"expected_sha256": "a" * 64, "pin_verified": True, "saved_setup_verified": True,
                                   "configuration_verified": True},
                  ambient_config={"platform": "win32", "checked": True,
                                  "environment": {"APPDATA": "", "PROGRAMDATA": ""}, "absent_paths": []},
                  data_folder_unchanged=True, data_folder_evidence={"version": "file-fingerprint-1", "before": fingerprint,
                  "after": deepcopy(fingerprint), "before_complete": True, "after_complete": True})
    loads = []
    for index, row in enumerate(record["trials"]):
        load_id = f"synthetic-test-only/load-{index+1:06d}"
        row["reading"]["load_evidence_id"] = load_id
        row["reading"]["counters"]["generated_token_ids"] = list(row["reading"]["token_ids"])
        active = row["arm"] == "lookup"
        params = runtime_params_from(config).model_copy(update={"log_verbosity": 4,
            "lookup_decoding": LookupDecodingParams(algorithm="ngram-simple" if active else "none")})
        text = ("adding speculative implementation 'ngram-simple'\n- size_n=12, size_m=16, min_hits=1\n"
                "speculative decoding context initialized\n" if active else "no implementations specified for speculative decoding\n")
        text += "model loaded\n"
        startup = {"available": True, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                   "truncated": False, "reader_errors": [], "lines": len(text.splitlines()), "line_limit": 5000,
                   "dropped_lines": 0, "generation_current": True}
        image = {"pid": 10000 + index, "create_time_s": float(index + 1), "status": "observed", "files": files,
                 "executable": files[0]["path"], "problems": []}
        loads.append({"load_id": load_id, "variant": asdict(LOOKUP if active else CONTROL),
                      "command": ["--offline", "--spec-type", "ngram-simple" if active else "none"]
                      + (["--spec-ngram-simple-size-n", "12", "--spec-ngram-simple-size-m", "16"] if active else []),
                      "requested_params": params.model_dump(mode="json"), "capture_errors": [],
                      "snapshots": [{"phase": phase, "startup": deepcopy(startup), "image": deepcopy(image), "process_current": True}
                                    for phase in ("ready", "before_stop")],
                      "state": "released", "cleanup": row["cleanup"]})
    record["load_evidence"] = {"version": "reuse-load-evidence-1", "simulated": False, "loads": loads}
    record["summary"] = summarize(record["trials"], sha)
    return record, rules, rules_sha


def test_complete_synthetic_observations_recompute_but_never_admit_a_path() -> None:
    record, rules, sha = synthetic_record()
    before = deepcopy(record)
    checked = audit(record, rules, sha)
    assert checked["observations_complete"] and checked["performance_interpretable"]
    assert checked["automatic_admission"] is False
    assert checked["summary"] == record["summary"]
    assert record == before


@pytest.mark.parametrize("fault", ["duplicate-first", "duplicate-last", "missing", "order", "variant", "config",
                                   "receipt", "model", "model-changed", "input-changed", "data-changed", "map-changed",
                                   "before-incomplete", "cleanup", "load-ref", "load-duplicate", "params", "types",
                                   "missing-slot", "raw-ids", "draft-counter", "log-truncated", "log-digest", "log-marker",
                                   "no-activation", "wrong-size", "disabled", "library", "image", "pid-reuse", "stored-summary",
                                   "stopped", "held", "simulated", "rules"])
def test_audit_refuses_false_winners_from_all_evidence_boundaries(fault: str) -> None:
    r, rules, sha = synthetic_record()
    row = r["trials"][1]
    load = r["load_evidence"]["loads"][1]
    snapshot = load["snapshots"][0]
    if fault == "duplicate-first": r["trials"].insert(0, deepcopy(row))
    elif fault == "duplicate-last": r["trials"].append(deepcopy(row))
    elif fault == "missing": r["trials"].pop()
    elif fault == "order": r["trials"].reverse()
    elif fault == "variant": load["variant"]["variant_id"] = "control"
    elif fault == "config": r["configuration"]["threads"] = 99
    elif fault == "receipt": r["model_preflight"]["saved_setup_verified"] = False
    elif fault == "model": r["model_file_identity"]["sha256"] = "0" * 64
    elif fault == "model-changed": r["model_file_unchanged"] = False
    elif fault == "input-changed": r["frozen_inputs_unchanged"] = False
    elif fault == "data-changed": r["data_folder_unchanged"] = False
    elif fault == "map-changed": r["data_folder_evidence"]["after"]["only-file"]["sha256"] = "0" * 64
    elif fault == "before-incomplete": r["data_folder_evidence"]["before_complete"] = False
    elif fault == "cleanup": load["cleanup"] = {"process_gone": False, "credential_removed": True, "problem": None}
    elif fault == "load-ref": row["reading"]["load_evidence_id"] = "not-this-load"
    elif fault == "load-duplicate": r["load_evidence"]["loads"].insert(0, deepcopy(load))
    elif fault == "params": load["requested_params"]["lookup_decoding"]["draft_length"] = 8
    elif fault == "types": row["reading"]["counters"]["generation_settings"]["speculative.types"] = "other"
    elif fault == "missing-slot": row["reading"]["counters"]["id_slot"] = None
    elif fault == "raw-ids": row["reading"]["token_ids"][-1] = True
    elif fault == "draft-counter": row["reading"]["counters"]["draft_n_accepted"] = 999
    elif fault == "log-truncated": snapshot["startup"]["truncated"] = True
    elif fault == "log-digest": snapshot["startup"]["text_sha256"] = "0" * 64
    elif fault == "log-marker": snapshot["startup"]["text"] = "no end marker"
    elif fault == "no-activation": snapshot["startup"]["text"] = "model loaded"
    elif fault == "wrong-size": snapshot["startup"]["text"] = snapshot["startup"]["text"].replace("size_m=16", "size_m=32")
    elif fault == "disabled": snapshot["startup"]["text"] += "speculative decoding not supported by this context"
    elif fault == "library": snapshot["image"]["files"][0]["sha256"] = "0" * 64
    elif fault == "image": snapshot["image"]["status"] = "incomplete"
    elif fault == "pid-reuse":
        for s in load["snapshots"]: s["image"].update(pid=10000, create_time_s=1.0)
    elif fault == "stored-summary": r["summary"]["cases"]["copy-prose"]["outcome"] = "best ever"
    elif fault == "stopped": r["status"] = "interrupted"
    elif fault == "held": r["held_engine_retry"] = {"clean": True}
    elif fault == "simulated": r["simulated"] = True
    elif fault == "rules": r["preregistration"]["sha256"] = "0" * 64
    checked = audit(r, rules, sha)
    assert not checked["performance_interpretable"] and not checked["workload_findings"]
    assert checked["problems"]


def fixture(tmp_path: Path, fault: str = "") -> tuple[Any, list[str], LookupSimulator, list[Any], bytes, Path]:
    r = runner()
    model = tmp_path / "models/a.gguf"
    model.parent.mkdir()
    _write_gguf(model, 3, [_kv_string("general.architecture", "qwen2"), _kv_u32("qwen2.block_count", 2),
                          _kv_u32("qwen2.context_length", 4096), _kv_u32("qwen2.embedding_length", 16),
                          _kv_u32("qwen2.attention.head_count", 2), _kv_u32("qwen2.attention.head_count_kv", 1)], [])
    model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    data = tmp_path / "data"
    data.mkdir()
    config = RuntimeConfiguration(context_length=4096)
    st = model.stat()
    saved = SavedSetup(setup_id="fixture", model_name="fixture",
                       model=ModelFileIdentity(path=str(model), size_bytes=st.st_size, modified_ns=st.st_mtime_ns,
                                               sha256="b"*64 if fault == "saved" else model_sha),
                       context_length=4096, power_source=PowerState.AC, machine=MachineCheck(),
                       configuration=config, starting_configuration=config)
    store = data / "saved_setups.json"
    store.write_text(json.dumps({"setups": [] if fault == "no-saved" else [saved.model_dump(mode="json")]}))
    original = store.read_bytes()
    source, engine = tmp_path / "source.py", tmp_path / "engine.dll"
    source.write_text("source")
    engine.write_bytes(b"engine fixture, never executed")
    rules = rules_for({"models/a.gguf": {"sha256": "b"*64 if fault == "model" else model_sha},
                       "models/b.gguf": {"sha256": "b"*64}})
    rules.update(python_files={"source.py": hashlib.sha256(source.read_bytes()).hexdigest()},
                 engine_files={"engine.dll": hashlib.sha256(engine.read_bytes()).hexdigest()})
    if fault == "source": rules["python_files"]["source.py"] = "b"*64
    if fault == "engine": rules["engine_files"]["engine.dll"] = "b"*64
    if fault == "configuration": rules["models"]["models/a.gguf"]["configuration_sha256"] = "b"*64
    if fault == "plan": rules["parameters"]["tokens"] = 64
    if fault == "version": rules["version"] = "different"
    if fault == "required": rules["require_saved_setup"] = False
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules))
    rules_sha = hashlib.sha256(rules_path.read_bytes()).hexdigest()
    if fault == "rule-pin": rules_sha = "0"*64
    out = tmp_path / "out.json"
    args = ["--model", str(model), "--preregistration", str(rules_path), "--preregistration-sha256", rules_sha, "--out", str(out)]
    made: list[Any] = []
    simulator = LookupSimulator()
    def make(**kwargs: Any) -> Any:
        made.append(kwargs)
        return simulator
    original_inputs = r.FrozenInputs
    def windows_inputs(*args: Any, **kwargs: Any) -> Any:
        # The production study is Windows-only. Keep synthetic preflight controls
        # portable without bypassing any real FrozenInputs validation.
        return original_inputs(*args, **kwargs, platform="win32")
    patches = [patch.object(r, "ROOT", tmp_path), patch.object(r, "Settings", return_value=Settings().model_copy(update={"data_dir": data})),
               patch.object(r, "engine_environment", return_value={"APPDATA": "", "PROGRAMDATA": ""}),
               patch.object(r, "FrozenInputs", side_effect=windows_inputs),
               patch.object(r, "RealLookupEngine", side_effect=make), patch.object(r, "power_state", return_value="ac")]
    return r, args, simulator, [made, patches], original, store


@pytest.mark.parametrize("fault", ["model", "saved", "no-saved", "source", "engine", "configuration", "plan", "version", "required", "rule-pin"])
def test_real_runner_refuses_before_engine_construction(fault: str, tmp_path: Path) -> None:
    from contextlib import ExitStack
    r, args, simulator, info, original, store = fixture(tmp_path, fault)
    made, patches = info
    with ExitStack() as stack:
        for p in patches: stack.enter_context(p)
        code = r.main(args)
    record = json.loads((tmp_path / "out.json").read_bytes())
    assert code in (3, 4, 6) and record["status"] != "done"
    assert not made and simulator.starts == 0
    assert store.read_bytes() == original


def test_real_setup_and_metadata_positive_control_reaches_only_fake_engine(tmp_path: Path) -> None:
    from contextlib import ExitStack
    r, args, simulator, info, original, store = fixture(tmp_path)
    made, patches = info
    with ExitStack() as stack:
        for p in patches: stack.enter_context(p)
        code = r.main(args)
    record = json.loads((tmp_path / "out.json").read_bytes())
    assert code == 2 and record["status"] == "incomplete"  # fake engine has no real process snapshots
    assert len(made) == 1 and simulator.starts == simulator.stops == 40
    assert made[0]["model"].architecture == "qwen2"
    assert record["model_preflight"]["saved_setup_verified"]
    assert store.read_bytes() == original


def test_simulation_is_complete_but_has_no_live_model_or_input_attestation(tmp_path: Path) -> None:
    out = tmp_path / "sim.json"
    assert runner().main(["--simulate", "--out", str(out)]) == 0
    r = json.loads(out.read_bytes())
    assert r["status"] == "done" and r["simulated"] is True
    assert len(r["trials"]) == 40 and not r["load_evidence"]["loads"]
    assert "model_file_identity" not in r


def test_existing_output_is_preserved_and_no_engine_constructed(tmp_path: Path) -> None:
    out = tmp_path / "existing.json"
    out.write_bytes(b"old evidence")
    with patch.object(runner(), "LookupSimulator") as make:
        assert runner().main(["--simulate", "--out", str(out)]) == 4
    assert not make.called and out.read_bytes() == b"old evidence"


def test_simulation_cannot_write_into_data_or_product_source(tmp_path: Path) -> None:
    r = runner()
    with patch.object(r, "Settings", return_value=Settings().model_copy(update={"data_dir": tmp_path})):
        assert r.main(["--simulate", "--out", str(tmp_path / "saved_setups.json")]) == 4
    assert r.main(["--simulate", "--out", str(ROOT / "src/never-create.json")]) == 4


@pytest.mark.parametrize("fault", ["start-error", "request-error", "start-interrupt", "request-interrupt", "cleanup", "quality"])
def test_runner_checkpoints_stops_and_does_not_publish_done(fault: str, tmp_path: Path) -> None:
    out = tmp_path / "partial.json"
    sim = LookupSimulator(fault=fault)
    with patch.object(runner(), "LookupSimulator", return_value=sim):
        if "interrupt" in fault:
            with pytest.raises(KeyboardInterrupt): runner().main(["--simulate", "--out", str(out)])
        else:
            assert runner().main(["--simulate", "--out", str(out)]) in (2, 3)
    r = json.loads(out.read_bytes())
    assert r["status"] != "done" and r["trials"]
    assert sim.starts == sim.stops <= 2


def test_preparing_and_checkpoint_publication_never_claim_final_done(tmp_path: Path) -> None:
    r = runner()
    published: list[dict[str, Any]] = []
    original = r._write
    def capture(path: Path, value: dict[str, Any]) -> None:
        published.append(deepcopy(value))
        original(path, value)
    with patch.object(r, "_write", side_effect=capture):
        assert r.main(["--simulate", "--out", str(tmp_path / "x.json")]) == 0
    assert published[-1]["status"] == "done"
    assert all(x["status"] != "done" for x in published[:-1])


@pytest.mark.parametrize("name", ["source.py", "engine.dll", "rules.json"])
def test_frozen_inputs_refuse_replacement_or_deletion_before_another_load(tmp_path: Path, name: str) -> None:
    r = runner()
    source, engine, rules_file = (tmp_path / p for p in ("source.py", "engine.dll", "rules.json"))
    source.write_bytes(b"python"); engine.write_bytes(b"engine"); rules_file.write_bytes(b"rules")
    rules = {"python_files": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()},
             "engine_files": {"engine.dll": hashlib.sha256(engine.read_bytes()).hexdigest()}}
    inputs = r.FrozenInputs(tmp_path, rules, rules_path=rules_file, rules_sha=hashlib.sha256(rules_file.read_bytes()).hexdigest(),
                           environment=lambda: {"APPDATA": "", "PROGRAMDATA": ""}, platform="win32")
    inputs.check(full=True)
    (tmp_path / name).write_bytes(b"replacement with a different generation")
    with pytest.raises((ValueError, OSError)): inputs.check()
    (tmp_path / name).unlink()
    with pytest.raises((ValueError, OSError)): inputs.check()


@pytest.mark.parametrize("raw", ['{"protocol":1,"protocol":2}', '{"models":{},"Models":{}}', '{"bad":NaN}', '[]', '{}'])
def test_malformed_or_ambiguous_frozen_rules_refuse(tmp_path: Path, raw: str) -> None:
    p = tmp_path / "rules.json"
    p.write_text(raw)
    with pytest.raises(ValueError): runner().read_rules(p, hashlib.sha256(p.read_bytes()).hexdigest())


def test_a_completed_error_does_not_start_again_even_when_checkpoint_publication_fails() -> None:
    original = RuntimeError("original request failure")
    class Failed(LookupSimulator):
        def request(self, case: Any) -> Any:
            raise original
    engine = Failed()
    calls = 0
    def publish(record: Any) -> None:
        nonlocal calls
        calls += 1
        if calls > 1: raise OSError("cannot publish")
    with pytest.raises(RuntimeError) as caught:
        evaluate(engine, config_sha="c", power=lambda: "ac", checkpoint=publish)
    assert caught.value is original and engine.starts == engine.stops == 1


def test_interrupt_remains_interrupt_when_final_publication_fails(tmp_path: Path) -> None:
    r = runner()
    sim = LookupSimulator(fault="request-interrupt")
    with patch.object(r, "LookupSimulator", return_value=sim), patch.object(r, "_write", side_effect=OSError("disk")):
        # The first preparing publication fails before a load is owed, so no
        # interrupt has actually occurred. Use a late-only failure below instead.
        assert r.main(["--simulate", "--out", str(tmp_path / "early.json")]) == 7
    sim = LookupSimulator(fault="request-interrupt")
    original_write = r._write
    def late(path: Path, record: dict[str, Any]) -> None:
        if record.get("trials"): raise OSError("disk after trial")
        original_write(path, record)
    with patch.object(r, "LookupSimulator", return_value=sim), patch.object(r, "_write", side_effect=late):
        with pytest.raises(KeyboardInterrupt): r.main(["--simulate", "--out", str(tmp_path / "late.json")])
    assert sim.starts == sim.stops == 1


def test_real_lookup_bridge_uses_one_backend_and_exact_opt_in_requests(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from inferenceos.models.identity import model_file_signature
    from inferenceos.interfaces.backend_adapter import GenerationResult
    from tests.resident_helpers import synthetic_model

    r = runner()
    source = tmp_path / "source.py"; source.write_bytes(b"source")
    engine_file = tmp_path / "engine.dll"; engine_file.write_bytes(b"engine")
    model = tmp_path / "model.gguf"; model.write_bytes(b"never loaded")
    inputs = r.FrozenInputs(tmp_path, {"python_files": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()},
                                    "engine_files": {"engine.dll": hashlib.sha256(engine_file.read_bytes()).hexdigest()}},
                           environment=lambda: {"APPDATA": "", "PROGRAMDATA": ""}, platform="win32")
    made: list[Any] = []
    class Backend:
        last_command = None
        undeleted_key_files: tuple[Path, ...] = ()
        unconfirmed_pid = None
        api_key_file = None
        def __init__(self) -> None:
            self.active = False
            self.params: list[Any] = []
            self.requests: list[Any] = []
            self.last_response_counters: dict[str, Any] = {}
        def start(self, model: Any, params: Any) -> None:
            self.active = True; self.params.append(params)
        def is_active(self) -> bool: return self.active
        def tokenize(self, text: str) -> list[int]: return [1, 2, 3]
        def generate(self, request: Any) -> Any:
            self.requests.append(request)
            self.last_response_counters = {"generated_token_ids": [1, 2, 3], "stop_seen": True}
            return GenerationResult(text="reply", generated_tokens=3, total_s=1, ttft_s=.1)
        def stop(self) -> Any:
            self.active = False
            return SimpleNamespace(cleanup_complete=True, credential_removed=True)
    def factory() -> Any:
        made.append(Backend()); return made[-1]
    bridge = r.RealLookupEngine(model=synthetic_model(str(model)), configuration=RuntimeConfiguration(context_length=4096),
                               backend_factory=factory, inputs=inputs, model_generation=model_file_signature(model))
    for variant in (CONTROL, LOOKUP):
        bridge.start(variant)
        bridge.request(build_cases()[0])
        assert bridge.stop().clean
    assert len(made) == 1 and len(made[0].params) == 2
    assert [p.lookup_decoding.algorithm for p in made[0].params] == ["none", "ngram-simple"]
    assert all(q.return_tokens and not q.cache_prompt and not q.ignore_eos for q in made[0].requests)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError): bridge.start(LOOKUP)
    assert len(made[0].params) == 2 and not made[0].active


def test_offline_cli_is_exclusive_and_protects_originals(tmp_path: Path) -> None:
    cli = importlib.import_module("audit_lookup_decoding")
    record, rules, _ = synthetic_record()
    rules_path, source, out = (tmp_path / n for n in ("rules.json", "record.json", "audit.json"))
    rules_path.write_text(json.dumps(rules))
    sha = hashlib.sha256(rules_path.read_bytes()).hexdigest()
    record["preregistration"]["sha256"] = sha
    source.write_text(json.dumps(record))
    original = source.read_bytes()
    args = [str(source), "--rules", str(rules_path), "--rules-sha256", sha, "--out"]
    assert cli.main([*args, str(out)]) == 0
    previous = out.read_bytes()
    assert cli.main([*args, str(out)]) == 4 and out.read_bytes() == previous
    for target in (source, rules_path):
        assert cli.main([*args, str(target)]) == 4
    assert source.read_bytes() == original


def test_incomplete_or_simulated_evidence_is_never_relabelled_as_hardware(tmp_path: Path) -> None:
    cli = importlib.import_module("audit_lookup_decoding")
    record, rules, _ = synthetic_record()
    record["simulated"] = True
    record["summary"] = summarize(record["trials"], record["configuration_sha256"], simulated=True)
    rules_path, source, out = (tmp_path / n for n in ("rules.json", "sim.json", "audit.json"))
    rules_path.write_text(json.dumps(rules))
    sha = hashlib.sha256(rules_path.read_bytes()).hexdigest()
    record["preregistration"]["sha256"] = sha
    source.write_text(json.dumps(record))
    assert cli.main([str(source), "--rules", str(rules_path), "--rules-sha256", sha, "--out", str(out)]) == 2
    checked = json.loads(out.read_bytes())
    assert not checked["performance_interpretable"] and not checked["workload_findings"]


def test_interrupt_during_held_cleanup_rethrows_the_same_object(tmp_path: Path) -> None:
    original = KeyboardInterrupt("cleanup interrupted")
    class Held(LookupSimulator):
        def retry_held(self, *, propagate_interrupt: bool = False) -> None:
            assert propagate_interrupt
            raise original
    with patch.object(runner(), "LookupSimulator", return_value=Held()):
        with pytest.raises(KeyboardInterrupt) as caught:
            runner().main(["--simulate", "--out", str(tmp_path / "result.json")])
    assert caught.value is original
    saved = json.loads((tmp_path / "result.json").read_bytes())
    assert saved["status"] == "interrupted" and saved["interruption"]["stage"] == "held cleanup"


def test_publication_interrupt_after_replacement_is_not_left_as_done(tmp_path: Path) -> None:
    r = runner()
    original = KeyboardInterrupt("after replace")
    writer = r._write
    thrown = False
    def publish(path: Path, record: dict[str, Any]) -> None:
        nonlocal thrown
        writer(path, record)
        if record["status"] == "done" and not thrown:
            thrown = True
            raise original
    with patch.object(r, "_write", side_effect=publish):
        with pytest.raises(KeyboardInterrupt) as caught:
            r.main(["--simulate", "--out", str(tmp_path / "result.json")])
    assert caught.value is original
    assert json.loads((tmp_path / "result.json").read_bytes())["status"] == "interrupted"


def test_shared_release_retry_opt_in_preserves_interrupt_without_changing_default_contract() -> None:
    from types import SimpleNamespace
    from evaluate_prompt_reuse import RealReuseEngine
    original = KeyboardInterrupt("native cleanup")
    def fail() -> None: raise original
    backend = SimpleNamespace(cleanup=fail, confirm_previous_gone=lambda: False, is_active=lambda: True,
                              unconfirmed_pid=123, undeleted_key_files=(), api_key_file=None)
    bridge = RealReuseEngine(model=None, configuration=None, backend_factory=lambda: backend)
    bridge._backend, bridge._loaded, bridge.unreleased = backend, True, "held"
    with pytest.raises(KeyboardInterrupt) as caught: bridge.retry_held(propagate_interrupt=True)
    assert caught.value is original and bridge.held is backend
    default = bridge.retry_held()
    assert default and default["error"].startswith("KeyboardInterrupt") and not default["released"]


@pytest.mark.parametrize("fault", ["missing-marker", "unsupported", "wrong-size", "no-context", "truncated", "stale", "image", "ambient", "stale-log", "capture-error"])
def test_actual_lookup_start_gate_prevents_candidate_measurement(fault: str, tmp_path: Path) -> None:
    from evaluate_prompt_reuse import RealReuseEngine
    from inferenceos.tune.prompt_reuse import Cleanup
    r = runner()
    inputs, directory = frozen_engine_fixture(tmp_path)
    bridge = r.RealLookupEngine(inputs=inputs, model=None, configuration=RuntimeConfiguration(context_length=4096),
                               backend_factory=lambda: None, capture_load_evidence=True)
    sim = LookupSimulator()
    counts = {"start": 0, "request": 0, "stop": 0}
    def fake_start(self: Any, variant: Any) -> None:
        counts["start"] += 1
        sim.start(variant)
        active = variant.variant_id == "lookup"
        text = ("adding speculative implementation 'ngram-simple'\nsize_n=12, size_m=16, min_hits=1\n"
                "speculative decoding context initialized\n" if active else "no implementations specified\n") + "model loaded"
        log = {"available": True, "truncated": False, "reader_errors": [], "text": text, "generation_current": True}
        image = {"status": "observed", "executable": str(directory / "llama-server.exe"),
                 "files": [{"path": str(tmp_path / name), "sha256": sha}
                           for name, sha in inputs.hashes.items() if name.endswith((".exe", ".dll"))]}
        snapshot = {"phase": "ready", "startup": log, "image": image, "process_current": True}
        if active:
            if fault == "missing-marker": log["text"] = text.replace("model loaded", "")
            elif fault == "unsupported": log["text"] = text + "speculative decoding not supported by this context"
            elif fault == "wrong-size": log["text"] = text.replace("size_m=16", "size_m=32")
            elif fault == "no-context": log["text"] = text.replace("speculative decoding context initialized", "")
            elif fault == "truncated": log["truncated"] = True
            elif fault == "stale": snapshot["process_current"] = False
            elif fault == "image": snapshot["image"]["status"] = "incomplete"  # type: ignore[index]
            elif fault == "ambient": log["text"] = text + "\nusing config file: surprise.ini"
            elif fault == "stale-log": log["generation_current"] = False
        self.load_evidence = [{"snapshots": [snapshot], "capture_errors": ["incomplete capture"] if active and fault == "capture-error" else []}]
    def request(case: Any) -> Any:
        counts["request"] += 1
        return sim.request(case)
    def stop() -> Cleanup:
        counts["stop"] += 1
        return sim.stop()
    with patch.object(RealReuseEngine, "start", new=fake_start), patch.object(bridge, "request", side_effect=request), patch.object(bridge, "stop", side_effect=stop), patch.object(bridge, "load_problems", return_value=[]):
        with pytest.raises(ValueError):
            evaluate(bridge, config_sha="config", power=lambda: "ac", checkpoint=lambda r: None)
    assert counts == {"start": 2, "request": 1, "stop": 2}


@pytest.mark.parametrize("variable", ["APPDATA", "PROGRAMDATA"])
def test_ambient_ini_refuses_and_appearance_between_loads_cannot_dispatch(variable: str, tmp_path: Path) -> None:
    r = runner()
    env = {"APPDATA": "", "PROGRAMDATA": ""}
    env[variable] = str(tmp_path / variable)
    source = tmp_path / "s.py"; source.write_bytes(b"source")
    engine = tmp_path / "e.dll"; engine.write_bytes(b"never executed")
    rules = {"python_files": {"s.py": hashlib.sha256(source.read_bytes()).hexdigest()},
             "engine_files": {"e.dll": hashlib.sha256(engine.read_bytes()).hexdigest()}}
    inputs = r.FrozenInputs(tmp_path, rules, environment=lambda: env, platform="win32")
    inputs.check(full=True)
    ini = Path(env[variable]) / "llama.cpp/config.ini"
    ini.parent.mkdir(parents=True); ini.write_text("")
    with pytest.raises(ValueError, match="ambient engine configuration exists"): inputs.check()
    with pytest.raises(ValueError, match="ambient engine configuration exists"):
        r.FrozenInputs(tmp_path, rules, environment=lambda: env, platform="win32")
    ini.unlink()
    env[variable] = str(tmp_path / "different")
    with pytest.raises(ValueError, match="environment changed"): inputs.check()


def test_unreadable_ambient_path_and_unregistered_platform_refuse(tmp_path: Path) -> None:
    r = runner()
    env = {"APPDATA": str(tmp_path), "PROGRAMDATA": ""}
    with patch.object(Path, "stat", side_effect=PermissionError("denied")):
        with pytest.raises(ValueError, match="cannot be checked"): r.check_ambient_config(env, platform="win32")
    with pytest.raises(ValueError, match="Windows only"): r.check_ambient_config(env, platform="linux")


def test_known_post_request_capture_failure_stops_before_another_load() -> None:
    class BadCapture(LookupSimulator):
        def load_problems(self) -> list[str]: return ["before-stop log was truncated"]
    engine = BadCapture()
    saved = evaluate(engine, config_sha="config", power=lambda: "simulated", simulated=True, checkpoint=lambda r: None)
    assert saved["status"] == "stopped-observation"
    assert engine.starts == engine.stops == engine.requests == 1
    assert saved["trials"][0]["reading"] and saved["trials"][0]["cleanup"]["process_gone"] is True


@pytest.mark.parametrize("fault", ["truncated", "stale-log", "reader", "image", "process", "capture", "missing"])
def test_actual_released_load_inspector_stops_late_observation_failures(fault: str) -> None:
    from types import SimpleNamespace
    r = runner()
    record, rules, _ = synthetic_record()
    load = record["load_evidence"]["loads"][0]
    bridge = r.RealLookupEngine(inputs=SimpleNamespace(hashes=rules["engine_files"]), model=None,
                               configuration=None, backend_factory=lambda: None, capture_load_evidence=True)
    bridge.load_evidence = [load]
    assert not bridge.load_problems()
    before_stop = load["snapshots"][1]
    if fault == "truncated": before_stop["startup"]["truncated"] = True
    elif fault == "stale-log": before_stop["startup"]["generation_current"] = False
    elif fault == "reader": before_stop["startup"]["reader_errors"] = ["reader failed"]
    elif fault == "image": before_stop["image"]["status"] = "incomplete"
    elif fault == "process": before_stop["process_current"] = False
    elif fault == "capture": load["capture_errors"] = ["capture failed"]
    elif fault == "missing": load["snapshots"].pop()
    class Captured(LookupSimulator):
        def load_problems(self) -> list[str]: return bridge.load_problems()  # type: ignore[no-any-return]
    engine = Captured()
    result = evaluate(engine, config_sha="config", power=lambda: "simulated", simulated=True, checkpoint=lambda r: None)
    assert result["status"] == "stopped-observation" and engine.starts == engine.stops == engine.requests == 1
    assert result["trials"][0]["observation_problems"]
    assert result["trials"][0]["cleanup"] == {"process_gone": True, "credential_removed": True, "problem": None}


@pytest.mark.parametrize("interruption", [False, True])
def test_released_load_inspection_error_keeps_work_cleanup_and_original_error(interruption: bool) -> None:
    original = KeyboardInterrupt("observation") if interruption else RuntimeError("observation")
    class Failed(LookupSimulator):
        def load_problems(self) -> list[str]: raise original
    engine = Failed();saved=[]
    with pytest.raises(type(original)) as caught:
        evaluate(engine, config_sha="config", power=lambda: "simulated", simulated=True,
                 checkpoint=lambda r:saved.append(deepcopy(r)))
    assert caught.value is original and engine.starts == engine.stops == engine.requests == 1
    row=saved[-1]["trials"][0]
    assert row["reading"] and row["cleanup"]["process_gone"] is True
    assert row["errors"][-1]["stage"] == "observations"


@pytest.mark.parametrize("fault", ["ambient-missing", "ambient-unchecked", "ambient-path", "ambient-log",
                                   "offline", "type", "n", "m"])
def test_ambient_and_explicit_command_evidence_are_required_offline(fault: str) -> None:
    record, rules, sha = synthetic_record()
    load = record["load_evidence"]["loads"][1]
    if fault == "ambient-missing": record.pop("ambient_config")
    elif fault == "ambient-unchecked": record["ambient_config"]["checked"] = False
    elif fault == "ambient-path": record["ambient_config"]["absent_paths"] = ["unknown.ini"]
    elif fault == "ambient-log":
        for s in load["snapshots"]:
            text = s["startup"]["text"] + "using config file: unknown.ini\n"
            s["startup"].update(text=text, text_sha256=hashlib.sha256(text.encode()).hexdigest())
    elif fault == "offline": load["command"].remove("--offline")
    else:
        flag = {"type": "--spec-type", "n": "--spec-ngram-simple-size-n", "m": "--spec-ngram-simple-size-m"}[fault]
        load["command"][load["command"].index(flag) + 1] = "wrong"
    checked = audit(record, rules, sha)
    assert not checked["observations_complete"] and not checked["workload_findings"]


ENGINE_FILES = ("llama-server.exe", "llama-server-impl.dll", "llama.dll", "ggml-base.dll", "ggml-vulkan.dll")


@pytest.mark.parametrize("phase", ["ready", "before_stop"])
@pytest.mark.parametrize("name", ENGINE_FILES)
@pytest.mark.parametrize("fault", ["hash", "missing"])
def test_mapped_identity_contradiction_stops_at_its_first_observable_boundary(
    phase: str, name: str, fault: str,
) -> None:
    from types import SimpleNamespace
    from evaluate_prompt_reuse import RealReuseEngine

    r = runner()
    baseline, rules, _ = synthetic_record()
    sim = LookupSimulator()
    bridge = r.RealLookupEngine(inputs=SimpleNamespace(hashes=rules["engine_files"]),
                               model=None, configuration=RuntimeConfiguration(context_length=4096),
                               backend_factory=lambda: None, capture_load_evidence=True)

    def snapshot(load: dict[str, Any], at: str) -> dict[str, Any]:
        value: dict[str, Any] = deepcopy(next(s for s in load["snapshots"] if s["phase"] == at))
        if at == phase and sim.starts == 1:
            files = value["image"]["files"]
            if fault == "missing":
                value["image"]["files"] = [f for f in files if not f["path"].endswith(name)]
            else:
                next(f for f in files if f["path"].endswith(name))["sha256"] = "0" * 64
        return value

    def start(self: Any, variant: Any) -> None:
        sim.start(variant)
        load = deepcopy(baseline["load_evidence"]["loads"][sim.starts - 1])
        load["snapshots"] = [snapshot(load, "ready")]
        load["state"] = "ready"
        self.load_evidence.append(load)

    def stop() -> Any:
        cleanup = sim.stop()
        source = baseline["load_evidence"]["loads"][sim.starts - 1]
        bridge.load_evidence[-1]["snapshots"].append(snapshot(source, "before_stop"))
        bridge.load_evidence[-1]["state"] = "released"
        return cleanup

    saved: list[dict[str, Any]] = []
    with patch.object(RealReuseEngine, "start", new=start), patch.object(bridge, "request", side_effect=sim.request), patch.object(bridge, "stop", side_effect=stop):
        if phase == "ready":
            with pytest.raises(r.Refused):
                evaluate(bridge, config_sha="config", power=lambda: "ac", checkpoint=lambda x: saved.append(deepcopy(x)))
            assert saved[-1]["status"] != "done" and sim.requests == 0
        else:
            result = evaluate(bridge, config_sha="config", power=lambda: "ac", checkpoint=lambda x: saved.append(deepcopy(x)))
            assert result["status"] == "stopped-observation" and sim.requests == 1
            assert result["trials"][0]["reading"] and result["trials"][0]["cleanup"]["process_gone"] is True
    assert sim.starts == sim.stops == 1


def frozen_engine_fixture(tmp_path: Path) -> tuple[Any, Path]:
    r = runner()
    directory = tmp_path / "tools/llama.cpp"
    directory.mkdir(parents=True)
    source = tmp_path / "source.py"
    source.write_bytes(b"synthetic source")
    files = {}
    for name in ENGINE_FILES:
        path = directory / name
        path.write_bytes(("synthetic, never executable: " + name).encode())
        files[path.relative_to(tmp_path).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = r.FrozenInputs(tmp_path, {"engine_files": files,
                           "python_files": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}},
                           environment=lambda: {"APPDATA": "", "PROGRAMDATA": ""}, platform="win32")
    return inputs, directory


@pytest.mark.parametrize("when", ["construction", "first-start"])
@pytest.mark.parametrize("same_bytes", [False, True])
def test_configured_executable_must_be_the_frozen_file_before_backend_construction(
    tmp_path: Path, when: str, same_bytes: bool,
) -> None:
    from tests.resident_helpers import synthetic_model

    r = runner()
    inputs, canonical = frozen_engine_fixture(tmp_path)
    other = tmp_path / "different-toolchain"
    other.mkdir()
    alternate = other / "llama-server.exe"
    alternate.write_bytes((canonical / "llama-server.exe").read_bytes() if same_bytes else b"unreviewed fixture")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not parsed or loaded")
    settings = Settings().model_copy(update={"llama_cpp_dir": other if when == "construction" else canonical})
    calls: list[Any] = []

    class Backend:
        last_command = None
        def start(self, *args: Any) -> None: calls.append("start")

    def factory(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return Backend()

    with patch("subprocess.Popen", side_effect=AssertionError("no process permitted")) as launches, patch("inferenceos.backends.resident_server.LlamaServerResidentBackend", side_effect=factory):
        with pytest.raises(r.Refused):
            bridge = r.RealLookupEngine(inputs=inputs, settings=settings, model=synthetic_model(str(model)),
                                       configuration=RuntimeConfiguration(context_length=4096))
            settings.llama_cpp_dir = other
            bridge.start(CONTROL)
        launches.assert_not_called()
    assert not calls
    inputs.check(full=True)  # Canonical hashes alone do not detect this wrong selection.


@pytest.mark.parametrize("fault", ["none", "executable", "deleted", "selection", "library"])
def test_production_factory_binds_checked_path_and_rechecks_before_reusing_backend(
    fault: str, tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    from inferenceos.backends.llama_server_toolchain import build_server_command
    from inferenceos.backends.resident_server import _options_from
    from inferenceos.tune.reuse_evidence import file_identity
    from tests.resident_helpers import synthetic_model

    r = runner()
    inputs, directory = frozen_engine_fixture(tmp_path)
    settings = Settings().model_copy(update={"llama_cpp_dir": directory})
    model = tmp_path / "model.gguf"
    model.write_bytes(b"not parsed or loaded")
    made: list[Any] = []

    class Backend:
        unconfirmed_pid = None
        undeleted_key_files: tuple[Path, ...] = ()
        api_key_file = None
        def __init__(self, *, server: Path, **kwargs: Any) -> None:
            self.server = server
            self.pid: int | None = None
            self.identity: Any = None
            self.params: list[Any] = []
            self.last_command: list[str] = []
            made.append(self)
        def start(self, profile: Any, params: Any) -> None:
            self.params.append(params)
            self.pid = 12345 + len(self.params)
            self.identity = SimpleNamespace(pid=self.pid, create_time=float(self.pid))
            self.last_command = build_server_command(self.server, _options_from(Path(profile.path), params),
                host="127.0.0.1", port=1234, api_key_file=tmp_path / "never-created")
        def server_startup_log(self) -> str:
            if self.params[-1].lookup_decoding.algorithm == "none":
                return "no implementations specified\nmodel loaded\n"
            return ("adding speculative implementation 'ngram-simple'\nsize_n=12, size_m=16, min_hits=1\n"
                    "speculative decoding context initialized\nmodel loaded\n")
        def startup_log_snapshot(self) -> dict[str, Any]:
            text = self.server_startup_log()
            return {"available": True, "text": text, "truncated": False, "reader_errors": [],
                    "generation_current": True, "lines": len(text.splitlines()), "line_limit": 5000, "dropped_lines": 0}
        def is_active(self) -> bool: return self.pid is not None
        def stop(self) -> Any:
            self.pid = self.identity = None
            return SimpleNamespace(cleanup_complete=True, credential_removed=True)

    def image(pid: int) -> dict[str, Any]:
        return {"pid": pid, "create_time_s": float(pid), "status": "observed", "problems": [],
                "files": [file_identity(directory / name) for name in ENGINE_FILES],
                "executable": str(directory / "llama-server.exe")}

    with patch("subprocess.Popen", side_effect=AssertionError("all process creation blocked")) as launches, patch("inferenceos.backends.resident_server.LlamaServerResidentBackend", Backend), patch("inferenceos.tune.reuse_evidence.process_image", side_effect=image):
        bridge = r.RealLookupEngine(inputs=inputs, settings=settings, model=synthetic_model(str(model)),
                                   configuration=RuntimeConfiguration(context_length=4096),
                                   capture_load_evidence=True, run_id="synthetic-only")
        bridge.start(CONTROL)
        assert bridge.stop().clean and bridge.load_problems() == []
        if fault == "executable": (directory / "llama-server.exe").write_bytes(b"changed")
        elif fault == "deleted": (directory / "llama-server.exe").unlink()
        elif fault == "library": (directory / "llama.dll").write_bytes(b"changed")
        elif fault == "selection":
            other = tmp_path / "different"
            other.mkdir()
            (other / "llama-server.exe").write_bytes((directory / "llama-server.exe").read_bytes())
            settings.llama_cpp_dir = other
        if fault != "none":
            with pytest.raises((r.Refused, OSError)): bridge.start(LOOKUP)
            assert len(made[0].params) == 1
        else:
            bridge.start(LOOKUP)
            assert bridge.stop().clean and bridge.load_problems() == []
            assert len(made[0].params) == 2
        assert len(made) == 1 and made[0].server == directory / "llama-server.exe"
        assert bridge.held is None
        launches.assert_not_called()


@pytest.mark.parametrize("fault", ["image-missing", "not-object", "files-missing", "entry-missing", "not-executable",
                                   "pin-missing", "pin-invalid", "pin-ambiguous", "file-duplicate", "optional-hash"])
def test_live_and_offline_identity_checker_has_one_fail_closed_policy(fault: str) -> None:
    from inferenceos.tune.lookup_audit import engine_identity_problems

    record, rules, sha = synthetic_record()
    load = record["load_evidence"]["loads"][0]
    snapshot = load["snapshots"][0]
    image = snapshot["image"]
    if fault == "image-missing": snapshot["image"] = None
    elif fault == "not-object": image["files"][0] = "not a file identity"
    elif fault == "files-missing": image["files"] = None
    elif fault == "entry-missing": image["files"].append({})
    elif fault == "not-executable": image["executable"] = "C:/other/not-the-engine.exe"
    elif fault == "pin-missing": rules["engine_files"].pop("tools/llama.dll")
    elif fault == "pin-invalid": rules["engine_files"]["tools/llama.dll"] = "not a digest"
    elif fault == "pin-ambiguous": rules["engine_files"]["different/LLAMA.DLL"] = "e" * 64
    elif fault == "file-duplicate": image["files"].append(deepcopy(image["files"][0]))
    elif fault == "optional-hash":
        rules["engine_files"]["tools/ggml-cpu.dll"] = "e" * 64
        image["files"].append({"path": "C:/synthetic-engine/ggml-cpu.dll", "sha256": "0" * 64})
    assert engine_identity_problems(snapshot.get("image"), rules["engine_files"])
    # Malformed legacy capture shapes are still rejected at the shared gate. The
    # full saved-record parser expects object-shaped captures; it is not a new schema.
    if fault not in ("image-missing", "not-object", "files-missing", "entry-missing"):
        checked = audit(record, rules, sha)
        assert not checked["observations_complete"] and not checked["workload_findings"]


def test_identity_checker_does_not_require_frozen_windows_system_library_bytes() -> None:
    from inferenceos.tune.lookup_audit import engine_identity_problems

    record, rules, _ = synthetic_record()
    image = record["load_evidence"]["loads"][0]["snapshots"][0]["image"]
    image["files"].append({"path": "C:/Windows/System32/kernel32.dll", "sha256": "f" * 64})
    assert engine_identity_problems(image, rules["engine_files"]) == []
