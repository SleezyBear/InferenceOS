"""Frozen model-free lookup development evaluation. No tune, saves or default changes.

Real runs require reviewed rules plus their SHA256; simulation needs neither model
nor engine. Exit 0=complete, 2=incomplete/stopped, 3=error, 4=refused, 6=changed input
or data, 7=publication failed. Ctrl+C is checkpointed and re-raised after cleanup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inferenceos.config import Settings
from inferenceos.interfaces.backend_adapter import GenerationRequest, LookupDecodingParams
from inferenceos.models.identity import model_file_signature
from inferenceos.models.loader import load_model_profile
from inferenceos.storage import atomic_write_text
from inferenceos.tune.lookup_decoding import (
    VERSION, BLOCKS, TOKENS, SEED, LOOKUP_LENGTH, DRAFT_LENGTH, ARMS,
    LookupCase, build_cases, digest, evaluate, summarize, activation_problems,
)
from inferenceos.tune.prompt_reuse import EngineVariant
from inferenceos.tune.reuse_evidence import EVIDENCE_VERSION, capture_problems, file_identity
from inferenceos.tune.lookup_sim import LookupSimulator
from inferenceos.tune.lookup_audit import engine_identity_problems
from evaluate_prompt_reuse import RealReuseEngine, _fingerprint, _norm, _setup, output_problem, power_state

PROTOCOL = "lookup-study-1"


class Refused(ValueError):
    pass


def engine_environment() -> dict[str, str]:
    return {name: os.environ.get(name, "") for name in ("APPDATA", "PROGRAMDATA")}


def check_ambient_config(environment: dict[str, str], *, platform: str) -> dict[str, Any]:
    """b11139 reads these Windows ini files BEFORE CLI arguments. Never edit them."""
    if platform != "win32":
        raise Refused("this real lookup protocol is registered for Windows only")
    paths = [Path(value) / "llama.cpp/config.ini" for value in environment.values() if value]
    for path in paths:
        try:
            path.stat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise Refused(f"ambient engine configuration cannot be checked: {path}") from exc
        raise Refused(f"ambient engine configuration exists: {path}; no engine is constructed")
    return {"platform": platform, "checked": True, "environment": dict(environment),
            "absent_paths": [str(p) for p in paths]}


def parameters() -> dict[str, Any]:
    return {"blocks": BLOCKS, "tokens": TOKENS, "seed": SEED, "context_length": 4096,
            "ignore_eos": False,
            "lookup_length": LOOKUP_LENGTH, "draft_length": DRAFT_LENGTH,
            "case_sha256": {c.case_id: c.sha256 for c in build_cases()}}


class FrozenInputs:
    """Hash frozen files before construction, then bracket loads by generation.

    Bounded path/metadata observations, not OS locking or resident-memory attestation.
    """
    def __init__(self, root: Path, rules: dict[str, Any], *, rules_path: Path | None = None,
                 rules_sha: str | None = None, environment: Callable[[], dict[str, str]] | None = None,
                 platform: str | None = None) -> None:
        self.root = root
        self.signatures: dict[Path, tuple[str, int, int, int]] = {}
        self.hashes: dict[str, str] = {}
        self._rules: tuple[Path, tuple[str, int, int, int], str] | None = None
        self._environment = environment or engine_environment
        self._platform = platform or sys.platform
        self.ambient = check_ambient_config(self._environment(), platform=self._platform)
        for group in ("python_files", "engine_files"):
            items = rules.get(group)
            if not isinstance(items, dict) or not items:
                raise Refused(f"missing frozen {group}")
            for name, expected in items.items():
                if not isinstance(name, str) or Path(name).as_posix() != name:
                    raise Refused("frozen paths must use canonical forward slashes")
                path = (root / name).resolve()
                if not path.is_relative_to(root.resolve()) or not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
                    raise Refused("invalid frozen input path/hash")
                before = model_file_signature(path)
                actual = file_identity(path)
                if actual["sha256"] != expected or model_file_signature(path) != before:
                    raise Refused(f"frozen input differs: {name}")
                self.signatures[path] = before
                self.hashes[name] = expected
        if rules_path is not None:
            signature = model_file_signature(rules_path)
            if file_identity(rules_path)["sha256"] != rules_sha or model_file_signature(rules_path) != signature:
                raise Refused("governing rules changed before construction")
            self._rules = (rules_path, signature, str(rules_sha))

    def check(self, *, full: bool = False) -> None:
        if check_ambient_config(self._environment(), platform=self._platform) != self.ambient:
            raise Refused("ambient engine configuration environment changed")
        if self._rules is not None:
            path, signature, expected = self._rules
            if model_file_signature(path) != signature or (full and file_identity(path)["sha256"] != expected):
                raise Refused("governing rules changed during study")
        for path, signature in self.signatures.items():
            if model_file_signature(path) != signature:
                raise Refused(f"study input generation changed: {path}")
            if full and file_identity(path)["sha256"] != self.hashes[path.relative_to(self.root.resolve()).as_posix()]:
                raise Refused(f"study input bytes changed: {path}")

    def executable(self, directory: Path) -> Path:
        """Bind actual configurable selection to the one frozen executable before use."""
        from inferenceos.backends.llama_server_toolchain import resolve_server_executable

        self.check()
        pins = [(name, expected) for name, expected in self.hashes.items()
                if Path(name).name.lower() == "llama-server.exe"]
        if len(pins) != 1:
            raise Refused("one frozen engine executable required before construction")
        name, expected = pins[0]
        registered = (self.root / name).resolve(strict=True)
        selected = resolve_server_executable(directory, platform="win32").resolve(strict=True)
        if selected != registered:
            raise Refused("selected engine executable is not the frozen path; no backend constructed")
        signature = model_file_signature(selected)
        if (signature != self.signatures[registered] or file_identity(selected)["sha256"] != expected
                or model_file_signature(selected) != signature):
            raise Refused("selected engine executable differs from frozen input; no start")
        return selected


class RealLookupEngine(RealReuseEngine):
    def __init__(self, *, inputs: FrozenInputs, **kwargs: Any) -> None:
        self.inputs = inputs
        self._selection_settings: Any = kwargs.get("settings")
        self._selected_server: Path | None = None
        if kwargs.get("backend_factory") is None and self._selection_settings is not None:
            from inferenceos.backends.resident_server import LlamaServerResidentBackend

            server = inputs.executable(self._selection_settings.llama_cpp_dir)
            self._selected_server = server

            def factory() -> Any:
                # Bind the checked path once. The inherited factory must not resolve
                # a configurable toolchain independently after frozen validation.
                return LlamaServerResidentBackend(
                    server=server,
                    readiness_timeout_s=self._selection_settings.resident_readiness_timeout_s,
                    request_timeout_s=self._selection_settings.resident_request_timeout_s)

            kwargs["backend_factory"] = factory
        super().__init__(**kwargs)

    def _capture_load(self, phase: str) -> None:
        if phase == "ready" and self.capture_load_evidence:
            # Readers can lag the health response. Wait only for existing log text,
            # before the one ready snapshot; no request, probe or measurement added.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                text = self.startup_log()
                if isinstance(text, str) and "model loaded" in text:
                    break
                time.sleep(.01)
        super()._capture_load(phase)

    def start(self, variant: EngineVariant) -> None:
        super().start(variant)
        if self.capture_load_evidence:
            load = self.load_evidence[-1] if self.load_evidence else {}
            snapshots = load.get("snapshots", [])
            ready = [s for s in snapshots if s.get("phase") == "ready"]
            if len(ready) != 1:
                raise Refused("one complete ready observation required before measurement")
            snapshot = ready[0]
            log = snapshot.get("startup", {})
            if (load.get("capture_errors") or snapshot.get("process_current") is not True
                    or snapshot.get("image", {}).get("status") != "observed"
                    or log.get("generation_current") is not True or log.get("available") is not True
                    or log.get("truncated") is not False or log.get("reader_errors")):
                raise Refused("ready process/log observations incomplete; no measured request")
            problems = activation_problems(log.get("text"), variant.variant_id)
            problems.extend(engine_identity_problems(snapshot.get("image"), self.inputs.hashes))
            if "using config file:" in str(log.get("text", "")):
                problems.append("engine loaded ambient configuration")
            if problems:
                raise Refused("; ".join(problems) + "; no measured request")

    def _params_for(self, variant: EngineVariant) -> Any:
        if variant.variant_id not in ARMS or variant.cache_reuse is not None or variant.swa_full:
            raise Refused("unknown lookup-study variant")
        self.inputs.check()
        if self._selected_server is not None:
            if self.inputs.executable(self._selection_settings.llama_cpp_dir) != self._selected_server:
                raise Refused("selected engine executable changed; no further start")
        base = super()._params_for(variant)
        from inferenceos.backends.engine_env import MANAGED_SWITCHES, parse_switches
        if set(parse_switches(base.engine_switches)) - MANAGED_SWITCHES:
            raise Refused("saved setup contains non-registered environment overrides")
        group = LookupDecodingParams(algorithm="none" if variant.variant_id == "control" else "ngram-simple",
                                     lookup_length=LOOKUP_LENGTH, draft_length=DRAFT_LENGTH)
        return base.model_copy(update={"lookup_decoding": group})

    def load_problems(self) -> list[str]:
        """Inspect the just-released load; no request, retry or process control."""
        if not self.capture_load_evidence:
            return []
        if not self.load_evidence:
            return ["load observations absent"]
        load = self.load_evidence[-1]
        problems = capture_problems([load])
        if load.get("state") != "released":
            problems.append("captured load release not confirmed")
        arm = load.get("variant", {}).get("variant_id")
        for snapshot in load.get("snapshots", []):
            log = snapshot.get("startup", {})
            if log.get("generation_current") is not True:
                problems.append("captured log generation not current")
            problems.extend(activation_problems(log.get("text"), str(arm)))
            problems.extend(engine_identity_problems(snapshot.get("image"), self.inputs.hashes))
        return list(dict.fromkeys(problems))

    def request(self, case: LookupCase) -> dict[str, Any]:
        from inferenceos.backends.resident_server import ResidentGenerationProtocolError

        tokens = self.tokenize(case.prompt)
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise Refused("tokenizer did not return a complete prompt")
        if len(tokens) + TOKENS + DRAFT_LENGTH >= self._configuration.context_length:
            raise Refused("fixed work does not fit the unchanged context")
        try:
            result = self._backend.generate(GenerationRequest(
                prompt=case.prompt, n_predict=TOKENS, seed=SEED, temperature=0,
                ignore_eos=False, cache_prompt=False, return_tokens=True))
        except ResidentGenerationProtocolError as exc:
            result = exc.result
        counters = dict(self._backend.last_response_counters)
        return {"text": result.text, "token_ids": counters.get("generated_token_ids"),
                "prompt_tokens": len(tokens), "prompt_ids_sha256": digest(tokens),
                "total_s": result.total_s, "ttft_s": result.ttft_s,
                "peak_rss_bytes": result.peak_rss_bytes, "counters": counters,
                "load_evidence_id": self.load_evidence_id}


def read_rules(path: Path, expected: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected.lower():
        raise Refused("frozen rules SHA256 differs")
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        names: set[str] = set()
        for key, value in items:
            if key.casefold() in names:
                raise Refused("duplicate/case-colliding frozen rule name")
            names.add(key.casefold())
            result[key] = value
        return result
    def invalid_constant(value: str) -> Any:
        raise Refused(f"non-JSON numeric constant: {value}")
    try:
        rules = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (ValueError, TypeError) as exc:
        raise Refused("governing rules are not valid JSON") from exc
    if (not isinstance(rules, dict) or rules.get("protocol") != PROTOCOL
            or rules.get("version") != VERSION or rules.get("parameters") != parameters()
            or rules.get("require_saved_setup") is not True):
        raise Refused("rules protocol/version/plan or saved-setup requirement differs")
    models = rules.get("models")
    if not isinstance(models, dict) or len(models) != 2:
        raise Refused("exactly two pre-registered model identities required")
    for name, value in models.items():
        if not isinstance(name, str) or not isinstance(value, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256"))):
            raise Refused("frozen model identity malformed")
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get("configuration_sha256"))):
            raise Refused("frozen saved configuration identity missing or malformed")
        if Path(name).is_absolute() or ".." in Path(name).parts or Path(name).as_posix() != name:
            raise Refused("frozen model paths must be canonical relative paths")
    order = rules.get("model_order")
    if not isinstance(order, list) or len(order) != 2 or len(set(order)) != 2 or set(order) != set(models):
        raise Refused("frozen two-model order missing or contradictory")
    return rules, actual


def _write(path: Path, record: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--preregistration-sha256")
    args = parser.parse_args(argv)
    settings = Settings()
    protected = [Path(settings.data_dir), ROOT / "src", ROOT / "tests", ROOT / "scripts", ROOT / "tools"]
    if args.model is not None:
        protected.append(args.model)
    if args.preregistration is not None:
        protected.append(args.preregistration)
    problem = output_problem(args.out, protected=protected)
    if problem or args.out.exists() or args.out.is_symlink():
        print(f"refused: {problem or 'output already exists'}")
        return 4
    record: dict[str, Any] = {"protocol": PROTOCOL, "version": VERSION,
                              "run_id": args.run_id or str(uuid.uuid4()), "simulated": bool(args.simulate),
                              "started_at": datetime.now(timezone.utc).isoformat(), "parameters": parameters(),
                              "status": "preparing", "trials": [], "stop_reason": ""}
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(record, handle)
    except OSError as exc:
        print(f"refused: cannot reserve output: {exc}")
        return 4
    engine: Any = None
    inputs: FrozenInputs | None = None
    model_before: dict[str, Any] | None = None
    data_dir = None if args.simulate else Path(settings.data_dir)
    data_evidence: dict[str, Any] | None = None
    interrupted: KeyboardInterrupt | None = None
    code = 3
    announced = 0

    def note_interrupt(exc: KeyboardInterrupt, stage: str) -> None:
        nonlocal interrupted
        interrupted = exc
        record.update(status="interrupted", stop_reason=f"KeyboardInterrupt during {stage}",
                      interruption={"stage": stage, "type": type(exc).__name__, "message": str(exc)})

    def checkpoint(partial: dict[str, Any]) -> None:
        nonlocal announced
        record.update(partial)
        # Only final publication, after ownership/input/data checks, may say done.
        published = dict(record)
        if published.get("status") == "done":
            published["status"] = "running"
        _write(args.out, published)
        rows = partial.get("trials", [])
        if len(rows) > announced:
            announced = len(rows)
            row = rows[-1]
            print(f"trial {announced}/{partial.get('planned_trials', 40)}: {row.get('case_id')} "
                  f"block {int(row.get('block', 0)) + 1} {row.get('arm')}; {row.get('stage')}", flush=True)

    try:
        rules = None
        if args.preregistration is not None or args.preregistration_sha256 is not None:
            if args.preregistration is None or args.preregistration_sha256 is None:
                raise Refused("both rules arguments are required")
            rules, rules_sha = read_rules(args.preregistration, args.preregistration_sha256)
            record["preregistration"] = {"file": str(args.preregistration), "sha256": rules_sha}
        if args.simulate:
            record["configuration"] = {"simulated": True, "context_length": 4096}
            engine = LookupSimulator()
            power = lambda: "simulated"
        else:
            if rules is None or args.model is None:
                raise Refused("real evaluation requires model and reviewed frozen rules")
            path = args.model.resolve(strict=True)
            matches = [v for name, v in rules["models"].items() if _norm(ROOT / name) == _norm(path)]
            if len(matches) != 1:
                raise Refused("model path is not the registered model")
            signature = model_file_signature(path)
            model_before = file_identity(path)
            record["model"] = str(path)
            record["model_file_identity"] = model_before
            if model_before["sha256"] != matches[0]["sha256"]:
                raise Refused("model does not match frozen identity; no engine constructed")
            data_evidence = {"version": "file-fingerprint-1",
                             "root": str(data_dir.resolve()) if data_dir is not None else "",
                             "before": _fingerprint(data_dir) if data_dir is not None else {}, "after": None,
                             "before_complete": True, "after_complete": False,
                             "before_captured_at": datetime.now(timezone.utc).isoformat()}
            record["data_folder_evidence"] = data_evidence
            configuration, source = _setup(settings, path, 4096, None, identity=model_before, require_saved=True)
            if digest(configuration.model_dump(mode="json")) != matches[0]["configuration_sha256"]:
                raise Refused("saved configuration differs from the frozen baseline; no engine constructed")
            if model_file_signature(path) != signature:
                raise Refused("model generation changed during setup validation")
            inputs = FrozenInputs(ROOT, rules, rules_path=args.preregistration,
                                  rules_sha=record["preregistration"]["sha256"])
            model = load_model_profile(path)
            if model_file_signature(path) != signature:
                raise Refused("model generation changed during metadata preparation")
            record.update(configuration=configuration.model_dump(mode="json"), setup_source=source,
                          model_preflight={"expected_sha256": matches[0]["sha256"], "pin_verified": True,
                                           "saved_setup_verified": True, "configuration_verified": True},
                          data_folder_evidence=data_evidence)
            record["ambient_config"] = inputs.ambient
            engine = RealLookupEngine(settings=settings, model=model, configuration=configuration,
                                      capture_load_evidence=True, run_id=record["run_id"],
                                      model_generation=signature, inputs=inputs)
            power = power_state
        record["configuration_sha256"] = digest(record["configuration"])
        checkpoint({"status": "running"})
        result = evaluate(engine, config_sha=record["configuration_sha256"], power=power,
                          checkpoint=checkpoint, simulated=args.simulate)
        record.update(result)
        code = 0 if result["status"] == "done" else 2
    except KeyboardInterrupt as exc:
        note_interrupt(exc, "evaluation/preparation")
    except (Refused, LookupError, FileNotFoundError) as exc:
        record.update(status="refused", stop_reason=f"{type(exc).__name__}: {exc}")
        code = 4
    except Exception as exc:
        record.update(status="error", stop_reason=f"{type(exc).__name__}: {exc}")
        code = 3
    finally:
        if engine is not None:
            try:
                retry = engine.retry_held(propagate_interrupt=True)
                if retry is not None:
                    record.update(held_engine_retry=retry)
                    if record.get("status") == "done":
                        record.update(status="stopped-cleanup", stop_reason="a held engine required cleanup retry")
                        code = 2
            except KeyboardInterrupt as exc:
                note_interrupt(exc, "held cleanup")
            except Exception as exc:
                record.update(status="stopped-cleanup", stop_reason=f"final cleanup: {type(exc).__name__}: {exc}")
                code = 2
            try:
                loads = getattr(engine, "load_evidence", [])
                record["engine_commands"] = getattr(engine, "commands", [])
                capture_errors = [] if args.simulate else capture_problems(loads)
                record["load_evidence"] = {"version": EVIDENCE_VERSION, "simulated": bool(args.simulate),
                                           "loads": loads, "problems": capture_errors}
                if not args.simulate and (not loads or capture_errors) and record.get("status") == "done":
                    record.update(status="incomplete", stop_reason="per-load observations incomplete")
                    code = 2
            except KeyboardInterrupt as exc:
                note_interrupt(exc, "capture accounting")
            except Exception as exc:
                record.update(status="error", stop_reason=f"capture accounting: {type(exc).__name__}: {exc}")
                code = 3
        try:
            if inputs is not None:
                inputs.check(full=True)
                record["frozen_inputs_unchanged"] = True
            if model_before is not None:
                record["model_file_unchanged"] = file_identity(Path(model_before["path"]))["sha256"] == model_before["sha256"]
                if record["model_file_unchanged"] is not True:
                    raise Refused("model content changed")
            if data_evidence is not None and data_dir is not None:
                after = _fingerprint(data_dir)
                data_evidence["after"] = after
                data_evidence["after_complete"] = True
                data_evidence["after_captured_at"] = datetime.now(timezone.utc).isoformat()
                record["data_folder_unchanged"] = after == data_evidence["before"]
                if after != data_evidence["before"]:
                    raise Refused("data-folder map changed")
        except KeyboardInterrupt as exc:
            note_interrupt(exc, "final input/data checks")
        except Exception as exc:
            record.update(status="input-or-data-changed", stop_reason=f"{type(exc).__name__}: {exc}")
            code = 6
        if interrupted is not None:
            record["status"] = "interrupted"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            if record.get("configuration_sha256"):
                record["summary"] = summarize(record["trials"], record["configuration_sha256"], simulated=args.simulate)
        except KeyboardInterrupt as exc:
            note_interrupt(exc, "final summary")
        except Exception as exc:
            record.update(status="error", stop_reason=f"summary: {type(exc).__name__}: {exc}")
            code = 3
        try:
            _write(args.out, record)
        except KeyboardInterrupt as exc:
            note_interrupt(exc, "publication")
            # A signal can arrive just after atomic replacement. Best effort only:
            # update that completed write to interrupted, without retrying any work.
            try:
                _write(args.out, record)
            except (OSError, ValueError, TypeError, KeyboardInterrupt):
                print("interrupted result could not be republished; wrapper exit is not success")
        except (OSError, ValueError, TypeError) as exc:
            print(f"publication failed: {type(exc).__name__}: {exc}")
            code = 7
    if interrupted is not None:
        raise interrupted
    print(f"status: {record['status']} (exit {code}); results: {args.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
