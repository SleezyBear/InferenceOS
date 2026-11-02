"""Stage-A fit-only collector. Real dispatch requires separately reviewed frozen inputs.

No generation command is implemented. --simulate uses fabricated observations only.
Exit 0=collection complete (not accepted/qualified), 2=stopped/incomplete, 3=error,
4=refused, 6=changed inputs/data, 7=publication failed. Interrupts are checkpointed
and re-raised after the existing backend owner has attempted release.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inferenceos.config import Settings
from inferenceos.experiment.restoration import runtime_params_from
from inferenceos.interfaces.backend_adapter import LookupDecodingParams
from inferenceos.models.identity import model_file_signature
from inferenceos.models.loader import load_model_profile
from inferenceos.storage import atomic_write_text
from inferenceos.tune.lookup_corpus import digest
from inferenceos.tune.lookup_fit import FitManifest, FitModel, FitRefused, collect_fit, parse_manifest, strict_json, preservation_problems
from inferenceos.tune.lookup_fit_io import InspectionTimeout, bounded_read, bounded_file_bytes, inspect_json
from inferenceos.tune.lookup_fit_observers import observe_effects
from inferenceos.tune.reuse_evidence import file_identity, process_image
from evaluate_lookup_decoding import FrozenInputs
from evaluate_prompt_reuse import _fingerprint, _setup, output_problem, power_state


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_pinned(path: Path, expected: str) -> bytes:
    raw = bounded_file_bytes(path)
    if sha(raw) != expected:
        raise FitRefused("frozen JSON/document digest differs")
    return raw


def local_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve(strict=True).is_relative_to(root.resolve()):
        raise FitRefused("frozen input escapes root")
    for p in (path, *path.parents):
        if p == root.parent:
            break
        if p.is_symlink() or getattr(p,"is_junction",lambda:False)():
            raise FitRefused("frozen input/root links are not supported")
    return path.resolve(strict=True)


def machine_inventory() -> dict[str, Any]:
    from inferenceos.native import get_platform
    native = get_platform()
    if not native.is_windows():
        raise FitRefused("live Stage A is Windows-only")
    power = native.power_status()
    memory = native.global_memory_status()
    if power is None or power.ac_line is not True or memory is None:
        raise FitRefused("live AC and physical memory observations required")
    output = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True,
                            timeout=10, check=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    guids = re.findall(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", output)
    if len(guids) != 1:
        raise FitRefused("one active power-plan GUID required")
    return dict(hostname=socket.gethostname(), cim=asdict(native.collect_cim_snapshot()),
                cpuid=native.cpuid_features(), os_flags=native.os_cpu_features(), memory=asdict(memory),
                vulkan=[asdict(d) for d in native.vulkan_enumerate()], power_plan=guids[0].lower(), power_source="ac")


class Preflight:
    """Read-only actual-model/setup/toolchain validation before backend construction."""
    def __init__(self, root: Path, settings: Settings, manifest: FitManifest, rules: Path,
                 rules_sha: str, proposal: Path, proposal_sha: str) -> None:
        self.root, self.settings, self.manifest = root, settings, manifest
        if (settings.project_root.resolve() != root.resolve() or settings.data_dir.resolve() != (root / "data").resolve()
                or settings.data_dir.is_symlink() or getattr(settings.data_dir,"is_junction",lambda:False)()):
            raise FitRefused("canonical repository/data root required")
        paths = set(manifest.python_files) | set(manifest.engine_files)
        for relative in paths:
            local_path(root, relative)
        source_paths = {p.relative_to(root).as_posix() for folder in ("src", "scripts", "tests")
                  for p in (root / folder).rglob("*.py")}
        if source_paths != set(manifest.python_files):
            raise FitRefused("complete current Python source population required")
        self.inputs = FrozenInputs(root, manifest.model_dump(mode="json"), rules_path=rules, rules_sha=rules_sha)
        self.selected = self.inputs.executable(settings.llama_cpp_dir)
        self.documents = [(rules, rules_sha, model_file_signature(rules)),
                          (proposal, proposal_sha, model_file_signature(proposal)),
                          (root / "docs/lookup-qualification-protocol.md", manifest.protocol_document_sha256,
                           model_file_signature(root / "docs/lookup-qualification-protocol.md"))]
        for path, expected, _ in self.documents:
            read_pinned(path, expected)
        self.before = _fingerprint(root / "data")
        self.saved_path = local_path(root, "data/saved_setups.json")
        self.saved_bytes = self.saved_path.read_bytes()
        self.saved_signature = model_file_signature(self.saved_path)
        if self.before.get('saved_setups.json',{}).get('sha256') != sha(self.saved_bytes):
            raise FitRefused('saved setups changed during initial capture')
        self.models: dict[str, Any] = {}
        self.signatures: dict[Path, Any] = {}
        self.receipts = []
        for pin in manifest.models:
            path = local_path(root, pin.path)
            signature = model_file_signature(path)
            identity = file_identity(path)
            if identity["sha256"] != pin.sha256:
                raise FitRefused("model identity differs; no backend constructed")
            configuration, source = _setup(settings, path, 4096, None, identity=identity, require_saved=True)
            if (configuration.model_dump(mode="json") != pin.configuration.model_dump(mode="json")
                    or source != pin.saved_setup_source or model_file_signature(path) != signature):
                raise FitRefused("saved setup/base identity differs; no backend constructed")
            if any(getattr(configuration, k) is not None for k in ("engine_switches", "placement_overrides", "kv_cache_type")):
                raise FitRefused("this Stage-A proposal has no aliases or custom overrides")
            self.signatures[path] = signature
            self.models[pin.label] = load_model_profile(path)
            self.receipts.append(dict(label=pin.label, model_file=identity, configuration=configuration.model_dump(mode="json"),
                                      source=source))
        self.environment = dict(os.environ)
        from inferenceos.backends.engine_env import process_environment, ENGINE_PREFIXES
        effective_environment = process_environment(self.environment, None)
        if any(k.upper().startswith(ENGINE_PREFIXES) for k in effective_environment):
            raise FitRefused("unregistered inherited engine environment override")
        self.check(full=True)
        problems = preservation_problems(self.preservation(),manifest)
        if problems:
            raise FitRefused('; '.join(problems)+'; no backend constructed')

    def check(self, *, full: bool = False) -> None:
        self.inputs.check(full=full)
        current_sources = {p.relative_to(self.root).as_posix() for folder in ('src','scripts','tests')
                           for p in (self.root / folder).rglob('*.py')}
        if current_sources != set(self.manifest.python_files):
            raise FitRefused('Python source population changed')
        from inferenceos.backends.llama_server_toolchain import resolve_server_executable
        if resolve_server_executable(self.settings.llama_cpp_dir,platform="win32").resolve(strict=True) != self.selected:
            raise FitRefused("selected executable changed")
        if hasattr(self, "environment") and dict(os.environ) != self.environment:
            raise FitRefused("parent environment changed")
        for path, expected, signature in self.documents:
            if model_file_signature(path) != signature or (full and sha(path.read_bytes()) != expected):
                raise FitRefused("governing document changed")
        for path, signature in self.signatures.items():
            if model_file_signature(path) != signature:
                raise FitRefused("model generation changed")
            if full and file_identity(path)["sha256"] != next(m.sha256 for m in self.manifest.models if local_path(self.root, m.path) == path):
                raise FitRefused("model bytes changed")
        if model_file_signature(self.saved_path) != self.saved_signature or self.saved_path.read_bytes() != self.saved_bytes:
            raise FitRefused("saved setups changed")

    def preservation(self) -> dict[str, Any]:
        self.check(full=True)
        after = _fingerprint(self.root / "data")
        if after != self.before:
            raise FitRefused("protected data changed")
        return dict(root=str(self.root.resolve()), data_before=self.before, data_after=after,
                    saved_file=dict(bytes_base64=base64.b64encode(self.saved_bytes).decode("ascii"),
                                    sha256=sha(self.saved_bytes), path=str(self.saved_path)),
                    models=self.receipts, selected_executable=str(self.selected),
                    frozen_hashes={**self.manifest.python_files, **self.manifest.engine_files},
                    environment_sha256=digest(self.environment),
                    ambient=self.inputs.ambient)


class LiveFitEngine:
    """Exactly one native backend owner; all observational workers are read-only."""
    def __init__(self, preflight: Preflight, *, backend_factory: Any = None, inventory: Any = machine_inventory) -> None:
        from inferenceos.backends.resident_server import LlamaServerResidentBackend
        self.preflight = preflight
        self.backend = (backend_factory or (lambda: LlamaServerResidentBackend(
            server=preflight.selected, readiness_timeout_s=120, request_timeout_s=30)))()
        self.inventory = inventory
        self.cancelled = threading.Event()
        self.owed = False

    def start(self, model: FitModel, arm: str, *, deadline: float | None = None) -> None:
        """Use one supplied absolute deadline for the complete Stage-B start.

        Without a deadline, retain Stage A's separately bounded checks and start.
        The optional deadline tightens that existing owner; it does not transfer
        process cancellation or cleanup to an additional owner.
        """
        if self.cancelled.is_set() or self.owed or arm not in ("control", "lookup"):
            raise FitRefused("engine admission closed")
        if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
            raise ValueError("finite absolute start deadline required")
        shared_deadline = deadline

        def remaining(cap: float) -> float:
            if shared_deadline is None:
                return cap  # The accepted Stage-A call retains its old bounds.
            if self.cancelled.is_set():
                raise FitRefused("engine start cancelled before dispatch")
            left = shared_deadline-time.monotonic()
            if left <= 0:
                raise InspectionTimeout("complete engine start deadline expired")
            return min(cap, left)

        try:
            bounded_read(lambda: self.preflight.inputs.executable(self.preflight.settings.llama_cpp_dir), timeout_s=remaining(30))
            bounded_read(self.preflight.check,timeout_s=remaining(30))
            params = runtime_params_from(model.configuration).model_copy(update={
                "log_verbosity":4, "lookup_decoding":LookupDecodingParams(algorithm="none" if arm == "control" else "ngram-simple",
                                                                         lookup_length=12, draft_length=16)})
            allowance = remaining(120)
            self.owed = True
            start_deadline = shared_deadline if shared_deadline is not None else time.monotonic()+120
            bounded_read(lambda: self.backend.start_if_current(self.preflight.models[model.label], params,
                                                               current=lambda: not self.cancelled.is_set() and time.monotonic()<start_deadline), timeout_s=allowance)
            if shared_deadline is not None:
                remaining(120)  # Refuse a late return even from a faulty IO double.
        except BaseException:
            # Legacy pre-start validation failures historically propagated without
            # changing this owner. Stage B must close admission at every boundary.
            if shared_deadline is not None or self.owed:
                self.cancelled.set()  # also closes admission to late start workers
            raise

    def _inspect(self, path: str, body: dict[str, Any] | None = None) -> Any:
        if self.cancelled.is_set() or not self.owed:
            raise FitRefused("no active fit load")
        deadline = time.monotonic()+30
        def request() -> Any:
            url, password = self.backend.base_url, self.backend.api_key_file
            if url is None or password is None:
                raise FitRefused("owned inspection endpoint unavailable")
            key = password.read_text(encoding="utf-8").strip()
            remaining = deadline-time.monotonic()
            if remaining<=0 or self.cancelled.is_set():
                raise FitRefused('inspection expired/cancelled during credential read')
            return inspect_json(url,key,path,body=body,timeout_s=remaining)
        return bounded_read(request, timeout_s=30)

    def tokenize(self, text: str, *, add_special: bool) -> list[int]:
        payload = self._inspect("/tokenize", dict(content=text, add_special=add_special))
        ids = payload.get("tokens") if isinstance(payload, dict) and 'error' not in payload else None
        if not isinstance(ids, list):
            raise FitRefused("tokenizer response lacks a token list")
        return ids

    def observe(self) -> dict[str, Any]:
        from inferenceos.backends.resident_server import redacted_command
        identity,pid = bounded_read(lambda:(self.backend.identity,self.backend.pid),timeout_s=30)
        startup = bounded_read(self.backend.startup_log_snapshot, timeout_s=30)
        deadline = time.monotonic()+2
        while 'model loaded' not in str(startup.get('text','')) and time.monotonic()<deadline:
            time.sleep(.01)
            startup = bounded_read(self.backend.startup_log_snapshot,timeout_s=max(.001,deadline-time.monotonic()))
        startup["text_sha256"] = sha(str(startup.get("text", "")).encode())
        image = bounded_read(lambda: process_image(pid), timeout_s=30)
        slots = self._inspect("/slots")
        machine = bounded_read(self.inventory, timeout_s=30)
        command = bounded_read(lambda: list(self.backend.last_command or []), timeout_s=30)
        # The shared display redactor deliberately drops argv[0]. This evidence
        # needs the actual launch executable, independently checked against preflight.
        shown_command = [command[0], *redacted_command(command)] if command else []
        final_identity,final_pid = bounded_read(lambda:(self.backend.identity,self.backend.pid),timeout_s=30)
        current = (identity is not None and identity.pid == pid and identity.create_time == image.get("create_time_s")
                   and final_identity == identity and final_pid == pid)
        return dict(startup=startup, image=image, command=shown_command,
                    engine_environment=self.backend.engine_settings(),selected_executable=str(self.preflight.selected), process_current=current, slots=slots,
                    machine=machine, effects=observe_effects(startup.get("text", ""), slots))

    def release(self, *, abort: bool) -> dict[str, Any]:
        problem = None
        if abort:
            self.cancelled.set()
        try:
            if abort:
                measurement = self.backend.halt()
            else:
                measurement = self.backend.stop()
            if measurement is not None and (measurement.cleanup_complete is not True or measurement.credential_removed is False):
                problem = "backend release measurement is not clean"
        except Exception as exc:
            problem = type(exc).__name__+": "+str(exc)
            self.cancelled.set()
            self.backend.cleanup()
        finally:
            self.owed = False
        active = self.backend.is_active()
        gone = not active and self.backend.unconfirmed_pid is None
        credential = self.backend.api_key_file is None and not self.backend.undeleted_key_files
        if not gone or not credential or problem is not None:
            self.cancelled.set()
            problem = problem or "process/password release unconfirmed"
        return dict(process_gone=gone, credential_removed=credential, problem=problem)

    def retry_held(self) -> dict[str, Any] | None:
        if (not self.owed and not self.backend.is_active() and self.backend.unconfirmed_pid is None
                and self.backend.api_key_file is None and not self.backend.undeleted_key_files):
            return None
        self.cancelled.set()
        self.backend.halt()
        self.backend.cleanup()
        self.backend.confirm_previous_gone()
        return self.release(abort=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--rules-sha256")
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--proposal-sha256")
    parser.add_argument("--authorized-stage-a", action="store_true", help="operator acknowledgement of separate owner authorization")
    args = parser.parse_args(argv)
    settings = Settings()
    protected = [ROOT / n for n in ("data", "models", "tools", "src", "tests", "scripts", "docs")]
    protected += [settings.data_dir, ROOT / "reviewer-implementation-handoff-current.md"]
    protected += [p for p in (args.rules, args.proposal) if p is not None]
    if output_problem(args.out, protected=protected) or args.out.exists() or args.out.is_symlink():
        print("refused: protected or existing output")
        return 4
    if not args.simulate and not all((args.authorized_stage_a,args.rules,args.rules_sha256,args.proposal,args.proposal_sha256)):
        print("refused: reviewed manifest/proposal hashes and separate Stage-A authorization required")
        return 4
    record: dict[str, Any] = dict(status="preparing", simulated=args.simulate, loads=[])
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        initial_text = json.dumps(record)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(initial_text)
    except OSError:
        return 4

    expected_output = initial_text.encode('utf-8')
    final_publish = False

    def checkpoint(value: dict[str, Any]) -> None:
        nonlocal expected_output
        record.update(value)
        if (output_problem(args.out,protected=protected) or args.out.is_symlink()
                or bounded_file_bytes(args.out,maximum_bytes=64*1024*1024) != expected_output):
            raise OSError("owned output changed; no overwrite")
        published = dict(record)
        if published.get("status") == "done" and not final_publish:
            published["status"] = "collected-pending-preservation"
        text = json.dumps(published,indent=2,allow_nan=False)+"\n"
        atomic_write_text(args.out,text)
        written = bounded_file_bytes(args.out,maximum_bytes=64*1024*1024)
        if written.decode('utf-8').replace('\r\n','\n') != text:
            raise OSError('owned output changed after publication')
        expected_output = written

    interrupted: BaseException | None = None
    code = 3
    preflight: Preflight | None = None
    engine: LiveFitEngine | None = None
    try:
        if args.simulate:
            from inferenceos.tune.lookup_fit_sim import simulate_fit
            record.update(simulate_fit(checkpoint=checkpoint))
            code = 0 if record["status"] == "done" else 2
        else:
            assert args.rules and args.rules_sha256 and args.proposal and args.proposal_sha256
            proposal = strict_json(read_pinned(args.proposal,args.proposal_sha256))
            manifest = parse_manifest(read_pinned(args.rules,args.rules_sha256),proposal,args.proposal_sha256)
            initial_power = power_state()
            record['initial_power'] = initial_power
            if initial_power != "ac":
                raise FitRefused("initial AC required; no backend constructed")
            preflight = bounded_read(lambda: Preflight(ROOT,settings,manifest,args.rules,args.rules_sha256,
                                                       args.proposal,args.proposal_sha256), timeout_s=120)
            engine = LiveFitEngine(preflight)
            record.update(collect_fit(engine,manifest,run_id=uuid.uuid4().hex,power=power_state,
                                      guard=lambda:bounded_read(preflight.check,timeout_s=30),checkpoint=checkpoint))
            code = 0 if record["status"] == "done" else 2
    except (KeyboardInterrupt, SystemExit) as exc:
        interrupted = exc
        record.update(status="interrupted", error=type(exc).__name__)
    except (FitRefused, ValueError, LookupError, OSError) as exc:
        record.update(status="refused",error=type(exc).__name__+": "+str(exc))
        code = 4
    except Exception as exc:
        record.update(status="error",error=type(exc).__name__+": "+str(exc))
        code = 3
    finally:
        if engine is not None:
            try:
                retry = engine.retry_held()
                if retry is not None:
                    record.update(status="stopped-cleanup",cleanup_retry=retry)
                    code = 2
            except BaseException as exc:
                record.update(status="stopped-cleanup",cleanup_problem=type(exc).__name__+": "+str(exc))
                if isinstance(exc,(KeyboardInterrupt,SystemExit)):
                    interrupted = exc
                code = 2
        if preflight is not None:
            try:
                record["preservation"] = bounded_read(preflight.preservation,timeout_s=120)
            except BaseException as exc:
                record.update(status="changed-input-or-data", preservation_problem=type(exc).__name__+": "+str(exc))
                if isinstance(exc,(KeyboardInterrupt,SystemExit)):
                    interrupted = exc
                code = 6
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), generation_allowed=False,
                      stage_B_authorized=False,automatic_admission=False,deployment_allowed=False)
        final_publish = True
        try:
            checkpoint(record)
        except Exception as exc:
            print("publication failed: "+str(exc))
            code = 7
    if interrupted is not None:
        raise interrupted
    return code


if __name__ == "__main__":
    raise SystemExit(main())
