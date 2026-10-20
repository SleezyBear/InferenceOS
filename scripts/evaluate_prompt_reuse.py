"""Prompt reuse, development evaluation (16 Oct 2026; revised after review): run it on a
real engine, or on the simulated one (``--simulate``, no engine, no model).

What it measures and how it decides what counts is in ``src/inferenceos/tune/prompt_reuse.py``.
In short: per case, blocks of a fresh-arm trial and a reuse-arm trial on separate fresh
engine loads, identical work in both, the same priming request in both; a block's time
difference counts only with complete work evidence from the engine in both arms, reuse
shown in the reuse arm and none in the fresh arm, within what is possible, in the
priming slot, on mains power, with identical answers.

The engine setup is the model's saved setup for this context on mains power (read only),
or ``--layers N`` when there is none. The context size is the one given and is never
changed. Nothing is tuned or saved: saved setups and tune history are only read, every
file in the data folder is fingerprinted before and after, and the results file must lie
outside the data folder and the model file, in simulation too, and must not exist yet
(checked before anything starts; earlier results are never overwritten or merged).

The results file is rewritten (atomically) after every trial from this run's own
record in memory, so an interruption keeps everything this run did, with status
"interrupted". A refusal after the checks (no saved setup) is written with status
"refused".

Exit status: 0 "done" (every planned trial and the cancellation check ran, every
cleanup complete, no validity gate fired, the check passed; findings such as "no reuse"
are results, not failures); 2 "incomplete" (a trial failed or the cancellation check did
not pass); 3 an error outside the trials; 4 refused before starting (no setup, no model,
a results path inside the data folder or already existing); 5 stopped: an engine stop
was not confirmed clean (later models must not run); 6 the data folder changed; 7
stopped: a pre-registered validity gate (E1 a fresh reading reused, E4 impossible reuse,
in any reading);
9 stopped: not on mains power. An interruption (Ctrl+C) writes the record and is
re-raised.

    python scripts/evaluate_prompt_reuse.py --model models/study/Llama-3.2-3B-Instruct-Q4_K_M.gguf \\
        --context 4096 --blocks 4 --out _jobs/out/127_reuse_llama.json --run-id <id>
    python scripts/evaluate_prompt_reuse.py --simulate --blocks 3 --out reuse_sim.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
import uuid
from dataclasses import asdict, replace
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from inferenceos.tune.prompt_reuse import (  # noqa: E402
    DEFAULT_VARIANTS,
    DONE,
    INCOMPLETE,
    INTERRUPTED,
    STOPPED_CLEANUP,
    STOPPED_POWER,
    STOPPED_VALIDITY,
    Abandoned,
    Cleanup,
    EngineVariant,
    Reading,
    build_cases,
    evaluate,
)

EXIT_BY_STATUS = {DONE: 0, INCOMPLETE: 2, STOPPED_CLEANUP: 5, STOPPED_VALIDITY: 7, STOPPED_POWER: 9}


class RealReuseEngine:
    """The prompt-reuse engine interface over the real resident llama-server backend.

    **One backend object for the whole run**, so there is one cleanup owner: the backend
    keeps the identity of any process whose exit it could not confirm
    (``unconfirmed_pid``) and its undeleted password files, and refuses to overlap them.
    A stop counts as clean only from that evidence: the backend no longer active, no
    unconfirmed process, no password file left. A cleared process handle is not
    confirmation. While a stop is unconfirmed (or was interrupted), no further start is
    allowed; ``retry_held()`` makes one more release attempt through the backend's own
    ``cleanup()`` and a bounded wait for the unconfirmed process, and reports what is
    still unresolved."""

    def __init__(self, *, settings: Any = None, model: Any, configuration: Any,
                 backend_factory: Callable[[], Any] | None = None,
                 capture_load_evidence: bool = False, run_id: str = "",
                 model_generation: tuple[str, int, int, int] | None = None) -> None:
        if backend_factory is None:
            from inferenceos.backends.llama_server_toolchain import resolve_server_executable
            from inferenceos.backends.resident_server import LlamaServerResidentBackend

            def backend_factory() -> Any:
                return LlamaServerResidentBackend(
                    server=resolve_server_executable(settings.llama_cpp_dir),
                    readiness_timeout_s=settings.resident_readiness_timeout_s,
                    request_timeout_s=settings.resident_request_timeout_s,
                )
        self._make = backend_factory
        self._model = model
        self._configuration = configuration
        self._backend: Any = None
        self._params: Any = None
        #: A load is (or may be) running: set before ``start``, cleared by a finished stop.
        self._loaded = False
        #: Why the last release is not confirmed (None: confirmed). Set before a stop
        #: begins, so an interrupted stop leaves it set.
        self.unreleased: str | None = None
        self.commands: list[list[str]] = []
        self.engine_binaries: list[dict[str, Any]] = []
        self.input_problem: str | None = None
        self._binary_stat: tuple[Any, ...] | None = None
        self.capture_load_evidence = capture_load_evidence
        self.load_evidence: list[dict[str, Any]] = []
        self._active_load: dict[str, Any] | None = None
        self._run_id = run_id
        self._model_generation = model_generation

    @property
    def load_evidence_id(self) -> str | None:
        return str(self._active_load["load_id"]) if self._active_load is not None else None

    def _capture_load(self, phase: str) -> None:
        """Observe only: ordinary diagnostic failures never replace lifecycle errors."""
        load = self._active_load
        if load is None:
            return
        started = time.perf_counter()
        try:
            from inferenceos.tune.reuse_evidence import process_image
            backend = self._backend
            pid = backend.pid
            expected = backend.identity
            reader = getattr(backend, "startup_log_snapshot", None)
            startup = reader() if callable(reader) else {
                "available": False, "text": self.startup_log(), "truncated": None,
                "reader_errors": ["startup loss metadata unavailable"],
            }
            image = process_image(pid)
            current = (expected is not None and expected.pid == pid
                       and expected.create_time == image.get("create_time_s")
                       and backend.pid == pid and backend.identity == expected
                       and startup.get("generation_current") is True)
            text = startup.get("text")
            if isinstance(text, str):
                startup["text_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            load["snapshots"].append({"phase": phase, "startup": startup, "image": image,
                                      "process_current": current})
        except Exception as exc:  # noqa: BLE001 - evidence failure stays explicit
            load["capture_errors"].append({"phase": phase, "error": type(exc).__name__})
        finally:
            load["capture_seconds"] = load.get("capture_seconds", 0.0) + time.perf_counter() - started

    @property
    def held(self) -> Any:
        """The backend whose release is not confirmed, or None."""
        return self._backend if (self.unreleased is not None or self._loaded) else None

    def start(self, variant: EngineVariant) -> None:
        from inferenceos.backends.resident_server import redacted_command
        from inferenceos.experiment.restoration import runtime_params_from

        if self.input_problem:
            raise RuntimeError(self.input_problem)
        if self.unreleased is not None or self._loaded:
            raise RuntimeError("an earlier engine's cleanup is not confirmed; no new engine is started "
                               f"({self.unreleased or 'still loaded'})")
        if self._model_generation is not None:
            from inferenceos.models.identity import model_file_signature

            try:
                unchanged = model_file_signature(Path(self._model_generation[0])) == self._model_generation
            except OSError:
                unchanged = False
            if not unchanged:
                self.input_problem = "model file changed after preflight; no engine is started"
                raise RuntimeError(self.input_problem)
        params = runtime_params_from(self._configuration).model_copy(
            update={"cache_reuse": variant.cache_reuse, "swa_full": bool(variant.swa_full)})
        if self.capture_load_evidence:
            # b11139 maps the core's INFO messages (including cache sizes) to trace.
            # Both study arms use the same diagnostic level; production is unchanged.
            params = params.model_copy(update={"log_verbosity": 4})
        if params.context_length != self._configuration.context_length:
            raise RuntimeError("context size would change; refused")
        if self._backend is None:
            self._backend = self._make()
        self._params = params
        self._loaded = True  # from here a stop is owed, whatever happens in start
        if self.capture_load_evidence:
            self._active_load = {"load_id": f"{self._run_id}/load-{len(self.load_evidence) + 1:06d}",
                                 "variant": asdict(variant), "requested_params": params.model_dump(mode="json"),
                                 "snapshots": [], "capture_errors": [], "state": "starting"}
            self.load_evidence.append(self._active_load)
        try:
            self._backend.start(self._model, params)
        except BaseException:
            self._capture_load("start_failed")
            raise
        if self._active_load is not None:
            self._active_load["state"] = "ready"
        self._capture_load("ready")
        command = getattr(self._backend, "last_command", None)
        if command:
            self.commands.append(redacted_command(command))
            if self._active_load is not None:
                self._active_load["command"] = redacted_command(command)
            binary = Path(command[0])
            try:
                stat = binary.stat()
                signature = (str(binary.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino)
                if signature != self._binary_stat:
                    identity = _file_identity(binary)
                    known = [item["sha256"] for item in self.engine_binaries if item.get("sha256")]
                    if known and identity["sha256"] != known[0]:
                        self.input_problem = "the engine executable changed during the study"
                    self.engine_binaries.append(identity)
                    self._binary_stat = signature
            except OSError as exc:
                # A process double or execute-only binary may not be readable. Report
                # the limitation rather than inventing a build identity.
                if any(item.get("sha256") for item in self.engine_binaries):
                    self.input_problem = "the previously identified engine executable can no longer be verified"
                self.engine_binaries.append({"path": str(binary), "identity_error": type(exc).__name__})
            if self.input_problem:
                raise RuntimeError(self.input_problem)

    def tokenize(self, text: str) -> list[int]:
        return list(self._backend.tokenize(text))

    def generate(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int) -> Reading:
        from inferenceos.interfaces.backend_adapter import GenerationRequest
        from inferenceos.backends.resident_server import ResidentGenerationProtocolError

        try:
            result = self._backend.generate(GenerationRequest(prompt=prompt, n_predict=n_predict, seed=seed,
                                                              temperature=0.0, ignore_eos=True, cache_prompt=cache_prompt))
        except ResidentGenerationProtocolError as exc:
            # Preserve reported counters for E1/E4 on unusable work. The explicit
            # protocol problem keeps this reading out of timing/correctness claims.
            result = exc.result
        raw = dict(getattr(self._backend, "last_response_counters", {}) or {})
        engine_generated = raw.get("predicted_n") if type(raw.get("predicted_n")) is int else raw.get("tokens_predicted")
        return Reading(ttft_s=result.ttft_s, total_s=result.total_s, prefill_s=result.prefill_s,
                       processed=raw.get("prompt_n") if type(raw.get("prompt_n")) is int else None,
                       cached=result.prompt_tokens_cached, total=result.prompt_tokens_total, text=result.text,
                       cache_prompt=cache_prompt,
                       generated=engine_generated if type(engine_generated) is int else None,
                       completed=raw.get("stop_seen") is True,
                       slot=raw.get("id_slot") if type(raw.get("id_slot")) is int else None,
                       occupancy=raw.get("tokens_cached_occupancy")
                       if type(raw.get("tokens_cached_occupancy")) is int else None,
                       engine_counters=raw, load_evidence_id=self.load_evidence_id)

    def abandon(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int, after_chunks: int) -> Abandoned:
        from inferenceos.interfaces.backend_adapter import GenerationRequest

        request = GenerationRequest(prompt=prompt, n_predict=n_predict, seed=seed, temperature=0.0,
                                    ignore_eos=True, cache_prompt=cache_prompt)
        evidence = self._backend.abandon_generation(request, after_chunks=after_chunks)
        return Abandoned(pieces=int(evidence["pieces"]), ending=str(evidence["ending"]),
                         idle_confirmed=evidence["idle_confirmed"], idle_detail=evidence.get("idle_detail"),
                         error=evidence.get("error"))

    def healthy(self) -> bool:
        return bool(self._loaded and self._backend is not None and self._backend.health())

    def chat(self, messages: list[dict[str, str]], *, cache_prompt: bool | None, max_tokens: int,
             seed: int) -> Any:
        """One ordinary chat request (reuse-correctness exposure check)."""
        from inferenceos.tune.reuse_correctness import chat_reading

        return replace(chat_reading(self._backend.chat_request(messages, max_tokens=max_tokens, seed=seed,
                                                              cache_prompt=cache_prompt)),
                       load_evidence_id=self.load_evidence_id)

    def startup_log(self) -> str | None:
        """The load's startup log, or None when it cannot show warnings (log verbosity
        below 2 in b11139's scale) or there is no running load."""
        verbosity = getattr(self._params, "log_verbosity", None)
        if not self._loaded or self._backend is None or (verbosity is not None and verbosity < 2):
            return None
        return str(self._backend.server_startup_log())

    def release_evidence(self) -> dict[str, Any]:
        """What the backend itself says is still unreleased."""
        backend = self._backend
        if backend is None:
            return {"active": False, "unconfirmed_pid": None, "password_files_left": []}
        left = [str(p) for p in backend.undeleted_key_files]
        current = getattr(backend, "api_key_file", None)  # still held as the current key
        if current is not None and str(current) not in left:
            left.append(str(current))
        return {"active": bool(backend.is_active()), "unconfirmed_pid": backend.unconfirmed_pid,
                "password_files_left": left}

    @staticmethod
    def _unresolved(evidence: dict[str, Any]) -> list[str]:
        problems = []
        if evidence["active"]:
            problems.append("the engine is still active")
        if evidence["unconfirmed_pid"] is not None:
            problems.append(f"the exit of process {evidence['unconfirmed_pid']} is not confirmed")
        if evidence["password_files_left"]:
            problems.append(f"password file not deleted: {', '.join(evidence['password_files_left'])}")
        return problems

    def stop(self) -> Cleanup:
        # A Ctrl+C during optional capture still reaches the existing cleanup owner;
        # its exception propagates after cleanup, with no next measurement.
        try:
            self._capture_load("before_stop")
        finally:
            load = self._active_load
            try:
                result = self._stop_backend()
                if load is not None:
                    load.update(state="released" if result.clean else "unreleased", cleanup=asdict(result))
            except BaseException as exc:
                if load is not None:
                    load.update(state="stop_error", stop_error=type(exc).__name__)
                raise
            finally:
                self._active_load = None
        return result

    def _stop_backend(self) -> Cleanup:
        backend = self._backend
        if backend is None or not self._loaded:
            # Nothing was loaded by this start: clean unless an earlier release is unconfirmed.
            return Cleanup(process_gone=self.unreleased is None, credential_removed=None, problem=self.unreleased)
        self.unreleased = "stop not finished"
        problems: list[str] = []
        measured_gone: bool | None = None
        credential: bool | None = None
        try:
            if backend.is_active():
                measurement = backend.stop()
                measured_gone = bool(measurement.cleanup_complete)
                credential = measurement.credential_removed
                if not measured_gone:
                    problems.append("the engine's stop did not confirm the process exited")
            # Not active: a failed start already stopped what it owned; the evidence
            # below (unconfirmed identity, password files) says whether that worked.
        except Exception as exc:  # noqa: BLE001 - try the no-raise release, report unclean
            problems.append(f"stop failed: {type(exc).__name__}: {exc}")
            try:
                backend.cleanup()
            except Exception:  # noqa: BLE001
                pass
        evidence = self.release_evidence()
        problems.extend(self._unresolved(evidence))
        self._loaded = False
        self.unreleased = "; ".join(problems) or None
        return Cleanup(process_gone=bool(not evidence["active"] and evidence["unconfirmed_pid"] is None
                                         and measured_gone is not False),
                       credential_removed=(not evidence["password_files_left"]) and credential is not False,
                       problem=self.unreleased)

    def retry_held(self) -> dict[str, Any] | None:
        """One more release attempt when a release is unconfirmed or was interrupted
        (None when there is nothing to release). Uses the backend's own ``cleanup()`` and
        a bounded wait for an unconfirmed process; reports what is still unresolved."""
        if self._backend is None or (self.unreleased is None and not self._loaded):
            return None
        backend = self._backend
        result: dict[str, Any] = {"retried": True, "before": self.unreleased or "still loaded"}
        try:
            backend.cleanup()
            backend.confirm_previous_gone()
        except BaseException as exc:  # noqa: BLE001 - recorded; the caller keeps the original outcome
            result["error"] = f"{type(exc).__name__}: {exc}"
        evidence = self.release_evidence()
        unresolved = self._unresolved(evidence)
        result.update(evidence)
        result["released"] = not unresolved
        result["unresolved"] = unresolved
        if not unresolved:
            self._loaded, self.unreleased = False, None
        return result


def power_state(*, platform: str | None = None, ac_line: Callable[[], int] | None = None) -> str:
    """'ac', 'no battery', 'battery' or 'unknown (...)': only the first two allow a trial.

    Windows: the system's own AC line status (``GetSystemPowerStatus``): 1 = ac, 0 =
    battery, anything else (255 = unknown) or a failed read = unknown. psutil is not used
    there: it reports "no battery" (None) whenever the battery flag has bit 128 set,
    which the unknown flag 255 also has. Elsewhere: psutil's battery reading."""
    if (platform or sys.platform) == "win32":
        try:
            line = (ac_line or _windows_ac_line)()
        except Exception as exc:  # noqa: BLE001 - an unreadable state is not mains power
            return f"unknown (the power status could not be read: {type(exc).__name__})"
        if line == 1:
            return "ac"
        if line == 0:
            return "battery"
        return f"unknown (AC line status {line})"
    import psutil

    battery = psutil.sensors_battery()
    if battery is None:
        return "no battery"
    if battery.power_plugged is True:
        return "ac"
    if battery.power_plugged is False:
        return "battery"
    return "unknown (the battery does not say whether it is plugged in)"


def _windows_ac_line() -> int:
    """``SYSTEM_POWER_STATUS.ACLineStatus`` from ``GetSystemPowerStatus`` (Windows only)."""
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        raise OSError("GetSystemPowerStatus failed")
    return int(status.ACLineStatus)


def _setup(settings: Any, model_path: Path, context: int, layers: int | None, *,
           identity: dict[str, Any] | None = None, require_saved: bool = False) -> tuple[Any, str]:
    from inferenceos.contracts.enums import PowerState
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.tune.profiles import SetupStore

    saved = SetupStore(settings).find(model_path, context_length=context, power=PowerState.AC)
    if saved is not None:
        current = identity if identity is not None else _file_identity(model_path)
        if (_norm(Path(current["path"])) != _norm(model_path)
                or _norm(Path(saved.model.path)) != _norm(model_path)
                or saved.model.size_bytes != current["size_bytes"]
                or saved.model.sha256 is None
                or saved.model.sha256.lower() != current["sha256"]):
            # Development studies already compute a full model hash. Do not let
            # matching size/date bypass a contradictory or absent saved content hash.
            raise LookupError("saved setup model identity does not match the current model bytes; refused")
        config = saved.configuration
        source = f"saved setup {saved.setup_id} (read only)"
    elif layers is not None and not require_saved:
        config = RuntimeConfiguration(context_length=context, gpu_offload_layers=layers)
        source = f"--layers {layers}; the rest engine default"
    else:
        if require_saved:
            raise LookupError("frozen study requires a verified saved setup; --layers cannot replace it")
        raise LookupError("no saved setup for this model and context on mains power; give --layers")
    if config.context_length != context:
        raise LookupError(f"the setup's context {config.context_length} is not {context}")
    return config, source


def _norm(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path))


def output_problem(out: Path, *, protected: list[Path]) -> str | None:
    """Why ``out`` may not be written (None: it may). Resolved through links: it must not
    be, or lie inside, a protected folder or file, and must not be an existing folder."""
    target = _norm(out)
    for item in protected:
        guard = _norm(item)
        if target == guard or target.startswith(guard.rstrip(os.sep) + os.sep):
            return f"the results file {out} is inside the protected {item}"
    if out.is_dir():
        return f"the results path {out} is a folder"
    return None


def _fingerprint(folder: Path) -> dict[str, dict[str, Any]]:
    from inferenceos.tune.reuse_evidence import file_identity

    if not folder.exists():
        return {}
    if not folder.is_dir():
        raise OSError(f"data folder is not a directory: {folder}")
    def enumerate_files() -> list[Path]:
        # glob/rglob can silently skip unreadable subdirectories. An incomplete map
        # must not establish preservation. Refuse links/junctions rather than follow
        # targets outside the declared root or silently ignore their contents.
        pending, files = [folder], []
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if (entry.is_symlink() or getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
                            & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                        raise OSError(f"unsupported data-folder link/junction: {path}")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif entry.is_file(follow_symlinks=False):
                        files.append(path)
                    else:
                        raise OSError(f"unsupported data-folder entry: {path}")
        return sorted(files)

    paths = enumerate_files()
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        identity = file_identity(path)
        result[path.relative_to(folder).as_posix()] = {
            "size": identity["size_bytes"], "mtime_ns": identity["modified_ns"], "sha256": identity["sha256"]}
    if paths != enumerate_files():
        raise OSError("data-folder file population changed during fingerprint capture")
    return result


def _file_identity(path: Path) -> dict[str, Any]:
    """Content identity outside timed measurements; refuse a file changing while read."""
    from inferenceos.tune.reuse_evidence import file_identity

    return file_identity(path)


def _write(out: Path, record: dict[str, Any]) -> None:
    from inferenceos.storage import atomic_write_text

    atomic_write_text(out, json.dumps(record, indent=2, default=str) + "\n")


def _study_metadata(pairs: Any) -> dict[str, Any]:
    from inferenceos.tune.reuse_correctness import CORRECTNESS_VERSION, CHAT_MAX_TOKENS
    from inferenceos.tune.prompt_reuse import SEED, MEASURED_PIECES, PRIMING_PIECES

    return {"name": "reuse correctness", "version": CORRECTNESS_VERSION,
            "judge_scope": "first-answer numeric/refusal grammar; not general semantic or model safety proof",
            "generation": {"seed": SEED, "temperature": 0.0, "native_measured_pieces": MEASURED_PIECES,
                           "native_priming_pieces": PRIMING_PIECES, "chat_max_tokens": CHAT_MAX_TOKENS},
            "pairs": {p.pair_id: {"expected": p.expected, "source": p.source, "regression": p.regression,
                                  "question_prime": p.question_prime, "question_measured": p.question_measured}
                      for p in pairs}}


def _correctness(record: dict[str, Any], result: dict[str, Any], engine: Any, pairs: Any, *, chat: bool,
                 power: Any, power_check: bool, checkpoint: Any) -> str:
    """The correctness study's additions to the record (job 128): the pre-registered
    correctness summary of the trials and, when asked and the trials did not stop, the
    chat exposure check with its own immediate stops. Returns the status."""
    from inferenceos.tune.prompt_reuse import TrialResult
    from inferenceos.tune.reuse_correctness import (
        CHAT_BLOCKS,
        chat_summary,
        correctness_summary,
        run_chat_exposure,
    )

    status = str(result["status"])
    record.setdefault("study", _study_metadata(pairs))
    record["correctness"] = correctness_summary([TrialResult(**t) for t in result["trials"]], pairs)
    if status not in (DONE, INCOMPLETE):
        record["correctness"]["swa_full_comparison"]["interpretable"] = False
        record["correctness"]["swa_full_comparison"]["reason"] = f"study execution stopped: {status}"
    if not chat or status not in (DONE, INCOMPLETE):
        return status

    def chat_checkpoint(chat_trials: list[dict[str, Any]], stop: str | None, reason: str | None) -> None:
        checkpoint({**result, "study": record["study"], "correctness": record["correctness"],
                    "chat_trials": chat_trials, **({"status": stop, "stop_reason": reason} if stop else {})})

    chat_trials, stop, reason = run_chat_exposure(engine, pairs, blocks=CHAT_BLOCKS, power=power,
                                                  power_check=power_check, checkpoint=chat_checkpoint)
    record["chat_trials"] = chat_trials
    record["chat_summary"] = chat_summary(chat_trials, pairs)
    if stop:
        record["status"], record["stop_reason"] = stop, reason
        return str(stop)
    unusable = [f"{pair_id} block {row['block'] + 1}" for pair_id, item in record["chat_summary"]["pairs"].items()
                for row in item["trials"] if not row["usable"]]
    if status == DONE and (unusable or len(chat_trials) != CHAT_BLOCKS * len(pairs)):
        # A missing or unusable reply is never an observed answer: the run is incomplete.
        record["status"] = INCOMPLETE
        record["stop_reason"] = f"chat exposure trials not usable: {', '.join(unusable) or 'missing trials'}"
        return str(INCOMPLETE)
    return status


class _Refused(Exception):
    """Refused before any engine started (recorded with status "refused", exit 4)."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--blocks", type=int, default=None,
                        help="blocks per case and variant (default: 4 for the reuse study, 3 for correctness)")
    parser.add_argument("--sections", type=int, default=12)
    parser.add_argument("--cases", default="", help="comma-separated case ids (default: all)")
    parser.add_argument("--no-variants", action="store_true", help="only the engine-default variant")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--simulate", action="store_true", help="use the simulated engine (no model, no engine)")
    parser.add_argument("--no-power-check", action="store_true",
                        help="test runs only: do not require mains power (recorded in the results)")
    parser.add_argument("--study", choices=("reuse", "correctness"), default="reuse",
                        help="reuse: the speed study (job 127); correctness: known-answer pairs (job 128)")
    parser.add_argument("--swa-full-variant", action="store_true",
                        help="correctness study: also run every pair with --swa-full")
    parser.add_argument("--chat-exposure", action="store_true",
                        help="correctness study: also ask every pair as an ordinary chat client would")
    parser.add_argument("--capture-load-evidence", action="store_true",
                        help="opt-in development diagnostics: trace logs and per-load observed mapped-file identities")
    parser.add_argument("--preregistration", type=Path, help="frozen governing rules (paired with SHA-256)")
    parser.add_argument("--preregistration-sha256", help="expected byte SHA-256 of the frozen rules")
    args = parser.parse_args(argv)
    run_id = args.run_id or str(uuid.uuid4())
    if args.study != "correctness" and (args.swa_full_variant or args.chat_exposure):
        print("--swa-full-variant and --chat-exposure belong to --study correctness")
        return 3
    from inferenceos.tune.reuse_correctness import BLOCKS as CORRECTNESS_BLOCKS
    from inferenceos.tune.reuse_correctness import DEFAULT as CORRECTNESS_DEFAULT
    from inferenceos.tune.reuse_correctness import SWA_FULL, build_pairs
    from inferenceos.tune.reuse_correctness import CORRECTNESS_VERSION

    if args.sections < (12 if args.study == "correctness" else 1) or (args.blocks is not None and args.blocks < 1):
        print("refused: positive block count and enough document sections are required (correctness: at least 12)")
        return 4
    registration = None
    frozen_model_sha256: str | None = None
    require_saved_setup = False
    if args.preregistration is not None or args.preregistration_sha256 is not None:
        try:
            if args.preregistration is None or args.preregistration_sha256 is None or args.study != "correctness":
                raise ValueError("both preregistration arguments belong to the correctness study")
            rules = args.preregistration.read_bytes()
            digest = hashlib.sha256(rules).hexdigest()
            if digest != args.preregistration_sha256.lower() or CORRECTNESS_VERSION not in rules.decode("utf-8"):
                raise ValueError("governing rules digest or study version does not match")
            # Historical job128 rules are text. Structured frozen rules bind the
            # model too: missing/malformed pins cannot silently become an unpinned run.
            if args.preregistration.suffix.lower() == ".json" or rules.lstrip().startswith((b"{", b"[")):
                frozen = json.loads(rules)
                if not isinstance(frozen, dict) or frozen.get("study_version") != CORRECTNESS_VERSION:
                    raise ValueError("structured rules need the current study version and an object layout")
                pin = frozen.get("model_sha256")
                if not isinstance(pin, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", pin):
                    raise ValueError("frozen rules need a valid model SHA-256 pin")
                frozen_model_sha256 = pin.lower()
                require_saved_setup = frozen.get("require_saved_setup", False)
                if (type(require_saved_setup) is not bool
                        or (frozen.get("protocol") == "g3-evidence-2" and not require_saved_setup)):
                    raise ValueError("G3 rules require a saved setup with a verified model identity")
            registration = {"sha256": digest, "file": str(args.preregistration), "study_version": CORRECTNESS_VERSION}
        except (OSError, UnicodeError, ValueError) as exc:
            print(f"refused: {exc}")
            return 4

    pairs = build_pairs(args.sections) if args.study == "correctness" else ()
    blocks = args.blocks if args.blocks is not None else (CORRECTNESS_BLOCKS if pairs else 4)
    cases = tuple(pair.case for pair in pairs) if pairs else build_cases(args.sections)
    if args.cases:
        wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
        if not wanted:
            print("refused: the selected case list is empty")
            return 4
        unknown = sorted(set(wanted) - {c.case_id for c in cases})
        if unknown:
            print(f"unknown case(s): {unknown}")
            return 3
        cases = tuple(c for c in cases if c.case_id in wanted)
    if pairs:
        variants: tuple[EngineVariant, ...] = (CORRECTNESS_DEFAULT,) + ((SWA_FULL,) if args.swa_full_variant else ())
        wanted_ids = {c.case_id for c in cases}
        pairs = tuple(pair for pair in pairs if pair.case.case_id in wanted_ids)
    else:
        variants = DEFAULT_VARIANTS[:1] if args.no_variants else DEFAULT_VARIANTS
    record: dict[str, Any] = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
                              "simulated": bool(args.simulate)}
    if registration is not None:
        record["preregistration"] = registration
    if pairs:
        record["study"] = _study_metadata(pairs)
    from inferenceos.config import Settings

    if not args.simulate and (args.model is None or not args.model.is_file()):
        print("refused: --model must name an existing model file")
        return 4
    settings: Any = Settings()
    # Protected in every mode, simulation included: the data folder, and the model file.
    protected: list[Path] = [Path(settings.data_dir)] + ([args.model] if args.model is not None else [])
    data_dir: Path | None = None if args.simulate else Path(settings.data_dir)
    problem = output_problem(args.out, protected=protected)
    if problem is None and (args.out.exists() or args.out.is_symlink()):
        # Results are only ever written by the run that made the file: an existing file
        # (an earlier run's evidence, or anything else) is never overwritten or merged.
        problem = f"the results file {args.out} already exists; give a new name"
    if problem:
        print(f"refused: {problem}")
        return 4
    # Claim the name atomically before any engine starts. The earlier exists() check
    # alone allowed two invocations to overwrite each other's checkpoints.
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as claimed:
            json.dump({**record, "status": "preparing"}, claimed)
    except OSError as exc:
        print(f"refused: cannot reserve results file: {exc}")
        return 4
    before: dict[str, Any] = {}
    before_taken = False
    engine: Any = None
    code = 3
    interrupted = False
    refused: str | None = None
    #: This invocation's latest checkpoint, kept in memory: the only partial evidence
    #: the final record may use (never read back from the file).
    latest: dict[str, Any] = {}

    def checkpoint(partial: dict[str, Any]) -> None:
        latest.clear()
        latest.update(partial)
        if engine is not None and getattr(engine, "engine_binaries", None):
            record["engine_binaries"] = engine.engine_binaries
        if args.capture_load_evidence:
            from inferenceos.tune.reuse_evidence import EVIDENCE_VERSION
            record["load_evidence"] = {"version": EVIDENCE_VERSION,
                                       "simulated": bool(args.simulate),
                                       "loads": getattr(engine, "load_evidence", [])}
        published = {**record, **partial}
        if args.study == "correctness" and published.get("status") in (DONE, INCOMPLETE):
            # Completing native work (or one chat block) is not completing the study.
            # Only final publication after analysis and input checks can say "done".
            published["status"] = "running"
            published["phase"] = "collecting or analysing correctness evidence"
        try:
            _write(args.out, published)
        except OSError as exc:
            if (partial.get("status") == INTERRUPTED
                    or any(t.get("interrupted") for key in ("trials", "chat_trials") for t in partial.get(key, []))):
                raise KeyboardInterrupt from exc
            raise

    try:
        if data_dir is not None:
            record["data_folder_evidence"] = {"version": "file-fingerprint-1", "root": str(data_dir.resolve()),
                                               "before": None, "after": None,
                                               "before_complete": False, "after_complete": False}
        before = _fingerprint(data_dir) if data_dir is not None else {}
        before_taken = True
        if data_dir is not None:
            record["data_folder_evidence"].update(before=before, before_complete=True,
                                                   before_captured_at=datetime.now(timezone.utc).isoformat())
            _write(args.out, {**record, "status": "preparing"})
        if args.simulate:
            from inferenceos.tune.prompt_reuse_sim import SimulatedReuseEngine

            if pairs:
                from inferenceos.tune.prompt_reuse_sim import SimFaults
                from inferenceos.tune.reuse_correctness import answer_oracle

                # Simulated only: a sliding-window model whose partial reuse is wrong without a
                # full-size cache (the job 127 observation, as a fault), to exercise the rules.
                engine = SimulatedReuseEngine(oracle=answer_oracle(pairs), sliding_window=True,
                                              faults=SimFaults(partial_reuse_wrong=True))
                record["setup_source"] = ("simulated engine (sliding-window model whose partial reuse is wrong "
                                          "without a full-size cache)")
            else:
                engine = SimulatedReuseEngine()
                record["setup_source"] = "simulated engine"
            configuration: dict[str, Any] = {"simulated": True, "context_length": args.context}
            probe = lambda: "not checked"  # noqa: E731
            power_check = False
        else:
            from inferenceos.models.loader import load_model_profile
            from inferenceos.models.identity import model_file_signature

            try:
                model_path = args.model.resolve()
                record["model"] = str(model_path)
                generation = model_file_signature(model_path)
                identity = _file_identity(model_path)
                record["model_file_identity"] = identity
                record["model_preflight"] = {"expected_sha256": frozen_model_sha256,
                                             "pin_verified": False, "saved_setup_required": require_saved_setup,
                                             "saved_setup_verified": False}
                if frozen_model_sha256 is not None and identity["sha256"] != frozen_model_sha256:
                    raise LookupError("model SHA-256 differs from the frozen rules; no engine is constructed")
                record["model_preflight"]["pin_verified"] = frozen_model_sha256 is not None
                if model_file_signature(model_path) != generation:
                    raise LookupError("model file changed during preflight; no engine is constructed")
                config, source = _setup(settings, model_path, args.context, args.layers,
                                        identity=identity, require_saved=require_saved_setup)
                record["model_preflight"]["saved_setup_verified"] = require_saved_setup
                model = load_model_profile(model_path)
                if model_file_signature(model_path) != generation:
                    raise LookupError("model file changed during setup/metadata preparation; no engine is constructed")
            except LookupError as exc:
                refused = str(exc)
                raise _Refused from exc
            record["setup_source"] = source
            configuration = config.model_dump(mode="json")
            engine = RealReuseEngine(settings=settings, model=model,
                                     configuration=config, capture_load_evidence=args.capture_load_evidence,
                                     run_id=run_id, model_generation=generation)
            power_check = not args.no_power_check
            probe = power_state if power_check else (lambda: "not checked")
        result = evaluate(engine, cases=cases, variants=variants, blocks=blocks, context_length=args.context,
                          configuration=configuration, run_id=run_id, progress=lambda line: print(line, flush=True),
                          power=probe, power_check=power_check, checkpoint=checkpoint,
                          cancellation=not pairs)
        record.update(result)
        status = result["status"]
        if pairs:
            status = _correctness(record, result, engine, pairs, chat=args.chat_exposure, power=probe,
                                  power_check=power_check, checkpoint=checkpoint)
        code = EXIT_BY_STATUS.get(status, 3)
    except _Refused:
        record["status"] = "refused"
        record["refusal"] = refused
        code = 4
    except KeyboardInterrupt:
        interrupted = True
    except Exception as exc:  # noqa: BLE001 - recorded; exit 3
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        record["status"] = "error"
        code = 3
    finally:
        for key, value in latest.items():  # this invocation's own partial evidence only
            record.setdefault(key, value)
        if interrupted:
            record["status"] = INTERRUPTED
        if engine is not None:
            record["engine_commands"] = getattr(engine, "commands", [])
            retry = getattr(engine, "retry_held", lambda: None)()
            if retry is not None:
                record["held_engine_retry"] = retry
                if str(retry.get("error", "")).startswith("KeyboardInterrupt"):
                    interrupted = True
                    record["status"] = INTERRUPTED
            if args.capture_load_evidence:
                from inferenceos.tune.reuse_evidence import EVIDENCE_VERSION, capture_problems
                loads = getattr(engine, "load_evidence", [])
                record["load_evidence"] = {"version": EVIDENCE_VERSION, "simulated": bool(args.simulate),
                                           "loads": loads, "problems": capture_problems(loads)
                                           if loads else ["no per-load observations available"]}
        if not args.simulate and record.get("model_file_identity"):
            try:
                after_model = _file_identity(args.model.resolve())
                record["model_file_unchanged"] = after_model["sha256"] == record["model_file_identity"]["sha256"]
                if not record["model_file_unchanged"]:
                    record["status"], code = "model file changed", 6
                    if record.get("correctness"):
                        record["correctness"]["swa_full_comparison"]["interpretable"] = False
            except KeyboardInterrupt:
                interrupted = True
            except OSError as exc:
                record["model_file_unchanged"] = None
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                record["status"], code = "error", 3
        if data_dir is not None and before_taken:
            from run_practical_eval import compare
            try:
                after = _fingerprint(data_dir)
                record["data_folder_evidence"].update(after=after, after_complete=True,
                                                       after_captured_at=datetime.now(timezone.utc).isoformat())
                changes = compare(before, after)
                record["data_folder_unchanged"] = not changes
                record["data_folder_changes"] = changes
                if changes:
                    record["status"] = "data folder changed"
                    code = 6
            except KeyboardInterrupt:
                interrupted = True
                record["status"] = INTERRUPTED
            except OSError as exc:
                record["data_folder_unchanged"] = None
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                record["status"], code = "error", 3
        if interrupted:
            record["status"] = INTERRUPTED
        if record.get("status") not in (DONE, INCOMPLETE) and record.get("correctness"):
            comparison = record["correctness"]["swa_full_comparison"]
            comparison["interpretable"] = False
            comparison["reason"] = f"study execution stopped: {record.get('status')}"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            _write(args.out, record)  # outside protected data; owned by this invocation
        except KeyboardInterrupt:
            interrupted = True
        except OSError as exc:
            # Cleanup is already complete. Never turn failed publication into success,
            # or mask a pending interruption with a second filesystem exception.
            code = 3
            record["status"] = "output error"
            print(f"could not write final evidence: {type(exc).__name__}: {exc}")
    if interrupted:
        print(f"status: interrupted; partial results: {args.out}")
        raise KeyboardInterrupt
    if refused is not None:
        print(f"refused: {refused}")
    print(f"status: {record.get('status')} (exit {code}); results: {args.out}")
    if record.get("stop_reason"):
        print(f"  stop reason: {record['stop_reason']}")
    for key, item in ((record.get("correctness") or {}).get("pairs") or {}).items():
        print(f"  correctness {key}: {item['outcome']} (usable {item['usable_blocks']}/{len(item['blocks'])})")
    for key, item in ((record.get("chat_summary") or {}).get("pairs") or {}).items():
        print(f"  chat exposure {key}: {item['outcome']}")
    for key, item in (record.get("summary") or {}).items() if not record.get("correctness") else ():
        gain = item.get("mean_relative_ttft_gain")
        print(f"  {key}: {item['role']}, usable {item['usable_blocks']}/{item['blocks']}, "
              f"fresh arm stayed fresh: {item['fresh_arm_stayed_fresh']}, "
              f"mean first-word gain: {'n/a' if gain is None else f'{gain:+.1%}'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
