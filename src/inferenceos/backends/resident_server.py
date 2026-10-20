"""Real resident llama.cpp server backend used by :class:`LlamaCppResidentAdapter`.

``LlamaServerResidentBackend`` owns one real ``llama-server`` process and exposes
measured load/unload/generation evidence. It never falls back to spawning
``llama-cli``: if the resident server is not running, generation fails loudly.
"""

from __future__ import annotations

import inspect
import math
import os
import re
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from inferenceos.backends import llama_server_toolchain as srv
from inferenceos.backends.engine_env import parse_switches
from inferenceos.backends.llama_cpp_toolchain import (
    DeviceInfo,
    parse_device_info,
    parse_effective_settings,
    parse_effective_threads,
    parse_llama_memory,
    parse_load_mode,
)
from inferenceos.backends.process_control import (
    PeakRssAccumulator,
    resource_snapshot,
    wait_process_exit,
)
from inferenceos.contracts.lifecycle import (
    ProcessIdentity,
    ResidentGenerationMeasurement,
    ResidentLoadMeasurement,
    ResidentUnloadMeasurement,
)
from inferenceos.contracts.experiment import effective_kv_types
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import (
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
    TokenChoice,
)
from inferenceos.logging_config import get_logger

try:  # psutil is a dependency; guarded like process_control
    import psutil
except ImportError:  # pragma: no cover - defensive
    psutil = None  # type: ignore[assignment]

logger = get_logger("backends.resident_server")

# One cheap psutil memory sample per interval during a generation: bounded overhead
# that cannot materially distort the measured inference.
_RSS_SAMPLE_INTERVAL_S = 0.1

# H3: temporary API-key files are uniquely InferenceOS-owned. A crashed process cannot
# run its cleanup, so a bounded stale-file sweep runs at each server start. Review R5:
# age alone cannot tell an abandoned key from the key of a server that has simply been
# running for a long time (``start`` runs until Ctrl+C), and clients read that file to
# connect. So each key's name records the InferenceOS process that owns it (process id
# and start time); a key whose owner is still running is never removed, however old.
# Only keys whose owner is gone, or old-style keys without an owner, are removed by age.
_KEY_PREFIX = "inferenceos-key-"
_KEY_SUFFIX = ".txt"
_STALE_KEY_AGE_S = 24 * 3600.0
_MAX_KEY_SWEEP_FILES = 200
#: ``o<pid>x<start-ms>-``; ``x u`` (``o<pid>xu-``) means the start time could not be read
#: when the key was made: such a key is kept while any process has that id.
_KEY_OWNER_RE = re.compile(r"o(\d+)x(\d+|u)-")
#: Start times are recorded in whole milliseconds; allow for rounding.
_OWNER_START_TOLERANCE_MS = 2


def _process_start_ms(pid: int) -> int | None:
    """The process's start time in milliseconds, or ``None`` if it is not running.
    Raises when the answer is unknown (e.g. access denied) -- callers treat that as
    "still running"."""
    if psutil is None:
        raise RuntimeError("process information is unavailable")
    try:
        return int(round(float(psutil.Process(pid).create_time()) * 1000))
    except psutil.NoSuchProcess:
        return None


def key_file_prefix() -> str:
    """Name prefix for a new key file: the owner's process id and start time, so the
    stale-key sweep can tell a live owner's key from an abandoned one. The process id
    is always recorded. If the start time cannot be read, it is recorded as unknown
    (``u``) and the key is kept while any process has that id -- never treated like an
    ownerless old-style key (review of the R5 fix)."""
    pid = os.getpid()
    try:
        started: int | None = _process_start_ms(pid)
    except Exception:  # noqa: BLE001 - start time unknown; the id still protects it
        started = None
    return f"{_KEY_PREFIX}o{pid}x{'u' if started is None else started}-"


def key_owner_alive(name: str) -> bool | None:
    """Whether the InferenceOS process named in key file ``name`` is still running:
    ``True`` (running, or unknown -- kept to be safe), ``False`` (gone, or its id now
    belongs to a different process), ``None`` (an old-style name with no owner)."""
    match = _KEY_OWNER_RE.match(name[len(_KEY_PREFIX):])
    if match is None:
        return None
    pid = int(match.group(1))
    try:
        started = _process_start_ms(pid)
    except Exception:  # noqa: BLE001 - unknown: keep the key
        return True
    if started is None:
        return False
    if match.group(2) == "u":
        return True  # some process has that id; its start time was never recorded
    return abs(started - int(match.group(2))) <= _OWNER_START_TOLERANCE_MS


def restrict_api_key_file(path: Path, *, platform: str | None = None) -> None:
    """Restrict a temporary API-key file to owner-only (``0600``) on Unix.

    Windows has no equivalent POSIX mode; its behavior is unchanged (the file lives in
    the per-user ``%TEMP%``). Best-effort: a permissions failure must never prevent the
    server from starting, and it never widens access.
    """
    if (platform or sys.platform) == "win32":
        return
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover - defensive
        pass


def sweep_stale_key_files(
    *,
    temp_dir: Path | None = None,
    max_age_s: float = _STALE_KEY_AGE_S,
    now: float | None = None,
) -> int:
    """Remove stale InferenceOS-owned temporary API-key files. Never raises.

    Only files matching the exact ``inferenceos-key-*.txt`` pattern in the system
    temporary directory, older than ``max_age_s``, and not symlinks are removed, and
    never a key whose owning InferenceOS process is still running (review R5; see
    :func:`key_owner_alive`). Unrelated temporary files are untouched.
    """
    root = Path(temp_dir) if temp_dir is not None else Path(tempfile.gettempdir())
    if not root.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - max_age_s
    removed = 0
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if removed >= _MAX_KEY_SWEEP_FILES:
                    break
                name = entry.name
                if not (name.startswith(_KEY_PREFIX) and name.endswith(_KEY_SUFFIX)):
                    continue
                try:
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        continue
                    stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.st_mtime > cutoff:
                    continue
                if key_owner_alive(name):
                    continue
                try:
                    os.unlink(entry.path)
                    removed += 1
                except OSError:
                    continue
    except OSError:  # pragma: no cover - defensive
        return removed
    return removed


class ResidentStartCancelled(RuntimeError):
    """A start was cancelled by ``stop()`` before the server was published as ready."""


class ResidentBackendError(RuntimeError):
    """A resident backend operation failed and must not be silently retried."""


class ResidentGenerationProtocolError(ResidentBackendError):
    """Failed request with actual partial evidence, never a successful measurement.

    Development evaluations may inspect its counters for validity breaches even
    when the request's work is incomplete. Ordinary callers still receive an error.
    """

    def __init__(self, message: str, result: GenerationResult) -> None:
        super().__init__(message)
        self.result = result


class ResidentServerBackend(Protocol):
    """Structural interface used by the adapter and the deep-diagnosis lifecycle."""

    def start(self, model: ModelProfile, params: RuntimeParams) -> ResidentLoadMeasurement:
        ...

    def stop(self) -> ResidentUnloadMeasurement:
        ...

    def cleanup(self) -> None:
        """Release resources for a (possibly already-dead) target without raising."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...

    def is_running(self) -> bool:
        ...

    @property
    def pid(self) -> int | None:
        ...

    @property
    def identity(self) -> ProcessIdentity | None:
        ...

    @property
    def base_url(self) -> str | None:
        ...

    @property
    def last_load(self) -> ResidentLoadMeasurement | None:
        ...

    @property
    def last_unload(self) -> ResidentUnloadMeasurement | None:
        ...

    @property
    def last_generation(self) -> ResidentGenerationMeasurement | None:
        ...

    def telemetry(self) -> BackendTelemetry:
        ...


def token_choices_from(chunk: dict[str, object]) -> list[TokenChoice]:
    """The chosen words and their probabilities in one streamed chunk.

    llama-server sends ``completion_probabilities``: per word ``token`` with ``logprob``
    and ``top_logprobs`` (or ``prob`` / ``top_probs`` when probabilities after sampling
    were asked for). Anything else is ignored.
    """
    entries = chunk.get("completion_probabilities")
    if not isinstance(entries, list):
        return []
    result: list[TokenChoice] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        token = entry.get("token")
        prob = _probability(entry)
        if not isinstance(token, str) or prob is None:
            continue
        others = entry.get("top_logprobs", entry.get("top_probs"))
        runner_up: float | None = None
        if isinstance(others, list):
            values = []
            for other in others:
                if not isinstance(other, dict):
                    continue
                same = other.get("id") == entry.get("id") if "id" in entry else other.get("token") == token
                p = _probability(other)
                if p is not None and not same:
                    values.append(p)
            runner_up = max(values) if values else None
        result.append(TokenChoice(token=token, prob=prob, runner_up_prob=runner_up))
    return result


def _probability(entry: dict[str, object]) -> float | None:
    if isinstance(entry.get("prob"), (int, float)):
        return float(entry["prob"])  # type: ignore[arg-type]
    logprob = entry.get("logprob")
    if isinstance(logprob, (int, float)):
        return float(math.exp(logprob))
    return None


@dataclass
class _StopOperation:
    process: srv.LlamaServerProcess
    pid: int | None
    keys: tuple[Path, ...]
    owner: int = field(default_factory=threading.get_ident)
    done: threading.Event = field(default_factory=threading.Event)
    result: ResidentUnloadMeasurement | None = None


class LlamaServerResidentBackend:
    """Concrete resident backend over an authenticated loopback ``llama-server``."""

    def __init__(
        self,
        *,
        server: Path,
        host: str = "127.0.0.1",
        port: int = 0,
        readiness_timeout_s: float = 180.0,
        request_timeout_s: float = 600.0,
        process_factory: type[srv.LlamaServerProcess] | None = None,
    ) -> None:
        self._server = server
        self._host = host
        self._configured_port = port
        self._readiness_timeout_s = readiness_timeout_s
        self._request_timeout_s = request_timeout_s
        self._process_factory = process_factory or srv.LlamaServerProcess
        self._process: srv.LlamaServerProcess | None = None
        self._client: srv.LlamaServerClient | None = None
        self._key_file: Path | None = None
        #: Key files whose deletion failed (e.g. another program holds them open). They
        #: stay remembered and are retried at every later cleanup and start.
        self._undeleted_keys: list[Path] = []
        #: The last server command line started (for evidence; shown only redacted).
        self.last_command: list[str] | None = None
        #: The engine's raw counters from the last answer (see ``generate``).
        self.last_response_counters: dict[str, object] = {}
        self._model: ModelProfile | None = None
        self._params = RuntimeParams()
        self._port: int | None = None
        self._load: ResidentLoadMeasurement | None = None
        self._unload: ResidentUnloadMeasurement | None = None
        self._generation: ResidentGenerationMeasurement | None = None
        self._telemetry = BackendTelemetry()
        self._device: DeviceInfo | None = None
        # Startup lifecycle (review 10). ``idle`` -> ``starting`` -> ``ready`` ->
        # ``stopping`` -> ``idle``. The process is owned (``_owned``) from the moment it
        # is created, so ``stop()`` can end a load that has not finished; a cancelled
        # start never publishes a ready server. One lock guards every transition.
        self._lifecycle = threading.RLock()
        # Never hold the lifecycle lock across process waits. Starting-thread
        # unwind and cancellation may both terminate the same captured process.
        self._termination = threading.RLock()
        self._stop_operation: _StopOperation | None = None
        self._state = "idle"
        self._owned: srv.LlamaServerProcess | None = None
        self._cancel = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        #: Bumped by every stop: a generation started before a stop sees it and ends.
        self._epoch = 0
        #: A process whose exit could not be confirmed: no new start until it is gone.
        self._unconfirmed_pid: int | None = None
        #: Set by ``halt()`` (a user stop): no start is allowed until ``resume()``, so a
        #: stop that lands between the two halves of a reload cannot be undone by them.
        self._halted = False
        self._starter: int | None = None
        #: Bounded waits (seconds): a start to unwind after a stop, a process to exit
        #: after it was told to stop, a previous stop / process to finish before a new
        #: start. Attributes so tests can shorten them; behaviour is the same.
        self.unwind_timeout_s = 30.0
        self.exit_timeout_s = 10.0
        self.previous_gone_timeout_s = 30.0

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #
    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def identity(self) -> ProcessIdentity | None:
        return self._process.identity() if self._process is not None else None

    @property
    def base_url(self) -> str | None:
        return self._client.base_url if self._client is not None else None

    @property
    def client(self) -> srv.LlamaServerClient | None:
        return self._client

    @property
    def unconfirmed_pid(self) -> int | None:
        """A process this backend started whose exit it could not confirm (None: none).
        A cleared process handle is not a confirmation; only this is."""
        return self._unconfirmed_pid

    def confirm_previous_gone(self, timeout_s: float | None = None) -> bool:
        """Wait (bounded) for a stop in progress and for an unconfirmed process to exit;
        True when nothing this backend started is left unconfirmed. Never raises for an
        unconfirmed process (the identity stays recorded)."""
        try:
            self._wait_previous_gone(timeout_s)
        except ResidentBackendError:
            return False
        return self._unconfirmed_pid is None

    @property
    def undeleted_key_files(self) -> list[Path]:
        """Key files this backend made whose deletion has failed so far."""
        return list(self._undeleted_keys)

    @property
    def api_key_file(self) -> Path | None:
        """The owner-only file holding this run's password (deleted at stop)."""
        return self._key_file

    @property
    def last_load(self) -> ResidentLoadMeasurement | None:
        return self._load

    @property
    def last_unload(self) -> ResidentUnloadMeasurement | None:
        return self._unload

    @property
    def last_generation(self) -> ResidentGenerationMeasurement | None:
        return self._generation

    def telemetry(self) -> BackendTelemetry:
        return self._telemetry

    @property
    def last_device(self) -> DeviceInfo | None:
        return self._device

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def state(self) -> str:
        """``idle``, ``starting``, ``ready`` or ``stopping``."""
        return self._state

    def is_active(self) -> bool:
        """True while this backend owns a process (starting, ready or stopping)."""
        return self._state != "idle" or self.is_running()

    @property
    def halted(self) -> bool:
        return self._halted

    def halt(self) -> ResidentUnloadMeasurement | None:
        """A stop that sticks: end whatever runs (a start in progress or a ready server)
        and refuse any new start until :meth:`resume`. Safe in any state, from any
        thread, more than once. Returns the stop's measurement, or None if nothing ran."""
        with self._lifecycle:
            self._halted = True
            active = self._state != "idle" or self._process is not None
        if not active:
            return None
        try:
            return self.stop()
        except ResidentBackendError:
            return None  # it reached idle on its own in the meantime

    def resume(self, timeout_s: float | None = None, *, unless: Callable[[], bool] | None = None) -> bool:
        """Allow starts again, once the previous process is confirmed gone.

        ``unless`` is checked under the lifecycle lock right before the halt is lifted;
        if it returns True the backend stays halted (a user stop that raced this resume
        wins). Returns whether starts are allowed again."""
        self._wait_previous_gone(timeout_s)
        with self._lifecycle:
            if unless is not None and unless():
                return False
            self._halted = False
            return True

    def server_log(self) -> str:
        """The server's recent log lines (engine diagnostics such as timing blocks)."""
        process = self._process
        return process.log_text() if process is not None else ""

    def engine_settings(self) -> dict[str, str]:
        """Engine variables the running server was started with."""
        settings = getattr(self._process, "engine_settings", None)
        return dict(settings) if isinstance(settings, dict) else {}

    def server_startup_log(self) -> str:
        """The server's first log lines (devices, buffers), kept however long it ran."""
        process = self._process
        reader = getattr(process, "startup_text", None)
        return reader() if callable(reader) else ""

    # ------------------------------------------------------------------ #
    # Load / unload                                                        #
    # ------------------------------------------------------------------ #
    def start_if_current(
        self, model: ModelProfile, params: RuntimeParams, *, current: Callable[[], bool],
    ) -> ResidentLoadMeasurement:
        """Admit adapter work under the process owner's lifecycle lock."""
        return self.start(model, params, _current=current)

    def start(
        self, model: ModelProfile, params: RuntimeParams, *, _current: Callable[[], bool] | None = None,
    ) -> ResidentLoadMeasurement:
        # A cancelled caller must not wait for an unrelated replacement to unload.
        # Recheck below as well: this preflight alone cannot authorize dispatch.
        if _current is not None and not _current():
            raise ResidentStartCancelled("resident start superseded before preparation")
        self._wait_previous_gone()
        with self._lifecycle:
            if _current is not None and not _current():
                raise ResidentStartCancelled("resident start superseded before admission")
            if self._halted:
                raise ResidentStartCancelled("resident server start refused: stopped until resumed")
            if self._state != "idle" or self.is_running():
                raise ResidentBackendError("a resident server is already running")
            self._state = "starting"
            self._starter = threading.get_ident()
            self._cancel.clear()
            self._idle.clear()
        try:
            return self._start(model, params)
        except BaseException:
            # Failure or cancellation: terminate only the process this start owns and
            # release the credentials; never leave a half-started server behind. The
            # backend always returns to idle, and anything not released stays on record
            # (unconfirmed PID, undeleted key files) for the next cleanup. An ordinary
            # error during this cleanup is logged and the original exception propagates;
            # a new interruption (Ctrl+C) during it propagates instead, with the original
            # failure kept as its ``__context__``.
            try:
                try:
                    with self._lifecycle:
                        owned, self._owned = self._owned, None
                    if owned is not None:
                        self._note_pending_exit(getattr(owned, "pid", None))
                        try:
                            self._terminate_process(owned, timeout_s=5.0)
                        except Exception:  # noqa: BLE001 - cleanup must not mask the cause
                            pass
                        self._confirm_gone(getattr(owned, "pid", None))
                finally:
                    try:
                        self._cleanup_key_file()
                    except Exception as exc:  # noqa: BLE001 - kept on record; never masks the cause
                        logger.warning("password file cleanup after a failed start: %s", exc)
            finally:
                with self._lifecycle:
                    if self._state in ("starting", "stopping"):
                        self._state = "idle"
                        self._idle.set()
            raise

    def _check_cancel(self, where: str) -> None:
        if self._cancel.is_set():
            raise ResidentStartCancelled(f"resident server start cancelled {where}")

    def _wait_previous_gone(self, timeout_s: float | None = None) -> None:
        """A start never overlaps a stop in progress, nor a process whose exit was not
        confirmed (recovery waits until the previous process is gone)."""
        if timeout_s is None:
            timeout_s = self.previous_gone_timeout_s
        if not self._idle.wait(timeout=timeout_s):
            raise ResidentBackendError("the previous resident server is still stopping")
        with self._lifecycle:
            pid = self._unconfirmed_pid
        if pid is not None:
            if wait_process_exit(pid, timeout_s=timeout_s):
                with self._lifecycle:
                    if self._unconfirmed_pid == pid:
                        self._unconfirmed_pid = None
            else:
                raise ResidentBackendError(
                    f"the previous resident server (pid {pid}) has not exited; not starting another"
                )

    def _note_pending_exit(self, pid: int | None) -> None:
        """Record ``pid`` as not yet confirmed gone *before* asking it to stop, so an
        interruption anywhere in a stop keeps its identity (cleared on confirmation). An
        older unconfirmed identity is never overwritten."""
        with self._lifecycle:
            if pid and self._unconfirmed_pid is None:
                self._unconfirmed_pid = pid

    def _confirm_gone(self, pid: int | None) -> bool:
        if not pid:
            return True
        gone = bool(wait_process_exit(pid, timeout_s=self.exit_timeout_s))
        with self._lifecycle:
            if not gone:
                self._note_pending_exit(pid)
            elif self._unconfirmed_pid == pid:
                self._unconfirmed_pid = None
        return gone

    def _terminate_process(self, process: srv.LlamaServerProcess, *, timeout_s: float) -> tuple[int | None, str, bool]:
        if not self._termination.acquire(timeout=max(timeout_s, 0.0)):
            raise ResidentBackendError("resident process termination is already in progress")
        try:
            return process.stop(timeout_s=timeout_s)
        finally:
            self._termination.release()

    def _start(self, model: ModelProfile, params: RuntimeParams) -> ResidentLoadMeasurement:
        model_path = Path(model.path)
        if not model_path.is_file():
            raise ResidentBackendError(f"model file not found: {model_path}")
        # A prior crashed target may have left its temporary key file behind; never
        # leak it across a recovery/restart. A bounded sweep also clears stale
        # InferenceOS-owned keys left by previously crashed processes.
        sweep_stale_key_files()
        self._cleanup_key_file()

        before = resource_snapshot(None)
        api_key = srv.generate_api_key()
        # The whole preparation of the password file runs under the lifecycle lock, the
        # lock every cleanup holds for its key-file bookkeeping. A cleanup or stop that
        # arrives meanwhile waits until the file is complete and on record, then deletes
        # it; after this block the start never writes the file again, so a cleanup can
        # never be followed by a re-created, untracked file. A start already cancelled
        # creates no file at all.
        with self._lifecycle:
            self._check_cancel("before the password file was created")
            handle, key_path = tempfile.mkstemp(prefix=key_file_prefix(), suffix=_KEY_SUFFIX)
            try:
                # Owned from the moment it exists: if anything below fails or is
                # interrupted, the failed-start cleanup deletes it (or keeps it on record).
                self._key_file = Path(key_path)
            finally:
                os.close(handle)
            key_file = Path(key_path)
            key_file.write_text(api_key + "\n", encoding="utf-8")
            # Unix: owner-only permissions for the credential material.
            restrict_api_key_file(key_file)

        port = self._configured_port or srv.pick_free_port(self._host)
        options = _options_from(model_path, params)
        command = srv.build_server_command(
            self._server, options, host=self._host, port=port, api_key_file=key_file
        )
        logger.info(
            "Starting resident llama-server for model %s on %s:%s (key redacted)",
            model.name,
            self._host,
            port,
        )
        # The exact options, so a report can be checked against what really ran.
        self.last_command = list(command)
        logger.info("llama-server options: %s", " ".join(redacted_command(command)))
        if options.environment:
            logger.info(
                "llama-server engine switches: %s",
                ";".join(f"{k}={v}" for k, v in sorted(options.environment.items())),
            )
            process = self._process_factory(command, environment=options.environment)
        else:
            process = self._process_factory(command)
        # Owned before it runs: a stop from now on terminates it.
        with self._lifecycle:
            self._owned = process
        self._check_cancel("before the process was created")
        started = time.perf_counter()
        pid = process.start()
        self._check_cancel("while the process was starting")
        # What the engine really reads (inherited variables included, diagnostic ones
        # removed unless asked for): recorded so a speed test can be checked.
        settings = getattr(process, "engine_settings", None)
        if isinstance(settings, dict) and settings:
            logger.info(
                "llama-server engine settings in effect: %s",
                ";".join(f"{k}={v}" for k, v in sorted(settings.items())),
            )
        client = srv.LlamaServerClient(
            host=self._host, port=port, api_key=api_key, timeout_s=self._request_timeout_s
        )
        wait = process.wait_ready
        try:
            if "should_stop" in inspect.signature(wait).parameters:
                ready_s = wait(client, timeout_s=self._readiness_timeout_s, should_stop=self._cancel.is_set)
            else:  # process doubles without a cancellable wait
                ready_s = wait(client, timeout_s=self._readiness_timeout_s)
        except Exception as exc:
            if self._cancel.is_set():
                # The wait ended because of a stop: report it as a cancellation, so no
                # caller treats it as an ordinary failure (e.g. restores a setup).
                raise ResidentStartCancelled(
                    "resident server start cancelled while waiting for readiness"
                ) from exc
            raise
        self._check_cancel("while waiting for readiness")
        startup_s = time.perf_counter() - started
        after = resource_snapshot(pid)
        identity = process.identity()
        log_text = process.log_text()
        device = parse_device_info(log_text)
        memory = parse_llama_memory(log_text)
        effective_threads = parse_effective_threads(log_text)
        # Effective weight-load mode from backend evidence only (startup log); ``None``
        # when the log does not carry it (never inferred from the requested params).
        mmap_effective = parse_load_mode(log_text)
        self._device = device
        self._telemetry = BackendTelemetry(
            device=device.device if device is not None else None,
            gpu_layers_offloaded=device.layers_offloaded if device is not None else None,
            gpu_layers_total=device.layers_total if device is not None else None,
            kv_cache_bytes=memory.kv_cache_bytes if memory is not None else None,
            model_buffer_bytes=memory.model_buffer_bytes if memory is not None else None,
            context_length=params.context_length,
            effective_threads=effective_threads,
            mmap_effective=mmap_effective,
            observed=parse_effective_settings(log_text),
        )
        capacity = self._telemetry.observed.get("context_length")
        if (type(capacity) is int and params.context_length is not None
                and capacity < params.context_length):
            raise ResidentBackendError(
                f"engine slot context capacity {capacity} is below requested context "
                f"{params.context_length}; refusing to run with a smaller context"
            )
        from inferenceos.backends.observed import observed_configuration
        from inferenceos.contracts.experiment import RuntimeConfiguration

        requested_fields = {name: getattr(params, name) for name in RuntimeConfiguration.model_fields}
        _, contradictions, _ = observed_configuration(
            RuntimeConfiguration.model_validate(requested_fields), self._telemetry)
        if contradictions:
            raise ResidentBackendError("requested settings contradicted by engine: " + "; ".join(contradictions))
        resident_verified = bool(
            process.is_alive()
            and identity is not None
            and after.exists
            and (after.rss_bytes or 0) > 0
        )
        measurement = ResidentLoadMeasurement(
            model_ref=model.name,
            model_path=str(model_path),
            pid=pid,
            identity=identity,
            host=self._host,
            port=port,
            ready=ready_s is not None,
            startup_s=startup_s,
            ready_s=ready_s,
            before=before,
            after=after,
            resident_verified=resident_verified,
            notes=[
                "resident process started and authenticated health check passed",
                "residency verified via OS process identity and RSS measurement",
            ],
        )
        with self._lifecycle:
            # The last moment a stop can still cancel this start: after this the server
            # is published as ready and a stop unloads it normally.
            self._check_cancel("immediately after readiness")
            self._process = process
            self._owned = None
            self._client = client
            self._model = model
            self._params = params
            self._port = port
            self._load = measurement
            self._state = "ready"
        return measurement

    def stop_if_current(self, *, current: Callable[[], bool]) -> ResidentUnloadMeasurement:
        return self.stop(_current=current)

    def stop(self, *, _current: Callable[[], bool] | None = None) -> ResidentUnloadMeasurement:
        """Stop the server in any state: a start in progress is cancelled (its owned
        process tree terminated, no ready server published), a ready server unloaded.
        Safe to call from another thread and more than once."""
        result = self._request_stop(wait_start=True, missing_ok=False, current=_current)
        assert result is not None
        return result

    def _request_stop(
        self, *, wait_start: bool, missing_ok: bool, current: Callable[[], bool] | None = None,
    ) -> ResidentUnloadMeasurement | None:
        # Claim the generation under the same lock as start/publication. cleanup
        # must never independently publish idle while a stop can still mutate it.
        run = False
        operation: _StopOperation | None = None
        with self._lifecycle:
            if current is not None and not current():
                raise ResidentStartCancelled("resident stop superseded before admission")
            state = self._state
            process: srv.LlamaServerProcess | None
            keys = tuple(dict.fromkeys([*self._undeleted_keys, *([self._key_file] if self._key_file else [])]))
            if state == "starting":
                self._cancel.set()
                process = self._owned
                self._epoch += 1
                cancellation_epoch = self._epoch
            elif state == "stopping":
                process = None
                operation = self._stop_operation
            elif self._process is not None or (self._stop_operation is not None
                    and self._unconfirmed_pid is not None
                    and self._stop_operation.pid == self._unconfirmed_pid):
                # A failed stop retains its process owner for a later cleanup;
                # clearing the public ready handle is not a release confirmation.
                process = self._process if self._process is not None else self._stop_operation.process  # type: ignore[union-attr]
                operation = _StopOperation(process, process.pid, keys)
                self._stop_operation = operation
                self._note_pending_exit(operation.pid)
                self._state = "stopping"
                self._idle.clear()
                self._epoch += 1
                run = True
            else:
                if missing_ok:
                    self._cleanup_key_file(paths=keys)
                    return None
                raise ResidentBackendError("no resident server to unload")
        if state == "stopping":
            # This operation's event/result cannot be reset or replaced by a new
            # generation. A bounded wait never authorizes clearing shared state.
            if operation is not None and operation.owner != threading.get_ident():
                if operation.done.wait(timeout=self.unwind_timeout_s) and operation.result is not None:
                    return operation.result
            if missing_ok:
                return None
            raise ResidentBackendError("the resident stop has not completed successfully")
        if state == "starting":
            if wait_start:
                return self._cancel_start(process, epoch=cancellation_epoch)
            # Cleanup during startup does not wait for the starter (it may be
            # called from its callback). Only the starter can publish idle.
            if process is not None:
                self._note_pending_exit(getattr(process, "pid", None))
                try:
                    try:
                        self._terminate_process(process, timeout_s=5.0)
                    except Exception as exc:
                        logger.warning("resident startup cleanup failed: %s", exc)
                finally:
                    try:
                        self._confirm_gone(getattr(process, "pid", None))
                    finally:
                        self._cleanup_key_file(paths=keys)
            else:
                self._cleanup_key_file(paths=keys)
            return None
        assert run and operation is not None
        return self._finish_stop(operation)

    def _finish_stop(self, operation: _StopOperation) -> ResidentUnloadMeasurement:
        process, pid = operation.process, operation.pid
        started = time.perf_counter()
        failure: BaseException | None = None
        try:
            before = resource_snapshot(pid)
            requested = datetime.now(timezone.utc)
            exit_code, method, exited = self._terminate_process(process, timeout_s=self.exit_timeout_s)
            exited = bool(exited and wait_process_exit(pid, timeout_s=self.exit_timeout_s)) if pid else exited
            after = resource_snapshot(pid)
            cleanup_complete = bool(exited and not after.exists)
            with self._lifecycle:
                if cleanup_complete and self._unconfirmed_pid == pid:
                    self._unconfirmed_pid = None
                credential_removed = self._cleanup_key_file(paths=operation.keys)
                measurement = ResidentUnloadMeasurement(
                    pid=pid, requested_s=requested.timestamp(), exited=exited,
                    exit_code=exit_code, stop_method=method, before=before, after=after,
                    unload_s=time.perf_counter() - started, cleanup_complete=cleanup_complete,
                    credential_removed=credential_removed,
                    notes=["resident process tree terminated and process exit verified"
                           if cleanup_complete else "resident process did not fully exit; cleanup incomplete"]
                    + ([] if credential_removed else ["the password file could not be deleted: "
                        + ", ".join(str(path) for path in self._undeleted_keys)
                        + " (kept on record; retried at the next cleanup or start)"]),
                )
                operation.result = measurement
                self._unload = measurement
            return measurement
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                with self._lifecycle:
                    if self._process is process:
                        self._process = None
                        self._client = None
                    if failure is not None:
                        self._cleanup_key_file(paths=operation.keys)
            except Exception as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"resident stop credential cleanup also failed: {cleanup_error}")
            finally:
                with self._lifecycle:
                    if self._stop_operation is operation:
                        self._state = "idle"
                        self._idle.set()
                    operation.done.set()

    def _cancel_start(self, process: srv.LlamaServerProcess | None, *, epoch: int) -> ResidentUnloadMeasurement:
        """Cancel a start in progress: terminate the owned process (if it exists yet)
        and wait until the start has unwound (bounded)."""
        pid = getattr(process, "pid", None) if process is not None else None
        self._note_pending_exit(pid)
        requested = datetime.now(timezone.utc)
        started = time.perf_counter()
        exit_code: int | None = None
        method, exited = "none", True
        if process is not None:
            try:
                exit_code, method, exited = self._terminate_process(process, timeout_s=self.exit_timeout_s)
            except Exception as exc:  # noqa: BLE001 - reported in the measurement
                method, exited = f"stop failed: {type(exc).__name__}", False
        # Called from the starting thread itself (a callback): it unwinds after we return.
        same_thread = self._starter == threading.get_ident()
        unwound = True if same_thread else self._idle.wait(timeout=self.unwind_timeout_s)
        gone = self._confirm_gone(pid) if exited else False
        measurement = ResidentUnloadMeasurement(
            pid=pid,
            requested_s=requested.timestamp(),
            exited=bool(exited),
            exit_code=exit_code,
            stop_method=method,
            before=resource_snapshot(None),
            after=resource_snapshot(pid),
            unload_s=time.perf_counter() - started,
            cleanup_complete=bool(unwound and gone),
            notes=["start cancelled before the server was ready"
                   + ("" if unwound else f"; the start had not unwound within {self.unwind_timeout_s:g} s")],
        )
        with self._lifecycle:
            # A new start may begin once the starter unwinds. This late diagnostic
            # must not replace evidence belonging to that subsequent generation.
            if self._epoch == epoch and self._cancel.is_set():
                self._unload = measurement
        return measurement

    def cleanup(self) -> None:
        """Release the target and its Job Object handle after a loss, without raising.

        Safe to call when the process already exited (``stop`` is tolerant of an
        already-exited process) and when there is no process at all.
        """
        try:
            self._request_stop(wait_start=False, missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - process/key evidence stays on record
            logger.warning("resident cleanup failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #
    def generate(self, request: GenerationRequest) -> GenerationResult:
        # Evidence belongs to this request even when it fails before producing a
        # result. Never expose the preceding answer as this attempt's measurement.
        self._generation = None
        self.last_response_counters = {}
        with self._lifecycle:
            client = self._client
            process = self._process
            epoch = self._epoch
            # Never answer on a server that is still starting (not yet published) or
            # being stopped.
            ready = self._state not in ("starting", "stopping")
        if not ready or client is None or process is None or not process.is_alive():
            raise ResidentBackendError(
                "resident server is not running; refusing to spawn a replacement process"
            )
        pid = process.pid
        if pid is None:  # pragma: no cover - a live process always has a pid
            raise ResidentBackendError("resident server has no pid")
        submitted = datetime.now(timezone.utc)
        started = time.perf_counter()
        first_token_at: datetime | None = None
        ttft_s: float | None = None
        pieces: list[str] = []
        choices: list[TokenChoice] = []
        final: dict[str, object] | None = None
        counter_evidence: dict[str, object] = {}
        generated: int | None = None
        prompt_tokens: int | None = None
        prompt_total: int | None = None
        prompt_cached: int | None = None
        # True peak-vs-final memory over the generation interval: sampled at a bounded
        # cadence and combined with the OS peak working set when available.
        peak_tracker = PeakRssAccumulator()
        peak_tracker.observe(pid)
        next_sample = started + _RSS_SAMPLE_INTERVAL_S
        try:
            for chunk in client.stream_completion(
                prompt=request.prompt,
                n_predict=request.n_predict,
                seed=request.seed,
                temperature=request.temperature,
                stop=list(request.stop),
                json_schema=None,
                ignore_eos=bool(request.ignore_eos),
                cache_prompt=bool(request.cache_prompt),
                **({"n_probs": int(request.n_probs)} if request.n_probs > 0 else {}),
            ):
                if self._epoch != epoch:
                    # Stopped (cancelled) while answering: end here.
                    raise ResidentBackendError("resident generation stopped by a stop request")
                if request.n_probs > 0:
                    choices.extend(token_choices_from(chunk))
                content = chunk.get("content")
                if isinstance(content, str) and content:
                    if first_token_at is None:
                        first_token_at = datetime.now(timezone.utc)
                        ttft_s = time.perf_counter() - started
                    pieces.append(content)
                if type(chunk.get("tokens_predicted")) is int:
                    generated = chunk["tokens_predicted"]
                if type(chunk.get("tokens_evaluated")) is int:
                    prompt_tokens = chunk["tokens_evaluated"]
                    prompt_total = chunk["tokens_evaluated"]
                if any(key in chunk for key in ("timings", "tokens_evaluated", "tokens_predicted")):
                    counter_evidence = chunk
                # Not ``tokens_cached``: in b11139 that is the slot's size after the answer
                # (occupancy, including what was just written), not prompt work reused.
                if chunk.get("stop") is True:
                    final = chunk
                now = time.perf_counter()
                if now >= next_sample:
                    peak_tracker.observe(pid)
                    next_sample = now + _RSS_SAMPLE_INTERVAL_S
        except srv.ServerRequestError as exc:
            raise ResidentBackendError(f"resident generation failed: {exc}") from exc
        completed = datetime.now(timezone.utc)
        total_s = time.perf_counter() - started
        if not process.is_alive():
            raise ResidentBackendError("resident server died during generation")
        peak_tracker.observe(pid)

        last = final or counter_evidence
        timings = last.get("timings")
        decode_tps: float | None = None
        prompt_tps: float | None = None
        prefill_s: float | None = None
        if isinstance(timings, dict):
            value = timings.get("predicted_per_second")
            decode_tps = float(value) if isinstance(value, (int, float)) else None
            # The engine's own prompt-reading time and speed (llama-server sends them
            # in the last streamed chunk; 4 Oct 2026: they were read for writing speed
            # but dropped for reading, so tune worked around a "missing" reading speed).
            reading = timings.get("prompt_per_second")
            prompt_tps = float(reading) if isinstance(reading, (int, float)) and reading > 0 else None
            prompt_ms = timings.get("prompt_ms")
            if isinstance(prompt_ms, (int, float)) and prompt_ms >= 0:
                prefill_s = float(prompt_ms) / 1000.0
            if generated is None and type(timings.get("predicted_n")) is int:
                generated = timings["predicted_n"]
            if type(timings.get("prompt_n")) is int:
                prompt_tokens = timings["prompt_n"]
            if type(timings.get("cache_n")) is int:
                prompt_cached = timings["cache_n"]
        # The engine's own counters, raw, for evidence (prompt-reuse evaluation): what it
        # processed, reused and wrote, whether it said it finished, and in which slot.
        timing_map = timings if isinstance(timings, dict) else {}
        self.last_response_counters = {
            "tokens_evaluated": last.get("tokens_evaluated"),
            "tokens_predicted": last.get("tokens_predicted"),
            "prompt_n": timing_map.get("prompt_n"),
            "cache_n": timing_map.get("cache_n"),
            "predicted_n": timing_map.get("predicted_n"),
            "stop_seen": bool(final is not None and final.get("stop") is True),
            "stop_type": last.get("stop_type"),
            "id_slot": last.get("id_slot"),
            "tokens_cached_occupancy": last.get("tokens_cached"),
        }
        # Native /completion sends stop:true on the terminal response (b11139
        # server_task_result_cmpl_final). Clean HTTP EOF, text, or timing fields
        # alone do not establish that the requested generation completed.
        protocol_problem: str | None = None
        if final is None:
            protocol_problem = "resident stream ended without an engine completion marker"
        counts = (("tokens_predicted", last.get("tokens_predicted")),
                  ("predicted_n", timing_map.get("predicted_n")),
                  ("cache_n", timing_map.get("cache_n")))
        for label, count in counts:
            if count is not None and (type(count) is not int or count < 0):
                protocol_problem = protocol_problem or f"engine {label} is not a nonnegative integer"
        if (last.get("tokens_predicted") is not None and timing_map.get("predicted_n") is not None
                and last["tokens_predicted"] != timing_map["predicted_n"]):
            protocol_problem = protocol_problem or "engine written-token counters disagree"
        if not request.cache_prompt and prompt_cached is not None and prompt_cached > 0:
            protocol_problem = protocol_problem or "fresh generation reported reused prompt work"
        if self._epoch != epoch:
            raise ResidentBackendError("resident generation stopped by a stop request")
        if (request.ignore_eos and not request.stop and request.n_predict > 0
                and (type(generated) is not int or generated != request.n_predict)):
            protocol_problem = protocol_problem or f"fixed-length generation requested {request.n_predict} tokens; engine reported {generated!r}"
        text = "".join(pieces)
        measurement = ResidentGenerationMeasurement(
            submitted_at=submitted,
            first_token_at=first_token_at,
            completed_at=completed,
            ttft_s=ttft_s,
            total_s=total_s,
            generated_tokens=generated,
            prompt_tokens=prompt_tokens,
            decode_tokens_per_sec=decode_tps,
            peak_rss_bytes=peak_tracker.peak_rss_bytes,
            final_rss_bytes=peak_tracker.final_rss_bytes,
            rss_sample_count=peak_tracker.sample_count,
            streamed=True,
            text=text[:4000],
            cache_prompt=bool(request.cache_prompt),
            notes=[
                "TTFT measured from request submission to first streamed content chunk",
                (
                    "peak RSS is the maximum of sampled RSS and the OS peak working set"
                    if peak_tracker.os_peak_available
                    else "peak RSS is the maximum sampled RSS over the generation interval"
                ),
                f"rss_samples={peak_tracker.sample_count}",
            ],
        )
        result = GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            prompt_tokens_per_sec=prompt_tps,
            decode_tokens_per_sec=decode_tps,
            model_load_s=None,
            prefill_s=prefill_s,
            decode_s=None,
            ttft_s=ttft_s,
            total_s=total_s,
            process_wall_s=total_s,
            peak_rss_bytes=peak_tracker.peak_rss_bytes,
            final_rss_bytes=peak_tracker.final_rss_bytes,
            peak_process_cpu_percent=peak_tracker.peak_process_cpu_percent,
            cpu_seconds=peak_tracker.cpu_seconds,
            context_switches_voluntary=peak_tracker.context_switches_voluntary,
            context_switches_involuntary=peak_tracker.context_switches_involuntary,
            context_switches_total=peak_tracker.context_switches_total,
            token_choices=choices if request.n_probs > 0 else None,
            prompt_tokens_cached=prompt_cached,
            prompt_tokens_total=prompt_total,
            cache_prompt=bool(request.cache_prompt),
        )
        if protocol_problem is not None:
            self.last_response_counters["protocol_problem"] = protocol_problem
            raise ResidentGenerationProtocolError(protocol_problem, result)
        self._generation = measurement
        return result

    def health(self) -> bool:
        """Whether the running engine answers its health check (False when not running)."""
        with self._lifecycle:
            client = self._client
            ready = self._state not in ("starting", "stopping")
        process = self._process
        if not ready or client is None or process is None or not process.is_alive():
            return False
        return client.health()

    def tokenize(self, content: str, *, add_special: bool = True) -> list[int]:
        """The running engine's tokens for ``content`` (for prompt-reuse evidence)."""
        with self._lifecycle:
            client = self._client
            ready = self._state not in ("starting", "stopping")
        if not ready or client is None:
            raise ResidentBackendError("resident server is not running")
        try:
            return client.tokenize(content, add_special=add_special)
        except srv.ServerRequestError as exc:
            raise ResidentBackendError(f"tokenize failed: {exc}") from exc

    def chat_request(self, messages: list[dict[str, str]], *, max_tokens: int, seed: int,
                     cache_prompt: bool | None) -> dict[str, object]:
        """One ordinary chat request to the running engine, greedy (temperature 0), and
        its whole reply (reuse-correctness evaluation only). ``cache_prompt`` None leaves
        the field out: the engine's default applies, as for an ordinary client."""
        with self._lifecycle:
            client = self._client
            ready = self._state not in ("starting", "stopping")
        if not ready or client is None:
            raise ResidentBackendError("resident server is not running")
        try:
            return client.chat_request(messages=messages, max_tokens=max_tokens, seed=seed, temperature=0.0,
                                       cache_prompt=cache_prompt)
        except srv.ServerRequestError as exc:
            raise ResidentBackendError(f"chat request failed: {exc}") from exc

    def abandon_generation(self, request: GenerationRequest, *, after_chunks: int,
                           idle_timeout_s: float = 30.0) -> dict[str, object]:
        """Start ``request`` and walk away after ``after_chunks`` streamed pieces of text
        (the client closes the connection, as an impatient user would), then wait (at most
        ``idle_timeout_s``) until the engine's ``/slots`` shows no slot processing.

        Returns ``pieces`` (text pieces received); ``ending``: ``"client cutoff"`` (the
        client left after ``after_chunks`` pieces, the only interruption), ``"engine
        finished"`` (the engine said it finished first), ``"stream ended early"`` (the
        stream ended without either) or ``"request failed"`` (``error`` says how);
        ``idle_confirmed``: True (every slot idle, the reply received within the wait),
        False (still processing at the limit, or no reply within it), None (the engine
        could not say); ``idle_detail``. Each ``/slots`` request is limited to the time
        left, and a reply that arrives after the limit is not accepted. Used only by the
        prompt-reuse evaluation."""
        with self._lifecycle:
            client = self._client
            ready = self._state not in ("starting", "stopping")
        if not ready or client is None:
            raise ResidentBackendError("resident server is not running")
        seen = 0
        ending = "stream ended early"
        error: str | None = None
        stream: Any = None
        try:
            stream = client.stream_completion(
                prompt=request.prompt, n_predict=request.n_predict, seed=request.seed,
                temperature=request.temperature, stop=list(request.stop), json_schema=None,
                ignore_eos=bool(request.ignore_eos), cache_prompt=bool(request.cache_prompt),
            )
            for chunk in stream:
                if chunk.get("stop") is True:
                    ending = "engine finished"
                    break
                content = chunk.get("content")
                if isinstance(content, str) and content:
                    seen += 1
                    if seen >= after_chunks:
                        ending = "client cutoff"
                        break
        except srv.ServerRequestError as exc:
            ending, error = "request failed", str(exc)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()  # closes the HTTP response: the engine sees the client leave
        idle: bool | None
        detail: str | None = None
        deadline = time.monotonic() + idle_timeout_s
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                idle, detail = False, f"a slot was still processing at the {idle_timeout_s:g} s limit"
                break
            try:
                slots = client.slots(timeout_s=max(left, 0.05))
            except (srv.ServerRequestError, OSError, ValueError) as exc:
                if time.monotonic() >= deadline:
                    idle, detail = False, f"no slot reply within the {idle_timeout_s:g} s limit"
                else:
                    idle, detail = None, f"the slots could not be read: {exc}"
                break
            if time.monotonic() > deadline:
                idle, detail = False, f"the slot reply arrived after the {idle_timeout_s:g} s limit"
                break
            if not slots or not all(isinstance(s, dict) and type(s.get("is_processing")) is bool for s in slots):
                # b11139 sends a true/false ``is_processing`` for every slot; anything else
                # (missing, null, 0, "") is unknown, never idle.
                idle, detail = None, "the slots do not all report a true/false is_processing"
                break
            if not any(s["is_processing"] for s in slots):
                idle = True
                break
            time.sleep(max(min(0.2, deadline - time.monotonic()), 0.0))
        return {"pieces": seen, "ending": ending, "error": error, "idle_confirmed": idle, "idle_detail": detail}

    def _cleanup_key_file(self, *, paths: tuple[Path, ...] | None = None) -> bool:
        """Delete this backend's key file and any earlier one whose deletion failed.
        Returns True when none is left. A file that cannot be deleted (for example
        another program holds it open on Windows) is logged and kept on record in
        ``undeleted_key_files``, so it is retried later and never reported as removed.

        Each path is put on that record *before* its deletion is attempted and taken off
        only once the deletion succeeded, so an interruption (Ctrl+C) at any point leaves
        every unfinished deletion on record for the next cleanup.

        Every caller (a failed start, ``stop``, ``cleanup``, a new start) holds the
        lifecycle lock (re-entrant) for the whole bookkeeping, so two overlapping cleanups
        never work on the same record at once."""
        with self._lifecycle:
            if paths is None:
                paths = tuple(dict.fromkeys([*self._undeleted_keys, *([self._key_file] if self._key_file else [])]))
            if self._key_file in paths:
                if self._key_file is not None and self._key_file not in self._undeleted_keys:
                    self._undeleted_keys.append(self._key_file)
                self._key_file = None
            for path in paths:
                if path not in self._undeleted_keys:
                    self._undeleted_keys.append(path)
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("could not delete a temporary password file %s: %s", path, exc)
                    continue
                if path in self._undeleted_keys:
                    self._undeleted_keys.remove(path)
            return not self._undeleted_keys and self._key_file is None


def redacted_command(command: list[str]) -> list[str]:
    """The server command without the model's folder or the key file location."""
    shown: list[str] = []
    hide_next = False
    for part in command[1:]:
        if hide_next:
            shown.append("<hidden>")
            hide_next = False
            continue
        shown.append(part)
        if part in ("--api-key-file", "--api-key"):
            hide_next = True
    return shown


def _options_from(model_path: Path, params: RuntimeParams) -> srv.ServerOptions:
    cache_k, cache_v = effective_kv_types(
        params.kv_cache_type, params.kv_cache_type_k, params.kv_cache_type_v
    )
    return srv.ServerOptions(
        model_path=model_path,
        n_gpu_layers=(
            params.gpu_offload_layers if params.gpu_offload_layers is not None else 99
        ),
        context_length=params.context_length,
        threads=params.threads or None,
        threads_batch=params.threads_batch or None,
        environment=parse_switches(params.engine_switches),
        batch_size=params.batch_size,
        ubatch_size=params.ubatch_size,
        flash_attention=params.flash_attention,
        device=params.device,
        load_mode=("mmap" if params.use_mmap else "none") if params.use_mmap is not None else None,
        cache_type_k=cache_k,
        cache_type_v=cache_v,
        tensor_overrides=params.placement_overrides,
        cache_reuse=params.cache_reuse,
        swa_full=bool(params.swa_full),
        log_verbosity=params.log_verbosity,
        ignore_eos=bool(params.ignore_eos),
    )
