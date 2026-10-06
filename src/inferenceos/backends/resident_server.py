"""Real resident llama.cpp server backend used by :class:`LlamaCppResidentAdapter`.

``LlamaServerResidentBackend`` owns one real ``llama-server`` process and exposes
measured load/unload/generation evidence. It never falls back to spawning
``llama-cli``: if the resident server is not running, generation fails loudly.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from inferenceos.backends import llama_server_toolchain as srv
from inferenceos.backends.engine_env import parse_switches
from inferenceos.backends.llama_cpp_toolchain import (
    DeviceInfo,
    parse_device_info,
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
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import (
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
    TokenChoice,
)
from inferenceos.logging_config import get_logger

logger = get_logger("backends.resident_server")

# One cheap psutil memory sample per interval during a generation: bounded overhead
# that cannot materially distort the measured inference.
_RSS_SAMPLE_INTERVAL_S = 0.1

# H3: temporary API-key files are uniquely InferenceOS-owned. A crashed process cannot
# run its cleanup, so a bounded stale-file sweep runs at each server start. The age
# threshold is far longer than any real server lifetime and deleting an already-loaded
# key file does not affect a running server, so a concurrent owner is never harmed.
_KEY_PREFIX = "inferenceos-key-"
_KEY_SUFFIX = ".txt"
_STALE_KEY_AGE_S = 24 * 3600.0
_MAX_KEY_SWEEP_FILES = 200


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
    temporary directory, older than ``max_age_s``, and not symlinks are removed.
    Unrelated temporary files and another process's active key are untouched.
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
                try:
                    os.unlink(entry.path)
                    removed += 1
                except OSError:
                    continue
    except OSError:  # pragma: no cover - defensive
        return removed
    return removed


class ResidentBackendError(RuntimeError):
    """A resident backend operation failed and must not be silently retried."""


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
        self._model: ModelProfile | None = None
        self._params = RuntimeParams()
        self._port: int | None = None
        self._load: ResidentLoadMeasurement | None = None
        self._unload: ResidentUnloadMeasurement | None = None
        self._generation: ResidentGenerationMeasurement | None = None
        self._telemetry = BackendTelemetry()
        self._device: DeviceInfo | None = None

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
    def start(self, model: ModelProfile, params: RuntimeParams) -> ResidentLoadMeasurement:
        if self.is_running():
            raise ResidentBackendError("a resident server is already running")
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
        handle, key_path = tempfile.mkstemp(prefix="inferenceos-key-", suffix=".txt")
        os.close(handle)
        key_file = Path(key_path)
        key_file.write_text(api_key + "\n", encoding="utf-8")
        # Unix: owner-only permissions for the credential material.
        restrict_api_key_file(key_file)
        self._key_file = key_file

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
        logger.info("llama-server options: %s", " ".join(redacted_command(command)))
        if options.environment:
            logger.info(
                "llama-server engine switches: %s",
                ";".join(f"{k}={v}" for k, v in sorted(options.environment.items())),
            )
            process = self._process_factory(command, environment=options.environment)
        else:
            process = self._process_factory(command)
        started = time.perf_counter()
        try:
            pid = process.start()
        except Exception:
            self._cleanup_key_file()
            raise
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
        try:
            ready_s = process.wait_ready(client, timeout_s=self._readiness_timeout_s)
        except Exception:
            process.stop(timeout_s=5.0)
            self._cleanup_key_file()
            raise
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
        )
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
        self._process = process
        self._client = client
        self._model = model
        self._params = params
        self._port = port
        self._load = measurement
        return measurement

    def stop(self) -> ResidentUnloadMeasurement:
        process = self._process
        if process is None:
            raise ResidentBackendError("no resident server to unload")
        pid = process.pid
        before = resource_snapshot(pid)
        requested = datetime.now(timezone.utc)
        started = time.perf_counter()
        exit_code, method, exited = process.stop(timeout_s=10.0)
        exited = bool(exited and wait_process_exit(pid, timeout_s=10.0)) if pid else exited
        unload_s = time.perf_counter() - started
        after = resource_snapshot(pid)
        cleanup_complete = bool(exited and not after.exists)
        measurement = ResidentUnloadMeasurement(
            pid=pid,
            requested_s=requested.timestamp(),
            exited=exited,
            exit_code=exit_code,
            stop_method=method,
            before=before,
            after=after,
            unload_s=unload_s,
            cleanup_complete=cleanup_complete,
            notes=[
                "resident process tree terminated and process exit verified"
                if cleanup_complete
                else "resident process did not fully exit; cleanup incomplete",
            ],
        )
        self._unload = measurement
        self._process = None
        self._client = None
        self._cleanup_key_file()
        return measurement

    def cleanup(self) -> None:
        """Release the target and its Job Object handle after a loss, without raising.

        Safe to call when the process already exited (``stop`` is tolerant of an
        already-exited process) and when there is no process at all.
        """
        process = self._process
        if process is not None:
            try:
                process.stop(timeout_s=5.0)
            except Exception:  # noqa: BLE001 - cleanup must never raise
                pass
        self._process = None
        self._client = None
        self._cleanup_key_file()

    # ------------------------------------------------------------------ #
    # Generation                                                           #
    # ------------------------------------------------------------------ #
    def generate(self, request: GenerationRequest) -> GenerationResult:
        client = self._client
        process = self._process
        if client is None or process is None or not process.is_alive():
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
        generated: int | None = None
        prompt_tokens: int | None = None
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
                **({"n_probs": int(request.n_probs)} if request.n_probs > 0 else {}),
            ):
                if request.n_probs > 0:
                    choices.extend(token_choices_from(chunk))
                content = chunk.get("content")
                if isinstance(content, str) and content:
                    if first_token_at is None:
                        first_token_at = datetime.now(timezone.utc)
                        ttft_s = time.perf_counter() - started
                    pieces.append(content)
                if "tokens_predicted" in chunk and isinstance(chunk["tokens_predicted"], int):
                    generated = chunk["tokens_predicted"]
                if "tokens_evaluated" in chunk and isinstance(chunk["tokens_evaluated"], int):
                    prompt_tokens = chunk["tokens_evaluated"]
                if chunk.get("stop") is True:
                    final = chunk
                if "timings" in chunk and isinstance(chunk["timings"], dict):
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

        timings = final.get("timings") if final else None
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
            if isinstance(timings.get("predicted_n"), int):
                generated = timings["predicted_n"]
            if isinstance(timings.get("prompt_n"), int):
                prompt_tokens = timings["prompt_n"]
        text = "".join(pieces)
        if generated is None and text:
            generated = len(text.split())
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
            cache_prompt=False,
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
        self._generation = measurement
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
        )
        return result

    def _cleanup_key_file(self) -> None:
        if self._key_file is not None:
            try:
                self._key_file.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - defensive
                pass
            self._key_file = None


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
        cache_type_k=params.kv_cache_type,
        cache_type_v=params.kv_cache_type,
        log_verbosity=params.log_verbosity,
        ignore_eos=bool(params.ignore_eos),
    )
