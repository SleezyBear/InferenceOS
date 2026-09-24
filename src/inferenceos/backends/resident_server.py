"""Real resident llama.cpp server backend used by :class:`LlamaCppResidentAdapter`.

``LlamaServerResidentBackend`` owns one real ``llama-server`` process and exposes
measured load/unload/generation evidence. It never falls back to spawning
``llama-cli``: if the resident server is not running, generation fails loudly.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from inferenceos.backends import llama_server_toolchain as srv
from inferenceos.backends.llama_cpp_toolchain import (
    parse_device_info,
    parse_llama_memory,
)
from inferenceos.backends.process_control import (
    resource_snapshot,
    wait_process_exit,
)
from inferenceos.contracts.lifecycle import (
    ProcessIdentity,
    ResidentGenerationMeasurement,
    ResidentLoadMeasurement,
    ResidentUnloadMeasurement,
    StopMethod,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import (
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.logging_config import get_logger

logger = get_logger("backends.resident_server")


class ResidentBackendError(RuntimeError):
    """A resident backend operation failed and must not be silently retried."""


class ResidentServerBackend(Protocol):
    """Structural interface used by the adapter and the deep-diagnosis lifecycle."""

    def start(self, model: ModelProfile, params: RuntimeParams) -> ResidentLoadMeasurement:
        ...

    def stop(self) -> ResidentUnloadMeasurement:
        ...

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

    def is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    # ------------------------------------------------------------------ #
    # Load / unload                                                        #
    # ------------------------------------------------------------------ #
    def start(self, model: ModelProfile, params: RuntimeParams) -> ResidentLoadMeasurement:
        if self.is_running():
            raise ResidentBackendError("a resident server is already running")
        model_path = Path(model.path)
        if not model_path.is_file():
            raise ResidentBackendError(f"model file not found: {model_path}")

        before = resource_snapshot(None)
        api_key = srv.generate_api_key()
        handle, key_path = tempfile.mkstemp(prefix="inferenceos-key-", suffix=".txt")
        os.close(handle)
        key_file = Path(key_path)
        key_file.write_text(api_key + "\n", encoding="utf-8")
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
        process = self._process_factory(command)
        started = time.perf_counter()
        try:
            pid = process.start()
        except Exception:
            self._cleanup_key_file()
            raise
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
        self._telemetry = BackendTelemetry(
            device=device.device if device is not None else None,
            gpu_layers_offloaded=device.layers_offloaded if device is not None else None,
            gpu_layers_total=device.layers_total if device is not None else None,
            kv_cache_bytes=memory.kv_cache_bytes if memory is not None else None,
            model_buffer_bytes=memory.model_buffer_bytes if memory is not None else None,
            context_length=params.context_length,
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
        submitted = datetime.now(timezone.utc)
        started = time.perf_counter()
        first_token_at: datetime | None = None
        ttft_s: float | None = None
        pieces: list[str] = []
        final: dict[str, object] | None = None
        generated: int | None = None
        prompt_tokens: int | None = None
        try:
            for chunk in client.stream_completion(
                prompt=request.prompt,
                n_predict=request.n_predict,
                seed=request.seed,
                temperature=request.temperature,
                stop=list(request.stop),
                json_schema=None,
            ):
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
        except srv.ServerRequestError as exc:
            raise ResidentBackendError(f"resident generation failed: {exc}") from exc
        completed = datetime.now(timezone.utc)
        total_s = time.perf_counter() - started
        if not process.is_alive():
            raise ResidentBackendError("resident server died during generation")

        timings = final.get("timings") if final else None
        decode_tps: float | None = None
        if isinstance(timings, dict):
            value = timings.get("predicted_per_second")
            decode_tps = float(value) if isinstance(value, (int, float)) else None
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
            streamed=True,
            text=text[:4000],
            cache_prompt=False,
            notes=[
                "TTFT measured from request submission to first streamed content chunk",
            ],
        )
        self._generation = measurement
        result = GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated,
            prompt_tokens_per_sec=None,
            decode_tokens_per_sec=decode_tps,
            model_load_s=None,
            prefill_s=None,
            decode_s=None,
            ttft_s=ttft_s,
            total_s=total_s,
            process_wall_s=total_s,
            peak_rss_bytes=resource_snapshot(process.pid).rss_bytes,
            peak_process_cpu_percent=None,
        )
        return result

    def _cleanup_key_file(self) -> None:
        if self._key_file is not None:
            try:
                self._key_file.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - defensive
                pass
            self._key_file = None


def _options_from(model_path: Path, params: RuntimeParams) -> srv.ServerOptions:
    return srv.ServerOptions(
        model_path=model_path,
        n_gpu_layers=(
            params.gpu_offload_layers if params.gpu_offload_layers is not None else 99
        ),
        context_length=params.context_length,
        threads=params.threads or None,
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
