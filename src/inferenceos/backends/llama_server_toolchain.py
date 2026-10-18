"""Low-level management of a real resident ``llama-server`` process.

This module is deliberately separate from :mod:`inferenceos.backends.llama_cpp_adapter`
(the process-per-generation CLI adapter, kept for isolated/cold benchmarking). It
cannot run inference by spawning ``llama-cli``; generation goes through the resident
server's authenticated HTTP API so the prompt is carried in the request body rather
than a command line.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterator

from inferenceos.backends.engine_env import engine_settings, process_environment
from inferenceos.backends.tensor_overrides import engine_argument
from inferenceos.backends.process_control import (
    ProcessGuard,
    PsutilProcessGuard,
    create_process_guard,
    process_identity,
    subprocess_session_kwargs,
)
from inferenceos.contracts.lifecycle import ProcessIdentity
from inferenceos.logging_config import get_logger

logger = get_logger("backends.llama_server")

SERVER_NAMES = ("llama-server.exe", "llama-server")
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class ServerError(RuntimeError):
    """Base class for resident-server failures."""


class ServerStartError(ServerError):
    """The server could not be started or never became ready."""


class ServerRequestError(ServerError):
    """An authenticated request to the resident server failed."""


def default_server_name(*, platform: str | None = None) -> str:
    """Platform-appropriate resident server executable name.

    Windows uses ``llama-server.exe``; Unix uses ``llama-server``. This is the single
    extension point for future platform asset layouts.
    """
    return "llama-server.exe" if (platform or sys.platform) == "win32" else "llama-server"


def server_process_names(*, platform: str | None = None) -> tuple[str, ...]:
    """Process names that identify an InferenceOS-owned resident server.

    Deliberately narrow (only the platform's own server name) so unrelated processes
    are never mistaken for the resident server.
    """
    return ("llama-server.exe",) if (platform or sys.platform) == "win32" else ("llama-server",)


def resolve_server_executable(llama_dir: Path, *, platform: str | None = None) -> Path:
    """Return the existing server executable, or the platform-appropriate default path.

    Unlike :func:`find_server` this never raises, so construction-time resolution
    preserves the previous behavior (``start`` reports a clear error if it is missing).
    """
    for name in SERVER_NAMES:
        candidate = llama_dir / name
        if candidate.is_file():
            return candidate
    return llama_dir / default_server_name(platform=platform)


def find_server(llama_dir: Path) -> Path:
    for name in SERVER_NAMES:
        candidate = llama_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"llama-server not found in {llama_dir}")


def generate_api_key() -> str:
    return secrets.token_urlsafe(32)


def pick_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass
class ServerOptions:
    model_path: Path
    n_gpu_layers: int = 99
    context_length: int | None = None
    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    flash_attention: bool | None = None
    device: str | None = None
    threads_batch: int | None = None
    load_mode: str | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    log_verbosity: int | None = None
    ignore_eos: bool = False
    no_jinja: bool = False
    tensor_overrides: str | None = None
    # Minimum chunk size for reusing cached prompt work by shifting it (``--cache-reuse``);
    # None = engine default. Set only by the prompt-reuse development evaluation.
    cache_reuse: int | None = None
    # Full-size sliding-window cache (``--swa-full``); set only by the reuse-correctness
    # development evaluation.
    swa_full: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    # Environment variables for the server process only (engine switches).
    environment: dict[str, str] = field(default_factory=dict)


def build_server_command(
    server: Path,
    options: ServerOptions,
    *,
    host: str,
    port: int,
    api_key_file: Path,
) -> list[str]:
    """Translate shared runtime knobs into a real ``llama-server`` invocation.

    The authentication key is passed via ``--api-key-file`` (never ``--api-key``) so
    it does not appear in the process command line.
    """
    command: list[str] = [
        str(server),
        "-m",
        str(options.model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--no-webui",
        "--api-key-file",
        str(api_key_file),
        "-ngl",
        str(options.n_gpu_layers),
    ]
    if options.context_length:
        command += ["-c", str(options.context_length)]
    if options.threads:
        command += ["-t", str(options.threads)]
    if options.threads_batch:
        command += ["-tb", str(options.threads_batch)]
    if options.batch_size:
        command += ["-b", str(options.batch_size)]
    if options.ubatch_size:
        command += ["-ub", str(options.ubatch_size)]
    if options.flash_attention is not None:
        command += ["-fa", "on" if options.flash_attention else "off"]
    if options.device:
        command += ["-dev", options.device]
    if options.load_mode:
        command += ["-lm", options.load_mode]
    if options.cache_type_k:
        command += ["-ctk", options.cache_type_k]
    if options.cache_type_v:
        command += ["-ctv", options.cache_type_v]
    override_argument = engine_argument(options.tensor_overrides)
    if override_argument:
        command += ["-ot", override_argument]
    if options.cache_reuse is not None:
        command += ["--cache-reuse", str(int(options.cache_reuse))]
    if options.swa_full:
        command += ["--swa-full"]
    if options.ignore_eos:
        command += ["--ignore-eos"]
    if options.no_jinja:
        command += ["--no-jinja"]
    if options.log_verbosity is not None:
        command += ["-lv", str(options.log_verbosity)]
    command += list(options.extra_args)
    return command


class LlamaServerClient:
    """Authenticated HTTP client for a resident server on loopback."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        api_key: str,
        timeout_s: float = 600.0,
    ) -> None:
        self._host = host
        self._port = port
        self._key = api_key
        self._timeout_s = timeout_s

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _open(
        self,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        method: str = "GET",
        timeout_s: float | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Authorization": f"Bearer {self._key}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            return urllib.request.urlopen(
                request, timeout=timeout_s or self._timeout_s
            )
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # pragma: no cover - defensive
                pass
            raise ServerRequestError(
                f"{method} {path} failed: HTTP {exc.code} {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ServerRequestError(f"{method} {path} failed: {exc}") from exc

    def health(self, *, timeout_s: float = 3.0) -> bool:
        try:
            with self._open("/health", timeout_s=timeout_s) as response:
                return bool(response.status == 200)
        except ServerRequestError:
            return False

    def props(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        with self._open("/props", timeout_s=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        return payload if isinstance(payload, dict) else {}

    def tokenize(self, content: str, *, add_special: bool = True, timeout_s: float = 30.0) -> list[int]:
        """The engine's own tokens for ``content`` (``POST /tokenize``). ``add_special``
        adds the start token exactly as ``/completion`` does for a prompt."""
        body = {"content": content, "add_special": bool(add_special)}
        with self._open("/tokenize", body=body, method="POST", timeout_s=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(tokens, list) or not all(type(t) is int and t >= 0 for t in tokens):
            raise ServerRequestError("POST /tokenize returned no token list")
        return [int(t) for t in tokens]

    def slots(self, *, timeout_s: float = 10.0) -> list[dict[str, Any]]:
        """The engine's slot list (``GET /slots``, enabled by default in b11139). A
        timeout while reading or an unreadable reply is a ``ServerRequestError``."""
        try:
            with self._open("/slots", timeout_s=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except (OSError, ValueError) as exc:
            raise ServerRequestError(f"GET /slots failed: {exc}") from exc
        if not isinstance(payload, list):
            raise ServerRequestError("GET /slots returned no slot list")
        if not all(isinstance(item, dict) for item in payload):
            # Never filtered away: an unreadable entry could be the busy slot.
            raise ServerRequestError("GET /slots returned an entry that is not a slot object")
        return list(payload)

    def chat_request(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        seed: int,
        temperature: float,
        cache_prompt: bool | None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """One ``/v1/chat/completions`` request as an ordinary client sends it (the
        reuse-correctness evaluation's exposure check). ``cache_prompt`` None leaves the
        field out, so the engine's own default applies, exactly as for a client that does
        not know about it; True or False send it. Returns the engine's whole reply."""
        body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": temperature,
            "seed": seed,
            "stream": False,
        }
        if cache_prompt is not None:
            body["cache_prompt"] = bool(cache_prompt)
        try:
            with self._open("/v1/chat/completions", body=body, method="POST", timeout_s=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except (OSError, ValueError) as exc:
            raise ServerRequestError(f"POST /v1/chat/completions failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise ServerRequestError("POST /v1/chat/completions returned no object")
        return payload

    def chat_complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float,
        seed: int,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "reasoning_result",
                    "strict": True,
                    "schema": json_schema,
                },
            }
        with self._open(
            "/v1/chat/completions", body=body, method="POST", timeout_s=timeout_s
        ) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        return payload if isinstance(payload, dict) else {}

    def stream_completion(
        self,
        *,
        prompt: str,
        n_predict: int,
        seed: int,
        temperature: float,
        stop: list[str],
        json_schema: dict[str, Any] | None,
        timeout_s: float | None = None,
        ignore_eos: bool = False,
        n_probs: int = 0,
        cache_prompt: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield streaming chunks from ``/completion`` (server-sent events).

        ``cache_prompt`` is always sent explicitly. It is False unless a caller asks for
        prompt reuse, so measurements read the whole prompt fresh even though the
        engine's own default is to reuse."""
        body: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": n_predict,
            "seed": seed,
            "temperature": temperature,
            "stop": stop,
            "stream": True,
            "cache_prompt": bool(cache_prompt),
        }
        if ignore_eos:
            body["ignore_eos"] = True
        if n_probs > 0:
            # Probabilities before sampling (log form), for the chosen word and the
            # closest alternatives.
            body["n_probs"] = int(n_probs)
        if json_schema is not None:
            body["json_schema"] = json_schema
        response = self._open(
            "/completion", body=body, method="POST", timeout_s=timeout_s
        )
        try:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload in ("[DONE]", ""):
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict):
                    yield chunk
        finally:
            response.close()


class LlamaServerProcess:
    """Owns the real server process, a Job Object guard, and its recent logs."""

    def __init__(
        self,
        command: list[str],
        *,
        guard: ProcessGuard | None = None,
        log_limit: int = 6000,
        environment: dict[str, str] | None = None,
    ) -> None:
        self._command = command
        # Extra environment variables for this process only (engine switches); the
        # rest of the environment is inherited, without diagnostic variables
        # (``engine_env.process_environment``).
        self._environment = dict(environment or {})
        #: Engine variables the process was actually started with.
        self.engine_settings: dict[str, str] = {}
        self._guard: ProcessGuard | None = guard
        self._proc: subprocess.Popen[str] | None = None
        self._logs: deque[str] = deque(maxlen=log_limit)
        # The first lines (devices, buffers, build) are kept apart: a long run with the
        # engine's timer on pushes them out of the recent-lines buffer. Not cut at
        # readiness: the reader threads can lag behind the health check (laptop job
        # 083 kept only 22 lines that way).
        self._startup: list[str] = []
        self._startup_limit = 5000
        self._lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._reader_threads: list[threading.Thread] = []

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def start(self) -> int:
        guard = self._guard or create_process_guard()
        self._guard = guard  # own the handle even if environment/Popen preparation fails
        try:
            env = process_environment(dict(os.environ), self._environment)
            self.engine_settings = engine_settings(env)
            self._proc = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=_CREATE_NO_WINDOW,
                env=env,
                **subprocess_session_kwargs(),  # type: ignore[call-overload]
            )
        except OSError as exc:
            try:
                guard.close()
                self._guard = None
            except Exception:
                logger.exception("process guard cleanup after launch failure")
            raise ServerStartError(f"failed to start llama-server: {exc}") from exc
        except BaseException:
            try:
                guard.close()
                self._guard = None
            except Exception:
                logger.exception("process guard cleanup after interrupted launch")
            raise
        try:
            assigned = guard.assign(self._proc.pid)
            if assigned is False:
                raise ServerStartError("server process containment could not be established")
        except BaseException:
            # No request may run on an uncontained child. Keep an explicit tree owner
            # for the backend's existing unwind, including failed/interrupted assignment.
            self._guard = PsutilProcessGuard(self._proc.pid)
            try:
                guard.close()
            except Exception:
                logger.exception("failed containment handle cleanup")
            raise
        self._reader_threads = [
            self._spawn_reader(self._proc.stderr, "stderr"),
            self._spawn_reader(self._proc.stdout, "stdout"),
        ]
        return self._proc.pid

    def _spawn_reader(self, stream: Any, name: str) -> threading.Thread:
        def _read() -> None:
            if stream is None:  # pragma: no cover - defensive
                return
            try:
                for line in stream:
                    with self._lock:
                        text = line.rstrip("\n")
                        self._logs.append(text)
                        if len(self._startup) < self._startup_limit:
                            self._startup.append(text)
            except (ValueError, OSError):  # pragma: no cover - stream closed
                return

        thread = threading.Thread(target=_read, name=f"llama-server-{name}", daemon=True)
        thread.start()
        return thread

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def returncode(self) -> int | None:
        return self._proc.poll() if self._proc is not None else None

    def identity(self) -> ProcessIdentity | None:
        pid = self.pid
        return process_identity(pid) if pid is not None else None

    def log_text(self) -> str:
        with self._lock:
            return "\n".join(self._logs)

    def startup_text(self) -> str:
        """The server's first lines (devices, buffers), kept however long it ran."""
        with self._lock:
            return "\n".join(self._startup)

    def wait_ready(
        self,
        client: LlamaServerClient,
        *,
        timeout_s: float,
        poll_s: float = 0.25,
        should_stop: Callable[[], bool] | None = None,
    ) -> float:
        """Wait until the server answers its health check. ``should_stop`` is polled
        between checks (review 10): a cancelled startup ends the wait at once."""
        started = time.perf_counter()
        while time.perf_counter() - started < timeout_s:
            if should_stop is not None and should_stop():
                raise ServerStartError("llama-server startup was cancelled")
            if self._proc is not None and self._proc.poll() is not None:
                tail = "\n".join(list(self._logs)[-8:])
                raise ServerStartError(
                    f"llama-server exited early with code {self._proc.returncode}: {tail}"
                )
            if client.health():
                return time.perf_counter() - started
            time.sleep(poll_s)
        raise ServerStartError(
            f"llama-server was not ready within {timeout_s:.0f}s"
        )

    def stop(self, *, timeout_s: float = 10.0) -> tuple[int | None, str, bool]:
        # Serialised: a cancellation from another thread and the owner's own cleanup
        # may both stop the process; the second call sees it already exited.
        with self._stop_lock:
            return self._stop(timeout_s=timeout_s)

    def _stop(self, *, timeout_s: float) -> tuple[int | None, str, bool]:
        proc = self._proc
        if proc is None:
            if self._guard is not None:
                self._guard.close()
                self._guard = None
            return None, "none", True
        if proc.poll() is not None:
            guard = self._guard
            if guard is not None:
                guard.close()
            return proc.returncode, "already_exited", True
        method = self._guard.kind if self._guard is not None else "process_tree"
        if self._guard is not None:
            self._guard.terminate()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                return proc.poll(), method, False
        finally:
            if self._guard is not None:
                self._guard.close()
        return proc.returncode, method, True
