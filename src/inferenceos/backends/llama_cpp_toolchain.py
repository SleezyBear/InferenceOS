"""Thin, real process wrapper around the acquired llama.cpp CLI.

This is *not* the ``BackendAdapter`` (that interface is defined in Phase 0 and
implemented in Phase 2). It is a low-level helper used by the Phase 0 verification
command to run the real binary, capture the real device/offload logs, and parse the
real generated text. Optimizer components must go through ``BackendAdapter``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferenceos.backends.engine_env import process_environment, parse_switches
from inferenceos.backends.process_control import terminate_process_tree
from inferenceos.interfaces.backend_adapter import BackendCleanupUnresolved
from inferenceos.backends.tensor_overrides import engine_argument
from inferenceos.contracts.enums import BackendType
from inferenceos.logging_config import get_logger

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a hard dependency
    psutil = None  # type: ignore[assignment]

logger = get_logger("backends.llama_cpp")

CLI_NAMES = ("llama-cli.exe", "llama-cli")

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_DEVICE_LINE_RE = re.compile(r"using device\s+(\w+)\s+\(([^)]+)\)")
_DEVICE_SUMMARY_RE = re.compile(
    r"-\s*(\w+)\s*:\s*([^(\n]+?)\(([\d,]+)\s*MiB,\s*([\d,]+)\s*MiB free\)"
)
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
_PROMPT_STAT_RE = re.compile(
    r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]"
)
_KV_SIZE_RE = re.compile(r"llama_kv_cache:\s+size\s*=\s*([\d.]+)\s*MiB")
_MODEL_BUFFER_RE = re.compile(
    r"load_tensors:\s+([A-Za-z0-9_]+)\s+model buffer size\s*=\s*([\d.]+)\s*MiB"
)
_N_CTX_RE = re.compile(r"llama_context:\s+n_ctx\s*=\s*(\d+)")
# The effective generation threadpool size llama.cpp actually initialized. This is the
# authoritative runtime value; it may differ from the requested ``-t`` when the backend
# chooses it dynamically.
_N_THREADS_RE = re.compile(r"llama threadpool init,\s*n_threads\s*=\s*(\d+)")
_N_THREADS_GENERIC_RE = re.compile(r"n_threads\s*=\s*(\d+)")
# The effective weight-load mode llama.cpp actually used, from its
# ``load_tensors: loading model tensors ... (load_mode = mmap|none)`` line. llama.cpp
# emits this only at high log verbosity; when it is absent the effective state is
# unknown (never inferred from the configured value or launch arguments).
_LOAD_MODE_RE = re.compile(r"\(\s*load_mode\s*=\s*(mmap|none)\s*\)", re.IGNORECASE)
#: The whole ``version:`` line: build number and code version (commit) are both kept,
#: so two different builds never look the same.
_VERSION_RE = re.compile(r"version:\s*([^\r\n]+)")

# llama.cpp prefixes each log line with a monotonic ``a.bb.ccc.ddd`` stamp.
_LOG_TS_RE = re.compile(r"^(?P<ts>\d+\.\d{2}\.\d{3}\.\d{3})\s")
_LOAD_START_RE = re.compile(r"load_model:\s+loading model")
_READY_RE = re.compile(r"load_model:\s+initializing,\s+n_slots")
_TIMING_NUM_RE = re.compile(r"=\s*([\d.]+)\s*ms")
_TIMING_TOKENS_RE = re.compile(r"/\s*(\d+)\s*tokens")
_TIMING_RUNS_RE = re.compile(r"/\s*(\d+)\s*runs")
_TIMING_TPS_RE = re.compile(r"([\d.]+)\s*tokens per second")


@dataclass
class DeviceInfo:
    backend: str
    device: str
    vram_total_bytes: int | None = None
    vram_free_bytes: int | None = None
    layers_offloaded: int | None = None
    layers_total: int | None = None


@dataclass
class LlamaMemoryReport:
    """Real memory figures reported by llama.cpp for a context allocation."""

    kv_cache_bytes: int | None = None
    model_buffer_bytes: int | None = None
    context_length: int | None = None
    buffers: dict[str, int] = field(default_factory=dict)


@dataclass
class MemoryObservation:
    """An externally measured process memory observation (not an estimate)."""

    peak_rss_bytes: int | None = None
    final_rss_bytes: int | None = None
    peak_cpu_percent: float | None = None
    sample_count: int = 0


@dataclass
class LlamaTimings:
    """Engine phase counters, with server and CLI dialects kept distinct.

    ``model_load_s`` uses the logged interval or the engine's printed load counter.
    Actual TTFT is unknown without streaming; ``ttft_estimate_s`` is a diagnostic
    approximation and is never exported as measured TTFT. Decode runs are not
    silently converted into written-token counts.
    """

    prompt_eval_s: float | None = None
    prompt_tokens: int | None = None
    prefill_tokens_per_sec: float | None = None
    eval_s: float | None = None
    generated_tokens: int | None = None
    # CLI performance counters report decode evaluations (runs), which are not
    # necessarily the number of pieces written to the response.
    decode_evaluations: int | None = None
    decode_tokens_per_sec: float | None = None
    total_s: float | None = None
    total_tokens: int | None = None
    load_start_s: float | None = None
    ready_s: float | None = None
    timing_s: float | None = None
    reported_model_load_s: float | None = None

    @property
    def model_load_s(self) -> float | None:
        if self.load_start_s is None or self.ready_s is None:
            return self.reported_model_load_s
        return max(self.ready_s - self.load_start_s, 0.0)

    @property
    def ttft_s(self) -> float | None:
        # An average decode step cannot establish the actual first-token time.
        # Resident streaming measures TTFT independently; CLI capture does not.
        return None

    @property
    def ttft_estimate_s(self) -> float | None:
        if self.prompt_eval_s is None:
            return None
        evaluations = self.generated_tokens or self.decode_evaluations
        if self.eval_s is not None and evaluations:
            return self.prompt_eval_s + self.eval_s / evaluations
        return self.prompt_eval_s


@dataclass
class LlamaRunResult:
    pid: int
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    command: list[str] = field(default_factory=list)
    peak_rss_bytes: int | None = None
    final_rss_bytes: int | None = None
    peak_process_cpu_percent: float | None = None
    memory: LlamaMemoryReport | None = None
    timings: LlamaTimings | None = None

    @property
    def combined(self) -> str:
        return f"{self.stdout}\n{self.stderr}"



def default_cli_name(*, platform: str | None = None) -> str:
    """Platform-appropriate ``llama-cli`` executable name (Windows ``.exe``, else none)."""
    return "llama-cli.exe" if (platform or sys.platform) == "win32" else "llama-cli"


def resolve_cli_executable(llama_dir: Path, *, platform: str | None = None) -> Path:
    """Return the existing CLI executable, or the platform-appropriate default path."""
    for name in CLI_NAMES:
        candidate = llama_dir / name
        if candidate.is_file():
            return candidate
    return llama_dir / default_cli_name(platform=platform)


def find_cli(llama_dir: Path) -> Path:
    for name in CLI_NAMES:
        candidate = llama_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"llama-cli not found in {llama_dir}")


def run_cli(
    cli: Path,
    model_path: Path,
    prompt: str,
    *,
    n_predict: int = 64,
    seed: int = 1234,
    temperature: float = 0.0,
    n_gpu_layers: int = 99,
    threads: int = 0,
    threads_batch: int | None = None,
    engine_switches: str | None = None,
    context_length: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    flash_attention: bool | None = None,
    device: str | None = None,
    load_mode: str | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    tensor_overrides: str | None = None,
    verbose: bool = True,
    log_verbosity: int | None = None,
    ignore_eos: bool | None = None,
    no_jinja: bool = False,
    json_schema: str | None = None,
    extra_args: tuple[str, ...] = (),
    timeout_s: float = 900.0,
    track_rss: bool = False,
    rss_interval_s: float = 0.05,
) -> LlamaRunResult:
    """Run one real, single-turn inference and return captured real output."""
    command = build_run_command(
        cli,
        model_path,
        prompt,
        n_predict=n_predict,
        seed=seed,
        temperature=temperature,
        n_gpu_layers=n_gpu_layers,
        threads=threads,
        threads_batch=threads_batch,
        context_length=context_length,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        flash_attention=flash_attention,
        device=device,
        load_mode=load_mode,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        tensor_overrides=tensor_overrides,
        verbose=verbose,
        log_verbosity=log_verbosity,
        ignore_eos=ignore_eos,
        no_jinja=no_jinja,
        json_schema=json_schema,
        extra_args=extra_args,
    )

    shown = list(command)
    for index, argument in enumerate(shown[:-1]):
        if argument in ("-p", "--prompt", "-j", "--json-schema"):
            shown[index + 1] = "<omitted>"
    logger.info("Running llama.cpp: %s", " ".join(shown))
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        # Never let an inherited diagnostic variable (e.g. an engine timer) slow a run.
        env=process_environment(dict(os.environ), parse_switches(engine_switches)),
        creationflags=_CREATE_NO_WINDOW,
    )

    observation = MemoryObservation()
    stop_event = threading.Event()
    rss_thread: threading.Thread | None = None
    try:
        if track_rss:
            rss_thread = threading.Thread(
                target=lambda: _observe_rss(proc.pid, stop_event, rss_interval_s, observation),
                name="rss-observer", daemon=True,
            )
            rss_thread.start()
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except BaseException as exc:
        # This function owns the CLI child even when the caller cancels, a pipe
        # fails, or the RSS observer cannot start. Cleanup must be bounded too.
        try:
            terminate_process_tree(proc.pid, timeout_s=5.0)
            if proc.poll() is None:
                proc.kill()
            proc.communicate(timeout=5.0)
            if proc.poll() is None:
                raise RuntimeError("CLI process exit unconfirmed")
        except Exception as cleanup_error:
            exc.add_note(f"CLI cleanup also failed: {cleanup_error}")
            setattr(exc, "unreleased_process", proc)
            if isinstance(exc, Exception):
                blocked = BackendCleanupUnresolved(f"CLI process cleanup unconfirmed: {cleanup_error}")
                blocked.unreleased_process = proc  # type: ignore[attr-defined]
                raise blocked from exc
            raise exc from cleanup_error
        if isinstance(exc, subprocess.TimeoutExpired):
            raise TimeoutError(f"llama-cli exceeded {timeout_s:.0f}s") from exc
        raise
    finally:
        stop_event.set()
        if rss_thread is not None and rss_thread.ident is not None:
            rss_thread.join(timeout=2.0)

    duration = time.perf_counter() - started
    stderr_text = stderr or ""
    return LlamaRunResult(
        pid=proc.pid,
        returncode=proc.returncode,
        stdout=stdout or "",
        stderr=stderr_text,
        duration_s=duration,
        command=command,
        peak_rss_bytes=observation.peak_rss_bytes,
        final_rss_bytes=observation.final_rss_bytes,
        peak_process_cpu_percent=observation.peak_cpu_percent,
        memory=parse_llama_memory(stderr_text),
        timings=parse_llama_timings(stderr_text),
    )


def build_run_command(
    cli: Path,
    model_path: Path,
    prompt: str,
    *,
    n_predict: int = 64,
    seed: int = 1234,
    temperature: float = 0.0,
    n_gpu_layers: int = 99,
    threads: int = 0,
    threads_batch: int | None = None,
    context_length: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    flash_attention: bool | None = None,
    device: str | None = None,
    load_mode: str | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
    tensor_overrides: str | None = None,
    verbose: bool = True,
    log_verbosity: int | None = None,
    ignore_eos: bool | None = None,
    no_jinja: bool = False,
    json_schema: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    """Translate generic runtime knobs into a real llama.cpp CLI invocation."""
    command: list[str] = [
        str(cli),
        "-m",
        str(model_path),
        "-ngl",
        str(n_gpu_layers),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-s",
        str(seed),
        "--temp",
        str(temperature),
        "-st",
        "--no-warmup",
    ]
    if threads and threads > 0:
        command += ["-t", str(threads)]
    if threads_batch is not None and threads_batch > 0:
        command += ["-tb", str(threads_batch)]
    if context_length and context_length > 0:
        command += ["-c", str(context_length)]
    if batch_size and batch_size > 0:
        command += ["-b", str(batch_size)]
    if ubatch_size and ubatch_size > 0:
        command += ["-ub", str(ubatch_size)]
    if flash_attention is not None:
        command += ["-fa", "on" if flash_attention else "off"]
    if device:
        command += ["-dev", device]
    if load_mode:
        command += ["-lm", load_mode]
    if cache_type_k:
        command += ["-ctk", cache_type_k]
    if cache_type_v:
        command += ["-ctv", cache_type_v]
    override_argument = engine_argument(tensor_overrides)
    if override_argument:
        command += ["-ot", override_argument]
    if ignore_eos:
        command += ["--ignore-eos"]
    if no_jinja:
        command += ["--no-jinja"]
    if json_schema:
        command += ["-j", json_schema]
    if log_verbosity is not None:
        command += ["-lv", str(log_verbosity)]
    elif verbose:
        command += ["-v"]
    command += list(extra_args)
    return command


def _observe_rss(
    pid: int,
    stop_event: threading.Event,
    interval_s: float,
    observation: MemoryObservation,
) -> None:
    if psutil is None:  # pragma: no cover - psutil is required in practice
        return
    try:
        process = psutil.Process(pid)
    except psutil.Error:  # pragma: no cover - process may already have exited
        return
    peak = 0
    final = 0
    samples = 0
    peak_cpu = 0.0
    last_time = time.perf_counter()
    last_cpu = _process_cpu_seconds(process)
    while not stop_event.is_set():
        try:
            rss = int(process.memory_info().rss)
            peak = max(peak, rss)
            final = rss
            now = time.perf_counter()
            cpu = _process_cpu_seconds(process)
            if last_cpu is not None and cpu is not None and now > last_time:
                percent = max((cpu - last_cpu) / (now - last_time) * 100.0, 0.0)
                peak_cpu = max(peak_cpu, percent)
            last_time, last_cpu = now, cpu
            samples += 1
        except psutil.Error:
            break
        stop_event.wait(interval_s)
    observation.peak_rss_bytes = peak or None
    observation.final_rss_bytes = final or None
    observation.peak_cpu_percent = peak_cpu or None
    observation.sample_count = samples


def _process_cpu_seconds(process: Any) -> float | None:
    try:
        times = process.cpu_times()
    except psutil.Error:
        return None
    return float(getattr(times, "user", 0.0) or 0.0) + float(getattr(times, "system", 0.0) or 0.0)


def parse_log_seconds(stamp: str) -> float | None:
    """Parse llama.cpp's ``a.bb.ccc.ddd`` monotonic log stamp into seconds."""
    parts = stamp.split(".")
    if len(parts) != 4:
        return None
    try:
        minutes, seconds, millis, micros = (int(part) for part in parts)
    except ValueError:
        return None
    return minutes * 60 + seconds + millis / 1000.0 + micros / 1_000_000.0


def parse_llama_timings(stderr: str) -> LlamaTimings | None:
    """Parse llama.cpp's authoritative per-phase timing counters."""
    timings = LlamaTimings()
    found = False
    for raw in stderr.splitlines():
        stamp = _LOG_TS_RE.match(raw)
        seconds = parse_log_seconds(stamp.group("ts")) if stamp else None
        if seconds is not None and timings.load_start_s is None and _LOAD_START_RE.search(raw):
            timings.load_start_s = seconds
        if seconds is not None and timings.ready_s is None and _READY_RE.search(raw):
            timings.ready_s = seconds
        if "print_timing" not in raw and "llama_perf_context_print" not in raw:
            continue
        found = True
        if seconds is not None:
            timings.timing_s = seconds
        if "prompt eval time" in raw:
            timings.prompt_eval_s = _ms(raw)
            timings.prompt_tokens = _tokens(raw)
            timings.prefill_tokens_per_sec = _tps(raw)
        elif "eval time" in raw:
            timings.eval_s = _ms(raw)
            timings.generated_tokens = _tokens(raw)
            runs = _TIMING_RUNS_RE.search(raw)
            timings.decode_evaluations = int(runs[1]) if runs is not None else None
            timings.decode_tokens_per_sec = _tps(raw)
        elif "total time" in raw:
            timings.total_s = _ms(raw)
            timings.total_tokens = _tokens(raw)
        elif "load time" in raw:
            timings.reported_model_load_s = _ms(raw)
    if not found:
        return None
    return timings


def _ms(line: str) -> float | None:
    match = _TIMING_NUM_RE.search(line)
    return float(match.group(1)) / 1000.0 if match else None


def _tokens(line: str) -> int | None:
    match = _TIMING_TOKENS_RE.search(line)
    return int(match.group(1)) if match else None


def _tps(line: str) -> float | None:
    match = _TIMING_TPS_RE.search(line)
    return float(match.group(1)) if match else None


def parse_llama_memory(stderr: str) -> LlamaMemoryReport | None:
    """Parse the real memory figures llama.cpp logs for its context allocation."""
    kv_matches = _KV_SIZE_RE.findall(stderr)
    buffers: dict[str, int] = {}
    for name, value in _MODEL_BUFFER_RE.findall(stderr):
        size = int(round(float(value) * 1024 * 1024))
        buffers[name] = max(buffers.get(name, 0), size)
    ctx_matches = _N_CTX_RE.findall(stderr)
    if not kv_matches and not buffers and not ctx_matches:
        return None
    kv_bytes = max(
        (int(round(float(value) * 1024 * 1024)) for value in kv_matches),
        default=None,
    )
    model_buffer = max(buffers.values(), default=None)
    context = int(ctx_matches[-1]) if ctx_matches else None
    return LlamaMemoryReport(
        kv_cache_bytes=kv_bytes,
        model_buffer_bytes=model_buffer,
        context_length=context,
        buffers=buffers,
    )


_KV_TYPES_RE = re.compile(
    r"llama_kv_cache\w*:.*?size\s*=\s*([\d.]+)\s*MiB.*?K \((\w+)\):\s*([\d.]+)\s*MiB,\s*V \((\w+)\):\s*([\d.]+)\s*MiB"
)
_KV_BUFFER_RE = re.compile(r"llama_kv_cache\w*:\s+(\S+)\s+KV buffer size\s*=\s*([\d.]+)\s*MiB")
_STATE_BUFFER_RE = re.compile(r"llama_memory_recurrent\w*:\s+(\S+)\s+RS buffer size\s*=\s*([\d.]+)\s*MiB")
_STATE_TYPES_RE = re.compile(r"llama_memory_recurrent\w*:.*?R \((\w+)\).*?S \((\w+)\)")
_COMPUTE_BUFFER_RE = re.compile(r"(?:sched_reserve|llama_context):\s+(\S+)\s+compute buffer size\s*=\s*([\d.]+)\s*MiB")
_FLASH_SET_RE = re.compile(r"Flash Attention was auto, set to (enabled|disabled)")
_FLASH_RE = re.compile(r"llama_context:\s+flash_attn\s*=\s*(\w+)")
_OVERRIDDEN_RE = re.compile(r"tensor (\S+) \([^)]*\) buffer type overridden to (\S+)")
_SLOT_CONTEXT_RE = re.compile(r"initializing, n_slots\s*=\s*\d+, n_ctx_slot\s*=\s*(\d+)")


def _mib_table(pattern: re.Pattern[str], text: str) -> dict[str, int]:
    table: dict[str, int] = {}
    for name, value in pattern.findall(text):
        table[name] = table.get(name, 0) + int(round(float(value) * 1024 * 1024))
    return table


def parse_effective_settings(log_text: str) -> dict[str, object]:
    """What llama.cpp's own startup log shows about how it set itself up.

    Keys (each present only when the log showed it): ``cache_type_k`` /
    ``cache_type_v`` (the conversation-memory formats; a model with a sliding-window
    and a full cache prints both, and differing formats are reported as a list),
    ``kv_buffers`` / ``state_buffers`` / ``model_buffers`` / ``compute_buffers``
    (bytes per engine buffer), ``state_types`` (recurrent state formats),
    ``flash_attention`` (True/False once the engine decided), ``overridden_tensors``
    (piece -> buffer, printed only at debug log level). Missing evidence stays missing.
    """
    observed: dict[str, object] = {}
    slots = _SLOT_CONTEXT_RE.findall(log_text)
    if slots:
        observed["context_length"] = int(slots[-1])
    kv_types = _KV_TYPES_RE.findall(log_text)
    if kv_types:
        keys = sorted({item[1] for item in kv_types})
        values = sorted({item[3] for item in kv_types})
        observed["cache_type_k"] = keys[0] if len(keys) == 1 else keys
        observed["cache_type_v"] = values[0] if len(values) == 1 else values
    for key, pattern in (
        ("kv_buffers", _KV_BUFFER_RE),
        ("state_buffers", _STATE_BUFFER_RE),
        ("compute_buffers", _COMPUTE_BUFFER_RE),
    ):
        table = _mib_table(pattern, log_text)
        if table:
            observed[key] = table
    models: dict[str, int] = {}
    for name, value in _MODEL_BUFFER_RE.findall(log_text):
        models[name] = max(models.get(name, 0), int(round(float(value) * 1024 * 1024)))
    if models:
        observed["model_buffers"] = models
    state = _STATE_TYPES_RE.findall(log_text)
    if state:
        observed["state_types"] = sorted({kind for pair in state for kind in pair})
    flash = _FLASH_SET_RE.findall(log_text)
    if flash:
        observed["flash_attention"] = flash[-1] == "enabled"
    else:
        named = _FLASH_RE.findall(log_text)
        if named and named[-1].lower() in ("enabled", "on", "1", "true"):
            observed["flash_attention"] = True
        elif named and named[-1].lower() in ("disabled", "off", "0", "false"):
            observed["flash_attention"] = False
    overridden = _OVERRIDDEN_RE.findall(log_text)
    if overridden:
        observed["overridden_tensors"] = {name: buffer for name, buffer in overridden}
    return observed


def list_devices(cli: Path, *, timeout_s: float = 60.0) -> list[str]:
    """Return the real devices llama.cpp reports via ``--list-devices``."""
    try:
        proc = subprocess.run(
            [str(cli), "--list-devices"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        logger.warning("llama.cpp --list-devices failed: %s", exc)
        return []
    devices: list[str] = []
    for line in (proc.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("available devices"):
            continue
        if ":" in stripped:
            devices.append(stripped)
    return devices


def cli_version(cli: Path, *, timeout_s: float = 60.0) -> str | None:
    """Return the real llama.cpp version string (best-effort)."""
    try:
        proc = subprocess.run(
            [str(cli), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    match = _VERSION_RE.search(combined)
    if match:
        return match.group(1).strip()
    lines = combined.strip().splitlines()
    return lines[0].strip() if lines else None


def parse_effective_threads(log_text: str) -> int | None:
    """Return the *effective* generation thread count llama.cpp actually initialized.

    This is the value the backend selected or was configured with at startup
    (``llama threadpool init, n_threads = N``); it is authoritative runtime state and
    may differ from the requested ``-t``. It is **never** inferred from CPU topology:
    when the log does not carry the value, ``None`` is returned (unknown / fail closed).
    """
    match = _N_THREADS_RE.search(log_text) or _N_THREADS_GENERIC_RE.search(log_text)
    if match is None:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def parse_load_mode(log_text: str) -> bool | None:
    """Return the *effective* weight-load mode observed in a llama.cpp log.

    ``True`` = mmap, ``False`` = non-mmap, ``None`` = unknown.

    The value comes only from llama.cpp's own ``load_tensors: loading model tensors ...
    (load_mode = mmap|none)`` evidence line (emitted at high log verbosity). It is
    deliberately **never** inferred from the configured ``RuntimeParams.use_mmap``, the
    launch arguments, or OS memory statistics: those describe what was requested or are
    merely correlated, not what the backend actually did. Missing, malformed, or
    unrecognized evidence returns ``None`` (unknown) and is never classified as disabled.
    When several loads appear in one log the most recent one wins.
    """
    if not log_text:
        return None
    matches = _LOAD_MODE_RE.findall(log_text)
    if not matches:
        return None
    mode = matches[-1].strip().lower()
    if mode == "mmap":
        return True
    if mode == "none":
        return False
    return None  # pragma: no cover - the regex only captures mmap|none


def parse_device_info(stderr: str) -> DeviceInfo | None:
    backend: str | None = None
    device: str | None = None
    vram_total: int | None = None
    vram_free: int | None = None

    summary = _DEVICE_SUMMARY_RE.search(stderr)
    if summary:
        backend = summary.group(1).strip()
        device = summary.group(2).strip()
        vram_total = int(summary.group(3).replace(",", "")) * 1024 * 1024
        vram_free = int(summary.group(4).replace(",", "")) * 1024 * 1024

    device_line = _DEVICE_LINE_RE.search(stderr)
    if device_line:
        backend = backend or device_line.group(1).strip()
        device = device or device_line.group(2).strip()

    layers_offloaded: int | None = None
    layers_total: int | None = None
    offload = _OFFLOAD_RE.search(stderr)
    if offload:
        layers_offloaded = int(offload.group(1))
        layers_total = int(offload.group(2))

    if backend is None and device is None:
        return None
    return DeviceInfo(
        backend=backend or "unknown",
        device=device or "unknown",
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        layers_offloaded=layers_offloaded,
        layers_total=layers_total,
    )


def parse_generated_text(stdout: str) -> str | None:
    lines = stdout.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if line.startswith("> "):
            start = index + 1
    if start is None:
        return None
    collected: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if _PROMPT_STAT_RE.search(line) or stripped.startswith("Exiting"):
            break
        collected.append(line)
    text = "\n".join(collected).strip()
    return text or None


def parse_timings(stdout: str) -> tuple[float | None, float | None]:
    match = _PROMPT_STAT_RE.search(stdout)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def backend_type_from_name(name: str) -> BackendType:
    lowered = name.lower()
    if "vulkan" in lowered:
        return BackendType.VULKAN
    if "cuda" in lowered:
        return BackendType.CUDA
    if "rocm" in lowered or "hip" in lowered:
        return BackendType.ROCM
    if "metal" in lowered:
        return BackendType.METAL
    if "opencl" in lowered:
        return BackendType.OPENCL
    if "sycl" in lowered:
        return BackendType.SYCL
    if "cpu" in lowered:
        return BackendType.CPU
    return BackendType.CPU
