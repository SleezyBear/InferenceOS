"""Thin, real process wrapper around the acquired llama.cpp CLI.

This is *not* the ``BackendAdapter`` (that interface is defined in Phase 0 and
implemented in Phase 2). It is a low-level helper used by the Phase 0 verification
command to run the real binary, capture the real device/offload logs, and parse the
real generated text. Optimizer components must go through ``BackendAdapter``.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
_VERSION_RE = re.compile(r"version:\s*(\S+)")

# llama.cpp prefixes each log line with a monotonic ``a.bb.ccc.ddd`` stamp.
_LOG_TS_RE = re.compile(r"^(?P<ts>\d+\.\d{2}\.\d{3}\.\d{3})\s")
_LOAD_START_RE = re.compile(r"load_model:\s+loading model")
_READY_RE = re.compile(r"load_model:\s+initializing,\s+n_slots")
_TIMING_NUM_RE = re.compile(r"=\s*([\d.]+)\s*ms")
_TIMING_TOKENS_RE = re.compile(r"/\s*(\d+)\s*tokens")
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
    """Authoritative per-phase timings parsed from llama.cpp's own log.

    ``model_load_s`` is model-load-through-context-creation; ``ttft_s`` is
    model-ready to first generated token (prefill + one decode step).
    """

    prompt_eval_s: float | None = None
    prompt_tokens: int | None = None
    prefill_tokens_per_sec: float | None = None
    eval_s: float | None = None
    generated_tokens: int | None = None
    decode_tokens_per_sec: float | None = None
    total_s: float | None = None
    total_tokens: int | None = None
    load_start_s: float | None = None
    ready_s: float | None = None
    timing_s: float | None = None

    @property
    def model_load_s(self) -> float | None:
        if self.load_start_s is None or self.ready_s is None:
            return None
        return max(self.ready_s - self.load_start_s, 0.0)

    @property
    def ttft_s(self) -> float | None:
        if self.prompt_eval_s is None:
            return None
        if self.eval_s is not None and self.generated_tokens:
            return self.prompt_eval_s + self.eval_s / self.generated_tokens
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
    context_length: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    flash_attention: bool | None = None,
    device: str | None = None,
    load_mode: str | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
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
        context_length=context_length,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        flash_attention=flash_attention,
        device=device,
        load_mode=load_mode,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        verbose=verbose,
        log_verbosity=log_verbosity,
        ignore_eos=ignore_eos,
        no_jinja=no_jinja,
        json_schema=json_schema,
        extra_args=extra_args,
    )

    logger.info("Running llama.cpp: %s", " ".join(command))
    started = time.perf_counter()
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    observation = MemoryObservation()
    stop_event = threading.Event()
    rss_thread: threading.Thread | None = None
    if track_rss:
        rss_thread = threading.Thread(
            target=lambda: _observe_rss(proc.pid, stop_event, rss_interval_s, observation),
            name="rss-observer",
            daemon=True,
        )
        rss_thread.start()

    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise TimeoutError(f"llama-cli exceeded {timeout_s:.0f}s") from exc
    finally:
        stop_event.set()
        if rss_thread is not None:
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
    context_length: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    flash_attention: bool | None = None,
    device: str | None = None,
    load_mode: str | None = None,
    cache_type_k: str | None = None,
    cache_type_v: str | None = None,
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
        if "print_timing" not in raw:
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
            timings.decode_tokens_per_sec = _tps(raw)
        elif "total time" in raw:
            timings.total_s = _ms(raw)
            timings.total_tokens = _tokens(raw)
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
        return match.group(1)
    lines = combined.strip().splitlines()
    return lines[0].strip() if lines else None


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
