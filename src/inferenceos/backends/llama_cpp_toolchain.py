"""Thin, real process wrapper around the acquired llama.cpp CLI.

This is *not* the ``BackendAdapter`` (that interface is defined in Phase 0 and
implemented in Phase 2). It is a low-level helper used by the Phase 0 verification
command to run the real binary, capture the real device/offload logs, and parse the
real generated text. Optimizer components must go through ``BackendAdapter``.
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from inferenceos.contracts.enums import BackendType
from inferenceos.logging_config import get_logger

logger = get_logger("backends.llama_cpp")

CLI_NAMES = ("llama-cli.exe", "llama-cli")

_DEVICE_LINE_RE = re.compile(r"using device\s+(\w+)\s+\(([^)]+)\)")
_DEVICE_SUMMARY_RE = re.compile(
    r"-\s*(\w+)\s*:\s*([^(\n]+?)\(([\d,]+)\s*MiB,\s*([\d,]+)\s*MiB free\)"
)
_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU")
_PROMPT_STAT_RE = re.compile(
    r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]"
)


@dataclass
class DeviceInfo:
    backend: str
    device: str
    vram_total_bytes: int | None = None
    vram_free_bytes: int | None = None
    layers_offloaded: int | None = None
    layers_total: int | None = None


@dataclass
class LlamaRunResult:
    pid: int
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    command: list[str] = field(default_factory=list)

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
    verbose: bool = True,
    extra_args: tuple[str, ...] = (),
    timeout_s: float = 900.0,
) -> LlamaRunResult:
    """Run one real, single-turn inference and return captured real output."""
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
    if verbose:
        command += ["-v"]
    command += list(extra_args)

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
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise TimeoutError(
            f"llama-cli exceeded {timeout_s:.0f}s"
        ) from exc
    duration = time.perf_counter() - started
    return LlamaRunResult(
        pid=proc.pid,
        returncode=proc.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        duration_s=duration,
        command=command,
    )


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
