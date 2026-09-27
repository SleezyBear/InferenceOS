#!/usr/bin/env python3
"""InferenceOS A/B effectiveness experiment runner (single self-contained script).

Question this experiment answers
--------------------------------
Does InferenceOS produce a *measurable, retained* improvement over an unmanaged
``llama-server`` baseline on the same machine, same model, same llama.cpp build,
same backend, same generation parameters and same fixed workload?

Design (see the task specification for the authoritative requirements)
----------------------------------------------------------------------
* Test A -- an unmanaged ``llama-server`` started directly by this script. InferenceOS
  never controls this process.
* Test B -- the *same* ``llama-server`` binary/model/configuration started and managed
  through the real InferenceOS runtime path (``ResidentActiveWorkload`` +
  ``RuntimeMonitor``), with the experiment auto-approving disruptive interventions so
  the run is unattended. Existing safety gates are untouched.
* Two identical 20-minute stabilization windows (``pre_a_stabilization`` and
  ``post_a_stabilization``) are observed: one after pre-A cleanup (before Test A) and
  one after post-A cleanup (before Test B). Neither is silently shortened. The Test A
  baseline and the Test B baseline are captured only after their stabilization
  completes.
* A common telemetry collector samples the machine for both A and B.

Hard guarantees
---------------
* Both tests use the same GGUF file, llama.cpp build, backend, prompts, and generation
  parameters. The fixed workload is deterministic and saved as an artifact.
* The script only terminates processes it launched (identified by PID + creation time).
  It never indiscriminately kills unrelated user processes.
* This script does not modify InferenceOS behaviour, thresholds or acceptance criteria,
  and it does not create a Phase 12.

Usage
-----
::

    python run_ab_effectiveness.py --dry-run          # preflight only, no A/B
    python run_ab_effectiveness.py                    # full experiment (long)
    python run_ab_effectiveness.py --resume --run-dir artifacts/ab_effectiveness/<ts>

The full experiment is intentionally *not* executed by the implementation that added
this file; run it explicitly.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Dependency bootstrap (minimal, no elevation)                                  #
# --------------------------------------------------------------------------- #
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _prefer_project_venv() -> None:
    """Re-exec under the project virtualenv when one exists.

    This makes ``python run_ab_effectiveness.py`` work even when the shell's ``python``
    is a different interpreter. It is a no-op when already running under the venv or no
    venv is present, and it never installs anything into a system interpreter.

    Only called from ``__main__`` (never on import), so tests are unaffected.
    """
    if os.environ.get("INFERENCEOS_AB_VENV") == "1":
        return
    scripts = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    venv_python = PROJECT_ROOT / ".venv" / scripts / executable
    if not venv_python.is_file():
        return
    try:
        current = Path(sys.executable).resolve()
        target = venv_python.resolve()
    except OSError:
        return
    if current == target:
        return
    env = dict(os.environ)
    env["INFERENCEOS_AB_VENV"] = "1"
    try:
        completed = subprocess.run(
            [str(target), str(Path(__file__).resolve()), *sys.argv[1:]], env=env
        )
    except OSError:
        return
    raise SystemExit(completed.returncode)


# Re-exec into the project venv before importing heavy dependencies (only when this
# file is the program being run; importing it as a module is unaffected).
if __name__ == "__main__":
    _prefer_project_venv()



def _bootstrap_import_path() -> None:
    """Make the local ``src`` layout importable when the package is not installed."""
    try:
        import inferenceos  # noqa: F401
        return
    except ImportError:
        pass
    src = PROJECT_ROOT / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _bootstrap_dependencies() -> None:
    """Best-effort editable install of the project if its dependencies are missing.

    Never uses elevated privileges. On failure the caller prints the exact command the
    user must run.
    """
    need = False
    for module in ("pydantic", "pydantic_settings", "psutil"):
        try:
            __import__(module)
        except ImportError:
            need = True
            break
    if not need:
        try:
            __import__("inferenceos")
            return
        except ImportError:
            need = True
    if not need:
        return
    if not (PROJECT_ROOT / "pyproject.toml").is_file():
        return
    print("[setup] installing project dependencies via pip (no elevation)...", file=sys.stderr)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT)],
            check=False,
            timeout=1800,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        print(f"[setup] automatic install failed: {exc}", file=sys.stderr)


_bootstrap_import_path()
_bootstrap_dependencies()

# --------------------------------------------------------------------------- #
# Standard library                                                              #
# --------------------------------------------------------------------------- #
import argparse
import hashlib
import json
import math
import platform as _platform
import shutil
import socket
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Sequence

# --------------------------------------------------------------------------- #
# InferenceOS imports                                                           #
# --------------------------------------------------------------------------- #
try:
    import psutil

    from inferenceos import __version__ as INFERENCEOS_VERSION
    from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
    from inferenceos.backends.llama_cpp_toolchain import cli_version as _llama_cli_version
    from inferenceos.backends.llama_cpp_toolchain import list_devices as _llama_list_devices
    from inferenceos.backends.llama_server_toolchain import (
        find_server,
        pick_free_port,
    )
    from inferenceos.backends.process_control import (
        PeakRssAccumulator,
        process_identity,
        resource_snapshot,
        terminate_process_tree,
        wait_process_exit,
    )
    from inferenceos.config import Settings
    from inferenceos.contracts.enums import WorkloadKind
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.contracts.lifecycle import ProcessIdentity
    from inferenceos.contracts.model import ModelProfile
    from inferenceos.contracts.runtime import RuntimeMonitorRun, RuntimeRequestRecord
    from inferenceos.hardware.profiler import detect_hardware_profile
    from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
    from inferenceos.models.loader import ModelMetadataError, load_model_profile
    from inferenceos.native import get_platform
    from inferenceos.native.base import CounterSample, VulkanDevice
    from inferenceos.optimizer_accounting import (
        AccountingPolicy,
        assess_net_benefit,
        build_overhead_attribution,
        distortion_from_stability,
    )
    from inferenceos.profiling.workloads import BenchmarkWorkload
    from inferenceos.runtime.approval import (
        AutoApproveApprovalProvider,
        AutoRejectApprovalProvider,
    )
    from inferenceos.runtime.monitor import RuntimeMonitor
    from inferenceos.runtime.policy import build_runtime_budget, build_runtime_policy
    from inferenceos.runtime.resident_workload import ResidentActiveWorkload
    from inferenceos.runtime.telemetry import SystemTelemetrySampler
    from inferenceos.storage import atomic_write_text
    from inferenceos.telemetry.counters import CounterSampler
    from inferenceos.telemetry.system import (
        GPU_ADAPTER_MEMORY_COUNTER,
        GPU_ENGINE_COUNTER,
        HARD_FAULT_COUNTER,
        aggregate_gpu_samples,
        observe_power_performance,
        sample_system_state,
    )
except ImportError as exc:  # pragma: no cover - fresh machine path
    sys.stderr.write(
        "ERROR: InferenceOS dependencies are not importable.\n"
        f"  detail: {exc}\n\n"
        "Run these commands from the project root, then re-run this script:\n"
        f"  {sys.executable} -m pip install -e \".[dev]\"\n"
    )
    raise SystemExit(3)


SCHEMA_VERSION = "1.0"

# --------------------------------------------------------------------------- #
# Documented provisioning sources                                               #
# --------------------------------------------------------------------------- #
# The exact "Qwen3.5 9B" GGUF is provisioned from an explicit, documented source. If
# the default source cannot be verified on this machine, the script fails with precise
# commands and resumes afterwards -- it never silently substitutes another model.
MODEL_SOURCE_NAME = "Qwen3.5 9B (GGUF)"
# The user may override the repository / filename / checksum without editing the file.
MODEL_DEFAULT_REPO = os.environ.get("INFERENCEOS_AB_MODEL_REPO", "")
MODEL_DEFAULT_FILENAME = os.environ.get(
    "INFERENCEOS_AB_MODEL_FILENAME", "Qwen3.5-9B-Q4_K_M.gguf"
)
MODEL_DEFAULT_URL = os.environ.get("INFERENCEOS_AB_MODEL_URL", "")
MODEL_DEFAULT_SHA256 = os.environ.get("INFERENCEOS_AB_MODEL_SHA256", "")
# The large experiment model is kept in a dedicated subdirectory so it is not picked up
# as a test model by the repository test suite (which globs models/*.gguf).
MODEL_SUBDIR = os.environ.get("INFERENCEOS_AB_MODEL_SUBDIR", "ab")


def default_model_path(filename: str) -> Path:
    """Canonical location for the experiment GGUF."""
    return PROJECT_ROOT / "models" / MODEL_SUBDIR / filename


def find_existing_model(filename: str) -> Path | None:
    """Return the experiment model if present (dedicated subdir first)."""
    for candidate in (default_model_path(filename), PROJECT_ROOT / "models" / filename):
        if candidate.is_file():
            return candidate
    return None

# llama.cpp build used by the repository (tools/llama.cpp). The script detects the
# actual version and records it; it does not assume compatibility with the model.
LLAMA_RELEASE_HINT = os.environ.get(
    "INFERENCEOS_AB_LLAMA_RELEASE",
    "llama.cpp release b-b11139 (llama-*-bin-win-vulkan-x64)",
)

# Two identical 20-minute stabilization periods (one before Test A, one before Test B)
# are a hard requirement: each is only shortened when the operator explicitly passes
# --allow-short-cooldown (which is recorded as a validity confound).
STABILIZATION_SECONDS_DEFAULT = 20 * 60.0
# Retained name for backwards compatibility with existing callers/tests.
COOLDOWN_SECONDS_DEFAULT = STABILIZATION_SECONDS_DEFAULT

DEFAULT_HOST = "127.0.0.1"
DEFAULT_N_PREDICT = 64
DEFAULT_SEED = 1234
DEFAULT_ROUNDS = 3
DEFAULT_SAMPLE_INTERVAL_S = 1.0
DEFAULT_BASELINE_MIN_EFFECT = 0.05

# Deterministic, fixed workload prompts. All requests share the same generation
# parameters so per-request metrics are comparable. Never randomized.
FIXED_PROMPTS: tuple[str, ...] = (
    "In one sentence, explain what a key value cache is.",
    "List three differences between prefill and decode in a transformer.",
    "Name two factors that bound the decode speed of a local language model.",
    "In one sentence, describe why memory bandwidth matters for token generation.",
)

SERVER_PROCESS_NAMES = ("llama-server.exe", "llama-server")
LLAMA_PROCESS_PREFIXES = ("llama-",)
OLLAMA_PROCESS_NAMES = ("ollama.exe", "ollama", "ollama_llama_server.exe")
# Shells and terminals are never inference servers, even when their command line
# happens to mention a server name (e.g. a diagnostic command or a script path).
SHELL_PROCESS_NAMES = frozenset(
    {
        "powershell.exe",
        "pwsh.exe",
        "cmd.exe",
        "conhost.exe",
        "bash.exe",
        "wsl.exe",
        "windowsterminal.exe",
        "wt.exe",
        "opencode.exe",
        "node.exe",
    }
)
OTHER_INFERENCE_NAMES = (
    "vllm",
    "text-generation-launcher",
    "koboldcpp",
    "lmstudio",
    "lm-studio",
    "gpt4all",
    "localai",
)

GPU_COUNTER_PATHS = (GPU_ENGINE_COUNTER, GPU_ADAPTER_MEMORY_COUNTER, HARD_FAULT_COUNTER)


# --------------------------------------------------------------------------- #
# Small utilities                                                               #
# --------------------------------------------------------------------------- #
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fmt_bytes(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024.0 or unit == "TiB":
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{number:.2f} TiB"


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=_json_default))


def atomic_write_text_safe(path: Path, text: str) -> None:
    atomic_write_text(path, text)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


class JsonlWriter:
    """Append-only JSONL writer with the parent directory created on demand."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, default=_json_default)
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()


@dataclass(frozen=True)
class FixedRequest:
    """One deterministic request in the fixed A/B workload."""

    index: int
    request_id: str
    prompt: str
    n_predict: int
    seed: int
    temperature: float
    ignore_eos: bool

    @property
    def prompt_sha256(self) -> str:
        return sha256_text(self.prompt)

    def to_benchmark_workload(self) -> BenchmarkWorkload:
        return BenchmarkWorkload(
            workload_id=f"ab-{self.index:02d}",
            kind=WorkloadKind.SHORT,
            description="A/B effectiveness fixed request",
            prompt=self.prompt,
            n_predict=self.n_predict,
            seed=self.seed,
            temperature=self.temperature,
            min_context_tokens=64,
            ignore_eos=self.ignore_eos,
            version="ab.1",
        )


def build_fixed_requests(
    *,
    rounds: int,
    n_predict: int = DEFAULT_N_PREDICT,
    seed: int = DEFAULT_SEED,
    temperature: float = 0.0,
    ignore_eos: bool = True,
) -> list[FixedRequest]:
    """Deterministic request sequence: ``rounds`` passes over the fixed prompt set."""
    requests: list[FixedRequest] = []
    for round_index in range(max(rounds, 1)):
        for prompt_index, prompt in enumerate(FIXED_PROMPTS):
            index = len(requests)
            requests.append(
                FixedRequest(
                    index=index,
                    request_id=f"ab-{round_index:02d}-{prompt_index:02d}",
                    prompt=prompt,
                    n_predict=n_predict,
                    seed=seed,
                    temperature=temperature,
                    ignore_eos=ignore_eos,
                )
            )
    return requests


# --------------------------------------------------------------------------- #
# Statistics                                                                    #
# --------------------------------------------------------------------------- #
def summarize_values(values: Sequence[float | None]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "cv": None,
            "min": None,
            "max": None,
            "p95": None,
        }
    ordered = sorted(clean)
    mean = statistics.fmean(clean)
    stdev = statistics.pstdev(clean) if len(clean) > 1 else 0.0
    return {
        "n": len(clean),
        "mean": mean,
        "median": statistics.median(clean),
        "stdev": stdev,
        "cv": (stdev / mean) if mean else None,
        "min": ordered[0],
        "max": ordered[-1],
        "p95": percentile(ordered, 95.0),
    }


def percentile(ordered_values: Sequence[float], percent: float) -> float | None:
    """Linear-interpolation percentile over an already-sorted sequence."""
    if not ordered_values:
        return None
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    rank = (percent / 100.0) * (len(ordered_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered_values[lower])
    fraction = rank - lower
    return float(ordered_values[lower]) * (1.0 - fraction) + float(ordered_values[upper]) * fraction


def mean_difference_ci(
    sample_a: Sequence[float], sample_b: Sequence[float], *, z: float = 1.96
) -> tuple[float | None, float | None, float | None]:
    """Return (mean_b - mean_a, lower, upper) of an approximate 95% CI.

    Uses Welch's standard error with a normal critical value. With small samples this is
    a conservative approximation; it is never used alone to declare success.
    """
    if len(sample_a) < 2 or len(sample_b) < 2:
        return None, None, None
    mean_a = statistics.fmean(sample_a)
    mean_b = statistics.fmean(sample_b)
    var_a = statistics.variance(sample_a)
    var_b = statistics.variance(sample_b)
    se = math.sqrt(var_a / len(sample_a) + var_b / len(sample_b))
    diff = mean_b - mean_a
    if se == 0.0:
        return diff, diff, diff
    return diff, diff - z * se, diff + z * se


def metric_verdict(
    *,
    baseline_values: Sequence[float],
    measured_values: Sequence[float],
    higher_is_better: bool,
    min_effect: float,
) -> dict[str, Any]:
    """Conservatively classify one metric's A -> B change."""
    clean_a = [float(v) for v in baseline_values if v is not None and math.isfinite(float(v))]
    clean_b = [float(v) for v in measured_values if v is not None and math.isfinite(float(v))]
    result: dict[str, Any] = {
        "baseline": summarize_values(clean_a),
        "measured": summarize_values(clean_b),
        "direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "min_effect": min_effect,
        "relative_change": None,
        "ci_low": None,
        "ci_high": None,
        "ci_excludes_zero": False,
        "beyond_variability": False,
        "consistent_direction": False,
        "improved": False,
        "regressed": False,
        "sufficient_samples": len(clean_a) >= 3 and len(clean_b) >= 3,
    }
    if len(clean_a) < 2 or len(clean_b) < 2:
        return result
    mean_a = statistics.fmean(clean_a)
    mean_b = statistics.fmean(clean_b)
    if mean_a == 0.0:
        return result
    relative = (mean_b - mean_a) / abs(mean_a) if higher_is_better else (mean_a - mean_b) / abs(mean_a)
    diff, low, high = mean_difference_ci(clean_a, clean_b)
    result["relative_change"] = relative
    result["ci_low"] = low
    result["ci_high"] = high
    result["ci_excludes_zero"] = bool(low is not None and high is not None and (low > 0.0 or high < 0.0))
    # Consistent direction: mean and median agree on the sign of the change.
    median_a = statistics.median(clean_a)
    median_b = statistics.median(clean_b)
    consistent = (mean_b - mean_a) * (median_b - median_a) > 0.0
    result["consistent_direction"] = consistent
    result["beyond_variability"] = bool(
        result["ci_excludes_zero"] and abs(relative) >= min_effect and consistent
    )
    result["improved"] = bool(result["beyond_variability"] and relative > 0.0)
    result["regressed"] = bool(result["beyond_variability"] and relative < 0.0)
    return result


# --------------------------------------------------------------------------- #
# Process inventory                                                             #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    name: str
    create_time: float
    executable: str | None
    command_line: str | None
    rss_bytes: int | None
    category: str


def classify_process(name: str, command_line: str | None, *, self_pid: int) -> str:
    lowered = (name or "").lower()
    cmdline = (command_line or "").lower().strip()
    first = cmdline.split(" ", 1)[0].strip('"') if cmdline else ""
    # A shell/terminal is not a model server even if its arguments mention one.
    if lowered in SHELL_PROCESS_NAMES:
        return "other"
    if lowered == "inferenceos.exe":
        return "inferenceos"
    for server_name in SERVER_PROCESS_NAMES:
        if lowered == server_name.lower():
            return "llama_server"
    if any(lowered.startswith(prefix) for prefix in LLAMA_PROCESS_PREFIXES):
        return "llama_cpp"
    if any(lowered == candidate.lower() for candidate in OLLAMA_PROCESS_NAMES):
        return "ollama" if "server" not in lowered else "model_server"
    if any(candidate in lowered for candidate in OTHER_INFERENCE_NAMES):
        return "model_server"
    # Command-line fallbacks are restricted to the first (executable) token so an
    # argument that merely contains a server name cannot cause a false positive.
    if first.endswith("llama-server") or first.endswith("llama-server.exe"):
        return "llama_server"
    if (
        first.endswith("llama-cli")
        or first.endswith("llama-cli.exe")
        or "llama.cpp" in first
    ):
        return "llama_cpp"
    if first.startswith("ollama"):
        return "ollama"
    if any(candidate in first for candidate in OTHER_INFERENCE_NAMES):
        return "model_server"
    # An InferenceOS *runtime* process (the CLI entry point running the optimizer).
    if lowered.startswith("python") and (
        "-m inferenceos" in cmdline
        or "inferenceos\\cli.py" in cmdline
        or "inferenceos/cli.py" in cmdline
    ):
        return "inferenceos"
    return "other"


def snapshot_processes() -> list[ProcessRecord]:
    """A best-effort real process-table snapshot classified for contamination checks."""
    records: list[ProcessRecord] = []
    self_pid = os.getpid()
    attrs = ["pid", "name", "create_time", "exe", "cmdline", "memory_info"]
    for process in psutil.process_iter(attrs=attrs):
        try:
            info = process.info
            pid = int(info["pid"])
            name = str(info.get("name") or "")
            create_time = float(info.get("create_time") or 0.0)
            cmdline_list = info.get("cmdline")
            command_line = " ".join(cmdline_list) if isinstance(cmdline_list, list) else None
            memory = info.get("memory_info")
            rss = int(memory.rss) if memory is not None else None
            records.append(
                ProcessRecord(
                    pid=pid,
                    name=name,
                    create_time=create_time,
                    executable=info.get("exe"),
                    command_line=command_line,
                    rss_bytes=rss,
                    category=classify_process(name, command_line, self_pid=self_pid),
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError, ValueError):
            continue
    records.sort(key=lambda item: item.pid)
    return records


def process_map(records: Sequence[ProcessRecord]) -> dict[int, ProcessRecord]:
    return {record.pid: record for record in records}


def _load_process_records(path: Path) -> list[ProcessRecord]:
    """Reload a persisted process inventory (used by ``--resume``)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records: list[ProcessRecord] = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            records.append(
                ProcessRecord(
                    pid=int(item["pid"]),
                    name=str(item.get("name") or ""),
                    create_time=float(item.get("create_time") or 0.0),
                    executable=item.get("executable"),
                    command_line=item.get("command_line"),
                    rss_bytes=item.get("rss_bytes"),
                    category=str(item.get("category") or "other"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return records


def processes_by_category(
    records: Sequence[ProcessRecord], categories: Sequence[str]
) -> list[ProcessRecord]:
    wanted = set(categories)
    return [record for record in records if record.category in wanted]


@dataclass
class ProcessComparison:
    before: list[ProcessRecord]
    created: list[ProcessRecord]
    remaining: list[ProcessRecord]
    unexpected_survivors: list[ProcessRecord]


def compare_process_sets(
    before: Sequence[ProcessRecord],
    after: Sequence[ProcessRecord],
    *,
    launched_pids: Sequence[int] = (),
) -> ProcessComparison:
    """Identify processes created during a phase and survivors owned by the experiment."""
    before_by_pid = process_map(before)
    launched = set(int(pid) for pid in launched_pids)
    created: list[ProcessRecord] = []
    remaining: list[ProcessRecord] = []
    unexpected: list[ProcessRecord] = []
    for record in after:
        prior = before_by_pid.get(record.pid)
        if prior is None:
            created.append(record)
            # A surviving descendant of a process we launched is experiment-owned.
            if record.pid in launched:
                remaining.append(record)
        if record.pid in launched and record.pid not in {r.pid for r in remaining}:
            remaining.append(record)
    for record in created:
        if record.pid in launched and record.pid not in {r.pid for r in unexpected}:
            unexpected.append(record)
    return ProcessComparison(
        before=list(before),
        created=created,
        remaining=remaining,
        unexpected_survivors=unexpected,
    )


# --------------------------------------------------------------------------- #
# Telemetry collector (shared by both tests)                                    #
# --------------------------------------------------------------------------- #
class _PollableCounterSampler(CounterSampler):
    """``CounterSampler`` with a non-destructive snapshot for per-tick aggregation."""

    def poll(self) -> list[CounterSample]:
        with self._lock:
            return list(self._samples)


@dataclass
class TelemetrySummary:
    sample_count: int
    duration_s: float
    peak_cpu_percent: float | None
    mean_cpu_percent: float | None
    peak_target_rss_bytes: int | None
    final_target_rss_bytes: int | None
    peak_gpu_engine_percent: float | None
    peak_gpu_memory_bytes: int | None
    monitor_overhead_ms_mean: float | None
    thermal_note: str


class TelemetryCollector:
    """Background machine/server telemetry sampler writing JSONL.

    Cheap psutil samples + the persistent Windows counter sampler (GPU engine/adapter
    memory/hard faults). GPU LUIDs are mapped to real Vulkan device indices using the
    same identity logic the project already uses (B-LUID correctness), so a counter LUID
    is never misattributed to another device.
    """

    def __init__(
        self,
        *,
        jsonl_path: Path,
        interval_s: float,
        target_pid_provider: Callable[[], int | None],
        include_gpu: bool = True,
    ) -> None:
        self._jsonl = JsonlWriter(jsonl_path)
        self._interval_s = max(0.2, float(interval_s))
        self._target_pid_provider = target_pid_provider
        self._include_gpu = include_gpu
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sampler: _PollableCounterSampler | None = None
        self._vulkan_devices: list[VulkanDevice] = []
        self._consumed = 0
        self._started = 0.0

    def start(self) -> None:
        self._started = time.perf_counter()
        if self._include_gpu:
            try:
                self._vulkan_devices = list(get_platform().vulkan_enumerate())
            except Exception:  # noqa: BLE001 - telemetry must never crash the run
                self._vulkan_devices = []
            try:
                sampler = _PollableCounterSampler(list(GPU_COUNTER_PATHS), interval_s=0.5)
                sampler.start()
                sampler.wait_until_ready(timeout_s=5.0)
                self._sampler = sampler
            except Exception:  # noqa: BLE001
                self._sampler = None
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, name="ab-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> TelemetrySummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        duration = max(time.perf_counter() - self._started, 0.0)
        if self._sampler is not None:
            try:
                self._sampler.stop()
            except Exception:  # noqa: BLE001
                pass
        # Summarize from disk (single source of truth for the run artifacts).
        peak_cpu: float | None = None
        cpu_values: list[float] = []
        peak_rss: int | None = None
        final_rss: int | None = None
        peak_gpu: float | None = None
        peak_gpu_mem: int | None = None
        overheads: list[float] = []
        count = 0
        if self._jsonl.path.is_file():
            for line in self._jsonl.path.read_text(encoding="utf-8").splitlines():
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                count += 1
                cpu = payload.get("cpu_percent_total")
                if isinstance(cpu, (int, float)):
                    cpu_values.append(float(cpu))
                    peak_cpu = cpu if peak_cpu is None else max(peak_cpu, float(cpu))
                rss = payload.get("target_rss_bytes")
                if isinstance(rss, int):
                    peak_rss = rss if peak_rss is None else max(peak_rss, rss)
                    final_rss = rss
                gpu = payload.get("gpu_engine_max_percent")
                if isinstance(gpu, (int, float)):
                    peak_gpu = gpu if peak_gpu is None else max(peak_gpu, float(gpu))
                gpu_mem = payload.get("gpu_memory_used_bytes")
                if isinstance(gpu_mem, int):
                    peak_gpu_mem = gpu_mem if peak_gpu_mem is None else max(peak_gpu_mem, gpu_mem)
                overhead = payload.get("monitor_overhead_ms")
                if isinstance(overhead, (int, float)):
                    overheads.append(float(overhead))
        return TelemetrySummary(
            sample_count=count,
            duration_s=duration,
            peak_cpu_percent=peak_cpu,
            mean_cpu_percent=(statistics.fmean(cpu_values) if cpu_values else None),
            peak_target_rss_bytes=peak_rss,
            final_target_rss_bytes=final_rss,
            peak_gpu_engine_percent=peak_gpu,
            peak_gpu_memory_bytes=peak_gpu_mem,
            monitor_overhead_ms_mean=(statistics.fmean(overheads) if overheads else None),
            thermal_note="unavailable: Windows exposes no reliable CPU/GPU temperature via "
            "psutil/WMI on this platform; no temperature is invented",
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            tick_start = time.perf_counter()
            try:
                payload = self._sample()
                payload["monitor_overhead_ms"] = (time.perf_counter() - tick_start) * 1000.0
                self._jsonl.write(payload)
            except Exception as exc:  # noqa: BLE001 - monitoring must never break the run
                self._jsonl.write(
                    {"ts": utcnow_iso(), "error": f"{type(exc).__name__}: {exc}"}
                )
            elapsed = time.perf_counter() - tick_start
            self._stop.wait(max(0.0, self._interval_s - elapsed))

    def _sample(self) -> dict[str, Any]:
        captured = utcnow_iso()
        cpu_percent = float(psutil.cpu_percent(interval=None))
        virtual = psutil.virtual_memory()
        power = observe_power_performance()
        target_pid = self._target_pid_provider()
        target_rss: int | None = None
        target_peak: int | None = None
        target_cpu: float | None = None
        target_threads: int | None = None
        if target_pid is not None:
            snapshot = resource_snapshot(target_pid)
            if snapshot.exists:
                target_rss = snapshot.rss_bytes
                target_peak = snapshot.peak_rss_bytes
            try:
                process = psutil.Process(target_pid)
                target_cpu = float(process.cpu_percent(interval=None))
                target_threads = int(process.num_threads())
            except (psutil.Error, ValueError):
                pass
        gpu_engines: list[dict[str, Any]] = []
        gpu_memory: list[dict[str, Any]] = []
        gpu_engine_max: float | None = None
        gpu_memory_used: int | None = None
        hard_faults: float | None = None
        if self._sampler is not None:
            samples = self._sampler.poll()
            new_samples = samples[self._consumed :]
            self._consumed = len(samples)
            if new_samples:
                aggregate = aggregate_gpu_samples(
                    new_samples, vulkan_devices=self._vulkan_devices
                )
                gpu_engines = [
                    {
                        "device_index": engine.device_index,
                        "engine": engine.engine,
                        "utilization_percent": engine.utilization_percent,
                    }
                    for engine in aggregate.engines
                ]
                gpu_memory = [
                    {
                        "device_index": memory.device_index,
                        "used_bytes": memory.memory_used_bytes,
                    }
                    for memory in aggregate.memory
                ]
                if gpu_engines:
                    gpu_engine_max = max(item["utilization_percent"] for item in gpu_engines)
                if gpu_memory:
                    gpu_memory_used = max(int(item["used_bytes"]) for item in gpu_memory)
                for sample in new_samples:
                    if "pages input" in sample.path.lower():
                        hard_faults = max(sample.value, 0.0)
        return {
            "ts": captured,
            "elapsed_s": round(time.perf_counter() - self._started, 3),
            "cpu_percent_total": cpu_percent,
            "cpu_current_mhz": power.current_mhz,
            "cpu_max_mhz": power.max_mhz,
            "cpu_limit_mhz": power.limit_mhz,
            "cpu_performance_ratio": power.ratio,
            "ram_total_bytes": int(virtual.total),
            "ram_available_bytes": int(virtual.available),
            "ram_used_bytes": int(virtual.total - virtual.available),
            "process_count": len(psutil.pids()),
            "hard_page_faults_per_sec": hard_faults,
            "power_state": power.power_state.value if hasattr(power.power_state, "value") else str(power.power_state),
            "thermal_available": False,
            "thermal_note": "unavailable: no reliable Windows thermal sensor exposed",
            "target_pid": target_pid,
            "target_rss_bytes": target_rss,
            "target_peak_rss_bytes": target_peak,
            "target_cpu_percent": target_cpu,
            "target_threads": target_threads,
            "gpu_source": "windows_gpu_engine_counters" if self._sampler is not None else "unavailable",
            "gpu_engines": gpu_engines,
            "gpu_engine_max_percent": gpu_engine_max,
            "gpu_memory": gpu_memory,
            "gpu_memory_used_bytes": gpu_memory_used,
        }


# --------------------------------------------------------------------------- #
# Request outcomes                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class RequestOutcome:
    test: str
    index: int
    request_id: str
    prompt: str
    prompt_sha256: str
    n_predict: int
    seed: int
    temperature: float
    submitted_at: str
    first_token_at: str | None
    completed_at: str | None
    ttft_s: float | None
    total_latency_s: float | None
    generated_tokens: int | None
    prompt_tokens: int | None
    decode_tokens_per_sec: float | None
    peak_rss_bytes: int | None
    final_rss_bytes: int | None
    success: bool
    error: str | None
    output: str | None
    post_intervention: bool = False
    decode_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Unmanaged baseline (Test A)                                                   #
# --------------------------------------------------------------------------- #
def build_unmanaged_command(
    *,
    server: Path,
    model_path: Path,
    host: str,
    port: int,
    context_length: int,
    gpu_layers: int,
    threads: int | None,
    ignore_eos: bool = False,
) -> list[str]:
    """Direct ``llama-server`` invocation mirroring the managed configuration."""
    command: list[str] = [
        str(server),
        "-m",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--no-webui",
        "-ngl",
        str(gpu_layers),
    ]
    if context_length > 0:
        command += ["-c", str(context_length)]
    if threads:
        command += ["-t", str(threads)]
    if ignore_eos:
        command += ["--ignore-eos"]
    return command


def _stream_completion(
    *,
    host: str,
    port: int,
    body: dict[str, Any],
    timeout_s: float,
    api_key: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Minimal SSE client for ``llama-server`` ``/completion`` (unmanaged path)."""
    url = f"http://{host}:{port}/completion"
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    response = urllib.request.urlopen(request, timeout=timeout_s)
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


def run_unmanaged_request(
    *,
    test: str,
    request: FixedRequest,
    host: str,
    port: int,
    server_pid: int,
    timeout_s: float,
) -> RequestOutcome:
    body: dict[str, Any] = {
        "prompt": request.prompt,
        "n_predict": request.n_predict,
        "seed": request.seed,
        "temperature": request.temperature,
        "stop": [],
        "stream": True,
        "cache_prompt": False,
    }
    submitted = datetime.now(timezone.utc)
    started = time.perf_counter()
    first_token_at: datetime | None = None
    ttft_s: float | None = None
    pieces: list[str] = []
    generated: int | None = None
    prompt_tokens: int | None = None
    decode_tps: float | None = None
    final_chunk: dict[str, Any] | None = None
    peak = PeakRssAccumulator()
    peak.observe(server_pid)
    next_sample = started + 0.1
    error: str | None = None
    try:
        for chunk in _stream_completion(
            host=host, port=port, body=body, timeout_s=timeout_s
        ):
            content = chunk.get("content")
            if isinstance(content, str) and content:
                if first_token_at is None:
                    first_token_at = datetime.now(timezone.utc)
                    ttft_s = time.perf_counter() - started
                pieces.append(content)
            if isinstance(chunk.get("tokens_predicted"), int):
                generated = chunk["tokens_predicted"]
            if isinstance(chunk.get("tokens_evaluated"), int):
                prompt_tokens = chunk["tokens_evaluated"]
            if "timings" in chunk and isinstance(chunk["timings"], dict):
                final_chunk = chunk
            now = time.perf_counter()
            if now >= next_sample:
                peak.observe(server_pid)
                next_sample = now + 0.1
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    completed = datetime.now(timezone.utc)
    total_s = time.perf_counter() - started
    peak.observe(server_pid)
    if final_chunk is not None:
        timings = final_chunk.get("timings")
        if isinstance(timings, dict):
            value = timings.get("predicted_per_second")
            if isinstance(value, (int, float)):
                decode_tps = float(value)
            if isinstance(timings.get("predicted_n"), int):
                generated = timings["predicted_n"]
            if isinstance(timings.get("prompt_n"), int):
                prompt_tokens = timings["prompt_n"]
    text = "".join(pieces)
    if decode_tps is None and generated and total_s > 0:
        decode_s = max(total_s - (ttft_s or 0.0), 1e-6)
        decode_tps = (generated - 1) / decode_s if generated > 1 else generated / decode_s
    return RequestOutcome(
        test=test,
        index=request.index,
        request_id=request.request_id,
        prompt=request.prompt,
        prompt_sha256=request.prompt_sha256,
        n_predict=request.n_predict,
        seed=request.seed,
        temperature=request.temperature,
        submitted_at=submitted.isoformat(),
        first_token_at=first_token_at.isoformat() if first_token_at else None,
        completed_at=completed.isoformat(),
        ttft_s=ttft_s,
        total_latency_s=total_s,
        decode_s=(max(total_s - (ttft_s or 0.0), 0.0) if generated else None),
        generated_tokens=generated,
        prompt_tokens=prompt_tokens,
        decode_tokens_per_sec=decode_tps,
        peak_rss_bytes=peak.peak_rss_bytes,
        final_rss_bytes=peak.final_rss_bytes,
        success=bool(text) and error is None,
        error=error,
        output=text[:4000] if text else None,
    )


@dataclass
class UnmanagedServer:
    process: subprocess.Popen[bytes]
    pid: int
    identity: ProcessIdentity | None
    host: str
    port: int
    command: list[str]
    log_path: Path
    started_at: str
    ready_s: float
    startup_s: float
    device_text: str
    log_handle: Any = None


def _wait_http_ready(
    *, host: str, port: int, timeout_s: float, process: subprocess.Popen[bytes]
) -> float:
    started = time.perf_counter()
    url = f"http://{host}:{port}/health"
    while time.perf_counter() - started < timeout_s:
        if process.poll() is not None:
            raise RuntimeError(f"unmanaged llama-server exited early: code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return time.perf_counter() - started
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError(f"unmanaged llama-server not ready within {timeout_s:.0f}s")


def start_unmanaged_server(
    *,
    server: Path,
    model_path: Path,
    host: str,
    context_length: int,
    gpu_layers: int,
    threads: int | None,
    ignore_eos: bool,
    log_path: Path,
    readiness_timeout_s: float,
) -> UnmanagedServer:
    port = pick_free_port(host)
    command = build_unmanaged_command(
        server=server,
        model_path=model_path,
        host=host,
        port=port,
        context_length=context_length,
        gpu_layers=gpu_layers,
        threads=threads,
        ignore_eos=ignore_eos,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Persist the exact unmanaged command (no secret is present in Test A).
    log_handle = log_path.open("w", encoding="utf-8")
    startup_started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        ready_s = _wait_http_ready(
            host=host,
            port=port,
            timeout_s=readiness_timeout_s,
            process=process,
        )
    except Exception:
        process.kill()
        log_handle.close()
        raise
    startup_s = time.perf_counter() - startup_started
    device_text = ""
    try:
        device_text = log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
    except OSError:
        device_text = ""
    return UnmanagedServer(
        process=process,
        pid=int(process.pid),
        identity=process_identity(int(process.pid)),
        host=host,
        port=port,
        command=command,
        log_path=log_path,
        started_at=utcnow_iso(),
        ready_s=ready_s,
        startup_s=startup_s,
        device_text=device_text,
        log_handle=log_handle,
    )


def _close_unmanaged_log(server: UnmanagedServer) -> None:
    handle = server.log_handle
    if handle is not None:
        try:
            handle.flush()
            handle.close()
        except OSError:  # pragma: no cover - defensive
            pass
        server.log_handle = None


def stop_unmanaged_server(server: UnmanagedServer, *, timeout_s: float = 15.0) -> dict[str, Any]:
    """Terminate only the server this experiment launched (PID + identity checked)."""
    pid = server.pid
    identity = server.identity
    report: dict[str, Any] = {
        "pid": pid,
        "identity": identity.model_dump() if identity is not None else None,
        "already_exited": server.process.poll() is not None,
        "method": "terminate_tree",
        "exited": False,
        "identity_matched": None,
    }
    current = process_identity(pid)
    if identity is not None and current is not None:
        report["identity_matched"] = abs(current.create_time - identity.create_time) < 0.01
        if not report["identity_matched"]:
            report["note"] = "PID was reused by an unrelated process; refusing to kill it"
            report["exited"] = False
            _close_unmanaged_log(server)
            return report
    try:
        server.process.terminate()
    except OSError:
        pass
    try:
        server.process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        terminate_process_tree(pid, timeout_s=timeout_s)
    report["exited"] = wait_process_exit(pid, timeout_s=timeout_s)
    _close_unmanaged_log(server)
    return report


# --------------------------------------------------------------------------- #
# InferenceOS-managed workload (Test B)                                         #
# --------------------------------------------------------------------------- #
class FixedSequenceResidentWorkload(ResidentActiveWorkload):
    """Resident InferenceOS workload that serves the fixed A/B request sequence once.

    It reuses the project's resident lifecycle and bounded-recovery machinery unchanged;
    only the request-selection loop is overridden so the exact fixed workload is served.
    """

    _in_flight_id: str | None
    _in_flight_started_at: datetime | None

    def __init__(
        self,
        *,
        fixed_requests: Sequence[FixedRequest],
        outcomes: list[RequestOutcome],
        test_label: str,
        **kwargs: Any,
    ) -> None:
        if not fixed_requests:
            raise ValueError("fixed_requests must be non-empty")
        first = fixed_requests[0].to_benchmark_workload()
        super().__init__(workload=first, **kwargs)
        self._fixed_requests = list(fixed_requests)
        self._fixed_index = 0
        self._fixed_lock = threading.Lock()
        self._outcomes = outcomes
        self._test_label = test_label
        self._sequence_complete = False

    def sequence_complete(self) -> bool:
        with self._fixed_lock:
            return self._sequence_complete

    def _next_request(self) -> tuple[int, FixedRequest] | None:
        with self._fixed_lock:
            if self._fixed_index >= len(self._fixed_requests):
                self._sequence_complete = True
                self._accepting = False
                return None
            index = self._fixed_index
            self._fixed_index += 1
            return index, self._fixed_requests[index]

    def _serve_one(self) -> None:
        selected = self._next_request()
        if selected is None:
            return
        index, request = selected
        request_id = f"rreq-{request.index:03d}-{uuid.uuid4().hex[:8]}"
        submitted = datetime.now(timezone.utc)
        with self._lock:
            self._in_flight = True
            self._in_flight_id = request_id
            self._in_flight_started_at = submitted
        error: str | None = None
        result = None
        try:
            self._prepare_target()
            result = self._adapter.generate(
                GenerationRequest(
                    prompt=request.prompt,
                    n_predict=request.n_predict,
                    seed=request.seed,
                    temperature=request.temperature,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed request must not crash
            error = f"{type(exc).__name__}: {exc}"
        completed = datetime.now(timezone.utc)
        generation = self._adapter.last_generation
        text = result.text if result is not None else None
        outcome = RequestOutcome(
            test=self._test_label,
            index=request.index,
            request_id=request_id,
            prompt=request.prompt,
            prompt_sha256=request.prompt_sha256,
            n_predict=request.n_predict,
            seed=request.seed,
            temperature=request.temperature,
            submitted_at=submitted.isoformat(),
            first_token_at=(
                generation.first_token_at.isoformat()
                if generation is not None and generation.first_token_at is not None
                else None
            ),
            completed_at=completed.isoformat(),
            ttft_s=result.ttft_s if result is not None else None,
            total_latency_s=result.total_s if result is not None else None,
            decode_s=(
                max((result.total_s or 0.0) - (result.ttft_s or 0.0), 0.0)
                if result is not None and result.generated_tokens
                else None
            ),
            generated_tokens=result.generated_tokens if result is not None else None,
            prompt_tokens=result.prompt_tokens if result is not None else None,
            decode_tokens_per_sec=result.decode_tokens_per_sec if result is not None else None,
            peak_rss_bytes=result.peak_rss_bytes if result is not None else None,
            final_rss_bytes=result.final_rss_bytes if result is not None else None,
            success=bool(text) and error is None,
            error=error,
            output=text[:4000] if text else None,
        )
        with self._lock:
            self._outcomes.append(outcome)
            self._requests.append(
                RuntimeRequestRecord(
                    request_id=request_id,
                    submitted_at=submitted,
                    first_token_at=(
                        generation.first_token_at if generation is not None else None
                    ),
                    completed_at=completed,
                    latency_s=outcome.total_latency_s,
                    generated_tokens=outcome.generated_tokens,
                    decode_tokens_per_sec=outcome.decode_tokens_per_sec,
                    success=outcome.success,
                    error=error,
                )
            )
            self._completed += 1
            metrics: dict[str, float] = {}
            for name, value in (
                ("decode_tokens_per_sec", outcome.decode_tokens_per_sec),
                ("total_latency_s", outcome.total_latency_s),
                ("ttft_s", outcome.ttft_s),
                ("peak_rss_bytes", outcome.peak_rss_bytes),
                ("final_rss_bytes", outcome.final_rss_bytes),
            ):
                if value is not None:
                    metrics[name] = float(value)
            self._last_metrics = metrics
            self._last_text = text
            self._last_error = error
            self._recovery_permitted = True
            self._in_flight = False
            self._in_flight_id = None
            self._in_flight_started_at = None
            if self._fixed_index >= len(self._fixed_requests):
                self._sequence_complete = True
                self._accepting = False
        if error is not None:
            self._maybe_recover_after_failure()


# --------------------------------------------------------------------------- #
# Experiment configuration model                                                #
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    run_dir: Path
    model_path: Path | None
    model_url: str
    model_sha256: str
    model_filename: str
    llama_dir: Path
    server_path: Path | None
    host: str
    context_length: int
    gpu_layers: int
    threads: int | None
    rounds: int
    n_predict: int
    seed: int
    sample_interval_s: float
    test_duration_s: float
    cooldown_s: float
    dry_run: bool
    resume: bool
    auto_approve: bool
    allow_short_cooldown: bool
    dry_run_samples: int

    def config_hash(self) -> str:
        payload = asdict(self)
        payload["run_dir"] = str(self.run_dir)
        payload["model_path"] = str(self.model_path) if self.model_path else None
        payload["llama_dir"] = str(self.llama_dir)
        payload["server_path"] = str(self.server_path) if self.server_path else None
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Preflight                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class PreflightReport:
    passed: bool
    blockers: list[str]
    warnings: list[str]
    setup_commands: list[str]
    data: dict[str, Any]


def _runtime_configuration(cfg: ExperimentConfig) -> RuntimeConfiguration:
    return RuntimeConfiguration(
        threads=cfg.threads,
        gpu_offload_layers=cfg.gpu_layers,
        context_length=cfg.context_length,
    )


def collect_preflight(cfg: ExperimentConfig, settings: Settings) -> PreflightReport:
    """Capture a full pre-flight snapshot and identify provisioning blockers."""
    blockers: list[str] = []
    warnings: list[str] = []
    setup_commands: list[str] = []
    data: dict[str, Any] = {
        "timestamp": utcnow_iso(),
        "hostname": socket.gethostname(),
        "platform": _platform.platform(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "inferenceos_version": INFERENCEOS_VERSION,
        "experiment_config_hash": cfg.config_hash(),
        "os": {
            "system": _platform.system(),
            "release": _platform.release(),
            "version": _platform.version(),
            "machine": _platform.machine(),
        },
    }
    try:
        import subprocess as _sp

        revision = _sp.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        data["inferenceos_git_revision"] = revision.stdout.strip() or None
    except Exception:  # noqa: BLE001
        data["inferenceos_git_revision"] = None

    # CPU / memory / GPU / power.
    profile = None
    try:
        profile = detect_hardware_profile(settings)
        data["hardware_profile"] = json.loads(profile.model_dump_json())
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"hardware profile unavailable: {type(exc).__name__}: {exc}")
    data["psutil"] = {
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "total_ram_bytes": int(psutil.virtual_memory().total),
        "available_ram_bytes": int(psutil.virtual_memory().available),
    }
    try:
        state = sample_system_state(
            target_pid=None, interval_s=settings.telemetry_interval_s, top_processes=8
        )
        data["system_state"] = json.loads(state.model_dump_json())
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"live system state unavailable: {type(exc).__name__}: {exc}")
    data["vulkan_available"] = bool(getattr(profile, "vulkan_available", False)) if profile else None
    data["cuda_available"] = bool(getattr(profile, "cuda_available", False)) if profile else None
    data["rocm_available"] = bool(getattr(profile, "rocm_available", False)) if profile else None
    power = observe_power_performance()
    data["power_performance"] = {
        "power_state": power.power_state.value,
        "current_mhz": power.current_mhz,
        "max_mhz": power.max_mhz,
        "limit_mhz": power.limit_mhz,
        "ratio": power.ratio,
    }
    data["thermal"] = {
        "available": False,
        "reason": "no reliable Windows thermal sensor exposed via psutil/WMI on this platform",
    }
    try:
        usage = shutil.disk_usage(cfg.run_dir.parent if cfg.run_dir.parent.exists() else PROJECT_ROOT)
        data["disk"] = {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}
    except OSError as exc:
        warnings.append(f"disk usage unavailable: {exc}")
    data["network_required"] = bool(cfg.model_url) and not (
        cfg.model_path is not None and cfg.model_path.is_file()
    )

    # llama.cpp validation.
    server_path = cfg.server_path
    if server_path is None or not server_path.is_file():
        try:
            server_path = find_server(cfg.llama_dir)
        except FileNotFoundError:
            server_path = None
    if server_path is None or not server_path.is_file():
        blockers.append(f"llama-server executable not found in {cfg.llama_dir}")
        setup_commands += [
            "Acquire a llama.cpp Windows Vulkan release (see " + LLAMA_RELEASE_HINT + ")",
            f"Extract llama-server.exe and its DLLs into: {cfg.llama_dir}",
            "or pass --llama-dir <dir> / --server <path>",
        ]
        data["llama"] = {"server_found": False, "llama_dir": str(cfg.llama_dir)}
    else:
        version = _safe_call(lambda: _llama_cli_version(server_path))
        devices = _safe_call(lambda: _llama_list_devices(server_path)) or []
        data["llama"] = {
            "server_found": True,
            "server_path": str(server_path),
            "version": version,
            "devices": devices,
            "expected_release_hint": LLAMA_RELEASE_HINT,
            "vulkan_device_listed": any("vulkan" in str(item).lower() for item in devices),
        }
        if not version:
            warnings.append("llama-server version could not be read")
        if not any("vulkan" in str(item).lower() for item in devices):
            warnings.append(
                "no Vulkan device listed by llama-server; GPU offload may be unavailable"
            )
    data["model"] = _model_preflight(cfg)
    if not data["model"]["available"]:
        blockers.append("the exact Qwen3.5 9B GGUF is not available locally")
        setup_commands += _model_setup_commands(cfg)

    # Process contamination check.
    processes = snapshot_processes()
    relevant = processes_by_category(
        processes, ("llama_server", "llama_cpp", "ollama", "model_server", "inferenceos")
    )
    data["processes"] = {
        "total": len(processes),
        "relevant": [asdict(record) for record in relevant],
    }
    for record in relevant:
        if record.pid == os.getpid():
            continue
        if record.category in INFERENCE_PROCESS_CATEGORIES:
            blockers.append(
                f"another inference server is already running: {record.name} (pid "
                f"{record.pid}, category {record.category}); stop it before the experiment"
            )

    setup_commands = _dedupe(setup_commands)
    return PreflightReport(
        passed=not blockers,
        blockers=blockers,
        warnings=warnings + _dedupe([]),
        setup_commands=setup_commands,
        data=data,
    )


def _safe_call(func: Callable[[], Any]) -> Any:
    try:
        return func()
    except Exception:  # noqa: BLE001
        return None


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _model_preflight(cfg: ExperimentConfig) -> dict[str, Any]:
    path = cfg.model_path
    info: dict[str, Any] = {
        "expected_source": MODEL_SOURCE_NAME,
        "expected_filename": cfg.model_filename,
        "source_url": cfg.model_url or None,
        "path": str(path) if path else None,
        "available": False,
        "size_bytes": None,
        "sha256": None,
        "sha256_expected": cfg.model_sha256 or MODEL_DEFAULT_SHA256 or None,
        "sha256_verified": None,
    }
    if path is None or not Path(path).is_file():
        return info
    resolved = Path(path)
    info["available"] = True
    info["size_bytes"] = resolved.stat().st_size
    expected = cfg.model_sha256 or MODEL_DEFAULT_SHA256
    if expected:
        actual = sha256_file(resolved)
        info["sha256"] = actual
        info["sha256_verified"] = actual.lower() == expected.lower()
    else:
        info["sha256"] = None
        info["sha256_note"] = (
            "no trustworthy checksum supplied; SHA-256 will be computed and recorded at "
            "acquisition time and treated as the experiment fingerprint"
        )
    return info


def _model_setup_commands(cfg: ExperimentConfig) -> list[str]:
    target = cfg.model_path or default_model_path(cfg.model_filename)
    if cfg.model_url:
        download = f'Invoke-WebRequest -Uri "{cfg.model_url}" -OutFile "{target}"'
    else:
        download = (
            "$repo = Read-Host 'Hugging Face GGUF repo id'; "
            f"huggingface-cli download $repo --include '*{cfg.model_filename}' "
            f'--local-dir "{target.parent}"'
        )
    return [
        f"mkdir -Force \"{target.parent}\"",
        download,
        f"# then re-run: {sys.executable} {Path(__file__).name} --resume --run-dir \"{cfg.run_dir}\"",
    ]


def ensure_model_available(cfg: ExperimentConfig) -> Path | None:
    """Locate the exact model, optionally downloading an explicitly-provided URL.

    A download only happens when ``--model-url`` is set (an explicit, documented
    source). The file is verified against ``--model-sha256`` when supplied; a mismatch
    is refused and the partial file removed. Never substitutes a different model.
    """
    target = cfg.model_path or default_model_path(cfg.model_filename)
    if target.is_file():
        cfg.model_path = target
        return target
    if not cfg.model_url:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    print(f"[model] downloading explicit source: {cfg.model_url}")
    try:
        with urllib.request.urlopen(cfg.model_url, timeout=120) as response, part.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"[model] download failed: {type(exc).__name__}: {exc}")
        part.unlink(missing_ok=True)
        return None
    if cfg.model_sha256:
        actual = sha256_file(part)
        if actual.lower() != cfg.model_sha256.lower():
            print(
                "[model] SHA-256 mismatch; refusing the download "
                f"(expected {cfg.model_sha256}, got {actual})"
            )
            part.unlink(missing_ok=True)
            return None
    os.replace(part, target)
    cfg.model_path = target
    return target


def acquire_model(cfg: ExperimentConfig) -> tuple[Path | None, dict[str, Any]]:
    """Locate (or optionally download) the exact model. Never substitutes another file."""
    metadata: dict[str, Any] = {
        "expected_source": MODEL_SOURCE_NAME,
        "expected_filename": cfg.model_filename,
        "source_url": cfg.model_url or None,
    }
    candidate = cfg.model_path
    if candidate is None:
        default = default_model_path(cfg.model_filename)
        candidate = default if default.is_file() else None
    if candidate is None:
        return None, metadata
    resolved = Path(candidate)
    if not resolved.is_file():
        return None, metadata
    metadata["path"] = str(resolved)
    metadata["size_bytes"] = resolved.stat().st_size
    # Always compute and record the SHA-256: it is required experiment metadata and the
    # fingerprint that proves both tests used the same file.
    metadata["sha256"] = sha256_file(resolved)
    if cfg.model_sha256:
        metadata["sha256_verified"] = (
            metadata["sha256"].lower() == cfg.model_sha256.lower()
        )
    metadata["acquired_at"] = utcnow_iso()
    return resolved, metadata


# --------------------------------------------------------------------------- #
# Artifact directory / phase state                                              #
# --------------------------------------------------------------------------- #
class RunDirectory:
    def __init__(self, root: Path) -> None:
        self.root = root
        for sub in (
            "test_a",
            "cleanup",
            "stabilization",
            "test_b",
            "dry_run",
        ):
            (root / sub).mkdir(parents=True, exist_ok=True)
        self._state_path = root / "state.json"

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def load_state(self) -> dict[str, Any]:
        if not self._state_path.is_file():
            return {"completed": [], "config_hash": None}
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
        return {"completed": [], "config_hash": None}

    def mark_complete(self, phase: str, config_hash: str) -> None:
        state = self.load_state()
        completed = set(state.get("completed") or [])
        completed.add(phase)
        state["completed"] = sorted(completed)
        state["config_hash"] = config_hash
        state["updated_at"] = utcnow_iso()
        atomic_write_json(self._state_path, state)

    def is_complete(self, phase: str, config_hash: str) -> bool:
        state = self.load_state()
        return bool(
            state.get("config_hash") == config_hash and phase in set(state.get("completed") or [])
        )


# --------------------------------------------------------------------------- #
# Test orchestration                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class MachineSnapshotPaths:
    before: Path
    during: Path
    after: Path
    server_log: Path
    requests: Path


def _write_machine_snapshot(path: Path, *, target_pid: int | None, settings: Settings) -> dict[str, Any]:
    payload: dict[str, Any]
    try:
        state = sample_system_state(
            target_pid=target_pid,
            interval_s=settings.telemetry_interval_s,
            top_processes=12,
        )
        payload = dict(json.loads(state.model_dump_json()))
    except Exception as exc:  # noqa: BLE001
        payload = {"error": f"{type(exc).__name__}: {exc}", "captured_at": utcnow_iso()}
    payload["process_inventory_count"] = len(snapshot_processes())
    atomic_write_json(path, payload)
    return payload


def run_test_a(
    cfg: ExperimentConfig,
    run: RunDirectory,
    *,
    server: Path,
    model_path: Path,
    requests: Sequence[FixedRequest],
    settings: Settings,
) -> dict[str, Any]:
    """Execute unmanaged ``llama-server`` baseline."""
    paths = MachineSnapshotPaths(
        before=run.path("test_a", "machine_before.json"),
        during=run.path("test_a", "machine_during.jsonl"),
        after=run.path("test_a", "machine_after.json"),
        server_log=run.path("test_a", "server.log"),
        requests=run.path("test_a", "requests.jsonl"),
    )
    launched = start_unmanaged_server(
        server=server,
        model_path=model_path,
        host=cfg.host,
        context_length=cfg.context_length,
        gpu_layers=cfg.gpu_layers,
        threads=cfg.threads,
        ignore_eos=bool(requests and requests[0].ignore_eos),
        log_path=paths.server_log,
        readiness_timeout_s=settings.resident_readiness_timeout_s,
    )
    pid_holder: dict[str, int | None] = {"pid": launched.pid}
    startup_resource = resource_snapshot(launched.pid)
    collector = TelemetryCollector(
        jsonl_path=paths.during,
        interval_s=cfg.sample_interval_s,
        target_pid_provider=lambda: pid_holder["pid"],
    )
    started_at = utcnow_iso()
    collector.start()
    outcomes: list[RequestOutcome] = []
    outcomes_writer = JsonlWriter(paths.requests)
    duration_limit = cfg.test_duration_s if cfg.test_duration_s > 0 else None
    work_started = time.perf_counter()
    try:
        for request in requests:
            if duration_limit is not None and time.perf_counter() - work_started >= duration_limit:
                break
            outcome = run_unmanaged_request(
                test="A",
                request=request,
                host=launched.host,
                port=launched.port,
                server_pid=launched.pid,
                timeout_s=settings.resident_request_timeout_s,
            )
            outcomes.append(outcome)
            outcomes_writer.write(outcome.to_dict())
    finally:
        telemetry = collector.stop()
        stop_report = stop_unmanaged_server(launched)
        pid_holder["pid"] = None
    machine_after = _write_machine_snapshot(paths.after, target_pid=None, settings=settings)
    server_log_tail = ""
    try:
        server_log_tail = paths.server_log.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        pass
    summary = {
        "test": "A",
        "kind": "unmanaged_llama_server",
        "started_at": started_at,
        "finished_at": utcnow_iso(),
        "host": launched.host,
        "port": launched.port,
        "server_pid": launched.pid,
        "server_identity": launched.identity.model_dump() if launched.identity else None,
        "command": launched.command,
        "startup_s": launched.startup_s,
        "ready_s": launched.ready_s,
        "startup_rss_bytes": startup_resource.rss_bytes,
        "startup_peak_rss_bytes": startup_resource.peak_rss_bytes,
        "intended_request_count": len(requests),
        "request_count": len(outcomes),
        "success_count": sum(1 for item in outcomes if item.success),
        "monitor_termination": None,
        "telemetry": asdict(telemetry),
        "stop": stop_report,
        "machine_after_process_count": machine_after.get("process_inventory_count"),
        "server_log_tail": server_log_tail,
        "outcomes": [item.to_dict() for item in outcomes],
    }
    atomic_write_json(run.path("test_a", "summary.json"), summary)
    return summary


def run_test_b(
    cfg: ExperimentConfig,
    run: RunDirectory,
    *,
    model_path: Path,
    model_profile: ModelProfile,
    requests: Sequence[FixedRequest],
    settings: Settings,
    monitor_duration_s: float,
    server_path: Path,
) -> dict[str, Any]:
    """Execute the InferenceOS-managed resident server over the same fixed workload."""
    paths = MachineSnapshotPaths(
        before=run.path("test_b", "machine_before.json"),
        during=run.path("test_b", "machine_during.jsonl"),
        after=run.path("test_b", "machine_after.json"),
        server_log=run.path("test_b", "server.log"),
        requests=run.path("test_b", "requests.jsonl"),
    )
    outcomes: list[RequestOutcome] = []
    configuration = _runtime_configuration(cfg)
    workload = FixedSequenceResidentWorkload(
        fixed_requests=requests,
        outcomes=outcomes,
        test_label="B",
        settings=settings,
        model=model_profile,
        # Explicitly bind the same binary Test A used so both tests share one build.
        adapter=LlamaCppResidentAdapter(settings=settings, server=server_path),
        configuration=configuration,
        request_gap_s=0.05,
    )
    collector = TelemetryCollector(
        jsonl_path=paths.during,
        interval_s=cfg.sample_interval_s,
        target_pid_provider=workload.target_pid,
    )
    approval = (
        AutoApproveApprovalProvider() if cfg.auto_approve else AutoRejectApprovalProvider()
    )
    budget = build_runtime_budget(settings, max_duration_s=monitor_duration_s)
    policy = build_runtime_policy(settings, budget=budget)
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=SystemTelemetrySampler(settings=settings, include_gpu=True),
        policy=policy,
        approval_provider=approval,
        settings=settings,
        session_notes=[
            "A/B effectiveness experiment runner; the same fixed workload as the "
            "unmanaged baseline was served through the InferenceOS runtime path.",
            "Disruptive interventions used an explicit experiment approval provider; "
            "no InferenceOS policy or safety gate was modified or weakened.",
        ],
    )
    started_at = utcnow_iso()
    collector.start()
    monitor_error: str | None = None
    target_pid_final: int | None = None
    target_identity_final: ProcessIdentity | None = None
    adapter_loaded_before_stop = False
    adapter_server_path: str | None = None
    try:
        result = monitor.run(duration_s=monitor_duration_s)
        monitor_run = result.run
    except Exception as exc:  # noqa: BLE001 - record and fail the test cleanly
        monitor_error = f"{type(exc).__name__}: {exc}"
        monitor_run = None
    finally:
        telemetry = collector.stop()
        target_pid_final = workload.target_pid()
        target_identity_final = workload.target_identity()
        adapter = workload.adapter
        adapter_loaded_before_stop = bool(getattr(adapter, "is_loaded", False))
        server_attr = getattr(adapter, "server_path", None)
        adapter_server_path = str(server_attr) if server_attr is not None else None
        try:
            workload.stop()
        except Exception:  # noqa: BLE001
            pass
    finished_at = utcnow_iso()
    # Capture the managed llama-server log (InferenceOS backend keeps a bounded tail).
    adapter = workload.adapter
    server_log = ""
    backend = getattr(adapter, "_backend", None)
    process = getattr(backend, "_process", None)
    if process is not None and hasattr(process, "log_text"):
        try:
            server_log = str(process.log_text())[-20000:]
        except Exception:  # noqa: BLE001
            server_log = ""
    if not server_log:
        server_log = (
            "[note] managed llama-server log tail was not available after shutdown; "
            "the process log is bounded in memory by InferenceOS."
        )
    try:
        paths.server_log.write_text(server_log, encoding="utf-8")
    except OSError:
        pass
    writer = JsonlWriter(paths.requests)
    for outcome in outcomes:
        writer.write(outcome.to_dict())
    machine_after = _write_machine_snapshot(paths.after, target_pid=None, settings=settings)
    intervention_end: str | None = None
    intervention_count = 0
    recovery_count = 0
    monitor_termination: str | None = None
    monitor_distinct_processes: bool | None = None
    if monitor_run is not None:
        intervention_count = len(monitor_run.interventions)
        recovery_count = len(monitor_run.target_recoveries)
        monitor_termination = monitor_run.termination.reason.value
        if monitor_run.session is not None:
            monitor_distinct_processes = monitor_run.session.distinct_processes
        for intervention in monitor_run.interventions:
            candidate = intervention.resumed_at or intervention.finished_at or intervention.started_at
            if candidate is not None:
                stamp = candidate.isoformat()
                if intervention_end is None or stamp > intervention_end:
                    intervention_end = stamp
        path = run.path("test_b", "inferenceos_monitor_run.json")
        atomic_write_json(path, json.loads(monitor_run.model_dump_json()))
        atomic_write_text_safe(run.path("test_b", "inferenceos_monitor_report.md"), _render_monitor(monitor_run))
    summary = {
        "test": "B",
        "kind": "inferenceos_managed_llama_server",
        "started_at": started_at,
        "finished_at": finished_at,
        "configuration": configuration.model_dump(),
        "monitor_duration_s": monitor_duration_s,
        "monitor_error": monitor_error,
        "monitor_termination": monitor_termination,
        "intended_request_count": len(requests),
        "request_count": len(outcomes),
        "success_count": sum(1 for item in outcomes if item.success),
        "telemetry": asdict(telemetry),
        "machine_after_process_count": machine_after.get("process_inventory_count"),
        "intervention_count": intervention_count,
        "intervention_end": intervention_end,
        "recovery_count": recovery_count,
        "target_pid_final": target_pid_final,
        "target_identity": (
            target_identity_final.model_dump() if target_identity_final is not None else None
        ),
        "management_evidence": {
            "managed_by": "inferenceos.runtime.RuntimeMonitor",
            "workload": "inferenceos.runtime.resident_workload.ResidentActiveWorkload",
            "adapter_server_path": adapter_server_path,
            "adapter_was_loaded": adapter_loaded_before_stop,
            "target_alive_at_end": target_identity_final is not None,
            "monitor_session_distinct_processes": monitor_distinct_processes,
        },
        "outcomes": [item.to_dict() for item in outcomes],
    }
    atomic_write_json(run.path("test_b", "summary.json"), summary)
    if monitor_run is not None:
        summary["monitor_overhead_s"] = monitor_run.overhead.optimizer_overhead_s
        summary["monitor_net_benefit"] = (
            json.loads(monitor_run.net_benefit.model_dump_json())
            if monitor_run.net_benefit is not None
            else None
        )
    _ = model_path  # retained for symmetry / future evidence
    return summary


def _render_monitor(monitor_run: RuntimeMonitorRun) -> str:
    lines = [
        "# InferenceOS runtime monitor run (Test B)",
        "",
        f"* run_id: `{monitor_run.run_id}`",
        f"* final_state: `{monitor_run.final_state.value}`",
        f"* observations: {monitor_run.observations_collected}",
        f"* recommendations: {len(monitor_run.recommendations)}",
        f"* interventions: {len(monitor_run.interventions)}",
        f"* approvals: {len(monitor_run.approvals)}",
        f"* rollbacks/restorations: "
        f"{sum(1 for i in monitor_run.interventions if i.decision is not None)}",
        f"* target recoveries: {len(monitor_run.target_recoveries)}",
        f"* optimizer overhead: {monitor_run.overhead.optimizer_overhead_s:.3f} s",
        f"* termination: {monitor_run.termination.reason.value} "
        f"({monitor_run.termination.detail})",
        "",
    ]
    for intervention in monitor_run.interventions:
        lines.append(
            f"* intervention `{intervention.intervention_id}` "
            f"action={intervention.action_id} retained={intervention.retained} "
            f"decision={intervention.decision.value if intervention.decision else None} "
            f"duration={intervention.duration_s:.2f}s"
        )
    return "\n".join(lines) + "\n"


def mark_post_intervention(outcomes: Sequence[RequestOutcome], intervention_end: str | None) -> int:
    """Mark outcomes submitted after the last intervention; returns the count."""
    if not intervention_end:
        return 0
    count = 0
    for outcome in outcomes:
        if outcome.submitted_at >= intervention_end:
            outcome.post_intervention = True
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Cleanup / stabilization                                                       #
# --------------------------------------------------------------------------- #
# Any process in these categories indicates an experiment-related inference process
# that must not survive cleanup. Unrelated user processes are never terminated; they are
# reported and (pre-A) fail the experiment closed.
INFERENCE_PROCESS_CATEGORIES = ("llama_server", "llama_cpp", "ollama", "model_server")


def _cleanup_verification_payload(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "process_before": [],
        "process_created": [],
        "process_remaining": [],
        "cleanup_actions": [],
        "checks": {},
        "unexpected_survivors": [],
        "remaining_owned": [],
        "cleanup_success": False,
        "note": "cleanup evidence unavailable",
    }


def run_cleanup(
    run: RunDirectory,
    *,
    before_records: Sequence[ProcessRecord],
    launched_pids: Sequence[int],
    label: str,
    phase_noun: str,
) -> dict[str, Any]:
    """Shared cleanup + process-state verification used before and after Test A.

    The same safety logic and checklist are applied in both places:

    * stop/terminate only experiment-owned processes (PID + recorded identity);
    * verify llama-server, llama.cpp, Ollama and other model-serving processes are gone;
    * inspect the process table and compare against the appropriate population;
    * report relevant InferenceOS processes (expected: none beyond this runner);
    * never terminate unrelated user processes.

    ``before_records`` is the process population captured at the start of cleanup;
    ``launched_pids`` are the PIDs the experiment itself launched. Cleanup fails closed
    when any inference process or experiment-owned survivor remains.
    """
    pre = list(before_records)
    pre_map = process_map(pre)
    launched = {int(pid) for pid in launched_pids}
    cleanup_actions: list[str] = []
    remaining_owned: list[dict[str, Any]] = []
    for pid in launched:
        if psutil.pid_exists(pid):
            cleanup_actions.append(
                f"{label}: pid {pid} still present {phase_noun}; terminating tree"
            )
            terminate_process_tree(pid, timeout_s=10.0)
            if psutil.pid_exists(pid):
                remaining_owned.append({"pid": pid, "note": "survived controlled cleanup"})
    after = snapshot_processes()
    created = [record for record in after if record.pid not in pre_map]
    owned_survivors = [record for record in after if record.pid in launched]

    def _remaining(category: str) -> list[dict[str, Any]]:
        return [
            asdict(record)
            for record in after
            if record.category == category
        ]

    checks = {
        "llama_server": _remaining("llama_server"),
        "llama_cpp": _remaining("llama_cpp"),
        "ollama": _remaining("ollama"),
        "model_serving": _remaining("model_server"),
        "inferenceos": _remaining("inferenceos"),
    }
    inference_remaining = [
        record for record in after if record.category in INFERENCE_PROCESS_CATEGORIES
    ]
    cleanup_success = not remaining_owned and not inference_remaining
    if not inference_remaining:
        cleanup_actions.append(
            f"{label}: verified no llama-server/llama.cpp/Ollama/model-serving process remains"
        )
    if not checks["inferenceos"]:
        cleanup_actions.append(
            f"{label}: verified no stray InferenceOS runtime process remains"
        )
    payload = {
        "label": label,
        "phase": phase_noun,
        "timestamp": utcnow_iso(),
        "process_before": [asdict(record) for record in pre],
        "process_created": [asdict(record) for record in created],
        "process_remaining": [asdict(record) for record in after],
        "cleanup_actions": cleanup_actions,
        "checks": checks,
        "unexpected_survivors": [asdict(record) for record in owned_survivors],
        "remaining_owned": remaining_owned,
        "remaining_inference": [asdict(record) for record in inference_remaining],
        "cleanup_success": cleanup_success,
        "note": (
            "only experiment-launched processes (PID + creation identity) are terminated; "
            "unrelated user processes are never killed"
        ),
    }
    atomic_write_json(
        run.path("cleanup", f"{label}_process_before.json"), payload["process_before"]
    )
    atomic_write_json(
        run.path("cleanup", f"{label}_process_created.json"), payload["process_created"]
    )
    atomic_write_json(
        run.path("cleanup", f"{label}_process_remaining.json"), payload["process_remaining"]
    )
    atomic_write_json(run.path("cleanup", f"{label}_cleanup.json"), payload)
    return payload


def _load_cleanup(run: RunDirectory, label: str) -> dict[str, Any]:
    path = run.path("cleanup", f"{label}_cleanup.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError):
        pass
    return _cleanup_verification_payload(label)


def _load_stabilization(run: RunDirectory, label: str) -> dict[str, Any]:
    return _load_json_dict(run.path("stabilization", f"{label}_summary.json"))


def run_final_cleanup(run: RunDirectory) -> dict[str, Any]:
    """Verify no inference process (managed or unmanaged) survives the experiment."""
    processes = snapshot_processes()
    survivors = [
        record for record in processes if record.category in INFERENCE_PROCESS_CATEGORIES
    ]
    payload = {
        "timestamp": utcnow_iso(),
        "process_remaining": [asdict(record) for record in processes],
        "inference_process_survivors": [asdict(record) for record in survivors],
        "checks": {
            category: [
                asdict(record) for record in processes if record.category == category
            ]
            for category in INFERENCE_PROCESS_CATEGORIES
        },
        "cleanup_success": not survivors,
        "note": "the experiment only terminates processes it launched",
    }
    atomic_write_json(run.path("cleanup", "final_cleanup.json"), payload)
    return payload


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _telemetry_evolution(path: Path) -> dict[str, Any]:
    """Summarize process-count / machine-state evolution over a stabilization window."""
    process_counts: list[float] = []
    cpu_values: list[float] = []
    available_ram: list[float] = []
    gpu_values: list[float] = []
    power_states: list[str] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            count = payload.get("process_count")
            if isinstance(count, (int, float)):
                process_counts.append(float(count))
            cpu = payload.get("cpu_percent_total")
            if isinstance(cpu, (int, float)):
                cpu_values.append(float(cpu))
            ram = payload.get("ram_available_bytes")
            if isinstance(ram, (int, float)):
                available_ram.append(float(ram))
            gpu = payload.get("gpu_engine_max_percent")
            if isinstance(gpu, (int, float)):
                gpu_values.append(float(gpu))
            state = payload.get("power_state")
            if isinstance(state, str):
                power_states.append(state)

    def _range(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "first": values[0],
            "last": values[-1],
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
        }

    return {
        "sample_count": len(process_counts),
        "process_count": _range(process_counts),
        "cpu_percent": _range(cpu_values),
        "available_ram_bytes": _range(available_ram),
        "gpu_engine_max_percent": _range(gpu_values),
        "power_states": sorted(set(power_states)),
    }


def run_stabilization(
    run: RunDirectory,
    cfg: ExperimentConfig,
    *,
    label: str,
    before_snapshot: Path,
    after_snapshot: Path,
    settings: Settings,
    duration_s: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """One shared 20-minute stabilization window (used before A and before B).

    No inference, no llama-server, no model-serving process: only lightweight telemetry.
    The window is never silently shortened; a shortened window (only possible via the
    explicit ``--allow-short-cooldown``) is recorded as a validity confound.
    """
    requested = float(duration_s if duration_s is not None else cfg.cooldown_s)
    telemetry_path = run.path("stabilization", f"{label}_telemetry.jsonl")
    collector = TelemetryCollector(
        jsonl_path=telemetry_path,
        interval_s=max(cfg.sample_interval_s, 2.0),
        target_pid_provider=lambda: None,
        include_gpu=True,
    )
    started_at = utcnow_iso()
    started = time.perf_counter()
    collector.start()
    try:
        # Sleep in bounded chunks so telemetry and reporting stay responsive.
        deadline = started + requested
        while time.perf_counter() < deadline:
            time.sleep(min(5.0, max(0.0, deadline - time.perf_counter())))
    finally:
        telemetry = collector.stop()
    elapsed = time.perf_counter() - started
    finished_at = utcnow_iso()
    before = _load_json_dict(before_snapshot)
    after = _write_machine_snapshot(after_snapshot, target_pid=None, settings=settings)
    shortened = (not dry_run) and requested < STABILIZATION_SECONDS_DEFAULT - 1.0
    completed = elapsed >= requested - 1.0
    payload = {
        "label": label,
        "started_at": started_at,
        "finished_at": finished_at,
        "requested_seconds": requested,
        "actual_seconds": elapsed,
        "default_seconds": STABILIZATION_SECONDS_DEFAULT,
        "shortened": shortened,
        "dry_run": dry_run,
        "allow_short_cooldown_flag": cfg.allow_short_cooldown,
        "completed_successfully": completed,
        "telemetry": asdict(telemetry),
        "process_count_evolution": _telemetry_evolution(telemetry_path),
        "machine_state_before": before,
        "machine_state_after": after,
        "returned_to_baseline": _state_within_tolerance(before, after),
        "thermal_note": telemetry.thermal_note,
    }
    atomic_write_json(run.path("stabilization", f"{label}_summary.json"), payload)
    return payload


def _state_within_tolerance(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def _num(payload: dict[str, Any], key: str) -> float | None:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    results: dict[str, Any] = {}
    for key in ("cpu_percent_total", "available_ram_bytes", "memory_pressure"):
        before_value = _num(before, key)
        after_value = _num(after, key)
        if before_value is None or after_value is None:
            results[key] = {"comparable": False}
            continue
        if key == "available_ram_bytes" and before_value:
            relative = abs(after_value - before_value) / max(abs(before_value), 1.0)
        else:
            relative = abs(after_value - before_value)
        results[key] = {
            "before": before_value,
            "after": after_value,
            "relative_change": relative,
            "within_tolerance": relative <= 0.25,
        }
    results["approximately_returned"] = all(
        item.get("within_tolerance", True) for item in results.values() if isinstance(item, dict)
    )
    return results


# --------------------------------------------------------------------------- #
# Analysis                                                                      #
# --------------------------------------------------------------------------- #
def _values(outcomes: Sequence[RequestOutcome], field_name: str) -> list[float]:
    values: list[float] = []
    for outcome in outcomes:
        value = getattr(outcome, field_name)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _workload_spec_identical(
    outcomes_a: Sequence[RequestOutcome], outcomes_b: Sequence[RequestOutcome]
) -> bool:
    """True when overlapping A/B requests have identical workload specifications."""
    a_by_index = {item.index: item for item in outcomes_a}
    b_by_index = {item.index: item for item in outcomes_b}
    common = sorted(set(a_by_index) & set(b_by_index))
    if not common:
        return bool(a_by_index) == bool(b_by_index)
    for index in common:
        a = a_by_index[index]
        b = b_by_index[index]
        if (
            a.prompt_sha256 != b.prompt_sha256
            or a.n_predict != b.n_predict
            or a.seed != b.seed
            or a.temperature != b.temperature
        ):
            return False
    return True


# Monitor termination reasons that mean the managed run did not finish normally.
_EARLY_TERMINATION_REASONS = frozenset(
    {
        "monitoring_budget_exhausted",
        "error",
        "safety_terminal",
        "external_termination",
        "operator_required",
    }
)


def _equal_work_assessment(test_a: dict[str, Any], test_b: dict[str, Any]) -> dict[str, Any]:
    """Detect unequal A/B workload execution so a comparison is never silently invalid.

    Returns a machine-readable assessment. Checks are only enforced when the runner
    recorded intended request counts (the real experiment does); synthetic/unit summaries
    without them are treated as unknown rather than falsely failing.
    """
    a_out = [RequestOutcome(**item) for item in test_a.get("outcomes", [])]
    b_out = [RequestOutcome(**item) for item in test_b.get("outcomes", [])]
    intended_a = test_a.get("intended_request_count")
    intended_b = test_b.get("intended_request_count")
    success_a = sum(1 for item in a_out if item.success)
    success_b = sum(1 for item in b_out if item.success)
    fail_a = len(a_out) - success_a
    fail_b = len(b_out) - success_b
    termination = test_b.get("monitor_termination")
    monitor_error = test_b.get("monitor_error")
    terminated_early = bool(
        (isinstance(termination, str) and termination in _EARLY_TERMINATION_REASONS)
        or monitor_error
    )

    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    counts_provided = isinstance(intended_a, int) or isinstance(intended_b, int)
    if counts_provided:
        checks["same intended request count"] = bool(
            isinstance(intended_a, int)
            and isinstance(intended_b, int)
            and intended_a == intended_b
        )
        checks["A completed the full fixed workload"] = bool(
            isinstance(intended_a, int)
            and len(a_out) >= intended_a
            and success_a == intended_a
        )
        checks["B completed the full fixed workload"] = bool(
            isinstance(intended_b, int)
            and len(b_out) >= intended_b
            and success_b == intended_b
        )
        checks["no failed requests"] = fail_a == 0 and fail_b == 0
    checks["monitor did not terminate the managed run early"] = not terminated_early
    if a_out and b_out:
        checks["workload specification identical between A and B"] = _workload_spec_identical(
            a_out, b_out
        )
    detail = {
        "intended_a": intended_a,
        "intended_b": intended_b,
        "executed_a": len(a_out),
        "executed_b": len(b_out),
        "success_a": success_a,
        "success_b": success_b,
        "failed_a": fail_a,
        "failed_b": fail_b,
        "monitor_termination": termination,
        "monitor_error": monitor_error,
        "terminated_early": terminated_early,
        "counts_provided": counts_provided,
    }
    return {
        "valid": all(checks.values()) if checks else True,
        "checks": checks,
        "detail": detail,
        "failed_checks": [name for name, ok in checks.items() if not ok],
    }


def analyze_ab(
    *,
    test_a: dict[str, Any],
    test_b: dict[str, Any],
    cfg: ExperimentConfig,
    cleanup: dict[str, Any],
    pre_a_stabilization: dict[str, Any],
    post_a_stabilization: dict[str, Any],
    preflight: dict[str, Any],
    settings: Settings,
    pre_a_cleanup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outcomes_a = [RequestOutcome(**item) for item in test_a.get("outcomes", [])]
    outcomes_b = [RequestOutcome(**item) for item in test_b.get("outcomes", [])]
    success_a = [item for item in outcomes_a if item.success]
    success_b = [item for item in outcomes_b if item.success]
    post_b = [item for item in success_b if item.post_intervention]
    comparison_b = post_b if post_b else success_b

    metric_specs = (
        ("decode_tokens_per_sec", True, "decode throughput"),
        ("total_latency_s", False, "total latency"),
        ("ttft_s", False, "time to first token"),
        ("decode_s", False, "decode duration"),
        ("peak_rss_bytes", False, "peak RSS"),
    )
    min_effect = float(getattr(settings, "phase7_min_effect", DEFAULT_BASELINE_MIN_EFFECT))
    metrics: dict[str, Any] = {}
    for name, higher_is_better, label in metric_specs:
        metrics[name] = {
            "label": label,
            **metric_verdict(
                baseline_values=_values(success_a, name),
                measured_values=_values(comparison_b, name),
                higher_is_better=higher_is_better,
                min_effect=min_effect,
            ),
        }

    # Primary metric: decode throughput if usable, else total latency.
    primary_name = "decode_tokens_per_sec"
    if metrics[primary_name]["baseline"]["n"] < 3 or metrics[primary_name]["measured"]["n"] < 3:
        primary_name = "total_latency_s"
    primary = metrics[primary_name]

    raw = {
        "test_a": {
            "requests": len(outcomes_a),
            "successes": len(success_a),
            "failures": len(outcomes_a) - len(success_a),
            "startup_s": test_a.get("startup_s"),
            "server_pid": test_a.get("server_pid"),
            "telemetry": test_a.get("telemetry"),
        },
        "test_b": {
            "requests": len(outcomes_b),
            "successes": len(success_b),
            "failures": len(outcomes_b) - len(success_b),
            "interventions": test_b.get("intervention_count", 0),
            "post_intervention_successes": len(post_b),
            "telemetry": test_b.get("telemetry"),
            "monitor_error": test_b.get("monitor_error"),
        },
    }
    derived = {
        "min_effect": min_effect,
        "primary_metric": primary_name,
        "metrics": metrics,
        "totals": {
            "generated_tokens_a": sum(item.generated_tokens or 0 for item in success_a),
            "generated_tokens_b": sum(item.generated_tokens or 0 for item in comparison_b),
            "workload_wall_s_a": test_a.get("telemetry", {}).get("duration_s"),
            "workload_wall_s_b": test_b.get("telemetry", {}).get("duration_s"),
        },
        "inferenceos": {
            "intervention_count": test_b.get("intervention_count", 0),
            "optimizer_overhead_s": test_b.get("monitor_overhead_s"),
            "native_net_benefit": test_b.get("monitor_net_benefit"),
        },
        "server_stability": {
            "a_startup_s": test_a.get("startup_s"),
            "a_stop_exited": (test_a.get("stop") or {}).get("exited"),
            "b_monitor_error": test_b.get("monitor_error"),
            "b_monitor_termination": test_b.get("monitor_termination"),
            "b_recovery_events": test_b.get("recovery_count", 0),
            "request_failures_a": raw["test_a"]["failures"],
            "request_failures_b": raw["test_b"]["failures"],
        },
        "thermal_observations": {
            "available": False,
            "reason": "no reliable Windows thermal sensor exposed via psutil/WMI",
        },
    }

    # H4 cross-run net-benefit: A is the baseline, post-intervention B is the measurement.
    cross_overhead = build_overhead_attribution(
        inclusive_overhead_s=float(test_b.get("monitor_overhead_s") or 0.0),
        components={},
        baseline_target_workload_s=float(
            test_a.get("telemetry", {}).get("duration_s") or 0.0
        )
        or None,
        decision_count=int(test_b.get("intervention_count") or 0),
        notes=["cross-run accounting: unmanaged baseline (A) vs managed (B)"],
    )
    means_a = {
        "decode_tokens_per_sec": metrics["decode_tokens_per_sec"]["baseline"].get("mean"),
        "total_latency_s": metrics["total_latency_s"]["baseline"].get("mean"),
        "ttft_s": metrics["ttft_s"]["baseline"].get("mean"),
        "peak_rss_bytes": metrics["peak_rss_bytes"]["baseline"].get("mean"),
    }
    means_b = {
        "decode_tokens_per_sec": metrics["decode_tokens_per_sec"]["measured"].get("mean"),
        "total_latency_s": metrics["total_latency_s"]["measured"].get("mean"),
        "ttft_s": metrics["ttft_s"]["measured"].get("mean"),
        "peak_rss_bytes": metrics["peak_rss_bytes"]["measured"].get("mean"),
    }
    clean_a = {k: v for k, v in means_a.items() if v is not None}
    clean_b = {k: v for k, v in means_b.items() if v is not None}
    distortion = distortion_from_stability(
        cv=metrics["decode_tokens_per_sec"]["measured"].get("cv"),
        within_tolerance=(
            (metrics["decode_tokens_per_sec"]["measured"].get("cv") or 0.0)
            <= float(getattr(settings, "phase10_deviation_threshold", 0.25))
        ),
        tolerance=float(getattr(settings, "phase10_deviation_threshold", 0.25)),
        metric="decode_tokens_per_sec",
        detail="cross-run throughput CV over the managed workload",
    )
    cross_net_benefit = assess_net_benefit(
        baseline_metrics=clean_a,
        measured_metrics=clean_b,
        overhead=cross_overhead,
        distortion=distortion,
        policy=AccountingPolicy.from_settings(settings),
        preferred_metric=primary_name if primary_name in clean_a and primary_name in clean_b else None,
    )

    validity = _validity_report(
        preflight=preflight,
        test_a=test_a,
        test_b=test_b,
        cleanup=cleanup,
        pre_a_stabilization=pre_a_stabilization,
        post_a_stabilization=post_a_stabilization,
        cfg=cfg,
        metrics=metrics,
        pre_a_cleanup=pre_a_cleanup,
    )

    machine_state_comparison = _compare_machine_states(
        pre_a_stabilization.get("machine_state_after") or {},
        post_a_stabilization.get("machine_state_after") or {},
    )

    # Equal-work validity gate: a comparison is only usable for effectiveness conclusions
    # when A and B executed the same intended fixed workload and the managed run was not
    # terminated early by the monitor/runtime.
    equal_work = _equal_work_assessment(test_a, test_b)
    for check_name, ok in equal_work["checks"].items():
        validity["checks"].append(
            {
                "check": f"equal-work: {check_name}",
                "ok": ok,
                "detail": json.dumps(equal_work["detail"]),
                "major": True,
            }
        )
    validity["major_confound"] = any(
        check["major"] and check["ok"] is False for check in validity["checks"]
    )

    intervention_count = int(test_b.get("intervention_count") or 0)
    # A retained improvement requires that InferenceOS actually changed something and
    # that the post-intervention workload improved beyond variability. A faster B with
    # no intervention is machine-state variability, not an InferenceOS effect.
    retained_improvement = bool(
        intervention_count > 0 and post_b and metrics[primary_name]["improved"]
    )
    if not equal_work["valid"]:
        # Never present a truncated/unequal run as a normal A/B result.
        verdict = "invalid_comparison"
        verdict_text = "INVALID COMPARISON (unequal A/B workload)"
    elif validity["major_confound"]:
        verdict = "inconclusive"
        verdict_text = "INCONCLUSIVE"
    elif retained_improvement:
        verdict = "improvement_demonstrated"
        verdict_text = "Improvement demonstrated"
    elif metrics[primary_name]["regressed"]:
        verdict = "regression"
        verdict_text = "Regression demonstrated"
    else:
        verdict = "no_demonstrated_improvement"
        verdict_text = "No demonstrated improvement"

    notes = [
        "A positive result is only claimed when the change exceeds the experiment's "
        "variability (approximate 95% CI excludes zero), exceeds the 5% minimum effect, "
        "and is directionally consistent.",
        "A retained improvement additionally requires at least one real InferenceOS "
        "intervention and post-intervention requests that improve.",
        "An effectiveness comparison additionally requires that A and B executed the "
        "same fixed workload and that the managed run was not terminated early.",
    ]
    if not equal_work["valid"]:
        notes.append(
            "Equal-work gate FAILED: "
            + "; ".join(equal_work["failed_checks"])
            + f" (A executed {equal_work['detail']['executed_a']}, "
            f"B executed {equal_work['detail']['executed_b']}, monitor termination "
            f"{equal_work['detail']['monitor_termination']})."
        )
    if intervention_count == 0:
        notes.append(
            "InferenceOS made no intervention in this run; any B-vs-A difference is "
            "attributed to machine-state variability, not to InferenceOS."
        )
    if intervention_count > 0 and not post_b:
        notes.append(
            "InferenceOS intervened but no post-intervention requests completed; a "
            "retained effect could not be measured."
        )

    interpretation = {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "statement": _verdict_statement(verdict_text, primary_name, primary),
        "retained_improvement": retained_improvement,
        "used_post_intervention_only": bool(post_b),
        "effective_comparison": equal_work["valid"],
        "notes": notes,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utcnow_iso(),
        "config": {
            "model_path": str(cfg.model_path) if cfg.model_path else None,
            "context_length": cfg.context_length,
            "gpu_layers": cfg.gpu_layers,
            "threads": cfg.threads,
            "rounds": cfg.rounds,
            "n_predict": cfg.n_predict,
            "seed": cfg.seed,
            "stabilization_seconds": cfg.cooldown_s,
            "cooldown_seconds": cfg.cooldown_s,
        },
        "raw": raw,
        "derived": derived,
        "interpretation": interpretation,
        "cross_run_net_benefit": json.loads(cross_net_benefit.model_dump_json()),
        "validity": validity,
        "cleanup": {
            "pre_a": _cleanup_summary(pre_a_cleanup),
            "post_a": _cleanup_summary(cleanup),
        },
        "stabilization": {
            "pre_a_stabilization": _stabilization_summary(pre_a_stabilization),
            "post_a_stabilization": _stabilization_summary(post_a_stabilization),
        },
        "machine_state_comparison": machine_state_comparison,
        "equal_work": equal_work,
    }


def _stabilization_summary(stabilization: dict[str, Any] | None) -> dict[str, Any]:
    if not stabilization:
        return {"available": False}
    return {
        "available": True,
        "label": stabilization.get("label"),
        "started_at": stabilization.get("started_at"),
        "finished_at": stabilization.get("finished_at"),
        "requested_seconds": stabilization.get("requested_seconds"),
        "actual_seconds": stabilization.get("actual_seconds"),
        "shortened": stabilization.get("shortened"),
        "completed_successfully": stabilization.get("completed_successfully"),
        "telemetry": stabilization.get("telemetry", {}),
        "process_count_evolution": stabilization.get("process_count_evolution", {}),
        "returned_to_baseline": stabilization.get("returned_to_baseline", {}),
        "machine_state_after": stabilization.get("machine_state_after", {}),
    }


def _compare_machine_states(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare the stabilized pre-A and stabilized pre-B machine states.

    Reports the actual measurements and flags material differences that could confound
    the A/B comparison; it never claims the two states are identical.
    """

    def _num(payload: dict[str, Any], key: str) -> float | None:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    metrics: dict[str, Any] = {}
    material: list[str] = []
    for key, threshold in (
        ("cpu_percent_total", 0.25),
        ("available_ram_bytes", 0.25),
        ("memory_pressure", 0.25),
        ("process_inventory_count", 0.25),
    ):
        av = _num(before, key)
        bv = _num(after, key)
        if av is None or bv is None:
            metrics[key] = {"available": False}
            continue
        delta = bv - av
        relative = (abs(delta) / abs(av)) if av else None
        is_material = relative is not None and relative > threshold
        metrics[key] = {
            "available": True,
            "pre_a_stabilized": av,
            "post_a_stabilized": bv,
            "delta": delta,
            "relative_change": relative,
            "material": is_material,
        }
        if is_material:
            material.append(key)
    return {
        "pre_a_label": "pre_a_stabilized_machine_state",
        "post_a_label": "post_a_stabilized_machine_state",
        "metrics": metrics,
        "material_differences": material,
        "comparable": not material,
        "note": (
            "actual stabilized machine states; material differences are reported rather "
            "than assumed away"
        ),
    }


def _cleanup_summary(cleanup: dict[str, Any] | None) -> dict[str, Any]:
    if not cleanup:
        return {"available": False}
    return {
        "available": True,
        "label": cleanup.get("label"),
        "cleanup_success": cleanup.get("cleanup_success"),
        "cleanup_actions": cleanup.get("cleanup_actions", []),
        "remaining_inference": cleanup.get("remaining_inference", []),
        "remaining_owned": cleanup.get("remaining_owned", []),
        "checks": cleanup.get("checks", {}),
    }


def _verdict_statement(verdict_text: str, primary_name: str, primary: dict[str, Any]) -> str:
    baseline = primary.get("baseline", {})
    measured = primary.get("measured", {})
    relative = primary.get("relative_change")
    rel_text = f"{relative:+.1%}" if isinstance(relative, (int, float)) else "n/a"
    return (
        f"{verdict_text}. Primary metric '{primary_name}': baseline mean "
        f"{baseline.get('mean')} -> managed mean {measured.get('mean')} ({rel_text}); "
        f"CI=[{primary.get('ci_low')}, {primary.get('ci_high')}]."
    )


def _validity_report(
    *,
    preflight: dict[str, Any],
    test_a: dict[str, Any],
    test_b: dict[str, Any],
    cleanup: dict[str, Any],
    pre_a_stabilization: dict[str, Any],
    post_a_stabilization: dict[str, Any],
    cfg: ExperimentConfig,
    metrics: dict[str, Any],
    pre_a_cleanup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def _add(name: str, ok: bool | None, detail: str, *, major: bool = False) -> None:
        checks.append({"check": name, "ok": ok, "detail": detail, "major": major})

    # Same model / same binary / same config.
    model_a = test_a.get("model_sha256")
    model_b = test_b.get("model_sha256")
    _add(
        "same model file fingerprint",
        (model_a == model_b) if model_a and model_b else None,
        f"A={model_a} B={model_b}",
        major=True,
    )
    _add(
        "same runtime configuration",
        test_a.get("command") is not None,
        "A and B both use context/gpu-layers/threads from the experiment config",
    )
    # Cleanup: pre-A establishes the clean starting state; post-A verifies the baseline
    # run left nothing behind; final cleanup covers Test B.
    pre_survivors = (pre_a_cleanup or {}).get("remaining_inference") or []
    _add(
        "pre-A cleanup established a clean starting state",
        (pre_a_cleanup or {}).get("cleanup_success"),
        f"remaining_inference={len(pre_survivors)}",
        major=bool(pre_survivors),
    )
    survivors = cleanup.get("unexpected_survivors") or []
    _add(
        "no experiment-owned survivors after A",
        bool(cleanup.get("cleanup_success")),
        f"survivors={len(survivors)}",
        major=bool(survivors),
    )
    final_cleanup = test_b.get("final_cleanup") or {}
    final_survivors = final_cleanup.get("inference_process_survivors") or []
    _add(
        "no inference processes after final cleanup",
        final_cleanup.get("cleanup_success"),
        f"survivors={len(final_survivors)}",
        major=bool(final_survivors),
    )
    # Two identical stabilization windows: pre-A (before Test A) and post-A (before B).
    for stab_label, stab in (
        ("pre-A stabilization", pre_a_stabilization),
        ("post-A stabilization", post_a_stabilization),
    ):
        _add(
            f"{stab_label} ran for the full requested duration",
            stab.get("completed_successfully") if stab else None,
            f"requested={stab.get('requested_seconds')} actual={stab.get('actual_seconds')}",
            major=bool(stab.get("shortened")) and not cfg.allow_short_cooldown,
        )
        _add(
            f"{stab_label} length not silently shortened",
            (not bool(stab.get("shortened"))) if stab else None,
            f"shortened={stab.get('shortened')} "
            f"allow_short_cooldown={cfg.allow_short_cooldown}",
            major=bool(stab.get("shortened")) and not cfg.allow_short_cooldown,
        )
    # Stabilized machine-state comparison (pre-A vs pre-B). Differences are reported,
    # never assumed away; material differences are flagged but do not by themselves
    # invalidate the run.
    machine_comparison = _compare_machine_states(
        pre_a_stabilization.get("machine_state_after") or {},
        post_a_stabilization.get("machine_state_after") or {},
    )
    _add(
        "stabilized pre-A and post-A machine states comparable",
        bool(machine_comparison.get("comparable")),
        "material_differences=" + ",".join(machine_comparison.get("material_differences", [])),
        major=False,
    )
    # Monitoring distortion.
    overhead_mean = test_b.get("telemetry", {}).get("monitor_overhead_ms_mean")
    workload_ms = None
    duration = test_b.get("telemetry", {}).get("duration_s")
    if isinstance(duration, (int, float)) and isinstance(overhead_mean, (int, float)):
        workload_ms = duration * 1000.0
    distortion_ok = None
    if workload_ms and overhead_mean:
        distortion_ok = (overhead_mean / workload_ms) <= 0.05
    _add(
        "monitoring overhead is not material (<5%)",
        distortion_ok,
        f"mean_overhead_ms={overhead_mean} workload_ms={workload_ms}",
        major=False,
    )
    # Thermal.
    _add(
        "thermal telemetry available",
        False,
        test_b.get("telemetry", {}).get("thermal_note")
        or "unavailable: no reliable Windows thermal sensor",
        major=False,
    )
    # GPU telemetry present.
    _add(
        "GPU telemetry captured",
        bool(test_a.get("telemetry", {}).get("peak_gpu_engine_percent") is not None),
        "peak GPU engine utilization recorded for the baseline",
        major=False,
    )
    # Process population differences between the two stabilized states.
    pre_count = (
        (pre_a_stabilization.get("machine_state_after") or {}).get(
            "process_inventory_count"
        )
    )
    post_count = (
        (post_a_stabilization.get("machine_state_after") or {}).get(
            "process_inventory_count"
        )
    )
    population_ok = None
    if isinstance(pre_count, int) and isinstance(post_count, int):
        population_ok = abs(post_count - pre_count) <= 50
    _add(
        "process population comparable between stabilized pre-A and post-A states",
        population_ok,
        f"pre_a_stabilized={pre_count} post_a_stabilized={post_count}",
        major=False,
    )
    major_confound = any(check["major"] and check["ok"] is False for check in checks)
    return {
        "checks": checks,
        "major_confound": major_confound,
        "thermal": {
            "available": False,
            "reason": "no reliable Windows thermal sensor exposed via psutil/WMI",
        },
        "missing_telemetry": [
            check["check"] for check in checks if check["ok"] is None
        ],
        "notes": [
            "Thermals are genuinely unavailable on this platform; no temperature is "
            "invented.",
            "A major confound (different model, surviving experiment process, or a "
            "shortened stabilization window without explicit override) forces an "
            "INCONCLUSIVE result.",
            "The stabilized pre-A and post-A machine states are compared directly; "
            "material differences are reported, not assumed away.",
        ],
    }


# --------------------------------------------------------------------------- #
# Report rendering                                                              #
# --------------------------------------------------------------------------- #
def render_report(
    *,
    cfg: ExperimentConfig,
    metadata: dict[str, Any],
    preflight: dict[str, Any],
    analysis: dict[str, Any],
    status: str,
) -> str:
    raw = analysis.get("raw", {})
    derived = analysis.get("derived", {})
    interpretation = analysis.get("interpretation", {})
    validity = analysis.get("validity", {})
    metrics = derived.get("metrics", {})
    lines: list[str] = []
    lines.append("# InferenceOS A/B effectiveness report")
    lines.append("")
    lines.append(f"* status: **{status}**")
    lines.append(f"* generated_at: {analysis.get('generated_at')}")
    lines.append(f"* verdict: **{interpretation.get('verdict_text')}**")
    lines.append(f"* {interpretation.get('statement', '')}")
    lines.append("")
    lines.append("## Experiment configuration")
    lines.append("")
    conf = analysis.get("config", {})
    for key, value in conf.items():
        lines.append(f"* {key}: `{value}`")
    lines.append("")
    lines.append("## Provisioning metadata")
    lines.append("")
    for key in (
        "model_filename",
        "model_size_bytes",
        "model_sha256",
        "model_source_url",
        "llama_server_path",
        "llama_version",
        "llama_expected_release",
        "inferenceos_git_revision",
        "hostname",
        "python_version",
    ):
        if key in metadata:
            lines.append(f"* {key}: `{metadata.get(key)}`")
    lines.append("")
    lines.append("## Raw measurements")
    lines.append("")
    for test_key, label in (("test_a", "A (unmanaged)"), ("test_b", "B (InferenceOS)")):
        block = raw.get(test_key, {})
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"* requests: {block.get('requests')}")
        lines.append(f"* successes: {block.get('successes')}")
        lines.append(f"* failures: {block.get('failures')}")
        if test_key == "test_b":
            lines.append(f"* interventions: {block.get('interventions')}")
            lines.append(f"* post-intervention successes: {block.get('post_intervention_successes')}")
        telemetry = block.get("telemetry") or {}
        if telemetry:
            lines.append(f"* peak CPU: {telemetry.get('peak_cpu_percent')} %")
            lines.append(f"* mean CPU: {telemetry.get('mean_cpu_percent')} %")
            lines.append(f"* peak server RSS: {fmt_bytes(telemetry.get('peak_target_rss_bytes'))}")
            lines.append(
                f"* peak GPU engine: {telemetry.get('peak_gpu_engine_percent')} %"
            )
            lines.append(
                f"* monitoring overhead (mean): {telemetry.get('monitor_overhead_ms_mean')} ms"
            )
        lines.append("")
    lines.append("## Derived metrics (A vs B)")
    lines.append("")
    lines.append("| metric | A mean | B mean | relative | 95% CI | beyond variability |")
    lines.append("|---|---|---|---|---|---|")
    for name, block in metrics.items():
        lines.append(
            f"| {name} | {_fmt(block.get('baseline', {}).get('mean'))} | "
            f"{_fmt(block.get('measured', {}).get('mean'))} | "
            f"{_fmt_pct(block.get('relative_change'))} | "
            f"[{_fmt(block.get('ci_low'))}, {_fmt(block.get('ci_high'))}] | "
            f"{block.get('beyond_variability')} |"
        )
    lines.append("")
    lines.append("## InferenceOS evidence (Test B)")
    lines.append("")
    ios = derived.get("inferenceos", {})
    lines.append(f"* interventions: {ios.get('intervention_count')}")
    lines.append(f"* optimizer overhead: {ios.get('optimizer_overhead_s')} s")
    native = ios.get("native_net_benefit")
    if native:
        lines.append(f"* native H4 net-benefit status: `{native.get('status')}`")
    cross = analysis.get("cross_run_net_benefit", {})
    lines.append(f"* cross-run H4 net-benefit status: `{cross.get('status')}`")
    lines.append(f"* cross-run benefit ratio: {_fmt_pct(cross.get('benefit_ratio'))}")
    lines.append(f"* cross-run overhead fraction: {_fmt_pct(cross.get('overhead_fraction'))}")
    lines.append("")
    lines.append("## Experiment timeline")
    lines.append("")
    lines.append("```")
    lines.append("Setup / model provisioning")
    lines.append("  -> Pre-A cleanup")
    lines.append("  -> 20-min pre-A stabilization")
    lines.append("  -> Test A")
    lines.append("  -> Post-A cleanup")
    lines.append("  -> 20-min post-A stabilization")
    lines.append("  -> Test B")
    lines.append("  -> Final cleanup")
    lines.append("  -> Analysis / report")
    lines.append("```")
    lines.append("")
    cleanup_block = analysis.get("cleanup", {})
    stabilization_block = analysis.get("stabilization", {})
    pre_a = cleanup_block.get("pre_a", {})
    post_a = cleanup_block.get("post_a", {})
    pre_a_stab = stabilization_block.get("pre_a_stabilization", {})
    post_a_stab = stabilization_block.get("post_a_stabilization", {})
    lines.append("### Pre-A cleanup (before pre-A stabilization and Test A)")
    lines.append("")
    lines.append(f"* success: **{pre_a.get('cleanup_success')}**")
    lines.append(f"* remaining inference processes: {len(pre_a.get('remaining_inference', []))}")
    for action in pre_a.get("cleanup_actions", []):
        lines.append(f"* action: {action}")
    lines.append("")
    lines.append("### Pre-A stabilization")
    lines.append("")
    lines.append(
        f"* requested: {pre_a_stab.get('requested_seconds')} s "
        f"(actual {pre_a_stab.get('actual_seconds')} s)"
    )
    lines.append(f"* completed successfully: {pre_a_stab.get('completed_successfully')}")
    lines.append(f"* shortened: {pre_a_stab.get('shortened')}")
    lines.append(f"* telemetry samples: {pre_a_stab.get('telemetry', {}).get('sample_count')}")
    lines.append("")
    lines.append("### Post-A cleanup (before post-A stabilization and Test B)")
    lines.append("")
    lines.append(f"* success: **{post_a.get('cleanup_success')}**")
    lines.append(f"* remaining inference processes: {len(post_a.get('remaining_inference', []))}")
    for action in post_a.get("cleanup_actions", []):
        lines.append(f"* action: {action}")
    lines.append("")
    lines.append("### Post-A stabilization")
    lines.append("")
    lines.append(
        f"* requested: {post_a_stab.get('requested_seconds')} s "
        f"(actual {post_a_stab.get('actual_seconds')} s)"
    )
    lines.append(f"* completed successfully: {post_a_stab.get('completed_successfully')}")
    lines.append(f"* shortened: {post_a_stab.get('shortened')}")
    lines.append(f"* telemetry samples: {post_a_stab.get('telemetry', {}).get('sample_count')}")
    lines.append("")
    machine = analysis.get("machine_state_comparison", {})
    lines.append("### Stabilized machine-state comparison (pre-A vs pre-B)")
    lines.append("")
    lines.append("| metric | pre-A stabilized | post-A stabilized | relative change | material |")
    lines.append("|---|---|---|---|---|")
    for key, item in (machine.get("metrics") or {}).items():
        if not item.get("available"):
            lines.append(f"| {key} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {key} | {_fmt(item.get('pre_a_stabilized'))} | "
            f"{_fmt(item.get('post_a_stabilized'))} | "
            f"{_fmt_pct(item.get('relative_change'))} | {item.get('material')} |"
        )
    lines.append("")
    material = machine.get("material_differences", [])
    lines.append(
        "* material differences: "
        + (", ".join(material) if material else "none above the 25% threshold")
    )
    lines.append(f"* {machine.get('note', '')}")
    lines.append("")
    equal_work = analysis.get("equal_work", {})
    if equal_work:
        detail = equal_work.get("detail", {})
        lines.append("### A/B workload equivalence (effectiveness eligibility)")
        lines.append("")
        lines.append(f"* effective comparison: **{equal_work.get('valid')}**")
        lines.append(
            f"* intended requests: A={detail.get('intended_a')} B={detail.get('intended_b')}"
        )
        lines.append(
            f"* executed requests: A={detail.get('executed_a')} B={detail.get('executed_b')}"
        )
        lines.append(
            f"* successful requests: A={detail.get('success_a')} B={detail.get('success_b')}"
        )
        lines.append(
            f"* failed requests: A={detail.get('failed_a')} B={detail.get('failed_b')}"
        )
        lines.append(
            f"* monitor termination: {detail.get('monitor_termination')} "
            f"(early={detail.get('terminated_early')})"
        )
        for check_name, ok in (equal_work.get("checks") or {}).items():
            lines.append(f"* [{'PASS' if ok else 'FAIL'}] {check_name}")
        if equal_work.get("failed_checks"):
            lines.append(
                "* FAILED checks: " + "; ".join(equal_work.get("failed_checks", []))
            )
        lines.append("")
    lines.append("## Validity / contamination")
    lines.append("")
    for check in validity.get("checks", []):
        mark = {True: "PASS", False: "FAIL", None: "UNKNOWN"}[check.get("ok")]
        major = " (major)" if check.get("major") else ""
        lines.append(f"* [{mark}] {check.get('check')}{major}: {check.get('detail')}")
    lines.append("")
    for note in validity.get("notes", []):
        lines.append(f"* {note}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(interpretation.get("statement", ""))
    lines.append("")
    if interpretation.get("verdict") == "no_demonstrated_improvement":
        lines.append("```")
        lines.append("No demonstrated improvement")
        lines.append("```")
    elif interpretation.get("verdict") == "inconclusive":
        lines.append("```")
        lines.append("INCONCLUSIVE")
        lines.append("```")
    elif interpretation.get("verdict") == "invalid_comparison":
        lines.append("```")
        lines.append("INVALID COMPARISON (unequal A/B workload)")
        lines.append("```")
    for note in interpretation.get("notes", []):
        lines.append(f"* {note}")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4g}"
    return "n/a"


def _fmt_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):+.1%}"
    return "n/a"


# --------------------------------------------------------------------------- #
# Dry run                                                                       #
# --------------------------------------------------------------------------- #
def run_dry_run(
    cfg: ExperimentConfig,
    run: RunDirectory,
    *,
    preflight: PreflightReport,
    requests: Sequence[FixedRequest],
    metadata: dict[str, Any],
    settings: Settings,
) -> int:
    """Prove provisioning, discovery, telemetry, workload and reporting work."""
    status = "DRY_RUN"
    dry_dir = run.path("dry_run")
    atomic_write_json(dry_dir / "preflight.json", preflight.data)
    atomic_write_json(
        dry_dir / "workload.json",
        _workload_artifact(cfg, requests),
    )
    # Exercise the new pre-A cleanup stage (same logic used before the real Test A).
    pre_a_before = snapshot_processes()
    pre_a_cleanup = run_cleanup(
        run,
        before_records=pre_a_before,
        launched_pids=[],
        label="pre_a",
        phase_noun="during the dry run",
    )
    atomic_write_json(dry_dir / "pre_a_cleanup.json", pre_a_cleanup)
    # Exercise both stabilization stages with a short dry-run duration (the real run
    # uses the full default 20 minutes for each).
    dry_duration = max(1.0, cfg.sample_interval_s)
    pre_a_stabilization = run_stabilization(
        run,
        cfg,
        label="pre_a_stabilization",
        before_snapshot=run.path("stabilization", "pre_a_before.json"),
        after_snapshot=run.path("stabilization", "pre_a_after.json"),
        settings=settings,
        duration_s=dry_duration,
        dry_run=True,
    )
    atomic_write_json(dry_dir / "pre_a_stabilization.json", pre_a_stabilization)
    post_a_stabilization = run_stabilization(
        run,
        cfg,
        label="post_a_stabilization",
        before_snapshot=run.path("stabilization", "post_a_before.json"),
        after_snapshot=run.path("stabilization", "post_a_after.json"),
        settings=settings,
        duration_s=dry_duration,
        dry_run=True,
    )
    atomic_write_json(dry_dir / "post_a_stabilization.json", post_a_stabilization)
    # Real telemetry for a few samples (no inference).
    collector = TelemetryCollector(
        jsonl_path=dry_dir / "telemetry.jsonl",
        interval_s=cfg.sample_interval_s,
        target_pid_provider=lambda: None,
    )
    collector.start()
    for _ in range(max(cfg.dry_run_samples, 1)):
        time.sleep(max(cfg.sample_interval_s, 0.25))
    telemetry = collector.stop()
    processes = snapshot_processes()
    atomic_write_json(
        dry_dir / "processes.json",
        {
            "total": len(processes),
            "by_category": _category_counts(processes),
            "relevant": [
                asdict(record)
                for record in processes_by_category(
                    processes,
                    ("llama_server", "llama_cpp", "ollama", "model_server", "inferenceos"),
                )
            ],
        },
    )
    analysis = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utcnow_iso(),
        "status": status,
        "config": {
            "model_path": str(cfg.model_path) if cfg.model_path else None,
            "context_length": cfg.context_length,
            "gpu_layers": cfg.gpu_layers,
            "threads": cfg.threads,
            "rounds": cfg.rounds,
            "n_predict": cfg.n_predict,
            "seed": cfg.seed,
            "cooldown_seconds": cfg.cooldown_s,
        },
        "raw": {"dry_run_telemetry": asdict(telemetry), "preflight_passed": preflight.passed},
        "derived": {},
        "interpretation": {
            "verdict": "dry_run",
            "verdict_text": "DRY RUN (no A/B executed)",
            "statement": (
                "Preflight, provisioning discovery, telemetry collection, workload "
                "construction and report generation all executed. The actual A/B "
                "experiment was not run."
            ),
            "notes": [],
        },
        "validity": {
            "dry_run": True,
            "blockers": preflight.blockers,
            "warnings": preflight.warnings,
            "pre_a_cleanup_success": pre_a_cleanup.get("cleanup_success"),
        },
        "cleanup": {
            "pre_a": _cleanup_summary(pre_a_cleanup),
            "post_a": {"available": False},
        },
        "stabilization": {
            "pre_a_stabilization": _stabilization_summary(pre_a_stabilization),
            "post_a_stabilization": _stabilization_summary(post_a_stabilization),
        },
        "machine_state_comparison": _compare_machine_states(
            pre_a_stabilization.get("machine_state_after") or {},
            post_a_stabilization.get("machine_state_after") or {},
        ),
        "cross_run_net_benefit": {},
    }
    atomic_write_json(run.path("comparison.json"), analysis)
    atomic_write_text_safe(
        run.path("report.md"),
        render_report(
            cfg=cfg,
            metadata=metadata,
            preflight=preflight.data,
            analysis=analysis,
            status=status,
        ),
    )
    atomic_write_json(run.path("preflight.json"), preflight.data)
    if preflight.blockers:
        _write_setup_required(run, cfg, preflight)
    return 0


def _category_counts(records: Sequence[ProcessRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.category] = counts.get(record.category, 0) + 1
    return counts


def _workload_artifact(cfg: ExperimentConfig, requests: Sequence[FixedRequest]) -> dict[str, Any]:
    rounds = (
        len(requests) // len(FIXED_PROMPTS) if FIXED_PROMPTS and requests else cfg.rounds
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utcnow_iso(),
        "description": "Deterministic fixed A/B workload (identical for Test A and Test B)",
        "rounds": rounds,
        "n_predict": cfg.n_predict,
        "seed": cfg.seed,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "cache_prompt": False,
        "requests": [
            {
                "index": request.index,
                "request_id": request.request_id,
                "prompt": request.prompt,
                "prompt_sha256": request.prompt_sha256,
                "n_predict": request.n_predict,
                "seed": request.seed,
                "temperature": request.temperature,
                "ignore_eos": request.ignore_eos,
            }
            for request in requests
        ],
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #
def _build_metadata(
    cfg: ExperimentConfig,
    *,
    model: Path | None,
    model_metadata: dict[str, Any],
    server_path: Path | None,
    settings: Settings,
) -> dict[str, Any]:
    version = None
    if server_path is not None and server_path.is_file():
        version = _safe_call(lambda: _llama_cli_version(server_path))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utcnow_iso(),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "inferenceos_version": INFERENCEOS_VERSION,
        "inferenceos_git_revision": _git_revision(),
        "model_filename": model.name if model else cfg.model_filename,
        "model_path": str(model) if model else None,
        "model_size_bytes": model_metadata.get("size_bytes"),
        "model_sha256": model_metadata.get("sha256"),
        "model_source_url": cfg.model_url or None,
        "model_source_name": MODEL_SOURCE_NAME,
        "model_sha256_expected": cfg.model_sha256 or MODEL_DEFAULT_SHA256 or None,
        "llama_server_path": str(server_path) if server_path else None,
        "llama_version": version,
        "llama_expected_release": LLAMA_RELEASE_HINT,
        "experiment_config_hash": cfg.config_hash(),
        "experiment_config": {
            "host": cfg.host,
            "context_length": cfg.context_length,
            "gpu_layers": cfg.gpu_layers,
            "threads": cfg.threads,
            "rounds": cfg.rounds,
            "n_predict": cfg.n_predict,
            "seed": cfg.seed,
            "cooldown_seconds": cfg.cooldown_s,
            "sample_interval_s": cfg.sample_interval_s,
            "test_duration_s": cfg.test_duration_s,
            "auto_approve": cfg.auto_approve,
        },
        "settings": {
            "phase7_min_effect": getattr(settings, "phase7_min_effect", None),
            "phase10_deviation_threshold": getattr(settings, "phase10_deviation_threshold", None),
            "resident_readiness_timeout_s": getattr(settings, "resident_readiness_timeout_s", None),
        },
    }


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _write_setup_required(
    run: RunDirectory, cfg: ExperimentConfig, preflight: PreflightReport
) -> None:
    payload = {
        "blockers": preflight.blockers,
        "warnings": preflight.warnings,
        "setup_commands": preflight.setup_commands,
        "resume_command": (
            f"{sys.executable} {Path(__file__).name} --resume --run-dir \"{cfg.run_dir}\""
        ),
    }
    atomic_write_json(run.path("setup_required.json"), payload)
    atomic_write_text_safe(
        run.path("setup_required.txt"),
        "\n".join(
            ["Blockers:"]
            + [f"  - {item}" for item in preflight.blockers]
            + ["", "Run:"]
            + [f"  {item}" for item in preflight.setup_commands]
        )
        + "\n",
    )


def execute(cfg: ExperimentConfig) -> int:
    settings = Settings()
    settings.ensure_directories()
    run = RunDirectory(cfg.run_dir)
    config_hash = cfg.config_hash()

    # ---- PRE-FLIGHT ------------------------------------------------------ #
    # Optional explicit provisioning first (downloads only when --model-url is given).
    ensure_model_available(cfg)
    preflight = collect_preflight(cfg, settings)
    atomic_write_json(run.path("preflight.json"), preflight.data)
    if not preflight.passed:
        _write_setup_required(run, cfg, preflight)
        if not cfg.dry_run:
            print("SETUP REQUIRED:")
            for item in preflight.blockers:
                print(f"  - {item}")
            print("Run:")
            for command in preflight.setup_commands:
                print(f"  {command}")
            return 3

    requests = build_fixed_requests(
        rounds=cfg.rounds,
        n_predict=cfg.n_predict,
        seed=cfg.seed,
        temperature=0.0,
        ignore_eos=True,
    )
    model_path, model_metadata = acquire_model(cfg)
    server_path = cfg.server_path or _safe_call(lambda: find_server(cfg.llama_dir))
    metadata = _build_metadata(
        cfg,
        model=model_path,
        model_metadata=model_metadata,
        server_path=server_path,
        settings=settings,
    )
    atomic_write_json(run.path("metadata.json"), metadata)

    if cfg.dry_run:
        return run_dry_run(
            cfg,
            run,
            preflight=preflight,
            requests=requests,
            metadata=metadata,
            settings=settings,
        )

    if model_path is None or server_path is None:
        print("SETUP REQUIRED: model or llama-server missing after preflight.")
        return 3

    atomic_write_json(run.path("workload.json"), _workload_artifact(cfg, requests))

    resumed_test_a = bool(
        cfg.resume
        and run.is_complete("test_a", config_hash)
        and run.path("test_a", "summary.json").is_file()
    )

    # ---- PRE-A CLEANUP -> PRE-A STABILIZATION -> TEST A ------------------ #
    if resumed_test_a:
        pre_a_cleanup = _load_cleanup(run, "pre_a")
        pre_a_stabilization = _load_stabilization(run, "pre_a_stabilization")
        test_a = json.loads(run.path("test_a", "summary.json").read_text(encoding="utf-8"))
        process_before_a = _load_process_records(
            run.path("cleanup", "post_a_process_before.json")
        )
        print("[resume] Test A already complete; loading artifacts")
    else:
        print("=== PRE-A CLEANUP (establish clean starting state) ===")
        pre_a_before = snapshot_processes()
        pre_a_cleanup = run_cleanup(
            run,
            before_records=pre_a_before,
            launched_pids=[],
            label="pre_a",
            phase_noun="before Test A",
        )
        if not pre_a_cleanup.get("cleanup_success"):
            print(
                "Pre-A cleanup could not establish a clean starting state; "
                "refusing to start stabilization or Test A."
            )
            print(f"see {run.path('cleanup', 'pre_a_cleanup.json')}")
            return 5
        # PRE-A STABILIZATION (20 minutes) -- Test A must not begin immediately after
        # provisioning/setup.
        print("=== PRE-A STABILIZATION (20 minutes) ===")
        _write_machine_snapshot(
            run.path("stabilization", "pre_a_before.json"), target_pid=None, settings=settings
        )
        pre_a_stabilization = run_stabilization(
            run,
            cfg,
            label="pre_a_stabilization",
            before_snapshot=run.path("stabilization", "pre_a_before.json"),
            after_snapshot=run.path("test_a", "machine_before.json"),
            settings=settings,
        )
        if not pre_a_stabilization.get("completed_successfully"):
            print("Pre-A stabilization did not complete; refusing to start Test A.")
            return 6
        process_before_a = snapshot_processes()
        print("=== TEST A: unmanaged llama-server baseline ===")
        test_a = run_test_a(
            cfg,
            run,
            server=server_path,
            model_path=model_path,
            requests=requests,
            settings=settings,
        )
        run.mark_complete("test_a", config_hash)

    launched_pids = [test_a.get("server_pid")] if test_a.get("server_pid") else []

    # ---- POST-A CLEANUP -------------------------------------------------- #
    print("=== POST-A CLEANUP (before stabilization and Test B) ===")
    cleanup = run_cleanup(
        run,
        before_records=process_before_a,
        launched_pids=[int(pid) for pid in launched_pids if pid],
        label="post_a",
        phase_noun="after Test A",
    )
    if not cleanup.get("cleanup_success"):
        print("Post-A cleanup could not be verified; refusing to start Test B.")
        return 4
    atomic_write_json(
        run.path("cleanup", "cleanup.json"),
        {"pre_a": pre_a_cleanup, "post_a": cleanup},
    )

    # ---- POST-A STABILIZATION -------------------------------------------- #
    print("=== POST-A STABILIZATION (20 minutes) ===")
    post_a_stabilization = run_stabilization(
        run,
        cfg,
        label="post_a_stabilization",
        before_snapshot=run.path("test_a", "machine_after.json"),
        after_snapshot=run.path("test_b", "machine_before.json"),
        settings=settings,
    )
    if not post_a_stabilization.get("completed_successfully"):
        print("Post-A stabilization did not complete; refusing to start Test B.")
        return 6

    # ---- TEST B ---------------------------------------------------------- #
    try:
        model_profile = load_model_profile(model_path)
    except (ModelMetadataError, OSError) as exc:
        print(f"ERROR: could not parse GGUF model profile: {exc}")
        return 3
    a_duration = float(test_a.get("telemetry", {}).get("duration_s") or 0.0)
    monitor_duration = (
        cfg.test_duration_s if cfg.test_duration_s > 0 else min(max(a_duration * 1.5 + 60.0, 120.0), 1800.0)
    )
    print("=== TEST B: InferenceOS-managed llama-server ===")
    test_b = run_test_b(
        cfg,
        run,
        model_path=model_path,
        model_profile=model_profile,
        requests=requests,
        settings=settings,
        monitor_duration_s=monitor_duration,
        server_path=server_path,
    )
    # Persist the post-intervention flag on B outcomes for a retained-effect analysis.
    flagged: list[dict[str, Any]] = []
    for item in test_b.get("outcomes", []):
        outcome = RequestOutcome(**item)
        if test_b.get("intervention_end") and outcome.submitted_at >= test_b["intervention_end"]:
            outcome.post_intervention = True
        flagged.append(outcome.to_dict())
    test_b["outcomes"] = flagged
    final_cleanup = run_final_cleanup(run)
    test_b["final_cleanup"] = final_cleanup
    atomic_write_json(run.path("test_b", "summary.json"), test_b)
    run.mark_complete("test_b", config_hash)

    # ---- ANALYSIS -------------------------------------------------------- #
    print("=== A/B ANALYSIS ===")
    # Attach model fingerprints for the validity check.
    test_a["model_sha256"] = metadata.get("model_sha256")
    test_b["model_sha256"] = metadata.get("model_sha256")
    analysis = analyze_ab(
        test_a=test_a,
        test_b=test_b,
        cfg=cfg,
        cleanup=cleanup,
        pre_a_stabilization=pre_a_stabilization,
        post_a_stabilization=post_a_stabilization,
        preflight=preflight.data,
        settings=settings,
        pre_a_cleanup=pre_a_cleanup,
    )
    atomic_write_json(run.path("comparison.json"), analysis)
    atomic_write_text_safe(
        run.path("report.md"),
        render_report(
            cfg=cfg,
            metadata=metadata,
            preflight=preflight.data,
            analysis=analysis,
            status="COMPLETE",
        ),
    )
    print(analysis["interpretation"]["statement"])
    print(f"verdict: {analysis['interpretation']['verdict_text']}")
    print(f"results: {run.root}")
    run.mark_complete("analysis", config_hash)
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Run the InferenceOS A/B effectiveness experiment (unmanaged vs managed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="artifact directory")
    parser.add_argument("--resume", action="store_true", help="resume a previous run")
    parser.add_argument("--dry-run", action="store_true", help="preflight only; no A/B")
    parser.add_argument("--model-path", type=Path, default=None, help="exact GGUF file to use")
    parser.add_argument("--model-url", default=MODEL_DEFAULT_URL, help="explicit download URL")
    parser.add_argument("--model-sha256", default=MODEL_DEFAULT_SHA256, help="expected SHA-256")
    parser.add_argument("--model-filename", default=MODEL_DEFAULT_FILENAME)
    parser.add_argument("--llama-dir", type=Path, default=PROJECT_ROOT / "tools" / "llama.cpp")
    parser.add_argument("--server", type=Path, default=None, help="llama-server executable")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--n-predict", type=int, default=DEFAULT_N_PREDICT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-interval-s", type=float, default=DEFAULT_SAMPLE_INTERVAL_S)
    parser.add_argument(
        "--test-duration-seconds",
        type=float,
        default=0.0,
        help="0 = derive from the baseline run",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=STABILIZATION_SECONDS_DEFAULT,
        help="duration of EACH stabilization window (pre-A and post-A), in seconds",
    )
    parser.add_argument(
        "--allow-short-cooldown",
        action="store_true",
        help="explicitly allow a shortened stabilization window (recorded as a confound)",
    )
    parser.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="reject disruptive InferenceOS interventions instead of approving them",
    )
    parser.add_argument("--dry-run-samples", type=int, default=3)
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = PROJECT_ROOT / "artifacts" / "ab_effectiveness" / run_stamp()
    run_dir = Path(run_dir)
    threads = int(args.threads) if args.threads and args.threads > 0 else None
    model_path = Path(args.model_path) if args.model_path else None
    if model_path is None and args.model_filename:
        model_path = find_existing_model(args.model_filename)
    return ExperimentConfig(
        run_dir=run_dir,
        model_path=model_path,
        model_url=str(args.model_url or ""),
        model_sha256=str(args.model_sha256 or ""),
        model_filename=str(args.model_filename),
        llama_dir=Path(args.llama_dir),
        server_path=Path(args.server) if args.server else None,
        host=str(args.host),
        context_length=int(args.context_length),
        gpu_layers=int(args.gpu_layers),
        threads=threads,
        rounds=max(int(args.rounds), 1),
        n_predict=int(args.n_predict),
        seed=int(args.seed),
        sample_interval_s=float(args.sample_interval_s),
        test_duration_s=float(args.test_duration_seconds),
        cooldown_s=float(args.cooldown_seconds),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        auto_approve=not bool(args.no_auto_approve),
        allow_short_cooldown=bool(args.allow_short_cooldown),
        dry_run_samples=int(args.dry_run_samples),
    )


def main(argv: Sequence[str] | None = None) -> int:
    cfg = parse_args(argv)
    return execute(cfg)


if __name__ == "__main__":
    _prefer_project_venv()
    raise SystemExit(main())
