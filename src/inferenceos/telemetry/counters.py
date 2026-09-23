"""Persistent multi-counter sampler over real Windows performance counters.

Generalizes the Phase 0/1 single-purpose GPU sampler: it loops ``Get-Counter`` for
an arbitrary set of counter paths inside one long-lived PowerShell process and
streams ``path|instance|value`` lines. The profiler uses it to capture peak GPU
utilization, GPU adapter memory and hard page-fault rate *over an inference
interval* rather than from a single arbitrary sample.

Rate counters need more than one read to be meaningful, so the loop keeps sampling
and the caller aggregates (e.g. takes the max) after ``stop``.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Sequence

from inferenceos.logging_config import get_logger
from inferenceos.native.windows import CounterSample

logger = get_logger("telemetry.counters")

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_PS_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$paths = @(__PATHS__)
while ($true) {
    'HB'
    [Console]::Out.Flush()
    $samples = (Get-Counter -Counter $paths).CounterSamples
    foreach ($s in $samples) {
        if ($null -ne $s.CookedValue) {
            '{0}|{1}|{2}' -f $s.Path, $s.InstanceName, $s.CookedValue
        }
    }
    [Console]::Out.Flush()
    Start-Sleep -Milliseconds __INTERVAL__
}
"""


class CounterSampler:
    """Background sampler over a fixed set of Windows performance counters."""

    def __init__(self, paths: Sequence[str], interval_s: float = 0.25) -> None:
        if not paths:
            raise ValueError("at least one counter path is required")
        self._paths = list(paths)
        self._interval_ms = max(100, int(interval_s * 1000))
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._samples: list[CounterSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._started_at = 0.0
        self._stopped_at = 0.0

    @property
    def paths(self) -> list[str]:
        return list(self._paths)

    def start(self) -> None:
        self._stop.clear()
        self._ready.clear()
        escaped = ",".join("'" + path.replace("'", "''") + "'" for path in self._paths)
        script = _PS_SCRIPT.replace("__PATHS__", escaped).replace(
            "__INTERVAL__", str(self._interval_ms)
        )
        try:
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=_CREATE_NO_WINDOW,
            )
        except OSError as exc:  # pragma: no cover - environment dependent
            logger.warning("Failed to start counter sampler: %s", exc)
            self._proc = None
            return
        self._started_at = time.perf_counter()
        self._stopped_at = self._started_at
        self._thread = threading.Thread(
            target=self._read_loop, name="counter-sampler", daemon=True
        )
        self._thread.start()

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:  # pragma: no cover - defensive
            return
        for line in proc.stdout:
            if self._stop.is_set():
                break
            self._ready.set()
            parsed = _parse_line(line)
            if parsed is not None:
                with self._lock:
                    self._samples.append(parsed)

    def wait_until_ready(self, timeout_s: float = 5.0) -> bool:
        return self._ready.wait(timeout=timeout_s)

    def wait_for_first_sample(self, timeout_s: float = 5.0, poll_s: float = 0.05) -> bool:
        """Wait until the first counter reading has been captured.

        Ensuring at least one real reading exists before a (possibly very short)
        workload starts means the sampler is already producing data, rather than
        still paying its first ``Get-Counter`` initialization cost during the run.
        """
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self._stop.is_set():
                return False
            with self._lock:
                if self._samples:
                    return True
            time.sleep(poll_s)
        return False

    def stop(self) -> list[CounterSample]:
        self._stop.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except OSError:  # pragma: no cover - race on exit
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._stopped_at = time.perf_counter()
        with self._lock:
            return list(self._samples)

    @property
    def duration_s(self) -> float:
        if not self._started_at:
            return 0.0
        end = self._stopped_at or time.perf_counter()
        return max(end - self._started_at, 0.0)


def _parse_line(line: str) -> CounterSample | None:
    parts = line.strip().split("|")
    if len(parts) != 3:
        return None
    try:
        value = float(parts[2])
    except ValueError:
        return None
    return CounterSample(path=parts[0].strip(), instance=parts[1].strip() or None, value=value)


def max_value(samples: Sequence[CounterSample], path_contains: str) -> float | None:
    """Maximum value among samples whose path contains ``path_contains``."""
    needle = path_contains.lower()
    values = [sample.value for sample in samples if needle in sample.path.lower()]
    return max(values) if values else None


_ENGINE_INSTANCE_RE = re.compile(
    r"luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)_phys_(\d+)_eng_(\d+)_engtype_(\w+)"
)
_ADAPTER_INSTANCE_RE = re.compile(r"luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)_phys_(\d+)")


def max_gpu_engine_percent(
    samples: Sequence[CounterSample], engine: str | None = None
) -> float | None:
    """Maximum GPU engine utilization (optionally for one engine type)."""
    values: list[float] = []
    for sample in samples:
        if "gpu engine" not in sample.path.lower() or sample.instance is None:
            continue
        match = _ENGINE_INSTANCE_RE.search(sample.instance)
        if not match:
            continue
        engine_type = match.group(5).lower()
        if engine is None or engine_type == engine.lower():
            values.append(max(sample.value, 0.0))
    return max(values) if values else None


def max_gpu_adapter_memory_bytes(samples: Sequence[CounterSample]) -> int | None:
    values: list[float] = []
    for sample in samples:
        if "gpu adapter memory" not in sample.path.lower() or sample.instance is None:
            continue
        if _ADAPTER_INSTANCE_RE.search(sample.instance) and sample.value > 0:
            values.append(sample.value)
    return int(max(values)) if values else None


def gpu_engine_peaks(samples: Sequence[CounterSample]) -> list[tuple[str, str, float]]:
    """Peak utilization per (luid, engine) across the interval, as single-sample maxima."""
    peaks: dict[tuple[str, str], float] = {}
    for sample in samples:
        if "gpu engine" not in sample.path.lower() or sample.instance is None:
            continue
        match = _ENGINE_INSTANCE_RE.search(sample.instance)
        if not match:
            continue
        luid = f"{match.group(1).lower()}:{match.group(2).lower()}"
        key = (luid, match.group(5).lower())
        peaks[key] = max(peaks.get(key, 0.0), max(sample.value, 0.0))
    return [(luid, engine, value) for (luid, engine), value in sorted(peaks.items())]


def gpu_adapter_memory_peaks(samples: Sequence[CounterSample]) -> list[tuple[str, int]]:
    """Peak dedicated memory per adapter LUID across the interval."""
    peaks: dict[str, float] = {}
    for sample in samples:
        if "gpu adapter memory" not in sample.path.lower() or sample.instance is None:
            continue
        match = _ADAPTER_INSTANCE_RE.search(sample.instance)
        if not match:
            continue
        luid = f"{match.group(1).lower()}:{match.group(2).lower()}"
        peaks[luid] = max(peaks.get(luid, 0.0), max(sample.value, 0.0))
    return [(luid, int(value)) for luid, value in sorted(peaks.items())]
