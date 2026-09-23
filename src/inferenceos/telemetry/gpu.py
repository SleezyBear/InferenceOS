"""GPU telemetry (Windows-native).

Samples the real Windows "GPU Engine" performance counters while an inference runs,
so Phase 0 can show *observed* GPU utilization rather than asserting it.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

from inferenceos.logging_config import get_logger

logger = get_logger("telemetry.gpu")

_ENGINE_RE = re.compile(r"pid_(\d+)_.*?engtype_([A-Za-z0-9_]+)")

_PS_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
while ($true) {
    'HB'
    [Console]::Out.Flush()
    $t = [DateTime]::UtcNow.Ticks / 10000000.0
    $samples = (Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples
    foreach ($c in $samples) {
        if ($c.CookedValue -gt 0) {
            '{0}|{1}|{2}' -f $t, $c.InstanceName, $c.CookedValue
        }
    }
    [Console]::Out.Flush()
    Start-Sleep -Milliseconds __INTERVAL__
}
"""


@dataclass(frozen=True)
class GpuSample:
    timestamp: float
    pid: int | None
    engine: str
    utilization_percent: float


@dataclass
class GpuUtilizationSummary:
    samples: list[GpuSample] = field(default_factory=list)
    max_overall_percent: float = 0.0
    max_engine: str | None = None
    max_target_percent: float = 0.0
    target_pid: int | None = None
    duration_s: float = 0.0
    engine_max_percent: dict[str, float] = field(default_factory=dict)


class GpuUtilizationSampler:
    """Background sampler over Windows GPU Engine utilization counters."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self._interval_ms = max(100, int(interval_s * 1000))
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._samples: list[GpuSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._start_time = 0.0

    def start(self) -> None:
        self._stop.clear()
        self._ready.clear()
        script = _PS_SCRIPT.replace("__INTERVAL__", str(self._interval_ms))
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:  # pragma: no cover - environment dependent
            logger.warning("Failed to start GPU sampler: %s", exc)
            self._proc = None
            return
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(
            target=self._read_loop, name="gpu-sampler", daemon=True
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
        """Wait until the counter subprocess is running and reporting.

        The stream emits a heartbeat line (ignored by the sample parser) at the
        start of every sampling loop, so this confirms the sampler is live before
        a short workload begins without depending on GPU activity being non-zero.
        """
        return self._ready.wait(timeout=timeout_s)

    def wait_for_first_sample(self, timeout_s: float = 4.0, poll_s: float = 0.1) -> bool:
        """Wait until the real counter stream has produced at least one sample.

        Useful before starting a short workload so the sampler is known to be
        capturing. Returns False if no sample arrived within the timeout (e.g. a
        completely idle GPU that reports no non-zero engine).
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

    def stop(self, target_pid: int | None = None) -> GpuUtilizationSummary:
        self._stop.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except OSError:  # pragma: no cover - race on exit
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        duration = time.perf_counter() - self._start_time if self._start_time else 0.0
        with self._lock:
            samples = list(self._samples)
        return _summarize(samples, target_pid, duration)


def _parse_line(line: str) -> GpuSample | None:
    parts = line.strip().split("|")
    if len(parts) != 3:
        return None
    try:
        timestamp = float(parts[0])
        value = float(parts[2])
    except ValueError:
        return None
    match = _ENGINE_RE.search(parts[1])
    pid = int(match.group(1)) if match else None
    engine = match.group(2) if match else "unknown"
    return GpuSample(
        timestamp=timestamp, pid=pid, engine=engine, utilization_percent=value
    )


def _summarize(
    samples: list[GpuSample], target_pid: int | None, duration: float
) -> GpuUtilizationSummary:
    engine_max: dict[str, float] = {}
    for sample in samples:
        engine_max[sample.engine] = max(
            engine_max.get(sample.engine, 0.0), sample.utilization_percent
        )
    if samples:
        top = max(samples, key=lambda s: s.utilization_percent)
        max_overall = top.utilization_percent
        max_engine: str | None = top.engine
    else:
        max_overall = 0.0
        max_engine = None

    max_target = 0.0
    if target_pid is not None:
        max_target = max(
            (s.utilization_percent for s in samples if s.pid == target_pid),
            default=0.0,
        )

    return GpuUtilizationSummary(
        samples=samples,
        max_overall_percent=max_overall,
        max_engine=max_engine,
        max_target_percent=max_target,
        target_pid=target_pid,
        duration_s=duration,
        engine_max_percent=engine_max,
    )
