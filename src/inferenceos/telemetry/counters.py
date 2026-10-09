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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Sequence

from inferenceos.logging_config import get_logger
from inferenceos.native.base import CounterSample
from inferenceos.telemetry.attribution import (
    CLOCK_SLACK_S,
    CPU_PERFORMANCE_SPEC,
    PAGING_SPEC,
    PagingVerdict,
    RawReading,
    interval_mean,
    paging_verdict,
    ticks_to_unix,
)

HARD_FAULT_PATH = r"\Memory\Pages Input/sec"


class MonitorCleanupUnresolved(BaseException):
    """A counter sampler's cleanup could not be confirmed (its process or reader may
    still be alive). No further measurement, sampler or published result may follow.
    A ``BaseException`` on purpose: generic ``except Exception`` retry handling (a
    failed run is normally recorded and the next one started) must not swallow it."""

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
        #: Arrival time (perf_counter) of each sample, parallel to ``_samples``.
        self._times: list[float] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._started_at = 0.0
        self._stopped_at = 0.0
        #: False after a ``stop()`` whose reader or process could not be confirmed finished.
        self.cleanup_confirmed = True

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
                arrived = time.perf_counter()
                with self._lock:
                    self._samples.append(parsed)
                    self._times.append(arrived)

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

    def snapshot(self) -> list[CounterSample]:
        """Return the samples captured so far without stopping the sampler.

        Enables a low-cadence consumer (e.g. runtime monitoring) to poll the persistent
        counter stream non-destructively.
        """
        with self._lock:
            return list(self._samples)

    def timed_snapshot(self) -> list[tuple[float, CounterSample]]:
        """Samples captured so far with their arrival times (non-destructive)."""
        with self._lock:
            return list(zip(self._times, self._samples))

    def discard_before(self, cutoff: float) -> int:
        """Forget samples that arrived before ``cutoff`` (bounds what a long-lived
        sampler keeps); returns how many were dropped."""
        with self._lock:
            keep = next((i for i, t in enumerate(self._times) if t >= cutoff), len(self._times))
            del self._times[:keep]
            del self._samples[:keep]
            return keep

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
        # Whether the cleanup is confirmed (reviewer, monitor round 2): the reader has
        # finished and the PowerShell process has exited (a bounded wait; immediate in
        # the normal case). The profiler refuses further work while this is False.
        exited = True
        if proc is not None and proc.poll() is None:
            try:
                proc.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, OSError):
                exited = False
        self.cleanup_confirmed = exited and not (self._thread is not None and self._thread.is_alive())
        self._stopped_at = time.perf_counter()
        with self._lock:
            return list(self._samples)

    @property
    def duration_s(self) -> float:
        if not self._started_at:
            return 0.0
        end = self._stopped_at or time.perf_counter()
        return max(end - self._started_at, 0.0)


#: Counter paths the measurement-scoped monitor can read with ``NextSample()``
#: (category, counter, instance). Wildcard-instance GPU counters are not collected
#: in scoped mode (descriptive only; no tune decision uses them).
SCOPED_SPECS: dict[str, tuple[str, str, str]] = {
    r"\Memory\Pages Input/sec": ("Memory", "Pages Input/sec", ""),
    r"\Processor Information(_Total)\% Processor Performance": (
        "Processor Information", "% Processor Performance", "_Total"),
}

#: Reads every counter with .NET ``NextSample()`` -- no Get-Counter one-second wait --
#: bracketing each read with the system clock (UTC ticks). Each line:
#: ``S|spec|before_ticks|after_ticks|raw|base|ts100|type|calculated``; ``calculated``
#: is ``CounterSample.Calculate(previous, current)`` (empty for the first reading).
#: Invariant culture throughout, so decimals never depend on the laptop's locale.
_RAW_PS_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$inv = [Globalization.CultureInfo]::InvariantCulture
$specs = @(__SPECS__)
$counters = @()
foreach ($spec in $specs) {
    $p = $spec.Split('|')
    try {
        $c = New-Object System.Diagnostics.PerformanceCounter($p[0], $p[1], $p[2], $true)
        $counters += ,@($spec, $c)
    } catch {
        [Console]::Out.WriteLine('E|' + $spec + '|' + ($_.Exception.Message -replace '[\r\n|]', ' '))
    }
}
[Console]::Out.Flush()
$prev = @{}
while ($true) {
    foreach ($pair in $counters) {
        $spec = $pair[0]
        $b = [DateTime]::UtcNow.Ticks
        $s = $pair[1].NextSample()
        $a = [DateTime]::UtcNow.Ticks
        $calc = ''
        if ($prev.ContainsKey($spec)) {
            try { $calc = ([System.Diagnostics.CounterSample]::Calculate($prev[$spec], $s)).ToString('R', $inv) } catch { $calc = '' }
        }
        $prev[$spec] = $s
        [Console]::Out.WriteLine([string]::Format($inv, 'S|{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}',
            [object[]]@($spec, $b, $a, $s.RawValue, $s.BaseValue, $s.TimeStamp100nSec, [int]$s.CounterType, $calc)))
    }
    [Console]::Out.Flush()
    Start-Sleep -Milliseconds __INTERVAL__
}
"""


def parse_raw_line(line: str, arrival_wall: float, arrival_perf: float) -> RawReading | str | None:
    """A reading, an error text (``E|...`` lines) or None for anything unparsable."""
    parts = line.rstrip("\r\n").split("|")
    if parts and parts[0] == "E" and len(parts) >= 3:
        return f"{'|'.join(parts[1:-1])}: {parts[-1]}"
    # S | category | counter | instance | before | after | raw | base | ts100 | type | calculated
    if len(parts) != 11 or parts[0] != "S":
        return None
    spec = "|".join(parts[1:4])
    try:
        calc = float(parts[10]) if parts[10] else None
        return RawReading(spec=spec, read_before=ticks_to_unix(int(parts[4])), read_after=ticks_to_unix(int(parts[5])),
                          raw=int(parts[6]), base=int(parts[7]), ts100=int(parts[8]), counter_type=int(parts[9]),
                          calculated=calc, arrival_wall=arrival_wall, arrival_perf=arrival_perf)
    except ValueError:
        return None


class RawCounterSampler:
    """One PowerShell process reading counters with ``NextSample()``; every line keeps
    its own read bracket. Stopping is bounded in every step (terminate, wait, kill,
    join the reader, close the pipe only once the reader has returned) and reports
    what could not be resolved."""

    EXIT_S = 5.0
    READER_JOIN_S = 2.0
    PIPE_CLOSE_S = 1.0

    def __init__(self, specs: Sequence[tuple[str, str, str]], interval_s: float = 0.1) -> None:
        if not specs:
            raise ValueError("at least one counter is required")
        self._specs = ["|".join(spec) for spec in specs]
        self._interval_ms = max(20, int(interval_s * 1000))
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._readings: list[RawReading] = []
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self.stop_record: dict[str, object] = {}

    def start(self) -> bool:
        escaped = ",".join("'" + spec.replace("'", "''") + "'" for spec in self._specs)
        script = _RAW_PS_SCRIPT.replace("__SPECS__", escaped).replace("__INTERVAL__", str(self._interval_ms))
        try:
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                errors="replace", bufsize=1, creationflags=_CREATE_NO_WINDOW,
            )
        except OSError as exc:  # pragma: no cover - environment dependent
            self.errors.append(f"could not start the sampler: {exc}")
            self._proc = None
            return False
        self._thread = threading.Thread(target=self._read_loop, name="raw-counter-sampler", daemon=True)
        self._thread.start()
        return True

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:  # pragma: no cover - defensive
            return
        try:
            for line in proc.stdout:
                parsed = parse_raw_line(line, time.time(), time.perf_counter())
                with self._lock:
                    if isinstance(parsed, RawReading):
                        self._readings.append(parsed)
                    elif isinstance(parsed, str):
                        self.errors.append(parsed)
        except (OSError, ValueError):  # the pipe closed under the reader
            pass

    def readings(self) -> list[RawReading]:
        with self._lock:
            return list(self._readings)

    def has_reading(self, spec: str) -> bool:
        with self._lock:
            return any(r.spec == spec for r in self._readings)

    def discard_before(self, wall: float) -> int:
        """Forget readings made before ``wall`` except the last one of each counter
        (it opens the next interval)."""
        with self._lock:
            keep: list[RawReading] = []
            last: dict[str, RawReading] = {}
            for r in self._readings:
                if r.read_after < wall:
                    last[r.spec] = r
                else:
                    keep.append(r)
            dropped = len(self._readings) - len(keep) - len(last)
            self._readings = sorted([*last.values(), *keep], key=lambda r: r.read_after)
            return dropped

    def stop(self) -> dict[str, object]:
        """Bounded stop; never raises. Returns what happened (``resolved`` False means a
        sampler process or its reader may still be alive)."""
        record: dict[str, object] = {"exit_confirmed": None, "reader_joined": None, "pipe_closed": False,
                                     "errors": []}
        errors: list[str] = []
        proc, reader = self._proc, self._thread
        if proc is None:
            record.update(exit_confirmed=True, reader_joined=True, pipe_closed=True, resolved=True)
            self.stop_record = record
            return record
        try:
            proc.terminate()
        except Exception as exc:  # noqa: BLE001 - recorded
            errors.append(f"terminate: {type(exc).__name__}: {exc}")
        confirmed = False
        try:
            proc.wait(timeout=self.EXIT_S)
            confirmed = True
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=self.EXIT_S)
                confirmed = True
            except Exception as exc:  # noqa: BLE001 - recorded
                errors.append(f"kill: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - recorded
            errors.append(f"wait: {type(exc).__name__}: {exc}")
        joined = True
        if reader is not None:
            reader.join(self.READER_JOIN_S)
            joined = not reader.is_alive()
        closed = False
        if joined and proc.stdout is not None:
            box: dict[str, str] = {}

            def close() -> None:
                try:
                    assert proc.stdout is not None
                    proc.stdout.close()
                except Exception as exc:  # noqa: BLE001 - recorded
                    box["error"] = f"{type(exc).__name__}: {exc}"

            closer = threading.Thread(target=close, daemon=True)
            closer.start()
            closer.join(self.PIPE_CLOSE_S)
            closed = not closer.is_alive() and "error" not in box
            if "error" in box:
                errors.append(f"close: {box['error']}")
            elif closer.is_alive():
                errors.append(f"close: did not return within {self.PIPE_CLOSE_S:g} s")
        elif not joined:
            errors.append(f"reader: still blocked after {self.READER_JOIN_S:g} s (pipe left open)")
        record.update(exit_confirmed=confirmed, reader_joined=joined, pipe_closed=closed, errors=errors,
                      resolved=bool(confirmed and joined and closed))
        self.stop_record = record
        return record


@dataclass
class AnswerTelemetry:
    """What the scoped monitor attributes to one answer."""

    paging: PagingVerdict
    cpu_performance_percent: float | None
    start_wall: float
    end_wall: float
    clock_problem: str | None = None

    def record(self) -> dict[str, object]:
        return {"paging": self.paging.record(),
                "cpu_performance_percent": self.cpu_performance_percent,
                "answer_s": round(self.end_wall - self.start_wall, 3), "clock_problem": self.clock_problem,
                # Unrounded, for an exact replay against saved raw readings.
                "start_wall": self.start_wall, "end_wall": self.end_wall}


class MeasurementMonitor:
    """One counter sampler for a whole measurement (experimental; off by default).

    Opened once (its start-up cost paid once), it serves the measurement's warm-up and
    scored answers. Each answer's wall-clock interval is noted when it ends; nothing is
    decided then. :meth:`finalize` attributes readings to answers by the intervals the
    readings describe (``inferenceos.telemetry.attribution``), waiting -- bounded, and
    only once per measurement -- for the reading that closes the last answer. Late
    readings are therefore kept, not lost. Missing, incomplete or ambiguous evidence
    stays unknown. :meth:`close` is idempotent and bounded.
    """

    def __init__(self, paths: Sequence[str], interval_s: float = 0.1, *,
                 sampler_factory: Callable[..., RawCounterSampler] | None = None,
                 finalize_wait_s: float = 2.0) -> None:
        self.collected = [path for path in paths if path in SCOPED_SPECS]
        self.not_collected = [path for path in paths if path not in SCOPED_SPECS]
        specs = [SCOPED_SPECS[path] for path in self.collected] or [SCOPED_SPECS[HARD_FAULT_PATH]]
        self._sampler = (sampler_factory or RawCounterSampler)(specs, interval_s=interval_s)
        self.finalize_wait_s = float(finalize_wait_s)
        self.start_cost_s = 0.0
        self.stop_cost_s = 0.0
        self.stop_record: dict[str, object] = {}
        self.windows = 0
        self.closed = False
        self.opened = False
        self._answers: dict[str, tuple[float, float, float, float]] = {}
        self.last_finalize_readings: tuple[RawReading, ...] = ()

    def open(self, *, first_sample_timeout_s: float = 6.0) -> float:
        began = time.perf_counter()
        self._sampler.start()
        deadline = time.perf_counter() + first_sample_timeout_s
        while time.perf_counter() < deadline and not self._sampler.has_reading(PAGING_SPEC):
            time.sleep(0.02)
        self.opened = True
        self.start_cost_s = time.perf_counter() - began
        return self.start_cost_s

    def note_answer(self, key: str, *, wall_start: float, wall_end: float,
                    perf_start: float, perf_end: float) -> None:
        self._answers[key] = (wall_start, wall_end, perf_start, perf_end)
        self.windows += 1

    def finalize(self, keys: Sequence[str]) -> tuple[dict[str, AnswerTelemetry], float]:
        """Attribute readings to the noted answers ``keys``; returns their telemetry and
        the time spent waiting for the last closing reading (bounded)."""
        wanted = [key for key in keys if key in self._answers]
        waited = 0.0
        if wanted and not self.closed:
            last_end = max(self._answers[key][1] for key in wanted)
            began = time.perf_counter()
            deadline = began + self.finalize_wait_s
            while time.perf_counter() < deadline:
                if any(r.spec == PAGING_SPEC and r.read_before - CLOCK_SLACK_S >= last_end
                       for r in self._sampler.readings()):
                    break
                time.sleep(0.01)
            waited = time.perf_counter() - began
        readings = self._sampler.readings()
        #: Immutable copy of exactly the readings these answers were judged on, taken
        #: before retention trims history (validation and reports read this).
        self.last_finalize_readings = tuple(readings)
        out: dict[str, AnswerTelemetry] = {}
        for key in wanted:
            wall_start, wall_end, perf_start, perf_end = self._answers.pop(key)
            # The answer was timed on both clocks; if they disagree the wall clock moved
            # during the answer and its placement is not trustworthy.
            drift = (wall_end - wall_start) - (perf_end - perf_start)
            clock_problem = (f"the system clock moved {drift:+.3f} s during the answer"
                             if abs(drift) > CLOCK_SLACK_S else None)
            if clock_problem:
                verdict = PagingVerdict("unknown", None, None, f"unusable timing: {clock_problem}")
                cpu = None
            else:
                verdict = paging_verdict(readings, wall_start, wall_end)
                if verdict.status == "pending":
                    verdict = PagingVerdict("unknown", None, None,
                                            f"no reading after the answer arrived within {self.finalize_wait_s:g} s")
                cpu = interval_mean(readings, wall_start, wall_end, CPU_PERFORMANCE_SPEC)
            out[key] = AnswerTelemetry(verdict, cpu, wall_start, wall_end, clock_problem)
        if out:
            # Answers noted before these and never finalized (warm-ups) are dropped,
            # then readings no later answer can need: keep the last reading before the
            # newest finalized end (it opens the next interval) and anything later.
            earliest = min(t.start_wall for t in out.values())
            for key in [k for k, a in self._answers.items() if a[1] <= earliest]:
                del self._answers[key]
            if not self._answers:
                self._sampler.discard_before(max(t.end_wall for t in out.values()))
        return out, waited

    def forget(self) -> None:
        self._answers.clear()

    @property
    def errors(self) -> list[str]:
        return list(getattr(self._sampler, "errors", []))

    def close(self) -> float:
        if self.closed:
            return 0.0
        self.closed = True
        if not self.opened:
            return 0.0
        began = time.perf_counter()
        try:
            self.stop_record = dict(self._sampler.stop())
        finally:
            self.stop_cost_s = time.perf_counter() - began
        return self.stop_cost_s


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
