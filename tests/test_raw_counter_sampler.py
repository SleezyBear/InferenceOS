"""The raw-counter sampler behind the scoped monitor: its line format, its bounded
stop (same principles as the audited probe cleanup) and its PowerShell script."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

import inferenceos.telemetry.counters as counters
from inferenceos.telemetry.attribution import PAGING_SPEC, ticks_to_unix
from inferenceos.telemetry.counters import RawCounterSampler, parse_raw_line

EPOCH_TICKS = 621355968000000000


def _ticks(unix: float) -> int:
    return int(unix * 1e7) + EPOCH_TICKS


def test_a_reading_line_is_parsed_with_its_bracket_and_type() -> None:
    before = _ticks(1_800_000_000.0)
    line = f"S|Memory|Pages Input/sec||{before}|{before + 20000}|4621826|0|133000000000000000|272696320|\r\n"
    reading = parse_raw_line(line, 1_800_000_000.01, 5.0)
    assert reading is not None and not isinstance(reading, str)
    assert reading.spec == PAGING_SPEC and reading.raw == 4621826 and reading.counter_type == 272696320
    assert abs(reading.read_before - 1_800_000_000.0) < 1e-6 and abs(reading.read_after - 1_800_000_000.002) < 1e-6
    assert reading.calculated is None  # the first reading has no interval yet
    assert reading.timing_problem() is None


def test_calculated_values_use_invariant_decimals() -> None:
    t = _ticks(1_800_000_000.0)
    line = f"S|Processor Information|% Processor Performance|_Total|{t}|{t}|5|7|1|542573824|97.25"
    reading = parse_raw_line(line, 1_800_000_000.0, 0.0)
    assert reading is not None and not isinstance(reading, str) and reading.calculated == 97.25
    comma = line.replace("97.25", "97,25")  # a locale-formatted value is refused, never misread
    assert parse_raw_line(comma, 1_800_000_000.0, 0.0) is None


def test_error_and_garbage_lines() -> None:
    assert parse_raw_line("E|Memory|Pages Input/sec||Category does not exist.", 0.0, 0.0) == \
        "Memory|Pages Input/sec|: Category does not exist."
    assert parse_raw_line("HB", 0.0, 0.0) is None
    assert parse_raw_line("S|too|few", 0.0, 0.0) is None


def test_ticks_conversion() -> None:
    assert abs(ticks_to_unix(EPOCH_TICKS + 10_000_000) - 1.0) < 1e-9


def test_stop_returns_while_a_real_pipe_writer_stays_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sampler that will not exit and a reader blocked on a real pipe: stop() returns
    within its bounds, closes nothing the reader holds, and reports itself unresolved."""
    monkeypatch.setattr(RawCounterSampler, "EXIT_S", 0.1)
    monkeypatch.setattr(RawCounterSampler, "READER_JOIN_S", 0.3)
    read_fd, write_fd = os.pipe()
    t = _ticks(time.time())
    os.write(write_fd, f"S|Memory|Pages Input/sec||{t}|{t}|1|0|1|272696320|\n".encode())
    stream = os.fdopen(read_fd, "r")

    class StuckProc:
        stdout = stream

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired("powershell", timeout or 0)

    monkeypatch.setattr(counters.subprocess, "Popen", lambda *a, **k: StuckProc())
    sampler = RawCounterSampler([("Memory", "Pages Input/sec", "")])
    assert sampler.start()
    deadline = time.time() + 2.0
    while time.time() < deadline and not sampler.readings():
        time.sleep(0.01)
    began = time.perf_counter()
    try:
        record = sampler.stop()
        took = time.perf_counter() - began
        reader_alive = sampler._thread is not None and sampler._thread.is_alive()  # noqa: SLF001
    finally:
        os.close(write_fd)  # release the writer: the reader sees end-of-file
    assert sampler._thread is not None  # noqa: SLF001
    sampler._thread.join(5.0)  # noqa: SLF001
    stream.close()
    print(f"\nstop returned after {took:.2f} s with the writer open; reader blocked then: {reader_alive}; {record}")
    assert took < 2.0 and reader_alive
    assert record["exit_confirmed"] is False and record["reader_joined"] is False
    assert record["pipe_closed"] is False and record["resolved"] is False
    assert len(sampler.readings()) == 1  # the reading captured before stop is kept
    assert not sampler._thread.is_alive()  # noqa: SLF001


def _shell() -> str | None:
    names = ("powershell", "pwsh") if sys.platform == "win32" else ("pwsh",)
    return next((found for name in names if (found := shutil.which(name))), None)


@pytest.mark.skipif(_shell() is None, reason="no PowerShell on this machine")
def test_the_sampler_script_parses(tmp_path: Path) -> None:
    script = counters._RAW_PS_SCRIPT.replace("__SPECS__", "'Memory|Pages Input/sec|'").replace(  # noqa: SLF001
        "__INTERVAL__", "100")
    path = tmp_path / "sampler.ps1"
    path.write_text(script, encoding="ascii")
    check = tmp_path / "check.ps1"
    check.write_text(
        "param([string]$File)\n$t=$null;$e=$null\n"
        "[void][System.Management.Automation.Language.Parser]::ParseFile($File,[ref]$t,[ref]$e)\n"
        "foreach($x in $e){\"{0}: {1}\" -f $x.Extent.StartLineNumber,$x.Message}\nexit $e.Count\n", encoding="ascii")
    done = subprocess.run([str(_shell()), "-NoProfile", "-NonInteractive", "-File", str(check), "-File", str(path)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


def test_monitor_open_waits_for_the_first_paging_reading_only_once() -> None:
    made: list[Any] = []

    class Fake:
        def __init__(self, specs: Any, interval_s: float = 0.1) -> None:
            self.calls = 0
            made.append(self)

        def start(self) -> bool:
            return True

        def has_reading(self, spec: str) -> bool:
            self.calls += 1
            return self.calls > 3

    monitor = counters.MeasurementMonitor([r"\Memory\Pages Input/sec", r"\GPU Engine(*)\Utilization Percentage"],
                                          sampler_factory=Fake)  # type: ignore[arg-type]
    cost = monitor.open(first_sample_timeout_s=1.0)
    assert cost < 0.5 and made[0].calls == 4
    assert monitor.collected == [r"\Memory\Pages Input/sec"]
    assert monitor.not_collected == [r"\GPU Engine(*)\Utilization Percentage"]  # stated, not silently dropped
