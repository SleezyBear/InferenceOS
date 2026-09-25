"""Controlled fixtures for ``verify-phase11r`` (verification-only).

The real acceptance scenarios use the real Windows process observer and real controlled
processes. Deterministic scenario doubles (scripted observer, synthetic reports) are
shipped here explicitly labelled as verification-only; the Phase 11R analogue of
``reasoning.providers.fake``. Nothing here is used by ``inferenceos external`` or any
production path.
"""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from inferenceos.contracts.enums import ExternalProcessAccess, ExternalScope
from inferenceos.contracts.external import (
    ExternalObservationReport,
    ExternalProcessIdentity,
    ExternalProcessObservation,
)
from inferenceos.external.identity import build_identity
from inferenceos.external.observer import ProcessObserver

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


def make_identity(
    *,
    pid: int,
    name: str = "external.exe",
    is_target: bool = False,
    create_time: float = 1000.0,
    access: ExternalProcessAccess = ExternalProcessAccess.READABLE,
    executable_path: str | None = None,
) -> ExternalProcessIdentity:
    return build_identity(
        pid=pid,
        name=name,
        executable_path=executable_path if executable_path is not None else sys.executable,
        owner="test-user",
        create_time_epoch=create_time + pid,
        access=access,
        is_target=is_target,
    )


def make_process_observation(
    *,
    pid: int,
    name: str = "external.exe",
    rss_bytes: int | None = 512 * 1024 * 1024,
    cpu_percent: float | None = 5.0,
    is_target: bool = False,
    access: ExternalProcessAccess = ExternalProcessAccess.READABLE,
    create_time: float = 1000.0,
) -> ExternalProcessObservation:
    identity = make_identity(
        pid=pid, name=name, is_target=is_target, create_time=create_time, access=access
    )
    measurable: list[str] = []
    if rss_bytes is not None:
        measurable.append("rss_bytes")
    if cpu_percent is not None:
        measurable.append("cpu_percent")
    return ExternalProcessObservation(
        observation_id=f"extproc-{uuid.uuid4().hex[:8]}",
        identity=identity,
        captured_at=_NOW,
        cpu_percent=cpu_percent,
        cpu_time_s=(cpu_percent or 0.0) / 100.0,
        rss_bytes=rss_bytes,
        measurable=measurable,
        access=access,
    )


def make_report(
    *,
    processes: list[ExternalProcessObservation],
    target_pid: int | None = None,
    memory_pressure: float = 0.5,
    total_ram_bytes: int = 16 * 1024**3,
    available_ram_bytes: int = 8 * 1024**3,
) -> ExternalObservationReport:
    return ExternalObservationReport(
        report_id=f"extobs-{uuid.uuid4().hex[:8]}",
        captured_at=_NOW,
        scope=ExternalScope.EXTERNAL_PROCESS,
        target_pid=target_pid,
        total_ram_bytes=total_ram_bytes,
        available_ram_bytes=available_ram_bytes,
        memory_pressure=memory_pressure,
        logical_cores=8,
        processes=processes,
        privacy_notes=["read-only psutil counters only"],
    )


class ScriptedProcessObserver(ProcessObserver):
    def __init__(self, report: ExternalObservationReport) -> None:
        self._report = report
        self.calls = 0

    def observe(
        self,
        *,
        target_pid: int | None = None,
        own_pid: int | None = None,
        max_processes: int = 40,
        scope: ExternalScope = ExternalScope.EXTERNAL_PROCESS,
    ) -> ExternalObservationReport:
        self.calls += 1
        return self._report


def _python_executable() -> str:
    """The real interpreter path (venv Scripts\\python.exe is a redirector on Windows)."""
    try:
        import psutil

        return psutil.Process().exe()
    except Exception:  # noqa: BLE001 - fall back to sys.executable
        return sys.executable


@contextmanager
def launch_sleeper(seconds: float = 60.0) -> Iterator[subprocess.Popen[bytes]]:
    proc = subprocess.Popen([_python_executable(), "-c", f"import time; time.sleep({seconds})\n"])
    try:
        _wait_for_process(proc.pid)
        yield proc
    finally:
        _terminate(proc)


@contextmanager
def launch_cpu_spinner(seconds: float = 60.0) -> Iterator[subprocess.Popen[bytes]]:
    code = (
        "import time\n"
        f"end = time.time() + {seconds}\n"
        "x = 0\n"
        "while time.time() < end:\n"
        "    x = (x * 1103515245 + 12345) & 0x7FFFFFFF\n"
    )
    proc = subprocess.Popen([_python_executable(), "-c", code])
    try:
        _wait_for_process(proc.pid)
        yield proc
    finally:
        _terminate(proc)


@contextmanager
def launch_memory_holder(hold_bytes: int = 2 * 1024**3) -> Iterator[subprocess.Popen[bytes]]:
    code = (
        "import sys, time\n"
        f"size = int({hold_bytes})\n"
        "buf = bytearray(size)\n"
        "for i in range(0, size, 4096):\n"
        "    buf[i] = 1\n"
        "print('READY', flush=True)\n"
        "time.sleep(600)\n"
    )
    proc = subprocess.Popen(
        [_python_executable(), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=False,
    )
    try:
        if proc.stdout is not None:
            proc.stdout.readline()
        _wait_for_process(proc.pid)
        yield proc
    finally:
        _terminate(proc)


def _wait_for_process(pid: int, timeout_s: float = 10.0) -> None:
    try:
        import psutil
    except ImportError:  # pragma: no cover - hard dependency
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            psutil.Process(pid)
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.05)


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            proc.kill()


__all__ = [
    "FakeClock",
    "ScriptedProcessObserver",
    "launch_cpu_spinner",
    "launch_memory_holder",
    "launch_sleeper",
    "make_identity",
    "make_process_observation",
    "make_report",
]
