"""Read-only, minimally-invasive external process observation.

Only ordinary psutil counters are collected (name, pid, start time, executable path,
owner, CPU time/percent, RSS, private bytes if cheaply available, page faults, I/O
counters, thread count). This module never reads process memory, command lines,
environment variables, credentials, application contents, or user data.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from inferenceos.config import Settings
from inferenceos.contracts.enums import ExternalProcessAccess, ExternalScope
from inferenceos.contracts.external import (
    ExternalObservationReport,
    ExternalProcessIdentity,
    ExternalProcessObservation,
)
from inferenceos.external.identity import identity_from_process

try:  # pragma: no cover - psutil is a hard dependency in practice
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

_PRIVACY_NOTES = [
    "read-only psutil counters only",
    "no process memory contents, command lines, environment, credentials, or app data",
]


@dataclass(frozen=True)
class _DeltaCpu:
    cpu_seconds: float
    wall: float


class ProcessObserver(ABC):
    """Abstract read-only external process observer."""

    @abstractmethod
    def observe(
        self,
        *,
        target_pid: int | None = None,
        own_pid: int | None = None,
        max_processes: int = 40,
        scope: ExternalScope = ExternalScope.EXTERNAL_PROCESS,
    ) -> ExternalObservationReport:
        ...


class WindowsProcessObserver(ProcessObserver):
    """Real process observer over the live Windows process table."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._last_cpu: dict[int, _DeltaCpu] = {}

    def observe(
        self,
        *,
        target_pid: int | None = None,
        own_pid: int | None = None,
        max_processes: int = 40,
        scope: ExternalScope = ExternalScope.EXTERNAL_PROCESS,
    ) -> ExternalObservationReport:
        if psutil is None:  # pragma: no cover - hard dependency
            raise RuntimeError("psutil is required for external process observation")

        now = time.monotonic()
        virtual = psutil.virtual_memory()
        total = int(virtual.total)
        available = int(virtual.available)
        cpu_total = float(psutil.cpu_percent(interval=None))
        logical = psutil.cpu_count(logical=True)

        observations: list[ExternalProcessObservation] = []
        seen: set[int] = set()
        for process in psutil.process_iter():
            try:
                observation = self._observe_process(process, now, target_pid, own_pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
            if observation is None:
                continue
            seen.add(observation.identity.pid)
            observations.append(observation)

        # Drop stale delta-tracking entries.
        for pid in [pid for pid in self._last_cpu if pid not in seen]:
            del self._last_cpu[pid]

        observations.sort(key=lambda item: item.rss_bytes or 0, reverse=True)
        bounded = observations[: max(max_processes, 1)]
        if target_pid is not None and not any(
            item.identity.pid == target_pid for item in bounded
        ):
            target = next((item for item in observations if item.identity.pid == target_pid), None)
            if target is not None:
                bounded.append(target)

        memory_pressure = self._memory_pressure(total, available)
        return ExternalObservationReport(
            report_id=f"extobs-{uuid.uuid4().hex[:10]}",
            captured_at=datetime.now(timezone.utc),
            scope=scope,
            target_pid=target_pid,
            own_pid=own_pid,
            logical_cores=logical,
            total_ram_bytes=total,
            available_ram_bytes=available,
            memory_pressure=memory_pressure,
            cpu_percent_total=cpu_total,
            processes=bounded,
            limitations=[
                "GPU counters and hard-page-fault perf counters are not sampled at runtime",
                "private/commit bytes may be unavailable without deeper process inspection",
            ],
            privacy_notes=list(_PRIVACY_NOTES),
        )

    def _observe_process(
        self,
        process: "psutil.Process",
        now: float,
        target_pid: int | None,
        own_pid: int | None,
    ) -> ExternalProcessObservation | None:
        try:
            pid = int(process.pid)
        except (psutil.NoSuchProcess, OSError):
            return None
        is_target = pid == target_pid
        identity = identity_from_process(process, is_target=is_target)
        measurable: list[str] = []
        cpu_percent: float | None = None
        cpu_time_s: float | None = None
        rss_bytes: int | None = None
        page_faults_per_sec: float | None = None
        io_read: int | None = None
        io_write: int | None = None
        threads: int | None = None
        access = identity.access
        inaccessible_reason: str | None = None

        try:
            times = process.cpu_times()
            cpu_sec = float(times.user) + float(times.system)
            previous = self._last_cpu.get(pid)
            if previous is not None and now > previous.wall:
                cpu_percent = max(0.0, (cpu_sec - previous.cpu_seconds) / (now - previous.wall) * 100.0)
            self._last_cpu[pid] = _DeltaCpu(cpu_seconds=cpu_sec, wall=now)
            cpu_time_s = cpu_sec
            measurable.append("cpu_time_s")
            if cpu_percent is not None:
                measurable.append("cpu_percent")
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            access = ExternalProcessAccess.PARTIAL
            inaccessible_reason = "cpu times not accessible"

        try:
            memory = process.memory_info()
            rss_bytes = int(memory.rss)
            measurable.append("rss_bytes")
            faults = getattr(memory, "num_page_faults", None)
            if faults is not None:
                page_faults_per_sec = float(faults)
                measurable.append("page_faults")
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            access = ExternalProcessAccess.PARTIAL
            inaccessible_reason = inaccessible_reason or "memory info not accessible"

        try:
            counters = process.io_counters()
            io_read = int(counters.read_bytes)
            io_write = int(counters.write_bytes)
            measurable.append("io_bytes")
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, AttributeError):
            pass

        try:
            threads = int(process.num_threads())
            measurable.append("num_threads")
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

        if pid == own_pid and pid != target_pid:
            # H6: the optimizer's own process is never an external candidate. It is
            # only retained when it is explicitly the monitored target (self-target
            # monitoring is a separate, guarded arrangement). Children are not
            # excluded here: the resident server is a legitimate child/target.
            return None
        return ExternalProcessObservation(
            observation_id=f"extproc-{uuid.uuid4().hex[:10]}",
            identity=identity,
            captured_at=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            cpu_time_s=cpu_time_s,
            rss_bytes=rss_bytes,
            page_faults_per_sec=page_faults_per_sec,
            io_read_bytes=io_read,
            io_write_bytes=io_write,
            num_threads=threads,
            measurable=measurable,
            access=access,
            inaccessible_reason=inaccessible_reason,
        )

    def _memory_pressure(self, total: int, available: int) -> float:
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - available / total))
