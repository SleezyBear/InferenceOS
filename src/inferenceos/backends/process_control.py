"""Platform-aware process control for the resident backend.

Windows: a Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` guarantees that
the resident server *and its descendants* are terminated when the job is closed or
terminated, so an unload/timeout cannot leave orphaned inference processes behind.

Other platforms: a psutil-based recursive termination fallback that keeps the same
interface (Linux development and tests are not broken).

This module also provides real process-identity and resource snapshots used to prove
residency and to detect PID reuse.
"""

from __future__ import annotations

import hashlib
import ctypes
import os
import signal
import sys
import time
from abc import ABC, abstractmethod
from ctypes import wintypes
from typing import Callable

from inferenceos.contracts.lifecycle import ProcessIdentity, ResourceSnapshot
from inferenceos.logging_config import get_logger

try:
    import psutil
except ImportError:  # pragma: no cover - hard dependency
    psutil = None  # type: ignore[assignment]

logger = get_logger("backends.process_control")

_IS_WINDOWS = sys.platform == "win32"


# --------------------------------------------------------------------------- #
# Job Object (Windows)                                                         #
# --------------------------------------------------------------------------- #
class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


class ProcessGuard(ABC):
    """Assigns a process to a containment group and can terminate the whole group."""

    @abstractmethod
    def assign(self, pid: int) -> bool:
        """Best-effort assignment; returns True when the process is contained."""

    @abstractmethod
    def terminate(self) -> None:
        """Terminate every process in the group."""

    @abstractmethod
    def close(self) -> None:
        """Release the containment handle (may itself terminate on Windows)."""

    @property
    def kind(self) -> str:
        return "process_guard"


class WindowsJobObject(ProcessGuard):
    """Windows Job Object wrapper with kill-on-close semantics."""

    def __init__(self) -> None:
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._handle: int | None = None
        self._assigned = False
        if not _IS_WINDOWS:  # pragma: no cover - guarded by factory
            raise OSError("Job Objects are Windows-only")
        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        ok = self._kernel32.SetInformationJobObject(
            wintypes.HANDLE(handle),
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            self._kernel32.CloseHandle(wintypes.HANDLE(handle))
            raise ctypes.WinError(ctypes.get_last_error())
        self._handle = int(handle)

    @property
    def kind(self) -> str:
        return "windows_job_object"

    def assign(self, pid: int) -> bool:
        if self._handle is None:
            return False
        process_handle = self._kernel32.OpenProcess(
            _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid
        )
        if not process_handle:
            logger.warning("OpenProcess for job assignment failed (pid=%s)", pid)
            return False
        try:
            ok = self._kernel32.AssignProcessToJobObject(
                wintypes.HANDLE(self._handle), wintypes.HANDLE(process_handle)
            )
            self._assigned = bool(ok)
            return bool(ok)
        finally:
            self._kernel32.CloseHandle(wintypes.HANDLE(process_handle))

    def terminate(self) -> None:
        if self._handle is None:
            return
        self._kernel32.TerminateJobObject(wintypes.HANDLE(self._handle), 1)

    def close(self) -> None:
        if self._handle is None:
            return
        self._kernel32.CloseHandle(wintypes.HANDLE(self._handle))
        self._handle = None


class PsutilProcessGuard(ProcessGuard):
    """Non-Windows containment: POSIX process group, then recursive psutil fallback.

    When the child was started in its own session/group (see
    :func:`subprocess_session_kwargs`), the whole group is signalled so descendants
    cannot be orphaned. If the process is not a group leader (or group termination is
    unavailable), the original recursive psutil tree termination is used.
    """

    def __init__(self, root_pid: int | None = None) -> None:
        self._root_pid = root_pid

    @property
    def kind(self) -> str:
        return "psutil_process_tree"

    def assign(self, pid: int) -> bool:
        self._root_pid = pid
        return True

    def terminate(self) -> None:
        pid = self._root_pid
        if pid is None or psutil is None:
            return
        if not _IS_WINDOWS and terminate_process_group(pid):
            return
        terminate_process_tree(pid)

    def close(self) -> None:
        return


def subprocess_session_kwargs(*, platform: str | None = None) -> dict[str, object]:
    """Popen kwargs placing a child in its own POSIX session/process group.

    Windows uses Job Objects instead, so this returns ``{}`` there. On Unix the new
    session/group is what :func:`terminate_process_group` reaps, preserving the
    no-orphan guarantee.
    """
    if (platform or sys.platform) == "win32":
        return {}
    return {"start_new_session": True}


def terminate_process_group(pid: int, *, timeout_s: float = 5.0) -> bool:
    """Terminate a POSIX process group led by ``pid`` (Windows: always ``False``)."""
    if _IS_WINDOWS or psutil is None:
        return False
    getpgid = getattr(os, "getpgid", None)
    killpg = getattr(os, "killpg", None)
    if getpgid is None or killpg is None:  # pragma: no cover - POSIX-only APIs
        return False
    try:
        pgid = getpgid(pid)
    except OSError:
        return False
    if pgid != pid:
        # Not a group leader we own; let the caller use tree termination.
        return False
    try:
        killpg(pgid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if not _process_group_alive(pgid):
            return True
        time.sleep(0.1)
    try:
        killpg(pgid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except OSError:
        pass
    time.sleep(0.1)
    return not _process_group_alive(pgid)


def _process_group_alive(pgid: int) -> bool:
    getpgid = getattr(os, "getpgid", None)
    if psutil is None or getpgid is None:  # pragma: no cover - POSIX-only API
        return False
    for process in psutil.process_iter(["pid"]):
        try:
            if getpgid(int(process.pid)) == pgid:
                return True
        except OSError:
            continue
    return False


def create_process_guard() -> ProcessGuard:
    """Create the strongest available containment guard for this platform."""
    if _IS_WINDOWS:
        try:
            return WindowsJobObject()
        except OSError as exc:  # pragma: no cover - environment dependent
            logger.warning("Job Object unavailable, using psutil fallback: %s", exc)
    return PsutilProcessGuard()


# --------------------------------------------------------------------------- #
# Identity / resource / termination                                            #
# --------------------------------------------------------------------------- #
def process_exists(pid: int) -> bool:
    if psutil is None:
        return False
    try:
        return psutil.pid_exists(pid)
    except Exception:  # pragma: no cover - defensive
        return False


def process_identity(pid: int) -> ProcessIdentity | None:
    """Return a PID-reuse-safe identity, or ``None`` if the process is gone."""
    if psutil is None:
        return None
    try:
        process = psutil.Process(pid)
        create_time = float(process.create_time())
        name = _safe(lambda: process.name())
        executable = _safe(lambda: process.exe())
        cmdline = process.cmdline()
        cmdline_hash = (
            hashlib.sha256(" ".join(cmdline).encode("utf-8", "replace")).hexdigest()[:16]
            if cmdline
            else None
        )
        return ProcessIdentity(
            pid=pid,
            create_time=create_time,
            name=name,
            executable=executable,
            cmdline_hash=cmdline_hash,
        )
    except Exception:  # noqa: BLE001 - psutil raises several error types
        return None


def identity_matches(identity: ProcessIdentity, *, tolerance_s: float = 0.01) -> bool:
    current = process_identity(identity.pid)
    if current is None:
        return False
    return abs(current.create_time - identity.create_time) <= tolerance_s


def resource_snapshot(pid: int | None) -> ResourceSnapshot:
    """Measure the process (if any) and the whole-system physical memory."""
    system_available: int | None = None
    system_total: int | None = None
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            system_available = int(vm.available)
            system_total = int(vm.total)
        except Exception:  # pragma: no cover - defensive
            pass
    if pid is None or psutil is None:
        return ResourceSnapshot(
            pid=pid,
            exists=False,
            system_available_bytes=system_available,
            system_total_bytes=system_total,
        )
    try:
        process = psutil.Process(pid)
        info = process.memory_info()
        # Windows exposes the authoritative peak working set (``peak_wset``); other
        # platforms do not, in which case peak tracking relies on sampling.
        peak = getattr(info, "peak_wset", None)
        return ResourceSnapshot(
            pid=pid,
            exists=True,
            rss_bytes=int(getattr(info, "rss", 0) or 0),
            peak_rss_bytes=int(peak) if peak else None,
            vms_bytes=int(getattr(info, "vms", 0) or 0),
            system_available_bytes=system_available,
            system_total_bytes=system_total,
        )
    except Exception:  # noqa: BLE001
        return ResourceSnapshot(
            pid=pid,
            exists=False,
            system_available_bytes=system_available,
            system_total_bytes=system_total,
        )


class PeakRssAccumulator:
    """Track true peak vs final RSS from a stream of real process memory samples.

    It accepts current RSS samples and (where the OS exposes it) an authoritative
    monotonic peak working set. The reported peak is the maximum of all of them; the
    final RSS is the last current sample. This makes ``peak != final`` representable
    rather than aliasing one to the other.

    It also derives the target's peak **process CPU utilisation** and the total
    **CPU-seconds** and **context-switch** deltas consumed over the observed window, from
    the same real psutil ``cpu_times`` / ``num_ctx_switches`` readings already used by the
    process-per-generation observer (``llama_cpp_toolchain._observe_rss``). The window is
    bounded by the first and last observation (the generation interval), so reload/startup
    CPU is not mixed in; a value is reported only once at least two readings exist, and it
    is ``None`` (unknown) otherwise — never fabricated.
    """

    def __init__(self) -> None:
        self._peak: int | None = None
        self._final: int | None = None
        self._sample_count = 0
        self._os_peak_seen = False
        self._peak_cpu_percent: float | None = None
        self._last_cpu_seconds: float | None = None
        self._last_cpu_at: float | None = None
        # Bounded CPU-utilisation / efficiency window (first and last readings).
        self._cpu_sample_count = 0
        self._first_cpu_seconds: float | None = None
        self._final_cpu_seconds: float | None = None
        self._first_ctx_voluntary: int | None = None
        self._final_ctx_voluntary: int | None = None
        self._first_ctx_involuntary: int | None = None
        self._final_ctx_involuntary: int | None = None

    def add(self, *, rss_bytes: int | None, os_peak_rss_bytes: int | None = None) -> None:
        if rss_bytes is not None:
            value = int(rss_bytes)
            self._peak = value if self._peak is None else max(self._peak, value)
            self._final = value
            self._sample_count += 1
        if os_peak_rss_bytes is not None:
            value = int(os_peak_rss_bytes)
            self._peak = value if self._peak is None else max(self._peak, value)
            self._os_peak_seen = True

    def observe(self, pid: int) -> None:
        """Take one real snapshot of ``pid`` and fold it into the accumulator."""
        snapshot = resource_snapshot(pid)
        if not snapshot.exists:
            return
        self.add(
            rss_bytes=snapshot.rss_bytes,
            os_peak_rss_bytes=snapshot.peak_rss_bytes,
        )
        self._observe_cpu(pid)

    def _observe_cpu(self, pid: int) -> None:
        """Record process CPU time and context switches at this instant.

        Uses the existing psutil mechanism; an unavailable reading is skipped (never
        fabricated), so the derived values stay ``None`` when the process cannot be read.
        """
        if psutil is None:
            return
        try:
            process = psutil.Process(pid)
            times = process.cpu_times()
            switches = process.num_ctx_switches()
        except Exception:  # noqa: BLE001 - a closed/gone process is not an error here
            return
        seconds = float(getattr(times, "user", 0.0) or 0.0) + float(
            getattr(times, "system", 0.0) or 0.0
        )
        voluntary = int(getattr(switches, "voluntary", 0) or 0)
        involuntary = int(getattr(switches, "involuntary", 0) or 0)
        now = time.perf_counter()
        if (
            self._last_cpu_seconds is not None
            and self._last_cpu_at is not None
            and now > self._last_cpu_at
        ):
            percent = max(
                (seconds - self._last_cpu_seconds) / (now - self._last_cpu_at) * 100.0,
                0.0,
            )
            self._peak_cpu_percent = (
                percent
                if self._peak_cpu_percent is None
                else max(self._peak_cpu_percent, percent)
            )
        self._last_cpu_seconds = seconds
        self._last_cpu_at = now
        if self._cpu_sample_count == 0:
            self._first_cpu_seconds = seconds
            self._first_ctx_voluntary = voluntary
            self._first_ctx_involuntary = involuntary
        self._cpu_sample_count += 1
        self._final_cpu_seconds = seconds
        self._final_ctx_voluntary = voluntary
        self._final_ctx_involuntary = involuntary

    @property
    def peak_rss_bytes(self) -> int | None:
        return self._peak

    @property
    def final_rss_bytes(self) -> int | None:
        return self._final

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def os_peak_available(self) -> bool:
        return self._os_peak_seen

    @property
    def peak_process_cpu_percent(self) -> float | None:
        return self._peak_cpu_percent

    @property
    def cpu_seconds(self) -> float | None:
        """Process CPU-seconds consumed over the bounded observation window."""
        if (
            self._cpu_sample_count < 2
            or self._first_cpu_seconds is None
            or self._final_cpu_seconds is None
        ):
            return None
        return max(self._final_cpu_seconds - self._first_cpu_seconds, 0.0)

    @property
    def context_switches_voluntary(self) -> int | None:
        if self._cpu_sample_count < 2 or self._first_ctx_voluntary is None or self._final_ctx_voluntary is None:
            return None
        return max(self._final_ctx_voluntary - self._first_ctx_voluntary, 0)

    @property
    def context_switches_involuntary(self) -> int | None:
        if (
            self._cpu_sample_count < 2
            or self._first_ctx_involuntary is None
            or self._final_ctx_involuntary is None
        ):
            return None
        return max(self._final_ctx_involuntary - self._first_ctx_involuntary, 0)

    @property
    def context_switches_total(self) -> int | None:
        voluntary = self.context_switches_voluntary
        involuntary = self.context_switches_involuntary
        if voluntary is None or involuntary is None:
            return None
        return voluntary + involuntary


def terminate_process_tree(pid: int, *, timeout_s: float = 5.0) -> bool:
    """Recursively terminate a process tree (fallback when no Job Object exists)."""
    if psutil is None:
        return False
    try:
        parent = psutil.Process(pid)
    except Exception:  # noqa: BLE001
        return False
    processes = _safe_list(parent.children(recursive=True))
    processes.append(parent)
    for process in processes:
        try:
            process.terminate()
        except Exception:  # noqa: BLE001
            pass
    _gone, alive = psutil.wait_procs(processes, timeout=timeout_s)
    for process in alive:
        try:
            process.kill()
        except Exception:  # noqa: BLE001
            pass
    _, still_alive = psutil.wait_procs(alive, timeout=timeout_s)
    return not still_alive


def wait_process_exit(pid: int, *, timeout_s: float = 10.0, poll_s: float = 0.1) -> bool:
    if psutil is None:
        return False
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if not process_exists(pid):
            return True
        time.sleep(poll_s)
    return not process_exists(pid)


def _safe(getter: Callable[[], object]) -> str | None:
    try:
        value = getter()
        return str(value) if value else None
    except Exception:  # noqa: BLE001
        return None


def _safe_list(value: object) -> list:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []
