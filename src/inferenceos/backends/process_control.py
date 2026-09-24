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
    """Non-Windows fallback: recursive psutil termination of the process tree."""

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
        terminate_process_tree(pid)

    def close(self) -> None:
        return


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
        return ResourceSnapshot(
            pid=pid,
            exists=True,
            rss_bytes=int(getattr(info, "rss", 0) or 0),
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
