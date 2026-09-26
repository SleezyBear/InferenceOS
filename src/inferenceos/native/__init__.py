"""Platform-specific probes with an explicit selection seam.

The hardware and telemetry layers resolve the selected :class:`NativePlatform`
implementation through :func:`get_platform` instead of assuming Windows. The raw
data shapes and the provider surface live in :mod:`inferenceos.native.base`.

Currently only Windows is implemented. Linux and macOS are *recognized* but raise a
clear, explicit error rather than being silently treated as supported. The existing
Windows functions remain re-exported here for backwards compatibility.
"""

from __future__ import annotations

import sys
from typing import cast

from inferenceos.native import windows as _windows
from inferenceos.native.base import (
    CimSnapshot,
    CounterSample,
    GlobalMemory,
    NativePlatform,
    PowerStatus,
    ProcessorPowerInfo,
    VulkanDevice,
)
from inferenceos.native.windows import (
    collect_cim_snapshot,
    cpuid_cache_info,
    cpuid_features,
    format_device_luid,
    global_memory_status,
    is_windows,
    library_present,
    normalize_luid,
    os_cpu_features,
    parse_cim_snapshot,
    parse_wmi_datetime,
    power_status,
    processor_power,
    sample_perf_counters,
    vulkan_enumerate,
)

__all__ = [
    "CimSnapshot",
    "CounterSample",
    "GlobalMemory",
    "NativePlatform",
    "PowerStatus",
    "ProcessorPowerInfo",
    "VulkanDevice",
    "collect_cim_snapshot",
    "cpuid_cache_info",
    "cpuid_features",
    "format_device_luid",
    "get_platform",
    "global_memory_status",
    "is_windows",
    "library_present",
    "normalize_luid",
    "os_cpu_features",
    "parse_cim_snapshot",
    "parse_wmi_datetime",
    "power_status",
    "processor_power",
    "sample_perf_counters",
    "vulkan_enumerate",
]


def _platform_key() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    return "unsupported"


def get_platform(override: str | None = None) -> NativePlatform:
    """Return the native provider for this host.

    ``override`` selects a platform explicitly (for deterministic tests and future
    validation on another OS): ``"windows"``, ``"linux"``, or ``"macos"``. Only
    Windows is implemented today; Linux/macOS raise :class:`NotImplementedError` and
    an unknown value raises :class:`OSError` rather than silently pretending support.
    """
    key = (override or _platform_key()).lower()
    if key in ("windows", "win32"):
        return cast(NativePlatform, _windows)
    if key in ("linux", "macos", "darwin"):
        normalized = "macos" if key == "darwin" else key
        raise NotImplementedError(
            f"native platform {normalized!r} is recognized but not implemented yet"
        )
    raise OSError(
        f"unsupported native platform: {sys.platform!r}"
        + (f" (override={override!r})" if override is not None else "")
    )
