"""Platform-specific probes.

All Windows-native mechanisms live under this package so the rest of
InferenceOS consumes a host-independent contract instead of OS details.
"""

from __future__ import annotations

from inferenceos.native.windows import (
    CimSnapshot,
    CounterSample,
    GlobalMemory,
    VulkanDevice,
    collect_cim_snapshot,
    cpuid_cache_info,
    cpuid_features,
    global_memory_status,
    is_windows,
    library_present,
    os_cpu_features,
    parse_cim_snapshot,
    parse_wmi_datetime,
    sample_perf_counters,
    vulkan_enumerate,
)

__all__ = [
    "CimSnapshot",
    "CounterSample",
    "GlobalMemory",
    "VulkanDevice",
    "collect_cim_snapshot",
    "cpuid_cache_info",
    "cpuid_features",
    "global_memory_status",
    "is_windows",
    "library_present",
    "os_cpu_features",
    "parse_cim_snapshot",
    "parse_wmi_datetime",
    "sample_perf_counters",
    "vulkan_enumerate",
]
