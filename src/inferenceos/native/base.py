"""Platform-neutral native vocabulary and provider surface.

This module is the shared boundary for platform-specific probes. It contains:

* the *raw* data shapes produced by native probes (``CimSnapshot``,
  ``VulkanDevice``, ``GlobalMemory``, ``CounterSample``, ``PowerStatus``,
  ``ProcessorPowerInfo``) — produced identically by every platform implementation;
* :class:`NativePlatform`, the minimal documented operation surface consumed by the
  hardware and telemetry layers.

The data shapes are intentionally the same as before; this module does not redesign
any contract. Platform implementations (e.g. :mod:`inferenceos.native.windows`)
provide the operations; pure logic (diagnosis, optimization, experiment, estimators,
measurement-validity, runtime safety, quality, rollback, accounting) does not depend
on this module and continues to consume platform-neutral contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

# --------------------------------------------------------------------------- #
# Raw data structures (shared shape across platforms)                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CimSnapshot:
    """Raw system-inventory snapshot (Windows: WMI/CIM).

    Kept as the common inventory shape; a non-Windows backend populates the same
    fields from its own inventory mechanism rather than inventing a new contract.
    """

    os: dict[str, Any] = field(default_factory=dict)
    processors: list[dict[str, Any]] = field(default_factory=list)
    video_controllers: list[dict[str, Any]] = field(default_factory=list)
    physical_memory: list[dict[str, Any]] = field(default_factory=list)
    disk_drives: list[dict[str, Any]] = field(default_factory=list)
    logical_disks: list[dict[str, Any]] = field(default_factory=list)
    physical_disks: list[dict[str, Any]] = field(default_factory=list)
    associations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class GlobalMemory:
    total_physical_bytes: int
    available_physical_bytes: int
    total_pagefile_bytes: int
    available_pagefile_bytes: int
    total_virtual_bytes: int
    available_virtual_bytes: int
    memory_load_percent: int


@dataclass(frozen=True)
class VulkanDevice:
    index: int
    name: str
    vendor_id: int
    device_id: int
    device_type: int
    api_version: str
    driver_version: int
    device_local_bytes: int
    # Windows LUID (``0x<high>:0x<low>``), captured from
    # ``VkPhysicalDeviceIDProperties.deviceLUID`` when the driver provides it.
    # ``None`` means the identity extension/loader did not expose a valid LUID; it is
    # never invented.
    luid: str | None = None
    device_node_mask: int | None = None


@dataclass(frozen=True)
class CounterSample:
    path: str
    instance: str | None
    value: float


@dataclass(frozen=True)
class PowerStatus:
    """Real external-power state (``None`` means indeterminate; never coerced to AC)."""

    ac_line: bool | None
    battery_present: bool | None
    battery_percent: int | None
    charging: bool | None


@dataclass(frozen=True)
class ProcessorPowerInfo:
    """Per-logical-processor clock state."""

    number: int
    current_mhz: int
    max_mhz: int
    limit_mhz: int


# --------------------------------------------------------------------------- #
# Provider surface                                                            #
# --------------------------------------------------------------------------- #
def backend_library_names(backend: str, *, platform: str) -> tuple[str, ...]:
    """Candidate dynamic-library names for a backend on a given platform.

    This is the single extension point for platform library naming. It makes
    *detection* portable; it does not claim that the backend is supported or usable.
    """
    if backend == "vulkan":
        if platform == "win32":
            return ("vulkan-1.dll",)
        if platform == "darwin":
            return ("libMoltenVK.dylib", "libvulkan.1.dylib")
        return ("libvulkan.so.1", "libvulkan.so")
    if backend == "cuda":
        if platform == "win32":
            return ("nvcuda.dll",)
        return ("libcuda.so.1", "libcuda.so")
    if backend == "rocm":
        if platform == "win32":
            return ("amdhip64.dll", "hiprtc.dll")
        return ("libamdhip64.so.6", "libamdhip64.so", "libhiprtc.so")
    return ()


class NativePlatform(Protocol):
    """The raw native operations consumed by the hardware/telemetry layers.

    A platform implementation is a module (or object) exposing exactly these
    operations. It is intentionally narrow: only operations that genuinely need
    OS-specific code live here.
    """

    def is_windows(self) -> bool:
        ...

    # System inventory ------------------------------------------------------
    def collect_cim_snapshot(self) -> CimSnapshot:
        ...

    def cpuid_features(self) -> dict[str, bool]:
        ...

    def os_cpu_features(self) -> dict[str, bool]:
        ...

    def cpuid_cache_info(self) -> list[dict[str, int]]:
        ...

    def global_memory_status(self) -> GlobalMemory | None:
        ...

    # GPU ------------------------------------------------------------------
    def vulkan_enumerate(self) -> list[VulkanDevice]:
        ...

    def library_present(self, name: str) -> bool:
        ...

    def normalize_luid(self, value: str) -> str:
        ...

    # Power / performance --------------------------------------------------
    def power_status(self) -> PowerStatus | None:
        ...

    def processor_power(self) -> list[ProcessorPowerInfo]:
        ...

    # Performance counters / parsing ---------------------------------------
    def sample_perf_counters(
        self, paths: Sequence[str], *, warmup_s: float = 1.0
    ) -> list[CounterSample]:
        ...

    def parse_wmi_datetime(self, value: Any) -> str | None:
        ...
