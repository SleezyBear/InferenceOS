"""Windows-native probes, isolated behind this module.

Everything platform-specific for Phase 1 lives here so the rest of the
application (contracts, hardware profilers, telemetry) stays host-independent and
testable with synthetic fixtures. The functions here return *raw* values; the
normalization into contracts happens in :mod:`inferenceos.hardware`.

Mechanisms used (all real, standard-user, non-destructive):

* WMI / CIM (``Get-CimInstance``) for identity and inventory.
* CPUID executed through a tiny in-memory machine-code thunk (dependency-free
  ``cpuid`` equivalent), cross-checked against ``IsProcessorFeaturePresent``.
* ``GlobalMemoryStatusEx`` via ctypes for physical/commit memory.
* The Vulkan loader (``vulkan-1.dll``) via ctypes for real device enumeration.
* ``Get-Counter`` for live OS performance counters (page faults, GPU engines).
* Loader presence checks for CUDA / ROCm runtimes.

No destructive operation is performed and no large files are written.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Sequence, cast

from inferenceos.logging_config import get_logger
from inferenceos.native.base import (
    CimSnapshot,
    CounterSample,
    GlobalMemory,
    PowerStatus,
    ProcessorPowerInfo,
    VulkanDevice,
)

logger = get_logger("native.windows")

_IS_WINDOWS = sys.platform == "win32"
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_PS_TIMEOUT_S = 90.0


# --------------------------------------------------------------------------- #
# Raw data structures                                                         #
# --------------------------------------------------------------------------- #
# ``CimSnapshot``/``GlobalMemory``/``VulkanDevice``/``CounterSample`` are shared raw
# shapes defined in ``native.base`` and re-exported here for backwards compatibility.


def format_device_luid(low_part: int, high_part: int) -> str:
    """Format a Windows LUID as ``0x<high>:0x<low>`` (matching the PDH parser)."""
    return f"0x{high_part & 0xFFFFFFFF:08x}:0x{low_part & 0xFFFFFFFF:08x}"


def normalize_luid(value: str) -> str:
    """Canonicalize a LUID string for comparison (case/``0x``-insensitive).

    The Windows PDH instance parser yields ``"00000000:00009383"`` while Vulkan
    enumeration yields ``"0x00000000:0x00009383"``; both normalize to the latter's
    payload so identity comparison is prefix-insensitive.
    """
    return value.strip().lower().replace("0x", "")


# --------------------------------------------------------------------------- #
# Process / PowerShell plumbing                                               #
# --------------------------------------------------------------------------- #
def is_windows() -> bool:
    return _IS_WINDOWS


def run_powershell(script: str, *, timeout_s: float = _PS_TIMEOUT_S) -> str:
    """Run a PowerShell script and return stripped stdout (empty on failure)."""
    if not _IS_WINDOWS:
        return ""
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        logger.warning("PowerShell probe failed: %s", exc)
        return ""
    if proc.returncode != 0 and not proc.stdout.strip():
        logger.warning("PowerShell probe exit %s: %s", proc.returncode, proc.stderr.strip()[:300])
    return proc.stdout.strip()


# --------------------------------------------------------------------------- #
# WMI / CIM inventory                                                         #
# --------------------------------------------------------------------------- #
_CIM_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$out = [ordered]@{}
$out.OS = @(Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory)
$out.Processor = @(Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed,CurrentClockSpeed,L2CacheSize,L3CacheSize,Architecture,ProcessorId,Description)
$out.VideoController = @(Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate,AdapterRAM,PNPDeviceID,VideoProcessor,Status,AdapterCompatibility)
$out.PhysicalMemory = @(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,Manufacturer,DeviceLocator,BankLabel)
$out.DiskDrive = @(Get-CimInstance Win32_DiskDrive | Select-Object Model,MediaType,InterfaceType,Size,Index,PNPDeviceID)
$out.LogicalDisk = @(Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID,DriveType,Size,FreeSpace,FileSystem,VolumeName)
$out.PhysicalDisk = @(Get-CimInstance -Namespace 'root\Microsoft\Windows\Storage' -ClassName MSFT_PhysicalDisk | Select-Object FriendlyName,MediaType,BusType,Size,DeviceId)
$assoc = @()
foreach ($disk in @(Get-CimInstance Win32_DiskDrive)) {
    foreach ($part in @(Get-CimAssociatedInstance -InputObject $disk -Association Win32_DiskDriveToDiskPartition)) {
        foreach ($ld in @(Get-CimAssociatedInstance -InputObject $part -Association Win32_LogicalDiskToPartition)) {
            $assoc += [pscustomobject]@{ DiskIndex = [int]$disk.Index; Partition = $part.DeviceID; Letter = $ld.DeviceID }
        }
    }
}
$out.Association = @($assoc)
$out | ConvertTo-Json -Depth 5 -Compress
"""


def _as_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [cast(dict[str, Any], value)]
    if isinstance(value, list):
        return [cast(dict[str, Any], item) for item in value if isinstance(item, dict)]
    return []


def parse_cim_snapshot(payload: str) -> CimSnapshot:
    """Parse the JSON emitted by ``_CIM_SCRIPT`` into a :class:`CimSnapshot`.

    Missing or malformed sections degrade to empty lists rather than raising, so
    an unsupported management class never breaks characterization.
    """
    if not payload:
        return CimSnapshot()
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse CIM JSON: %s", exc)
        return CimSnapshot()
    if not isinstance(raw, dict):
        return CimSnapshot()
    data = cast(dict[str, Any], raw)
    os_value = data.get("OS")
    if isinstance(os_value, list):
        os_dict: dict[str, Any] = cast(dict[str, Any], os_value[0]) if os_value else {}
    elif isinstance(os_value, dict):
        os_dict = cast(dict[str, Any], os_value)
    else:
        os_dict = {}
    return CimSnapshot(
        os=os_dict,
        processors=_as_list(data.get("Processor")),
        video_controllers=_as_list(data.get("VideoController")),
        physical_memory=_as_list(data.get("PhysicalMemory")),
        disk_drives=_as_list(data.get("DiskDrive")),
        logical_disks=_as_list(data.get("LogicalDisk")),
        physical_disks=_as_list(data.get("PhysicalDisk")),
        associations=_as_list(data.get("Association")),
    )


def collect_cim_snapshot() -> CimSnapshot:
    return parse_cim_snapshot(run_powershell(_CIM_SCRIPT))


# --------------------------------------------------------------------------- #
# CPU: CPUID thunk + OS feature API                                           #
# --------------------------------------------------------------------------- #
# push rbx; mov eax,ecx; mov ecx,edx; cpuid;
# mov [r8],eax; mov [r8+4],ebx; mov [r8+8],ecx; mov [r8+12],edx; pop rbx; ret
_CPUID_CODE = bytes(
    [
        0x53,
        0x89,
        0xC8,
        0x89,
        0xD1,
        0x0F,
        0xA2,
        0x41,
        0x89,
        0x00,
        0x41,
        0x89,
        0x58,
        0x04,
        0x41,
        0x89,
        0x48,
        0x08,
        0x41,
        0x89,
        0x50,
        0x0C,
        0x5B,
        0xC3,
    ]
)

_MEM_COMMIT = 0x1000
_MEM_RESERVE = 0x2000
_MEM_RELEASE = 0x8000
_PAGE_EXECUTE_READWRITE = 0x40


class _CpuidThunk:
    """Executes the x86-64 ``CPUID`` instruction from Python without a compiler."""

    def __init__(self) -> None:
        self._addr: int = 0
        self._func: Any = None
        self._available = False
        if not _IS_WINDOWS:
            return
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.VirtualAlloc.restype = ctypes.c_void_p
            kernel32.VirtualAlloc.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint32,
                ctypes.c_uint32,
            ]
            kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
            addr = kernel32.VirtualAlloc(
                None, len(_CPUID_CODE), _MEM_COMMIT | _MEM_RESERVE, _PAGE_EXECUTE_READWRITE
            )
            if not addr:
                return
            ctypes.memmove(addr, _CPUID_CODE, len(_CPUID_CODE))
            proto = ctypes.CFUNCTYPE(
                None,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
            )
            self._addr = int(addr)
            self._func = proto(addr)
            self._available = True
        except (OSError, ValueError) as exc:  # pragma: no cover - environment dependent
            logger.warning("CPUID thunk unavailable: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def leaf(self, leaf_id: int, subleaf: int = 0) -> tuple[int, int, int, int]:
        if not self._available:
            raise RuntimeError("CPUID thunk unavailable")
        regs = (ctypes.c_uint32 * 4)()
        self._func(ctypes.c_uint32(leaf_id), ctypes.c_uint32(subleaf), regs)
        return int(regs[0]), int(regs[1]), int(regs[2]), int(regs[3])

    def close(self) -> None:
        if self._addr and _IS_WINDOWS:
            try:
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
                kernel32.VirtualFree(ctypes.c_void_p(self._addr), 0, _MEM_RELEASE)
            except OSError:  # pragma: no cover
                pass
        self._addr = 0
        self._func = None
        self._available = False


_THUNK: _CpuidThunk | None = None


def _cpuid_thunk() -> _CpuidThunk:
    global _THUNK
    if _THUNK is None:
        _THUNK = _CpuidThunk()
    return _THUNK


def cpuid_features() -> dict[str, bool]:
    """Return raw CPU feature flags from CPUID (empty if the thunk is unavailable)."""
    thunk = _cpuid_thunk()
    if not thunk.available:
        return {}
    try:
        _max_leaf, _b, _c, _d = thunk.leaf(0, 0)
        _eax1, _ebx1, ecx1, edx1 = thunk.leaf(1, 0)
        _eax7, ebx7, ecx7, _edx7 = thunk.leaf(7, 0)
    except (RuntimeError, OSError) as exc:  # pragma: no cover - execution dependent
        logger.warning("CPUID execution failed: %s", exc)
        return {}
    return {
        "mmx": bool(edx1 & (1 << 23)),
        "sse": bool(edx1 & (1 << 25)),
        "sse2": bool(edx1 & (1 << 26)),
        "sse3": bool(ecx1 & (1 << 0)),
        "ssse3": bool(ecx1 & (1 << 9)),
        "sse4_1": bool(ecx1 & (1 << 19)),
        "sse4_2": bool(ecx1 & (1 << 20)),
        "popcnt": bool(ecx1 & (1 << 23)),
        "aes": bool(ecx1 & (1 << 25)),
        "osxsave": bool(ecx1 & (1 << 27)),
        "avx": bool(ecx1 & (1 << 28)),
        "fma": bool(ecx1 & (1 << 12)),
        "f16c": bool(ecx1 & (1 << 29)),
        "bmi1": bool(ebx7 & (1 << 3)),
        "avx2": bool(ebx7 & (1 << 5)),
        "bmi2": bool(ebx7 & (1 << 8)),
        "avx512f": bool(ebx7 & (1 << 16)),
        "avx512dq": bool(ebx7 & (1 << 17)),
        "avx512bw": bool(ebx7 & (1 << 30)),
        "avx512vl": bool(ebx7 & (1 << 31)),
        "avx512vnni": bool(ecx7 & (1 << 11)),
    }


# IsProcessorFeaturePresent ids (winnt.h).
_OS_FEATURE_IDS: dict[str, int] = {
    "mmx": 3,
    "sse": 6,
    "sse2": 10,
    "sse3": 13,
    "ssse3": 36,
    "sse4_1": 37,
    "sse4_2": 38,
    "avx": 39,
    "avx2": 40,
    "avx512f": 41,
}


def os_cpu_features() -> dict[str, bool]:
    """Return CPU features reported by the OS (``IsProcessorFeaturePresent``)."""
    if not _IS_WINDOWS:
        return {}
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.IsProcessorFeaturePresent.argtypes = [ctypes.c_uint32]
        kernel32.IsProcessorFeaturePresent.restype = ctypes.c_int
    except OSError:  # pragma: no cover
        return {}
    result: dict[str, bool] = {}
    for name, feature_id in _OS_FEATURE_IDS.items():
        try:
            result[name] = bool(kernel32.IsProcessorFeaturePresent(feature_id))
        except OSError:  # pragma: no cover
            result[name] = False
    return result


def cpuid_cache_info() -> list[dict[str, int]]:
    """Return deterministic cache descriptors from CPUID leaf 4.

    Each entry has ``level``, ``type`` (1=data, 2=instruction, 3=unified) and
    ``size_bytes``. Empty when the thunk is unavailable or leaf 4 is unsupported.
    """
    thunk = _cpuid_thunk()
    if not thunk.available:
        return []
    _max_leaf, _b, _c, _d = thunk.leaf(0, 0)
    _eax1, _ebx1, _ecx1, _edx1 = thunk.leaf(1, 0)
    caches: list[dict[str, int]] = []
    try:
        for subleaf in range(8):
            eax, ebx, ecx, _edx = thunk.leaf(4, subleaf)
            cache_type = eax & 0x1F
            if cache_type == 0:
                break
            level = (eax >> 5) & 0x7
            line_size = (ebx & 0xFFF) + 1
            partitions = ((ebx >> 12) & 0x3FF) + 1
            ways = ((ebx >> 22) & 0x3FF) + 1
            sets = (ecx & 0xFFFFFFFF) + 1
            size = line_size * partitions * ways * sets
            caches.append({"level": level, "type": cache_type, "size_bytes": size})
    except (RuntimeError, OSError):  # pragma: no cover - execution dependent
        return caches
    return caches


# --------------------------------------------------------------------------- #
# Memory: GlobalMemoryStatusEx                                                #
# --------------------------------------------------------------------------- #
class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def global_memory_status() -> GlobalMemory | None:
    """Real physical/commit memory via the Win32 ``GlobalMemoryStatusEx`` API."""
    if not _IS_WINDOWS:
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            logger.warning("GlobalMemoryStatusEx failed")
            return None
    except OSError as exc:  # pragma: no cover
        logger.warning("GlobalMemoryStatusEx unavailable: %s", exc)
        return None
    return GlobalMemory(
        total_physical_bytes=int(status.ullTotalPhys),
        available_physical_bytes=int(status.ullAvailPhys),
        total_pagefile_bytes=int(status.ullTotalPageFile),
        available_pagefile_bytes=int(status.ullAvailPageFile),
        total_virtual_bytes=int(status.ullTotalVirtual),
        available_virtual_bytes=int(status.ullAvailVirtual),
        memory_load_percent=int(status.dwMemoryLoad),
    )


# --------------------------------------------------------------------------- #
# GPU: Vulkan loader enumeration + runtime presence                           #
# --------------------------------------------------------------------------- #
class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


class _VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_int),
        ("deviceName", ctypes.c_char * 256),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("_reserved", ctypes.c_uint8 * 2048),
    ]


class _VkMemoryType(ctypes.Structure):
    _fields_ = [("propertyFlags", ctypes.c_uint32), ("heapIndex", ctypes.c_uint32)]


class _VkMemoryHeap(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("flags", ctypes.c_uint32)]


class _VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", ctypes.c_uint32),
        ("memoryTypes", _VkMemoryType * 32),
        ("memoryHeapCount", ctypes.c_uint32),
        ("memoryHeaps", _VkMemoryHeap * 16),
    ]


_VK_SUCCESS = 0
_VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 = 1000059001
_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES = 1000071004
_VK_API_VERSION_1_0 = (1 << 22) | (0 << 12) | 0
# Vulkan 1.1 is required for the loader/driver to populate
# ``VkPhysicalDeviceIDProperties`` (LUID). A 1.0 instance leaves it zeroed.
_VK_API_VERSION_1_1 = (1 << 22) | (1 << 12) | 0
_VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x00000001


class _VkPhysicalDeviceIDProperties(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("deviceUUID", ctypes.c_uint8 * 16),
        ("driverUUID", ctypes.c_uint8 * 16),
        ("deviceLUID", ctypes.c_uint8 * 8),
        ("deviceNodeMask", ctypes.c_uint32),
        ("deviceLUIDValid", ctypes.c_int),  # VkBool32
    ]


class _VkPhysicalDeviceProperties2(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("properties", _VkPhysicalDeviceProperties),
    ]


def _decode_vulkan_version(version: int) -> str:
    major = (version >> 22) & 0x7F
    minor = (version >> 12) & 0x3FF
    patch = version & 0xFFF
    return f"{major}.{minor}.{patch}"


def vulkan_enumerate() -> list[VulkanDevice]:
    """Enumerate physical devices through the real Vulkan loader.

    Returns an empty list if the loader is absent or no device is present; this is
    how "Vulkan detected but no usable device" is represented (never assumed).
    """
    if not _IS_WINDOWS:
        return []
    try:
        vk = ctypes.WinDLL("vulkan-1.dll", use_last_error=True)
    except OSError as exc:
        logger.info("Vulkan loader not available: %s", exc)
        return []

    try:
        vk.vkCreateInstance.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        vk.vkCreateInstance.restype = ctypes.c_int
        vk.vkEnumeratePhysicalDevices.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_void_p,
        ]
        vk.vkEnumeratePhysicalDevices.restype = ctypes.c_int
        vk.vkGetPhysicalDeviceProperties.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_VkPhysicalDeviceProperties),
        ]
        vk.vkGetPhysicalDeviceMemoryProperties.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_VkPhysicalDeviceMemoryProperties),
        ]
        vk.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        # Optional (Vulkan 1.1+): physical-device identity (LUID) for real
        # telemetry↔device correlation. Absent symbols/structs degrade gracefully.
        properties2 = getattr(vk, "vkGetPhysicalDeviceProperties2", None)
        if properties2 is not None:
            properties2.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_VkPhysicalDeviceProperties2),
            ]
            properties2.restype = None

        # Prefer a 1.1 instance so ``VkPhysicalDeviceIDProperties`` (LUID) is
        # populated; fall back to 1.0 (device enumeration only, no LUID) when the
        # driver/loader cannot create a 1.1 instance.
        instance = ctypes.c_void_p()
        created = False
        for api_version in (_VK_API_VERSION_1_1, _VK_API_VERSION_1_0):
            app_info = _VkApplicationInfo(
                sType=_VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pApplicationName=b"InferenceOS",
                apiVersion=api_version,
            )
            create_info = _VkInstanceCreateInfo(
                sType=_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=ctypes.pointer(app_info),
            )
            instance = ctypes.c_void_p()
            if (
                vk.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))
                == _VK_SUCCESS
            ):
                created = True
                break
        if not created:
            logger.warning("vkCreateInstance failed")
            return []

        try:
            count = ctypes.c_uint32(0)
            if (
                vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
                != _VK_SUCCESS
                or count.value == 0
            ):
                return []
            handles = (ctypes.c_void_p * count.value)()
            if (
                vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles)
                != _VK_SUCCESS
            ):
                return []
            devices: list[VulkanDevice] = []
            for index, handle in enumerate(handles):
                props = _VkPhysicalDeviceProperties()
                vk.vkGetPhysicalDeviceProperties(handle, ctypes.byref(props))
                mem = _VkPhysicalDeviceMemoryProperties()
                vk.vkGetPhysicalDeviceMemoryProperties(handle, ctypes.byref(mem))
                local_bytes = sum(
                    int(mem.memoryHeaps[i].size)
                    for i in range(mem.memoryHeapCount)
                    if int(mem.memoryHeaps[i].flags) & _VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
                )
                luid: str | None = None
                node_mask: int | None = None
                if properties2 is not None:
                    id_props = _VkPhysicalDeviceIDProperties(
                        sType=_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES
                    )
                    props2 = _VkPhysicalDeviceProperties2(
                        sType=_VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                        pNext=ctypes.cast(ctypes.byref(id_props), ctypes.c_void_p),
                    )
                    try:
                        properties2(handle, ctypes.byref(props2))
                    except (OSError, ValueError):  # pragma: no cover - driver dependent
                        properties2 = None
                    else:
                        if int(id_props.deviceLUIDValid):
                            luid_bytes = bytes(id_props.deviceLUID)
                            low = int.from_bytes(luid_bytes[0:4], "little")
                            high = int.from_bytes(luid_bytes[4:8], "little")
                            luid = format_device_luid(low, high)
                            node_mask = int(id_props.deviceNodeMask)
                name = bytes(props.deviceName).split(b"\x00", 1)[0].decode("utf-8", "replace")
                devices.append(
                    VulkanDevice(
                        index=index,
                        name=name,
                        vendor_id=int(props.vendorID),
                        device_id=int(props.deviceID),
                        device_type=int(props.deviceType),
                        api_version=_decode_vulkan_version(int(props.apiVersion)),
                        driver_version=int(props.driverVersion),
                        device_local_bytes=local_bytes,
                        luid=luid,
                        device_node_mask=node_mask,
                    )
                )
            return devices
        finally:
            vk.vkDestroyInstance(instance, None)
    except (OSError, ValueError) as exc:  # pragma: no cover - driver dependent
        logger.warning("Vulkan enumeration failed: %s", exc)
        return []


def library_present(name: str) -> bool:
    """True if a dynamic library / executable is loadable by the OS loader."""
    if not _IS_WINDOWS:
        return False
    try:
        ctypes.WinDLL(name, use_last_error=True)
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# Power state / processor performance (B1 measurement validity)               #
# --------------------------------------------------------------------------- #
# ``PowerStatus``/``ProcessorPowerInfo`` are shared raw shapes defined in
# ``native.base`` and re-exported here for backwards compatibility.


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_uint32),
        ("BatteryFullLifeTime", ctypes.c_uint32),
    ]


def power_status() -> PowerStatus | None:
    """Real AC/battery state via the Win32 ``GetSystemPowerStatus`` API."""
    if not _IS_WINDOWS:
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetSystemPowerStatus.argtypes = [ctypes.POINTER(_SystemPowerStatus)]
        kernel32.GetSystemPowerStatus.restype = ctypes.c_int
        status = _SystemPowerStatus()
        if not kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            logger.warning("GetSystemPowerStatus failed")
            return None
    except OSError as exc:  # pragma: no cover - environment dependent
        logger.warning("GetSystemPowerStatus unavailable: %s", exc)
        return None

    ac = {0: False, 1: True}.get(int(status.ACLineStatus))
    battery_flag = int(status.BatteryFlag)
    indeterminate = battery_flag == 255
    battery_present = None if indeterminate else not bool(battery_flag & 0x80)
    percent_raw = int(status.BatteryLifePercent)
    return PowerStatus(
        ac_line=ac,
        battery_present=battery_present,
        battery_percent=None if percent_raw == 255 else percent_raw,
        charging=None if indeterminate else bool(battery_flag & 0x08),
    )


class _ProcessorPowerInformation(ctypes.Structure):
    _fields_ = [
        ("Number", ctypes.c_uint32),
        ("MaxMhz", ctypes.c_uint32),
        ("CurrentMhz", ctypes.c_uint32),
        ("MhzLimit", ctypes.c_uint32),
        ("MaxIdleState", ctypes.c_uint32),
        ("CurrentIdleState", ctypes.c_uint32),
    ]


# POWER_INFORMATION_LEVEL.ProcessorInformation == 11 (powrprof.h).
_PROCESSOR_INFORMATION_LEVEL = 11
_ALL_PROCESSORS = 0xFFFF


def processor_power() -> list[ProcessorPowerInfo]:
    """Real per-processor clock state via ``CallNtPowerInformation``.

    Returns an empty list when the platform does not expose it; the caller must treat
    an empty result as "unknown", never as "full speed".
    """
    if not _IS_WINDOWS:
        return []
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetActiveProcessorCount.argtypes = [ctypes.c_uint16]
        kernel32.GetActiveProcessorCount.restype = ctypes.c_uint32
        count = int(kernel32.GetActiveProcessorCount(_ALL_PROCESSORS))
        if count <= 0:
            return []
        powrprof = ctypes.WinDLL("powrprof.dll", use_last_error=True)
        powrprof.CallNtPowerInformation.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        powrprof.CallNtPowerInformation.restype = ctypes.c_int32
        buffer = (_ProcessorPowerInformation * count)()
        status = powrprof.CallNtPowerInformation(
            _PROCESSOR_INFORMATION_LEVEL,
            None,
            0,
            ctypes.byref(buffer),
            ctypes.sizeof(buffer),
        )
        if status != 0:  # STATUS_SUCCESS
            logger.warning("CallNtPowerInformation returned 0x%08X", status & 0xFFFFFFFF)
            return []
    except (OSError, ValueError) as exc:  # pragma: no cover - environment dependent
        logger.warning("processor power unavailable: %s", exc)
        return []

    result: list[ProcessorPowerInfo] = []
    for entry in buffer:
        max_mhz = int(entry.MaxMhz)
        if max_mhz <= 0:
            continue
        result.append(
            ProcessorPowerInfo(
                number=int(entry.Number),
                current_mhz=int(entry.CurrentMhz),
                max_mhz=max_mhz,
                limit_mhz=int(entry.MhzLimit),
            )
        )
    return result


# --------------------------------------------------------------------------- #
# Live performance counters                                                   #
# --------------------------------------------------------------------------- #
def sample_perf_counters(paths: Sequence[str], *, warmup_s: float = 1.0) -> list[CounterSample]:
    """Read real OS performance counters.

    Rate counters (page faults, GPU utilization) need two reads to produce a
    meaningful value, so a warm-up read is taken and discarded first.
    """
    if not paths:
        return []
    escaped = ",".join("'" + path.replace("'", "''") + "'" for path in paths)
    script = (
        "$ErrorActionPreference='SilentlyContinue'\n"
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
        f"$paths = @({escaped})\n"
        "$null = (Get-Counter -Counter $paths -MaxSamples 1 -ErrorAction SilentlyContinue)\n"
        f"Start-Sleep -Milliseconds {max(0, int(warmup_s * 1000))}\n"
        "$samples = (Get-Counter -Counter $paths -MaxSamples 1 -ErrorAction SilentlyContinue).CounterSamples\n"
        "foreach ($s in $samples) {\n"
        "  if ($null -eq $s.CookedValue) { continue }\n"
        "  '{0}|{1}|{2}' -f $s.Path, $s.InstanceName, $s.CookedValue\n"
        "}\n"
    )
    output = run_powershell(script, timeout_s=max(_PS_TIMEOUT_S, warmup_s + 30.0))
    samples: list[CounterSample] = []
    for line in output.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        try:
            value = float(parts[2])
        except ValueError:
            continue
        instance = parts[1].strip() or None
        samples.append(CounterSample(path=parts[0].strip(), instance=instance, value=value))
    return samples


# --------------------------------------------------------------------------- #
# Misc                                                                        #
# --------------------------------------------------------------------------- #
def parse_wmi_datetime(value: Any) -> str | None:
    """Normalize a WMI date (``/Date(ms)/`` or ISO) to an ISO-8601 date string."""
    if value is None:
        return None
    text = str(value)
    if text.startswith("/Date(") and text.endswith(")/"):
        number = text[len("/Date(") : -len(")/")].split("+", 1)[0].split("-", 1)[0]
        try:
            millis = int(number)
        except ValueError:
            return None
        return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc).date().isoformat()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text or None
