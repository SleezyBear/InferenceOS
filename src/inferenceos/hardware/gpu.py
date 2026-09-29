"""GPU characterization.

Detected hardware (WMI ``Win32_VideoController``) and actually usable execution
backends (real Vulkan physical-device enumeration, real CUDA/ROCm runtime
presence checks) are tracked separately. A GPU present in the OS inventory but not
enumerated by any usable backend is recorded as detected-but-not-usable rather
than being silently treated as available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from inferenceos.contracts.enums import BackendType, GPUDeviceType, GPUVendor
from inferenceos.contracts.hardware import (
    BackendAvailability,
    BackendDevice,
    CPUProfile,
    GPUProfile,
    HardwareProfile,
)
from inferenceos.native import backend_library_names, get_platform
from inferenceos.native.base import VulkanDevice

_VENDOR_BY_PCI: dict[int, GPUVendor] = {
    0x1002: GPUVendor.AMD,
    0x1022: GPUVendor.AMD,
    0x8086: GPUVendor.INTEL,
    0x10DE: GPUVendor.NVIDIA,
    0x106B: GPUVendor.APPLE,
}

_VULKAN_DEVICE_TYPES: dict[int, GPUDeviceType] = {
    0: GPUDeviceType.UNKNOWN,
    1: GPUDeviceType.INTEGRATED,
    2: GPUDeviceType.DISCRETE,
    3: GPUDeviceType.VIRTUAL,
    4: GPUDeviceType.CPU,
}


@dataclass(frozen=True)
class GpuCharacterization:
    gpus: list[GPUProfile]
    backend_availability: list[BackendAvailability]
    vulkan_devices: list[VulkanDevice]

    @property
    def vulkan_available(self) -> bool:
        return any(
            availability.backend == BackendType.VULKAN and availability.usable
            for availability in self.backend_availability
        )

    @property
    def cuda_available(self) -> bool:
        return any(
            availability.backend == BackendType.CUDA and availability.usable
            for availability in self.backend_availability
        )

    @property
    def rocm_available(self) -> bool:
        return any(
            availability.backend == BackendType.ROCM and availability.usable
            for availability in self.backend_availability
        )

    @property
    def backends_available(self) -> list[BackendType]:
        return [a.backend for a in self.backend_availability if a.usable]


def vendor_from_pci_id(vendor_id: int) -> GPUVendor:
    return _VENDOR_BY_PCI.get(vendor_id, GPUVendor.UNKNOWN)


def total_vram_bytes(hardware: HardwareProfile) -> int | None:
    """The largest real device-local VRAM reported by a usable backend device.

    Prefers actually-usable backend devices (e.g. Vulkan physical devices) over the
    OS inventory, so a device that is present but not executable does not inflate the
    total. Returns ``None`` when no device exposes a VRAM size (never fabricated).
    """
    candidates: list[int] = []
    for availability in hardware.backend_availability:
        if not availability.usable:
            continue
        for device in availability.devices:
            if device.usable and device.vram_bytes:
                candidates.append(int(device.vram_bytes))
    if candidates:
        return max(candidates)
    for gpu in hardware.gpus:
        if gpu.vram_bytes:
            candidates.append(int(gpu.vram_bytes))
    return max(candidates) if candidates else None


def device_type_from_vulkan(code: int) -> GPUDeviceType:
    return _VULKAN_DEVICE_TYPES.get(code, GPUDeviceType.UNKNOWN)


def pci_id_from_pnp(pnp_device_id: Any) -> str | None:
    """Extract ``vendor:device`` (e.g. ``1002:67ef``) from a PnP device id."""
    if not pnp_device_id:
        return None
    text = str(pnp_device_id).upper()
    vendor = _extract_pnp_field(text, "VEN_")
    device = _extract_pnp_field(text, "DEV_")
    if vendor and device:
        return f"{vendor.lower()}:{device.lower()}"
    return None


def _extract_pnp_field(text: str, marker: str) -> str | None:
    index = text.find(marker)
    if index == -1:
        return None
    start = index + len(marker)
    end = start
    while end < len(text) and text[end] in "0123456789ABCDEF":
        end += 1
    return text[start:end] or None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _strings_match(left: str, right: str) -> bool:
    a = left.strip().lower()
    b = right.strip().lower()
    if not a or not b:
        return False
    return a in b or b in a


def _controller_for(
    name: str,
    controllers: Sequence[Mapping[str, Any]],
    *,
    pci_id: str | None = None,
) -> Mapping[str, Any] | None:
    """Match a Vulkan device to a WMI video controller by name, then by PCI identity.

    Name substring matching is the historical behavior and is tried first. When it
    fails (e.g. different vendor naming between Vulkan and WMI), the real PCI
    ``VEN_/DEV_`` identity from the WMI ``PNPDeviceID`` is used as a more reliable
    fallback instead of leaving the devices uncorrelated.
    """
    for controller in controllers:
        if _strings_match(name, str(controller.get("Name", ""))):
            return controller
    if pci_id:
        for controller in controllers:
            if pci_id_from_pnp(controller.get("PNPDeviceID")) == pci_id:
                return controller
    return None


def build_gpu_profiles(
    vulkan_devices: Sequence[VulkanDevice],
    video_controllers: Sequence[Mapping[str, Any]],
) -> list[GPUProfile]:
    """Reconcile Vulkan-enumerated devices with WMI-inventoried controllers."""
    native = get_platform()
    profiles: list[GPUProfile] = []
    matched_controllers: set[int] = set()

    for device in vulkan_devices:
        device_pci_id = f"{device.vendor_id:04x}:{device.device_id:04x}"
        controller = _controller_for(
            device.name, video_controllers, pci_id=device_pci_id
        )
        if controller is not None:
            matched_controllers.add(id(controller))
        driver_version = None
        driver_date = None
        pci_id = device_pci_id
        if controller is not None:
            driver_version = _str_or_none(controller.get("DriverVersion"))
            driver_date = native.parse_wmi_datetime(controller.get("DriverDate"))
            pci_id = pci_id_from_pnp(controller.get("PNPDeviceID")) or pci_id
        profiles.append(
            GPUProfile(
                name=device.name,
                vendor=vendor_from_pci_id(device.vendor_id),
                device_type=device_type_from_vulkan(device.device_type),
                vram_bytes=device.device_local_bytes or None,
                driver_version=driver_version,
                driver_date=driver_date,
                supported_backends=[BackendType.VULKAN],
                vulkan_device_index=device.index,
                vulkan_api_version=device.api_version,
                pci_id=pci_id,
            )
        )

    for controller in video_controllers:
        if id(controller) in matched_controllers:
            continue
        name = str(controller.get("Name") or "Unknown GPU")
        adapter_ram = _as_int(controller.get("AdapterRAM"))
        profiles.append(
            GPUProfile(
                name=name,
                vendor=vendor_from_pci_id(_vendor_id_from_pnp(controller.get("PNPDeviceID")) or 0),
                device_type=GPUDeviceType.UNKNOWN,
                vram_bytes=adapter_ram if adapter_ram and adapter_ram > 0 else None,
                driver_version=_str_or_none(controller.get("DriverVersion")),
                driver_date=native.parse_wmi_datetime(controller.get("DriverDate")),
                supported_backends=[],
                pci_id=pci_id_from_pnp(controller.get("PNPDeviceID")),
            )
        )
    return profiles


def _vendor_id_from_pnp(pnp_device_id: Any) -> int | None:
    if not pnp_device_id:
        return None
    text = str(pnp_device_id).upper()
    vendor = _extract_pnp_field(text, "VEN_")
    if not vendor:
        return None
    try:
        return int(vendor, 16)
    except ValueError:
        return None


def build_backend_availability(
    vulkan_devices: Sequence[VulkanDevice],
    cpu_profile: CPUProfile,
    *,
    vulkan_loader_present: bool,
    cuda_runtime_present: bool,
    rocm_runtime_present: bool,
) -> list[BackendAvailability]:
    """Describe each execution backend as detected and/or usable.

    Usability is decided by real enumeration: Vulkan is usable only if at least
    one physical device is enumerated. CUDA/ROCm usability is not validated in
    Phase 1, so their runtimes are reported detected but not claimed usable.
    """
    vulkan_devices_contract = [
        BackendDevice(
            name=device.name,
            device_index=device.index,
            device_type=device_type_from_vulkan(device.device_type),
            vram_bytes=device.device_local_bytes or None,
            usable=True,
            detail=f"vulkan-api={device.api_version}",
        )
        for device in vulkan_devices
    ]
    vulkan_detected = vulkan_loader_present or bool(vulkan_devices)
    vulkan_detail = (
        f"{len(vulkan_devices)} Vulkan physical device(s) enumerated"
        if vulkan_devices
        else ("Vulkan loader present but no physical device enumerated"
              if vulkan_loader_present
              else "Vulkan loader not present")
    )

    availability = [
        BackendAvailability(
            backend=BackendType.CPU,
            detected=True,
            usable=True,
            devices=[
                BackendDevice(
                    name=cpu_profile.name,
                    device_type=GPUDeviceType.CPU,
                    usable=True,
                    detail=f"{cpu_profile.physical_cores} physical / "
                    f"{cpu_profile.logical_cores} logical cores",
                )
            ],
            detail="CPU execution always available",
        ),
        BackendAvailability(
            backend=BackendType.VULKAN,
            detected=vulkan_detected,
            usable=bool(vulkan_devices),
            devices=vulkan_devices_contract,
            detail=vulkan_detail,
        ),
        BackendAvailability(
            backend=BackendType.CUDA,
            detected=cuda_runtime_present,
            usable=False,
            devices=[],
            detail=(
                "CUDA runtime present; device usable not validated"
                if cuda_runtime_present
                else "CUDA runtime not present"
            ),
        ),
        BackendAvailability(
            backend=BackendType.ROCM,
            detected=rocm_runtime_present,
            usable=False,
            devices=[],
            detail=(
                "ROCm/HIP runtime present; device usable not validated"
                if rocm_runtime_present
                else "ROCm/HIP runtime not present"
            ),
        ),
    ]
    return availability


def characterize_gpus(
    cpu_profile: CPUProfile,
    *,
    vulkan_devices: Sequence[VulkanDevice] | None = None,
    video_controllers: Sequence[Mapping[str, Any]] | None = None,
    vulkan_loader_present: bool | None = None,
    cuda_runtime_present: bool | None = None,
    rocm_runtime_present: bool | None = None,
) -> GpuCharacterization:
    """Characterize GPUs and backends from real probes (injectable for tests)."""
    native = get_platform()
    vulkan_devices = (
        list(vulkan_devices) if vulkan_devices is not None else native.vulkan_enumerate()
    )
    if video_controllers is None:
        video_controllers = native.collect_cim_snapshot().video_controllers
    if vulkan_loader_present is None:
        vulkan_loader_present = any(
            native.library_present(name) for name in backend_library_names("vulkan")
        )
    if cuda_runtime_present is None:
        cuda_runtime_present = any(
            native.library_present(name) for name in backend_library_names("cuda")
        )
    if rocm_runtime_present is None:
        rocm_runtime_present = any(
            native.library_present(name) for name in backend_library_names("rocm")
        )
    gpus = build_gpu_profiles(vulkan_devices, video_controllers)
    availability = build_backend_availability(
        vulkan_devices,
        cpu_profile,
        vulkan_loader_present=vulkan_loader_present,
        cuda_runtime_present=cuda_runtime_present,
        rocm_runtime_present=rocm_runtime_present,
    )
    return GpuCharacterization(
        gpus=gpus,
        backend_availability=availability,
        vulkan_devices=vulkan_devices,
    )


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
