"""GPU/backend characterization: reconciliation and detected-vs-usable logic."""

from __future__ import annotations

from inferenceos.contracts.enums import BackendType, GPUDeviceType, GPUVendor
from inferenceos.hardware.cpu import build_cpu_profile
from inferenceos.hardware.gpu import (
    build_backend_availability,
    build_gpu_profiles,
    characterize_gpus,
    device_type_from_vulkan,
    pci_id_from_pnp,
    vendor_from_pci_id,
)
from inferenceos.native.windows import VulkanDevice

CPU = build_cpu_profile(
    [{"Name": "Test CPU", "NumberOfCores": 6, "NumberOfLogicalProcessors": 12}],
    cpuid={"sse2": True},
    os_flags={"sse2": True},
)

VULKAN_DEVICE = VulkanDevice(
    index=0,
    name="Radeon Pro 560X",
    vendor_id=0x1002,
    device_id=0x67EF,
    device_type=2,
    api_version="1.3.260",
    driver_version=123,
    device_local_bytes=4 * 1024**3,
)

CONTROLLER = {
    "Name": "Radeon Pro 560X",
    "DriverVersion": "31.0.21921.1000",
    "DriverDate": "/Date(1724025600000)/",
    "AdapterRAM": 4293918720,
    "PNPDeviceID": "PCI\\VEN_1002&DEV_67EF&SUBSYS_018F106B&REV_C2\\4&D48CA2C&0&0008",
}


def test_vendor_mapping() -> None:
    assert vendor_from_pci_id(0x1002) is GPUVendor.AMD
    assert vendor_from_pci_id(0x8086) is GPUVendor.INTEL
    assert vendor_from_pci_id(0x10DE) is GPUVendor.NVIDIA
    assert vendor_from_pci_id(0x9999) is GPUVendor.UNKNOWN


def test_device_type_mapping() -> None:
    assert device_type_from_vulkan(2) is GPUDeviceType.DISCRETE
    assert device_type_from_vulkan(1) is GPUDeviceType.INTEGRATED
    assert device_type_from_vulkan(4) is GPUDeviceType.CPU
    assert device_type_from_vulkan(99) is GPUDeviceType.UNKNOWN


def test_pci_id_from_pnp() -> None:
    assert pci_id_from_pnp(CONTROLLER["PNPDeviceID"]) == "1002:67ef"
    assert pci_id_from_pnp(None) is None
    assert pci_id_from_pnp("not-a-pnp-id") is None


def test_build_gpu_profiles_merges_vulkan_and_wmi() -> None:
    profiles = build_gpu_profiles([VULKAN_DEVICE], [CONTROLLER])
    assert len(profiles) == 1
    gpu = profiles[0]
    assert gpu.name == "Radeon Pro 560X"
    assert gpu.vendor is GPUVendor.AMD
    assert gpu.device_type is GPUDeviceType.DISCRETE
    assert gpu.supported_backends == [BackendType.VULKAN]
    assert gpu.vulkan_device_index == 0
    assert gpu.vulkan_api_version == "1.3.260"
    assert gpu.vram_bytes == 4 * 1024**3
    assert gpu.driver_version == "31.0.21921.1000"
    assert gpu.pci_id == "1002:67ef"
    assert gpu.driver_date == "2024-08-19"


def test_build_gpu_profiles_wmi_only_is_detected_not_usable() -> None:
    profiles = build_gpu_profiles([], [CONTROLLER])
    assert len(profiles) == 1
    assert profiles[0].supported_backends == []
    assert profiles[0].vulkan_device_index is None
    assert profiles[0].vendor is GPUVendor.AMD


def test_build_gpu_profiles_vulkan_only() -> None:
    profiles = build_gpu_profiles([VULKAN_DEVICE], [])
    assert len(profiles) == 1
    assert profiles[0].supported_backends == [BackendType.VULKAN]


def test_backend_availability_vulkan_usable() -> None:
    availability = build_backend_availability(
        [VULKAN_DEVICE],
        CPU,
        vulkan_loader_present=True,
        cuda_runtime_present=False,
        rocm_runtime_present=False,
    )
    by_backend = {a.backend: a for a in availability}
    assert by_backend[BackendType.CPU].usable
    assert by_backend[BackendType.VULKAN].detected
    assert by_backend[BackendType.VULKAN].usable
    assert by_backend[BackendType.VULKAN].devices[0].usable
    assert not by_backend[BackendType.CUDA].detected
    assert not by_backend[BackendType.CUDA].usable


def test_backend_availability_vulkan_detected_but_not_usable() -> None:
    availability = build_backend_availability(
        [],
        CPU,
        vulkan_loader_present=True,
        cuda_runtime_present=True,
        rocm_runtime_present=False,
    )
    by_backend = {a.backend: a for a in availability}
    assert by_backend[BackendType.VULKAN].detected
    assert not by_backend[BackendType.VULKAN].usable
    assert by_backend[BackendType.CUDA].detected
    assert not by_backend[BackendType.CUDA].usable


def test_characterize_gpus_uses_injected_probes() -> None:
    result = characterize_gpus(
        CPU,
        vulkan_devices=[VULKAN_DEVICE],
        video_controllers=[CONTROLLER],
        vulkan_loader_present=True,
        cuda_runtime_present=False,
        rocm_runtime_present=False,
    )
    assert result.vulkan_available
    assert not result.cuda_available
    assert not result.rocm_available
    assert BackendType.VULKAN in result.backends_available
    assert BackendType.CPU in result.backends_available
