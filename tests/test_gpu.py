"""GPU/backend characterization: reconciliation and detected-vs-usable logic."""

from __future__ import annotations

import pytest

from inferenceos.contracts.enums import BackendType, GPUDeviceType, GPUVendor
from inferenceos.contracts.hardware import BackendAvailability, BackendDevice
from inferenceos.hardware.cpu import build_cpu_profile
from inferenceos.hardware.gpu import (
    build_backend_availability,
    build_gpu_profiles,
    characterize_gpus,
    device_type_from_vulkan,
    pci_id_from_pnp,
    planning_vram_bytes,
    vendor_from_pci_id,
)
from inferenceos.native.windows import VulkanDevice
from inferenceos.selftest import sample_hardware, sample_model

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


@pytest.mark.parametrize(
    "inventory,backends,expected_gib",
    [
        pytest.param(False, "cpu_and_gpu", 4, id="cpu-is-not-a-second-gpu"),
        pytest.param(True, "cpu_and_gpu", 4, id="one-inventoried-gpu"),
        pytest.param(False, "matching_apis", 4, id="matching-gpu-in-two-apis"),
        pytest.param(True, "conflicting_apis", None, id="os-singleton-does-not-resolve-api-conflict"),
        pytest.param(False, "conflicting_apis", None, id="api-conflict-without-os-inventory"),
        pytest.param(True, "software_and_gpu", 4, id="software-vulkan-is-not-a-second-gpu"),
        pytest.param(False, "cpu_only", None, id="cpu-memory-is-not-vram"),
        pytest.param(True, "two_gpus", None, id="multiple-devices-within-one-api"),
    ],
)
def test_scalar_gpu_capacity_uses_only_unambiguous_gpu_evidence(
    inventory: bool, backends: str, expected_gib: int | None,
) -> None:
    hardware = sample_hardware()
    available = build_backend_availability(
        [VULKAN_DEVICE], CPU, vulkan_loader_present=True,
        cuda_runtime_present=False, rocm_runtime_present=False,
    )
    cpu, vulkan = available[:2]
    # Include real CPU availability, even when the OS GPU inventory is unavailable.
    if backends == "matching_apis":
        same = vulkan.devices[0].model_copy(update={"name": "  RADEON   Pro 560X  "})
        available.append(BackendAvailability(backend=BackendType.CUDA, usable=True, devices=[same]))
    elif backends == "conflicting_apis":
        other = BackendDevice(name="unrelated 16 GiB GPU", device_type=GPUDeviceType.DISCRETE,
                              vram_bytes=16 * 1024**3, usable=True)
        available.append(BackendAvailability(backend=BackendType.CUDA, usable=True, devices=[other]))
    elif backends == "software_and_gpu":
        software = BackendDevice(name="software renderer", device_type=GPUDeviceType.CPU,
                                 vram_bytes=32 * 1024**3, usable=True)
        vulkan.devices.append(software)
    elif backends == "cpu_only":
        cpu.devices[0] = cpu.devices[0].model_copy(update={"vram_bytes": 32 * 1024**3})
        available = [cpu]
    elif backends == "two_gpus":
        vulkan.devices.append(vulkan.devices[0].model_copy(update={"device_index": 1}))
    hardware = hardware.model_copy(update={
        "gpus": hardware.gpus if inventory else [], "backend_availability": available,
    })
    expected = expected_gib * 1024**3 if expected_gib is not None else None
    assert planning_vram_bytes(hardware) == expected


@pytest.mark.parametrize("case,expected_card", [
    ("software_renderer", False), ("inventory_only", False),
    ("unusable_backend", False), ("unusable_device", False),
    ("gpu_no_memory", True), ("gpu_present", True), ("two_gpus", True),
])
def test_gpu_capability_does_not_suppress_processor_paths_without_a_usable_gpu(
    case: str, expected_card: bool,
) -> None:
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.tune.model_facts import ModelFacts, machine_facts, processor_share
    from inferenceos.tune.paths import THREADS, PathContext

    hardware = sample_hardware()
    if case == "software_renderer":
        software = BackendDevice(name="CPU software renderer", device_type=GPUDeviceType.CPU,
                                 vram_bytes=32 * 1024**3, usable=True)
        hardware = hardware.model_copy(update={
            "gpus": [hardware.gpus[0].model_copy(update={"name": software.name,
                "device_type": GPUDeviceType.CPU, "vram_bytes": software.vram_bytes})],
            "backend_availability": [BackendAvailability(backend=BackendType.VULKAN, usable=True, devices=[software])],
        })
    elif case == "inventory_only":
        hardware = hardware.model_copy(update={"backend_availability": []})
    elif case in ("unusable_backend", "unusable_device", "gpu_no_memory"):
        usable = next(a for a in hardware.backend_availability if a.backend is BackendType.VULKAN)
        device = usable.devices[0].model_copy(update={
            "usable": case != "unusable_device",
            "vram_bytes": None if case == "gpu_no_memory" else usable.devices[0].vram_bytes,
        })
        hardware = hardware.model_copy(update={"backend_availability": [usable.model_copy(update={
            "usable": case != "unusable_backend", "devices": [device],
        })], "gpus": [hardware.gpus[0].model_copy(update={
            "vram_bytes": None if case == "gpu_no_memory" else hardware.gpus[0].vram_bytes,
        })]})
    elif case == "two_gpus":
        hardware.gpus.append(hardware.gpus[0].model_copy(update={"name": "another real GPU"}))
    machine = machine_facts(hardware, None)
    model = sample_model()
    facts = ModelFacts(total_layers=model.n_layers)
    config = RuntimeConfiguration(context_length=1024, gpu_offload_layers=model.n_layers + 1)
    share = processor_share(facts, config.gpu_offload_layers, card=machine.card)
    context = PathContext(hardware, model, model.n_layers, config, facts=facts,
                          machine=machine, processor_share=share)
    assert machine.card is expected_card
    assert share == (0.0 if expected_card else 1.0)
    assert THREADS.applies(context)[0] is (not expected_card)
