"""B-LUID: GPU telemetry ↔ Vulkan device correlation by real LUID identity.

The key property under test is that telemetry *appearance order* cannot corrupt the
device identity: a telemetry LUID is mapped to the matching Vulkan physical device's
real index even when the telemetry stream lists devices in the opposite order.
"""

from __future__ import annotations

from inferenceos.hardware.gpu import build_gpu_profiles
from inferenceos.native.windows import (
    CounterSample,
    VulkanDevice,
    format_device_luid,
    normalize_luid,
)
from inferenceos.telemetry.system import (
    aggregate_gpu_samples,
    map_luids_to_device_indices,
)

ENGINE_PATH = r"\\host\gpu engine(pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d)\utilization percentage"
ADAPTER_PATH = r"\\host\gpu adapter memory(luid_0x00000000_0x00009383_phys_0)\dedicated usage"

_INTEL_LUID = "0x00000000:0x0000a937"
_RADEON_LUID = "0x00000000:0x00009383"


def _device(index: int, name: str, luid: str | None) -> VulkanDevice:
    return VulkanDevice(
        index=index,
        name=name,
        vendor_id=0x8086 if "Intel" in name else 0x1002,
        device_id=0x3E9B if "Intel" in name else 0x67EF,
        device_type=1 if "Intel" in name else 2,
        api_version="1.3.260",
        driver_version=123,
        device_local_bytes=1 * 1024**3,
        luid=luid,
    )


def _engine_sample(luid: str, value: float) -> CounterSample:
    high, low = luid.replace("0x", "").split(":")
    instance = f"pid_1_luid_0x{high}_0x{low}_phys_0_eng_0_engtype_3d"
    return CounterSample(ENGINE_PATH, instance, value)


def test_luid_overrides_telemetry_appearance_order() -> None:
    # Vulkan: Intel iGPU is index 0, Radeon dGPU is index 1.
    intel = _device(0, "Intel UHD Graphics 630", _INTEL_LUID)
    radeon = _device(1, "Radeon Pro 560X", _RADEON_LUID)
    # Telemetry arrives in the OPPOSITE order: Radeon first, then Intel.
    samples = [
        _engine_sample(_RADEON_LUID, 40.0),
        _engine_sample(_INTEL_LUID, 5.0),
    ]

    result = aggregate_gpu_samples(samples, vulkan_devices=[intel, radeon])

    by_index = {engine.device_index: engine.utilization_percent for engine in result.engines}
    # Old appearance-order behavior would have assigned Radeon=0 and Intel=1.
    assert by_index[1] == 40.0  # Radeon telemetry -> Radeon Vulkan index 1
    assert by_index[0] == 5.0  # Intel telemetry -> Intel Vulkan index 0
    assert len(result.engines) == 2


def test_unmatched_telemetry_luid_is_not_misattributed() -> None:
    radeon = _device(1, "Radeon Pro 560X", _RADEON_LUID)
    unknown = "0x00000000:0x0000dead"
    samples = [
        _engine_sample(unknown, 77.0),
        _engine_sample(_RADEON_LUID, 12.0),
    ]

    result = aggregate_gpu_samples(samples, vulkan_devices=[radeon])

    by_index = {engine.device_index: engine.utilization_percent for engine in result.engines}
    assert by_index[1] == 12.0  # real match keeps the Vulkan index
    assert 0 not in by_index  # the unknown LUID must not borrow a real device index
    assert by_index[max(by_index)] == 77.0


def test_missing_luid_falls_back_to_appearance_order() -> None:
    a = _device(0, "Device A", None)
    b = _device(1, "Device B", None)
    samples = [
        _engine_sample("0x00000000:0x0000000a", 10.0),
        _engine_sample("0x00000000:0x0000000b", 20.0),
    ]
    mapping = map_luids_to_device_indices(
        ["00000000:0000000a", "00000000:0000000b"], [a, b]
    )
    assert mapping == {"00000000:0000000a": 0, "00000000:0000000b": 1}
    result = aggregate_gpu_samples(samples, vulkan_devices=[a, b])
    assert {e.device_index for e in result.engines} == {0, 1}


def test_no_vulkan_devices_preserves_legacy_order() -> None:
    samples = [
        _engine_sample("0x00000000:0x0000000a", 10.0),
        _engine_sample("0x00000000:0x0000000b", 20.0),
    ]
    result = aggregate_gpu_samples(samples)
    assert [engine.device_index for engine in result.engines] == [0, 1]


def test_luid_prefix_normalization_matches_pdh_and_vulkan() -> None:
    assert normalize_luid("0x00000000:0x00009383") == "00000000:00009383"
    assert normalize_luid("00000000:00009383") == "00000000:00009383"
    assert normalize_luid("0X00000000:0X00009383") == "00000000:00009383"
    # The PDH parser and the Vulkan formatter must agree.
    assert format_device_luid(0x9383, 0x0) == _RADEON_LUID


def test_vulkan_device_without_luid_is_backward_compatible() -> None:
    device = VulkanDevice(
        index=0,
        name="Radeon Pro 560X",
        vendor_id=0x1002,
        device_id=0x67EF,
        device_type=2,
        api_version="1.3.260",
        driver_version=123,
        device_local_bytes=4 * 1024**3,
    )
    assert device.luid is None
    assert device.device_node_mask is None


def test_wmi_vulkan_pci_identity_fallback_when_names_differ() -> None:
    # Vulkan name does not contain the WMI controller name; PCI VEN/DEV does.
    vulkan = _device(0, "Vulkan Device 1", _RADEON_LUID)
    controller = {
        "Name": "Radeon Pro 560X",
        "DriverVersion": "31.0.21921.1000",
        "DriverDate": "/Date(1724025600000)/",
        "AdapterRAM": 4293918720,
        "PNPDeviceID": "PCI\\VEN_1002&DEV_67EF&SUBSYS_018F106B&REV_C2\\4&D48CA2C&0&0008",
    }
    profiles = build_gpu_profiles([vulkan], [controller])
    assert len(profiles) == 1  # merged via PCI identity, not duplicated
    assert profiles[0].vulkan_device_index == 0
    assert profiles[0].pci_id == "1002:67ef"
