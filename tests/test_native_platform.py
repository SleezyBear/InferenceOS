"""Native platform seam: selection, explicit override, and consumer resolution."""

from __future__ import annotations

import sys

import pytest

from inferenceos.native import (
    CimSnapshot,
    CounterSample,
    GlobalMemory,
    PowerStatus,
    ProcessorPowerInfo,
    VulkanDevice,
    get_platform,
    is_windows,
)


def test_windows_platform_selection_returns_windows_provider() -> None:
    provider = get_platform("windows")
    assert provider.is_windows() is True
    assert hasattr(provider, "collect_cim_snapshot")
    assert hasattr(provider, "vulkan_enumerate")
    assert hasattr(provider, "sample_perf_counters")
    # On this host (win32) the default selection is the same provider.
    if sys.platform == "win32":
        assert get_platform() is provider
        assert is_windows() is True


def test_linux_and_macos_are_recognized_but_not_implemented() -> None:
    for key in ("linux", "macos", "darwin"):
        with pytest.raises(NotImplementedError):
            get_platform(key)


def test_unsupported_platform_raises_clear_error() -> None:
    with pytest.raises(OSError):
        get_platform("plan9")


def test_explicit_override_selects_windows_deterministically() -> None:
    # Override works regardless of the host OS detection path.
    assert get_platform("windows").is_windows() is True
    assert get_platform("win32").is_windows() is True


def test_native_reexports_remain_compatible() -> None:
    from inferenceos.native import (  # noqa: F401
        collect_cim_snapshot,
        cpuid_cache_info,
        cpuid_features,
        global_memory_status,
        library_present,
        os_cpu_features,
        parse_cim_snapshot,
        parse_wmi_datetime,
        sample_perf_counters,
        vulkan_enumerate,
    )
    from inferenceos.native.windows import (  # noqa: F401
        CimSnapshot as WinCimSnapshot,
        CounterSample as WinCounterSample,
        GlobalMemory as WinGlobalMemory,
        PowerStatus as WinPowerStatus,
        ProcessorPowerInfo as WinProcessorPowerInfo,
        VulkanDevice as WinVulkanDevice,
    )

    # Shared shapes are the same objects whether imported from base or windows.
    assert WinCimSnapshot is CimSnapshot
    assert WinVulkanDevice is VulkanDevice
    assert WinCounterSample is CounterSample
    assert WinGlobalMemory is GlobalMemory
    assert WinPowerStatus is PowerStatus
    assert WinProcessorPowerInfo is ProcessorPowerInfo


class _FakePlatform:
    """Minimal provider double that proves consumers resolve the seam."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def is_windows(self) -> bool:
        return True

    def collect_cim_snapshot(self) -> CimSnapshot:
        self.calls.append("collect_cim_snapshot")
        return CimSnapshot(
            processors=[
                {
                    "Name": "Fake CPU",
                    "Manufacturer": "FakeVendor",
                    "NumberOfCores": 4,
                    "NumberOfLogicalProcessors": 8,
                    "MaxClockSpeed": 3000,
                    "CurrentClockSpeed": 2500,
                    "Architecture": 9,
                }
            ],
            physical_memory=[{"Capacity": 8 * 1024**3, "Speed": 2400}],
        )

    def cpuid_features(self) -> dict[str, bool]:
        self.calls.append("cpuid_features")
        return {"sse2": True, "avx2": True}

    def os_cpu_features(self) -> dict[str, bool]:
        self.calls.append("os_cpu_features")
        return {"sse2": True, "avx2": True}

    def cpuid_cache_info(self) -> list[dict[str, int]]:
        self.calls.append("cpuid_cache_info")
        return [{"level": 3, "size_bytes": 8 * 1024**2}]

    def global_memory_status(self) -> GlobalMemory:
        self.calls.append("global_memory_status")
        return GlobalMemory(
            total_physical_bytes=8 * 1024**3,
            available_physical_bytes=4 * 1024**3,
            total_pagefile_bytes=12 * 1024**3,
            available_pagefile_bytes=6 * 1024**3,
            total_virtual_bytes=16 * 1024**3,
            available_virtual_bytes=8 * 1024**3,
            memory_load_percent=50,
        )

    def power_status(self) -> PowerStatus:
        self.calls.append("power_status")
        return PowerStatus(
            ac_line=True, battery_present=True, battery_percent=90, charging=True
        )

    def processor_power(self) -> list[ProcessorPowerInfo]:
        self.calls.append("processor_power")
        return [ProcessorPowerInfo(number=0, current_mhz=1000, max_mhz=2000, limit_mhz=2000)]


def test_hardware_cpu_resolves_the_platform_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inferenceos.hardware.cpu as cpu_module

    assert not hasattr(cpu_module, "windows")  # no direct Windows assumption
    fake = _FakePlatform()
    monkeypatch.setattr(cpu_module, "get_platform", lambda: fake)
    profile = cpu_module.detect_cpu()
    assert profile.name == "Fake CPU"
    assert profile.physical_cores == 4
    assert "sse2" in profile.instruction_sets
    assert "collect_cim_snapshot" in fake.calls


def test_hardware_memory_resolves_the_platform_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inferenceos.hardware.memory as memory_module

    assert not hasattr(memory_module, "windows")
    fake = _FakePlatform()
    monkeypatch.setattr(memory_module, "get_platform", lambda: fake)
    profile = memory_module.detect_memory()
    assert profile.total_bytes == 8 * 1024**3
    assert profile.modules[0].capacity_bytes == 8 * 1024**3
    assert "global_memory_status" in fake.calls


def test_telemetry_power_resolves_the_platform_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inferenceos.telemetry.system as system_module

    assert not hasattr(system_module, "windows")
    fake = _FakePlatform()
    monkeypatch.setattr(system_module, "get_platform", lambda: fake)
    performance = system_module.observe_power_performance()
    assert performance.current_mhz == 1000
    assert performance.max_mhz == 2000
    assert performance.ratio == 0.5
    assert "power_status" in fake.calls and "processor_power" in fake.calls


def test_consumer_modules_do_not_import_windows_directly() -> None:
    import inferenceos.hardware.gpu as gpu_module
    import inferenceos.hardware.profiler as profiler_module
    import inferenceos.hardware.storage as storage_module
    import inferenceos.profiling.runner as runner_module
    import inferenceos.runtime.verification as verification_module
    import inferenceos.telemetry.counters as counters_module

    for module in (
        gpu_module,
        profiler_module,
        storage_module,
        runner_module,
        verification_module,
        counters_module,
    ):
        assert not hasattr(module, "windows"), module.__name__


def test_luid_normalization_available_through_the_seam() -> None:
    provider = get_platform("windows")
    assert provider.normalize_luid("0x00000000:0x00009383") == "00000000:00009383"


def test_platform_provider_is_a_native_platform_shape() -> None:
    # Structural check: the selected provider exposes the documented surface.
    provider = get_platform("windows")
    surface = (
        "is_windows",
        "collect_cim_snapshot",
        "cpuid_features",
        "os_cpu_features",
        "cpuid_cache_info",
        "global_memory_status",
        "vulkan_enumerate",
        "library_present",
        "normalize_luid",
        "power_status",
        "processor_power",
        "sample_perf_counters",
        "parse_wmi_datetime",
    )
    for attribute in surface:
        assert callable(getattr(provider, attribute, None)), attribute
