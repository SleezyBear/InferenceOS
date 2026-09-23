"""Native probe parsing plus a real CPUID cross-check against the OS API."""

from __future__ import annotations

import pytest

from inferenceos.native import windows


def test_parse_cim_snapshot_normalizes_sections() -> None:
    payload = (
        '{"OS":{"Caption":"Microsoft Windows 10 Pro","Version":"10.0.19045"},'
        '"Processor":[{"Name":"CPU"}],'
        '"VideoController":{"Name":"GPU"},'
        '"Association":null}'
    )
    snapshot = windows.parse_cim_snapshot(payload)
    assert snapshot.os["Caption"] == "Microsoft Windows 10 Pro"
    assert len(snapshot.processors) == 1
    assert len(snapshot.video_controllers) == 1
    assert snapshot.associations == []


def test_parse_cim_snapshot_malformed_is_empty() -> None:
    snapshot = windows.parse_cim_snapshot("{not json")
    assert snapshot.processors == []
    assert snapshot.os == {}
    assert windows.parse_cim_snapshot("").processors == []


def test_parse_wmi_datetime_formats() -> None:
    assert windows.parse_wmi_datetime("/Date(1724025600000)/") == "2024-08-19"
    assert windows.parse_wmi_datetime("2024-08-19T00:00:00Z") == "2024-08-19"
    assert windows.parse_wmi_datetime(None) is None


@pytest.mark.skipif(not windows.is_windows(), reason="Windows-only probe")
def test_real_cpuid_agrees_with_os_feature_api() -> None:
    cpuid = windows.cpuid_features()
    os_flags = windows.os_cpu_features()
    if not cpuid or not os_flags:
        pytest.skip("CPU feature APIs unavailable")
    for feature in ("sse2", "avx", "avx2", "avx512f"):
        if feature in cpuid and feature in os_flags:
            assert cpuid[feature] == os_flags[feature], feature


@pytest.mark.skipif(not windows.is_windows(), reason="Windows-only probe")
def test_real_vulkan_enumeration_shape() -> None:
    devices = windows.vulkan_enumerate()
    for index, device in enumerate(devices):
        assert device.index == index
        assert device.name
        assert "." in device.api_version
        assert device.vendor_id > 0
