"""RAM characterization from synthetic GlobalMemoryStatusEx-shaped data."""

from __future__ import annotations

from inferenceos.hardware.memory import build_memory_profile
from inferenceos.native.windows import GlobalMemory


def _status() -> GlobalMemory:
    return GlobalMemory(
        total_physical_bytes=16 * 1024**3,
        available_physical_bytes=8 * 1024**3,
        total_pagefile_bytes=18 * 1024**3,
        available_pagefile_bytes=6 * 1024**3,
        total_virtual_bytes=128 * 1024**3,
        available_virtual_bytes=100 * 1024**3,
        memory_load_percent=50,
    )


def test_build_memory_profile_commit_math() -> None:
    profile = build_memory_profile(
        _status(),
        [{"Capacity": 8 * 1024**3, "Speed": 2400, "Manufacturer": "SK Hynix"}],
    )
    assert profile.total_bytes == 16 * 1024**3
    assert profile.available_bytes == 8 * 1024**3
    assert profile.commit_limit_bytes == 18 * 1024**3
    assert profile.committed_bytes == 12 * 1024**3
    assert profile.modules[0].capacity_bytes == 8 * 1024**3
    assert profile.modules[0].speed_mhz == 2400
    assert profile.modules[0].manufacturer == "SK Hynix"


def test_build_memory_profile_handles_partial_module_data() -> None:
    profile = build_memory_profile(_status(), [{"Capacity": 4 * 1024**3}])
    assert profile.modules[0].speed_mhz is None
    assert profile.modules[0].manufacturer is None


def test_build_memory_profile_no_modules() -> None:
    profile = build_memory_profile(_status(), [])
    assert profile.modules == []
