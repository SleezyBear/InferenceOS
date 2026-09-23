"""CPU characterization: feature normalization and profile building."""

from __future__ import annotations

import pytest

from inferenceos.hardware.cpu import (
    INSTRUCTION_SET_ORDER,
    architecture_from_code,
    base_clock_from_name,
    build_cpu_profile,
    caches_from_cpuid,
    caches_from_wmi,
    normalize_instruction_sets,
)

FULL_CPUID = {
    "mmx": True,
    "sse": True,
    "sse2": True,
    "sse3": True,
    "ssse3": True,
    "sse4_1": True,
    "sse4_2": True,
    "popcnt": True,
    "aes": True,
    "osxsave": True,
    "avx": True,
    "fma": True,
    "f16c": True,
    "bmi1": True,
    "avx2": True,
    "bmi2": True,
    "avx512f": True,
    "avx512dq": True,
    "avx512bw": True,
    "avx512vl": True,
    "avx512vnni": True,
}


def test_normalize_uses_canonical_order() -> None:
    result = normalize_instruction_sets(FULL_CPUID, {"avx": True})
    assert result == [name for name in INSTRUCTION_SET_ORDER if FULL_CPUID.get(name)]
    assert result.index("avx") < result.index("avx2") < result.index("avx512f")


def test_normalize_gates_avx_family_without_os_support() -> None:
    result = normalize_instruction_sets(FULL_CPUID, {"avx": False})
    assert "sse2" in result
    for gated in ("avx", "fma", "f16c", "avx2", "avx512f", "avx512vnni"):
        assert gated not in result


def test_normalize_falls_back_to_os_flags() -> None:
    assert normalize_instruction_sets({}, {"sse2": True, "avx": True}) == ["sse2", "avx"]


def test_normalize_empty_sources_yield_nothing() -> None:
    assert normalize_instruction_sets({}, {}) == []


def test_caches_from_cpuid_aggregates_levels() -> None:
    caches = caches_from_cpuid(
        [
            {"level": 1, "type": 1, "size_bytes": 32 * 1024},
            {"level": 1, "type": 2, "size_bytes": 32 * 1024},
            {"level": 2, "type": 3, "size_bytes": 256 * 1024},
            {"level": 3, "type": 3, "size_bytes": 8 * 1024**2},
        ]
    )
    assert caches.l1_bytes == 64 * 1024
    assert caches.l2_bytes == 256 * 1024
    assert caches.l3_bytes == 8 * 1024**2


def test_caches_from_cpuid_empty_is_none() -> None:
    caches = caches_from_cpuid([])
    assert caches.l1_bytes is None and caches.l2_bytes is None and caches.l3_bytes is None


def test_caches_from_wmi_kb_to_bytes() -> None:
    caches = caches_from_wmi({"L2CacheSize": 1536, "L3CacheSize": 9216})
    assert caches.l2_bytes == 1536 * 1024
    assert caches.l3_bytes == 9216 * 1024
    assert caches.l1_bytes is None


def test_architecture_and_base_clock() -> None:
    assert architecture_from_code(9) == "x86_64"
    assert architecture_from_code(12) == "arm64"
    assert architecture_from_code(None) == "unknown"
    assert base_clock_from_name("Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz") == 2600
    assert base_clock_from_name("Unknown CPU") is None


def _processor(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "Name": "Synthetic CPU @ 3.00GHz",
        "Manufacturer": "SyntheticVendor",
        "NumberOfCores": 4,
        "NumberOfLogicalProcessors": 8,
        "MaxClockSpeed": 3100,
        "L2CacheSize": 1024,
        "L3CacheSize": 8192,
        "Architecture": 9,
    }
    base.update(overrides)
    return base


def test_build_cpu_profile_from_real_shaped_input() -> None:
    profile = build_cpu_profile(
        [_processor()],
        cpuid=FULL_CPUID,
        os_flags={"avx": True},
        cache_descriptors=[{"level": 1, "type": 1, "size_bytes": 32768}],
    )
    assert profile.physical_cores == 4
    assert profile.logical_cores == 8
    assert profile.architecture == "x86_64"
    assert profile.base_clock_mhz == 3000
    assert profile.max_clock_mhz == 3100
    assert profile.caches.l1_bytes == 32768
    assert profile.detected_via.startswith("cpuid")
    assert "avx2" in profile.instruction_sets


def test_build_cpu_profile_handles_missing_optional_fields() -> None:
    profile = build_cpu_profile(
        [{"Name": "Bare CPU", "NumberOfCores": 2, "NumberOfLogicalProcessors": 2}],
        cpuid={},
        os_flags={"sse2": True},
    )
    assert profile.vendor is None
    assert profile.architecture == "unknown"
    assert profile.max_clock_mhz is None
    assert profile.caches.l2_bytes is None
    assert profile.instruction_sets == ["sse2"]


def test_build_cpu_profile_requires_processor_data() -> None:
    with pytest.raises(ValueError):
        build_cpu_profile([], cpuid={}, os_flags={})
