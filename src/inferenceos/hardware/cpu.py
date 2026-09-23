"""CPU characterization.

Instruction-set capabilities come from a real CPUID probe (cross-checked against
the OS ``IsProcessorFeaturePresent`` API) and cache sizes from CPUID leaf 4 with a
WMI fallback. Nothing here assumes a particular CPU model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from inferenceos.contracts.hardware import CacheProfile, CPUProfile
from inferenceos.native import windows

INSTRUCTION_SET_ORDER: tuple[str, ...] = (
    "mmx",
    "sse",
    "sse2",
    "sse3",
    "ssse3",
    "sse4_1",
    "sse4_2",
    "avx",
    "fma",
    "f16c",
    "avx2",
    "avx512f",
    "avx512dq",
    "avx512bw",
    "avx512vl",
    "avx512vnni",
    "bmi1",
    "bmi2",
    "popcnt",
    "aes",
)

_AVX_DEPENDENT = {
    "avx",
    "fma",
    "f16c",
    "avx2",
    "avx512f",
    "avx512dq",
    "avx512bw",
    "avx512vl",
    "avx512vnni",
}

_ARCHITECTURE_NAMES = {
    0: "x86",
    1: "mips",
    2: "alpha",
    3: "powerpc",
    5: "arm",
    6: "ia64",
    9: "x86_64",
    12: "arm64",
}

_SLOT_RE = re.compile(r"([\d.]+)\s*GHz", re.IGNORECASE)


def normalize_instruction_sets(
    cpuid: Mapping[str, bool],
    os_flags: Mapping[str, bool] | None = None,
) -> list[str]:
    """Return a stable, canonical list of real CPU instruction sets.

    CPUID is authoritative for the feature bits; the OS flags gate the AVX family
    because VEX/EVEX execution requires the OS to have enabled extended state
    (XSAVE/OSXSAVE). If no CPUID data is available the OS flags are used directly,
    and if neither source is available the result is empty (never invented).
    """
    os_flags = os_flags or {}
    if cpuid:
        enabled = {name for name, present in cpuid.items() if present}
        avx_ok = bool(cpuid.get("avx")) and (not os_flags or bool(os_flags.get("avx", True)))
        if cpuid.get("avx") and os_flags and not os_flags.get("avx", False):
            avx_ok = False
        if not avx_ok:
            enabled -= _AVX_DEPENDENT
    else:
        enabled = {name for name, present in os_flags.items() if present}
    return [name for name in INSTRUCTION_SET_ORDER if name in enabled]


def caches_from_cpuid(descriptors: Sequence[Mapping[str, int]]) -> CacheProfile:
    l1 = l2 = l3 = 0
    for entry in descriptors:
        level = int(entry.get("level", 0))
        size = int(entry.get("size_bytes", 0))
        if level == 1:
            l1 += size
        elif level == 2:
            l2 += size
        elif level >= 3:
            l3 += size
    return CacheProfile(
        l1_bytes=l1 or None,
        l2_bytes=l2 or None,
        l3_bytes=l3 or None,
    )


def caches_from_wmi(processor: Mapping[str, Any]) -> CacheProfile:
    l2_kb = _as_int(processor.get("L2CacheSize"))
    l3_kb = _as_int(processor.get("L3CacheSize"))
    return CacheProfile(
        l1_bytes=None,
        l2_bytes=l2_kb * 1024 if l2_kb else None,
        l3_bytes=l3_kb * 1024 if l3_kb else None,
    )


def architecture_from_code(code: Any) -> str:
    parsed = _as_int(code)
    return _ARCHITECTURE_NAMES.get(parsed or -1, "unknown")


def base_clock_from_name(name: str) -> int | None:
    match = _SLOT_RE.search(name)
    if not match:
        return None
    try:
        return int(round(float(match.group(1)) * 1000))
    except ValueError:  # pragma: no cover - defensive
        return None


def build_cpu_profile(
    processors: Sequence[Mapping[str, Any]],
    *,
    cpuid: Mapping[str, bool],
    os_flags: Mapping[str, bool],
    cache_descriptors: Sequence[Mapping[str, int]] = (),
) -> CPUProfile:
    """Build a :class:`CPUProfile` from real probe data."""
    if not processors:
        raise ValueError("no processor data available")
    processor = processors[0]
    name = str(processor.get("Name") or "Unknown CPU")
    physical = _as_int(processor.get("NumberOfCores")) or 1
    logical = _as_int(processor.get("NumberOfLogicalProcessors")) or physical
    caches = caches_from_cpuid(cache_descriptors)
    if caches.l1_bytes is None and caches.l2_bytes is None and caches.l3_bytes is None:
        caches = caches_from_wmi(processor)
    bases = [
        value
        for value in (
            _as_int(p.get("CurrentClockSpeed")) for p in processors
        )
        if value
    ]
    max_clock = max((_as_int(p.get("MaxClockSpeed")) or 0 for p in processors), default=0) or None
    return CPUProfile(
        name=name,
        vendor=_string_or_none(processor.get("Manufacturer")),
        architecture=architecture_from_code(processor.get("Architecture")),
        physical_cores=max(physical, 1),
        logical_cores=max(logical, 1),
        base_clock_mhz=base_clock_from_name(name) or (min(bases) if bases else None),
        max_clock_mhz=max_clock,
        caches=caches,
        instruction_sets=normalize_instruction_sets(cpuid, os_flags),
        detected_via="cpuid+IsProcessorFeaturePresent",
    )


def detect_cpu() -> CPUProfile:
    snapshot = windows.collect_cim_snapshot()
    return build_cpu_profile(
        snapshot.processors,
        cpuid=windows.cpuid_features(),
        os_flags=windows.os_cpu_features(),
        cache_descriptors=windows.cpuid_cache_info(),
    )


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
