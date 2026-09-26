"""System RAM characterization.

Primary source is the Win32 ``GlobalMemoryStatusEx`` API (real physical and
commit figures). Physical-module details come from WMI. Callers must be safe to
invoke repeatedly, so nothing here caches or assumes the machine's RAM size.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from inferenceos.contracts.hardware import MemoryModule, MemoryProfile
from inferenceos.native import get_platform
from inferenceos.native.base import GlobalMemory

try:  # psutil is a hard runtime dependency but keep the import defensive
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


def build_memory_profile(
    memory_status: GlobalMemory | None,
    modules: Sequence[Mapping[str, Any]],
) -> MemoryProfile:
    """Build a :class:`MemoryProfile` from real memory data."""
    total: int
    available: int
    commit_limit: int | None
    committed: int | None
    if memory_status is not None:
        total = memory_status.total_physical_bytes
        available = memory_status.available_physical_bytes
        commit_limit = memory_status.total_pagefile_bytes
        committed = max(memory_status.total_pagefile_bytes - memory_status.available_pagefile_bytes, 0)
    elif psutil is not None:
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        total = int(virtual.total)
        available = int(virtual.available)
        commit_limit = int(swap.total) or None
        committed = int(swap.used) if swap.total else None
    else:  # pragma: no cover - psutil is required in practice
        total = 0
        available = 0
        commit_limit = None
        committed = None

    return MemoryProfile(
        total_bytes=total,
        available_bytes=available,
        committed_bytes=committed,
        commit_limit_bytes=commit_limit,
        modules=[_module_from_wmi(module) for module in modules],
    )


def _module_from_wmi(module: Mapping[str, Any]) -> MemoryModule:
    capacity = _as_int(module.get("Capacity")) or 0
    return MemoryModule(
        capacity_bytes=capacity,
        speed_mhz=_as_int(module.get("Speed")),
        manufacturer=_str_or_none(module.get("Manufacturer")),
    )


def detect_memory() -> MemoryProfile:
    native = get_platform()
    snapshot = native.collect_cim_snapshot()
    return build_memory_profile(native.global_memory_status(), snapshot.physical_memory)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
