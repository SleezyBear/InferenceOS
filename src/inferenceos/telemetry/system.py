"""Live process / system telemetry.

Real per-process CPU and RSS (psutil), real physical/commit memory
(``GlobalMemoryStatusEx``), real hard page-fault rate (``\\Memory\\Pages Input/sec``),
and real GPU engine utilization / adapter memory (Windows GPU performance
counters). Unavailable signals are represented as ``None`` rather than fabricated.

Note on the Windows pressure signal: Windows has no swap/swappiness equivalent,
so committed-vs-available memory and hard page-fault counters are used instead of a
Linux analogy.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.system_state import (
    GPUEngineUtilization,
    GPUMemoryState,
    ProcessState,
    SystemState,
)
from inferenceos.native import get_platform
from inferenceos.native.base import CounterSample, VulkanDevice

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

HARD_FAULT_COUNTER = r"\Memory\Pages Input/sec"
GPU_ENGINE_COUNTER = r"\GPU Engine(*)\Utilization Percentage"
GPU_ADAPTER_MEMORY_COUNTER = r"\GPU Adapter Memory(*)\Dedicated Usage"

_ENGINE_INSTANCE_RE = re.compile(
    r"luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)_phys_(\d+)_eng_(\d+)_engtype_(\w+)"
)
_ADAPTER_INSTANCE_RE = re.compile(r"luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)_phys_(\d+)")


@dataclass(frozen=True)
class _ProcSnapshot:
    name: str
    rss_bytes: int
    vms_bytes: int
    cpu_seconds: float


@dataclass(frozen=True)
class _GpuAggregate:
    engines: list[GPUEngineUtilization]
    memory: list[GPUMemoryState]


@dataclass(frozen=True)
class PowerPerformance:
    """Aggregated B1 power/performance observation (all fields optional)."""

    power_state: PowerState = PowerState.UNKNOWN
    current_mhz: int | None = None
    max_mhz: int | None = None
    limit_mhz: int | None = None
    ratio: float | None = None


def observe_power_performance() -> PowerPerformance:
    """Observe real AC/battery state and aggregate processor clock performance.

    The ratio is ``current / (power-state ceiling or nominal max)`` and is clamped to
    ``[0, 1]`` (a processor cannot exceed its available ceiling). Unavailable values
    stay ``None``/``UNKNOWN``; they are never assumed healthy.
    """
    native = get_platform()
    status = native.power_status()
    if status is None:
        power = PowerState.UNKNOWN
    elif status.ac_line is True:
        power = PowerState.AC
    elif status.ac_line is False:
        power = PowerState.BATTERY
    elif status.battery_present is False:
        # Mains-only machine: no battery and no explicit AC-line flag.
        power = PowerState.AC
    else:
        power = PowerState.UNKNOWN

    entries = native.processor_power()
    if not entries:
        return PowerPerformance(power_state=power)

    currents = [entry.current_mhz for entry in entries if entry.current_mhz > 0]
    maxes = [entry.max_mhz for entry in entries if entry.max_mhz > 0]
    limits = [entry.limit_mhz for entry in entries if 0 < entry.limit_mhz]
    current = round(sum(currents) / len(currents)) if currents else None
    max_mhz = max(maxes) if maxes else None
    limit_mhz = min(limits) if limits else None
    reference: int | None
    if limit_mhz is not None and max_mhz is not None and 0 < limit_mhz <= max_mhz:
        reference = limit_mhz
    else:
        reference = max_mhz
    ratio = (
        min(max(current / reference, 0.0), 1.0)
        if current is not None and reference
        else None
    )
    return PowerPerformance(
        power_state=power,
        current_mhz=current,
        max_mhz=max_mhz,
        limit_mhz=limit_mhz,
        ratio=ratio,
    )


def compute_memory_pressure(
    total_bytes: int,
    available_bytes: int,
    committed_bytes: int | None,
    commit_limit_bytes: int | None,
) -> float:
    """Primary memory pressure: Windows commit pressure when available.

    This preserves the original Phase 3 semantics (commit-based primary signal with a
    physical-utilization fallback). Use :func:`compute_physical_pressure` and
    :func:`compute_commit_pressure` when the two must be distinguished explicitly.
    """
    commit = compute_commit_pressure(committed_bytes, commit_limit_bytes)
    if commit is not None:
        return commit
    return compute_physical_pressure(total_bytes, available_bytes)


def compute_physical_pressure(total_bytes: int, available_bytes: int) -> float:
    """Physical RAM utilization (0..1): ``1 - available/total``."""
    if total_bytes > 0:
        return min(max(1.0 - available_bytes / total_bytes, 0.0), 1.0)
    return 0.0


def compute_commit_pressure(
    committed_bytes: int | None, commit_limit_bytes: int | None
) -> float | None:
    """Windows commit pressure (0..1), or ``None`` when commit info is unavailable.

    ``None`` (not 0.0) means the OS did not expose commit usage/limit; it is never
    fabricated from physical RAM.
    """
    if commit_limit_bytes and committed_bytes is not None and commit_limit_bytes > 0:
        return min(max(committed_bytes / commit_limit_bytes, 0.0), 1.0)
    return None


def parse_gpu_engine_instance(instance: str) -> tuple[str, int, str] | None:
    match = _ENGINE_INSTANCE_RE.search(instance)
    if not match:
        return None
    luid = f"{match.group(1).lower()}:{match.group(2).lower()}"
    return luid, int(match.group(3)), match.group(5)


def parse_gpu_adapter_instance(instance: str) -> str | None:
    match = _ADAPTER_INSTANCE_RE.search(instance)
    if not match:
        return None
    return f"{match.group(1).lower()}:{match.group(2).lower()}"


def map_luids_to_device_indices(
    luids: Sequence[str],
    vulkan_devices: Sequence[VulkanDevice] | None = None,
) -> dict[str, int]:
    """Map telemetry LUID strings to real Vulkan device indices by LUID identity.

    When Vulkan physical devices with valid LUIDs are available (Windows), a telemetry
    LUID is assigned the *matching* device's real Vulkan index regardless of telemetry
    order. Unmatched telemetry LUIDs receive a distinct index beyond the Vulkan range —
    they are never attributed to a different real device. When no Vulkan LUID identity
    is available at all, the original appearance-order fallback is preserved.
    """
    mapping: dict[str, int] = {}
    matched: dict[str, int] = {}
    # Only consult the native platform when there are device identities to match;
    # the appearance-order fallback needs nothing platform-specific.
    native = get_platform() if vulkan_devices else None
    if vulkan_devices and native is not None:
        for device in vulkan_devices:
            if device.luid:
                matched[native.normalize_luid(device.luid)] = device.index
    if matched and native is not None:
        next_index = max(matched.values()) + 1
        for luid in luids:
            key = native.normalize_luid(luid)
            if key in matched:
                mapping[luid] = matched[key]
            else:
                mapping[luid] = next_index
                next_index += 1
        return mapping
    for position, luid in enumerate(luids):
        mapping[luid] = position
    return mapping


def aggregate_gpu_samples(
    samples: list[CounterSample],
    *,
    vulkan_devices: Sequence[VulkanDevice] | None = None,
) -> _GpuAggregate:
    """Aggregate raw counter samples into per-engine utilization and memory.

    ``device_index`` is the real Vulkan device index when the telemetry LUID matches a
    Vulkan physical device's LUID; otherwise it degrades to appearance order, and an
    unmatched LUID is kept distinct rather than misattributed.

    Utilization is counted without double-counting:

    * the same counter instance seen several times (one per sweep, when a caller
      passes everything collected since its last check) is averaged, not summed;
    * different processes on one physical engine are summed (they share it);
    * separate engines of the same type on one adapter report the busiest engine,
      not the sum (the same convention Windows Task Manager uses).
    """
    # instance -> (sum of readings, number of readings), to average repeated sweeps
    per_instance: dict[str, tuple[float, int]] = {}
    # instance -> (luid, physical adapter, engine index, engine type)
    instance_engine: dict[str, tuple[str, int, int, str]] = {}
    memory_used: dict[str, float] = {}
    luid_seen: list[str] = []

    def _note(luid: str) -> None:
        if luid not in luid_seen:
            luid_seen.append(luid)

    for sample in samples:
        text = f"{sample.path}|{sample.instance or ''}".lower()
        if "gpu engine" in text and "utilization" in text and sample.instance:
            parsed = parse_gpu_engine_instance(sample.instance)
            match = _ENGINE_INSTANCE_RE.search(sample.instance)
            if parsed is None or match is None:
                continue
            luid, phys, engine_type = parsed
            _note(luid)
            instance_key = sample.instance.lower()
            instance_engine[instance_key] = (luid, phys, int(match.group(4)), engine_type)
            total, count = per_instance.get(instance_key, (0.0, 0))
            per_instance[instance_key] = (total + max(sample.value, 0.0), count + 1)
        elif "gpu adapter memory" in text and sample.instance:
            adapter_luid = parse_gpu_adapter_instance(sample.instance)
            if adapter_luid is None:
                continue
            _note(adapter_luid)
            memory_used[adapter_luid] = max(sample.value, 0.0)

    # Sum processes sharing one physical engine (each process averaged over sweeps).
    per_engine: dict[tuple[str, int, int, str], float] = {}
    for instance_key, (total, count) in per_instance.items():
        engine_key = instance_engine[instance_key]
        per_engine[engine_key] = per_engine.get(engine_key, 0.0) + total / count
    # Report each engine type by its busiest engine on the adapter.
    engine_totals: dict[tuple[str, str], float] = {}
    for (luid, _phys, _engine_index, engine_type), busy in per_engine.items():
        key = (luid, engine_type)
        engine_totals[key] = max(engine_totals.get(key, 0.0), busy)

    index_by_luid = map_luids_to_device_indices(luid_seen, vulkan_devices)
    engines = [
        GPUEngineUtilization(
            device_index=index_by_luid[luid],
            engine=engine_type,
            utilization_percent=min(total, 100.0),
        )
        for (luid, engine_type), total in sorted(
            engine_totals.items(), key=lambda item: (index_by_luid[item[0][0]], item[0][1])
        )
    ]
    memory = [
        GPUMemoryState(
            device_index=index_by_luid[luid],
            memory_used_bytes=int(used),
            memory_total_bytes=None,
        )
        for luid, used in sorted(memory_used.items(), key=lambda item: index_by_luid[item[0]])
    ]
    return _GpuAggregate(engines=engines, memory=memory)


def hard_faults_from_samples(samples: list[CounterSample]) -> float:
    for sample in samples:
        if "pages input" in sample.path.lower():
            return max(sample.value, 0.0)
    return 0.0


def latest_hard_faults_from_samples(samples: list[CounterSample]) -> float | None:
    """The most recent ``\\Memory\\Pages Input/sec`` reading, or ``None`` if absent.

    Unlike :func:`hard_faults_from_samples` (Phase 3, which returns 0.0 when the
    counter is missing), this distinguishes *unavailable* telemetry (``None``) from a
    genuine measured zero (``0.0``) — required by the runtime paging sampler.
    """
    value: float | None = None
    for sample in samples:
        if "pages input" in sample.path.lower():
            value = max(sample.value, 0.0)
    return value


def _process_snapshot(processes: list[Any]) -> dict[int, _ProcSnapshot]:
    snapshot: dict[int, _ProcSnapshot] = {}
    for process in processes:
        try:
            info = process.info
            pid = int(info["pid"])
            mem = info.get("memory_info")
            cpu = info.get("cpu_times")
            if mem is None or cpu is None:
                continue
            snapshot[pid] = _ProcSnapshot(
                name=str(info.get("name") or f"pid-{pid}"),
                rss_bytes=int(mem.rss),
                vms_bytes=int(getattr(mem, "vms", 0) or 0),
                cpu_seconds=float(getattr(cpu, "user", 0.0) or 0.0)
                + float(getattr(cpu, "system", 0.0) or 0.0),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError, ValueError):
            continue
    return snapshot


def sample_system_state(
    *,
    target_pid: int | None = None,
    interval_s: float = 0.5,
    top_processes: int = 8,
    include_gpu: bool = True,
    vulkan_devices: Sequence[VulkanDevice] | None = None,
) -> SystemState:
    """Capture a real live :class:`SystemState` snapshot.

    When GPU telemetry is included, ``vulkan_devices`` (defaults to a fresh real
    Vulkan enumeration) supplies the LUID identity used to map GPU counter LUIDs to
    the correct Vulkan ``device_index``.
    """
    if psutil is None:  # pragma: no cover - psutil is required in practice
        raise RuntimeError("psutil is required for live telemetry")

    native = get_platform()
    psutil.cpu_percent(interval=None)
    attrs = ["pid", "name", "memory_info", "cpu_times"]
    before = _process_snapshot(list(psutil.process_iter(attrs=attrs)))

    paths = [HARD_FAULT_COUNTER]
    if include_gpu:
        paths += [GPU_ENGINE_COUNTER, GPU_ADAPTER_MEMORY_COUNTER]

    started = time.perf_counter()
    samples = native.sample_perf_counters(paths, warmup_s=interval_s)
    elapsed = max(time.perf_counter() - started, 1e-6)

    after = _process_snapshot(list(psutil.process_iter(attrs=attrs)))
    total_cpu = float(psutil.cpu_percent(interval=None))
    memory = native.global_memory_status()

    processes: list[ProcessState] = []
    for pid, current in after.items():
        previous = before.get(pid)
        cpu_percent = 0.0
        if previous is not None:
            delta = current.cpu_seconds - previous.cpu_seconds
            cpu_percent = max(delta / elapsed * 100.0, 0.0)
        processes.append(
            ProcessState(
                pid=pid,
                name=current.name,
                cpu_percent=cpu_percent,
                rss_bytes=current.rss_bytes,
                vms_bytes=current.vms_bytes,
            )
        )
    processes.sort(key=lambda state: state.rss_bytes, reverse=True)
    top = processes[: max(top_processes, 1)]

    target: ProcessState | None = None
    if target_pid is not None:
        target = next((state for state in processes if state.pid == target_pid), None)

    if include_gpu:
        identity = vulkan_devices if vulkan_devices is not None else native.vulkan_enumerate()
        gpu = aggregate_gpu_samples(samples, vulkan_devices=identity)
    else:
        gpu = _GpuAggregate([], [])
    power = observe_power_performance()

    if memory is not None:
        total = memory.total_physical_bytes
        available = memory.available_physical_bytes
        committed = max(memory.total_pagefile_bytes - memory.available_pagefile_bytes, 0)
        commit_limit = memory.total_pagefile_bytes
    else:  # pragma: no cover - API available on supported Windows
        virtual = psutil.virtual_memory()
        total = int(virtual.total)
        available = int(virtual.available)
        committed = None
        commit_limit = None

    hard_faults = latest_hard_faults_from_samples(samples)

    return SystemState(
        captured_at=datetime.now(timezone.utc),
        total_ram_bytes=total,
        available_ram_bytes=available,
        committed_bytes=committed,
        commit_limit_bytes=commit_limit,
        memory_pressure=compute_memory_pressure(total, available, committed, commit_limit),
        physical_memory_pressure=compute_physical_pressure(total, available),
        commit_pressure=compute_commit_pressure(committed, commit_limit),
        cpu_percent_total=total_cpu,
        hard_page_faults_per_sec=hard_faults if hard_faults is not None else 0.0,
        paging_telemetry_available=hard_faults is not None,
        paging_observed_at=datetime.now(timezone.utc) if hard_faults is not None else None,
        paging_source=(
            "windows_memory_pages_input_counter"
            if hard_faults is not None
            else "unavailable: pages input counter not returned"
        ),
        processes=top,
        gpu_engines=gpu.engines,
        gpu_memory=gpu.memory,
        gpu_telemetry_available=bool(gpu.engines or gpu.memory),
        gpu_observed_at=datetime.now(timezone.utc) if (gpu.engines or gpu.memory) else None,
        gpu_source="windows_gpu_engine_counters" if (gpu.engines or gpu.memory) else "unavailable",
        target_process=target,
        power_state=power.power_state,
        cpu_current_mhz=power.current_mhz,
        cpu_max_mhz=power.max_mhz,
        cpu_limit_mhz=power.limit_mhz,
        cpu_performance_ratio=power.ratio,
    )
