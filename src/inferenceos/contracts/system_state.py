"""Live system-state contracts.

Phase 1 fills these from real OS counters. Note the Windows-specific pressure
signal: committed-vs-available memory and hard page faults, not a Linux
swap/swappiness analogy.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import PowerState


class ProcessState(ContractBase):
    pid: int = Field(ge=0)
    name: str
    cpu_percent: float = Field(ge=0.0)
    rss_bytes: int = Field(ge=0)
    vms_bytes: int | None = None


class GPUEngineUtilization(ContractBase):
    device_index: int = Field(ge=0)
    engine: str
    utilization_percent: float = Field(ge=0.0)


class GPUMemoryState(ContractBase):
    device_index: int = Field(ge=0)
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None


class SystemState(ContractBase):
    captured_at: datetime
    total_ram_bytes: int = Field(ge=0)
    available_ram_bytes: int = Field(ge=0)
    committed_bytes: int | None = None
    commit_limit_bytes: int | None = None
    # ``memory_pressure`` is the primary pressure signal and matches the Phase 3
    # ``compute_memory_pressure`` semantics: Windows commit pressure
    # (committed/commit-limit) when commit information is available, otherwise the
    # physical utilization fallback (1 - available/total). The two underlying
    # concepts are exposed separately below so a consumer can distinguish them.
    memory_pressure: float = Field(ge=0.0, le=1.0)
    # Physical RAM utilization (1 - available_physical/total_physical). Always
    # available; never conflated with commit pressure.
    physical_memory_pressure: float | None = Field(default=None, ge=0.0, le=1.0)
    # Windows commit pressure (committed_bytes / commit_limit_bytes). ``None`` when
    # the OS did not expose commit information; never fabricated from physical RAM.
    commit_pressure: float | None = Field(default=None, ge=0.0, le=1.0)
    cpu_percent_total: float = Field(ge=0.0)
    hard_page_faults_per_sec: float = Field(ge=0.0)
    # Paging telemetry provenance, mirroring the GPU provenance fields below. A
    # genuine zero hard-fault rate is only distinguishable from missing telemetry via
    # ``paging_telemetry_available``; ``hard_page_faults_per_sec`` is 0.0 when
    # unavailable and must not be read as a measured zero.
    paging_telemetry_available: bool = False
    paging_observed_at: datetime | None = None
    paging_source: str | None = None
    processes: list[ProcessState] = Field(default_factory=list)
    gpu_engines: list[GPUEngineUtilization] = Field(default_factory=list)
    gpu_memory: list[GPUMemoryState] = Field(default_factory=list)
    # GPU telemetry provenance/freshness. ``gpu_engines``/``gpu_memory`` may be a cached
    # snapshot taken at ``gpu_observed_at``; consumers must not treat cached values as a
    # fresh observation. ``gpu_telemetry_available`` is False (with a reason in
    # ``gpu_source``) rather than presenting fabricated zeros when collection fails.
    gpu_telemetry_available: bool = False
    gpu_observed_at: datetime | None = None
    gpu_source: str | None = None
    target_process: ProcessState | None = None

    # B1 measurement-validity power/performance signals (all optional; ``None``/
    # ``UNKNOWN`` means the platform did not expose the value, never fabricated).
    power_state: PowerState = PowerState.UNKNOWN
    cpu_current_mhz: int | None = None
    cpu_max_mhz: int | None = None
    # Current power-state ceiling (``MhzLimit``) when the platform exposes it.
    cpu_limit_mhz: int | None = None
    # Observed current / (limit or max) clock ratio; directly observed, not inferred.
    cpu_performance_ratio: float | None = None
