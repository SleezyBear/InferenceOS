"""Live system-state contracts.

Phase 1 fills these from real OS counters. Note the Windows-specific pressure
signal: committed-vs-available memory and hard page faults, not a Linux
swap/swappiness analogy.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from inferenceos.contracts.base import ContractBase


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
    memory_pressure: float = Field(ge=0.0, le=1.0)
    cpu_percent_total: float = Field(ge=0.0)
    hard_page_faults_per_sec: float = Field(ge=0.0)
    processes: list[ProcessState] = Field(default_factory=list)
    gpu_engines: list[GPUEngineUtilization] = Field(default_factory=list)
    gpu_memory: list[GPUMemoryState] = Field(default_factory=list)
    target_process: ProcessState | None = None
