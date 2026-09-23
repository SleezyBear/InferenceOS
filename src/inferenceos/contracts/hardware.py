"""Hardware contracts.

Phase 1 populates these from real probes (CPUID-equivalent, Vulkan device
enumeration, Win32 memory APIs). Phase 0 defines the shape only.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import (
    BackendType,
    GPUDeviceType,
    GPUVendor,
    StorageMediaType,
)


class CacheProfile(ContractBase):
    l1_bytes: int | None = None
    l2_bytes: int | None = None
    l3_bytes: int | None = None


class CPUProfile(ContractBase):
    name: str
    vendor: str | None = None
    architecture: str | None = None
    physical_cores: int = Field(ge=1)
    logical_cores: int = Field(ge=1)
    base_clock_mhz: int | None = None
    max_clock_mhz: int | None = None
    caches: CacheProfile = Field(default_factory=CacheProfile)
    instruction_sets: list[str] = Field(default_factory=list)
    detected_via: str = "unknown"


class GPUProfile(ContractBase):
    name: str
    vendor: GPUVendor = GPUVendor.UNKNOWN
    device_type: GPUDeviceType = GPUDeviceType.UNKNOWN
    vram_bytes: int | None = None
    driver_version: str | None = None
    driver_date: str | None = None
    supported_backends: list[BackendType] = Field(default_factory=list)
    vulkan_device_index: int | None = None
    vulkan_api_version: str | None = None
    pci_id: str | None = None


class MemoryModule(ContractBase):
    capacity_bytes: int = Field(ge=0)
    speed_mhz: int | None = None
    manufacturer: str | None = None


class MemoryProfile(ContractBase):
    total_bytes: int = Field(ge=0)
    available_bytes: int = Field(ge=0)
    committed_bytes: int | None = None
    commit_limit_bytes: int | None = None
    modules: list[MemoryModule] = Field(default_factory=list)


class StorageDevice(ContractBase):
    name: str
    media_type: StorageMediaType = StorageMediaType.UNKNOWN
    bus_type: str | None = None
    size_bytes: int = Field(ge=0)
    free_bytes: int | None = None
    is_removable: bool = False
    volumes: list[str] = Field(default_factory=list)


class OSProfile(ContractBase):
    caption: str
    version: str
    build: str
    architecture: str


class HardwareProfile(ContractBase):
    captured_at: datetime
    hostname: str
    os: OSProfile
    cpu: CPUProfile
    gpus: list[GPUProfile] = Field(default_factory=list)
    memory: MemoryProfile
    storage: list[StorageDevice] = Field(default_factory=list)
    backends_available: list[BackendType] = Field(default_factory=list)
    vulkan_available: bool = False
    cuda_available: bool = False
    rocm_available: bool = False
