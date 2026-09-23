"""Real-machine integration checks for the Phase 1 characterization layer.

These exercise the actual probes against the running Windows host. They assert
behavioural invariants (not this development machine's specific values), so they
remain valid on any host meeting the project's platform baseline.
"""

from __future__ import annotations

import os
from pathlib import Path

import psutil
import pytest

from inferenceos.contracts.enums import BackendType, StorageMediaType
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.hardware.profiler import classify_storage_path, detect_hardware_profile
from inferenceos.selftest import sample_hardware
from inferenceos.telemetry.system import sample_system_state

pytestmark = pytest.mark.skipif(
    os.name != "nt", reason="Phase 1 characterization targets Windows"
)


@pytest.fixture(scope="module")
def profile() -> HardwareProfile:
    return detect_hardware_profile()


def test_profile_has_required_sections(profile: HardwareProfile) -> None:
    assert profile.hostname
    assert profile.os.caption
    assert profile.cpu.name
    assert profile.cpu.physical_cores >= 1
    assert profile.cpu.logical_cores >= profile.cpu.physical_cores
    assert profile.memory.total_bytes > 0
    assert profile.storage
    assert profile.backends_available


def test_cpu_matches_psutil(profile: HardwareProfile) -> None:
    logical = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    if logical is not None:
        assert profile.cpu.logical_cores == logical
    if physical is not None:
        assert profile.cpu.physical_cores == physical


def test_instruction_sets_detected(profile: HardwareProfile) -> None:
    assert profile.cpu.instruction_sets
    assert "sse2" in profile.cpu.instruction_sets
    assert profile.cpu.detected_via != "unknown"


def test_backend_usability_is_consistent(profile: HardwareProfile) -> None:
    for availability in profile.backend_availability:
        if availability.usable:
            assert availability.detected
    by_backend = {a.backend: a for a in profile.backend_availability}
    assert by_backend[BackendType.CPU].usable
    assert by_backend[BackendType.CUDA].usable == profile.cuda_available
    assert by_backend[BackendType.ROCM].usable == profile.rocm_available
    vulkan = by_backend[BackendType.VULKAN]
    assert vulkan.usable == profile.vulkan_available
    if vulkan.usable:
        assert vulkan.devices
        assert any(BackendType.VULKAN in gpu.supported_backends for gpu in profile.gpus)


def test_ram_matches_psutil(profile: HardwareProfile) -> None:
    total = psutil.virtual_memory().total
    assert abs(profile.memory.total_bytes - total) / total <= 0.05


def test_storage_path_resolution(profile: HardwareProfile) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    info = classify_storage_path(repo_root, profile)
    assert info is not None
    assert info.device_name
    assert info.media_type in (StorageMediaType.SSD, StorageMediaType.HDD)


def test_profile_json_roundtrip(profile: HardwareProfile) -> None:
    restored = HardwareProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile


def test_sample_hardware_contract_includes_backend_availability() -> None:
    sample = sample_hardware()
    assert sample.backend_availability
    restored = HardwareProfile.model_validate_json(sample.model_dump_json())
    assert restored == sample


def test_live_system_state_snapshot() -> None:
    state = sample_system_state(
        target_pid=os.getpid(), interval_s=0.3, top_processes=5, include_gpu=False
    )
    assert state.total_ram_bytes > 0
    assert 0 <= state.available_ram_bytes <= state.total_ram_bytes
    assert 0.0 <= state.memory_pressure <= 1.0
    assert state.processes
    assert state.target_process is not None
    assert state.target_process.pid == os.getpid()
