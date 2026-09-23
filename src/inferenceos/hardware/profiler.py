"""Hardware characterization orchestrator.

Produces a real :class:`HardwareProfile` from the actual machine by combining the
CPU, GPU/backend, memory, storage and OS probes. The hardware layer describes the
host independently — it does not assume llama.cpp or any other specific backend.
"""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.hardware.cpu import build_cpu_profile
from inferenceos.hardware.gpu import characterize_gpus
from inferenceos.hardware.memory import build_memory_profile
from inferenceos.hardware.os_profile import build_os_profile
from inferenceos.hardware.storage import (
    PathStorageInfo,
    build_storage_devices,
    classify_path,
)
from inferenceos.native import windows


def detect_hardware_profile(settings: Settings | None = None) -> HardwareProfile:
    """Run all real probes and build a :class:`HardwareProfile` for this host."""
    _ = settings  # reserved for future probe configuration
    snapshot = windows.collect_cim_snapshot()

    cpu = build_cpu_profile(
        snapshot.processors,
        cpuid=windows.cpuid_features(),
        os_flags=windows.os_cpu_features(),
        cache_descriptors=windows.cpuid_cache_info(),
    )
    gpu = characterize_gpus(cpu, video_controllers=snapshot.video_controllers)
    memory = build_memory_profile(windows.global_memory_status(), snapshot.physical_memory)
    storage = build_storage_devices(
        snapshot.disk_drives,
        snapshot.physical_disks,
        snapshot.logical_disks,
        snapshot.associations,
    )
    os_profile = build_os_profile(snapshot.os)

    return HardwareProfile(
        captured_at=datetime.now(timezone.utc),
        hostname=socket.gethostname(),
        os=os_profile,
        cpu=cpu,
        gpus=gpu.gpus,
        memory=memory,
        storage=storage,
        backends_available=gpu.backends_available,
        backend_availability=gpu.backend_availability,
        vulkan_available=gpu.vulkan_available,
        cuda_available=gpu.cuda_available,
        rocm_available=gpu.rocm_available,
    )


def classify_storage_path(
    path: Path | str,
    profile: HardwareProfile | None = None,
) -> PathStorageInfo | None:
    """Identify the physical storage backing a path (e.g. a model file)."""
    if profile is None:
        profile = detect_hardware_profile()
    return classify_path(path, profile.storage)
