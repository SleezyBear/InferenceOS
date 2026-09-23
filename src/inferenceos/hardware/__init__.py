"""Hardware characterization layer.

Platform-specific mechanisms live in :mod:`inferenceos.native`; this package turns
them into the host-independent contracts used by the rest of InferenceOS.
"""

from __future__ import annotations

from inferenceos.hardware.cpu import (
    INSTRUCTION_SET_ORDER,
    build_cpu_profile,
    normalize_instruction_sets,
)
from inferenceos.hardware.gpu import (
    GpuCharacterization,
    build_backend_availability,
    build_gpu_profiles,
    characterize_gpus,
    device_type_from_vulkan,
    vendor_from_pci_id,
)
from inferenceos.hardware.memory import build_memory_profile
from inferenceos.hardware.os_profile import build_os_profile
from inferenceos.hardware.profiler import classify_storage_path, detect_hardware_profile
from inferenceos.hardware.storage import (
    PathStorageInfo,
    build_storage_devices,
    classify_path,
    media_type_from_code,
)

__all__ = [
    "GpuCharacterization",
    "INSTRUCTION_SET_ORDER",
    "PathStorageInfo",
    "build_backend_availability",
    "build_cpu_profile",
    "build_gpu_profiles",
    "build_memory_profile",
    "build_os_profile",
    "build_storage_devices",
    "characterize_gpus",
    "classify_path",
    "classify_storage_path",
    "detect_hardware_profile",
    "device_type_from_vulkan",
    "media_type_from_code",
    "normalize_instruction_sets",
    "vendor_from_pci_id",
]
