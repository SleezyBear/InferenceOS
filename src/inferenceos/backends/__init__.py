"""Backend integrations (llama.cpp is the only v1 backend)."""

from __future__ import annotations

from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.backends.llama_cpp_toolchain import (
    DeviceInfo,
    LlamaMemoryReport,
    LlamaRunResult,
    LlamaTimings,
    MemoryObservation,
    cli_version,
    find_cli,
    list_devices,
    parse_device_info,
    parse_generated_text,
    parse_llama_memory,
    parse_llama_timings,
    parse_timings,
    run_cli,
)
from inferenceos.backends.process_control import (
    ProcessGuard,
    create_process_guard,
    process_identity,
    resource_snapshot,
    terminate_process_tree,
    wait_process_exit,
)
from inferenceos.backends.resident_server import (
    LlamaServerResidentBackend,
    ResidentBackendError,
    ResidentServerBackend,
)

__all__ = [
    "DeviceInfo",
    "LlamaCppAdapter",
    "LlamaCppResidentAdapter",
    "LlamaMemoryReport",
    "LlamaRunResult",
    "LlamaServerResidentBackend",
    "LlamaTimings",
    "MemoryObservation",
    "ProcessGuard",
    "ResidentBackendError",
    "ResidentServerBackend",
    "cli_version",
    "create_process_guard",
    "find_cli",
    "list_devices",
    "parse_device_info",
    "parse_generated_text",
    "parse_llama_memory",
    "parse_llama_timings",
    "parse_timings",
    "process_identity",
    "resource_snapshot",
    "run_cli",
    "terminate_process_tree",
    "wait_process_exit",
]
