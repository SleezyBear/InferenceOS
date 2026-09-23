"""Backend integrations (llama.cpp is the only v1 backend)."""

from __future__ import annotations

from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
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

__all__ = [
    "DeviceInfo",
    "LlamaCppAdapter",
    "LlamaMemoryReport",
    "LlamaRunResult",
    "LlamaTimings",
    "MemoryObservation",
    "cli_version",
    "find_cli",
    "list_devices",
    "parse_device_info",
    "parse_generated_text",
    "parse_llama_memory",
    "parse_llama_timings",
    "parse_timings",
    "run_cli",
]
