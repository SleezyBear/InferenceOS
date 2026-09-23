"""Deterministic parsers for real llama.cpp output (captured samples)."""

from __future__ import annotations

from inferenceos.backends.llama_cpp_toolchain import (
    backend_type_from_name,
    parse_device_info,
    parse_generated_text,
    parse_timings,
)
from inferenceos.contracts.enums import BackendType

DEVICE_STDERR = """
0.00.101.451 I cmn  common_param:   - Vulkan0 : Radeon Pro 560X (4096 MiB, 3433 MiB free)
0.00.286.038 I llama_prepare_model_devices: using device Vulkan0 (Radeon Pro 560X) (unknown id) - 3433 MiB free
0.00.500.937 I load_tensors: offloaded 29/29 layers to GPU
"""

OUTPUT_STDOUT = """
> Say hi in three words.
Hello there.

[ Prompt: 85.0 t/s | Generation: 24.1 t/s ]

Exiting...
"""


def test_parse_device_info_columns() -> None:
    info = parse_device_info(DEVICE_STDERR)
    assert info is not None
    assert info.backend == "Vulkan0"
    assert info.device == "Radeon Pro 560X"
    assert info.layers_offloaded == 29
    assert info.layers_total == 29
    assert info.vram_total_bytes == 4096 * 1024 * 1024
    assert info.vram_free_bytes == 3433 * 1024 * 1024


def test_parse_device_info_absent() -> None:
    assert parse_device_info("nothing to see") is None


def test_parse_generated_text() -> None:
    assert parse_generated_text(OUTPUT_STDOUT) == "Hello there."


def test_parse_timings() -> None:
    prompt_tps, gen_tps = parse_timings(OUTPUT_STDOUT)
    assert prompt_tps == 85.0
    assert gen_tps == 24.1


def test_backend_type_mapping() -> None:
    assert backend_type_from_name("Vulkan0") is BackendType.VULKAN
    assert backend_type_from_name("CUDA0") is BackendType.CUDA
    assert backend_type_from_name("CPU") is BackendType.CPU
    assert backend_type_from_name("something-else") is BackendType.CPU
