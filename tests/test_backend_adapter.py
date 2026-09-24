"""llama.cpp adapter contract behavior and CLI flag mapping (no inference)."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.backends.llama_cpp_toolchain import build_run_command, parse_llama_memory
from inferenceos.contracts.enums import QuantizationType
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendTelemetry,
    RuntimeParams,
)


def _model(path: str) -> ModelProfile:
    return ModelProfile(
        path=path,
        name="synthetic",
        format="gguf",
        architecture="synthetic",
        quantization=QuantizationType.Q4_K_M,
        file_size_bytes=10,
        n_layers=4,
        n_attention_heads=4,
        n_kv_heads=2,
        head_dim=64,
        key_length=64,
        value_length=64,
        weight_bytes=10,
        kv_cache_bytes_per_token=4096,
    )


def test_adapter_implements_interface() -> None:
    adapter = LlamaCppAdapter(cli=Path("unused-llama-cli"))
    assert isinstance(adapter, BackendAdapter)
    assert adapter.is_loaded is False


def test_load_requires_existing_file(tmp_path: Path) -> None:
    adapter = LlamaCppAdapter(cli=Path("unused-llama-cli"))
    with pytest.raises(FileNotFoundError):
        adapter.load(_model(str(tmp_path / "missing.gguf")), RuntimeParams())


def test_load_unload_lifecycle(tmp_path: Path) -> None:
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"\x00")
    adapter = LlamaCppAdapter(cli=Path("unused-llama-cli"))
    adapter.load(_model(str(model_file)), RuntimeParams(threads=4))
    assert adapter.is_loaded is True
    adapter.unload()
    assert adapter.is_loaded is False


def test_set_runtime_params_merges_non_none(tmp_path: Path) -> None:
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"\x00")
    adapter = LlamaCppAdapter(cli=Path("unused-llama-cli"))
    adapter.load(_model(str(model_file)), RuntimeParams(threads=4, context_length=1024))
    adapter.set_runtime_params(RuntimeParams(context_length=4096))
    assert adapter.params.threads == 4
    assert adapter.params.context_length == 4096


def test_set_runtime_params_can_clear_value(tmp_path: Path) -> None:
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"\x00")
    adapter = LlamaCppAdapter(cli=Path("unused-llama-cli"))
    adapter.load(_model(str(model_file)), RuntimeParams(threads=4, context_length=1024))
    adapter.set_runtime_params(RuntimeParams(threads=None))
    assert adapter.params.threads is None
    assert adapter.params.context_length == 1024


def test_generate_without_load_raises() -> None:
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    adapter = LlamaCppAdapter(cli=Path("unused-llama-cli"))
    with pytest.raises(RuntimeError):
        adapter.generate(GenerationRequest(prompt="hi"))


def test_telemetry_defaults_are_explicit() -> None:
    adapter = LlamaCppAdapter(cli=Path("unused-llama-cli"))
    telemetry = adapter.telemetry()
    assert isinstance(telemetry, BackendTelemetry)
    assert telemetry.device is None
    assert telemetry.kv_cache_bytes is None


def test_build_run_command_maps_runtime_params(tmp_path: Path) -> None:
    command = build_run_command(
        Path("llama-cli"),
        tmp_path / "m.gguf",
        "hello",
        n_predict=8,
        n_gpu_layers=0,
        context_length=2048,
        batch_size=256,
        ubatch_size=128,
        flash_attention=False,
        device="none",
        load_mode="none",
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )
    joined = " ".join(command)
    assert "-c 2048" in joined
    assert "-b 256" in joined
    assert "-ub 128" in joined
    assert "-fa off" in joined
    assert "-dev none" in joined
    assert "-lm none" in joined
    assert "-ctk q8_0" in joined
    assert "-ctv q8_0" in joined
    assert "--no-warmup" in joined
    assert "-st" in joined


def test_parse_llama_memory() -> None:
    stderr = (
        "0.00.483.800 I llama_kv_cache:        CPU KV buffer size =     0.00 MiB\n"
        "0.00.483.806 I llama_kv_cache: size =   56.00 MiB (  2048 cells,  28 layers,  1/1 seqs)\n"
        "0.01.050.028 I load_tensors:   CPU_Mapped model buffer size =  1059.89 MiB\n"
        "0.01.050.030 I load_tensors:   CPU_REPACK model buffer size =   596.53 MiB\n"
        "0.01.456.600 I llama_context: n_ctx                 = 2048\n"
    )
    report = parse_llama_memory(stderr)
    assert report is not None
    assert report.kv_cache_bytes == 56 * 1024 * 1024
    assert report.model_buffer_bytes == int(round(1059.89 * 1024 * 1024))
    assert report.context_length == 2048
    assert parse_llama_memory("no memory lines here") is None
