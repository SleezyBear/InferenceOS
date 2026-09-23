"""Real-machine integration checks for Phase 2 (metadata + adapter + KV).

These run against the actual GGUF files in ``models/`` and the real llama.cpp
build. They assert behavioural invariants, not machine-specific constants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.backends.llama_cpp_toolchain import find_cli
from inferenceos.config import Settings
from inferenceos.contracts.enums import AttentionType, BackendType
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
from inferenceos.models.loader import load_model_profile

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATHS = sorted(MODELS_DIR.glob("*.gguf"))

pytestmark = pytest.mark.skipif(
    len(MODEL_PATHS) < 2, reason="Phase 2 integration needs at least two real GGUF models"
)


def _llama_available() -> bool:
    try:
        find_cli(Settings().llama_cpp_dir)
    except FileNotFoundError:
        return False
    return True


@pytest.fixture(scope="module")
def profiles() -> list[ModelProfile]:
    return [load_model_profile(path) for path in MODEL_PATHS]


def test_real_models_parse(profiles: list[ModelProfile]) -> None:
    assert len(profiles) >= 2
    for model in profiles:
        assert model.format == "gguf"
        assert model.architecture != "unknown"
        assert model.n_layers >= 1
        assert model.n_attention_heads >= 1
        assert model.n_kv_heads >= 1
        assert model.head_dim >= 1
        assert model.trained_context_length is not None
        assert model.file_size_bytes > 0
        assert model.kv_cache_bytes_per_token > 0


def test_real_models_cover_mha_and_gqa(profiles: list[ModelProfile]) -> None:
    attention_types = {model.attention_type for model in profiles}
    assert AttentionType.MHA in attention_types
    assert AttentionType.GQA in attention_types or AttentionType.MQA in attention_types


def test_real_kv_estimate_uses_kv_heads(profiles: list[ModelProfile]) -> None:
    for model in profiles:
        key_length = model.key_length or model.head_dim
        value_length = model.value_length or model.head_dim
        expected = model.n_layers * model.n_kv_heads * (key_length + value_length) * 2
        assert model.kv_cache_bytes_per_token == expected


@pytest.mark.skipif(not _llama_available(), reason="llama.cpp toolchain not present")
def test_real_adapter_capabilities() -> None:
    capabilities = LlamaCppAdapter().capabilities()
    assert capabilities.name == "llama.cpp"
    assert BackendType.CPU in capabilities.backends
    assert capabilities.devices


@pytest.mark.skipif(not _llama_available(), reason="llama.cpp toolchain not present")
def test_real_predicted_matches_reported_kv(profiles: list[ModelProfile]) -> None:
    model = min(profiles, key=lambda item: item.file_size_bytes)
    context = 256
    adapter = LlamaCppAdapter()
    adapter.load(
        model,
        RuntimeParams(
            gpu_offload_layers=0, device="none", use_mmap=False, context_length=context
        ),
    )
    try:
        adapter.generate(GenerationRequest(prompt="Hello", n_predict=1, seed=1234))
        telemetry = adapter.telemetry()
        last_run = adapter.last_run
    finally:
        adapter.unload()

    assert last_run is not None
    assert last_run.returncode == 0
    reported = telemetry.kv_cache_bytes
    if reported:
        predicted = model.estimate_kv_cache_bytes(context)
        assert abs(predicted - reported) / reported <= 0.05
