"""ModelProfile construction from GGUF metadata and contract round-trips."""

from __future__ import annotations

from pathlib import Path

from inferenceos.contracts.enums import QuantizationType
from inferenceos.contracts.model import ModelFootprint, ModelProfile
from inferenceos.models.gguf import GgufArray, GgufFile, GgufTensorInfo
from inferenceos.models.loader import build_model_profile

BASE_METADATA: dict[str, object] = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.context_length": 32768,
    "qwen2.embedding_length": 1536,
    "qwen2.attention.head_count": 12,
    "qwen2.attention.head_count_kv": 2,
    "general.file_type": 15,
    "tokenizer.ggml.tokens": GgufArray(element_type=8, count=151936),
}

TENSORS = (
    GgufTensorInfo(name="blk.0.attn_qkv.weight", dimensions=(4, 8), ggml_type=12, offset=0),
    GgufTensorInfo(name="token_embd.weight", dimensions=(10, 8), ggml_type=12, offset=32),
)


def _gguf(metadata: dict[str, object], name: str = "model.gguf") -> GgufFile:
    return GgufFile(path=Path(name), version=3, metadata=metadata, tensors=TENSORS)


def test_build_profile_aggregates_tensor_parameters() -> None:
    metadata = dict(BASE_METADATA)
    metadata["general.name"] = "my-model"
    profile = build_model_profile(_gguf(metadata), file_size_bytes=1234)

    assert profile.format == "gguf"
    assert profile.name == "my-model"
    assert profile.architecture == "qwen2"
    assert profile.quantization is QuantizationType.Q4_K_M
    assert profile.parameter_count == 4 * 8 + 10 * 8
    assert profile.tensor_count == 2
    assert profile.file_size_bytes == 1234
    assert profile.weight_bytes == 1234
    assert profile.kv_cache_bytes_per_token == 28672
    assert profile.kv_cache_estimates == []
    assert profile.vocabulary_size == 151936


def test_name_falls_back_to_file_stem_when_generic() -> None:
    metadata = dict(BASE_METADATA)
    metadata["general.name"] = "qwen2"
    profile = build_model_profile(_gguf(metadata, "qwen2.5-1.5b.gguf"), file_size_bytes=1)
    assert profile.name == "qwen2.5-1.5b"


def test_quantization_falls_back_to_tensors() -> None:
    metadata = {key: value for key, value in BASE_METADATA.items() if key != "general.file_type"}
    profile = build_model_profile(_gguf(metadata), file_size_bytes=1)
    assert profile.quantization is QuantizationType.Q4_K_S


def test_profile_json_roundtrip() -> None:
    metadata = dict(BASE_METADATA)
    metadata["general.name"] = "roundtrip"
    profile = build_model_profile(_gguf(metadata), file_size_bytes=999)
    restored = ModelProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile


def test_footprint_method_on_real_shaped_profile() -> None:
    metadata = dict(BASE_METADATA)
    profile = build_model_profile(_gguf(metadata), file_size_bytes=1_000_000)
    footprint = profile.footprint(2048)
    assert footprint.kv_cache_bytes == 28672 * 2048
    assert footprint.total_estimate_bytes == 1_000_000 + 28672 * 2048
    restored = ModelFootprint.model_validate_json(footprint.model_dump_json())
    assert restored == footprint
