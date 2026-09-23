"""GGUF architecture normalization (architecture-agnostic, GQA/MHA/MQA)."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.contracts.enums import AttentionType, QuantizationType
from inferenceos.models.architecture import normalize_architecture
from inferenceos.models.gguf import GgufArray, GgufFile, GgufTensorInfo
from inferenceos.models.loader import ModelMetadataError, build_model_profile
from inferenceos.models.quantization import (
    quantization_from_file_type,
    quantization_from_tensors,
)

GQA_METADATA: dict[str, object] = {
    "general.architecture": "qwen2",
    "qwen2.block_count": 28,
    "qwen2.context_length": 32768,
    "qwen2.embedding_length": 1536,
    "qwen2.attention.head_count": 12,
    "qwen2.attention.head_count_kv": 2,
    "general.file_type": 15,
    "tokenizer.ggml.tokens": GgufArray(element_type=8, count=151936),
}

MHA_METADATA: dict[str, object] = {
    "general.architecture": "llama",
    "llama.block_count": 6,
    "llama.context_length": 128,
    "llama.embedding_length": 288,
    "llama.attention.head_count": 6,
    "general.file_type": 2,
    "tokenizer.ggml.tokens": GgufArray(element_type=8, count=32000),
}


def _gguf(metadata: dict[str, object], tensors: tuple[GgufTensorInfo, ...] = ()) -> GgufFile:
    return GgufFile(path=Path("synthetic.gguf"), version=3, metadata=metadata, tensors=tensors)


def test_normalize_gqa_architecture() -> None:
    info = normalize_architecture(_gguf(GQA_METADATA))
    assert info.architecture == "qwen2"
    assert info.n_layers == 28
    assert info.n_attention_heads == 12
    assert info.n_kv_heads == 2
    assert info.head_dim == 128
    assert info.embedding_dim == 1536
    assert info.context_length == 32768
    assert info.vocabulary_size == 151936
    assert info.quantization is QuantizationType.Q4_K_M
    assert info.file_type == 15


def test_normalize_mha_defaults_kv_to_heads() -> None:
    info = normalize_architecture(_gguf(MHA_METADATA))
    assert info.n_attention_heads == 6
    # No head_count_kv key -> standard MHA, KV heads == attention heads.
    assert info.n_kv_heads == 6
    assert info.head_dim == 48
    assert info.quantization is QuantizationType.Q4_0


def test_normalize_mqa_architecture() -> None:
    metadata = dict(GQA_METADATA)
    metadata["qwen2.attention.head_count_kv"] = 1
    info = normalize_architecture(_gguf(metadata))
    assert info.n_kv_heads == 1
    profile = build_model_profile(_gguf(metadata), file_size_bytes=1000)
    assert profile.attention_type is AttentionType.MQA


def test_normalize_respects_explicit_key_value_length() -> None:
    metadata = dict(GQA_METADATA)
    metadata["qwen2.attention.key_length"] = 64
    metadata["qwen2.attention.value_length"] = 96
    info = normalize_architecture(_gguf(metadata))
    assert info.key_length == 64
    assert info.value_length == 96
    assert info.head_dim == 64


def test_normalize_without_architecture_prefix_still_finds_keys() -> None:
    metadata: dict[str, object] = {
        "mystery.block_count": 4,
        "mystery.context_length": 512,
        "mystery.embedding_length": 256,
        "mystery.attention.head_count": 4,
        "mystery.attention.head_count_kv": 2,
    }
    info = normalize_architecture(_gguf(metadata))
    assert info.architecture is None
    assert info.n_layers == 4
    assert info.n_kv_heads == 2
    assert info.head_dim == 64


def test_quantization_from_file_type() -> None:
    assert quantization_from_file_type(15) is QuantizationType.Q4_K_M
    assert quantization_from_file_type(2) is QuantizationType.Q4_0
    assert quantization_from_file_type(None) is QuantizationType.UNKNOWN
    assert quantization_from_file_type(9999) is QuantizationType.UNKNOWN


def test_quantization_fallback_from_tensors() -> None:
    tensors = (
        GgufTensorInfo(name="blk.0.attn.weight", dimensions=(64, 64), ggml_type=12, offset=0),
        GgufTensorInfo(name="blk.0.ffn.weight", dimensions=(64, 64), ggml_type=12, offset=64),
    )
    assert quantization_from_tensors(tensors) is QuantizationType.Q4_K_S


def test_build_profile_requires_core_metadata() -> None:
    metadata: dict[str, object] = {"general.architecture": "qwen2", "qwen2.block_count": 28}
    with pytest.raises(ModelMetadataError):
        build_model_profile(_gguf(metadata), file_size_bytes=1)
