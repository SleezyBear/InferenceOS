"""Normalize raw GGUF metadata into a backend-independent architecture view.

Key lookup is suffix-based (``<anything>.attention.head_count``) so no specific
architecture name (Qwen, Llama, Phi, Gemma, …) is hardcoded. Where a field is not
present the value is left ``None`` rather than guessed; the loader decides which
missing fields make a reliable estimate impossible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferenceos.contracts.enums import QuantizationType
from inferenceos.models.gguf import GgufArray, GgufFile
from inferenceos.models.quantization import detect_quantization

# Suffix -> canonical field. The prefix before the suffix is the architecture id.
_BLOCK_COUNT = "block_count"
_CONTEXT_LENGTH = "context_length"
_EMBEDDING_LENGTH = "embedding_length"
_HEAD_COUNT = "attention.head_count"
_HEAD_COUNT_KV = "attention.head_count_kv"
_KEY_LENGTH = "attention.key_length"
_VALUE_LENGTH = "attention.value_length"
_ROPE_DIM = "rope.dimension_count"
_VOCAB_SIZE = "vocab_size"


@dataclass(frozen=True)
class ArchitectureInfo:
    architecture: str | None
    model_name: str | None
    context_length: int | None
    n_layers: int | None
    n_attention_heads: int | None
    n_kv_heads: int | None
    key_length: int | None
    value_length: int | None
    head_dim: int | None
    embedding_dim: int | None
    vocabulary_size: int | None
    parameter_count: int | None
    tensor_count: int
    quantization: QuantizationType
    file_type: int | None
    metadata_scalars: dict[str, Any] = field(default_factory=dict)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _as_str(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return None


def _as_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return None


def _find(gguf: GgufFile, suffix: str, architecture: str | None) -> object | None:
    if architecture is not None:
        prefixed = f"{architecture}.{suffix}"
        if prefixed in gguf.metadata:
            return gguf.metadata[prefixed]
    exact = gguf.metadata.get(suffix)
    if exact is not None:
        return exact
    dotted = f".{suffix}"
    for key, value in gguf.metadata.items():
        if key.endswith(dotted):
            return value
    return None


def _find_int(gguf: GgufFile, suffix: str, architecture: str | None) -> int | None:
    return _as_int(_find(gguf, suffix, architecture))


def _array_count(value: object | None) -> int | None:
    if isinstance(value, GgufArray):
        return value.count
    return None


def normalize_architecture(gguf: GgufFile) -> ArchitectureInfo:
    architecture = _as_str(gguf.get("general.architecture"))
    model_name = _as_str(gguf.get("general.name"))

    n_layers = _find_int(gguf, _BLOCK_COUNT, architecture)
    embedding_dim = _find_int(gguf, _EMBEDDING_LENGTH, architecture)
    n_attention_heads = _find_int(gguf, _HEAD_COUNT, architecture)
    n_kv_heads = _find_int(gguf, _HEAD_COUNT_KV, architecture)
    # Absence of an explicit KV-head count means the architecture is standard MHA
    # (llama.cpp's own default). Never assume head_count for a *present* value.
    if n_kv_heads is None and n_attention_heads is not None:
        n_kv_heads = n_attention_heads

    context_length = _find_int(gguf, _CONTEXT_LENGTH, architecture)
    if context_length is None:
        context_length = _as_int(gguf.get("general.context_length"))

    key_length = _find_int(gguf, _KEY_LENGTH, architecture)
    value_length = _find_int(gguf, _VALUE_LENGTH, architecture)

    head_dim = key_length
    if head_dim is None:
        if embedding_dim is not None and n_attention_heads:
            head_dim = embedding_dim // n_attention_heads
        else:
            head_dim = _find_int(gguf, _ROPE_DIM, architecture)

    file_type_raw = gguf.get("general.file_type")
    file_type = _as_int(file_type_raw)
    vocab_count = _array_count(gguf.get("tokenizer.ggml.tokens"))
    if vocab_count is None:
        vocab_count = _find_int(gguf, _VOCAB_SIZE, architecture)

    parameter_count = sum(tensor.element_count for tensor in gguf.tensors) or None

    scalar_metadata: dict[str, Any] = {}
    for key, value in gguf.metadata.items():
        if isinstance(value, (int, float, bool, str)):
            scalar_metadata[key] = value

    return ArchitectureInfo(
        architecture=architecture,
        model_name=model_name,
        context_length=context_length,
        n_layers=n_layers,
        n_attention_heads=n_attention_heads,
        n_kv_heads=n_kv_heads,
        key_length=key_length,
        value_length=value_length,
        head_dim=head_dim,
        embedding_dim=embedding_dim,
        vocabulary_size=vocab_count,
        parameter_count=parameter_count,
        tensor_count=gguf.tensor_count,
        quantization=detect_quantization(_as_number(file_type_raw), gguf.tensors),
        file_type=file_type,
        metadata_scalars=scalar_metadata,
    )
