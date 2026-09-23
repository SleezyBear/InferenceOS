"""Deterministic KV-cache and footprint estimation.

The KV-cache formula (for a transformer where every layer stores K and V for each
sequence position) is::

    kv_cache_bytes = n_layers
                     * n_kv_heads
                     * (key_length + value_length)
                     * element_bytes
                     * context_length

``n_kv_heads`` is the KV-head count, **not** the attention-head count: for GQA/MQA
models (``n_kv_heads < n_attention_heads``) using the attention-head count would
overestimate the cache, sometimes by more than an order of magnitude. ``element_bytes``
is 2 for f16 (the llama.cpp default), 4 for f32, etc.

``weight_bytes`` is the estimated loaded weight footprint. For a quantized GGUF
this is taken from the real file size (weights are stored in their final quantized
form); it is an *estimate*, never a measured RSS. Real RSS is produced by the
backend/telemetry layer.
"""

from __future__ import annotations

from collections.abc import Iterable

from inferenceos.contracts.model import KVEstimate, ModelFootprint, ModelProfile

F16_ELEMENT_BYTES = 2
F32_ELEMENT_BYTES = 4


def kv_cache_bytes_per_token(
    *,
    n_layers: int,
    n_kv_heads: int,
    key_length: int,
    value_length: int,
    element_bytes: int = F16_ELEMENT_BYTES,
) -> int:
    """Analytical KV-cache bytes per sequence position.

    Raises ``ValueError`` if any input is missing/invalid, so callers get an
    explicit indeterminate result instead of a fabricated value.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if n_kv_heads < 1:
        raise ValueError("n_kv_heads must be >= 1")
    if key_length < 1 or value_length < 1:
        raise ValueError("key_length and value_length must be >= 1")
    if element_bytes < 1:
        raise ValueError("element_bytes must be >= 1")
    per_layer = n_kv_heads * (key_length + value_length) * element_bytes
    return n_layers * per_layer


def estimate_kv_cache_bytes(
    kv_cache_bytes_per_token: int,
    context_length: int,
    *,
    element_bytes: int = F16_ELEMENT_BYTES,
) -> int:
    """Scale a per-token KV size (defined for f16) to a context length."""
    if context_length < 1:
        raise ValueError("context_length must be >= 1")
    if element_bytes < 1:
        raise ValueError("element_bytes must be >= 1")
    return kv_cache_bytes_per_token * context_length * element_bytes // F16_ELEMENT_BYTES


def estimate_kv_curve(
    kv_cache_bytes_per_token: int,
    contexts: Iterable[int],
    *,
    element_bytes: int = F16_ELEMENT_BYTES,
) -> list[KVEstimate]:
    """Build ``KVEstimate`` entries for a set of context lengths (de-duplicated)."""
    estimates: list[KVEstimate] = []
    for context in sorted(set(contexts)):
        estimates.append(
            KVEstimate(
                context_length=context,
                kv_cache_bytes=estimate_kv_cache_bytes(
                    kv_cache_bytes_per_token, context, element_bytes=element_bytes
                ),
            )
        )
    return estimates


def estimate_footprint(
    profile: ModelProfile,
    context_length: int,
    *,
    element_bytes: int = F16_ELEMENT_BYTES,
    dtype_name: str = "f16",
) -> ModelFootprint:
    """Estimated loaded footprint (weights + KV) at a context length."""
    return profile.footprint(
        context_length, kv_dtype_bytes=element_bytes, kv_dtype=dtype_name
    )
