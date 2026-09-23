"""Model contracts.

``ModelProfile`` captures *what the model is* (format, architecture, heads,
quantization) independently of any execution backend. It also carries explicit
memory estimates, kept deliberately separate from real measurements:

* ``file_size_bytes``      — bytes on disk (from the filesystem)
* ``weight_bytes``         — estimated loaded weight footprint
* ``kv_cache_bytes_*``     — estimated KV-cache size (architecture-derived)
* measured process RSS     — produced by the backend/telemetry layer, never here

The KV-cache estimate is per-token because Phase 2 must verify it against real
measured RSS deltas, and it must use the *KV-head* count (not the attention-head
count) so GQA/MQA architectures are handled correctly.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import AttentionType, QuantizationType


class KVEstimate(ContractBase):
    context_length: int = Field(ge=1)
    kv_cache_bytes: int = Field(ge=0)


class ModelFootprint(ContractBase):
    """Deterministic *estimated* memory footprint (never a measurement)."""

    context_length: int = Field(ge=1)
    file_size_bytes: int = Field(ge=0)
    weight_bytes: int = Field(ge=0)
    kv_cache_bytes: int = Field(ge=0)
    kv_cache_dtype: str = "f16"
    total_estimate_bytes: int = Field(ge=0)


class ModelProfile(ContractBase):
    path: str
    name: str
    format: str = "unknown"
    architecture: str
    quantization: QuantizationType = QuantizationType.UNKNOWN
    file_size_bytes: int = Field(ge=0)
    parameter_count: int | None = None
    tensor_count: int | None = None

    trained_context_length: int | None = None
    n_layers: int = Field(ge=1)
    n_attention_heads: int = Field(ge=1)
    n_kv_heads: int = Field(ge=1)
    head_dim: int = Field(ge=1)
    key_length: int | None = None
    value_length: int | None = None
    embedding_dim: int | None = None
    vocabulary_size: int | None = None

    attention_type: AttentionType = AttentionType.UNKNOWN

    weight_bytes: int = Field(ge=0)
    kv_cache_bytes_per_token: int = Field(ge=0)
    kv_cache_estimates: list[KVEstimate] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_attention_type(self) -> "ModelProfile":
        # Derive from the real head counts; never trust an asserted value.
        if self.n_kv_heads == 1:
            derived = AttentionType.MQA
        elif self.n_kv_heads < self.n_attention_heads:
            derived = AttentionType.GQA
        else:
            derived = AttentionType.MHA
        object.__setattr__(self, "attention_type", derived)
        return self

    def estimate_kv_cache_bytes(self, context_length: int, *, kv_dtype_bytes: int = 2) -> int:
        """Analytical KV-cache size for a context length.

        ``kv_cache_bytes_per_token`` is defined for a 2-byte (f16) element; other
        element widths are scaled linearly. Quantized KV-cache types have
        block-scale overhead not modelled here and must be treated as estimates.
        """
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if kv_dtype_bytes < 1:
            raise ValueError("kv_dtype_bytes must be >= 1")
        return self.kv_cache_bytes_per_token * context_length * kv_dtype_bytes // 2

    def kv_estimate(self, context_length: int) -> KVEstimate:
        return KVEstimate(
            context_length=context_length,
            kv_cache_bytes=self.estimate_kv_cache_bytes(context_length),
        )

    def footprint(
        self,
        context_length: int,
        *,
        kv_dtype_bytes: int = 2,
        kv_dtype: str = "f16",
    ) -> ModelFootprint:
        """Estimated loaded footprint at a given context length (not measured)."""
        kv_bytes = self.estimate_kv_cache_bytes(context_length, kv_dtype_bytes=kv_dtype_bytes)
        return ModelFootprint(
            context_length=context_length,
            file_size_bytes=self.file_size_bytes,
            weight_bytes=self.weight_bytes,
            kv_cache_bytes=kv_bytes,
            kv_cache_dtype=kv_dtype,
            total_estimate_bytes=self.weight_bytes + kv_bytes,
        )
