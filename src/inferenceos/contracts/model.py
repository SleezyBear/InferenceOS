"""Model contracts.

``ModelProfile`` carries GGUF metadata plus analytical memory estimates. The
KV-cache estimate is deliberately explicit (per-token) because Phase 2 must verify
it against real measured RSS deltas, and it must be correct for GQA/MQA models
where the KV-head count, not the attention-head count, drives the calculation.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import AttentionType, QuantizationType


class KVEstimate(ContractBase):
    context_length: int = Field(ge=1)
    kv_cache_bytes: int = Field(ge=0)


class ModelProfile(ContractBase):
    path: str
    name: str
    architecture: str
    quantization: QuantizationType = QuantizationType.UNKNOWN
    file_size_bytes: int = Field(ge=0)
    parameter_count: int | None = None

    trained_context_length: int | None = None
    n_layers: int = Field(ge=1)
    n_attention_heads: int = Field(ge=1)
    n_kv_heads: int = Field(ge=1)
    head_dim: int = Field(ge=1)
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

    def estimate_kv_cache_bytes(self, context_length: int) -> int:
        """Analytical KV-cache size for a given context length."""
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        return self.kv_cache_bytes_per_token * context_length

    def kv_estimate(self, context_length: int) -> KVEstimate:
        return KVEstimate(
            context_length=context_length,
            kv_cache_bytes=self.estimate_kv_cache_bytes(context_length),
        )
