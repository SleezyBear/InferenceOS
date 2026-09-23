"""KV-cache / footprint math, especially the GQA/MQA vs MHA distinction."""

from __future__ import annotations

import pytest

from inferenceos.contracts.enums import AttentionType, QuantizationType
from inferenceos.contracts.model import ModelProfile
from inferenceos.models.estimator import (
    estimate_kv_cache_bytes,
    estimate_kv_curve,
    kv_cache_bytes_per_token,
)


def _profile(
    *,
    n_layers: int,
    n_attention_heads: int,
    n_kv_heads: int,
    head_dim: int,
    weight_bytes: int = 1_000_000,
) -> ModelProfile:
    per_token = kv_cache_bytes_per_token(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        key_length=head_dim,
        value_length=head_dim,
    )
    return ModelProfile(
        path="synthetic.gguf",
        name="synthetic",
        format="gguf",
        architecture="synthetic",
        quantization=QuantizationType.Q4_K_M,
        file_size_bytes=weight_bytes,
        n_layers=n_layers,
        n_attention_heads=n_attention_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        key_length=head_dim,
        value_length=head_dim,
        weight_bytes=weight_bytes,
        kv_cache_bytes_per_token=per_token,
    )


def test_kv_per_token_formula() -> None:
    # 2 (K and V) * layers * kv_heads * head_dim * element_bytes
    assert (
        kv_cache_bytes_per_token(
            n_layers=28, n_kv_heads=2, key_length=128, value_length=128, element_bytes=2
        )
        == 28672
    )


def test_gqa_uses_kv_head_count_not_attention_head_count() -> None:
    heads = 12
    gqa = _profile(n_layers=28, n_attention_heads=heads, n_kv_heads=2, head_dim=128)
    mha = _profile(n_layers=28, n_attention_heads=heads, n_kv_heads=heads, head_dim=128)

    assert gqa.attention_type is AttentionType.GQA
    assert mha.attention_type is AttentionType.MHA
    # Same attention-head count, different KV-head count -> different cache.
    assert gqa.kv_cache_bytes_per_token * (heads // 2) == mha.kv_cache_bytes_per_token
    assert gqa.estimate_kv_cache_bytes(4096) < mha.estimate_kv_cache_bytes(4096)


def test_mqa_is_handled() -> None:
    mqa = _profile(n_layers=32, n_attention_heads=32, n_kv_heads=1, head_dim=128)
    assert mqa.attention_type is AttentionType.MQA
    assert mqa.kv_cache_bytes_per_token == 32 * 1 * (128 + 128) * 2


def test_estimate_scales_linearly_with_context() -> None:
    profile = _profile(n_layers=12, n_attention_heads=8, n_kv_heads=8, head_dim=64)
    one = profile.estimate_kv_cache_bytes(1024)
    assert profile.estimate_kv_cache_bytes(2048) == 2 * one


def test_estimate_scales_with_element_bytes() -> None:
    profile = _profile(n_layers=12, n_attention_heads=8, n_kv_heads=8, head_dim=64)
    f16 = profile.estimate_kv_cache_bytes(1024, kv_dtype_bytes=2)
    f32 = profile.estimate_kv_cache_bytes(1024, kv_dtype_bytes=4)
    assert f32 == 2 * f16


def test_estimate_kv_curve_sorts_and_dedups() -> None:
    curve = estimate_kv_curve(100, [4096, 1024, 2048, 1024])
    assert [entry.context_length for entry in curve] == [1024, 2048, 4096]
    assert curve[0].kv_cache_bytes == 100 * 1024


def test_footprint_separates_weight_and_kv() -> None:
    profile = _profile(n_layers=28, n_attention_heads=12, n_kv_heads=2, head_dim=128, weight_bytes=500)
    footprint = profile.footprint(2048)
    assert footprint.file_size_bytes == 500
    assert footprint.weight_bytes == 500
    assert footprint.kv_cache_bytes == profile.estimate_kv_cache_bytes(2048)
    assert footprint.total_estimate_bytes == 500 + footprint.kv_cache_bytes


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        kv_cache_bytes_per_token(n_layers=0, n_kv_heads=2, key_length=128, value_length=128)
    with pytest.raises(ValueError):
        kv_cache_bytes_per_token(n_layers=28, n_kv_heads=0, key_length=128, value_length=128)
    with pytest.raises(ValueError):
        estimate_kv_cache_bytes(100, 0)
    with pytest.raises(ValueError):
        _profile(n_layers=1, n_attention_heads=1, n_kv_heads=1, head_dim=1).estimate_kv_cache_bytes(
            0
        )
