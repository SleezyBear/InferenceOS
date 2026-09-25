"""Independent KV-cache validation: correct, inconsistent, and unavailable metadata."""

from __future__ import annotations

import pytest

from inferenceos.models.kv_validation import (
    KV_CACHE_RELATIVE_TOLERANCE,
    derive_expected_kv_cache_bytes,
    validate_reported_kv_cache,
)


def _expected(n_kv_heads: int, *, context: int = 4096) -> int:
    return derive_expected_kv_cache_bytes(
        n_layers=28,
        n_kv_heads=n_kv_heads,
        key_length=128,
        value_length=128,
        context_length=context,
        element_bytes=2,
    )


def test_derivation_is_exact_for_gqa() -> None:
    # 28 layers * 2 kv heads * (128+128) * 2 bytes * 4096 context
    assert _expected(2) == 28 * 2 * 256 * 2 * 4096


def test_correct_metadata_is_within_tolerance() -> None:
    expected = _expected(2)
    reported = int(expected * 1.02)  # small allocator overhead
    result = validate_reported_kv_cache(expected_bytes=expected, reported_bytes=reported)
    assert result.within_tolerance is True
    assert result.relative_error is not None
    assert result.relative_error <= KV_CACHE_RELATIVE_TOLERANCE


def test_inconsistent_metadata_fails_validation() -> None:
    # Metadata wrongly claims MHA (kv heads == attention heads == 12) for a GQA model.
    wrong_expected = _expected(12)
    reported = int(_expected(2) * 1.01)  # backend reports the true GQA allocation
    result = validate_reported_kv_cache(expected_bytes=wrong_expected, reported_bytes=reported)
    assert result.within_tolerance is False
    assert result.relative_error is not None
    assert result.relative_error > KV_CACHE_RELATIVE_TOLERANCE


def test_unavailable_reported_value_is_not_a_silent_pass() -> None:
    result = validate_reported_kv_cache(expected_bytes=_expected(2), reported_bytes=None)
    assert result.within_tolerance is None
    assert result.reported_bytes is None


def test_derivation_takes_explicit_fields_and_rejects_indeterminate_input() -> None:
    # The derivation accepts explicit architecture fields only (no llama.cpp value and
    # no precomputed per-token field), and rejects an indeterminate input.
    with pytest.raises(ValueError):
        derive_expected_kv_cache_bytes(
            n_layers=28,
            n_kv_heads=0,
            key_length=128,
            value_length=128,
            context_length=4096,
        )
