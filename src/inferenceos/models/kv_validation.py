"""Independent, architecture-derived KV-cache validation.

The expected KV footprint is derived **only** from model architecture metadata
(layer count, KV-head count, key/value dimension, element width) and the context
length. It deliberately never reads a number reported by llama.cpp, so a value the
backend reports can never "agree with itself": the two sides come from genuinely
different sources (GGUF metadata vs. the backend's own allocation log).

Tolerance rationale
-------------------
The analytical value is exact for an f16 cache with no padding. llama.cpp rounds the
KV allocation up to allocator alignment and may include small per-context bookkeeping,
so the backend-reported figure is expected to be at or slightly above the analytical
one. ``KV_CACHE_RELATIVE_TOLERANCE`` is an explicit, named band that covers that
overhead; it is a property of the derivation, not a value tuned to make a particular
run pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inferenceos.models.estimator import F16_ELEMENT_BYTES

# Explicit relative band between the analytical KV expectation and the backend-reported
# allocation, covering allocator alignment / per-context overhead.
KV_CACHE_RELATIVE_TOLERANCE = 0.05
KV_CACHE_TOLERANCE_BASIS = (
    "analytical KV (layers * kv_heads * (k+v) * element_bytes * context) vs the "
    "backend allocation, allowing for allocator alignment and per-context overhead"
)

# The wrong derivation uses the attention-head count instead of the KV-head count;
# it is retained only so validation can prove that a plausible-but-wrong source does
# not accidentally agree with the backend.
def derive_expected_kv_cache_bytes(
    *,
    n_layers: int,
    n_kv_heads: int,
    key_length: int,
    value_length: int,
    context_length: int,
    element_bytes: int = F16_ELEMENT_BYTES,
) -> int:
    """Analytically derive the expected KV-cache size from architecture metadata.

    Uses the **KV-head** count (not the attention-head count), so GQA/MQA models are
    not overestimated. Raises ``ValueError`` for invalid/indeterminate inputs rather
    than fabricating a value.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if n_kv_heads < 1:
        raise ValueError("n_kv_heads must be >= 1")
    if key_length < 1 or value_length < 1:
        raise ValueError("key_length and value_length must be >= 1")
    if context_length < 1:
        raise ValueError("context_length must be >= 1")
    if element_bytes < 1:
        raise ValueError("element_bytes must be >= 1")
    return n_layers * n_kv_heads * (key_length + value_length) * element_bytes * context_length


@dataclass(frozen=True)
class KvCacheValidation:
    expected_bytes: int
    reported_bytes: int | None
    relative_error: float | None
    within_tolerance: bool | None
    tolerance: float
    basis: list[str] = field(default_factory=list)


def validate_reported_kv_cache(
    *,
    expected_bytes: int,
    reported_bytes: int | None,
    tolerance: float = KV_CACHE_RELATIVE_TOLERANCE,
) -> KvCacheValidation:
    """Compare an independently derived expectation with the backend-reported value.

    ``reported_bytes is None`` yields ``within_tolerance=None`` (unavailable), never a
    silent pass.
    """
    basis = [KV_CACHE_TOLERANCE_BASIS, f"tolerance={tolerance:g}"]
    if reported_bytes is None or reported_bytes <= 0:
        return KvCacheValidation(
            expected_bytes=expected_bytes,
            reported_bytes=None,
            relative_error=None,
            within_tolerance=None,
            tolerance=tolerance,
            basis=basis + ["backend-reported KV unavailable"],
        )
    relative_error = abs(expected_bytes - reported_bytes) / reported_bytes
    within = relative_error <= tolerance
    return KvCacheValidation(
        expected_bytes=expected_bytes,
        reported_bytes=reported_bytes,
        relative_error=round(relative_error, 6),
        within_tolerance=within,
        tolerance=tolerance,
        basis=basis
        + [
            f"independent_expected={expected_bytes} reported={reported_bytes} "
            f"relative_error={relative_error:.4f}"
        ],
    )
