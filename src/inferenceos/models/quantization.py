"""GGUF quantization detection.

Two sources are used: the explicit ``general.file_type`` metadata (authoritative
when present) and, as a fallback, the composition of tensor types in the file.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from inferenceos.contracts.enums import QuantizationType
from inferenceos.models.gguf import GgufTensorInfo

# llama.cpp `llama_ftype` (general.file_type) -> QuantizationType.
_FILE_TYPE_TO_QUANT: dict[int, QuantizationType] = {
    0: QuantizationType.F32,
    1: QuantizationType.F16,
    2: QuantizationType.Q4_0,
    3: QuantizationType.Q4_1,
    7: QuantizationType.Q8_0,
    8: QuantizationType.Q5_0,
    9: QuantizationType.Q5_1,
    10: QuantizationType.Q2_K,
    11: QuantizationType.Q3_K_S,
    12: QuantizationType.Q3_K_M,
    13: QuantizationType.Q3_K_L,
    14: QuantizationType.Q4_K_S,
    15: QuantizationType.Q4_K_M,
    16: QuantizationType.Q5_K_S,
    17: QuantizationType.Q5_K_M,
    18: QuantizationType.Q6_K,
    23: QuantizationType.IQ4_NL,
    25: QuantizationType.IQ3_M,
    27: QuantizationType.IQ2_M,
    28: QuantizationType.IQ4_XS,
    30: QuantizationType.BF16,
}

# ggml tensor type ids (distinct from file_type) -> QuantizationType.
# K-quants are generic (no S/M variant) at the tensor level, so the nearest variant
# is used; ``general.file_type`` is the authoritative source when present.
_GGML_TYPE_TO_QUANT: dict[int, QuantizationType] = {
    0: QuantizationType.F32,
    1: QuantizationType.F16,
    2: QuantizationType.Q4_0,
    3: QuantizationType.Q4_1,
    6: QuantizationType.Q5_0,
    7: QuantizationType.Q5_1,
    8: QuantizationType.Q8_0,
    10: QuantizationType.Q2_K,
    11: QuantizationType.Q3_K_S,
    12: QuantizationType.Q4_K_S,
    13: QuantizationType.Q5_K_S,
    14: QuantizationType.Q6_K,
    15: QuantizationType.Q8_0,
    20: QuantizationType.IQ4_NL,
    23: QuantizationType.IQ4_XS,
    30: QuantizationType.BF16,
}


def quantization_from_file_type(file_type: int | float | None) -> QuantizationType:
    if file_type is None:
        return QuantizationType.UNKNOWN
    try:
        code = int(file_type)
    except (TypeError, ValueError):
        return QuantizationType.UNKNOWN
    return _FILE_TYPE_TO_QUANT.get(code, QuantizationType.UNKNOWN)


def quantization_from_tensors(tensors: Sequence[GgufTensorInfo]) -> QuantizationType:
    """Infer the dominant quantization from tensor types (fallback only)."""
    counts: Counter[int] = Counter()
    for tensor in tensors:
        # Skip 1-D norm/bias tensors (kept at F32) and prefer weight matrices.
        if len(tensor.dimensions) < 2:
            continue
        counts[tensor.ggml_type] += 1
    for ggml_type, _count in counts.most_common():
        quant = _GGML_TYPE_TO_QUANT.get(ggml_type)
        if quant is not None and quant not in (QuantizationType.F32, QuantizationType.F16):
            return quant
    if counts:
        return _GGML_TYPE_TO_QUANT.get(counts.most_common(1)[0][0], QuantizationType.UNKNOWN)
    return QuantizationType.UNKNOWN


def detect_quantization(
    file_type: int | float | None,
    tensors: Sequence[GgufTensorInfo],
) -> QuantizationType:
    quant = quantization_from_file_type(file_type)
    if quant is not QuantizationType.UNKNOWN:
        return quant
    return quantization_from_tensors(tensors)


def tensor_type_histogram(tensors: Sequence[GgufTensorInfo]) -> Mapping[int, int]:
    counts: Counter[int] = Counter(tensor.ggml_type for tensor in tensors)
    return dict(sorted(counts.items()))
