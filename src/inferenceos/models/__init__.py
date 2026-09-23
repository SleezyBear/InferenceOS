"""Model introspection: real GGUF parsing and architecture normalization.

Kept separate from contracts so the representation is not tied to a specific file
format or execution backend.
"""

from __future__ import annotations

from inferenceos.models.architecture import ArchitectureInfo, normalize_architecture
from inferenceos.models.estimator import (
    F16_ELEMENT_BYTES,
    F32_ELEMENT_BYTES,
    estimate_footprint,
    estimate_kv_cache_bytes,
    estimate_kv_curve,
    kv_cache_bytes_per_token,
)
from inferenceos.models.gguf import (
    GGUF_MAGIC,
    GgufArray,
    GgufError,
    GgufFile,
    GgufTensorInfo,
    read_gguf,
)
from inferenceos.models.loader import (
    ModelMetadataError,
    build_model_profile,
    load_model_profile,
)
from inferenceos.models.quantization import (
    detect_quantization,
    quantization_from_file_type,
    quantization_from_tensors,
)

__all__ = [
    "F16_ELEMENT_BYTES",
    "F32_ELEMENT_BYTES",
    "GGUF_MAGIC",
    "ArchitectureInfo",
    "GgufArray",
    "GgufError",
    "GgufFile",
    "GgufTensorInfo",
    "ModelMetadataError",
    "build_model_profile",
    "detect_quantization",
    "estimate_footprint",
    "estimate_kv_cache_bytes",
    "estimate_kv_curve",
    "kv_cache_bytes_per_token",
    "load_model_profile",
    "normalize_architecture",
    "quantization_from_file_type",
    "quantization_from_tensors",
    "read_gguf",
]
