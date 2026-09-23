"""Build an InferenceOS :class:`ModelProfile` from a real GGUF file.

Parsing (``gguf``/``architecture``) is kept separate from contract construction
here. If metadata required for a reliable KV estimate is missing, this raises
:class:`ModelMetadataError` instead of inventing values.
"""

from __future__ import annotations

from pathlib import Path

from inferenceos.contracts.enums import QuantizationType
from inferenceos.contracts.model import ModelProfile
from inferenceos.models.architecture import ArchitectureInfo, normalize_architecture
from inferenceos.models.estimator import F16_ELEMENT_BYTES, kv_cache_bytes_per_token
from inferenceos.models.gguf import GgufFile, read_gguf
from inferenceos.models.quantization import tensor_type_histogram


class ModelMetadataError(ValueError):
    """Raised when a model cannot be characterized from its metadata."""


_EXCLUDED_METADATA_KEYS = {"tokenizer.chat_template"}


def build_model_profile(
    gguf: GgufFile,
    *,
    file_size_bytes: int | None = None,
    weight_bytes: int | None = None,
) -> ModelProfile:
    info = normalize_architecture(gguf)
    missing = [
        label
        for label, value in (
            ("layers", info.n_layers),
            ("attention heads", info.n_attention_heads),
            ("KV heads", info.n_kv_heads),
            ("head dimension", info.head_dim),
        )
        if value is None
    ]
    if missing:
        raise ModelMetadataError(
            f"missing required model metadata for {gguf.path.name}: {', '.join(missing)}"
        )
    assert info.n_layers is not None  # narrowed by the check above
    assert info.n_attention_heads is not None
    assert info.n_kv_heads is not None
    assert info.head_dim is not None

    size = file_size_bytes
    if size is None:
        try:
            size = gguf.path.stat().st_size
        except OSError as exc:  # pragma: no cover - filesystem dependent
            raise ModelMetadataError(f"cannot stat model file {gguf.path}: {exc}") from exc

    key_length = info.key_length or info.head_dim
    value_length = info.value_length or info.head_dim
    per_token = kv_cache_bytes_per_token(
        n_layers=info.n_layers,
        n_kv_heads=info.n_kv_heads,
        key_length=key_length,
        value_length=value_length,
        element_bytes=F16_ELEMENT_BYTES,
    )

    metadata = dict(info.metadata_scalars)
    for excluded in _EXCLUDED_METADATA_KEYS:
        metadata.pop(excluded, None)
    metadata["_tensor_type_histogram"] = {
        str(key): value for key, value in tensor_type_histogram(gguf.tensors).items()
    }
    metadata["_source_format"] = "gguf"
    metadata["_gguf_version"] = gguf.version

    name = info.model_name
    if name is None or (info.architecture is not None and name.lower() == info.architecture.lower()):
        # ``general.name`` is sometimes a bare architecture id (e.g. "llama");
        # prefer the file stem when it carries no more information.
        name = gguf.path.stem

    return ModelProfile(
        path=str(gguf.path),
        name=name,
        format="gguf",
        architecture=info.architecture or "unknown",
        quantization=info.quantization,
        file_size_bytes=size,
        parameter_count=info.parameter_count,
        tensor_count=info.tensor_count,
        trained_context_length=info.context_length,
        n_layers=info.n_layers,
        n_attention_heads=info.n_attention_heads,
        n_kv_heads=info.n_kv_heads,
        head_dim=info.head_dim,
        key_length=key_length,
        value_length=value_length,
        embedding_dim=info.embedding_dim,
        vocabulary_size=info.vocabulary_size,
        weight_bytes=weight_bytes if weight_bytes is not None else size,
        kv_cache_bytes_per_token=per_token,
        kv_cache_estimates=[],
        metadata=metadata,
    )


def load_model_profile(path: Path | str) -> ModelProfile:
    """Read a real GGUF file and produce its normalized :class:`ModelProfile`."""
    model_path = Path(path)
    if not model_path.is_file():
        raise ModelMetadataError(f"model file not found: {model_path}")
    gguf = read_gguf(model_path)
    return build_model_profile(gguf)


def quantization_label(quantization: QuantizationType) -> str:
    return quantization.value
