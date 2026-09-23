"""Synthetic GGUF parsing: metadata, arrays, tensors, and malformed input."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from inferenceos.models.gguf import (
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_BOOL,
    GGUF_TYPE_FLOAT32,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT32,
    GgufArray,
    GgufError,
    read_gguf,
)


def _gstr(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_string(key: str, value: str) -> bytes:
    return _gstr(key) + struct.pack("<I", GGUF_TYPE_STRING) + _gstr(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _gstr(key) + struct.pack("<I", GGUF_TYPE_UINT32) + struct.pack("<I", value)


def _kv_f32(key: str, value: float) -> bytes:
    return _gstr(key) + struct.pack("<I", GGUF_TYPE_FLOAT32) + struct.pack("<f", value)


def _kv_bool(key: str, value: bool) -> bytes:
    return _gstr(key) + struct.pack("<I", GGUF_TYPE_BOOL) + struct.pack("<B", 1 if value else 0)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    return (
        _gstr(key)
        + struct.pack("<I", GGUF_TYPE_ARRAY)
        + struct.pack("<I", GGUF_TYPE_STRING)
        + struct.pack("<Q", len(values))
        + b"".join(_gstr(value) for value in values)
    )


def _tensor(name: str, dims: tuple[int, ...], ggml_type: int, offset: int) -> bytes:
    return (
        _gstr(name)
        + struct.pack("<I", len(dims))
        + b"".join(struct.pack("<Q", dim) for dim in dims)
        + struct.pack("<I", ggml_type)
        + struct.pack("<Q", offset)
    )


def _write_gguf(path: Path, version: int, kv: list[bytes], tensors: list[bytes]) -> None:
    header = (
        b"GGUF"
        + struct.pack("<I", version)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", len(kv))
    )
    path.write_bytes(header + b"".join(kv) + b"".join(tensors))


def test_reads_metadata_arrays_and_tensors(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    _write_gguf(
        path,
        3,
        [
            _kv_string("general.architecture", "qwen2"),
            _kv_u32("qwen2.block_count", 28),
            _kv_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6),
            _kv_bool("tokenizer.ggml.add_bos_token", False),
            _kv_string_array("tokenizer.ggml.tokens", ["a", "bb", "ccc"]),
            _kv_u32("tail.value", 7),
        ],
        [
            _tensor("token_embd.weight", (3, 4), 8, 0),
            _tensor("blk.0.attn_qkv.weight", (4, 4), 12, 64),
        ],
    )

    gguf = read_gguf(path)
    assert gguf.version == 3
    assert gguf.tensor_count == 2
    assert gguf.get("general.architecture") == "qwen2"
    assert gguf.get("qwen2.block_count") == 28
    assert abs(float(gguf.scalar("qwen2.attention.layer_norm_rms_epsilon") or 0.0) - 1e-6) < 1e-12
    assert gguf.get("tokenizer.ggml.add_bos_token") is False

    tokens = gguf.get("tokenizer.ggml.tokens")
    assert isinstance(tokens, GgufArray)
    assert tokens.count == 3
    assert tokens.element_type_name == "str"
    # Metadata after the skipped array must still parse correctly.
    assert gguf.get("tail.value") == 7

    first = gguf.tensors[0]
    assert first.name == "token_embd.weight"
    assert first.dimensions == (3, 4)
    assert first.element_count == 12
    assert first.ggml_type == 8
    assert gguf.tensors[1].offset == 64


def test_reads_version_two(tmp_path: Path) -> None:
    path = tmp_path / "m2.gguf"
    _write_gguf(path, 2, [_kv_string("general.architecture", "llama")], [])
    gguf = read_gguf(path)
    assert gguf.version == 2
    assert gguf.get("general.architecture") == "llama"


def test_rejects_bad_magic(tmp_path: Path) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"XXXX" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0))
    with pytest.raises(GgufError):
        read_gguf(path)


def test_rejects_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "trunc.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3))
    with pytest.raises(GgufError):
        read_gguf(path)


def test_rejects_unsupported_version(tmp_path: Path) -> None:
    path = tmp_path / "v9.gguf"
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 9) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    )
    with pytest.raises(GgufError):
        read_gguf(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        read_gguf(tmp_path / "does-not-exist.gguf")
