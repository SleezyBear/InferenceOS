"""Per-layer kinds read from GGUF tensor descriptors."""

from __future__ import annotations

import struct
from pathlib import Path

from inferenceos.tune.layer_search import (
    full_layer_kinds_from_gguf,
    layer_count_from_gguf,
    layer_kinds_from_gguf,
)


def _gstr(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _tensor(name: str, dims: tuple[int, ...], ggml_type: int = 0, offset: int = 0) -> bytes:
    return (
        _gstr(name)
        + struct.pack("<I", len(dims))
        + b"".join(struct.pack("<Q", dim) for dim in dims)
        + struct.pack("<I", ggml_type)
        + struct.pack("<Q", offset)
    )


def _write_gguf(path: Path, tensors: list[bytes]) -> None:
    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", 0)
    )
    path.write_bytes(header + b"".join(tensors))


def test_mixed_kinds_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "mixed.gguf"
    _write_gguf(
        path,
        [
            _tensor("token_embd.weight", (4, 4)),
            _tensor("blk.0.attn_q.weight", (4, 4)),
            _tensor("blk.0.attn_q.bias", (4,)),
            _tensor("blk.0.attn_k.weight", (4, 4)),
            _tensor("blk.1.ffn_gate.weight", (4, 4)),
            _tensor("blk.1.ffn_up.weight", (4, 4)),
        ],
    )
    assert layer_count_from_gguf(path) == 2
    kinds = layer_kinds_from_gguf(path)
    assert kinds == ["attn_k|attn_q", "ffn_gate|ffn_up"]
    assert full_layer_kinds_from_gguf(path) == ["attn_k|attn_q", "ffn_gate|ffn_up"]


def test_same_kind_collapses_to_none(tmp_path: Path) -> None:
    path = tmp_path / "same.gguf"
    _write_gguf(
        path,
        [
            _tensor("blk.0.attn_q.weight", (4, 4)),
            _tensor("blk.1.attn_q.weight", (4, 4)),
        ],
    )
    assert layer_kinds_from_gguf(path) is None
    assert full_layer_kinds_from_gguf(path) == ["attn_q", "attn_q"]


def test_no_blocks_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.gguf"
    _write_gguf(path, [_tensor("token_embd.weight", (4, 4))])
    assert layer_count_from_gguf(path) is None
    assert layer_kinds_from_gguf(path) is None
