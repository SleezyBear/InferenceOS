"""Minimal, dependency-free GGUF metadata reader.

Reads only the GGUF header, metadata key/value pairs and tensor descriptors —
never tensor data — so it is fast and memory-light even for multi-GB models. The
result is a raw representation; normalization into InferenceOS contracts happens in
:mod:`inferenceos.models.architecture` / :mod:`inferenceos.models.loader`.

Array payloads are skipped (their element type and count are retained) because the
only arrays needed for characterization are tokenizer vocabularies, where the count
is sufficient and materializing hundreds of thousands of strings is wasteful.

Reference: the GGUF specification and llama.cpp ``gguf_init_from_file``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = 0x46554747  # b"GGUF" little-endian

# Metadata value type ids (gguf_metadata_value_type)
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGUF_VALUE_TYPE_NAMES: dict[int, str] = {
    GGUF_TYPE_UINT8: "u8",
    GGUF_TYPE_INT8: "i8",
    GGUF_TYPE_UINT16: "u16",
    GGUF_TYPE_INT16: "i16",
    GGUF_TYPE_UINT32: "u32",
    GGUF_TYPE_INT32: "i32",
    GGUF_TYPE_FLOAT32: "f32",
    GGUF_TYPE_BOOL: "bool",
    GGUF_TYPE_STRING: "str",
    GGUF_TYPE_ARRAY: "arr",
    GGUF_TYPE_UINT64: "u64",
    GGUF_TYPE_INT64: "i64",
    GGUF_TYPE_FLOAT64: "f64",
}

_U8 = struct.Struct("<B")
_I8 = struct.Struct("<b")
_U16 = struct.Struct("<H")
_I16 = struct.Struct("<h")
_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_F32 = struct.Struct("<f")
_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")
_F64 = struct.Struct("<d")
_BOOL = struct.Struct("<?")

_SCALAR_STRUCTS: dict[int, struct.Struct] = {
    GGUF_TYPE_UINT8: _U8,
    GGUF_TYPE_INT8: _I8,
    GGUF_TYPE_UINT16: _U16,
    GGUF_TYPE_INT16: _I16,
    GGUF_TYPE_UINT32: _U32,
    GGUF_TYPE_INT32: _I32,
    GGUF_TYPE_FLOAT32: _F32,
    GGUF_TYPE_BOOL: _BOOL,
    GGUF_TYPE_UINT64: _U64,
    GGUF_TYPE_INT64: _I64,
    GGUF_TYPE_FLOAT64: _F64,
}

MAX_DIMS = 4


class GgufError(ValueError):
    """Raised when a file is not valid GGUF or is truncated/malformed."""


@dataclass(frozen=True)
class GgufArray:
    """A metadata array whose payload was skipped, retaining only its shape."""

    element_type: int
    count: int

    @property
    def element_type_name(self) -> str:
        return GGUF_VALUE_TYPE_NAMES.get(self.element_type, f"type:{self.element_type}")


@dataclass(frozen=True)
class GgufTensorInfo:
    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def element_count(self) -> int:
        total = 1
        for dim in self.dimensions:
            total *= dim
        return total


@dataclass(frozen=True)
class GgufFile:
    path: Path
    version: int
    metadata: dict[str, object]
    tensors: tuple[GgufTensorInfo, ...]

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    def get(self, key: str, default: object = None) -> object:
        return self.metadata.get(key, default)

    def scalar(self, key: str) -> int | float | bool | str | None:
        value = self.metadata.get(key)
        if isinstance(value, (int, float, bool, str)):
            return value
        return None


class _Reader:
    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        position = stream.tell()
        stream.seek(0, 2)
        self._end = stream.tell()
        stream.seek(position)

    def _require(self, size: int) -> None:
        if size < 0 or size > self._end - self._stream.tell():
            raise GgufError("unexpected end of file while reading GGUF metadata")

    def skip(self, size: int) -> None:
        self._require(size)
        self._stream.seek(size, 1)

    def read(self, size: int) -> bytes:
        self._require(size)
        data = self._stream.read(size)
        if len(data) != size:
            raise GgufError("unexpected end of file while reading GGUF metadata")
        return data

    def u8(self) -> int:
        return int(_U8.unpack(self.read(1))[0])

    def i8(self) -> int:
        return int(_I8.unpack(self.read(1))[0])

    def u16(self) -> int:
        return int(_U16.unpack(self.read(2))[0])

    def i16(self) -> int:
        return int(_I16.unpack(self.read(2))[0])

    def u32(self) -> int:
        return int(_U32.unpack(self.read(4))[0])

    def i32(self) -> int:
        return int(_I32.unpack(self.read(4))[0])

    def u64(self) -> int:
        return int(_U64.unpack(self.read(8))[0])

    def i64(self) -> int:
        return int(_I64.unpack(self.read(8))[0])

    def f32(self) -> float:
        return float(_F32.unpack(self.read(4))[0])

    def f64(self) -> float:
        return float(_F64.unpack(self.read(8))[0])

    def boolean(self) -> bool:
        value = self.u8()
        if value not in (0, 1):
            raise GgufError("invalid GGUF boolean: expected 0 or 1")
        return bool(value)

    def string(self) -> str:
        length = self.u64()
        raw = self.read(length)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GgufError("invalid UTF-8 in GGUF metadata or tensor name") from exc


def _read_scalar(reader: _Reader, value_type: int) -> int | float | bool:
    if value_type == GGUF_TYPE_STRING:
        raise GgufError("string is not a scalar numeric type")
    if value_type == GGUF_TYPE_ARRAY:
        raise GgufError("array is not a scalar type")
    struct_type = _SCALAR_STRUCTS.get(value_type)
    if struct_type is None:
        raise GgufError(f"unknown GGUF value type {value_type}")
    if value_type == GGUF_TYPE_UINT8:
        return reader.u8()
    if value_type == GGUF_TYPE_INT8:
        return reader.i8()
    if value_type == GGUF_TYPE_UINT16:
        return reader.u16()
    if value_type == GGUF_TYPE_INT16:
        return reader.i16()
    if value_type == GGUF_TYPE_UINT32:
        return reader.u32()
    if value_type == GGUF_TYPE_INT32:
        return reader.i32()
    if value_type == GGUF_TYPE_FLOAT32:
        return reader.f32()
    if value_type == GGUF_TYPE_BOOL:
        return reader.boolean()
    if value_type == GGUF_TYPE_UINT64:
        return reader.u64()
    if value_type == GGUF_TYPE_INT64:
        return reader.i64()
    if value_type == GGUF_TYPE_FLOAT64:
        return reader.f64()
    raise GgufError(f"unhandled GGUF value type {value_type}")  # pragma: no cover


def _skip_value(reader: _Reader, value_type: int) -> None:
    if value_type == GGUF_TYPE_STRING:
        length = reader.u64()
        reader.skip(length)
    elif value_type == GGUF_TYPE_ARRAY:
        element_type = reader.u32()
        count = reader.u64()
        _skip_array_payload(reader, element_type, count)
    else:
        struct_type = _SCALAR_STRUCTS.get(value_type)
        if struct_type is None:
            raise GgufError(f"unknown GGUF value type {value_type}")
        reader.skip(struct_type.size)


def _read_value(reader: _Reader, value_type: int) -> object:
    if value_type == GGUF_TYPE_STRING:
        return reader.string()
    if value_type == GGUF_TYPE_ARRAY:
        element_type = reader.u32()
        count = reader.u64()
        _skip_array_payload(reader, element_type, count)
        return GgufArray(element_type=element_type, count=count)
    return _read_scalar(reader, value_type)


def _skip_array_payload(reader: _Reader, element_type: int, count: int) -> None:
    # Nested arrays are valid GGUF. A file-controlled nesting depth must not
    # consume the Python call stack or be silently truncated. Each nested header
    # still passes the reader's remaining-file checks.
    pending = [(element_type, count)]
    while pending:
        kind, remaining = pending.pop()
        if kind == GGUF_TYPE_ARRAY:
            if remaining:
                child_kind, child_count = reader.u32(), reader.u64()
                if remaining > 1:
                    pending.append((kind, remaining - 1))
                pending.append((child_kind, child_count))
        elif kind == GGUF_TYPE_STRING:
            for _ in range(remaining):
                _skip_value(reader, GGUF_TYPE_STRING)
        else:
            struct_type = _SCALAR_STRUCTS.get(kind)
            if struct_type is None:
                raise GgufError(f"unknown GGUF array element type {kind}")
            reader.skip(struct_type.size * remaining)


def read_gguf(path: Path | str) -> GgufFile:
    """Read a GGUF file's metadata and tensor descriptors without loading tensors."""
    file_path = Path(path)
    with file_path.open("rb") as stream:
        reader = _Reader(stream)
        magic = reader.u32()
        if magic != GGUF_MAGIC:
            raise GgufError(f"not a GGUF file (magic=0x{magic:08x}): {file_path}")
        version = reader.u32()
        if version < 1 or version > 3:
            raise GgufError(f"unsupported GGUF version {version}: {file_path}")
        if version == 1:
            tensor_count = reader.u32()
            kv_count = reader.u32()
        else:
            tensor_count = reader.u64()
            kv_count = reader.u64()

        metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = reader.string()
            if key in metadata:
                raise GgufError(f"ambiguous duplicate GGUF metadata key: {key!r}")
            value_type = reader.u32()
            metadata[key] = _read_value(reader, value_type)

        tensors: list[GgufTensorInfo] = []
        for _ in range(tensor_count):
            name = reader.string()
            n_dims = reader.u32()
            if n_dims > MAX_DIMS:
                raise GgufError(f"tensor {name!r} declares {n_dims} dimensions")
            if version == 1:
                dimensions = tuple(reader.u32() for _ in range(n_dims))
            else:
                dimensions = tuple(reader.u64() for _ in range(n_dims))
            ggml_type = reader.u32()
            offset = reader.u64()
            tensors.append(
                GgufTensorInfo(
                    name=name,
                    dimensions=dimensions,
                    ggml_type=ggml_type,
                    offset=offset,
                )
            )
        return GgufFile(
            path=file_path,
            version=version,
            metadata=metadata,
            tensors=tuple(tensors),
        )
