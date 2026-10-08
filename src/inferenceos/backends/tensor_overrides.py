"""Placement overrides: which buffer single model pieces are loaded into.

Stored in a setup as one text, "PATTERN=BUFFER;PATTERN=BUFFER", in the order given.
``PATTERN`` is a regular expression searched in a piece's (tensor's) name from the
model file; ``BUFFER`` is "CPU" or a device buffer the engine reported (e.g.
"Vulkan0"). The first pattern that matches a piece decides, as in llama.cpp's
``--override-tensor`` (llama.cpp b11139, ``llama-model.cpp``: the overrides are tried
in order and the first match wins). Pieces no pattern matches stay where the layer
count (``-ngl``) puts them.

This module only reads, writes and checks the text against the model's own piece
names; what to try is decided in ``tune/tensor_placement.py`` and how the engine is
told is the backend's job.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

#: The buffer every build has.
CPU_BUFFER = "CPU"

_BUFFER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class OverrideError(ValueError):
    """A placement override text cannot be used."""


def parse_overrides(text: str | None) -> list[tuple[str, str]]:
    """``"a=CPU;b=CPU"`` -> ``[("a", "CPU"), ("b", "CPU")]`` (empty for None or "")."""
    result: list[tuple[str, str]] = []
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        pattern, sep, buffer = part.rpartition("=")
        pattern, buffer = pattern.strip(), buffer.strip()
        if not sep or not pattern or not _BUFFER_RE.match(buffer):
            raise OverrideError(f"not a placement override: {part!r} (expected PATTERN=BUFFER)")
        if "," in pattern:
            raise OverrideError(
                f"a pattern may not contain ',' (the engine separates overrides with it): {pattern!r}"
            )
        try:
            re.compile(pattern)
        except re.error as exc:
            raise OverrideError(f"not a valid pattern: {pattern!r} ({exc})") from exc
        result.append((pattern, buffer))
    return result


def format_overrides(overrides: Sequence[tuple[str, str]]) -> str | None:
    """The canonical text (None when there are none). Order is kept: it matters."""
    if not overrides:
        return None
    text = ";".join(f"{pattern}={buffer}" for pattern, buffer in overrides)
    parse_overrides(text)
    return text


def engine_argument(text: str | None) -> str | None:
    """The value for llama.cpp's ``--override-tensor`` (comma separated)."""
    overrides = parse_overrides(text)
    if not overrides:
        return None
    return ",".join(f"{pattern}={buffer}" for pattern, buffer in overrides)


def is_device_buffer(name: str) -> bool:
    """Whether an engine buffer name is graphics card memory (not CPU or host memory:
    "CPU", "CPU_Mapped", "CPU_REPACK", "Vulkan_Host" ... are on the processor side)."""
    upper = name.upper()
    return not upper.startswith("CPU") and "HOST" not in upper


@dataclass(frozen=True)
class OverrideCheck:
    """An override text checked against the model's piece names and known buffers."""

    ok: bool
    #: Piece name -> the buffer the first matching pattern sends it to.
    assigned: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def check_overrides(
    text: str | None,
    tensor_names: Iterable[str],
    *,
    known_buffers: Iterable[str] | None = None,
    protected: Iterable[str] = ("token_embd.weight",),
) -> OverrideCheck:
    """Check an override text before it is used.

    * every pattern must match at least one piece (an unmatched pattern does nothing
      and would make a candidate look like something it is not);
    * a pattern that matches only pieces an earlier pattern already decided is
      shadowed: refused, because the text would claim something the engine ignores;
    * two patterns matching the same piece with different buffers are reported (the
      first wins, as in the engine) and refused;
    * every buffer must be "CPU" or a buffer the engine reported (unknown buffers are
      refused, never guessed);
    * pieces in ``protected`` (the token table, which the engine keeps on the processor
      and also uses as the output step when the model ties them) are never moved.
    """
    try:
        overrides = parse_overrides(text)
    except OverrideError as exc:
        return OverrideCheck(False, problems=[str(exc)])
    names = list(tensor_names)
    buffers = {CPU_BUFFER, *(known_buffers or ())}
    assigned: dict[str, str] = {}
    problems: list[str] = []
    notes: list[str] = []
    protected_set = set(protected)
    for pattern, buffer in overrides:
        if buffer != CPU_BUFFER and buffer not in buffers:
            problems.append(f"buffer {buffer!r} is not one this engine reported")
        compiled = re.compile(pattern)
        matched = [name for name in names if compiled.search(name)]
        if not matched:
            problems.append(f"pattern {pattern!r} matches no piece of this model")
            continue
        hit_protected = [name for name in matched if name in protected_set]
        if hit_protected:
            problems.append(
                f"pattern {pattern!r} would move {', '.join(hit_protected)}, which is never moved"
            )
        fresh = [name for name in matched if name not in assigned]
        clashing = [name for name in matched if name in assigned and assigned[name] != buffer]
        if clashing:
            problems.append(
                f"pattern {pattern!r} overlaps an earlier one with a different buffer "
                f"({len(clashing)} piece(s), e.g. {clashing[0]})"
            )
        if not fresh:
            problems.append(f"pattern {pattern!r} only matches pieces an earlier pattern decided")
            continue
        if len(fresh) < len(matched):
            notes.append(f"pattern {pattern!r}: {len(matched) - len(fresh)} piece(s) already decided")
        for name in fresh:
            assigned[name] = buffer
    return OverrideCheck(not problems, assigned=assigned, problems=problems, notes=notes)


__all__ = [
    "CPU_BUFFER",
    "OverrideCheck",
    "OverrideError",
    "check_overrides",
    "engine_argument",
    "format_overrides",
    "is_device_buffer",
    "parse_overrides",
]
