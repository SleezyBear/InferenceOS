"""Which pieces of the model go on the graphics card for a layer count.

llama.cpp's ``-ngl N`` ("graphics card layers") is one number, but what it puts on the
card depends on the engine's own rule. The rule here was read from the llama.cpp
source of the installed build (b11139, commit ``42916d83f``, ``src/llama-model.cpp``,
``load_tensors``)::

    i_gpu_start    = max(n_layer + 1 - N, 0)
    act_gpu_layers = min(N, n_layer + 1)
    piece il (0..n_layer-1 = blocks, n_layer = the output layer) goes on the card when
    i_gpu_start <= il < i_gpu_start + act_gpu_layers

So with ``L`` blocks:

* ``N = 0``      nothing on the card,
* ``N = 1``      only the output layer (the step that picks the next word),
* ``N = k``      the output layer and the last ``k - 1`` blocks,
* ``N = L + 1``  everything (the largest useful count).

The token table used to look up the question's words always stays on the processor;
a model with a tied output layer uses a copy of that table as its output layer, placed
like an output layer.

An older rule ("blocks first": the last ``N`` blocks, the output layer only when
``N > L``) is kept for engines that use it. Every placement-dependent number in
``tune`` goes through :func:`placement` so the rule lives in one place.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: llama.cpp's current rule (output layer first). Verified on b11139 / 42916d83f.
OUTPUT_FIRST = "output_first"
#: The older rule: the last N blocks; the output layer only when N > blocks.
BLOCKS_FIRST = "blocks_first"

#: The kind name used for the output layer in per-piece kind lists.
OUTPUT_KIND = "output"


@dataclass(frozen=True)
class Placement:
    """Where the model's pieces are for one layer count."""

    count: int
    n_layers: int
    #: Indices of the blocks on the graphics card (0-based, in model order).
    blocks_on_card: tuple[int, ...]
    output_on_card: bool

    @property
    def blocks_on_processor(self) -> tuple[int, ...]:
        on_card = set(self.blocks_on_card)
        return tuple(index for index in range(self.n_layers) if index not in on_card)

    @property
    def everything_on_card(self) -> bool:
        return self.output_on_card and len(self.blocks_on_card) == self.n_layers

    @property
    def nothing_on_card(self) -> bool:
        return not self.output_on_card and not self.blocks_on_card

    def words(self) -> str:
        """The placement in plain words."""
        if self.nothing_on_card:
            return "nothing on the graphics card"
        if self.everything_on_card:
            return "everything on the graphics card"
        blocks = len(self.blocks_on_card)
        parts = []
        if self.output_on_card:
            parts.append("the output step")
        if blocks:
            parts.append(f"the last {blocks} of {self.n_layers} blocks")
        return " and ".join(parts) + " on the graphics card"


#: Engine builds whose placement rule was checked on a real load with
#: ``scripts/check_placement.py`` (laptop, 4 Oct 2026: Gemma exact, Phi 17 MiB off in
#: total, against 862 / 1,669 MiB for the other rule). Matched against the full version
#: line (build number or commit).
VERIFIED_OUTPUT_FIRST: tuple[str, ...] = ("build 11139", "42916d83f")


def engine_rule(engine_version: str | None) -> str:
    """The placement rule for an engine build.

    Every llama.cpp build InferenceOS has met so far (b11139) uses the output-first
    rule, and an unknown build is assumed to as well; ``rule_verified`` says whether
    that was checked for this build. ``scripts/check_placement.py`` checks it on a real
    load (the engine's own log says whether a count of 1 put the output layer on the
    card).
    """
    return OUTPUT_FIRST


def rule_verified(engine_version: str | None) -> bool:
    """Whether this engine build's placement rule was checked on a real load."""
    text = (engine_version or "").lower()
    return any(marker in text for marker in VERIFIED_OUTPUT_FIRST)


def max_count(n_layers: int, rule: str = OUTPUT_FIRST) -> int:
    """The count that puts everything on the card."""
    return max(int(n_layers), 0) + 1


def placement(count: int | None, n_layers: int, rule: str = OUTPUT_FIRST) -> Placement:
    """Which pieces are on the card for ``count`` under ``rule``."""
    total = max(int(n_layers), 0)
    n = max(int(count or 0), 0)
    if rule == BLOCKS_FIRST:
        blocks = min(n, total)
        on_card = tuple(range(total - blocks, total))
        return Placement(n, total, on_card, output_on_card=n > total)
    # output first
    start = max(total + 1 - n, 0)
    active = min(n, total + 1)
    pieces = [index for index in range(total + 1) if start <= index < start + active]
    on_card = tuple(index for index in pieces if index < total)
    return Placement(n, total, on_card, output_on_card=total in pieces)


def card_kinds(block_kinds: Sequence[str] | None, n_layers: int) -> list[str]:
    """Kind of the *first* piece moved onto the card for each count, indexed so that
    ``kinds[max_count - count]`` is the piece count ``count`` adds (output-first rule):
    the block kinds in model order, then the output layer last. Blocks of unknown kind
    are named ``block``."""
    kinds = list(block_kinds) if block_kinds is not None else ["block"] * max(int(n_layers), 0)
    return [*kinds, OUTPUT_KIND]


__all__ = [
    "BLOCKS_FIRST",
    "OUTPUT_FIRST",
    "OUTPUT_KIND",
    "Placement",
    "card_kinds",
    "engine_rule",
    "rule_verified",
    "max_count",
    "placement",
]
