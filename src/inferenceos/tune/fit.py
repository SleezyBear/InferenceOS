"""How many graphics card layers fit, worked out from the model file (step 5).

Uses the same safety margins as the layer search (a safety headroom and a working
reserve are never given to the model). Before the model is loaded, everything already
using graphics memory counts as "other programs". The context size is a fixed input:
the answer is how many layers fit *at that context*, never the other way round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from inferenceos.contracts.model import ModelProfile
from inferenceos.optimization.search.policy import SearchPolicy
from inferenceos.tune.placement import max_count

if TYPE_CHECKING:
    from inferenceos.tune.model_facts import ModelFacts


@dataclass(frozen=True)
class LayerFit:
    layers: int | None
    reasons: list[str] = field(default_factory=list)


def piece_bytes_in_order(facts: ModelFacts | None, model: ModelProfile) -> list[int]:
    """Bytes each extra count puts on the card, in the engine's order.

    llama.cpp b11139 puts the output step on the card first, then the blocks from the
    last one backwards (``tune/placement.py``), so count ``N`` holds the output step and
    the last ``N-1`` blocks, and count ``L+1`` holds everything. From the model file
    when its sizes were read; otherwise the weights split evenly over ``L+1`` pieces (an
    estimate).
    """
    if facts is not None and facts.layer_bytes is not None and facts.output_bytes is not None:
        return [int(facts.output_bytes), *[int(size) for size in reversed(facts.layer_bytes)]]
    pieces = max_count(int(model.n_layers))
    per_piece = max(math.ceil(model.weight_bytes / pieces), 1)
    return [per_piece] * pieces


def layers_that_fit(
    *,
    model: ModelProfile,
    context_length: int,
    total_graphics_memory_bytes: int | None,
    used_before_loading_bytes: int | None,
    policy: SearchPolicy,
    facts: ModelFacts | None = None,
) -> LayerFit:
    """Layer count that fits on the graphics card, or ``None`` when it can't be worked out.

    Counts go up to ``L+1`` (everything on the card) and follow the engine's placement
    order (review 2, 4 Oct 2026: the old version split the weights evenly over ``L``
    layers and stopped at ``L``).
    """
    if total_graphics_memory_bytes is None or total_graphics_memory_bytes <= 0:
        return LayerFit(None, ["graphics card memory size unknown"])
    if used_before_loading_bytes is None:
        return LayerFit(None, ["graphics card memory in use could not be read"])
    if model.n_layers < 1 or model.weight_bytes <= 0:
        return LayerFit(None, ["model layer sizes unknown"])
    kv = model.estimate_kv_cache_bytes(context_length, kv_dtype_bytes=policy.kv_element_bytes)
    headroom = int(round(policy.safety_headroom_fraction * total_graphics_memory_bytes))
    reserve = int(round(policy.runtime_reserve_fraction * total_graphics_memory_bytes))
    budget = total_graphics_memory_bytes - used_before_loading_bytes - headroom - reserve - kv
    pieces = piece_bytes_in_order(facts, model)
    fits = 0
    used = 0
    for size in pieces:
        if used + size > budget:
            break
        used += size
        fits += 1
    source = (
        "sizes from the model file"
        if facts is not None and facts.layer_bytes is not None and facts.output_bytes is not None
        else f"about {pieces[0]} bytes per piece (estimate)"
    )
    return LayerFit(
        fits,
        [
            f"graphics memory {total_graphics_memory_bytes} bytes, "
            f"{used_before_loading_bytes} already in use",
            f"kept free: safety {headroom} + working reserve {reserve}; "
            f"conversation memory at context {context_length}: {kv}",
            f"{source}; output step first, then the last blocks; "
            f"{fits} of {len(pieces)} counts fit",
        ],
    )


__all__ = ["LayerFit", "layers_that_fit", "piece_bytes_in_order"]
