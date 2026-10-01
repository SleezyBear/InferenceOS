"""How many graphics card layers fit, worked out from the model file (step 5).

Uses the same safety margins as the layer search (a safety headroom and a working
reserve are never given to the model). Before the model is loaded, everything already
using graphics memory counts as "other programs". The context size is a fixed input:
the answer is how many layers fit *at that context*, never the other way round.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from inferenceos.contracts.model import ModelProfile
from inferenceos.optimization.search.policy import SearchPolicy


@dataclass(frozen=True)
class LayerFit:
    layers: int | None
    reasons: list[str] = field(default_factory=list)


def layers_that_fit(
    *,
    model: ModelProfile,
    context_length: int,
    total_graphics_memory_bytes: int | None,
    used_before_loading_bytes: int | None,
    policy: SearchPolicy,
) -> LayerFit:
    """Layers that fit on the graphics card, or ``None`` when it can't be worked out."""
    if total_graphics_memory_bytes is None or total_graphics_memory_bytes <= 0:
        return LayerFit(None, ["graphics card memory size unknown"])
    if used_before_loading_bytes is None:
        return LayerFit(None, ["graphics card memory in use could not be read"])
    if model.n_layers < 1 or model.weight_bytes <= 0:
        return LayerFit(None, ["model layer sizes unknown"])
    kv = model.estimate_kv_cache_bytes(context_length, kv_dtype_bytes=policy.kv_element_bytes)
    headroom = int(round(policy.safety_headroom_fraction * total_graphics_memory_bytes))
    reserve = int(round(policy.runtime_reserve_fraction * total_graphics_memory_bytes))
    per_layer = max(math.ceil(model.weight_bytes / model.n_layers), 1)
    budget = total_graphics_memory_bytes - used_before_loading_bytes - headroom - reserve - kv
    fits = 0 if budget <= 0 else min(int(model.n_layers), int(budget // per_layer))
    return LayerFit(
        fits,
        [
            f"graphics memory {total_graphics_memory_bytes} bytes, "
            f"{used_before_loading_bytes} already in use",
            f"kept free: safety {headroom} + working reserve {reserve}; "
            f"conversation memory at context {context_length}: {kv}",
            f"about {per_layer} bytes per layer; {fits} of {model.n_layers} layers fit",
        ],
    )


__all__ = ["LayerFit", "layers_that_fit"]
