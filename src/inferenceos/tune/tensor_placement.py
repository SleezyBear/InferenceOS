"""Where every piece of the model sits: the layer count plus explicit overrides.

The layer count (``-ngl``) moves whole blocks; overrides (``placement_overrides``,
llama.cpp ``--override-tensor``) then move single pieces by name. Once a setup has
overrides they are the source of truth for where the weights are: every estimate here
(what is on the card, what the processor works through per word) is worked out piece
by piece from the model file's own piece names and sizes, never from the layer count
alone. When the piece sizes are unknown the answer is unknown (``None``), not the
layer-count layout.

What stays tied to the layer count: the conversation memory (KV cache) and the
recurrent state of a block live with the block's layer, wherever its weights are
moved (llama.cpp b11139 assigns them per layer). The token table always stays on the
processor; a model without its own output step reuses the token table as the output
step, so that step cannot be moved on its own (``tied_output``).

The placement path tries a few structural alternatives (never every subset of blocks):

* the output step on the processor, with the card room it frees used for more blocks;
* the expert pieces of a mixture-of-experts model on the processor (all of them, or
  those of the first blocks), with everything else on the card;
* for a model mixing attention blocks with other kinds (state-space, convolution): one
  kind's weights on the card and the other's on the processor.

Estimates only choose what to try. A candidate is kept only after the engine's own
load log shows the pieces really went where they were sent (``verify_placement``: the
engine's per-piece lines for identity, the buffer total as a consistency check), its
answers pass the normal answer checks and the normal speed and resource checks pass.

The block-kind grouping for hybrid models is generated and screened, but not eligible
for acceptance until a hardware probe has confirmed it runs correctly on this engine
(``tune_grouped_placement_probed``; reviewer, 6 Oct 2026).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from inferenceos.backends.tensor_overrides import (
    CPU_BUFFER,
    check_overrides,
    format_overrides,
    is_device_buffer,
    parse_overrides,
)
from inferenceos.tune.model_facts import ModelFacts
from inferenceos.tune.placement import OUTPUT_FIRST, placement

SHOWN, NOT_SEEN, UNVERIFIED = "shown", "not seen", "unverified"

#: Pieces of a block that hold the experts (llama.cpp names; ``--cpu-moe`` uses the same).
EXPERT_PATTERN = r"ffn_(up|down|gate)_exps"
#: The output step and its norm.
OUTPUT_PATTERN = r"^output"
#: How far the card's model buffers may be from the estimate and still count as the
#: requested placement (laptop, 4 Oct 2026: the layer-count estimate was exact for
#: Gemma and 17 MiB off for Phi).
VERIFY_TOLERANCE_BYTES = 48 * 1024 * 1024
VERIFY_TOLERANCE_SHARE = 0.04

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass(frozen=True)
class PiecePlacement:
    """Where the weights are for one setup."""

    card_pieces: frozenset[str]
    card_bytes: int
    processor_bytes: int
    #: Bytes worked through per written word (only the experts used per word count).
    card_active_bytes: int
    processor_active_bytes: int
    #: Blocks whose layer (and so conversation memory / state) is on the card.
    layers_on_card: tuple[int, ...]
    output_on_card: bool
    #: Pieces an override moved away from where the layer count put them.
    moved: dict[str, str] = field(default_factory=dict)

    @property
    def processor_share(self) -> float | None:
        whole = self.card_active_bytes + self.processor_active_bytes
        return self.processor_active_bytes / whole if whole > 0 else None


def _block(name: str) -> int | None:
    found = _BLOCK_RE.match(name)
    return int(found.group(1)) if found else None


def _active_fraction(facts: ModelFacts) -> float:
    if facts.experts and facts.experts_used and facts.experts > 0:
        return min(facts.experts_used / facts.experts, 1.0)
    return 1.0


def piece_placement(
    facts: ModelFacts,
    gpu_layers: int | None,
    overrides: str | None = None,
    *,
    card: bool = True,
    rule: str = OUTPUT_FIRST,
) -> PiecePlacement | None:
    """Where every piece is for ``gpu_layers`` and ``overrides``.

    ``None`` when it cannot be worked out: the piece sizes are unknown, the layer count
    is left to the engine, or the override text does not fit this model.
    """
    if facts.tensor_bytes is None or gpu_layers is None:
        return None
    total = int(facts.total_layers)
    layers = placement(gpu_layers if card else 0, total, rule)
    on_card_layers = set(layers.blocks_on_card)
    names = list(facts.tensor_bytes)
    checked = check_overrides(overrides, names)
    if overrides and not checked.ok:
        return None
    fraction = _active_fraction(facts)
    card_pieces: set[str] = set()
    card_bytes = processor_bytes = 0
    card_active = processor_active = 0
    moved: dict[str, str] = {}
    for name, size in facts.tensor_bytes.items():
        index = _block(name)
        if index is not None:
            default_card = index in on_card_layers
        elif name.startswith("output"):
            default_card = layers.output_on_card
        else:
            default_card = False
        target = checked.assigned.get(name)
        on_card = default_card if target is None else (card and target != CPU_BUFFER)
        if target is not None and on_card != default_card:
            moved[name] = target
        if index is None and not name.startswith("output"):
            # The token table is looked up one row per word, not worked through.
            work = 0
        else:
            work = int(size * fraction) if re.search(EXPERT_PATTERN, name) else int(size)
        if on_card:
            card_pieces.add(name)
            card_bytes += int(size)
            card_active += work
        else:
            processor_bytes += int(size)
            processor_active += work
    if facts.tied_output and "token_embd.weight" in facts.tensor_bytes:
        copy = int(facts.tensor_bytes["token_embd.weight"])
        if layers.output_on_card and card:
            card_bytes += copy
            card_active += copy
        else:
            processor_active += copy
    return PiecePlacement(
        card_pieces=frozenset(card_pieces),
        card_bytes=card_bytes,
        processor_bytes=processor_bytes,
        card_active_bytes=card_active,
        processor_active_bytes=processor_active,
        layers_on_card=tuple(sorted(on_card_layers)),
        output_on_card=layers.output_on_card,
        moved=moved,
    )


def placement_words(facts: ModelFacts, gpu_layers: int | None, overrides: str | None) -> str:
    """The placement in plain words (with overrides, the moved pieces are named by kind)."""
    where = piece_placement(facts, gpu_layers, overrides)
    base = placement(gpu_layers, int(facts.total_layers)).words() if gpu_layers is not None else "llama.cpp decides"
    if not overrides:
        return base
    if where is None:
        return base + "; some pieces moved by overrides (sizes unknown)"
    kinds: dict[str, int] = {}
    for name in where.moved:
        kind = "experts" if re.search(EXPERT_PATTERN, name) else "output step" if name.startswith("output") else "other pieces"
        kinds[kind] = kinds.get(kind, 0) + 1
    moved = ", ".join(f"{count} {kind} piece(s)" for kind, count in kinds.items()) or "nothing"
    return f"{base}; moved to the processor: {moved}"


# --------------------------------------------------------------------------- #
# Candidates                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlacementPlan:
    """One alternative: a layer count and the overrides that go with it."""

    kind: str
    gpu_layers: int
    overrides: str | None
    label: str
    #: Estimated card bytes for the weights (None = unknown).
    card_bytes: int | None
    #: A number the refinement step moves (blocks added, or expert blocks moved).
    knob: int | None = None


def _index_alternation(indices: Sequence[int]) -> str:
    return "|".join(str(index) for index in sorted(set(indices)))


def blocks_pattern(indices: Sequence[int], part: str = "") -> str:
    """A pattern matching the pieces of the given blocks (all of them, or ``part``)."""
    tail = rf"\.{part}" if part else r"\."
    return rf"^blk\.({_index_alternation(indices)}){tail}"


def output_on_processor_plan(
    facts: ModelFacts, gpu_layers: int, extra_blocks: int
) -> PlacementPlan | None:
    """The output step on the processor and ``extra_blocks`` more blocks on the card."""
    total = int(facts.total_layers)
    if extra_blocks < 1 or gpu_layers < 1 or gpu_layers > total:
        return None
    count = min(gpu_layers + extra_blocks, total + 1)
    overrides = format_overrides([(OUTPUT_PATTERN, CPU_BUFFER)])
    where = piece_placement(facts, count, overrides)
    blocks = len(placement(count, total).blocks_on_card)
    return PlacementPlan(
        kind="output",
        gpu_layers=count,
        overrides=overrides,
        label=f"output step on the processor, {blocks} blocks on the card",
        card_bytes=where.card_bytes if where is not None else None,
        knob=extra_blocks,
    )


def experts_plan(facts: ModelFacts, expert_blocks: int) -> PlacementPlan | None:
    """Every block on the card except the experts of the first ``expert_blocks`` blocks."""
    total = int(facts.total_layers)
    with_experts = [index for index, size in enumerate(facts.expert_bytes or ()) if size > 0]
    if not with_experts or expert_blocks < 1:
        return None
    chosen = with_experts[: min(expert_blocks, len(with_experts))]
    if len(chosen) == len(with_experts):
        pattern = EXPERT_PATTERN
        label = "all experts on the processor, everything else on the card"
    else:
        pattern = blocks_pattern(chosen, EXPERT_PATTERN)
        label = (
            f"experts of the first {len(chosen)} of {len(with_experts)} blocks on the "
            "processor, everything else on the card"
        )
    overrides = format_overrides([(pattern, CPU_BUFFER)])
    where = piece_placement(facts, total + 1, overrides)
    return PlacementPlan(
        kind="experts",
        gpu_layers=total + 1,
        overrides=overrides,
        label=label,
        card_bytes=where.card_bytes if where is not None else None,
        knob=len(chosen),
    )


def grouped_plan(facts: ModelFacts, on_card_kind: str) -> PlacementPlan | None:
    """All layers on the card, then the weights of every block *not* of
    ``on_card_kind`` moved to the processor (hybrid models)."""
    kinds = facts.layer_kinds
    if not kinds or len(set(kinds)) < 2:
        return None
    total = int(facts.total_layers)
    away = [index for index, kind in enumerate(kinds) if kind != on_card_kind]
    if not away or len(away) == len(kinds):
        return None
    overrides = format_overrides([(blocks_pattern(away), CPU_BUFFER)])
    where = piece_placement(facts, total + 1, overrides)
    kept = len(kinds) - len(away)
    return PlacementPlan(
        kind=f"grouped:{on_card_kind}",
        gpu_layers=total + 1,
        overrides=overrides,
        label=f"weights of the {kept} {on_card_kind} blocks on the card, the other {len(away)} on the processor",
        card_bytes=where.card_bytes if where is not None else None,
    )


@dataclass(frozen=True)
class PlacementOptions:
    plans: list[PlacementPlan]
    notes: list[str]


def placement_plans(
    facts: ModelFacts,
    *,
    gpu_layers: int | None,
    current_overrides: str | None,
    card_budget_bytes: int | None,
    card: bool,
) -> PlacementOptions:
    """The structural alternatives worth trying for this model and setup.

    ``card_budget_bytes``: room on the card for weights (free memory minus the safety
    margins and the conversation memory); plans estimated not to fit are left out with
    a note. ``None`` = unknown: nothing is left out for room.
    """
    notes: list[str] = []
    plans: list[PlacementPlan] = []
    if not card:
        return PlacementOptions([], ["no graphics card is used"])
    if facts.tensor_bytes is None or gpu_layers is None:
        return PlacementOptions([], ["the model's piece sizes are unknown"])
    total = int(facts.total_layers)

    if current_overrides:
        plans.append(
            PlacementPlan(
                kind="reset",
                gpu_layers=gpu_layers,
                overrides=None,
                label="no overrides (the layer count alone)",
                card_bytes=(where.card_bytes if (where := piece_placement(facts, gpu_layers, None)) else None),
            )
        )

    if facts.tied_output:
        notes.append(
            "output step not tried on its own: this model reuses the token table as its "
            "output step (tied), so it cannot be moved separately"
        )
    elif 1 <= gpu_layers <= total and facts.output_bytes:
        current = piece_placement(facts, gpu_layers, None)
        limit = card_budget_bytes if card_budget_bytes is not None else (current.card_bytes if current else None)
        extra = 0
        for blocks in range(1, total + 2 - gpu_layers):
            plan = output_on_processor_plan(facts, gpu_layers, blocks)
            if plan is None or plan.card_bytes is None or limit is None or plan.card_bytes > limit:
                break
            extra = blocks
        for blocks in sorted({extra, max(extra // 2, 1)}, reverse=True):
            plan = output_on_processor_plan(facts, gpu_layers, blocks) if extra else None
            if plan is not None:
                plans.append(plan)
        if extra < 1:
            notes.append("output step not moved: the room it frees holds no whole block")
    elif gpu_layers == total + 1:
        notes.append("output step not moved: everything already fits on the card")

    if facts.uses_experts and facts.expert_bytes and any(facts.expert_bytes):
        with_experts = [index for index, size in enumerate(facts.expert_bytes) if size > 0]
        plan_all = experts_plan(facts, len(with_experts))
        if plan_all is not None:
            plans.append(plan_all)
        if card_budget_bytes is not None and plan_all is not None and plan_all.card_bytes is not None:
            for count in range(1, len(with_experts)):
                plan = experts_plan(facts, count)
                if plan is not None and plan.card_bytes is not None and plan.card_bytes <= card_budget_bytes:
                    plans.append(plan)
                    break
    elif facts.uses_experts:
        notes.append("expert pieces not found by name; experts not moved")

    kinds = facts.layer_kinds or ()
    if len(set(kinds)) >= 2 and "attention" in kinds:
        for kind in ("attention", *sorted(set(kinds) - {"attention"})):
            plan = grouped_plan(facts, kind)
            if plan is not None:
                plans.append(plan)

    if any(plan.kind == "experts" or plan.kind.startswith("grouped") for plan in plans):
        notes.append(
            "pieces left on the processor are worked there: each word passes a block's "
            "intermediate numbers between card and processor (small), and for long "
            "questions the engine may copy those weights to the card per batch (large); "
            "the long question guards against that"
        )
    kept: list[PlacementPlan] = []
    for plan in plans:
        if (
            card_budget_bytes is not None
            and plan.card_bytes is not None
            and plan.card_bytes > card_budget_bytes
            and plan.kind != "reset"
        ):
            notes.append(
                f"not tried, estimated not to fit on the card ({plan.card_bytes / 1024**2:,.0f} "
                f"MiB of weights, room {card_budget_bytes / 1024**2:,.0f} MiB): {plan.label}"
            )
            continue
        if plan.overrides == current_overrides and plan.gpu_layers == gpu_layers:
            continue
        if any(item.overrides == plan.overrides and item.gpu_layers == plan.gpu_layers for item in kept):
            continue
        kept.append(plan)
    return PlacementOptions(kept, notes)


def refine_plans(facts: ModelFacts, best: PlacementPlan, tried: set[tuple[int, str | None]]) -> list[PlacementPlan]:
    """The neighbours of a promising plan: one block more or fewer."""
    out: list[PlacementPlan] = []
    if best.knob is None:
        return out
    for knob in (best.knob - 1, best.knob + 1):
        if knob < 1:
            continue
        if best.kind == "output":
            base = best.gpu_layers - best.knob
            plan = output_on_processor_plan(facts, base, knob)
        elif best.kind == "experts":
            plan = experts_plan(facts, knob)
        else:
            plan = None
        if plan is not None and (plan.gpu_layers, plan.overrides) not in tried:
            out.append(plan)
    return out


# --------------------------------------------------------------------------- #
# Did it take effect?                                                         #
# --------------------------------------------------------------------------- #
def device_model_bytes(observed: dict[str, object] | None) -> int | None:
    """Bytes of the card's model buffers in an engine log (None = not shown)."""
    if not observed:
        return None
    buffers = observed.get("model_buffers")
    if not isinstance(buffers, dict) or not buffers:
        return None
    return sum(int(value) for name, value in buffers.items() if is_device_buffer(str(name)))


def _buffer_side(name: str) -> str:
    """"card" or "processor" for an engine buffer name."""
    return "card" if is_device_buffer(name) else "processor"


def verify_placement(
    facts: ModelFacts,
    gpu_layers: int | None,
    overrides: str | None,
    observed: dict[str, object] | None,
    *,
    card: bool = True,
) -> tuple[str, str]:
    """Did the weights go where they were sent?

    Review 7: buffer totals alone do not prove a placement (an ignored override can leave
    a total within the tolerance, and equal-sized pieces swapped give the same total).
    So, with overrides, the **identity** comes from the engine's own per-piece lines
    ("tensor X ... buffer type overridden to Y", ``overridden_tensors``): every piece the
    overrides send somewhere must be listed with a buffer on the requested side, and no
    other piece may be listed. Without those lines the placement is "unverified". The
    card's model buffer total must still agree with the piece-by-piece estimate (a
    consistency check).

    Without overrides nothing may be listed as overridden, and the total must agree with
    the layer-count layout.
    """
    where = piece_placement(facts, gpu_layers, overrides, card=card)
    if where is None:
        return UNVERIFIED, "placement could not be worked out from the model file"
    listed_raw = (observed or {}).get("overridden_tensors")
    listed: dict[str, str] | None = (
        {str(name): str(buffer) for name, buffer in listed_raw.items()}
        if isinstance(listed_raw, dict)
        else None
    )
    identity_words = ""
    if overrides:
        if facts.tensor_bytes is None:
            return UNVERIFIED, "the model's piece names are unknown"
        expected = check_overrides(overrides, list(facts.tensor_bytes)).assigned
        if not expected:
            return UNVERIFIED, "the overrides match no piece of this model"
        if not listed:
            return UNVERIFIED, (
                "the engine did not list the pieces it moved (per-piece lines need the "
                "debug log level), so the placement cannot be told apart from an ignored "
                "or different one"
            )
        missing = sorted(name for name in expected if name not in listed)
        wrong = sorted(
            name for name, buffer in expected.items()
            if name in listed and _buffer_side(listed[name]) != _buffer_side(buffer)
        )
        extra = sorted(name for name in listed if name not in expected)
        if missing or wrong or extra:
            parts = []
            if missing:
                parts.append(f"{len(missing)} piece(s) not moved (e.g. {missing[0]})")
            if wrong:
                parts.append(f"{len(wrong)} piece(s) on the wrong side (e.g. {wrong[0]} -> {listed[wrong[0]]})")
            if extra:
                parts.append(f"{len(extra)} other piece(s) moved (e.g. {extra[0]})")
            return NOT_SEEN, "the engine's per-piece lines differ from the request: " + "; ".join(parts)
        identity_words = f"all {len(expected)} moved pieces listed by the engine where requested"
    elif listed:
        first = next(iter(listed))
        return NOT_SEEN, f"no overrides requested, but the engine moved {len(listed)} piece(s) (e.g. {first})"
    seen = device_model_bytes(observed)
    if seen is None:
        return UNVERIFIED, (
            (identity_words + "; but " if identity_words else "")
            + "the engine did not print its model buffer sizes in this run"
        )
    expected_bytes = where.card_bytes
    tolerance = max(VERIFY_TOLERANCE_BYTES, int(VERIFY_TOLERANCE_SHARE * max(expected_bytes, 1)))
    words = (
        f"card model buffers {seen / 1024**2:,.0f} MiB, expected "
        f"{expected_bytes / 1024**2:,.0f} MiB for this placement"
    )
    if identity_words:
        words = identity_words + "; " + words
    if abs(seen - expected_bytes) <= tolerance:
        return SHOWN, words
    return NOT_SEEN, words + " (not the requested placement)"


def override_problems(facts: ModelFacts, overrides: str | None, known_buffers: Sequence[str] = ()) -> list[str]:
    """Problems with an override text for this model (empty = usable)."""
    if not overrides:
        return []
    if facts.tensor_bytes is None:
        return ["the model's piece names are unknown"]
    try:
        parse_overrides(overrides)
    except ValueError as exc:
        return [str(exc)]
    return check_overrides(overrides, list(facts.tensor_bytes), known_buffers=known_buffers).problems


__all__ = [
    "EXPERT_PATTERN",
    "NOT_SEEN",
    "OUTPUT_PATTERN",
    "PiecePlacement",
    "PlacementOptions",
    "PlacementPlan",
    "SHOWN",
    "UNVERIFIED",
    "blocks_pattern",
    "device_model_bytes",
    "experts_plan",
    "grouped_plan",
    "output_on_processor_plan",
    "override_problems",
    "piece_placement",
    "placement_plans",
    "placement_words",
    "refine_plans",
    "verify_placement",
]
