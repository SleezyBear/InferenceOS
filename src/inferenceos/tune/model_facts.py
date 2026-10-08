"""Facts about the model and the machine that the optimization paths look at.

The paths after the layer path used to try the same values and the same test questions
for every model. These facts let each path decide whether it applies at all, what is
worth trying, and which test question shows its effect:

* where the model runs: how much of it (by size) is on the processor for the current
  number of graphics card layers. llama.cpp puts the *last* N layers on the card and
  the output layer only when N is larger than the layer count;
* the model's design, read from the tensor names in the model file: how many layers
  use attention (the others are recurrent / state-space or short-convolution layers),
  and whether it uses experts;
* the machine: processor cores, memory speed, free graphics card memory.

Nothing here is a measurement of speed. Unreadable values stay unknown (``None``) and a
path then behaves as before rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.hardware.gpu import total_vram_bytes
from inferenceos.models.gguf import GgufFile, read_gguf
from inferenceos.tune.placement import OUTPUT_FIRST, placement

#: Tensor-name parts that mark an attention layer.
_ATTENTION_PARTS = ("attn_q", "attn_k", "attn_v", "attn_qkv")
#: Tensor-name prefixes that mark a layer that is *not* attention, even when it has an
#: ``attn_*`` tensor (some hybrid models name their linear-attention input ``attn_qkv``).
_OTHER_KINDS: tuple[tuple[str, str], ...] = (
    ("ssm_", "recurrent (state-space)"),
    ("shortconv", "short convolution"),
    ("time_mix", "recurrent (RWKV)"),
)
_EXPERT_PARTS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


@dataclass(frozen=True)
class ModelFacts:
    """What the model file says about the model's design and size."""

    total_layers: int
    #: True when the per-layer facts below come from the model file.
    from_file: bool = False
    attention_layers: int | None = None
    #: Plain names of the other layer kinds, with counts.
    other_layers: dict[str, int] = field(default_factory=dict)
    experts: int | None = None
    experts_used: int | None = None
    #: Bytes of each ``blk.<i>.`` layer, in order.
    layer_bytes: tuple[int, ...] | None = None
    #: Bytes of each layer that are worked through per word: all of it, except that
    #: only ``experts_used / experts`` of the expert tensors count.
    active_layer_bytes: tuple[int, ...] | None = None
    #: Bytes of the output layer (and its norm).
    output_bytes: int | None = None
    #: Bytes of everything else that stays on the processor (token embeddings).
    other_bytes: int | None = None
    vocabulary_size: int | None = None
    embedding_dim: int | None = None
    attention_heads: int | None = None
    #: Every piece (tensor) name with its bytes, in file order (None = not read).
    tensor_bytes: dict[str, int] | None = None
    #: Kind of each block in order: "attention" or one of the other kinds' words.
    layer_kinds: tuple[str, ...] | None = None
    #: True when the model has no separate output step and reuses the token table.
    tied_output: bool | None = None
    #: Bytes of the expert pieces of each block (0 for blocks without experts).
    expert_bytes: tuple[int, ...] | None = None
    #: Width of one cached key and one cached value row per block (from the shapes of
    #: the ``attn_k`` / ``attn_v`` pieces; 0 for blocks without attention; None when
    #: a block's width could not be read).
    key_widths: tuple[int | None, ...] | None = None
    value_widths: tuple[int | None, ...] | None = None
    #: The model's sliding-window size when it uses one (some blocks then keep a
    #: smaller cache, so full-size cache estimates are upper bounds).
    sliding_window: int | None = None

    @property
    def total_bytes(self) -> int | None:
        if self.layer_bytes is None or self.output_bytes is None:
            return None
        return sum(self.layer_bytes) + self.output_bytes

    @property
    def uses_experts(self) -> bool:
        return bool(self.experts and self.experts > 1)

    def design_words(self) -> str:
        """The model's design in plain words."""
        parts: list[str] = [f"{self.total_layers} layers"]
        if self.attention_layers is not None:
            if self.other_layers:
                others = ", ".join(f"{count} {kind}" for kind, count in self.other_layers.items())
                parts.append(f"{self.attention_layers} use attention ({others})")
            else:
                parts.append("all use attention")
        else:
            parts.append("layer kinds unknown")
        if self.uses_experts:
            used = f", {self.experts_used} used per word" if self.experts_used else ""
            parts.append(f"{self.experts} experts{used}")
        return "; ".join(parts)


def _tensor_bytes(gguf: GgufFile) -> dict[str, int]:
    """Bytes of each tensor, from the gaps between data offsets (exact, with padding)."""
    ordered = sorted(gguf.tensors, key=lambda tensor: tensor.offset)
    sizes: dict[str, int] = {}
    try:
        data_end = Path(gguf.path).stat().st_size
    except OSError:
        data_end = None
    # Offsets are relative to the start of the data section, which we don't know
    # exactly; the last tensor's size is estimated from the average bytes per element.
    for current, following in zip(ordered, ordered[1:]):
        sizes[current.name] = max(following.offset - current.offset, 0)
    if ordered:
        last = ordered[-1]
        counted = sum(sizes.values())
        elements = sum(tensor.element_count for tensor in ordered[:-1]) or 1
        per_element = counted / elements
        estimate = int(last.element_count * per_element)
        if data_end is not None and counted + estimate > data_end:
            estimate = max(data_end - counted, 0)
        sizes[last.name] = estimate
    return sizes


def _block_index(name: str) -> tuple[int, str] | None:
    if not name.startswith("blk."):
        return None
    rest = name[len("blk.") :]
    dot = rest.find(".")
    if dot <= 0 or not rest[:dot].isdigit():
        return None
    return int(rest[:dot]), rest[dot + 1 :]


def _metadata_int(gguf: GgufFile, suffix: str) -> int | None:
    architecture = gguf.get("general.architecture")
    keys = []
    if isinstance(architecture, str):
        keys.append(f"{architecture}.{suffix}")
    keys.append(suffix)
    for key in keys:
        value = gguf.metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for key, value in gguf.metadata.items():
        if key.endswith("." + suffix) and isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def facts_from_gguf(gguf: GgufFile, model: ModelProfile) -> ModelFacts:
    sizes = _tensor_bytes(gguf)
    blocks: dict[int, list[str]] = {}
    layer_bytes: dict[int, int] = {}
    expert_bytes: dict[int, int] = {}
    output_bytes = 0
    other_bytes = 0
    for tensor in gguf.tensors:
        parsed = _block_index(tensor.name)
        size = sizes.get(tensor.name, 0)
        if parsed is None:
            if tensor.name.startswith("output"):
                output_bytes += size
            else:
                other_bytes += size
            continue
        index, part = parsed
        blocks.setdefault(index, []).append(part)
        layer_bytes[index] = layer_bytes.get(index, 0) + size
        if part.split(".")[0] in _EXPERT_PARTS:
            expert_bytes[index] = expert_bytes.get(index, 0) + size
    if not blocks:
        return ModelFacts(total_layers=int(model.n_layers))
    total = max(blocks) + 1
    attention = 0
    others: dict[str, int] = {}
    for index in range(total):
        parts = blocks.get(index, [])
        kind: str | None = None
        for prefix, words in _OTHER_KINDS:
            if any(part.startswith(prefix) for part in parts):
                kind = words
                break
        if kind is None and any(
            part.split(".")[0] in _ATTENTION_PARTS for part in parts
        ):
            attention += 1
            continue
        kind = kind or "other"
        others[kind] = others.get(kind, 0) + 1
    experts = _metadata_int(gguf, "expert_count")
    if experts is None and any(
        part.split(".")[0] in _EXPERT_PARTS for parts in blocks.values() for part in parts
    ):
        experts = 0  # experts present but their number is not stated
    if not any(tensor.name == "output.weight" for tensor in gguf.tensors):
        # Tied embeddings: llama.cpp uses a copy of the token table as the output layer,
        # placed like an output layer, so the processor works through it for each word
        # unless the output layer is on the card.
        output_bytes += sum(
            sizes.get(tensor.name, 0)
            for tensor in gguf.tensors
            if tensor.name == "token_embd.weight"
        )
    tied = not any(tensor.name == "output.weight" for tensor in gguf.tensors)
    kinds_list: list[str] = []
    for index in range(total):
        parts = blocks.get(index, [])
        found = next(
            (words for prefix, words in _OTHER_KINDS if any(part.startswith(prefix) for part in parts)),
            None,
        )
        if found is None and any(part.split(".")[0] in _ATTENTION_PARTS for part in parts):
            found = "attention"
        kinds_list.append(found or "other")
    shapes = {tensor.name: tensor.dimensions for tensor in gguf.tensors}
    key_widths: list[int | None] = []
    value_widths: list[int | None] = []
    for index in range(total):
        if kinds_list[index] != "attention":
            key_widths.append(0)
            value_widths.append(0)
            continue
        k_shape = shapes.get(f"blk.{index}.attn_k.weight")
        v_shape = shapes.get(f"blk.{index}.attn_v.weight")
        key_widths.append(int(k_shape[1]) if k_shape is not None and len(k_shape) > 1 else None)
        value_widths.append(int(v_shape[1]) if v_shape is not None and len(v_shape) > 1 else None)
    used = _metadata_int(gguf, "expert_used_count")
    fraction = (used / experts) if used and experts else 1.0
    active = tuple(
        int(layer_bytes.get(index, 0) - expert_bytes.get(index, 0) * (1.0 - fraction))
        for index in range(total)
    )
    return ModelFacts(
        total_layers=total,
        from_file=True,
        active_layer_bytes=active,
        attention_layers=attention,
        other_layers=others,
        experts=experts,
        experts_used=used,
        layer_bytes=tuple(layer_bytes.get(index, 0) for index in range(total)),
        output_bytes=output_bytes,
        other_bytes=other_bytes,
        vocabulary_size=model.vocabulary_size,
        embedding_dim=model.embedding_dim,
        attention_heads=model.n_attention_heads,
        tensor_bytes={tensor.name: sizes.get(tensor.name, 0) for tensor in gguf.tensors},
        layer_kinds=tuple(kinds_list),
        tied_output=tied,
        expert_bytes=tuple(expert_bytes.get(index, 0) for index in range(total)),
        key_widths=tuple(key_widths),
        value_widths=tuple(value_widths),
        sliding_window=_metadata_int(gguf, "attention.sliding_window"),
    )


def read_model_facts(path: Path | str, model: ModelProfile) -> ModelFacts:
    """The model's facts from its file, or only what the profile knows if unreadable."""
    try:
        gguf = read_gguf(path)
    except Exception:  # noqa: BLE001 - an unreadable file just means fewer facts
        return ModelFacts(
            total_layers=int(model.n_layers),
            vocabulary_size=model.vocabulary_size,
            embedding_dim=model.embedding_dim,
            attention_heads=model.n_attention_heads,
        )
    return facts_from_gguf(gguf, model)


def card_usable(hardware: HardwareProfile) -> bool:
    """A graphics card llama.cpp can run on is present."""
    return total_vram_bytes(hardware) is not None


def processor_share(
    facts: ModelFacts,
    gpu_layers: int | None,
    *,
    card: bool,
    rule: str = OUTPUT_FIRST,
    overrides: str | None = None,
) -> float | None:
    """Estimated share of the work per written word that runs on the processor.

    Where each piece sits comes from :func:`inferenceos.tune.placement.placement` (the
    engine's own rule). The work of a piece is estimated from its size; for a model with
    experts only the experts used per word count (``experts_used / experts`` of the
    expert tensors). This is an estimate from sizes, not a measured time. ``None`` when
    unknown (llama.cpp decides the layers itself); without sizes the blocks are taken
    as equal and the output layer is ignored.
    """
    if not card:
        return 1.0
    if gpu_layers is None:
        return None
    if overrides:
        # Overrides are the source of truth for the weights: piece by piece, or unknown.
        from inferenceos.tune.tensor_placement import piece_placement

        where_pieces = piece_placement(facts, gpu_layers, overrides, card=card, rule=rule)
        return where_pieces.processor_share if where_pieces is not None else None
    total = max(int(facts.total_layers), 1)
    where = placement(gpu_layers, total, rule)
    if facts.layer_bytes is None or facts.output_bytes is None:
        return len(where.blocks_on_processor) / total
    work = list(facts.active_layer_bytes or facts.layer_bytes)
    processor = sum(work[index] for index in where.blocks_on_processor if index < len(work))
    if not where.output_on_card:
        processor += facts.output_bytes
    whole = sum(work) + facts.output_bytes
    if whole <= 0:
        return None
    return processor / whole


def memory_speed_mhz(hardware: HardwareProfile) -> int | None:
    speeds = [module.speed_mhz for module in hardware.memory.modules if module.speed_mhz]
    return min(speeds) if speeds else None


@dataclass(frozen=True)
class MachineFacts:
    physical_cores: int
    logical_cores: int
    memory_mhz: int | None
    memory_modules: int
    card: bool
    free_card_bytes: int | None


def machine_facts(hardware: HardwareProfile, free_card_bytes: int | None) -> MachineFacts:
    physical = max(int(hardware.cpu.physical_cores or 1), 1)
    logical = max(int(hardware.cpu.logical_cores or physical), physical)
    return MachineFacts(
        physical_cores=physical,
        logical_cores=logical,
        memory_mhz=memory_speed_mhz(hardware),
        memory_modules=len(hardware.memory.modules),
        card=card_usable(hardware),
        free_card_bytes=free_card_bytes,
    )


def share_words(share: float | None) -> str:
    if share is None:
        return "unknown (llama.cpp decides the layers)"
    if share <= 0.0:
        return "none: everything, output step included, is on the graphics card"
    if share >= 0.999:
        return "all of the work"
    return f"about {share:.0%} of the work per word (estimated from sizes)"


__all__ = [
    "MachineFacts",
    "ModelFacts",
    "card_usable",
    "facts_from_gguf",
    "machine_facts",
    "memory_speed_mhz",
    "processor_share",
    "read_model_facts",
    "share_words",
]
