"""Conversation memory (KV cache) formats: what to try, what fits, what took effect.

The engine keeps, for every attention block, the keys and values of every word seen so
far. Storing them at lower precision (``q8_0``: 8-bit numbers in blocks of 32 with one
16-bit scale, 34 bytes per 32 numbers, against 64 bytes at ``f16``) roughly halves that
memory, which can let more of the model fit on the card. It can also change answers,
so it is a quality-risk change with its own acceptance policy
(``tune/known_answers.py``).

What is never changed: the model's weights and tokens, and the context size the user
chose (a smaller cache never means a shorter conversation window). Only attention
blocks have this memory: the state of recurrent (state-space, convolution) blocks is a
different kind of memory and its format is not touched.

Version 1 (``CACHE_POLICY_VERSION``) tries only ``q8_0``, for the keys, the values or
both. Lower precisions (``q4_0``, ``q5_*``, ``iq4_nl``) are left out until a stricter
quality policy for them exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.tune.model_facts import ModelFacts

SHOWN, NOT_SEEN, UNVERIFIED = "shown", "not seen", "unverified"

CACHE_POLICY_VERSION = "kv-cache-v1"
#: The engine's default format for both sides.
DEFAULT_FORMAT = "f16"
#: Formats version 1 may try.
TRIED_FORMATS = ("q8_0",)

#: Bytes per number, including the block scales (llama.cpp ggml block layouts).
FORMAT_BYTES: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}
#: Numbers per block of the block formats (a cached row must be a whole number of blocks).
BLOCK_SIZE: dict[str, int] = {name: 32 for name in ("q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl")}


def is_quantized(name: str | None) -> bool:
    return name in BLOCK_SIZE


@dataclass(frozen=True)
class CacheEstimate:
    """Estimated conversation memory for a context size and formats."""

    bytes: int | None
    #: "from the model's piece shapes", "upper bound (sliding window)", "estimate ..." or why unknown.
    basis: str
    exact_layout: bool


def cache_bytes(
    facts: ModelFacts | None,
    model: ModelProfile,
    context_length: int,
    key_format: str | None,
    value_format: str | None,
) -> CacheEstimate:
    """Conversation memory for ``context_length`` with the given formats.

    From the widths of each attention block's key and value pieces when known (only
    attention blocks counted); a model with a sliding window gets an upper bound. Without
    those widths, the model profile's per-block figures for every block (an estimate,
    over-counting non-attention blocks). Unknown formats give ``None``.
    """
    k_name = key_format or DEFAULT_FORMAT
    v_name = value_format or DEFAULT_FORMAT
    if k_name not in FORMAT_BYTES or v_name not in FORMAT_BYTES:
        return CacheEstimate(None, f"unknown format {k_name}/{v_name}", False)
    k_bytes, v_bytes = FORMAT_BYTES[k_name], FORMAT_BYTES[v_name]
    if facts is not None and facts.key_widths and facts.value_widths:
        if all(width is not None for width in facts.key_widths + facts.value_widths):
            per_word = sum(
                int(k or 0) * k_bytes + int(v or 0) * v_bytes
                for k, v in zip(facts.key_widths, facts.value_widths)
            )
            basis = "from the model's piece shapes (attention blocks only)"
            exact = True
            if facts.sliding_window:
                basis = "upper bound: some blocks use a sliding window and keep less"
                exact = False
            return CacheEstimate(int(per_word * context_length), basis, exact)
    key_length = model.key_length or model.head_dim
    value_length = model.value_length or model.head_dim
    per_word = model.n_layers * model.n_kv_heads * (key_length * k_bytes + value_length * v_bytes)
    return CacheEstimate(
        int(per_word * context_length),
        "estimate from the model's head counts for every block",
        False,
    )


@dataclass(frozen=True)
class CacheChoice:
    keys: str
    values: str
    #: Settings that must change with it (e.g. flash attention for a quantized V).
    also: tuple[tuple[str, object], ...]
    label: str


def _row_ok(widths: tuple[int | None, ...] | None, name: str) -> bool | None:
    if not is_quantized(name):
        return True
    if not widths:
        return None
    block = BLOCK_SIZE[name]
    known = [width for width in widths if width]
    if not known:
        return None
    return all(width % block == 0 for width in known)


def cache_choices(
    facts: ModelFacts | None,
    current: RuntimeConfiguration,
    *,
    accepts: object = None,
) -> tuple[list[CacheChoice], list[str]]:
    """The format pairs worth trying from the current setup, and notes on what was
    left out. ``accepts(setting, value) -> (Support, reason)`` is the engine's own
    "does it accept this value" (None = not known; nothing is ruled out for it)."""
    from inferenceos.contracts.capability import Support

    notes: list[str] = []
    keys_now, values_now = current.effective_kv_types()
    keys_now = keys_now or DEFAULT_FORMAT
    values_now = values_now or DEFAULT_FORMAT
    if facts is not None and facts.attention_layers == 0:
        return [], ["this model has no attention blocks, so it has no conversation memory of this kind"]
    wanted: list[tuple[str, str]] = []
    for name in TRIED_FORMATS:
        wanted += [(name, name), (name, values_now), (keys_now, name)]
    if is_quantized(keys_now) or is_quantized(values_now):
        wanted.append((DEFAULT_FORMAT, DEFAULT_FORMAT))
    choices: list[CacheChoice] = []
    seen: set[tuple[str, str]] = set()
    for keys, values in wanted:
        if (keys, values) == (keys_now, values_now) or (keys, values) in seen:
            continue
        seen.add((keys, values))
        if callable(accepts):
            refused = [
                reason
                for setting, value in (("kv_cache_type_k", keys), ("kv_cache_type_v", values))
                for status, reason in [accepts(setting, value)]
                if status is Support.NO
            ]
            if refused:
                notes.append(f"{keys}/{values} not tried: " + "; ".join(refused))
                continue
        row_k = _row_ok(facts.key_widths if facts else None, keys)
        row_v = _row_ok(facts.value_widths if facts else None, values)
        if row_k is False or row_v is False:
            notes.append(
                f"{keys}/{values} not tried: a cached row is not a whole number of "
                "32-number blocks for this model"
            )
            continue
        also: list[tuple[str, object]] = []
        if is_quantized(values):
            if current.flash_attention is False:
                notes.append(
                    f"{keys}/{values} not tried: a lower-precision value cache needs flash "
                    "attention, which this setup turns off"
                )
                continue
            if current.flash_attention is None:
                also.append(("flash_attention", True))
        label = f"conversation memory keys {keys}, values {values}"
        if also:
            label += " (with flash attention on, which it needs)"
        choices.append(CacheChoice(keys, values, tuple(also), label))
    if any(name not in TRIED_FORMATS + (DEFAULT_FORMAT,) for name in FORMAT_BYTES):
        notes.append(
            "lower precisions (q4/q5/iq4) are not tried in "
            f"{CACHE_POLICY_VERSION}: they need a stricter answer-quality policy first"
        )
    return choices, notes


def verify_cache(config: RuntimeConfiguration, observed: dict[str, object] | None) -> tuple[str, str]:
    """Did the engine use the requested formats? From its own memory line."""
    keys, values = config.effective_kv_types()
    keys, values = keys or DEFAULT_FORMAT, values or DEFAULT_FORMAT
    if not observed or "cache_type_k" not in observed or "cache_type_v" not in observed:
        return UNVERIFIED, "the engine did not print its conversation-memory formats in this run"
    seen_k, seen_v = observed["cache_type_k"], observed["cache_type_v"]
    words = f"engine reports keys {seen_k}, values {seen_v}; requested {keys}/{values}"
    if seen_k == keys and seen_v == values:
        return SHOWN, words
    return NOT_SEEN, words + " (not the requested formats)"


def observed_cache_bytes(observed: dict[str, object] | None) -> int | None:
    if not observed:
        return None
    buffers = observed.get("kv_buffers")
    if not isinstance(buffers, dict) or not buffers:
        return None
    return sum(int(value) for value in buffers.values())


__all__ = [
    "BLOCK_SIZE",
    "CACHE_POLICY_VERSION",
    "CacheChoice",
    "CacheEstimate",
    "DEFAULT_FORMAT",
    "FORMAT_BYTES",
    "NOT_SEEN",
    "SHOWN",
    "TRIED_FORMATS",
    "UNVERIFIED",
    "cache_bytes",
    "cache_choices",
    "is_quantized",
    "observed_cache_bytes",
    "verify_cache",
]
