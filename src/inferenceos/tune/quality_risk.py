"""How much proof of unchanged answers each kind of change needs.

Levels, strictest last:

* ``runtime`` (0): the same arithmetic, scheduled differently (threads, batch sizes,
  layer count). The normal answer check on the tuning questions.
* ``engine`` (1): may run different code for the same arithmetic (engine switches,
  flash attention, piece placement on another device); the stricter answer band
  (``experiment/quality_policy``: a change without a card, or a card marked "quality
  risk", is checked with the tightened band).
* ``precision`` (2): stores numbers at lower precision (the conversation memory
  format). The stricter band **and** the known-answer questions
  (``tune/known_answers.py``) on both setups.

A proposal that combines several changes is checked with the strictest requirement of
any change in it (review handoff, 6 Oct 2026: the final check used to take the first
change's card, so a cache change listed second would have been checked like a thread
count).

Review 7: the requirement of one change is derived from **every setting it changes**
(its own setting and the settings changed with it), not from its card alone. A layer
change merged with an earlier cache-format change still carries the cache format, so it
still needs the known-answer questions, whatever card the merged change ended up with.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inferenceos.contracts.tune import ProposedChange
from inferenceos.tune.known_answers import POLICY_VERSION as KNOWN_ANSWERS_VERSION

#: Version of this classification (changes whenever a level or rule changes).
QUALITY_POLICY_VERSION = "quality-risk-v2"


@dataclass(frozen=True)
class QualityRequirement:
    level: int
    name: str
    known_answers: bool
    why: str

    def record(self) -> dict[str, object]:
        return {
            "policy": QUALITY_POLICY_VERSION,
            "level": self.name,
            "known_answers": KNOWN_ANSWERS_VERSION if self.known_answers else None,
            "why": self.why,
        }


RUNTIME = QualityRequirement(0, "runtime", False, "same arithmetic, scheduled differently")
ENGINE = QualityRequirement(1, "engine", False, "may run different code for the same arithmetic")
PRECISION = QualityRequirement(2, "precision", True, "stores numbers at lower precision")

#: Card id -> requirement, for the changes tune makes.
REQUIREMENTS: dict[str, QualityRequirement] = {
    "runtime.reduce_threads": RUNTIME,
    "runtime.reading_threads": RUNTIME,
    "runtime.prompt_batch": RUNTIME,
    "runtime.increase_gpu_offload_layers": RUNTIME,
    "runtime.flash_attention": ENGINE,
    "engine.switches": ENGINE,
    "runtime.tensor_placement": ENGINE,
    "runtime.kv_cache_format": PRECISION,
}


#: Setting -> (card whose answer check fits it, requirement). Settings not listed are
#: scheduling only (runtime level) when in ``RUNTIME_SETTINGS``, else the engine level.
_HIGH_PRECISION = {None, "", "f16", "f32", "bf16"}
RUNTIME_SETTINGS = frozenset(
    {
        "gpu_offload_layers",
        "threads",
        "threads_batch",
        "batch_size",
        "ubatch_size",
        "use_mmap",
        "context_length",
    }
)
SETTING_CARDS: dict[str, str] = {
    "kv_cache_type": "runtime.kv_cache_format",
    "kv_cache_type_k": "runtime.kv_cache_format",
    "kv_cache_type_v": "runtime.kv_cache_format",
    "placement_overrides": "runtime.tensor_placement",
    "flash_attention": "runtime.flash_attention",
    "engine_switches": "engine.switches",
}


#: Cards whose requirement comes from the values they set, not the card itself.
VALUE_DECIDED_CARDS = frozenset({"runtime.kv_cache_format"})


def requirement_for(card_id: str) -> QualityRequirement:
    """The requirement for one card; an unknown card gets the stricter engine level."""
    return REQUIREMENTS.get(card_id, ENGINE)


def requirement_for_setting(setting: str, value: object) -> tuple[str | None, QualityRequirement]:
    """(card, requirement) for setting one value. A conversation-memory format below
    16-bit is precision level; going back to 16-bit is engine level."""
    if setting in ("kv_cache_type", "kv_cache_type_k", "kv_cache_type_v"):
        level = ENGINE if value in _HIGH_PRECISION else PRECISION
        return SETTING_CARDS[setting], level
    if setting in SETTING_CARDS:
        return SETTING_CARDS[setting], ENGINE
    if setting in RUNTIME_SETTINGS:
        return None, RUNTIME
    return None, ENGINE


def requirement_of_change(change: ProposedChange) -> tuple[str, QualityRequirement]:
    """The strictest of the change's card and every setting it changes: (card, level).

    The card returned is the one whose answer check fits the strictest part (the
    change's own card when that is already the strictest)."""
    card = change.card_id
    # The cache-format card is decided by the values it sets (back to 16-bit is not a
    # precision change); every other card sets a floor.
    best = RUNTIME if card in VALUE_DECIDED_CARDS else requirement_for(card)
    settings = [(change.setting, change.new_value), *change.related.items()]
    for setting, value in settings:
        found_card, found = requirement_for_setting(setting, value)
        if found.level > best.level:
            card, best = found_card or card, found
    return card, best


def strictest(changes: Sequence[ProposedChange]) -> tuple[str | None, QualityRequirement]:
    """The card id to check a combined proposal with, and its requirement: the strictest
    of all changes and of every setting they change (ties keep the first such change).
    ``(None, RUNTIME)`` for none."""
    best_card: str | None = None
    best = RUNTIME
    for change in changes:
        card, found = requirement_of_change(change)
        if best_card is None or found.level > best.level:
            best_card, best = card, found
    return best_card, best


__all__ = [
    "ENGINE",
    "PRECISION",
    "QUALITY_POLICY_VERSION",
    "QualityRequirement",
    "REQUIREMENTS",
    "RUNTIME",
    "SETTING_CARDS",
    "requirement_for",
    "requirement_for_setting",
    "requirement_of_change",
    "strictest",
]
