"""The automatic context-safety check.

InferenceOS must never change, suggest changing, or indirectly change context size.
This module enforces that in two places:

* **on registration** (:func:`check_card_context_safe`): a card must state that it is
  context safe, and none of its program settings or parameters may be the context
  size;
* **on every applied change** (:func:`check_context_unchanged`): the context size in
  the configuration before a card's change must equal the context size after it. The
  executors call this before a change is accepted, so a change that would alter the
  context size is refused and never reaches the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from inferenceos.cards.base import Card
    from inferenceos.contracts.experiment import RuntimeConfiguration

#: Names that mean "context size" anywhere in a card.
CONTEXT_FIELDS = frozenset({"context_length", "ctx_size", "n_ctx", "context_size"})


class ContextSafetyError(ValueError):
    """A card or a change would touch context size."""


def check_card_context_safe(card: Card) -> None:
    """Refuse a card that is not provably context safe."""
    if not card.context_safe:
        raise ContextSafetyError(
            f"card {card.card_id}: every card must be context safe (context_safe=True)"
        )
    for binding in card.programs:
        if binding.setting in CONTEXT_FIELDS:
            raise ContextSafetyError(
                f"card {card.card_id}: program {binding.program!r} setting "
                f"{binding.setting!r} is the context size, which is never changed"
            )
    for spec in card.parameters:
        if spec.name in CONTEXT_FIELDS:
            raise ContextSafetyError(
                f"card {card.card_id}: parameter {spec.name!r} is the context size, "
                "which is never changed"
            )


def check_context_unchanged(
    before: RuntimeConfiguration, after: RuntimeConfiguration, *, card_id: str
) -> None:
    """Refuse a change whose result has a different context size."""
    if before.context_length != after.context_length:
        raise ContextSafetyError(
            f"card {card_id}: change would alter context size "
            f"({before.context_length} -> {after.context_length}); refused"
        )
