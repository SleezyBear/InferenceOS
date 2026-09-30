"""The action registry, built from the optimization cards.

Every registered action is derived from one card in :mod:`inferenceos.cards`; this
module no longer holds its own list. Adding a card file is enough for its action to
appear here.
"""

from __future__ import annotations

from inferenceos.cards.catalog import CARD_CATALOG_VERSION, default_catalog
from inferenceos.optimization.registry import ActionRegistry

ACTION_REGISTRY_VERSION = f"actions-from-{CARD_CATALOG_VERSION}"


def build_default_registry() -> ActionRegistry:
    """A fresh registry holding the engine view of every card in the catalog."""
    registry = ActionRegistry()
    for card in default_catalog().all():
        registry.register(card.to_action())
    return registry
