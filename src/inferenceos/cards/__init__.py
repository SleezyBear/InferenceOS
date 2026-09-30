"""Optimization cards: one description per kind of optimization.

See :mod:`inferenceos.cards.base` for the card format and
:mod:`inferenceos.cards.catalog` for how cards are found and checked.
"""

from inferenceos.cards.base import (
    LLAMA_CPP,
    ActsOn,
    ApprovalRule,
    Card,
    ProgramSetting,
    When,
)
from inferenceos.cards.catalog import (
    CARD_CATALOG_VERSION,
    CardCatalog,
    CardError,
    default_catalog,
    discover_builtin_cards,
    get_card,
    register_card,
    unregister_card,
    validate_card,
)
from inferenceos.cards.safety import (
    CONTEXT_FIELDS,
    ContextSafetyError,
    check_card_context_safe,
    check_context_unchanged,
)

__all__ = [
    "ActsOn",
    "ApprovalRule",
    "CARD_CATALOG_VERSION",
    "CONTEXT_FIELDS",
    "Card",
    "CardCatalog",
    "CardError",
    "ContextSafetyError",
    "LLAMA_CPP",
    "ProgramSetting",
    "When",
    "check_card_context_safe",
    "check_context_unchanged",
    "default_catalog",
    "discover_builtin_cards",
    "get_card",
    "register_card",
    "unregister_card",
    "validate_card",
]
