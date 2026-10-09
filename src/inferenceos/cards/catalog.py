"""The card catalog: every card InferenceOS knows, found automatically.

Built-in cards live one per file in :mod:`inferenceos.cards.builtin`. Each file
exposes a module-level ``CARD`` (or ``CARDS`` for several). The default catalog finds
them by scanning that package, so adding a card file is enough: nothing else has to
be edited. Extra cards (for example a test card, or later a user's own card) can be
added at run time with :func:`register_card`.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import threading
from types import ModuleType

from inferenceos.cards.base import Card
from inferenceos.cards.safety import check_card_context_safe
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.path_declaration import PathDeclaration, validate_declaration

_CARD_ID_RE = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*)*$")

#: Changes whenever the built-in card set or the card format changes.
CARD_CATALOG_VERSION = "cards.v1"


class CardError(ValueError):
    """A card definition is invalid or duplicated."""


def validate_card(card: Card) -> None:
    """Check one card before it is accepted (the context check is part of this)."""
    if not _CARD_ID_RE.match(card.card_id):
        raise CardError(f"invalid card id: {card.card_id!r}")
    if not card.name or not card.what_it_changes:
        raise CardError(f"{card.card_id}: a card needs a name and 'what it changes'")
    if not card.answers:
        raise CardError(f"{card.card_id}: a card must say which problems it answers")
    if not card.undo:
        raise CardError(f"{card.card_id}: a card must say how it is undone")
    check_card_context_safe(card)
    _check_tune_link(card)
    known = set(RuntimeConfiguration.model_fields)
    for binding in card.programs:
        if binding.setting not in known:
            raise CardError(
                f"{card.card_id}: program {binding.program!r} setting "
                f"{binding.setting!r} is not a runtime setting InferenceOS can change"
            )
    if not card.suggest_only and not card.programs:
        raise CardError(
            f"{card.card_id}: a card that is applied needs at least one program setting "
            "(suggest-only cards need none)"
        )
    if not card.suggest_only and len(card.parameters) != 1:
        raise CardError(
            f"{card.card_id}: an applied card changes exactly one setting, so it needs "
            "exactly one parameter"
        )
    # The engine's own safety rules must accept the derived definition too.
    from inferenceos.optimization.registry import ActionRegistryError, validate_action

    try:
        validate_action(card.to_action())
    except ActionRegistryError as exc:
        raise CardError(str(exc)) from exc


def _check_tune_link(card: Card) -> None:
    """A card's optional ``tune`` field must be tune's canonical declaration for the same
    card id (``tune/declarations.py``): it is checked structurally, then against the
    registry, so a card can never carry a second, conflicting description of a path.
    Every problem is reported as a :class:`CardError`. Cards without it are unaffected."""
    link = card.tune
    if link is None:
        return
    if not isinstance(link, PathDeclaration):
        raise CardError(f"{card.card_id}: 'tune' must be a path declaration, not {type(link).__name__}")
    try:
        from inferenceos.tune.declarations import RUNTIME_SETTINGS, by_card

        problems = validate_declaration(link, RUNTIME_SETTINGS)
        if problems:
            raise CardError(f"{card.card_id}: invalid path declaration: {'; '.join(problems)}")
        canonical = by_card(card.card_id)
        if link.card_id != card.card_id:
            raise CardError(f"{card.card_id}: linked to the declaration of {link.card_id!r} (path {link.name!r})")
        if canonical is None:
            raise CardError(f"{card.card_id}: no path declaration exists for this card id")
        if link != canonical:
            raise CardError(f"{card.card_id}: its path declaration conflicts with the canonical declaration of "
                            f"path {canonical.name!r} in tune/declarations.py")
        if link.card_file is None:
            raise CardError(f"{card.card_id}: path {link.name!r} declares no card file but this card links to it")
        if card.setting_for() != link.settings_written[0]:
            raise CardError(f"{card.card_id}: changes {card.setting_for()!r} but path {link.name!r} "
                            f"declares {link.settings_written[0]!r} as its own setting")
    except CardError:
        raise
    except Exception as exc:  # noqa: BLE001 - any malformed link is a card error, never a crash
        raise CardError(f"{card.card_id}: unusable path declaration ({type(exc).__name__}: {exc})") from exc


class CardCatalog:
    """A validated, ordered set of cards."""

    def __init__(self, cards: list[Card] | None = None) -> None:
        self._cards: dict[str, Card] = {}
        for card in cards or []:
            self.register(card)

    def register(self, card: Card) -> Card:
        validate_card(card)
        if card.card_id in self._cards:
            raise CardError(f"duplicate card id: {card.card_id}")
        self._cards[card.card_id] = card
        return card

    def unregister(self, card_id: str) -> None:
        self._cards.pop(card_id, None)

    def __contains__(self, card_id: object) -> bool:
        return isinstance(card_id, str) and card_id in self._cards

    def __len__(self) -> int:
        return len(self._cards)

    def get(self, card_id: str) -> Card:
        try:
            return self._cards[card_id]
        except KeyError as exc:
            raise CardError(f"unknown card: {card_id}") from exc

    def get_optional(self, card_id: str) -> Card | None:
        return self._cards.get(card_id)

    def all(self) -> list[Card]:
        return [self._cards[key] for key in sorted(self._cards)]

    def ids(self) -> list[str]:
        return sorted(self._cards)

    def applied_cards(self) -> list[Card]:
        """Cards InferenceOS can actually apply and undo (not suggest-only)."""
        return [card for card in self.all() if not card.suggest_only and card.programs]


def _module_cards(module: ModuleType) -> list[Card]:
    found: list[Card] = []
    single = getattr(module, "CARD", None)
    if isinstance(single, Card):
        found.append(single)
    many = getattr(module, "CARDS", None)
    if isinstance(many, (list, tuple)):
        found.extend(card for card in many if isinstance(card, Card))
    return found


#: The package holding the built-in card files.
BUILTIN_CARD_PACKAGE = "inferenceos.cards.builtin"


def discover_builtin_cards(package_name: str = BUILTIN_CARD_PACKAGE) -> list[Card]:
    """Every card defined in the files of ``package_name``, in file-name order."""
    package = importlib.import_module(package_name)
    cards: list[Card] = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        found = _module_cards(module)
        for card in found:
            if card.tune is not None and card.tune.card_file != f"{info.name}.py":
                raise CardError(f"{card.card_id}: defined in {info.name}.py but its path declaration "
                                f"names {card.tune.card_file!r}")
        cards.extend(found)
    return cards


_lock = threading.Lock()
_default: CardCatalog | None = None


def default_catalog() -> CardCatalog:
    """The process-wide catalog: the built-in cards plus any registered extras."""
    global _default
    with _lock:
        if _default is None:
            _default = CardCatalog(discover_builtin_cards())
        return _default


def register_card(card: Card) -> Card:
    """Add a card to the default catalog (it is then used everywhere)."""
    return default_catalog().register(card)


def unregister_card(card_id: str) -> None:
    """Remove a previously registered extra card from the default catalog."""
    default_catalog().unregister(card_id)


def get_card(card_id: str) -> Card | None:
    """Look up a card in the default catalog (``None`` if there is no such card)."""
    return default_catalog().get_optional(card_id)
