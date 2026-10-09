"""Applicability: does a tune path apply here, and why (C3, 9 Oct 2026).

Each optimization path's declaration (``tune/declarations.py``) names its applicability
rule by a symbolic handler id. This module is the explicit runtime registry that turns
those ids into the existing, specialised rules, which stay in ``tune/paths.py`` and are
registered there with :func:`handler`. Nothing is evaluated or imported by name: an id
that is not registered is an error, never a guess.

:func:`check` is the one place the decision is made, with the semantics ``OptionPath.check``
always had:

1. a known missing engine option (the path declares one, the engine's option list is
   known, and the option is not in it) rejects the path with the existing reason, and the
   path's own rule is never called;
2. otherwise (the option is present, or the option list is unknown) the path's own rule
   decides, with its own reason. Unknown option support is not a rejection and not proof
   of support; it is left to the rule and to the engine's later checks, as before.

Registering a handler activates nothing: which paths run, in which order and in which
mode is decided elsewhere, exactly as before.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from inferenceos.contracts.path_declaration import HANDLER_ID, PathDeclaration

if TYPE_CHECKING:  # pragma: no cover - typing only (paths.py imports this module)
    from inferenceos.tune.paths import PathContext

Handler = Callable[["PathContext"], tuple[bool, str]]

_HANDLERS: dict[str, Handler] = {}


class ApplicabilityError(LookupError):
    """A handler id is invalid, registered twice, or not registered."""


def handler(handler_id: str) -> Callable[[Handler], Handler]:
    """Register ``fn`` as the applicability rule named ``handler_id`` (returns it unchanged)."""
    if not HANDLER_ID.match(handler_id):
        raise ApplicabilityError(f"invalid applicability handler id {handler_id!r}")

    def register(fn: Handler) -> Handler:
        existing = _HANDLERS.get(handler_id)
        if existing is not None and existing is not fn:
            raise ApplicabilityError(f"applicability handler {handler_id!r} is already registered")
        _HANDLERS[handler_id] = fn
        return fn

    return register


def resolve(handler_id: str) -> Handler:
    """The registered rule for ``handler_id``."""
    try:
        return _HANDLERS[handler_id]
    except KeyError:
        raise ApplicabilityError(f"no applicability handler is registered as {handler_id!r}") from None


def registered() -> frozenset[str]:
    return frozenset(_HANDLERS)


def missing_option(engine_option: str, context: PathContext) -> str | None:
    """The rejection reason when the engine is known not to have ``engine_option``."""
    if engine_option and context.supported_options is not None and engine_option not in context.supported_options:
        return f"this llama.cpp does not have the {engine_option} option"
    return None


def check(declaration: PathDeclaration, context: PathContext) -> tuple[bool, str]:
    """Whether the declared path applies in ``context``, and why (see the module text)."""
    if declaration.applicability is None:
        raise ApplicabilityError(f"path {declaration.name!r} decides applicability in its own search")
    rule = resolve(declaration.applicability)
    reason = missing_option(declaration.engine_option, context)
    if reason is not None:
        return False, reason
    return rule(context)


__all__ = ["ApplicabilityError", "Handler", "check", "handler", "missing_option", "registered", "resolve"]
