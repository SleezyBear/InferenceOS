"""The path declaration type shared by tune and the optimization cards (C3, 9 Oct 2026).

A :class:`PathDeclaration` says what one of tune's optimization paths is, reads, writes,
needs and invalidates. It is plain data: it imports nothing from InferenceOS, so the card
format (``cards/base.Card.tune``) and tune (``tune/declarations.py``, which holds the eight
declarations) can both use it without either depending on search, pipeline, model-loading
or execution code.

:func:`validate_declarations` checks a set of declarations before anything reads them:
unique names, card ids, practical categories and slots; settings that are real runtime
setting names; known questions, quality levels and capabilities; skip reasons only for
declared capabilities; and invalidations that name declared paths. Descriptive machine
and model facts (``facts_read``) are kept apart from runtime setting names
(``settings_read``) so that only the latter are checked against the runtime settings.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

SHORT, LONG = "short", "long"
QUESTIONS = (SHORT, LONG)
QUALITY_LEVELS = ("runtime", "engine", "precision")

#: Capabilities a path can require. Only those the practical search needs are checked
#: today; an unresolved capability is never taken as proof that a path is supported.
CAPABILITIES: dict[str, str] = {
    "graphics_card": "a graphics card is used",
    "engine_option": "the llama-server build has the path's option",
    "model_facts": "layer counts, kinds and sizes read from the model file",
    "long_question": "a long test question sized to the model",
    "engine_switch_list": "switches listed for this engine build and hardware",
    "known_answer_questions": "known-answer questions for the precision quality level",
}

_CARD_ID = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*)+$")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
#: A symbolic handler id: lower-case words joined by dots, e.g. "threads.processor_work".
HANDLER_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")


class DeclarationError(ValueError):
    """A path declaration, or the set of them, is invalid."""


@dataclass(frozen=True)
class PracticalUse:
    """How the practical search uses a path (only placement, threads and batch)."""

    category: str
    slot: int
    #: Shown in the practical search's log when a required capability is missing.
    skip_reasons: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PathDeclaration:
    name: str
    label: str
    #: The card id tune records a change under (and the quality policy's key).
    card_id: str
    #: The built-in card file for that id, if one exists (``cards/builtin``); None when
    #: tune's path has no card file.
    card_file: str | None
    #: Runtime settings the path changes (its own setting first, then any setting it
    #: changes together with it). Runtime setting names only.
    settings_written: tuple[str, ...]
    #: Runtime settings its candidates or applicability depend on. Runtime setting
    #: names only.
    settings_read: tuple[str, ...]
    #: Machine and model facts its candidates or applicability depend on (plain words,
    #: not runtime settings).
    facts_read: tuple[str, ...]
    #: llama-server option it needs ("" = none checked).
    engine_option: str
    requires: tuple[str, ...]
    #: Question a gain is judged on, and the question it must not lose on.
    judged_on: str
    guarded_on: str | None
    #: Quality level (``tune/quality_risk``): runtime, engine or precision; and whether
    #: the level is decided by the values set rather than the card.
    quality: str
    quality_by_value: bool
    #: How a measured setup is shown to have really run as requested. A description:
    #: the path's own verification code does the checking.
    effect_verification: str
    #: Paths to look at again after a change on this path is kept (Unbounded revisits).
    invalidates: tuple[str, ...]
    #: A kept change frees or uses card memory: the layer count is searched again.
    memory_shaping: bool
    #: Where the specialised search lives (not replaced by the declaration).
    search_handler: str
    in_unbounded: bool
    #: Symbolic id of the path's applicability rule ("does it apply here, and why"),
    #: resolved only through the explicit runtime registry in ``tune/applicability.py``
    #: (never evaluated or imported by name). None for a path whose specialised search
    #: decides applicability itself (the layer count).
    applicability: str | None
    practical: PracticalUse | None = None


def validate_declaration(d: PathDeclaration, runtime_settings: Collection[str]) -> list[str]:
    """Problems with one declaration (empty: valid)."""
    problems: list[str] = []
    where = f"path {d.name!r}"
    if not _NAME.match(d.name):
        problems.append(f"{where}: invalid path name")
    if not d.label:
        problems.append(f"{where}: no label")
    if not _CARD_ID.match(d.card_id):
        problems.append(f"{where}: invalid card id {d.card_id!r}")
    if d.card_file is not None and not re.match(r"^[a-z][a-z0-9_]*\.py$", d.card_file):
        problems.append(f"{where}: invalid card file {d.card_file!r}")
    if not d.settings_written:
        problems.append(f"{where}: writes no setting")
    for kind, names in (("settings_written", d.settings_written), ("settings_read", d.settings_read)):
        if len(set(names)) != len(names):
            problems.append(f"{where}: {kind} repeats a setting")
        for name in names:
            if name not in runtime_settings:
                problems.append(f"{where}: {kind} names {name!r}, which is not a runtime setting")
    for fact in d.facts_read:
        if fact in runtime_settings:
            problems.append(f"{where}: facts_read names the runtime setting {fact!r} (it belongs in settings_read)")
    for capability in d.requires:
        if capability not in CAPABILITIES:
            problems.append(f"{where}: unknown capability {capability!r}")
    if d.judged_on not in QUESTIONS:
        problems.append(f"{where}: unknown question {d.judged_on!r}")
    if d.guarded_on is not None and d.guarded_on not in QUESTIONS:
        problems.append(f"{where}: unknown guard question {d.guarded_on!r}")
    if d.guarded_on is not None and d.guarded_on == d.judged_on:
        problems.append(f"{where}: guarded on the question it is judged on")
    if d.quality not in QUALITY_LEVELS:
        problems.append(f"{where}: unknown quality level {d.quality!r}")
    if d.name in d.invalidates:
        problems.append(f"{where}: invalidates itself")
    if len(set(d.invalidates)) != len(d.invalidates):
        problems.append(f"{where}: invalidates a path twice")
    if not d.search_handler:
        problems.append(f"{where}: no search handler")
    if d.applicability is not None and (not isinstance(d.applicability, str)
                                        or not HANDLER_ID.match(d.applicability)):
        problems.append(f"{where}: invalid applicability handler id {d.applicability!r}")
    if d.practical is not None:
        slot = d.practical.slot
        if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
            problems.append(f"{where}: practical slot must be a whole number 0 or more, not {slot!r}")
        if not isinstance(d.practical.category, str) or not _NAME.match(d.practical.category):
            problems.append(f"{where}: invalid practical category {d.practical.category!r}")
        for capability in d.practical.skip_reasons:
            if capability not in d.requires:
                problems.append(f"{where}: skip reason for {capability!r}, which the path does not require")
    return problems


def validate_declarations(declarations: Iterable[PathDeclaration], runtime_settings: Collection[str]) -> None:
    """Raise :class:`DeclarationError` listing every problem in ``declarations``."""
    items = list(declarations)
    problems: list[str] = []
    for d in items:
        if not isinstance(d, PathDeclaration):
            raise DeclarationError(f"not a path declaration: {d!r}")
        problems += validate_declaration(d, runtime_settings)
    names = [d.name for d in items]
    columns: tuple[tuple[str, list[str]], ...] = (
        ("path name", names), ("card id", [d.card_id for d in items]),
        ("card file", [d.card_file for d in items if d.card_file is not None]),
        ("applicability handler", [d.applicability for d in items if isinstance(d.applicability, str)]),
        ("practical category", [d.practical.category for d in items if d.practical]),
        ("practical slot", [str(d.practical.slot) for d in items if d.practical]))
    for label, values in columns:
        repeated = sorted({v for v in values if values.count(v) > 1})
        if repeated:
            problems.append(f"duplicate {label}: {', '.join(repeated)}")
    slots = sorted(d.practical.slot for d in items if d.practical is not None
                   and isinstance(d.practical.slot, int) and not isinstance(d.practical.slot, bool))
    if slots and len(set(slots)) == len(slots) and slots != list(range(len(slots))):
        problems.append(f"practical slots must be 0..{len(slots) - 1} without gaps, not {slots}")
    declared = set(names)
    for d in items:
        for target in d.invalidates:
            if target not in declared:
                problems.append(f"path {d.name!r}: invalidates {target!r}, which is not a declared path")
    if problems:
        raise DeclarationError("; ".join(problems))


__all__ = [
    "CAPABILITIES",
    "HANDLER_ID",
    "LONG",
    "QUALITY_LEVELS",
    "QUESTIONS",
    "SHORT",
    "DeclarationError",
    "PathDeclaration",
    "PracticalUse",
    "validate_declaration",
    "validate_declarations",
]
