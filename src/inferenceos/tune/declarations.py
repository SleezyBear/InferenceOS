"""Path declarations: what every tune path is, reads, writes, needs and invalidates
(C3, 9 Oct 2026).

Each of tune's eight optimization paths has one stable, inspectable declaration here,
keyed by the card id tune already uses for it. The declarations are data (the type is
``contracts/path_declaration.PathDeclaration``); the specialised search code stays where
it is (``tune/layer_search.py``, ``tune/layer_top.py``, the ``OptionPath`` objects in
``tune/paths.py``, ``tune/planner.py``). A declaration says *what* a path is, never *how*
it searches, and its effect-verification text describes checks that the path's own code
performs.

The declarations are validated when this module is imported (``validate_declarations``):
nothing reads an invalid set.

Who reads them:

* the practical search, through :func:`practical_tables` (the adapter, first slice): its
  categories, settings, slots, questions, cards, evidence scope, final-check questions
  and the batch skip reason;
* Unbounded's static tables (second slice): ``tune/diagnose.AFFECTS`` and
  ``MEMORY_PATHS`` and ``tune/quality_risk.REQUIREMENTS`` are projections of the
  declarations (:func:`affects_table`, :func:`memory_shaping_paths`,
  :func:`quality_by_card`), with their contents, tuple order and fallbacks unchanged.
  Unbounded's search order, its applicability checks (``OptionPath.check`` / ``applies``)
  and its value-sensitive quality rules (``requirement_for_setting``,
  ``VALUE_DECIDED_CARDS``, merged-change analysis) are not replaced;
* the two legacy cards tune shares (layers and threads), whose optional ``tune`` field is
  the declaration itself, so there is one copy of these facts;
* tests, which check the declarations against ``OptionPath`` and the card catalog.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.path_declaration import (
    CAPABILITIES,
    LONG,
    SHORT,
    DeclarationError,
    PathDeclaration,
    PracticalUse,
    validate_declarations,
)

#: Version of the declaration schema and content (bump on any change). v2 (9 Oct 2026,
#: second C3 slice): the type moved to ``contracts/path_declaration.py``, descriptive
#: facts split from runtime settings read, declarations validated on import, and
#: Unbounded's static tables derived from them. v3 (9 Oct 2026, late): each OptionPath-backed
#: path names its applicability rule by a symbolic handler id (``tune/applicability.py``).
#: v4 (18 Oct 2026): layers invalidate engine switches; effect descriptions
#: distinguish observed contradiction checks from settings not yet observed.
DECLARATIONS_VERSION = "path-declarations-v4"

#: Runtime setting names a declaration may write or read.
RUNTIME_SETTINGS = frozenset(RuntimeConfiguration.model_fields)


DECLARATIONS: tuple[PathDeclaration, ...] = (
    PathDeclaration(
        name="layers", label="graphics-card layers (placement count)",
        card_id="runtime.increase_gpu_offload_layers", card_file="gpu_layers_more.py",
        settings_written=("gpu_offload_layers",),
        settings_read=("kv_cache_type_k", "kv_cache_type_v", "placement_overrides", "ubatch_size"),
        facts_read=("free card memory", "model facts (layer counts, kinds and sizes)"),
        engine_option="--n-gpu-layers", requires=("model_facts",),
        judged_on=SHORT, guarded_on=None, quality="runtime", quality_by_value=False,
        effect_verification="known startup layer-count contradictions are rejected; missing observations remain unverified",
        invalidates=("threads", "reading_threads", "prompt_batch", "placement", "kv_cache", "engine_switches"),
        memory_shaping=False,
        search_handler="tune/layer_top.py (load-only top) + tune/layer_search.py (Unbounded); "
                       "tune/planner.py placement steps (practical)",
        in_unbounded=True,
        applicability=None,
        practical=PracticalUse(category="placement", slot=0),
    ),
    PathDeclaration(
        name="threads", label="processor threads", card_id="runtime.reduce_threads", card_file="threads.py",
        settings_written=("threads",),
        settings_read=("gpu_offload_layers", "ubatch_size"),
        facts_read=("physical and logical cores", "processor share"),
        engine_option="--threads", requires=(),
        judged_on=SHORT, guarded_on=None, quality="runtime", quality_by_value=False,
        effect_verification="known startup thread-count contradictions are rejected; missing observations remain unverified",
        invalidates=("reading_threads", "prompt_batch"), memory_shaping=False,
        search_handler="tune/paths.py THREADS (Unbounded); tune/planner.py threads steps (practical)",
        in_unbounded=True,
        applicability="threads.processor_work",
        practical=PracticalUse(category="threads", slot=1),
    ),
    PathDeclaration(
        name="reading_threads", label="reading threads", card_id="runtime.reading_threads", card_file=None,
        settings_written=("threads_batch",), settings_read=("gpu_offload_layers", "ubatch_size", "threads"),
        facts_read=(),
        engine_option="--threads-batch", requires=("long_question",),
        judged_on=LONG, guarded_on=SHORT, quality="runtime", quality_by_value=False,
        effect_verification="requested reading threads are passed to the engine; effective reading threads are not yet verified",
        invalidates=("threads", "prompt_batch"), memory_shaping=False,
        search_handler="tune/paths.py READING_THREADS", in_unbounded=True,
        applicability="reading_threads.processor_reading",
    ),
    PathDeclaration(
        name="prompt_batch", label="prompt batch size", card_id="runtime.prompt_batch", card_file=None,
        settings_written=("ubatch_size",),
        settings_read=("gpu_offload_layers", "threads"),
        facts_read=("free card memory", "long question length"),
        engine_option="--ubatch-size", requires=("long_question",),
        judged_on=LONG, guarded_on=SHORT, quality="runtime", quality_by_value=False,
        effect_verification="requested batch size is passed to the engine; effective batch size is not yet verified",
        invalidates=("threads", "reading_threads", "flash_attention", "engine_switches", "placement"),
        memory_shaping=False,
        search_handler="tune/paths.py PROMPT_BATCH (Unbounded); tune/planner.py batch steps (practical)",
        in_unbounded=True,
        applicability="prompt_batch.always",
        practical=PracticalUse(category="batch", slot=2, skip_reasons={"long_question": "no long test question"}),
    ),
    PathDeclaration(
        name="flash_attention", label="flash attention", card_id="runtime.flash_attention", card_file=None,
        settings_written=("flash_attention",), settings_read=("kv_cache_type_v", "ubatch_size"), facts_read=(),
        engine_option="--flash-attn", requires=("long_question",),
        judged_on=LONG, guarded_on=SHORT, quality="engine", quality_by_value=False,
        effect_verification="known observed flash-attention contradictions are rejected; missing observations remain unverified; "
                            "trade-offs offered, never kept on their own",
        invalidates=("prompt_batch", "threads", "reading_threads", "engine_switches", "placement", "kv_cache"),
        memory_shaping=False, search_handler="tune/paths.py FLASH_ATTENTION", in_unbounded=True,
        applicability="flash_attention.attention_layers",
    ),
    PathDeclaration(
        name="engine_switches", label="engine switches", card_id="engine.switches", card_file=None,
        settings_written=("engine_switches",), settings_read=("gpu_offload_layers",), facts_read=("engine build",),
        engine_option="", requires=("graphics_card", "engine_switch_list"),
        judged_on=SHORT, guarded_on=LONG, quality="engine", quality_by_value=False,
        effect_verification="proof of effect from what the engine prints (never timing) and word-for-word "
                            "identical answers, before any timed check",
        invalidates=("threads", "reading_threads", "prompt_batch", "flash_attention"), memory_shaping=False,
        search_handler="tune/engine_switches.py + tune/paths.py ENGINE_SWITCHES", in_unbounded=True,
        applicability="engine_switches.listed_for_build",
    ),
    PathDeclaration(
        name="placement", label="per-piece placement", card_id="runtime.tensor_placement", card_file=None,
        settings_written=("placement_overrides", "gpu_offload_layers"),
        settings_read=("gpu_offload_layers",),
        facts_read=("model facts (layer counts, kinds and sizes)", "free card memory"),
        engine_option="--override-tensor", requires=("graphics_card", "model_facts"),
        judged_on=SHORT, guarded_on=LONG, quality="engine", quality_by_value=False,
        effect_verification="per-piece lines from a separate debug-level load of each setup; measured runs "
                            "checked by buffer totals",
        invalidates=("threads", "reading_threads", "prompt_batch", "flash_attention", "engine_switches", "kv_cache"),
        memory_shaping=True, search_handler="tune/tensor_placement.py + tune/paths.py PLACEMENT", in_unbounded=True,
        applicability="placement.piece_facts",
    ),
    PathDeclaration(
        name="kv_cache", label="conversation memory format", card_id="runtime.kv_cache_format", card_file=None,
        settings_written=("kv_cache_type_k", "kv_cache_type_v", "flash_attention"),
        settings_read=("flash_attention",),
        facts_read=("model facts (layer counts, kinds and sizes)", "free card memory"),
        engine_option="--cache-type-k", requires=("long_question", "known_answer_questions"),
        judged_on=LONG, guarded_on=SHORT, quality="precision", quality_by_value=True,
        effect_verification="memory format applied, from the engine's output; known-answer questions on both "
                            "setups when the format is below 16-bit",
        invalidates=("placement", "prompt_batch", "flash_attention", "threads"),
        memory_shaping=True, search_handler="tune/kv_cache.py + tune/paths.py KV_CACHE", in_unbounded=True,
        applicability="kv_cache.attention_blocks",
    ),
)

validate_declarations(DECLARATIONS, RUNTIME_SETTINGS)


#: Every card in the built-in catalog (``cards/builtin``) and what it is to tune: the path
#: it names, or why it is not a tune path.
CARD_CATALOG_MAP: dict[str, str] = {
    "runtime.increase_gpu_offload_layers": "layers (tune records layer changes in either direction under this id)",
    "runtime.reduce_gpu_offload_layers": "layers (older optimizer only; tune does not use this id)",
    "runtime.reduce_threads": "threads (tune tries more and fewer threads under this id)",
    "runtime.enable_mmap": "not a tune path (older optimizer; the memory-residency path is not built)",
    "external.request_close": "not a tune path (suggest-only, other programs; step 8)",
    "external.suspend_process": "not a tune path (suggest-only, other programs; step 8)",
    "external.reduce_priority": "not a tune path (suggest-only, other programs; step 8)",
    "system.increase_pagefile": "not a tune path (suggest-only, operating system; step 8)",
}


def declaration(name: str) -> PathDeclaration:
    for d in DECLARATIONS:
        if d.name == name:
            return d
    raise KeyError(f"no path declaration named {name!r}")


def by_card(card_id: str) -> PathDeclaration | None:
    return next((d for d in DECLARATIONS if d.card_id == card_id), None)


def revisits_after(name: str) -> tuple[str, ...]:
    """Paths to look at again after a change on ``name`` is kept (declared)."""
    return declaration(name).invalidates


def memory_shaping_paths() -> frozenset[str]:
    return frozenset(d.name for d in DECLARATIONS if d.memory_shaping)


def missing_capability(d: PathDeclaration, available: Mapping[str, bool]) -> str | None:
    """The first required capability that ``available`` says is missing, with the path's
    own reason when it declares one. ``None`` means only that nothing *known* is
    missing: a capability absent from ``available`` is unresolved, which is not proof
    that the path is supported (see :func:`unresolved_capabilities`)."""
    for capability in d.requires:
        if capability in available and not available[capability]:
            reasons = d.practical.skip_reasons if d.practical is not None else {}
            return reasons.get(capability, f"missing: {CAPABILITIES.get(capability, capability)}")
    return None


def unresolved_capabilities(d: PathDeclaration, available: Mapping[str, bool]) -> tuple[str, ...]:
    """Required capabilities ``available`` says nothing about (not checked by the caller)."""
    return tuple(c for c in d.requires if c not in available)


def affects_table() -> dict[str, tuple[str, ...]]:
    """Path -> paths to look at again after a change kept on it (``diagnose.AFFECTS``)."""
    return {d.name: d.invalidates for d in DECLARATIONS}


def quality_by_card() -> dict[str, str]:
    """Card id -> quality level name (``quality_risk.REQUIREMENTS`` maps the names to its
    requirement objects)."""
    return {d.card_id: d.quality for d in DECLARATIONS}


def value_decided_cards() -> frozenset[str]:
    """Cards whose quality level is decided by the values they set."""
    return frozenset(d.card_id for d in DECLARATIONS if d.quality_by_value)


@dataclass(frozen=True)
class PracticalTables:
    """The practical search's view of the declarations (the C3 adapter)."""

    categories: tuple[str, ...]
    setting: dict[str, str]
    slot: dict[str, int]
    question: dict[str, str]
    guard: dict[str, str | None]
    card_of: dict[str, str]
    #: Slots a category's readings are valid for: a reading counts only where these
    #: other settings equal the ones it was measured with (evidence scope).
    scope: dict[str, tuple[int, ...]]
    declarations: dict[str, PathDeclaration]

    def stale_after(self, setting: str) -> tuple[str, ...]:
        """Categories whose evidence a change of ``setting`` makes out of context."""
        changed = next((c for c in self.categories if self.setting[c] == setting), None)
        if changed is None:
            return ()
        return tuple(c for c in self.categories if c != changed and self.slot[changed] in self.scope[c])

    def missing_capability(self, category: str, available: Mapping[str, bool]) -> str | None:
        return missing_capability(self.declarations[category], available)

    def by_setting(self, setting: str) -> str | None:
        """The category that writes ``setting`` as its own setting (None: not practical)."""
        return next((c for c in self.categories if self.setting[c] == setting), None)

    def questions_of(self, category: str) -> tuple[str, ...]:
        """Questions a change in ``category`` is measured on: judged first, then guarded."""
        guard = self.guard[category]
        return (self.question[category],) + ((guard,) if guard is not None else ())

    def final_questions(self, changed: set[str]) -> dict[str, str]:
        """Question -> mode for a whole-setup final check of a setup changing ``changed``
        categories: "better" on every question a change is judged on, "not_worse" on
        every question a change is only guarded on."""
        better = [self.question[c] for c in self.categories if c in changed]
        guarded = [self.guard[c] for c in self.categories if c in changed and self.guard[c] is not None]
        out: dict[str, str] = {}
        for q in (SHORT, LONG):
            if q in better:
                out[q] = "better"
            elif q in guarded:
                out[q] = "not_worse"
        return out


#: The practical search's setup layout: (layers, threads, batch). Its planner, setups and
#: evidence scope index a three-value setup by these slots, so the adapter accepts exactly
#: this mapping (category -> slot and own setting); a declaration cannot add a fourth
#: category or move one.
PRACTICAL_LAYOUT: tuple[tuple[str, str], ...] = (
    ("placement", "gpu_offload_layers"),
    ("threads", "threads"),
    ("batch", "ubatch_size"),
)


def check_practical_layout(declarations: tuple[PathDeclaration, ...]) -> None:
    """Raise :class:`DeclarationError` unless the declared practical uses are exactly the
    adapter's layout: the same categories in the same slots, each writing that slot's
    setting as its own. This is the current three-category adapter's constraint, kept
    apart from the generic validation in ``contracts/path_declaration``."""
    declared = {d.practical.category: (d.practical.slot, d.settings_written[:1]) for d in declarations
                if d.practical is not None}
    expected = {category: (slot, (setting,)) for slot, (category, setting) in enumerate(PRACTICAL_LAYOUT)}
    if declared != expected:
        raise DeclarationError(f"practical uses {declared} do not match the practical search's layout {expected}")


check_practical_layout(DECLARATIONS)


def practical_tables(declarations: tuple[PathDeclaration, ...] | None = None) -> PracticalTables:
    source = DECLARATIONS if declarations is None else declarations
    if declarations is not None:
        validate_declarations(source, RUNTIME_SETTINGS)
    check_practical_layout(source)
    used = sorted((d for d in source if d.practical is not None), key=lambda d: d.practical.slot)  # type: ignore[union-attr]
    categories = tuple(d.practical.category for d in used)  # type: ignore[union-attr]
    slots = {d.practical.category: d.practical.slot for d in used}  # type: ignore[union-attr]
    return PracticalTables(
        categories=categories,
        setting={d.practical.category: d.settings_written[0] for d in used},  # type: ignore[union-attr]
        slot=slots,
        question={d.practical.category: d.judged_on for d in used},  # type: ignore[union-attr]
        guard={d.practical.category: d.guarded_on for d in used},  # type: ignore[union-attr]
        card_of={d.practical.category: d.card_id for d in used},  # type: ignore[union-attr]
        # Exact-setup evidence: a reading is valid only where every other practical
        # setting is unchanged (the practical search's rule since review 8).
        scope={c: tuple(slots[o] for o in categories if o != c) for c in categories},
        declarations={d.practical.category: d for d in used},  # type: ignore[union-attr]
    )


__all__ = [
    "CAPABILITIES",
    "CARD_CATALOG_MAP",
    "DECLARATIONS",
    "DECLARATIONS_VERSION",
    "LONG",
    "PRACTICAL_LAYOUT",
    "RUNTIME_SETTINGS",
    "SHORT",
    "DeclarationError",
    "PathDeclaration",
    "PracticalTables",
    "PracticalUse",
    "affects_table",
    "by_card",
    "check_practical_layout",
    "declaration",
    "memory_shaping_paths",
    "missing_capability",
    "practical_tables",
    "quality_by_card",
    "revisits_after",
    "unresolved_capabilities",
    "validate_declarations",
    "value_decided_cards",
]
