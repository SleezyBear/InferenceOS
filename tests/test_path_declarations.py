"""Path declarations and the practical search's adapter (C3, first slice, 9 Oct 2026).

Two kinds of check:

* consistency: the declarations agree with the code that still holds the same facts
  (``OptionPath`` objects, quality requirements, Unbounded's revisit table and memory
  paths, the card catalog), and the tables the practical search now derives from them
  are identical to the literals they replaced;
* interactions: through the adapter, the practical search still rechecks the right
  evidence after a setup change, keeps revisits bounded, skips a path whose capability
  is missing with its reason, keeps an earlier complete setup as a candidate, and never
  runs the five paths it does not use.
"""

from __future__ import annotations

import ast
from itertools import combinations
from pathlib import Path

import pytest

import inferenceos.cards.builtin as builtin_cards
from inferenceos.cards.builtin.gpu_layers_more import CARD_ID as LAYERS_CARD_ID
from inferenceos.cards.catalog import default_catalog
from inferenceos.tune import declarations as decl
from inferenceos.tune import planner as planner_module
from inferenceos.tune.budgeted import CARD_OF
from inferenceos.tune.diagnose import AFFECTS, MEMORY_PATHS
from inferenceos.tune.paths import ALL_PATHS, LONG_QUESTION, PROMPT_BATCH
from inferenceos.tune.paths import THREADS as THREADS_PATH
from inferenceos.tune.planner import (
    BATCH,
    MAX_RECHECKS,
    PLACEMENT,
    THREADS,
    Planner,
    Structure,
)
from inferenceos.tune.quality_risk import REQUIREMENTS
from tests.test_budgeted_tune import (  # noqa: F401
    Clock,
    _interaction,
    _practical,
    _round,
    _world,
    clock,
)

SRC = Path(decl.__file__)


# --------------------------------------------------------------------------- #
# Consistency                                                                  #
# --------------------------------------------------------------------------- #
def test_every_tune_path_has_exactly_one_declaration() -> None:
    names = [d.name for d in decl.DECLARATIONS]
    assert len(names) == len(set(names)) == 8
    assert set(names) == {p.name for p in ALL_PATHS} | {"layers"}
    assert len({d.card_id for d in decl.DECLARATIONS}) == 8


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}


def test_the_declarations_module_depends_on_no_search_code() -> None:
    """The declarations import only contracts; the shared type imports nothing from
    InferenceOS at all."""
    own = {m for m in _imports(SRC) if m.startswith("inferenceos")}
    assert own == {"inferenceos.contracts.experiment", "inferenceos.contracts.path_declaration"}, own
    shared = Path(decl.__file__).parents[1] / "contracts" / "path_declaration.py"
    assert not {m for m in _imports(shared) if m.startswith("inferenceos")}


def test_importing_the_declarations_loads_no_planner_pipeline_model_or_execution_code() -> None:
    import subprocess
    import sys

    code = ("import sys, inferenceos.contracts.path_declaration, inferenceos.tune.declarations; "
            "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('inferenceos'))))")
    loaded = set(subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                                text=True).stdout.split())
    forbidden = ("inferenceos.tune.planner", "inferenceos.tune.pipeline", "inferenceos.tune.budgeted",
                 "inferenceos.tune.paths", "inferenceos.backends", "inferenceos.model", "inferenceos.experiment",
                 "inferenceos.optimization", "inferenceos.cards", "inferenceos.runtime", "inferenceos.native")
    assert not {m for m in loaded if m.startswith(forbidden)}, sorted(loaded)


def test_declarations_match_the_option_paths() -> None:
    to_question = {None: decl.SHORT, LONG_QUESTION: decl.LONG}
    for path in ALL_PATHS:
        d = decl.declaration(path.name)
        assert d.card_id == path.card_id
        assert d.settings_written[0] == path.setting
        assert d.engine_option == path.option
        assert d.judged_on == to_question[path.workload_id], path.name
        guard = None if path.guard_workload_id is None else to_question.get(path.guard_workload_id, decl.SHORT)
        assert d.guarded_on == guard, path.name
    layers = decl.declaration("layers")
    assert layers.card_id == LAYERS_CARD_ID and layers.settings_written == ("gpu_offload_layers",)


def test_declarations_match_quality_revisits_and_memory_paths() -> None:
    for d in decl.DECLARATIONS:
        requirement = REQUIREMENTS[d.card_id]
        assert d.quality == requirement.name, d.name
        assert d.quality_by_value == requirement.known_answers, d.name
        assert d.invalidates == AFFECTS[d.name], d.name
        assert decl.revisits_after(d.name) == AFFECTS[d.name]
    assert decl.memory_shaping_paths() == MEMORY_PATHS
    assert set(AFFECTS) == {d.name for d in decl.DECLARATIONS}


def test_card_links_and_the_old_catalog_are_accounted_for() -> None:
    catalog = default_catalog()
    builtin = Path(builtin_cards.__file__).parent
    for d in decl.DECLARATIONS:
        if d.card_file is None:
            assert d.card_id not in catalog.ids(), d.name  # no card file yet: said so
        else:
            assert (builtin / d.card_file).is_file(), d.card_file
            assert d.card_id in catalog.ids(), d.name
    assert set(decl.CARD_CATALOG_MAP) == set(catalog.ids())
    for card_id, meaning in decl.CARD_CATALOG_MAP.items():
        found = decl.by_card(card_id)
        assert (found is not None) == (not meaning.startswith(("not a tune path", "layers (older"))), card_id


def test_only_three_paths_are_used_by_the_practical_search() -> None:
    used = [d.name for d in decl.DECLARATIONS if d.practical is not None]
    assert used == ["layers", "threads", "prompt_batch"]


# --------------------------------------------------------------------------- #
# The adapter reproduces the tables it replaced                                #
# --------------------------------------------------------------------------- #
def test_practical_tables_equal_the_old_literals() -> None:
    t = decl.practical_tables()
    assert t.categories == (PLACEMENT, THREADS, BATCH) == planner_module.CATEGORIES
    assert planner_module.QUESTION == {PLACEMENT: "short", THREADS: "short", BATCH: "long"}
    assert planner_module.SETTING == {PLACEMENT: "gpu_offload_layers", THREADS: "threads", BATCH: "ubatch_size"}
    assert planner_module.SLOT == {PLACEMENT: 0, THREADS: 1, BATCH: 2}
    assert CARD_OF == {PLACEMENT: LAYERS_CARD_ID, THREADS: THREADS_PATH.card_id, BATCH: PROMPT_BATCH.card_id}
    # Exact-setup evidence: a reading is valid for the other two settings.
    for category, slot in planner_module.SLOT.items():
        assert planner_module.SCOPE[category] == tuple(i for i in range(3) if i != slot)
    # A placement change makes thread and batch evidence stale (the old literal recheck set).
    assert t.stale_after("gpu_offload_layers") == (THREADS, BATCH)


def _old_final_plan(changed: set[str]) -> dict[str, str]:
    """The final-check questions as ``_final_plan`` decided them before C3."""
    settings = {planner_module.SETTING[c] for c in changed}
    out: dict[str, str] = {}
    short_found = bool(settings & {"gpu_offload_layers", "threads"})
    long_found = "ubatch_size" in settings
    if short_found:
        out["short"] = "better"
    elif long_found:
        out["short"] = "not_worse"
    if long_found:
        out["long"] = "better"
    return out


def test_final_check_questions_are_unchanged_for_every_combination() -> None:
    t = decl.practical_tables()
    for size in range(4):
        for changed in combinations(t.categories, size):
            got = t.final_questions(set(changed))
            assert got == _old_final_plan(set(changed)), changed
            assert list(got) == list(_old_final_plan(set(changed))), changed  # same order


def test_questions_measured_per_change_are_unchanged() -> None:
    t = decl.practical_tables()
    assert t.questions_of(PLACEMENT) == ("short",)
    assert t.questions_of(THREADS) == ("short",)
    assert t.questions_of(BATCH) == ("long", "short")
    assert t.by_setting("ubatch_size") == BATCH and t.by_setting("threads_batch") is None


# --------------------------------------------------------------------------- #
# Interactions through the adapter                                            #
# --------------------------------------------------------------------------- #
def _structure(**kw: object) -> Structure:
    base: dict[str, object] = dict(count_max=25, top=25, uses_experts=False, mixed_kinds=False,
                                   physical_cores=6, logical_cores=12, long_question_tokens=3000)
    base.update(kw)
    return Structure(**base)  # type: ignore[arg-type]


def _drive(planner: Planner, truth: dict[str, dict[int, float]], noise: float = 0.01) -> list[tuple[str, tuple[int, ...]]]:
    steps = []
    for _ in range(40):
        step = planner.next_step()
        if step is None:
            break
        steps.append((step.category, step.values))
        for value in step.values:
            planner.record(step.category, value, truth[step.category].get(value, 0.0), noise)
    return steps


def test_a_placement_change_makes_thread_evidence_out_of_context_and_rechecks_it() -> None:
    planner = Planner(_structure(), start_layers=25, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.8)
    _drive(planner, {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {10: 0.08, 9: 0.06}, BATCH: {}})
    assert planner.chosen[THREADS] == 10 and planner.in_context(THREADS)
    steps = planner.placement_changed(22)
    # Thread readings at 25 layers no longer count at 22: rechecked there, at 22 layers.
    assert not planner.in_context(THREADS)
    assert [s.category for s in steps] == [THREADS]
    assert steps[0].base[0] == 22 and set(steps[0].values) == {10, 9}
    # Every category it rechecks is one the declarations say a placement change affects.
    assert {s.category for s in steps} <= set(decl.practical_tables().stale_after("gpu_offload_layers"))


def test_a_batch_change_invalidates_thread_evidence() -> None:
    assert THREADS in decl.practical_tables().stale_after("ubatch_size")
    planner = Planner(_structure(), start_layers=25, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.8)
    _drive(planner, {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {10: 0.08, 8: 0.03, 9: 0.05},
                     BATCH: {1024: 0.12, 2048: 0.05}})
    assert planner.chosen[BATCH] == 1024
    # The thread readings at batch 512 were not reused at batch 1024: measured again there.
    assert any(r.setup[2] == 1024 for r in planner.readings[THREADS])
    assert all(r.context == (25, 1024) for r in planner.in_context(THREADS))


def test_an_unrelated_change_causes_no_revisit() -> None:
    t = decl.practical_tables()
    # Settings the practical search does not search invalidate none of its evidence.
    for setting in ("threads_batch", "flash_attention", "kv_cache_type_k", "placement_overrides"):
        assert t.stale_after(setting) == ()
    planner = Planner(_structure(), start_layers=25, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.8)
    _drive(planner, {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {10: 0.08}, BATCH: {}})
    log_before = list(planner.log)
    # The same placement again: nothing is out of context, nothing is rechecked.
    assert planner.placement_changed(25) == []
    assert planner.next_step() is None
    assert [line for line in planner.log[len(log_before):] if "recheck" in line] == []


def test_revisit_limits_hold() -> None:
    planner = Planner(_structure(), start_layers=25, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.8)
    truth = {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {10: 0.08, 9: 0.06, 8: 0.03},
             BATCH: {1024: 0.12, 2048: 0.05}}
    _drive(planner, truth)
    # Within one setup the recheck happened at most MAX_RECHECKS times per category.
    assert all(n <= MAX_RECHECKS for n in planner._rechecks_done.values())
    assert planner._recheck(THREADS) is None
    # A placement change allows one more recheck per affected category, not more.
    steps = planner.placement_changed(22)
    assert len([s for s in steps if s.category == THREADS]) <= 1
    _drive(planner, truth)
    assert all(n <= MAX_RECHECKS for n in planner._rechecks_done.values())
    assert planner._recheck(THREADS) is None and planner._recheck(BATCH) is None


def test_an_unsupported_path_is_skipped_with_its_declared_reason(tmp_path: Path, clock: Clock,  # noqa: F811
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    t = decl.practical_tables()
    assert t.missing_capability(BATCH, {"long_question": False}) == "no long test question"
    assert t.missing_capability(BATCH, {"long_question": True}) is None
    assert t.missing_capability(THREADS, {"long_question": False}) is None
    assert t.missing_capability(PLACEMENT, {"long_question": False}) is None  # model facts not judged here
    # A planned batch step with no long question: logged as skipped and never measured.
    _, _, driver, _ = _practical(tmp_path, _world(clock), 3600.0, clock=clock)
    def no_long_question(self: object, planner: Planner, reopen: bool = False) -> bool:
        # The planner is told a length (so it plans batch steps) but no long question exists.
        planner.set_question_length(3000)
        return True

    monkeypatch.setattr(type(driver), "_size_long", no_long_question)
    result = driver.run()
    search = _round(result, "time budget")
    assert "batch skipped: no long test question" in search["planner_log"]
    assert not [s for s in search["steps"] if s["category"] == BATCH]


def test_the_five_other_paths_never_run_in_the_practical_search(tmp_path: Path, clock: Clock) -> None:  # noqa: F811
    _, _, driver, _ = _practical(tmp_path, _world(clock, factor=_interaction), 100000.0, clock=clock)
    result = driver.run()
    budget = _round(result, "time budget")
    assert {s["category"] for s in budget["steps"]} <= {PLACEMENT, THREADS, BATCH}
    assert {c.card_id for c in result.proposed_changes} <= set(CARD_OF.values())
    unused = {d.card_id for d in decl.DECLARATIONS if d.practical is None}
    assert not any(card in str(budget["operations"]) for card in unused)


def test_an_earlier_complete_setup_remains_the_best_through_the_adapter(tmp_path: Path, clock: Clock) -> None:  # noqa: F811
    _, _, driver, _ = _practical(tmp_path, _world(clock, factor=_interaction), 100000.0, clock=clock)
    result = driver.run()
    search = _round(result, "practical search")
    earlier = next(o for o in search["options"] if o["settings"] == {"gpu_offload_layers": 0, "threads": 10})
    assert earlier["source"] == "earlier complete setup"
    assert result.outcome == "saved"
    final = result.final_configuration
    assert (final.gpu_offload_layers, final.threads, final.ubatch_size) == (0, 10, None)
    # Recorded under the declared cards, measured on the declared questions.
    assert {c.card_id for c in result.proposed_changes} == {CARD_OF[PLACEMENT], CARD_OF[THREADS]}
    check = _round(result, "final check")
    assert set(check["blocks_per_question"]) == set(decl.practical_tables().final_questions({PLACEMENT, THREADS}))


# --------------------------------------------------------------------------- #
# Second slice: validation, the optional card field, derived static tables      #
# --------------------------------------------------------------------------- #
from dataclasses import replace  # noqa: E402

from inferenceos.cards.base import Card  # noqa: E402
from inferenceos.cards.catalog import CardCatalog, CardError, discover_builtin_cards, validate_card  # noqa: E402
from inferenceos.contracts.path_declaration import DeclarationError, PracticalUse  # noqa: E402
from inferenceos.contracts.tune import ProposedChange  # noqa: E402
from inferenceos.tune import diagnose, quality_risk  # noqa: E402

#: The literals the derived tables replaced (9 Oct 2026), kept to prove equality.
OLD_AFFECTS: dict[str, tuple[str, ...]] = {
    "threads": ("reading_threads", "prompt_batch"),
    "reading_threads": ("threads", "prompt_batch"),
    "prompt_batch": ("threads", "reading_threads", "flash_attention", "engine_switches", "placement"),
    "flash_attention": ("prompt_batch", "threads", "reading_threads", "engine_switches", "placement", "kv_cache"),
    "engine_switches": ("threads", "reading_threads", "prompt_batch", "flash_attention"),
    "placement": ("threads", "reading_threads", "prompt_batch", "flash_attention", "engine_switches", "kv_cache"),
    "kv_cache": ("placement", "prompt_batch", "flash_attention", "threads"),
    "layers": ("threads", "reading_threads", "prompt_batch", "placement", "kv_cache"),
}
OLD_MEMORY_PATHS = frozenset({"placement", "kv_cache"})
OLD_REQUIREMENTS = {
    "runtime.reduce_threads": quality_risk.RUNTIME,
    "runtime.reading_threads": quality_risk.RUNTIME,
    "runtime.prompt_batch": quality_risk.RUNTIME,
    "runtime.increase_gpu_offload_layers": quality_risk.RUNTIME,
    "runtime.flash_attention": quality_risk.ENGINE,
    "engine.switches": quality_risk.ENGINE,
    "runtime.tensor_placement": quality_risk.ENGINE,
    "runtime.kv_cache_format": quality_risk.PRECISION,
}
OLD_CATALOG_ORDER = [
    "external.reduce_priority", "external.request_close", "external.suspend_process", "runtime.enable_mmap",
    "runtime.increase_gpu_offload_layers", "runtime.reduce_gpu_offload_layers", "runtime.reduce_threads",
    "system.increase_pagefile",
]


def test_derived_static_tables_equal_the_literals_they_replaced() -> None:
    # Keep the historical extraction reference intact. The full-project audit adds
    # the explicitly missing layer -> engine-switch dependency, not a new order.
    current = {**OLD_AFFECTS, "layers": (*OLD_AFFECTS["layers"], "engine_switches")}
    assert diagnose.AFFECTS == current
    for name, revisit in current.items():
        assert diagnose.AFFECTS[name] == revisit and type(diagnose.AFFECTS[name]) is tuple  # same tuple order
        assert diagnose.affected_paths(name) == revisit
    assert diagnose.MEMORY_PATHS == OLD_MEMORY_PATHS and isinstance(diagnose.MEMORY_PATHS, frozenset)
    assert quality_risk.REQUIREMENTS == OLD_REQUIREMENTS
    for card, requirement in OLD_REQUIREMENTS.items():
        assert quality_risk.REQUIREMENTS[card] is requirement
    # Fallbacks are unchanged: an unknown path revisits nothing, an unknown card is engine level.
    assert diagnose.affected_paths("no_such_path") == ()
    assert quality_risk.requirement_for("runtime.no_such_card") is quality_risk.ENGINE
    # The value-decided set stays its own rule, and agrees with the declarations.
    assert quality_risk.VALUE_DECIDED_CARDS == decl.value_decided_cards() == frozenset({"runtime.kv_cache_format"})


def _change(card: str, setting: str, value: int | str, **related: int | str) -> ProposedChange:
    return ProposedChange(change_id="c", card_id=card, setting=setting, description="", new_value=value,
                          related=dict(related))


def test_value_sensitive_quality_still_decides_combined_changes() -> None:
    """The static card table is only a floor: a layers or threads change that also carries
    a lower-precision memory format still needs the known-answer questions."""
    layers = decl.declaration("layers").card_id
    threads = decl.declaration("threads").card_id
    card, req = quality_risk.requirement_of_change(_change(layers, "gpu_offload_layers", 27, kv_cache_type_k="q8_0"))
    assert req is quality_risk.PRECISION and card == "runtime.kv_cache_format"
    _, req = quality_risk.requirement_of_change(_change(threads, "threads", 8, kv_cache_type_v="q4_0"))
    assert req is quality_risk.PRECISION and req.known_answers
    _, req = quality_risk.requirement_of_change(_change("runtime.kv_cache_format", "kv_cache_type_k", "f16"))
    assert req is quality_risk.ENGINE  # back to 16-bit is not a precision change
    _, req = quality_risk.requirement_of_change(_change(threads, "threads", 8, placement_overrides="x=CPU"))
    assert req is quality_risk.ENGINE
    _, req = quality_risk.requirement_of_change(_change("runtime.unknown", "threads", 8))
    assert req is quality_risk.ENGINE  # unknown card: conservative
    _, req = quality_risk.requirement_of_change(_change(threads, "some_new_setting", 1))
    assert req is quality_risk.ENGINE  # unknown setting: conservative
    best, req = quality_risk.strictest([_change(threads, "threads", 8),
                                        _change(layers, "gpu_offload_layers", 27, kv_cache_type_k="q8_0")])
    assert req is quality_risk.PRECISION and best == "runtime.kv_cache_format"


def _bad(**changes: object) -> list[decl.PathDeclaration]:
    items = list(decl.DECLARATIONS)
    items[1] = replace(items[1], **changes)  # type: ignore[arg-type]
    return items


@pytest.mark.parametrize(("changes", "message"), [
    ({"name": "layers"}, "duplicate path name"),
    ({"card_id": "runtime.increase_gpu_offload_layers"}, "duplicate card id"),
    ({"card_file": "gpu_layers_more.py"}, "duplicate card file"),
    ({"practical": PracticalUse(category="placement", slot=5)}, "duplicate practical category"),
    ({"practical": PracticalUse(category="threads2", slot=0)}, "duplicate practical slot"),
    ({"settings_written": ("free card memory",)}, "not a runtime setting"),
    ({"settings_read": ("processor share",)}, "not a runtime setting"),
    ({"facts_read": ("ubatch_size",)}, "belongs in settings_read"),
    ({"judged_on": "medium"}, "unknown question"),
    ({"guarded_on": "short"}, "guarded on the question it is judged on"),
    ({"quality": "risky"}, "unknown quality level"),
    ({"requires": ("telepathy",)}, "unknown capability"),
    ({"practical": PracticalUse(category="threads", slot=1, skip_reasons={"long_question": "x"})},
     "does not require"),
    ({"invalidates": ("reading_threads", "no_such_path")}, "not a declared path"),
    ({"invalidates": ("threads",)}, "invalidates itself"),
    ({"card_id": "Not An Id"}, "invalid card id"),
])
def test_malformed_declarations_are_refused(changes: dict[str, object], message: str) -> None:
    with pytest.raises(DeclarationError, match=message):
        decl.validate_declarations(_bad(**changes), decl.RUNTIME_SETTINGS)


def test_the_shipped_declarations_validate() -> None:
    decl.validate_declarations(decl.DECLARATIONS, decl.RUNTIME_SETTINGS)
    for d in decl.DECLARATIONS:
        assert set(d.settings_read) <= decl.RUNTIME_SETTINGS and not set(d.facts_read) & decl.RUNTIME_SETTINGS


def test_an_unresolved_capability_is_not_taken_as_supported() -> None:
    placement = decl.declaration("placement")
    assert decl.missing_capability(placement, {"long_question": True}) is None
    assert decl.unresolved_capabilities(placement, {"long_question": True}) == ("graphics_card", "model_facts")
    batch = decl.declaration("prompt_batch")
    assert decl.missing_capability(batch, {"long_question": False}) == "no long test question"
    assert decl.unresolved_capabilities(batch, {"long_question": False}) == ()


def test_old_cards_have_no_tune_field_and_the_shared_cards_link_to_the_canonical_declarations() -> None:
    catalog = default_catalog()
    assert catalog.ids() == OLD_CATALOG_ORDER and [c.card_id for c in catalog.all()] == OLD_CATALOG_ORDER
    linked = {c.card_id: c.tune for c in catalog.all() if c.tune is not None}
    assert set(linked) == {"runtime.increase_gpu_offload_layers", "runtime.reduce_threads"}
    assert linked["runtime.increase_gpu_offload_layers"] is decl.declaration("layers")
    assert linked["runtime.reduce_threads"] is decl.declaration("threads")
    assert catalog.get("runtime.reduce_gpu_offload_layers").tune is None  # historical id kept, not linked
    # Paths without a card file stay out of the legacy catalog and its applied cards.
    applied = [c.card_id for c in catalog.applied_cards()]
    assert applied == ["runtime.enable_mmap", "runtime.increase_gpu_offload_layers",
                       "runtime.reduce_gpu_offload_layers", "runtime.reduce_threads"]
    for d in decl.DECLARATIONS:
        if d.card_file is None:
            assert d.card_id not in catalog.ids()


def test_the_tune_field_changes_nothing_the_older_optimizer_sees() -> None:
    for card in default_catalog().all():
        bare = replace(card, tune=None)
        assert bare == card and repr(bare) == repr(card)
        assert bare.to_action().model_dump() == card.to_action().model_dump()
        assert bare.prepare is card.prepare and bare.settings() == card.settings()
        validate_card(bare)


def test_card_links_are_validated() -> None:
    threads_card = default_catalog().get("runtime.reduce_threads")
    with pytest.raises(CardError, match="linked to the declaration of"):
        validate_card(replace(threads_card, tune=decl.declaration("layers")))
    with pytest.raises(CardError, match="conflicts with the canonical declaration"):
        validate_card(replace(threads_card, tune=replace(decl.declaration("threads"), card_file=None)))
    with pytest.raises(CardError, match="must be a path declaration"):
        validate_card(replace(threads_card, tune={"name": "threads"}))  # type: ignore[arg-type]
    with pytest.raises(CardError, match="conflicts with the canonical declaration"):
        validate_card(replace(threads_card, tune=replace(decl.declaration("threads"), settings_written=("threads_batch",))))
    # The canonical declaration itself must still match the card's own setting.
    with pytest.raises(CardError, match="declares 'threads' as its own setting"):
        validate_card(replace(threads_card, programs=(replace(threads_card.programs[0], setting="threads_batch"),)))
    with pytest.raises(CardError):
        CardCatalog([replace(threads_card, tune=decl.declaration("layers"))])


def test_a_card_file_must_match_its_declaration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "fakecards"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "renamed.py").write_text(
        "from inferenceos.cards.builtin.threads import CARD as _C\n"
        "CARD = _C\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(CardError, match="defined in renamed.py but its path declaration names 'threads.py'"):
        discover_builtin_cards("fakecards")
    assert sorted(c.card_id for c in discover_builtin_cards()) == default_catalog().ids()


def test_cards_without_a_tune_field_still_construct_and_validate() -> None:
    card = default_catalog().get("runtime.enable_mmap")
    assert card.tune is None
    fresh = Card(**{f: getattr(card, f) for f in card.__dataclass_fields__ if f != "tune"})
    assert fresh.tune is None and fresh == card
    validate_card(fresh)


# --------------------------------------------------------------------------- #
# Review of the second slice: card links and practical slots                    #
# --------------------------------------------------------------------------- #
def test_a_card_link_cannot_carry_a_conflicting_declaration() -> None:
    """Reviewer reproduction (9 Oct): the threads card with an invalid, non-canonical
    declaration was accepted; an empty settings list raised IndexError."""
    threads_card = default_catalog().get("runtime.reduce_threads")
    bad = replace(decl.declaration("threads"), quality="unknown-quality", invalidates=("missing_path",),
                  settings_written=("threads", "not_a_setting"))
    with pytest.raises(CardError, match="invalid path declaration"):
        validate_card(replace(threads_card, tune=bad))
    with pytest.raises(CardError, match="writes no setting"):
        validate_card(replace(threads_card, tune=replace(decl.declaration("threads"), settings_written=())))
    # Structurally valid but not the canonical description: rejected as a conflict.
    for changes in ({"invalidates": ("missing_path",)}, {"label": "another description"},
                    {"invalidates": ("prompt_batch",)}, {"settings_read": ("gpu_offload_layers",)}):
        with pytest.raises(CardError, match="conflicts with the canonical declaration"):
            validate_card(replace(threads_card, tune=replace(decl.declaration("threads"), **changes)))
    with pytest.raises(CardError, match="no path declaration exists"):
        validate_card(replace(threads_card, card_id="runtime.no_such_path",
                              tune=replace(decl.declaration("threads"), card_id="runtime.no_such_path")))
    validate_card(threads_card)
    assert threads_card.tune is decl.declaration("threads")
    validate_card(replace(threads_card, tune=None))  # legacy: no link, unchanged


@pytest.mark.parametrize("slot", [-1, 99, True, 1.0, "1"])
def test_invalid_practical_slots_are_refused(slot: object) -> None:
    items = list(decl.DECLARATIONS)
    i = next(k for k, d in enumerate(items) if d.name == "threads")
    items[i] = replace(items[i], practical=PracticalUse(category="threads", slot=slot))  # type: ignore[arg-type]
    with pytest.raises(DeclarationError):
        decl.validate_declarations(items, decl.RUNTIME_SETTINGS)
    with pytest.raises(DeclarationError):
        decl.practical_tables(tuple(items))


def test_practical_slot_gaps_duplicates_and_layout_changes_are_refused() -> None:
    items = list(decl.DECLARATIONS)
    i = next(k for k, d in enumerate(items) if d.name == "threads")
    j = next(k for k, d in enumerate(items) if d.name == "prompt_batch")
    gap = list(items)
    gap[j] = replace(items[j], practical=PracticalUse(category="batch", slot=3,
                                                      skip_reasons={"long_question": "no long test question"}))
    with pytest.raises(DeclarationError, match="without gaps"):
        decl.validate_declarations(gap, decl.RUNTIME_SETTINGS)
    duplicate = list(items)
    duplicate[i] = replace(items[i], practical=PracticalUse(category="threads", slot=2))
    with pytest.raises(DeclarationError, match="duplicate practical slot"):
        decl.validate_declarations(duplicate, decl.RUNTIME_SETTINGS)
    swapped = list(items)
    swapped[i] = replace(items[i], practical=PracticalUse(category="threads", slot=2))
    swapped[j] = replace(items[j], practical=PracticalUse(category="batch", slot=1,
                                                          skip_reasons={"long_question": "no long test question"}))
    decl.validate_declarations(swapped, decl.RUNTIME_SETTINGS)  # generic rules hold ...
    with pytest.raises(DeclarationError, match="layout"):
        decl.practical_tables(tuple(swapped))  # ... but the adapter's layout does not
    fourth = list(items)
    k = next(n for n, d in enumerate(items) if d.name == "reading_threads")
    fourth[k] = replace(items[k], practical=PracticalUse(category="reading", slot=3))
    decl.validate_declarations(fourth, decl.RUNTIME_SETTINGS)
    with pytest.raises(DeclarationError, match="layout"):
        decl.practical_tables(tuple(fourth))  # no fourth category can be enabled by data
    bad_name = list(items)
    bad_name[i] = replace(items[i], practical=PracticalUse(category="Threads!", slot=1))
    with pytest.raises(DeclarationError, match="invalid practical category"):
        decl.validate_declarations(bad_name, decl.RUNTIME_SETTINGS)


def test_the_shipped_practical_layout_is_accepted() -> None:
    tables = decl.practical_tables(tuple(decl.DECLARATIONS))
    assert tables == decl.practical_tables()
    assert [(c, tables.slot[c], tables.setting[c]) for c in tables.categories] == [
        (c, i, s) for i, (c, s) in enumerate(decl.PRACTICAL_LAYOUT)]


def test_applicability_handler_ids_are_validated() -> None:
    with pytest.raises(DeclarationError, match="invalid applicability handler id"):
        decl.validate_declarations(_bad(applicability="threads processor"), decl.RUNTIME_SETTINGS)
    with pytest.raises(DeclarationError, match="invalid applicability handler id"):
        decl.validate_declarations(_bad(applicability="os.system"[:2]), decl.RUNTIME_SETTINGS)
    with pytest.raises(DeclarationError, match="duplicate applicability handler"):
        decl.validate_declarations(_bad(applicability="prompt_batch.always"), decl.RUNTIME_SETTINGS)
    assert [d.applicability is None for d in decl.DECLARATIONS] == [d.name == "layers" for d in decl.DECLARATIONS]
