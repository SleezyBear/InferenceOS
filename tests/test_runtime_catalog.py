"""The runtime catalog (C3, 10 Oct 2026): every declared path bound to its existing code.

The seven option paths are their ``OptionPath`` bindings, with the facts their
declarations hold exactly (label, card, own setting, questions, option, rule) read from
the declaration; the layer count is a specialised search with its modes' entry points;
the specialised verifier map is a projection of the bindings. These tests check the
catalog's validation, that every binding is what it was, that the verification lifecycle
is unchanged, and that importing the catalog starts nothing."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from inferenceos.contracts.experiment import ExperimentMeasurement, RuntimeConfiguration
from inferenceos.tune import budgeted, catalog as catalog_module, pipeline
from inferenceos.tune import paths as paths_module
from inferenceos.tune.catalog import CATALOG, LAYERS, PRACTICAL, UNBOUNDED, CatalogError, SpecializedSearch, build_catalog
from inferenceos.tune.declarations import DECLARATIONS
from inferenceos.tune.paths import (
    ALL_PATHS,
    KV_CACHE,
    LONG_QUESTION,
    PLACEMENT,
    SHORT_QUESTION,
    THREADS,
    VERIFIERS,
    cache_verify,
    paths_by_name,
    placement_verify,
    verifier_bindings,
    verify_every_run,
)

#: The bindings as the ``OptionPath`` objects held them before (literal copies).
OLD = {
    "threads": ("processor threads", "threads", "runtime.reduce_threads", None, None, "--threads"),
    "reading_threads": ("reading threads", "threads_batch", "runtime.reading_threads", LONG_QUESTION,
                        SHORT_QUESTION, "--threads-batch"),
    "flash_attention": ("flash attention", "flash_attention", "runtime.flash_attention", LONG_QUESTION,
                        SHORT_QUESTION, "--flash-attn"),
    "prompt_batch": ("prompt batch size", "ubatch_size", "runtime.prompt_batch", LONG_QUESTION, SHORT_QUESTION,
                     "--ubatch-size"),
    "engine_switches": ("engine switches", "engine_switches", "engine.switches", None, LONG_QUESTION, ""),
    "placement": ("per-piece placement", "placement_overrides", "runtime.tensor_placement", None, LONG_QUESTION,
                  "--override-tensor"),
    "kv_cache": ("conversation memory format", "kv_cache_type_k", "runtime.kv_cache_format", LONG_QUESTION,
                 SHORT_QUESTION, "--cache-type-k"),
}


def test_every_declaration_has_exactly_one_runtime_entry() -> None:
    assert CATALOG.names() == tuple(d.name for d in DECLARATIONS)
    assert CATALOG.option_paths is ALL_PATHS
    assert [p.name for p in ALL_PATHS] == list(OLD)  # order unchanged
    assert CATALOG.specialized == (LAYERS,)
    assert CATALOG.specialized_search("layers").modes == frozenset({UNBOUNDED, PRACTICAL})
    with pytest.raises(CatalogError):
        CATALOG.option_path("layers")  # the layer count is not an option path


def test_bindings_read_from_the_declarations_equal_the_old_fields() -> None:
    for path in ALL_PATHS:
        assert (path.label, path.setting, path.card_id, path.workload_id, path.guard_workload_id, path.option) == \
            OLD[path.name], path.name


def test_threads_keeps_the_callers_own_question() -> None:
    """The declaration says "short"; the binding keeps None, which means tune's own
    chosen question, not the literal short question."""
    assert THREADS.declaration.judged_on == "short" and THREADS.workload_id is None
    assert PLACEMENT.declaration.judged_on == "short" and PLACEMENT.workload_id is None


def test_paths_by_name_keeps_its_behaviour() -> None:
    names = [p.name for p in paths_by_name(["prompt_batch", "nope", " threads ", "threads", "layers", ""])]
    assert names == ["prompt_batch", "threads"]
    assert paths_by_name([]) == []
    assert [p.name for p in paths_by_name(list(reversed(OLD)))] == list(reversed(OLD))
    assert CATALOG.by_names(["kv_cache"]) == paths_by_name(["kv_cache"]) == [KV_CACHE]


def test_the_verifier_map_is_a_projection_with_the_same_coverage_and_order() -> None:
    assert list(VERIFIERS) == ["placement_overrides", "kv_cache_type_k", "kv_cache_type_v"]
    assert VERIFIERS["placement_overrides"] is placement_verify
    assert VERIFIERS["kv_cache_type_k"] is VERIFIERS["kv_cache_type_v"] is cache_verify
    assert CATALOG.verifiers() == VERIFIERS
    # Secondary settings keep their own checks: no specialised verifier is attached to them.
    assert "gpu_offload_layers" not in VERIFIERS and "flash_attention" not in VERIFIERS
    assert PLACEMENT.verifies == ("placement_overrides",)
    assert KV_CACHE.verifies == ("kv_cache_type_k", "kv_cache_type_v")


def test_conflicting_or_malformed_bindings_are_refused() -> None:
    other = replace(PLACEMENT, verify=cache_verify)
    with pytest.raises(RuntimeError, match="two different verifiers"):
        verifier_bindings((PLACEMENT, other))
    with pytest.raises(RuntimeError, match="must be given together"):
        paths_module._check_bindings((replace(THREADS, verify=cache_verify),) + ALL_PATHS[1:])
    with pytest.raises(RuntimeError, match="not distinct settings the path writes"):
        paths_module._check_bindings(ALL_PATHS[:-1] + (replace(KV_CACHE, verifies=("kv_cache_type_k", "typo")),))
    with pytest.raises(RuntimeError, match="not distinct settings the path writes"):
        paths_module._check_bindings(ALL_PATHS[:-1] + (replace(KV_CACHE, verifies=("kv_cache_type_k",
                                                                                   "kv_cache_type_k")),))
    with pytest.raises(RuntimeError, match="candidates is not callable"):
        paths_module._check_bindings((replace(THREADS, candidates="threads.candidates"),) + ALL_PATHS[1:])  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="notes is not callable"):
        paths_module._check_bindings((replace(THREADS, notes=3),) + ALL_PATHS[1:])  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no declaration"):
        paths_module._check_bindings(ALL_PATHS + (replace(THREADS, name="bogus"),))
    with pytest.raises(RuntimeError, match="duplicate path bindings"):
        paths_module._check_bindings(ALL_PATHS + (THREADS,))


def test_the_catalog_refuses_missing_extra_and_misplaced_entries() -> None:
    with pytest.raises(CatalogError, match="'kv_cache' has no runtime entry"):
        build_catalog(ALL_PATHS[:-1], (LAYERS,))
    with pytest.raises(CatalogError, match="'layers' has no runtime entry"):
        build_catalog(ALL_PATHS, ())
    with pytest.raises(CatalogError, match="more than one runtime entry"):
        build_catalog(ALL_PATHS + (THREADS,), (LAYERS,))
    with pytest.raises(CatalogError, match="'bogus' has no declaration"):
        build_catalog(ALL_PATHS, (LAYERS, SpecializedSearch("bogus", "x", {UNBOUNDED: (len,)})))
    with pytest.raises(CatalogError, match="declares an option-path applicability rule"):
        build_catalog(ALL_PATHS[1:], (LAYERS, SpecializedSearch("threads", "x", {UNBOUNDED: (len,), PRACTICAL: (len,)})))
    with pytest.raises(CatalogError, match="its declaration says"):
        build_catalog(ALL_PATHS, (SpecializedSearch("layers", "x", {UNBOUNDED: (len,)}),))
    with pytest.raises(CatalogError, match="needs callable entry points"):
        build_catalog(ALL_PATHS, (SpecializedSearch("layers", "x", {UNBOUNDED: ("find_layer_top",),  # type: ignore[dict-item]
                                                                     PRACTICAL: (len,)}),))
    with pytest.raises(CatalogError, match="two different verifiers"):
        build_catalog(ALL_PATHS + (replace(PLACEMENT, name="kv_cache", verify=cache_verify),), (LAYERS,))


def test_the_specialised_entry_points_are_the_objects_each_mode_calls() -> None:
    unbounded = LAYERS.entry_points[UNBOUNDED]
    practical = LAYERS.entry_points[PRACTICAL]
    assert unbounded == (pipeline.find_layer_top, pipeline.search_layers)
    assert practical == (budgeted.find_layer_top, budgeted.Planner)
    assert catalog_module.PRACTICAL_HELPERS["prompt_batch.sizes"] is budgeted._batch_sizes


def test_importing_the_catalog_starts_nothing() -> None:
    code = (
        "import subprocess, threading, sys\n"
        "calls = []\n"
        "subprocess.Popen.__init__ = lambda self, *a, **k: calls.append(a) or (_ for _ in ()).throw(RuntimeError('spawn'))\n"
        "before = threading.active_count()\n"
        "import inferenceos.tune.catalog\n"
        "print(len(calls), threading.active_count() - before)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.split()
    assert out == ["0", "0"]  # no process spawned, no thread (sampler) started


def _measurement(*runs: str) -> ExperimentMeasurement:
    return ExperimentMeasurement.model_construct(runs=[SimpleNamespace(tag=tag) for tag in runs])


def _tagged(context: Any, config: Any, measurement: ExperimentMeasurement) -> tuple[str, str]:
    tag = measurement.runs[0].tag  # type: ignore[attr-defined]
    return ("shown", "seen") if tag == "good" else ("not_seen", "missing")


def test_every_usable_run_must_show_the_effect() -> None:
    config = RuntimeConfiguration(context_length=4096)
    assert verify_every_run(_tagged, None, config, _measurement())[0] == "unverified"  # type: ignore[arg-type]
    assert verify_every_run(_tagged, None, config, _measurement("good", "good"))[0] == "shown"  # type: ignore[arg-type]
    status, words = verify_every_run(_tagged, None, config, _measurement("good", "bad", "good"))  # type: ignore[arg-type]
    assert status == "not_seen" and words.startswith("run 2 of 3")


def test_settings_to_verify_dedupes_cache_settings_and_keeps_removals() -> None:
    base = RuntimeConfiguration(context_length=4096)
    pick = pipeline.Tuner._settings_to_verify
    both = base.model_copy(update={"kv_cache_type_k": "q8_0", "kv_cache_type_v": "q8_0"})
    assert pick(None, base, both) == ["kv_cache_type_k"]  # type: ignore[arg-type]
    values_only = base.model_copy(update={"kv_cache_type_v": "q8_0"})
    assert pick(None, base, values_only) == ["kv_cache_type_v"]  # type: ignore[arg-type]
    keys_only = base.model_copy(update={"kv_cache_type_k": "q8_0"})
    assert pick(None, base, keys_only) == ["kv_cache_type_k"]  # type: ignore[arg-type]
    placed = both.model_copy(update={"placement_overrides": "^output=CPU"})
    assert pick(None, base, placed) == ["placement_overrides", "kv_cache_type_k"]  # type: ignore[arg-type]
    # Removal: the start had a placement, the setup has none -- still checked.
    assert pick(None, base.model_copy(update={"placement_overrides": "^output=CPU"}), base) == ["placement_overrides"]  # type: ignore[arg-type]
    assert pick(None, base, base.model_copy(update={"threads": 8, "gpu_offload_layers": 3})) == []  # type: ignore[arg-type]
