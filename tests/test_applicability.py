"""Applicability through the declarations (C3, 9 Oct 2026).

Each ``OptionPath``'s engine option and applicability rule now come from its declaration:
the declaration names a symbolic handler id, and ``tune/applicability.py`` resolves it
through an explicit registry of the existing, specialised rules in ``tune/paths.py``.
These tests check that the outcome and the reason are the same as before for every kind
of context, that a known missing option still rejects before the rule runs, and that the
registry cannot drift from the declarations or activate anything by itself."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from inferenceos.tune import applicability, declarations as decl
from inferenceos.tune import paths as paths_module
from inferenceos.tune.paths import ALL_PATHS, PathContext

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "applicability_before_c3c.json"

#: The engine options as the old ``OptionPath`` objects held them (before C3).
OLD_OPTIONS = {
    "threads": "--threads", "reading_threads": "--threads-batch", "flash_attention": "--flash-attn",
    "prompt_batch": "--ubatch-size", "engine_switches": "", "placement": "--override-tensor",
    "kv_cache": "--cache-type-k",
}
#: The rule each old ``OptionPath`` held (the functions themselves are unchanged).
OLD_RULES = {
    "threads": "_threads_applies", "reading_threads": "_reading_threads_applies", "flash_attention": "_flash_applies",
    "prompt_batch": "_prompt_batch_applies", "engine_switches": "_switches_apply", "placement": "_placement_applies",
    "kv_cache": "_cache_applies",
}


def _matrix() -> Any:
    path = ROOT / "scripts" / "applicability_matrix.py"
    spec = importlib.util.spec_from_file_location("applicability_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["applicability_matrix"] = module
    spec.loader.exec_module(module)
    return module


def _old_check(name: str, context: PathContext) -> tuple[bool, str]:
    """``OptionPath.check`` exactly as it was written before this slice."""
    option = OLD_OPTIONS[name]
    if option and context.supported_options is not None and option not in context.supported_options:
        return False, f"this llama.cpp does not have the {option} option"
    return getattr(paths_module, OLD_RULES[name])(context)  # type: ignore[no-any-return]


def test_outcomes_and_reasons_match_the_recorded_old_implementation() -> None:
    """Up to two contexts per distinct (outcome, reason) per path, recorded from the code
    before this slice with ``scripts/applicability_matrix.py``."""
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    by_label = dict(_matrix().contexts(paths_module))
    by_name = {p.name: p for p in ALL_PATHS}
    assert {c["path"] for c in cases} == set(by_name)
    for case in cases:
        got = by_name[case["path"]].check(by_label[case["context"]])
        assert list(got) == [case["applies"], case["reason"]], case


def test_the_whole_matrix_matches_the_old_check_semantics() -> None:
    contexts = _matrix().contexts(paths_module)
    assert len(contexts) > 30000
    for label, context in contexts[::7]:
        for path in ALL_PATHS:
            assert path.check(context) == _old_check(path.name, context), (path.name, label)


def test_every_path_binds_to_its_declared_option_and_rule() -> None:
    assert applicability.registered() == {d.applicability for d in decl.DECLARATIONS if d.applicability}
    for path in ALL_PATHS:
        d = decl.declaration(path.name)
        assert path.declaration is d
        assert path.option == d.engine_option == OLD_OPTIONS[path.name]
        assert d.applicability is not None
        assert path.applies is applicability.resolve(d.applicability) is getattr(paths_module, OLD_RULES[path.name])
    layers = decl.declaration("layers")
    assert layers.applicability is None  # its own top search decides
    with pytest.raises(applicability.ApplicabilityError, match="its own search"):
        applicability.check(layers, _matrix().contexts(paths_module)[0][1])


class _Spy:
    def __init__(self, result: tuple[bool, str] = (True, "spy")) -> None:
        self.calls = 0
        self.result = result

    def __call__(self, context: PathContext) -> tuple[bool, str]:
        self.calls += 1
        return self.result


@pytest.mark.parametrize("path", ALL_PATHS, ids=lambda p: p.name)
def test_a_known_missing_option_rejects_before_the_rule_runs(path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _Spy()
    handler_id = path.declaration.applicability
    monkeypatch.setitem(applicability._HANDLERS, handler_id, spy)
    base = _matrix().contexts(paths_module)[0][1]
    option = path.option
    if option:
        missing = replace(base, supported_options=frozenset({"--something-else"}))
        assert path.check(missing) == (False, f"this llama.cpp does not have the {option} option")
        assert spy.calls == 0
        present = replace(base, supported_options=frozenset({option}))
        assert path.check(present) == (True, "spy") and spy.calls == 1
    unknown = replace(base, supported_options=None)  # unknown support: the rule decides, as before
    calls = spy.calls
    assert path.check(unknown) == (True, "spy") and spy.calls == calls + 1
    empty = replace(base, supported_options=frozenset())
    if option:
        assert path.check(empty)[0] is False and spy.calls == calls + 1
    else:  # engine switches declare no option: the option list never rejects them
        assert path.check(empty) == (True, "spy")


def test_the_registry_refuses_bad_ids_duplicates_and_unknown_names() -> None:
    with pytest.raises(applicability.ApplicabilityError, match="invalid"):
        applicability.handler("Not An Id")
    with pytest.raises(applicability.ApplicabilityError, match="already registered"):
        applicability.handler("threads.processor_work")(lambda context: (True, ""))
    same = applicability.resolve("threads.processor_work")
    assert applicability.handler("threads.processor_work")(same) is same  # re-registering the same rule is harmless
    with pytest.raises(applicability.ApplicabilityError, match="no applicability handler"):
        applicability.resolve("threads.not_registered")
    with pytest.raises(applicability.ApplicabilityError, match="no applicability handler"):
        applicability.check(replace(decl.declaration("threads"), applicability="threads.not_registered"),
                            _matrix().contexts(paths_module)[0][1])


def test_bindings_cannot_drift_from_the_declarations(monkeypatch: pytest.MonkeyPatch) -> None:
    paths_module._check_bindings(ALL_PATHS)
    original = decl.declaration

    def changed(name: str) -> Any:
        d = original(name)
        return replace(d, applicability="threads.processor_work") if name == "kv_cache" else d

    monkeypatch.setattr(paths_module, "declaration", changed)
    with pytest.raises(RuntimeError, match="registered rules no declared path uses"):
        paths_module._check_bindings(ALL_PATHS)

    # Card, setting and questions are read from the declaration now (no second copy to
    # disagree); a declared question with no workload binding is still refused.
    def other_question(name: str) -> Any:
        d = original(name)
        return replace(d, judged_on="medium") if name == "threads" else d

    monkeypatch.setattr(paths_module, "declaration", other_question)
    with pytest.raises(RuntimeError, match="no workload binding"):
        paths_module._check_bindings(ALL_PATHS)


def test_registering_a_rule_activates_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(applicability._HANDLERS, "extra.rule", lambda context: (True, "extra"))
    with pytest.raises(RuntimeError, match="extra.rule"):
        paths_module._check_bindings(ALL_PATHS)  # an unused rule is refused, never run
    assert [p.name for p in ALL_PATHS] == list(OLD_OPTIONS)
    assert [d.name for d in decl.DECLARATIONS if d.practical is not None] == ["layers", "threads", "prompt_batch"]


def test_the_registry_module_imports_no_path_code_at_runtime() -> None:
    tree = ast.parse((ROOT / "src" / "inferenceos" / "tune" / "applicability.py").read_text(encoding="utf-8"))
    runtime = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    modules = {n.module for n in runtime if isinstance(n, ast.ImportFrom)}
    assert modules == {"__future__", "collections.abc", "typing", "inferenceos.contracts.path_declaration"}
