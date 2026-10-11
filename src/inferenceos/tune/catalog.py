"""The runtime catalog: every declared tune path bound to its existing executable code
(C3, 10 Oct 2026).

``tune/declarations.py`` says what each path is. This catalog says which code runs it:

* the seven option paths are their ``OptionPath`` objects (``tune/paths.py``), each the
  runtime binding of its declaration: candidates, refinement, notes, default resolver
  and specialised effect verifier, as ordinary callables;
* the layer count is a specialised search, listed with the modes it serves and the
  existing entry points each mode calls. These are recorded entry points: the layer
  searches are not dispatched through the catalog (it is not forced into an ``OptionPath``).

The catalog is validated whenever it is built (``build_catalog``, and so on import):
exactly one runtime entry per declaration, nothing without a declaration, mode support
matching the declarations, callable entry points, and every supplied option entry checked
with the binding rules of ``tune/paths.binding_problems`` (callable hooks, registered
applicability rule, verifier given with exactly the distinct written settings it covers,
one verifier per setting). Building it starts no
engine, sampler or probe and evaluates nothing from data.

Who reads it: Unbounded's path selection (``Tuner`` through :meth:`RuntimeCatalog.by_names`)
and its final-check verification (:meth:`RuntimeCatalog.verifiers`). Practical keeps its
planner and its adapter over the declarations; its entry points are recorded here and a
test checks they are the objects it calls. Registering or listing something here makes
no path active: which paths run, in which order and mode, is decided where it was before.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from inferenceos.tune.declarations import DECLARATIONS, PathDeclaration, declaration
from inferenceos.tune.layer_search import search_layers
from inferenceos.tune.layer_top import find_layer_top
from inferenceos.tune.paths import ALL_PATHS, OptionPath, Verifier, _batch_sizes, binding_problems, verifier_bindings
from inferenceos.tune.planner import Planner

UNBOUNDED, PRACTICAL = "unbounded", "practical"
MODES = (UNBOUNDED, PRACTICAL)


class CatalogError(RuntimeError):
    """The runtime catalog does not match the declarations, or an entry is malformed."""


@dataclass(frozen=True)
class SpecializedSearch:
    """A declared path whose own search is code that does not fit an ``OptionPath``,
    with the existing entry points each mode uses (recorded, not dispatched through)."""

    name: str
    description: str
    entry_points: Mapping[str, tuple[Callable[..., object], ...]] = field(default_factory=dict)

    @property
    def declaration(self) -> PathDeclaration:
        return declaration(self.name)

    @property
    def modes(self) -> frozenset[str]:
        return frozenset(self.entry_points)


LAYERS = SpecializedSearch(
    name="layers",
    description=("graphics-card layer count: a load-only search for the top that fits, then Unbounded's whole-range "
                 "layer search, or Practical's placement steps (extremes first, then intermediate counts)"),
    entry_points={UNBOUNDED: (find_layer_top, search_layers), PRACTICAL: (find_layer_top, Planner)},
)

#: Existing functions Practical calls for the option paths it uses (its planner reads the
#: declarations through ``practical_tables``); recorded so a test can check they are the
#: same objects Practical calls.
PRACTICAL_HELPERS: Mapping[str, Callable[..., object]] = {"prompt_batch.sizes": _batch_sizes}


def declared_modes(d: PathDeclaration) -> frozenset[str]:
    """The modes a declaration says a path serves."""
    return frozenset(m for m, on in ((UNBOUNDED, d.in_unbounded), (PRACTICAL, d.practical is not None)) if on)


@dataclass(frozen=True)
class RuntimeCatalog:
    option_paths: tuple[OptionPath, ...]
    specialized: tuple[SpecializedSearch, ...]

    def names(self) -> tuple[str, ...]:
        """Every catalogued path, in declaration order."""
        known = {p.name for p in self.option_paths} | {s.name for s in self.specialized}
        return tuple(d.name for d in DECLARATIONS if d.name in known)

    def option_path(self, name: str) -> OptionPath:
        for path in self.option_paths:
            if path.name == name:
                return path
        raise CatalogError(f"no option path named {name!r}")

    def specialized_search(self, name: str) -> SpecializedSearch:
        for search in self.specialized:
            if search.name == name:
                return search
        raise CatalogError(f"no specialised search named {name!r}")

    def by_names(self, names: list[str] | tuple[str, ...]) -> list[OptionPath]:
        """The named option paths in the order named; unknown names (the layer count
        included, which is not an option path) and repeats are left out."""
        known = {path.name: path for path in self.option_paths}
        chosen: list[OptionPath] = []
        for name in names:
            key = name.strip()
            if key in known and known[key] not in chosen:
                chosen.append(known[key])
        return chosen

    def verifiers(self) -> dict[str, Verifier]:
        """Setting -> specialised verifier, from the option paths' bindings, in path order."""
        return verifier_bindings(self.option_paths)

    def modes(self, name: str) -> frozenset[str]:
        return declared_modes(declaration(name))


def build_catalog(option_paths: tuple[OptionPath, ...], specialized: tuple[SpecializedSearch, ...]) -> RuntimeCatalog:
    """Check and return the catalog (raises :class:`CatalogError` listing every problem)."""
    problems: list[str] = []
    entries = [p.name for p in option_paths] + [s.name for s in specialized]
    declared = [d.name for d in DECLARATIONS]
    for name in sorted({n for n in entries if entries.count(n) > 1}):
        problems.append(f"{name!r} has more than one runtime entry")
    for name in sorted(set(declared) - set(entries)):
        problems.append(f"declared path {name!r} has no runtime entry")
    for name in sorted(set(entries) - set(declared)):
        problems.append(f"runtime entry {name!r} has no declaration")
    for path in option_paths:
        if path.name in declared and declaration(path.name).applicability is None:
            problems.append(f"{path.name!r} is bound as an option path but declares no applicability rule")
    for search in specialized:
        if search.name not in declared:
            continue
        d = declaration(search.name)
        if d.applicability is not None:
            problems.append(f"{search.name!r} is a specialised search but declares an option-path applicability rule")
        if search.modes != declared_modes(d):
            problems.append(f"{search.name!r} serves {sorted(search.modes)}, its declaration says "
                            f"{sorted(declared_modes(d))}")
        for mode, functions in search.entry_points.items():
            if mode not in MODES:
                problems.append(f"{search.name!r}: unknown mode {mode!r}")
            if not functions or not all(callable(f) for f in functions):
                problems.append(f"{search.name!r}: mode {mode!r} needs callable entry points")
    # The option entries actually supplied are checked with the same binding rules that
    # ``tune/paths.py`` applies to its own paths at import (one set of rules).
    problems += [f"binding {problem}" for problem in binding_problems(option_paths)]
    if problems:
        raise CatalogError("runtime catalog invalid: " + "; ".join(problems))
    return RuntimeCatalog(option_paths=option_paths, specialized=specialized)


CATALOG = build_catalog(ALL_PATHS, (LAYERS,))


__all__ = [
    "CATALOG",
    "LAYERS",
    "MODES",
    "PRACTICAL",
    "PRACTICAL_HELPERS",
    "UNBOUNDED",
    "CatalogError",
    "RuntimeCatalog",
    "SpecializedSearch",
    "build_catalog",
    "declared_modes",
]
