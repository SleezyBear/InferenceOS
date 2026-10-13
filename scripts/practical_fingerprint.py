"""Behaviour fingerprint of the practical search over simulated worlds, and an exact
comparison of two fingerprints (no model, no laptop).

``record ROOT OUT.json`` runs the practical search (the real ``BudgetedTune``) from the
code under ``ROOT`` over a fixed set of simulated tunes and writes every full result:

* the search-order worlds at 15 and 45 minutes, 6 seeds each;
* the confirmation worlds, 15 seeds each.

``compare A.json B.json`` compares two such files field by field. Only the fields in
``WALL_CLOCK`` (time stamps, random ids) and ``NORMALISED`` (paths inside the simulators'
own scratch folders, compared from that folder on, on POSIX or Windows) are set aside,
each by its exact path. Strict comparison is the default and is the check for a
same-platform refactor. ``--cross-platform`` (for a recording made on another system)
reports differences where both values are floats apart, with their largest size, and
never calls them identical; every other field must still match exactly. ``--allow finding1-reporting`` also sets aside the confirmation plan's reporting
fields added by the 9 Oct 2026 reporting fix, by exact path, and lets
``blocks_for_power`` change only from the block cap to null where the plan now records
the target as unattainable. ``--allow plan-format-version`` sets aside only the
addition of the plan's ``format_version`` field. Nothing else is ignored: every other difference is listed
by path and the exit status is 1. ``--allow coordinator-trace`` permits only an added
Practical ``search actions`` round, never changes to a pre-existing round.
Round order and multiplicity are compared separately from the per-round field
diagnostics; naming rounds does not permit reordering or dropping duplicates.

    python scripts/practical_fingerprint.py record . OUT.json
    python scripts/practical_fingerprint.py compare BEFORE.json AFTER.json [--allow finding1-reporting]
    python scripts/practical_fingerprint.py compare LINUX.json WINDOWS.json --cross-platform
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

#: A path inside one of the simulators' own scratch folders (``tempfile.mkdtemp`` with the
#: prefix ``simorder-`` or ``simconf-``), on POSIX or Windows: the root before it, the
#: folder itself and the model-relative rest.
TEMP_PATH = re.compile(r"^(?P<root>.*?)[\\/](?P<folder>sim(?:order|conf)-[A-Za-z0-9_]+)(?P<rest>(?:[\\/][^\0]*)?)$")
RUN_ID = re.compile(r"tune-practical-\d+(?:-[0-9a-f]{32})?")
META = "_meta"

#: Fields that carry wall-clock time, a random id or a temporary path, by exact path.
#: ``rounds[<name>]`` addresses a round by its ``round`` name; ``[]`` is every list item.
WALL_CLOCK: dict[str, str] = {
    "run_id": "the generated run id contains a wall clock and optional UUID",
    "started_at": "wall-clock start",
    "finished_at": "wall-clock end",
    "machine.checked_at": "wall-clock machine check time",
    "saved_setup.created_at": "wall-clock save time",
    "saved_setup.machine.checked_at": "wall-clock machine check time",
    "saved_setup.model.modified_ns": "the simulated model file's modification time",
    "saved_setup.setup_id": "a random id (uuid4)",
    "rounds[memory-safety check].readings.at": "wall-clock reading time",
}

#: Strings compared after replacing the temporary directory or the run id.
NORMALISED: dict[str, str] = {
    "model_path": "temporary directory",
    "saved_setup.model.path": "temporary directory",
    "saved_setup.tune_report": "report file named after the run id",
}

FINDING1_PLAN = "rounds[final check].confirmation_plan"
FINDING1_REPORTING: dict[str, str] = {
    f"{FINDING1_PLAN}.reason": "plan reason reworded to name the limit (reporting fix, 9 Oct)",
    f"{FINDING1_PLAN}.rule": "plan rule reworded (reporting fix, 9 Oct)",
    f"{FINDING1_PLAN}.selection_limitation": "new reporting field (9 Oct)",
    f"{FINDING1_PLAN}.questions.*.planned_power_limited_by": "new reporting field (9 Oct)",
    f"{FINDING1_PLAN}.questions.*.predicted.power_target_attainable": "new reporting field (9 Oct)",
    f"{FINDING1_PLAN}.questions.*.predicted.block_cap": "new reporting field (9 Oct)",
    f"{FINDING1_PLAN}.questions.*.predicted.power_at_cap": "new reporting field (9 Oct)",
}
FINDING1_BLOCKS_FOR_POWER = f"{FINDING1_PLAN}.questions.*.predicted.blocks_for_power"
#: ``--allow plan-format-version``: the version field new plans record (9 Oct 2026, late).
#: Only its addition is set aside; any other change to the plan is still a difference.
PLAN_FORMAT_VERSION_PATH = f"{FINDING1_PLAN}.format_version"


def record(root: Path, out: Path) -> int:
    root = root.resolve()
    sys.path[:0] = [str(root / "scripts"), str(root / "src"), str(root)]
    import inferenceos.tune.budgeted as budgeted
    if not Path(budgeted.__file__).resolve().is_relative_to(root):
        raise SystemExit(f"imported {budgeted.__file__}, not the code under {root}")
    import simulate_confirmation as sc
    import simulate_search_order as order

    last: dict[str, Any] = {}
    original = budgeted.BudgetedTune.run

    def run(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        last["r"] = result
        return result

    setattr(budgeted.BudgetedTune, "run", run)
    results: dict[str, Any] = {}
    for target in (900.0, 2700.0):
        for world in order.WORLDS:
            for k in range(6):
                last.clear()
                order.one(world, random.Random(f"fp-{world.name}-{k}"), target_s=target)
                results[f"order|{target:.0f}|{world.name}|{k}"] = last["r"].model_dump(mode="json")
    for cworld in sc.worlds():
        for k in range(15):
            last.clear()
            tmp = Path(tempfile.mkdtemp(prefix="simconf-"))
            try:
                sc._run(cworld, f"fp-{cworld.label}-{k}", tmp)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            results[f"conf|{cworld.label}|{k}"] = last["r"].model_dump(mode="json") if "r" in last else None
    results[META] = {"temp_root": tempfile.gettempdir(), "platform": sys.platform,
                     "python": sys.version.split()[0]}
    out.write_text(json.dumps(results, sort_keys=True), encoding="utf-8")
    print(f"{len(results) - 1} tunes recorded from {budgeted.__file__}")
    return 0


def _rounds_by_name(result: dict[str, Any]) -> dict[str, Any]:
    named = dict(result)
    rounds = result.get("rounds") or []
    names = [r.get("round") for r in rounds if isinstance(r, dict)]
    if len(names) != len(set(names)) or len(names) != len(rounds):
        return named
    named["rounds"] = {f"[{r['round']}]": r for r in rounds}
    return named


def _pattern(path: str) -> str:
    return re.sub(r"questions\.[^.]+\.", "questions.*.", path)


def _differences(a: Any, b: Any, path: str, out: list[tuple[str, Any, Any]]) -> None:
    if type(a) is not type(b):
        out.append((path, a, b))
    elif isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{path}{key}" if path.endswith("rounds") and key.startswith("[") else (
                f"{path}.{key}" if path else key)
            if key not in a or key not in b:
                out.append((sub, a.get(key, "<absent>"), b.get(key, "<absent>")))
            else:
                _differences(a[key], b[key], sub, out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append((f"{path}[len]", len(a), len(b)))
        for x, y in zip(a, b):
            _differences(x, y, f"{path}[]", out)
    elif a != b:
        out.append((path, a, b))


def scratch_relative(value: str, temp_root: str | None = None) -> str | None:
    """``<tmp>/models/sample.gguf`` for a path inside a simulator scratch folder (POSIX or
    Windows separators), keeping everything after the folder; None when the value is not
    such a path, or when ``temp_root`` is known and the path is not under it."""
    match = TEMP_PATH.match(value)
    if match is None:
        return None
    if temp_root is not None:
        def norm(text: str) -> str:
            return text.replace("\\", "/").rstrip("/").lower()
        if norm(match["root"]) != norm(temp_root):
            return None
    return "<tmp>" + match["rest"].replace("\\", "/")


def _normalised(path: str, value: Any, temp_root: str | None = None) -> Any:
    if path in NORMALISED and isinstance(value, str):
        relative = scratch_relative(value, temp_root)
        return RUN_ID.sub("tune-practical-<id>", relative if relative is not None else value)
    return value


def classify(a: Any, b: Any, allow: set[str], roots: tuple[str | None, str | None] = (None, None),
             numeric: bool = False) -> tuple[list[tuple[str, Any, Any]], dict[str, int], list[float]]:
    """The unexplained differences between two tune results, how many of each set-aside
    field differed, and (only when ``numeric``) the sizes of float-only differences, which
    are then reported apart instead of being listed as unexplained."""
    if a is None or b is None:
        return ([] if a == b else [("<result>", a is None, b is None)]), {}, []
    raw: list[tuple[str, Any, Any]] = []
    # Name-based diagnostics must not discard the report's ordered execution stages.
    old_rounds, new_rounds = a.get("rounds") or [], b.get("rounds") or []
    def identity(round_: Any) -> Any:
        return round_.get("round") if isinstance(round_, dict) else round_
    old_order = [identity(r) for r in old_rounds]
    new_order = [identity(r) for r in new_rounds]
    inserted = [r for r in new_rounds if isinstance(r, dict) and r.get("round") == "search actions"]
    if ("coordinator-trace" in allow and "search actions" not in old_order
            and len(inserted) == 1 and inserted[0].get("mode") == "practical"):
        new_order = [name for name in new_order if name != "search actions"]
    if old_order != new_order:
        raw.append(("rounds[order]", old_order, new_order))
    _differences(_rounds_by_name(a), _rounds_by_name(b), "", raw)
    unexplained: list[tuple[str, Any, Any]] = []
    set_aside: dict[str, int] = {}
    floats: list[float] = []
    for path, x, y in raw:
        pattern = _pattern(path)
        if path in WALL_CLOCK:
            set_aside[path] = set_aside.get(path, 0) + 1
        elif path in NORMALISED and _normalised(path, x, roots[0]) == _normalised(path, y, roots[1]):
            set_aside[path] = set_aside.get(path, 0) + 1
        elif ("plan-format-version" in allow and path == PLAN_FORMAT_VERSION_PATH and x == "<absent>"
              and y == "confirmation-plan-3"):
            set_aside[path + " (added)"] = set_aside.get(path + " (added)", 0) + 1
        elif ("coordinator-trace" in allow and path == "rounds[search actions]" and x == "<absent>"
              and isinstance(y, dict) and y.get("round") == "search actions" and y.get("mode") == "practical"):
            set_aside[path + " (added)"] = set_aside.get(path + " (added)", 0) + 1
        elif "finding1-reporting" in allow and pattern in FINDING1_REPORTING:
            set_aside[pattern] = set_aside.get(pattern, 0) + 1
        elif ("finding1-reporting" in allow and pattern == FINDING1_BLOCKS_FOR_POWER
              and _cap_became_null(b, path, x, y)):
            set_aside[pattern + " (cap -> null, unattainable)"] = set_aside.get(pattern, 0) + 1
        elif numeric and type(x) is float and type(y) is float:
            floats.append(abs(x - y))
        else:
            unexplained.append((path, x, y))
    return unexplained, set_aside, floats


def _cap_became_null(after: dict[str, Any], path: str, before: Any, now: Any) -> bool:
    if now is not None:
        return False
    question = path.split(".questions.")[1].split(".")[0]
    final: dict[str, Any] = next((r for r in after.get("rounds") or [] if r.get("round") == "final check"), {})
    predicted = (((final.get("confirmation_plan") or {}).get("questions") or {}).get(question) or {}).get("predicted")
    return bool(predicted and predicted.get("power_target_attainable") is False
                and before == predicted.get("block_cap"))


def compare(a_path: Path, b_path: Path, allow: set[str], cross_platform: bool = False) -> int:
    """Strict by default: every difference outside the set-aside fields is reported and the
    exit status is 1. With ``cross_platform``, float-only differences (both values floats)
    are counted and reported apart with their largest size, and do not fail the run; every
    other field, including every discrete decision, must still be exactly equal."""
    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    meta_a, meta_b = a.pop(META, None) or {}, b.pop(META, None) or {}
    if set(a) != set(b):
        print("different tune sets:", sorted(set(a) ^ set(b))[:5])
        return 1
    roots = (meta_a.get("temp_root"), meta_b.get("temp_root"))
    print(f"before: {meta_a.get('platform', 'unrecorded platform')}; after: {meta_b.get('platform', 'unrecorded platform')}")
    identical = numeric_only = 0
    totals: dict[str, int] = {}
    floats: list[float] = []
    bad: dict[str, list[tuple[str, Any, Any]]] = {}
    for key in sorted(a):
        unexplained, set_aside, tiny = classify(a[key], b[key], allow, roots, numeric=cross_platform)
        for field, n in set_aside.items():
            totals[field] = totals.get(field, 0) + n
        floats += tiny
        if unexplained:
            bad[key] = unexplained
        elif tiny:
            numeric_only += 1
        else:
            identical += 1
    print(f"{identical} of {len(a)} tunes identical apart from the set-aside fields")
    if cross_platform:
        print(f"{numeric_only} of {len(a)} tunes differ ONLY in floating-point values: {len(floats)} values, "
              f"largest difference {max(floats) if floats else 0.0!r} (not exact equality; every other field, "
              "including every decision, is equal in these tunes)")
    for field, n in sorted(totals.items()):
        print(f"  set aside: {field} ({n} differences)")
    for key, diffs in list(bad.items())[:10]:
        print(f"  DIFFERS {key}:")
        for path, x, y in diffs[:5]:
            print(f"    {path}: {str(x)[:80]!s} -> {str(y)[:80]!s}")
    return 0 if not bad else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("root", type=Path)
    rec.add_argument("out", type=Path)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("before", type=Path)
    cmp_.add_argument("after", type=Path)
    cmp_.add_argument("--allow", action="append", default=[], choices=["finding1-reporting", "plan-format-version", "coordinator-trace"])
    cmp_.add_argument("--cross-platform", action="store_true",
                      help="report float-only differences apart (for files recorded on different systems)")
    args = parser.parse_args(argv)
    if args.command == "record":
        return record(args.root, args.out)
    return compare(args.before, args.after, set(args.allow), args.cross_platform)


if __name__ == "__main__":
    raise SystemExit(main())
