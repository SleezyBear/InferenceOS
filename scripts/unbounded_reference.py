"""Full decision and execution reference of the Unbounded search, and an exact comparison
of two references (C3-6, 11 Oct 2026; no model, no laptop).

``record ROOT OUT.json`` runs the real ``Tuner`` from the code under ``ROOT`` and writes,
for every Unbounded tune it sees, the ordered trace of what it did:

* every call (with its arguments) and result of the search's own steps: the layer
  round, the facts round, diagnosis, the engine-switch check, each option-path round,
  the whole-setup check, known-answer checks, the heat check, the trade-off offer and
  saving; nested calls stay nested (``depth``);
* every engine measurement and load-only try, in order, with the exact settings and
  question (``run``, ``try_load``), every timed probe and piece-list read;
* every line the tune said to the user, in order (skip, time-limit and revisit reasons
  included), and every approval asked;
* the finished result (rounds, proposed changes, confirmation, final configuration,
  saved setup, outcome), or the exception it ended with.

Two scenario sets are recorded:

* ``sim``: the search-order worlds of ``scripts/simulate_search_order.py`` (the 160 tunes
  of the aggregate check, same seeds) plus fixed variants (effort modes, revisit limits,
  the layer path switched off, light and full path checks), all on the simulated clock;
* ``tests`` (with ``--tests``): every Unbounded tune started by the test suite, keyed by
  the test's id, so the scenarios written for diagnosis, engine switches, placement,
  memory format, layer revisits, trade-offs, quality and acceptance are covered too.

``compare A.json B.json`` checks that both files have the same scenarios (identities and
counts) and compares each trace and result field by field. Set aside, each by name:

* ``VOLATILE_KEYS``: keys holding wall-clock time, a random id or a temporary path;
* ``PATTERNS``: temporary folders, run ids, setup ids and time stamps inside strings;
* with ``--allow coordinator-trace``: the ``search actions`` round the coordinator adds
  to Unbounded reports (C3-6), and nothing else.

Every other difference is listed with its scenario and path; the exit status is then 1.

    python scripts/unbounded_reference.py record ROOT OUT.json.gz [--tests]
    python scripts/unbounded_reference.py compare BEFORE.json AFTER.json [--allow coordinator-trace]
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import gzip
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

#: Keys whose values are wall-clock time, a random id or a temporary path.
VOLATILE_KEYS: dict[str, str] = {
    "run_id": "made from the wall clock and a random id",
    "setup_id": "a random id (uuid4)",
    "created_at": "wall-clock save time",
    "checked_at": "wall-clock machine check time",
    "started_at": "wall-clock start",
    "finished_at": "wall-clock end",
    "modified_ns": "the model file's modification time",
    "tune_report": "report file named after the run id",
    "seconds": "wall-clock duration of a timed step (real clock in the tests)",
    "elapsed_s": "wall-clock duration of a measurement (real clock in the tests)",
}

#: Keys holding a deadline on the real clock: compared by kind (none, set, unlimited).
DEADLINE_KEYS = ("deadline",)

#: Strings rewritten before comparison.
PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("temporary folder", re.compile(
        r"(?:[A-Za-z]:)?(?:[\\/][^\s'\"\\/]+)*?[\\/](?:tmp|Temp)[\\/]"
        r"(?:pytest-of-[^\\/\s'\"]+[\\/]pytest-\d+|sim[a-z]+-[A-Za-z0-9_]+|tmp[A-Za-z0-9_]+)"), "<tmp>"),
    ("run id", re.compile(r"tune-\d{8}T\d{6}Z-[0-9a-f]{6}"), "<run-id>"),
    ("setup id", re.compile(r"setup-[0-9a-f]{10}"), "<setup-id>"),
    ("measurement id", re.compile(r"meas-[0-9a-f]{10}"), "<measurement-id>"),
    ("answer id", re.compile(r"run-\d{4}-[0-9a-f]{6}"), "<answer-id>"),
    ("time stamp", re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?"), "<time>"),
]

#: The round the coordinator adds to Unbounded reports (C3-6), set aside only with
#: ``--allow coordinator-trace``.
COORDINATOR_ROUND = "search actions"

#: Search steps on ``Tuner`` that are traced (call, arguments, result, nesting).
TRACED = (
    "_layer_round", "_facts_round", "_diagnose", "_check_switches", "_path_round",
    "_check_whole_setup", "_known_answers", "_run_heat_check", "_offer_tradeoff", "_save",
)

#: The fixed variants of the sim set: name -> setting overrides.
VARIANTS: dict[str, dict[str, Any]] = {
    "balanced": {"optimization_mode": "balanced"},
    "minimize": {"optimization_mode": "minimize_cost"},
    "revisits-0": {"tune_path_revisits": 0},
    "revisits-1": {"tune_path_revisits": 1},
    "no-layer-path": {"tune_layer_path": False},
    "full-checks": {"tune_path_check": "full"},
    "all-paths": {"tune_paths": "threads,reading_threads,flash_attention,prompt_batch,engine_switches,placement,kv_cache"},
}
VARIANT_TRIALS = 3


# --------------------------------------------------------------------------- #
# Summaries                                                                    #
# --------------------------------------------------------------------------- #
def summary(value: Any, depth: int = 0) -> Any:
    """A JSON value describing ``value`` exactly enough to compare decisions."""
    if depth > 12:
        return "<deep>"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return round(value, 9) if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): summary(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [summary(v, depth + 1) for v in value]
        return sorted(items, key=json.dumps) if isinstance(value, (set, frozenset)) else items
    name = type(value).__name__
    if name == "OptionPath":
        return {"path": value.name}
    if name == "BenchmarkWorkload":
        return {"workload": value.workload_id}
    if name == "DisturbanceWatch":
        return "<watch>"
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return {"type": name, **summary(dump(mode="json", exclude_none=True), depth + 1)}
        except Exception:  # noqa: BLE001 - a model that cannot dump is named only
            return {"type": name}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)
                  if not f.name.startswith("_") and getattr(value, f.name) is not None}
        return {"type": name, **summary(fields, depth + 1)}
    if callable(value):
        return f"<callable {getattr(value, '__qualname__', name)}>"
    return f"<{name}>"


def _deadline(value: Any) -> str | None:
    """A deadline as its kind: real-clock deadlines differ between runs."""
    if value is None:
        return None
    return "inf" if value == float("inf") else "set"


def _args(method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The arguments of a traced step; deadlines are reduced to their kind."""
    names = {
        "_layer_round": ("current", "start_measurement", "watch", "deadline"),
        "_facts_round": ("current",),
        "_diagnose": ("current", "watch", "label", "deadline"),
        "_check_switches": ("current", "depth", "deadline"),
        "_path_round": ("path", "current", "watch", "deadline"),
        "_check_whole_setup": ("start", "setup", "changes", "watch", "deadline"),
        "_run_heat_check": ("final", "watch", "runs"),
        "_offer_tradeoff": ("result",),
    }.get(method)
    named = dict(zip(names, args)) if names else {f"arg{i}": a for i, a in enumerate(args)}
    named.update(kwargs)
    out: dict[str, Any] = {}
    for key, value in named.items():
        if key in ("watch", "start_measurement", "result"):
            continue
        out[key] = _deadline(value) if "deadline" in key or key == "limit" else summary(value)
    return out


# --------------------------------------------------------------------------- #
# The tracer                                                                   #
# --------------------------------------------------------------------------- #
class Tracer:
    """Collects one trace per Unbounded ``Tuner.run``."""

    def __init__(self) -> None:
        self.traces: list[dict[str, Any]] = []
        self.current: list[dict[str, Any]] | None = None
        self.depth = 0

    def event(self, kind: str, **fields: Any) -> None:
        if self.current is not None:
            self.current.append({"kind": kind, "depth": self.depth, **fields})

    def install(self, tuner_cls: type) -> Callable[[], None]:
        originals: dict[str, Any] = {}
        tracer = self

        run = getattr(tuner_cls, "run")
        originals["run"] = run

        @functools.wraps(run)
        def traced_run(self: Any) -> Any:
            budget = float(getattr(self.settings, "tune_experimental_budget_s", 0.0) or 0.0)
            if budget > 0 or tracer.current is not None:
                return run(self)
            events: list[dict[str, Any]] = []
            tracer.current, tracer.depth = events, 0
            restore = tracer._wrap_instance(self)
            entry: dict[str, Any] = {"events": events}
            try:
                result = run(self)
            except BaseException as exc:
                entry["raised"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            else:
                entry["result"] = summary(json.loads(result.to_json()))
                return result
            finally:
                restore()
                tracer.current = None
                tracer.traces.append(entry)

        setattr(tuner_cls, "run", traced_run)
        for name in TRACED:
            original = getattr(tuner_cls, name)
            originals[name] = original
            setattr(tuner_cls, name, self._step(name, original))

        def uninstall() -> None:
            for key, value in originals.items():
                setattr(tuner_cls, key, value)

        return uninstall

    def _step(self, name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        tracer = self

        @functools.wraps(original)
        def traced(self: Any, *args: Any, **kwargs: Any) -> Any:
            if tracer.current is None:
                return original(self, *args, **kwargs)
            tracer.event("call", step=name, args=_args(name, args, kwargs))
            tracer.depth += 1
            try:
                result = original(self, *args, **kwargs)
            except BaseException as exc:
                tracer.depth -= 1
                tracer.event("raise", step=name, error=type(exc).__name__, message=str(exc))
                raise
            tracer.depth -= 1
            shown = summary(json.loads(result.to_json())) if name == "_offer_tradeoff" else summary(result)
            tracer.event("return", step=name, result=shown)
            return result

        return traced

    def _wrap_instance(self, tuner: Any) -> Callable[[], None]:
        """Record what the tune says, measures, loads, probes and asks, in order."""
        tracer = self
        restore: list[Callable[[], None]] = []

        def wrap_attr(owner: Any, attr: str, make: Callable[[Callable[..., Any]], Callable[..., Any]]) -> None:
            if owner is None:
                return
            had = attr in getattr(owner, "__dict__", {})
            original = getattr(owner, attr, None)
            if original is None:
                return
            try:
                setattr(owner, attr, make(original))
            except (AttributeError, TypeError):
                return

            def undo() -> None:
                if had:
                    setattr(owner, attr, original)
                else:
                    try:
                        delattr(owner, attr)
                    except AttributeError:
                        setattr(owner, attr, original)

            restore.append(undo)

        def say(original: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(original)
            def wrapped(line: str) -> Any:
                tracer.event("say", line=line)
                return original(line)
            return wrapped

        def run_once(original: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(original)
            def wrapped(task: Any, **kwargs: Any) -> Any:
                tracer.event("run", workload=task.workload.workload_id,
                             **{k: summary(v) for k, v in kwargs.items()})
                return original(task, **kwargs)
            return wrapped

        def plain(kind: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def make(original: Callable[..., Any]) -> Callable[..., Any]:
                # ``wraps`` keeps the signature: the tuner inspects ``try_load``'s.
                @functools.wraps(original)
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    tracer.event(kind, args=summary(list(args)), kwargs=summary(kwargs))
                    return original(*args, **kwargs)
                return wrapped
            return make

        wrap_attr(tuner, "say", say)
        wrap_attr(tuner.harness, "run_once", run_once)
        for attr, kind in (("try_load", "try_load"), ("probe_runner", "probe"),
                           ("piece_lines_reader", "piece_lines"), ("memory_speed_reader", "memory_speed")):
            if getattr(tuner, attr, None) is not None:
                wrap_attr(tuner, attr, plain(kind))
        wrap_attr(tuner.approver, "decide", plain("approval"))

        def undo_all() -> None:
            for undo in reversed(restore):
                undo()

        return undo_all


# --------------------------------------------------------------------------- #
# Scenarios                                                                    #
# --------------------------------------------------------------------------- #
def _sim_one(m: Any, world: Any, rng: random.Random, overrides: dict[str, Any]) -> None:
    """``simulate_search_order.one`` for Unbounded, with setting overrides."""
    from inferenceos.tune.layer_top import LoadAttempt

    tmp = Path(tempfile.mkdtemp(prefix="simorder-"))
    base: dict[str, Any] = {
        "optimization_mode": "performance_first",
        "tune_paths": "threads,reading_threads,flash_attention,prompt_batch",
        "tune_diagnosis": False, "tune_heat_check_runs": 0, "tune_path_check": "light",
        "tune_long_question_tokens": world.long_tokens, "tune_between_load_spread": world.between,
        "tune_offer_tradeoffs": False,
    }
    base.update(overrides)
    settings = m._settings(tmp, **base)
    machine = m.Machine(rng, world)
    tuner, _, _, _ = m._tuner(tmp, machine.respond, answers=[True] * 12, settings=settings, tuner_cls=m._SimTuner)
    tuner.world, tuner.machine_sim, tuner.finals = world, machine, []
    tuner.model = tuner.model.model_copy(update={"n_layers": world.blocks})
    tuner.context_length = 8192
    tuner.harness = m._Harness(machine)
    tuner.approver = m._TimedApprover(machine)
    tuner.say = lambda line: None
    tuner.cancel_hook = machine.stop_engine

    def try_load(layers: int, *rest: Any) -> Any:
        machine.loads += 1
        try:
            machine.advance(world.load_s, phase="load-only try")
        except RuntimeError as exc:
            return LoadAttempt(loaded=False, fits=False, error=str(exc))
        ok = layers <= world.top
        return LoadAttempt(loaded=ok, fits=ok)

    tuner.try_load = try_load
    with m.simulated_clock(machine):
        try:
            tuner.run()
        except Exception:  # noqa: BLE001 - the trace keeps the exception
            pass


def _record_sim(tracer: Tracer) -> list[dict[str, Any]]:
    import simulate_search_order as m

    out: list[dict[str, Any]] = []
    for w in m.WORLDS:
        rng = random.Random(f"1-{w.name}-unbounded")
        for i in range(20):
            before = len(tracer.traces)
            _sim_one(m, w, rng, {})
            out.append({"scenario": f"sim/{w.name}/{i}", **_only(tracer, before)})
    for variant, overrides in VARIANTS.items():
        for w in m.WORLDS:
            rng = random.Random(f"ref-{variant}-{w.name}")
            for i in range(VARIANT_TRIALS):
                before = len(tracer.traces)
                _sim_one(m, w, rng, overrides)
                out.append({"scenario": f"sim-{variant}/{w.name}/{i}", **_only(tracer, before)})
    return out


def _only(tracer: Tracer, before: int) -> dict[str, Any]:
    new = tracer.traces[before:]
    if len(new) != 1:
        return {"traces": new}
    return new[0]


class _Plugin:
    """Keys each Unbounded tune started by a test with the test's id."""

    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self.by_test: dict[str, list[dict[str, Any]]] = {}
        self.outcomes: dict[str, str] = {}
        self.node: str | None = None
        self.start = 0

    def pytest_runtest_setup(self, item: Any) -> None:
        self.node, self.start = item.nodeid, len(self.tracer.traces)

    def pytest_runtest_teardown(self, item: Any) -> None:
        new = self.tracer.traces[self.start:]
        if new:
            self.by_test[item.nodeid] = new

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when == "call" or report.outcome != "passed":
            self.outcomes[report.nodeid] = report.outcome


def _record_tests(tracer: Tracer, root: Path, extra: list[str]) -> list[dict[str, Any]]:
    import pytest

    plugin = _Plugin(tracer)
    args = [str(root / "tests"), "-q", "-p", "no:cacheprovider", "-x" if "-x" in extra else "--maxfail=100000",
            "-o", "addopts=", *[a for a in extra if a != "-x"]]
    code = int(pytest.main(args, plugins=[plugin]))
    if code not in (0, 1):  # interrupted, internal error, usage error or no tests
        raise SystemExit(f"the test suite did not run to the end (pytest exit code {code}); nothing recorded")
    out: list[dict[str, Any]] = []
    for node in sorted(plugin.by_test):
        for i, trace in enumerate(plugin.by_test[node]):
            out.append({"scenario": f"test/{node}/{i}", "test_outcome": plugin.outcomes.get(node), **trace})
    return out


def _inner_record(root: Path, out_path: Path, tests: bool, extra: list[str]) -> int:
    from inferenceos.tune.pipeline import Tuner

    tracer = Tracer()
    uninstall = tracer.install(Tuner)
    try:
        scenarios = _record_sim(tracer)
        if tests:
            scenarios += _record_tests(tracer, root, extra)
    finally:
        uninstall()
    data = {"_meta": {"root": str(root), "tests": tests, "scenarios": len(scenarios)},
            "scenarios": [_normalise(s) for s in scenarios]}
    _write(out_path, data)
    print(f"{len(scenarios)} scenarios recorded to {out_path}")
    return 0


def _write(path: Path, data: Any) -> None:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def _read(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: (f"<{k}>" if k in VOLATILE_KEYS
                    else _deadline(v) if k in DEADLINE_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool)
                    else _normalise(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    if isinstance(value, str):
        for _, pattern, replacement in PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    return value


# --------------------------------------------------------------------------- #
# Comparison                                                                   #
# --------------------------------------------------------------------------- #
def _strip_coordinator(value: Any) -> Any:
    """``value`` without the coordinator's round, wherever a report's rounds appear (the
    tune's result, and results returned inside the trace, such as the trade-off offer's)."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "rounds" and isinstance(item, list):
                item = [r for r in item if not (isinstance(r, dict) and r.get("round") == COORDINATOR_ROUND)]
            out[key] = _strip_coordinator(item)
        return out
    if isinstance(value, list):
        return [_strip_coordinator(item) for item in value]
    return value


def _diff(a: Any, b: Any, path: str, out: list[str], limit: int = 20) -> None:
    if len(out) >= limit:
        return
    if type(a) is not type(b):
        out.append(f"{path}: {json.dumps(a)[:200]} != {json.dumps(b)[:200]}")
    elif isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}.{key}: only in {'after' if key not in a else 'before'}")
            else:
                _diff(a[key], b[key], f"{path}.{key}", out, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items != {len(b)} items")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        out.append(f"{path}: {json.dumps(a)[:200]} != {json.dumps(b)[:200]}")


def compare(before: Path, after: Path, allow: list[str]) -> int:
    a = _read(before)["scenarios"]
    b = _read(after)["scenarios"]
    names_a, names_b = [s["scenario"] for s in a], [s["scenario"] for s in b]
    if names_a != names_b:
        missing = sorted(set(names_a) - set(names_b))
        added = sorted(set(names_b) - set(names_a))
        print(f"scenario identities differ: {len(a)} before, {len(b)} after; "
              f"missing {missing[:5]}, added {added[:5]}")
        return 1
    differing = 0
    events = 0
    for x, y in zip(a, b):
        x, y = _normalise(x), _normalise(y)
        if "coordinator-trace" in allow:
            x, y = _strip_coordinator(x), _strip_coordinator(y)
        events += len(x.get("events", []))
        problems: list[str] = []
        _diff(x, y, "", problems)
        if problems:
            differing += 1
            if differing <= 10:
                print(f"{x['scenario']}:")
                for line in problems:
                    print(f"  {line}")
    print(f"{len(a)} scenarios, {events} trace events; {differing} differ")
    return 1 if differing else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("root", type=Path)
    rec.add_argument("out", type=Path)
    rec.add_argument("--tests", action="store_true")
    rec.add_argument("--pytest-arg", action="append", default=[])
    inner = sub.add_parser("_inner")
    inner.add_argument("root", type=Path)
    inner.add_argument("out", type=Path)
    inner.add_argument("--tests", action="store_true")
    inner.add_argument("--pytest-arg", action="append", default=[])
    cmp = sub.add_parser("compare")
    cmp.add_argument("before", type=Path)
    cmp.add_argument("after", type=Path)
    cmp.add_argument("--allow", action="append", default=[], choices=["coordinator-trace"])
    args = parser.parse_args(argv)
    if args.cmd == "compare":
        return compare(args.before, args.after, args.allow)
    root = args.root.resolve()
    if args.cmd == "record":
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root), str(root / "scripts")])
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "_inner", str(root),
                   str(args.out.resolve()), *(["--tests"] if args.tests else []),
                   *[f"--pytest-arg={a}" for a in args.pytest_arg]]
        return subprocess.run(command, cwd=root, env=env, check=False).returncode
    sys.path[:0] = [str(root / "src"), str(root), str(root / "scripts")]
    return _inner_record(root, args.out, args.tests, list(args.pytest_arg))


if __name__ == "__main__":
    raise SystemExit(main())
