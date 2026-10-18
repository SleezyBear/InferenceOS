"""Confirmation planning, by simulation (no model, no laptop; 9 Oct 2026).

Compares the practical search's final-check block count chosen two ways, through the
**real** ``BudgetedTune`` (planner, selection, the one fresh final check, the existing
fair test and acceptance rule, pacing, saving) on one simulated clock:

* ``current``: the selection-time count (``_blocks``: screening noise width x 1.25, a
  normal-quantile power formula, at least 4 blocks), as jobs 123 and 124 ran;
* ``planned``: ``_plan_confirmation`` -- the count for 80% planned power under the
  screening width x an uncertainty allowance (Student t critical value), used only if
  the whole check is predicted to finish by the soft target.

Both use the same worlds, seeds and acceptance test. The worlds are shapes, not models:

* ``large``: a large placement gain (Granite-MoE-like, from the search-order simulator);
* ``weak, noise as assumed``: one usable placement +4% on the short question; the true
  load-to-load spread equals the tuner's fallback (1.2%);
* ``weak, noise underestimated``: the same gain, but the true load-to-load spread is
  3.5% while the tuner assumes the 1.2% fallback (a block spread of about 5 points:
  the size job 123 showed, but the gain and costs here are invented);
* ``small``: +2% (just above Performance first's smallest worthwhile gain), noise as
  assumed;
* ``no gain``: the usable placement is truly equal to the start (winner's curse only),
  noise underestimated;
* ``regression``: the usable placement is truly 2% worse, noise underestimated;
* ``drift``: the weak gain, with persistent load-to-load drift (AR 0.85) and a slow
  steady slow-down over the run (3% per 10 minutes);
* ``unequal costs``: the weak gain, but the candidate's answers take 1.8 x as long;
* ``stopped``: the underestimated weak world with a user stop at a random time between
  150 s and 750 s (nothing may be saved after a stop; the tune must end promptly).

Reported per world and planner: how often a setup with a true gain was confirmed, wrong
accepts (a confirmed setup whose true short-question effect is not a gain), harmful
accepts (truly worse than 2%), elapsed minutes (mean and 90th percentile), runs past the
soft target and past the hard limit, the block count used, and the final check's
predicted against measured time.

    python scripts/simulate_confirmation.py [--trials 200] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import statistics
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import simulate_search_order as base  # noqa: E402
from inferenceos.tune import budgeted, fair  # noqa: E402
from inferenceos.tune.layer_top import LoadAttempt  # noqa: E402
from tests.test_tune import _settings, _tuner  # noqa: E402

TARGET_S = 900.0


@dataclass
class World:
    model: base.ModelWorld
    #: True load-to-load spread (the tuner is told ``model.between``).
    true_between: float
    #: Persistence of the load-to-load drift (the search-order simulator uses 0.3).
    persistence: float = 0.3
    #: Log speed change per simulated second (negative: slows down over the run).
    trend_per_s: float = 0.0
    #: Answer-time multiplier of a setup (layers, threads, batch).
    cost: Callable[[int, int | None, int | None], float] = lambda layers, t, b: 1.0
    #: A user stop between these simulated seconds (None: no stop).
    stop_window: tuple[float, float] | None = None
    #: The candidate whose truth defines a wrong accept (setup whose short effect <= 0).
    label: str = ""


class Machine(base.Machine):
    def __init__(self, rng: random.Random, world: World) -> None:
        super().__init__(rng, world.model)
        self.sim = world

    def new_load(self) -> None:
        self.loads += 1
        self.engine_running = True
        self.advance(self.world.load_s, phase="loading")
        rho = self.sim.persistence
        self.drift = rho * self.drift + math.sqrt(1 - rho * rho) * self.rng.gauss(0.0, self.sim.true_between)

    def respond(self, task: Any, options: Any, call: int) -> Any:
        from tests.experiment_helpers import SampleResult

        q = "long" if task.workload.workload_id.startswith("long-") else "short"
        layers = int(options.gpu_offload_layers or 0)
        mult = self.sim.cost(layers, options.threads, options.ubatch_size)
        self.advance(self.world.answer_s[q] * mult + self.world.overhead_s, phase="answering")
        if self.world.fails(layers, options.threads, options.ubatch_size) or layers > self.world.top:
            return SampleResult(fail=True)
        s, lg = self.world.effect(layers, options.threads, options.ubatch_size)
        e = (s if q == "short" else lg) + self.sim.trend_per_s * self.seconds
        within = self.world.within[q]
        speed = 20.0 * math.exp(self.drift + e + self.rng.gauss(0.0, within))
        latency = 2.0 * math.exp(-(self.drift + e + self.rng.gauss(0.0, within)))
        return SampleResult(decode_tps=speed, total_latency_s=latency)


def _llama_today(gain: float, **over: Any) -> base.ModelWorld:
    """A Llama-like model where only 27 of 28 blocks fit (job 123's measured situation),
    so the usable placement step is ``gain`` (log) on both questions."""
    fields: dict[str, Any] = dict(
        name=f"one usable placement {math.exp(gain) - 1:+.0%}", blocks=28, top=27, start=(23, None, None),
        experts=False, share=lambda L: 0.10 if L >= 27 else 0.15 + (27 - L) * 0.03,
        layers={0: (-0.2, -0.2), 1: (-0.25, -0.25), 19: (-0.065, -0.06), 23: (0.0, 0.0), 27: (gain, gain)},
        batch=base._batch_table({64: -0.2, 128: -0.1, 256: -0.03, 1024: 0.0, 2048: 0.0}),
        load_s=8.0, answer_s={"short": 3.0, "long": 19.0}, overhead_s=1.6,
        within={"short": 0.03, "long": 0.01}, between=0.012, long_tokens=1850,
        ideal=(27, None, None) if gain > 0 else None, major=("gpu_offload_layers", 27),
    )
    fields.update(over)
    return base.ModelWorld(**fields)


def worlds() -> list[World]:
    granite = next(w for w in base.WORLDS if w.name.startswith("Granite-MoE"))
    weak = _llama_today(math.log(1.04))
    return [
        World(granite, true_between=granite.between, label="large"),
        World(weak, true_between=0.012, label="weak, noise as assumed"),
        World(weak, true_between=0.035, label="weak, noise underestimated"),
        World(_llama_today(math.log(1.02)), true_between=0.012, label="small (+2%)"),
        World(_llama_today(0.0), true_between=0.035, label="no gain"),
        World(_llama_today(math.log(0.98)), true_between=0.035, label="regression (-2%)"),
        World(weak, true_between=0.025, persistence=0.85, trend_per_s=-0.03 / 600.0, label="drift"),
        World(weak, true_between=0.012, cost=lambda L, t, b: 1.8 if L == 27 else 1.0, label="unequal costs"),
        World(weak, true_between=0.035, stop_window=(150.0, 750.0), label="stopped"),
    ]


def _current_plan(self: Any, setup: Any, plan: dict[str, tuple[str, int]]) -> tuple[dict[str, tuple[str, int]], dict[str, Any]]:
    """The selection-time count, as jobs 123 and 124 ran (for comparison only)."""
    return plan, {"questions": {q: {"workload": self._workload(q).workload_id, "mode": m, "selection_blocks": n,
                                    "chosen_blocks": n} for q, (m, n) in plan.items()},
                  "predicted_check_s": round(self._final_cost(setup), 1), "reason": "current planner"}


def _double_plan(self: Any, setup: Any, plan: dict[str, tuple[str, int]]) -> tuple[dict[str, tuple[str, int]], dict[str, Any]]:
    """Strawman: twice the selection count for every check, affordable or not."""
    doubled = {q: (m, min(2 * n, budgeted.MAX_FINAL_BLOCKS)) for q, (m, n) in plan.items()}
    cost = sum(2 * (n + 1) * (self._cost(q) or 0.0) for q, (_, n) in doubled.items())
    return doubled, {"questions": {q: {"workload": self._workload(q).workload_id, "mode": m, "selection_blocks": n,
                                       "chosen_blocks": doubled[q][1]} for q, (m, n) in plan.items()},
                     "predicted_check_s": round(cost, 1), "reason": "double (strawman)"}


#: Planners compared. Only "planned" is the implementation; the others are references:
#: "current" is how jobs 123/124 ran, "allowance 2.36" a sensitivity check (the 90th
#: percentile instead of the 80th), "two looks" a predeclared group-sequential
#: alternative (looks at 4 and the planned count, alpha split evenly -- the existing
#: Unbounded procedure), validated here separately and NOT adopted, and "double" a
#: strawman (twice every check).
VARIANTS = {"current": "current", "planned": "planned", "allow236": "planned, allowance 2.36",
            "twolook": "two looks [alternative]", "double": "double [strawman]"}


@contextmanager
def variant(name: str) -> Iterator[None]:
    original_plan = budgeted.BudgetedTune._plan_confirmation
    original_allowance = budgeted.NOISE_ALLOWANCE_FALLBACK
    original_looks = fair.planned_looks
    try:
        if name == "current":
            setattr(budgeted.BudgetedTune, "_plan_confirmation", _current_plan)
        elif name == "double [strawman]":
            setattr(budgeted.BudgetedTune, "_plan_confirmation", _double_plan)
        elif name == "planned, allowance 2.36":
            budgeted.NOISE_ALLOWANCE_FALLBACK = 2.36
        elif name == "two looks [alternative]":
            def looks(settings: Any, rules: Any, *, min_blocks: int | None = None,
                      max_blocks: int | None = None) -> list[int]:
                planned: list[int] = list(original_looks(settings, rules, min_blocks=min_blocks, max_blocks=max_blocks))
                first = max(int(settings.tune_experimental_final_blocks), 2)
                if len(planned) == 1 and planned[0] > first:
                    return [first, planned[0]]
                return planned
            setattr(fair, "planned_looks", looks)
        yield
    finally:
        setattr(budgeted.BudgetedTune, "_plan_confirmation", original_plan)
        budgeted.NOISE_ALLOWANCE_FALLBACK = original_allowance
        setattr(fair, "planned_looks", original_looks)


@dataclass
class Trial:
    confirmed: bool
    accepted: bool
    true_short: float
    total: float
    blocks: int | None
    predicted_check: float | None
    measured_check: float | None
    stop_status: str
    outcome: str
    #: When the final check started (None: no final check). Equal start times do not show
    #: equal searches; ``search_fingerprint`` is what pairs are compared on.
    final_start: float | None = None
    seed: str = ""
    #: Hash of every decision made before the final check (``search_fingerprint``); None
    #: when the run raised before returning a result.
    search_fingerprint: str | None = None


def search_fingerprint(rounds: list[dict[str, Any]], starting: Any) -> dict[str, Any]:
    """Every decision of a practical tune made before its final check started: the start,
    the layer-top tries, every operation in order with its exact settings, workload and
    result, every planner step, screening and planner-log entry, the search round's
    readings, deferrals, best setup and why it stopped, and the setup selected for the
    final check with its questions, modes and selection-time block counts. Operations and
    entries from the final check onward are left out, so a planner that only changes the
    final check's block count leaves this unchanged. A run without a final check is
    fingerprinted over its whole search, with ``final_check`` None."""
    named = {r.get("round"): r for r in rounds if isinstance(r, dict)}
    final = named.get("final check")
    started = final.get("started_at_s") if final else None

    def before(entries: list[dict[str, Any]] | None, at: str) -> list[dict[str, Any]]:
        return [e for e in entries or [] if started is None or not isinstance(e, dict)
                or not isinstance(e.get(at), (int, float)) or e[at] < started]

    budget = named.get("time budget") or {}
    operations = [op for op in before(budget.get("operations"), "began_s") if op.get("kind") != "final"]
    selected = None
    if final is not None:
        plan = final.get("confirmation_plan") or {}
        questions = plan.get("questions") or {}
        selected = {
            "started_at_s": started, "setup": final.get("setup"), "settings": final.get("settings"),
            "questions": {q: {"mode": (final.get("plan") or {}).get(q, (questions.get(q) or {}).get("mode")),
                              "workload": (questions.get(q) or {}).get("workload"),
                              "selection_blocks": (questions.get(q) or {}).get("selection_blocks")}
                          for q in sorted(set(final.get("plan") or {}) | set(questions))},
        }
    return {
        "starting": starting.model_dump(mode="json") if hasattr(starting, "model_dump") else starting,
        "layer_top": named.get("layer top"),
        "practical_search": named.get("practical search"),
        "operations": operations,
        "steps": before(budget.get("steps"), "at_s"),
        "screenings": before(budget.get("screenings"), "at_s"),
        "planner_log": before(budget.get("planner_log"), "at_s"),
        "final_check": selected,
    }


def fingerprint_hash(fingerprint: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def run(world: World, seed: str, planner: str) -> Trial:
    tmp = Path(tempfile.mkdtemp(prefix="simconf-"))
    with variant(planner):
        try:
            return _run(world, seed, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _run(world: World, seed: str, tmp: Path) -> Trial:
    rng = random.Random(seed)
    model = world.model
    settings = _settings(
        tmp, optimization_mode="performance_first",
        tune_paths="threads,reading_threads,flash_attention,prompt_batch",
        tune_diagnosis=False, tune_heat_check_runs=0, tune_path_check="light",
        tune_long_question_tokens=model.long_tokens, tune_between_load_spread=model.between,
        tune_offer_tradeoffs=False, tune_experimental_budget_s=TARGET_S,
    )
    machine = Machine(rng, world)
    raw_tuner, _, _, _ = _tuner(tmp, machine.respond, answers=[True] * 12, settings=settings,
                               tuner_cls=base._SimTuner)
    tuner = cast(base._SimTuner, raw_tuner)
    tuner.world, tuner.machine_sim, tuner.finals = model, machine, []
    tuner.model = tuner.model.model_copy(update={"n_layers": model.blocks})
    tuner.context_length = 8192
    tuner.harness = base._Harness(machine)
    tuner.approver = base._TimedApprover(machine)
    tuner.say = lambda line: None
    tuner.cancel_hook = machine.stop_engine

    def try_load(layers: int, *rest: Any) -> LoadAttempt:
        machine.loads += 1
        try:
            machine.advance(model.load_s, phase="load-only try")
        except RuntimeError as exc:
            return LoadAttempt(loaded=False, fits=False, error=str(exc))
        ok = layers <= model.top
        return LoadAttempt(loaded=ok, fits=ok)

    tuner.try_load = try_load
    driver = budgeted.BudgetedTune(tuner, TARGET_S, watchdog_factory=lambda cb: base.SimWatchdog(machine, cb))
    driver.save_events = machine.save_event
    stop_at = None
    if world.stop_window is not None:
        stop_at = rng.uniform(*world.stop_window)
        machine.stop_at = stop_at
        machine.stop_cb = lambda: driver.request_cancel("simulated user stop")
    with base.simulated_clock(machine):
        try:
            result = driver.run()
        except BaseException as exc:  # noqa: BLE001 - a stop may surface as an error here
            result = None
            outcome = f"raised {type(exc).__name__}"
    final_round: dict[str, Any] | None = None
    if result is not None:
        outcome = result.outcome
        final_round = next((r for r in result.rounds if isinstance(r, dict) and r.get("round") == "final check"), None)
    accepted = bool(result is not None and result.saved_setup is not None
                    and result.final_configuration != result.starting_configuration)
    setup = base._setup_values(result.final_configuration) if result is not None else model.start
    s, _ = model.effect(*setup)
    true_short = math.exp(s) - 1 if accepted else 0.0
    blocks = predicted = measured = final_start = None
    fingerprint = None
    if result is not None:
        fingerprint = fingerprint_hash(search_fingerprint(list(result.rounds), result.starting_configuration))
    if final_round is not None:
        final_start = final_round.get("started_at_s")
        blocks = int(sum((final_round.get("blocks_per_question") or {}).values()))
        predicted = final_round.get("predicted_s")
        measured = final_round.get("took_s")
    stop_status = ""
    if stop_at is not None:
        if machine.stop_fired_phase is None:
            stop_status = "unreached"
        elif accepted and not getattr(driver, "cancel_after_publication", False):
            stop_status = "published"
        elif machine.seconds - stop_at > base.STOP_LATENCY_LIMIT_S:
            stop_status = "slow"
        else:
            stop_status = "ok"
    return Trial(confirmed=accepted and true_short > 0, accepted=accepted, true_short=true_short,
                 total=machine.seconds, blocks=blocks, predicted_check=predicted, measured_check=measured,
                 stop_status=stop_status, outcome=outcome, final_start=final_start, seed=seed,
                 search_fingerprint=fingerprint)


def _q(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(int(p * len(values)), len(values) - 1)] if values else float("nan")


def _top(outcomes: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for o in outcomes:
        key = o.split(":")[0][:60]
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:4])


def wilson(successes: int, n: int, z: float = 1.959964) -> list[float] | None:
    """95% Wilson interval of a proportion (None without a denominator)."""
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def paired(base: list[Trial], other: list[Trial]) -> dict[str, Any]:
    """Seed-by-seed comparison of two planners on the same world. The search before the
    final check should be identical (same seed, planning changes nothing before it); this
    is checked on ``search_fingerprint`` (every decision before the final check), not on
    the start time. Every claim about runs newly past the target or newly confirmed comes
    from these pairs."""
    assert [t.seed for t in base] == [t.seed for t in other]
    pairs = list(zip(base, other))
    known = [(a, b) for a, b in pairs if a.search_fingerprint is not None and b.search_fingerprint is not None]
    return {
        "pairs": len(base),
        "same_search_until_final_check": sum(a.search_fingerprint == b.search_fingerprint for a, b in known),
        "search_differs": [b.seed for a, b in known if a.search_fingerprint != b.search_fingerprint],
        "search_unknown": [b.seed for a, b in pairs if a.search_fingerprint is None or b.search_fingerprint is None],
        "both_without_final_check": sum(a.final_start is None and b.final_start is None for a, b in pairs),
        "same_search_both_without_final_check": sum(a.final_start is None and b.final_start is None
                                                    and a.search_fingerprint == b.search_fingerprint
                                                    for a, b in known),
        "same_confirmation_start_time": sum(a.final_start == b.final_start and a.final_start is not None
                                            for a, b in pairs),
        "newly_past_target": [b.seed for a, b in zip(base, other) if b.total > TARGET_S >= a.total],
        "no_longer_past_target": [b.seed for a, b in zip(base, other) if a.total > TARGET_S >= b.total],
        "newly_past_hard_limit": [b.seed for a, b in zip(base, other) if b.total > TARGET_S + 180.0 >= a.total],
        "newly_confirmed": sum(b.confirmed and not a.confirmed for a, b in zip(base, other)),
        "no_longer_confirmed": sum(a.confirmed and not b.confirmed for a, b in zip(base, other)),
        "newly_wrong_accept": sum((b.accepted and b.true_short <= 0) and not (a.accepted and a.true_short <= 0)
                                  for a, b in zip(base, other)),
        "no_longer_wrong_accept": sum((a.accepted and a.true_short <= 0) and not (b.accepted and b.true_short <= 0)
                                      for a, b in zip(base, other)),
        "fewer_blocks": sum(a.blocks is not None and b.blocks is not None and b.blocks < a.blocks
                            for a, b in zip(base, other)),
        "mean_extra_minutes": round(statistics.mean(b.total - a.total for a, b in zip(base, other)) / 60, 3),
    }


def summarize(trials: list[Trial]) -> dict[str, Any]:
    n = len(trials)
    accepted = [t for t in trials if t.accepted]
    checks = [t for t in trials if t.blocks is not None]
    forecast = [(t.measured_check or 0.0) / t.predicted_check for t in checks if t.predicted_check]
    totals = [t.total for t in trials]
    stops: dict[str, int] = {}
    for t in trials:
        if t.stop_status:
            stops[t.stop_status] = stops.get(t.stop_status, 0) + 1
    return {
        "trials": n,
        "confirmed_gain_rate": round(sum(t.confirmed for t in trials) / n, 3),
        "wrong_accepts": sum(t.accepted and t.true_short <= 0 for t in trials),
        "harmful_accepts": sum(t.accepted and t.true_short < -0.02 for t in trials),
        "final_checks": len(checks),
        "wrong_accept_rate_of_checks": round(sum(t.accepted and t.true_short <= 0 for t in trials) / len(checks), 3)
        if checks else None,
        "wrong_accepts_wilson_95_of_checks": wilson(sum(t.accepted and t.true_short <= 0 for t in trials), len(checks)),
        "mean_min": round(statistics.mean(totals) / 60, 2), "p90_min": round(_q(totals, 0.9) / 60, 2),
        "past_target": sum(t.total > TARGET_S for t in trials),
        "past_hard_limit": sum(t.total > TARGET_S + 180.0 for t in trials),
        "mean_blocks": round(statistics.mean(t.blocks for t in checks if t.blocks is not None), 2) if checks else None,
        "block_counts": sorted({t.blocks for t in checks if t.blocks is not None}),
        "check_measured_over_predicted": None if not forecast else {
            "median": round(statistics.median(forecast), 3), "max": round(max(forecast), 3)},
        "accepted": len(accepted),
        "confirmed_of_checks": round(sum(t.confirmed for t in checks) / len(checks), 3) if checks else None,
        "confirmed_of_checks_wilson_95": wilson(sum(t.confirmed for t in checks), len(checks)),
        "no_final_check": _top([t.outcome for t in trials if t.blocks is None]),
        "stops": stops,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--json", default=None)
    parser.add_argument("--only", default=None, help="run only worlds whose label contains this text")
    parser.add_argument("--per-trial", default=None, help="write every trial (seed, planner, outcome) as JSON lines")
    parser.add_argument("--variants", default="current,planned",
                        help="comma-separated planners: " + ", ".join(f"{k} ({v})" for k, v in VARIANTS.items()))
    args = parser.parse_args(argv)
    out: dict[str, Any] = {"target_s": TARGET_S, "trials": args.trials, "worlds": {}}
    for world in worlds():
        if args.only and args.only not in world.label:
            continue
        row: dict[str, Any] = {}
        by_planner: dict[str, list[Trial]] = {}
        for planner in [v.strip() for v in args.variants.split(",")]:
            name = VARIANTS[planner]
            trials = by_planner[name] = [run(world, f"{world.label}-{k}", name) for k in range(args.trials)]
            if args.per_trial:
                with open(args.per_trial, "a", encoding="utf-8") as handle:
                    for t in trials:
                        handle.write(json.dumps({"world": world.label, "planner": name} | t.__dict__) + "\n")
            r = row[name] = summarize(trials)
            print(f"{world.label:28s} {name:26s} confirmed {r['confirmed_gain_rate']:.2f} (of checks "
                  f"{r['confirmed_of_checks']}) | wrong {r['wrong_accepts']}/{r['final_checks']} checks | "
                  f"mean {r['mean_min']} p90 {r['p90_min']} min | past target {r['past_target']} hard "
                  f"{r['past_hard_limit']} | blocks {r['mean_blocks']} | forecast {r['check_measured_over_predicted']}"
                  + (f" | stops {r['stops']}" if r["stops"] else ""), flush=True)
        if "current" in by_planner:
            for name, trials in by_planner.items():
                if name != "current":
                    row[name]["paired_with_current"] = pair = paired(by_planner["current"], trials)
                    print(f"{'':28s} {'  paired with current':26s} same search {pair['same_search_until_final_check']}/"
                          f"{pair['pairs']} (no final check in both: {pair['both_without_final_check']}; unknown "
                          f"{len(pair['search_unknown'])}) | newly past target {len(pair['newly_past_target'])}, no longer "
                          f"{len(pair['no_longer_past_target'])} | newly confirmed {pair['newly_confirmed']}, no longer "
                          f"{pair['no_longer_confirmed']} | wrong accepts +{pair['newly_wrong_accept']} "
                          f"-{pair['no_longer_wrong_accept']}", flush=True)
        out["worlds"][world.label] = row
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
