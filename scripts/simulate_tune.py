"""The whole tune selection procedure, by simulation (no model needed).

Owner (4 Oct 2026): prove the complete selection procedure by simulation before another
expensive full tune. This runs the real ``Tuner`` (every path, screening, looking again,
the per-path checks, the final check and trade-off offers) many times against a scripted
laptop whose true effects are known.

Corrected after review 5 (5 Oct 2026):

* the truth is scored the way production scores a change (relative change of each goal
  measure, weighted by the goal), not a log approximation;
* discovery is split into **all** real changes recovered and **part** of them;
* the cost is also given as **simulated time** (loads and answers at laptop-like times),
  not only fresh loads;
* more worlds: losses on either side of the 2% limit, a harmful combination of two good
  changes, more candidates (four paths), heavy-tailed noise, a candidate that heats the
  laptop, the time limit running out, and trade-offs the user says yes to that fail on
  answers or paging;
* zero counts come with their one-sided 95% upper bound.

Corrected after review 6 (6 Oct 2026): one controlled clock. Every deadline the real
``Tuner`` checks reads the simulated laptop's clock (loads and answers at laptop-like
times), so a time budget is a budget of simulated seconds, and the report shows by how
much the simulated time went past it (a measurement already started finishes).

Per row: harmful (truly loses more than 2% on a question without the user's yes),
not worthwhile, all / part found, average true gain, fresh loads, simulated minutes.

    python scripts/simulate_tune.py [--trials 100] [--seed 1] [--effort performance_first]
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferenceos.optimization.goals import improvement  # noqa: E402
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask  # noqa: E402
from inferenceos.tune.approval import ScriptedChangeApprover  # noqa: E402
from tests.experiment_helpers import SampleResult, ScriptedHarness  # noqa: E402
from tests.test_tune import _settings, _tuner  # noqa: E402

Key = tuple[str, object]
#: {question: (log change of writing speed, log change of first-word speed)}; + is better.
PerQuestion = dict[str, tuple[float, float]]

#: Laptop-like times (Gemma 3 1B, Oct 2026): a fresh load, and one answer per question
#: including the per-answer telemetry cost.
LOAD_S = 17.0
ANSWER_S = {"short": 1.9, "long": 11.7}


@dataclass
class World:
    name: str
    effects: dict[Key, PerQuestion] = field(default_factory=dict)
    #: Effects that apply only when every listed setting is set together.
    together: list[tuple[tuple[Key, ...], PerQuestion]] = field(default_factory=list)
    #: The changes a perfect tune would keep.
    real: tuple[Key, ...] = ()
    heavy_tails: bool = False
    #: A setting that heats the laptop: its writing effect falls by this much (log) per
    #: fresh load with it, from the effect given in ``effects``.
    heats: tuple[Key, float] | None = None
    #: Answers that change, or heavy paging, with this setting on the short question.
    wrong_answers: Key | None = None
    paging: Key | None = None
    #: Search time in simulated seconds (Balanced only), on the same clock as the
    #: simulated loads and answers.
    budget_s: float | None = None
    #: The user says yes to trade-offs.
    yes_to_tradeoffs: bool = False

    def change(self, values: dict[str, object], question: str, heat_loads: int = 0) -> tuple[float, float]:
        writing = first = 0.0
        for (setting, value), per_question in self.effects.items():
            if values.get(setting) == value and question in per_question:
                writing += per_question[question][0]
                first += per_question[question][1]
        for needs, per_question in self.together:
            if all(values.get(s) == v for s, v in needs) and question in per_question:
                writing += per_question[question][0]
                first += per_question[question][1]
        if self.heats is not None and values.get(self.heats[0][0]) == self.heats[0][1]:
            writing -= self.heats[1] * heat_loads
        return writing, first


def _values(config: Any) -> dict[str, object]:
    return {
        "threads": config.threads,
        "threads_batch": config.threads_batch,
        "ubatch_size": config.ubatch_size,
        "flash_attention": config.flash_attention,
    }


def production_score(writing_log: float, first_log: float, weights: dict[str, float]) -> float:
    """The goal score production gives a change: each goal measure's relative change,
    signed so + is better, weighted by the goal."""
    relative = {
        "decode_tokens_per_sec": math.exp(writing_log) - 1.0,
        # First-word *time* changes by exp(-first_log) - 1 when its speed changes.
        "ttft_s": math.exp(-first_log) - 1.0,
    }
    return sum(weight * improvement(metric, relative.get(metric, 0.0)) for metric, weight in weights.items())


WORLDS: list[World] = [
    World("nothing helps"),
    World("nothing helps, heavy-tailed noise", heavy_tails=True),
    World(
        "threads 4: +4% writing",
        {("threads", 4): {"short": (0.04, 0.0), "long": (0.04, 0.0)}},
        real=(("threads", 4),),
    ),
    World(
        "threads 4: +4% writing, heavy-tailed noise",
        {("threads", 4): {"short": (0.04, 0.0), "long": (0.04, 0.0)}},
        real=(("threads", 4),), heavy_tails=True,
    ),
    World("threads 4: +1% (too small)", {("threads", 4): {"short": (0.01, 0.0), "long": (0.01, 0.0)}}),
    World(
        "batch 1024: long +16%, short -1.5% (within 2%)",
        {("ubatch_size", 1024): {"long": (0.0, 0.16), "short": (-0.03, 0.0)}},
        real=(("ubatch_size", 1024),),
    ),
    World(
        "batch 1024: long +16%, short -2.5% (beyond 2%)",
        {("ubatch_size", 1024): {"long": (0.0, 0.16), "short": (-0.05, 0.0)}},
    ),
    World(
        "batch then threads (threads helps only after batch)",
        {("ubatch_size", 1024): {"long": (0.0, 0.16)}},
        together=[((("ubatch_size", 1024), ("threads", 4)), {"short": (0.08, 0.0), "long": (0.08, 0.0)})],
        real=(("ubatch_size", 1024), ("threads", 4)),
    ),
    World(
        "two good changes, harmful together",
        {
            ("threads", 4): {"short": (0.06, 0.0), "long": (0.06, 0.0)},
            ("ubatch_size", 1024): {"long": (0.0, 0.16)},
        },
        together=[((("ubatch_size", 1024), ("threads", 4)), {"short": (-0.16, 0.0), "long": (-0.16, 0.0)})],
        # Either change alone is good; keeping both is harmful (judged by "harmful").
    ),
    World(
        "threads 12 looks +5% but heats the laptop (-0.6% per load)",
        {("threads", 12): {"short": (0.05, 0.0), "long": (0.05, 0.0)}},
        heats=(("threads", 12), 0.006),
    ),
    World(
        "trade-off, user says yes: batch long +16%, short -8%",
        {("ubatch_size", 1024): {"long": (0.0, 0.16), "short": (-0.08, 0.0)}},
        yes_to_tradeoffs=True,
    ),
    World(
        "trade-off, user says yes, but answers change on short",
        {("ubatch_size", 1024): {"long": (0.0, 0.16), "short": (-0.08, 0.0)}},
        wrong_answers=("ubatch_size", 1024), yes_to_tradeoffs=True,
    ),
    World(
        "trade-off, user says yes, but heavy paging on short",
        {("ubatch_size", 1024): {"long": (0.0, 0.16), "short": (-0.08, 0.0)}},
        paging=("ubatch_size", 1024), yes_to_tradeoffs=True,
    ),
    World(
        "time limit runs out (Balanced budget 600 simulated s): threads 4 +8%",
        {("threads", 4): {"short": (0.08, 0.0), "long": (0.08, 0.0)}},
        real=(("threads", 4),), budget_s=600.0,
    ),
]


class Machine:
    """A laptop with per-load offsets that drift, run-to-run noise, and a clock."""

    def __init__(self, rng: random.Random, world: World, *, within_writing: float,
                 within_first: float, between: float, rho: float) -> None:
        self.rng, self.world = rng, world
        self.within_writing, self.within_first = within_writing, within_first
        self.between, self.rho = between, rho
        self.drift = 0.0
        self.loads = 0
        self.seconds = 0.0
        self.heat_loads = 0
        self._heating = False

    def _shock(self, sigma: float) -> float:
        if not self.world.heavy_tails:
            return self.rng.gauss(0.0, sigma)
        # Student-t, 3 degrees of freedom, scaled to the same spread.
        chi = sum(self.rng.gauss(0.0, 1.0) ** 2 for _ in range(3))
        return self.rng.gauss(0.0, 1.0) / math.sqrt(chi / 3.0) / math.sqrt(3.0) * sigma

    def new_load(self, options: ProfilerOptions) -> None:
        self.loads += 1
        self.seconds += LOAD_S
        self.drift = self.rho * self.drift + math.sqrt(1 - self.rho**2) * self._shock(self.between)
        heats = self.world.heats
        self._heating = heats is not None and _values(options).get(heats[0][0]) == heats[0][1]
        if self._heating:
            self.heat_loads += 1

    def respond(self, task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        question = "long" if task.workload.workload_id.startswith("long-") else "short"
        self.seconds += ANSWER_S[question]
        values = _values(options)
        writing, first = self.world.change(values, question, self.heat_loads if self._heating else 0)
        speed = 20.0 * math.exp(self.drift + writing + self._shock(self.within_writing))
        first_speed = math.exp(self.drift + first + self._shock(self.within_first))
        sample = SampleResult(decode_tps=speed, total_latency_s=2.0 / first_speed)
        if question == "short":
            if self.world.wrong_answers and values.get(self.world.wrong_answers[0]) == self.world.wrong_answers[1]:
                sample.text = "Bananas are yellow and 7 + 5 = 13."
            if self.world.paging and values.get(self.world.paging[0]) == self.world.paging[1]:
                sample.hard_faults_per_sec = 11_000.0
        return sample


@contextmanager
def simulated_clock(machine: Machine) -> Iterator[None]:
    """Every ``time.perf_counter()`` the tune makes reads the simulated clock."""
    original = time.perf_counter
    time.perf_counter = lambda: machine.seconds
    try:
        yield
    finally:
        time.perf_counter = original


class _Harness(ScriptedHarness):
    def __init__(self, machine: Machine) -> None:
        super().__init__(machine.respond)
        self.machine = machine

    def run_once(self, task: Any, **kwargs: Any) -> Any:
        if kwargs.get("warmup"):
            self.machine.new_load(kwargs["options"])
        return super().run_once(task, **kwargs)


@dataclass
class Outcome:
    harmful: bool
    not_worthwhile: bool
    found_all: bool
    found_part: bool
    gain: float
    loads: int
    minutes: float
    kept_tradeoff: bool
    unsafe_tradeoff: bool
    #: Simulated seconds past the budget (0 when within it or no budget).
    overrun_s: float = 0.0


def one(rng: random.Random, world: World, *, check: str, effort: str,
        noise: dict[str, float], runs: int) -> Outcome:
    tmp = Path(tempfile.mkdtemp(prefix="simtune-"))
    extra: dict[str, Any] = {}
    if world.budget_s is not None:
        effort = "balanced"
        extra["optimization_before_loading_search_time_s"] = world.budget_s
    check, _, floor = check.partition("@")
    if floor:
        extra["tune_path_check_min_first_look"] = float(floor)
    settings = _settings(
        tmp, optimization_mode=effort,
        tune_paths="threads,reading_threads,prompt_batch,flash_attention",
        tune_layer_path=False, tune_diagnosis=False, tune_heat_check_runs=0,
        tune_path_check=check, tune_path_runs=runs, tune_path_max_runs=runs, **extra,
    )
    machine = Machine(rng, world, **noise)
    tuner, _, _, _ = _tuner(tmp, machine.respond, answers=[True] * 8, settings=settings)
    tuner.harness = _Harness(machine)
    tuner.approver = ScriptedChangeApprover(
        [True] * 8, tradeoff_answers=[True] if world.yes_to_tradeoffs else []
    )
    tuner.say = lambda line: None
    with simulated_clock(machine):
        result = tuner.run()
    final, start = result.final_configuration, result.starting_configuration
    weights = dict(tuner.rules.goal.first)
    gains = {}
    for question in ("short", "long"):
        new = world.change(_values(final), question)
        old = world.change(_values(start), question)
        gains[question] = production_score(new[0] - old[0], new[1] - old[1], weights)
    if world.heats is not None and _values(final).get(world.heats[0][0]) == world.heats[0][1]:
        # Kept the heating setting: judge it by its steady state (after 20 loads).
        for question in gains:
            gains[question] -= world.heats[1] * 20
    kept_tradeoff = "trade-off you approved" in result.outcome
    unsafe = kept_tradeoff and (world.wrong_answers is not None or world.paging is not None)
    harmful = (min(gains.values()) < -float(settings.tune_guard_margin) and not kept_tradeoff) or unsafe
    smallest = float(tuner.rules.effort.minimum_improvement)
    kept = final != start
    not_worthwhile = kept and not kept_tradeoff and max(gains.values()) < smallest
    final_values = _values(final)
    got = [final_values.get(setting) == value for setting, value in world.real]
    return Outcome(
        harmful=harmful,
        not_worthwhile=not_worthwhile,
        found_all=bool(world.real) and all(got),
        found_part=bool(world.real) and any(got) and not all(got),
        gain=statistics.mean(gains.values()),
        loads=machine.loads,
        minutes=machine.seconds / 60.0,
        kept_tradeoff=kept_tradeoff,
        unsafe_tradeoff=unsafe,
        overrun_s=(
            max(machine.seconds - world.budget_s, 0.0) if world.budget_s is not None else 0.0
        ),
    )


def upper95(count: int, trials: int) -> float:
    """One-sided 95% upper bound of a rate (exact binomial for zero counts)."""
    if count == 0:
        return float(1.0 - 0.05 ** (1.0 / trials))
    rate = count / trials
    return min(1.0, rate + 1.645 * math.sqrt(rate * (1 - rate) / trials))


def run(trials: int, seed: int, effort: str, checks: tuple[str, ...], runs: int = 12,
        say: Callable[[str], None] = print) -> dict[str, dict[str, float]]:
    noise = {"within_writing": 0.03, "within_first": 0.06, "between": 0.012, "rho": 0.3}
    out: dict[str, dict[str, float]] = {}
    say(f"effort {effort}; {trials} tunes per row; {runs} runs per measurement; noise {noise}")
    say(f"{'world':62} {'check':10} {'harmful (95% up)':>17} {'not worth':>9} {'all':>5} {'part':>5} "
        f"{'gain':>6} {'loads':>6} {'min':>5} {'t-off':>5}")
    for world in WORLDS:
        for check in checks:
            rng = random.Random(f"{seed}-{world.name}-{check}")
            results = [
                one(rng, world, check=check, effort=effort, noise=noise, runs=runs)
                for _ in range(trials)
            ]
            harmful_n = sum(r.harmful for r in results)
            row = {
                "harmful": harmful_n / trials,
                "harmful_upper95": upper95(harmful_n, trials),
                "not_worthwhile": sum(r.not_worthwhile for r in results) / trials,
                "found_all": sum(r.found_all for r in results) / trials if world.real else float("nan"),
                "found_part": sum(r.found_part for r in results) / trials if world.real else float("nan"),
                "gain": statistics.mean(r.gain for r in results),
                "loads": statistics.mean(r.loads for r in results),
                "minutes": statistics.mean(r.minutes for r in results),
                "tradeoff_kept": sum(r.kept_tradeoff for r in results) / trials,
                "max_overrun_s": max(r.overrun_s for r in results),
            }
            out[f"{world.name} | {check}"] = row
            say(
                f"{world.name:62} {check:10} {row['harmful']:6.1%} ({row['harmful_upper95']:5.1%})   "
                f"{row['not_worthwhile']:8.1%} {row['found_all']:5.0%} {row['found_part']:5.0%} "
                f"{row['gain']:+6.1%} {row['loads']:6.0f} {row['minutes']:5.0f} {row['tradeoff_kept']:5.0%}"
                + (f"  over budget by up to {row['max_overrun_s']:.0f} s" if world.budget_s else "")
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--effort", default="performance_first")
    parser.add_argument("--checks", default="light,full")
    parser.add_argument("--worlds", default="")
    parser.add_argument("--runs", type=int, default=12, help="runs per measurement (laptop: Gemma short 11-13)")
    args = parser.parse_args(argv)
    if args.worlds:
        wanted = {int(i) for i in args.worlds.split(",")}
        WORLDS[:] = [w for i, w in enumerate(WORLDS) if i in wanted]
    run(args.trials, args.seed, args.effort, tuple(args.checks.split(",")), args.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
