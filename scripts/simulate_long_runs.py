"""Long-question runs per measurement (5 vs 2), through the practical mode's own selection,
sizing and acceptance procedure (revised after review 9).

Review 9: the first version sized blocks from the simulation's *known* effect and noise,
for one comparison under Gaussian noise. This version runs what the tuner would do with
only its *estimates*, for a prompt-batch candidate (found on the long question, guarded on
the short one):

1. **Screening**: one fresh-load measurement of the start and of the candidate on each
   question (the tuner's measurement code, ``measure_steady``); the reading is the
   tuner's screening gain (``_screen_gain``) and its noise width comes from the start's
   own run-to-run spread plus the tuner's fallback load-to-load spread (1.2%), not the
   truth.
2. **Selection**: the candidate goes on only if its long reading is *clear* (more than two
   noise widths and at least the smallest worthwhile gain), and its short reading does
   not lose more than max(3%, two widths) -- the practical mode's rules.
3. **Sizing**: final blocks from those estimates with the practical mode's rule
   (``BudgetedTune._blocks``: 80% power at 0.7 x the reading, spread x1.25, 4..16; the
   short guard sized for the 2% margin).
4. **Acceptance**: the real ``compare_fairly`` -- long "better" at N blocks, then short
   "not worse" at M blocks; accepted only if both pass.

The laptop is scripted with a per-load offset that drifts (AR(1), rho 0.5), per-answer
noise, and occasional slow answers (3% of answers 1.4x slower). Reported per runs count
and truth: how often the candidate is accepted (missed gains = 1 - that for real gains;
false or harmful acceptance for the others), and simulated minutes (screening + final).

    python scripts/simulate_long_runs.py [--trials 150] [--seed 1]
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferenceos.config import Settings  # noqa: E402
from inferenceos.contracts.experiment import RuntimeConfiguration  # noqa: E402
from inferenceos.optimization.keep import ModeRules  # noqa: E402
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask  # noqa: E402
from inferenceos.profiling.workloads import get_workload  # noqa: E402
from inferenceos.selftest import sample_model  # noqa: E402
from inferenceos.tune.budgeted import GUARD_SCREEN_TOLERANCE, SPREAD_MARGIN, Z_POWER  # noqa: E402
from inferenceos.tune.fair import compare_fairly  # noqa: E402
from inferenceos.tune.measure import DisturbanceWatch, measure_steady, run_score_spread  # noqa: E402
from inferenceos.tune.paths import long_question_workload, steady_workload  # noqa: E402
from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy  # noqa: E402

START = RuntimeConfiguration(context_length=8192, ubatch_size=512)
NEW = RuntimeConfiguration(context_length=8192, ubatch_size=1024)
LOAD_S = 11.0
ANSWER_S = {"short": 4.0, "long": 19.4}
OVERHEAD_S = 1.5
FALLBACK_BETWEEN = 0.012  # what the tuner assumes before it has learned the spread
SHORT_RUNS = 8


class Machine:
    def __init__(self, rng: random.Random, *, effect: dict[str, float], within: dict[str, float],
                 between: float, rho: float = 0.5, slow_share: float = 0.03) -> None:
        self.rng, self.effect, self.within, self.between = rng, effect, within, between
        self.rho, self.slow_share = rho, slow_share
        self.offset = 0.0
        self.seconds = 0.0

    def new_load(self) -> None:
        self.seconds += LOAD_S
        self.offset = self.rho * self.offset + math.sqrt(1 - self.rho**2) * self.rng.gauss(0.0, self.between)

    def respond(self, task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        q = "long" if task.workload.workload_id.startswith("long-") else "short"
        self.seconds += ANSWER_S[q] + OVERHEAD_S
        gain = self.effect[q] if options.ubatch_size == 1024 else 0.0
        slow = math.log(1.4) if self.rng.random() < self.slow_share else 0.0
        factor = math.exp(self.offset + gain - slow + self.rng.gauss(0.0, self.within[q]))
        return SampleResult(decode_tps=20.0 * factor, total_latency_s=2.0 / factor)


class _Harness(ScriptedHarness):
    def __init__(self, machine: Machine) -> None:
        super().__init__(machine.respond)
        self.machine = machine

    def run_once(self, task: Any, **kwargs: Any) -> Any:
        if kwargs.get("warmup"):
            self.machine.new_load()
        return super().run_once(task, **kwargs)


def _blocks(spread: float, effect: float) -> int:
    if effect <= 0:
        return 16
    return int(min(max(math.ceil((Z_POWER * SPREAD_MARGIN * spread / effect) ** 2), 4), 16))


def one(rng: random.Random, *, long_runs: int, effect: dict[str, float], within: dict[str, float],
        between: float) -> dict[str, Any]:
    from inferenceos.tune.pipeline import _screen_gain

    tmp = Path(tempfile.mkdtemp(prefix="simlong-"))
    settings = Settings(data_dir=tmp / "d", reports_dir=tmp / "r", logs_dir=tmp / "l",
                        optimization_mode="performance_first")
    rules = ModeRules.from_settings(settings)
    machine = Machine(rng, effect=effect, within=within, between=between)
    harness = _Harness(machine)
    model = sample_model()
    short = get_workload("short-short")
    long = long_question_workload(18, base=steady_workload("long-prefill-steady", default=short), per_passage=100.0)
    watch = DisturbanceWatch(logical_cores=12, limit_percent=1000.0, max_repeats=0)
    runs = {"short": SHORT_RUNS, "long": long_runs}
    workloads = {"short": short, "long": long}

    def policy(q: str) -> Any:
        return build_test_policy(warmup_repetitions=1, measured_repetitions=runs[q], max_repetitions=runs[q],
                                 min_valid_samples=runs[q], max_total_runs=runs[q] + 3, noise_tolerance=0.0)

    original = time.perf_counter
    time.perf_counter = lambda: machine.seconds
    try:
        readings: dict[str, float | None] = {}
        spreads: dict[str, float] = {}
        for q in ("long", "short"):
            ref = measure_steady(harness=harness, model=model, workload=workloads[q], config=START,
                                 policy=policy(q), phase="screen", watch=watch)
            cand = measure_steady(harness=harness, model=model, workload=workloads[q], config=NEW,
                                  policy=policy(q), phase="screen", watch=watch)
            within_est = run_score_spread(ref, rules.goal.first) or 0.0
            spreads[q] = math.sqrt(2.0 * (FALLBACK_BETWEEN**2 + within_est**2 / runs[q]))
            readings[q] = _screen_gain(ref, cand, rules.goal) if ref.valid_runs and cand.valid_runs else None
        screening_s = machine.seconds
        smallest = float(rules.effort.minimum_improvement)
        long_read, short_read = readings["long"], readings["short"]
        clear = long_read is not None and long_read > 2 * spreads["long"] and long_read >= smallest
        guard_ok = short_read is not None and short_read >= -max(GUARD_SCREEN_TOLERANCE, 2 * spreads["short"])
        out: dict[str, Any] = {"selected": bool(clear and guard_ok), "accepted": False,
                               "screening_s": screening_s, "final_s": 0.0}
        if not out["selected"]:
            out["total_s"] = machine.seconds
            return out
        n_long = _blocks(spreads["long"], max(0.7 * float(long_read or 0.0), smallest))
        n_short = _blocks(spreads["short"], float(settings.tune_guard_margin))
        out["blocks"] = (n_long, n_short)
        common = dict(harness=harness, model=model, start=START, candidate=NEW, settings=settings, rules=rules,
                      watch=watch, card_id="runtime.prompt_batch", phase="final")
        fair_long = compare_fairly(workload=long, policy=policy("long"), mode="better",
                                   min_blocks=n_long, max_blocks=n_long, **common)
        accepted = fair_long.kept
        out["long_passed"] = bool(fair_long.kept)
        if accepted:
            fair_short = compare_fairly(workload=short, policy=policy("short"), mode="not_worse",
                                        min_blocks=n_short, max_blocks=n_short, **common)
            accepted = fair_short.not_worse
        out["accepted"] = bool(accepted)
        out["final_s"] = machine.seconds - screening_s
        out["total_s"] = machine.seconds
        return out
    finally:
        time.perf_counter = original


SCENARIOS = [
    ("no gain", {"long": 0.0, "short": 0.0}, "false acceptance"),
    ("long +5%", {"long": 0.05, "short": 0.0}, "missed"),
    ("long +10%", {"long": 0.10, "short": 0.0}, "missed"),
    ("long +10%, short -1%", {"long": 0.10, "short": -0.01}, "missed"),
    ("long +10%, short -3% (harmful)", {"long": 0.10, "short": -0.03}, "harmful acceptance"),
]
NOISE = [
    ("Llama-like", {"short": 0.03, "long": 0.010}, 0.012),
    ("Granite-4H-like", {"short": 0.06, "long": 0.015}, 0.015),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenarios", default="", help="comma-separated parts of scenario names to run")
    args = parser.parse_args(argv)
    print(f"{args.trials} simulated candidates per cell; drift rho 0.5, 3% slow answers; short {SHORT_RUNS} runs")
    for noise_name, within, between in NOISE:
        print(f"\n{noise_name}: run-to-run short {within['short']:.1%} / long {within['long']:.1%}, "
              f"load-to-load {between:.1%}")
        print(f"  {'truth':32} {'runs':>4} {'selected':>8} {'accepted':>8} {'what it means':>26} "
              f"{'blocks long/short':>17} {'min when selected':>17} {'long passed / guard failed':>27}")
        for name, effect, meaning in SCENARIOS:
            if args.scenarios and not any(key in name for key in args.scenarios.split(",")):
                continue
            for long_runs in (5, 2):
                rng = random.Random(f"{args.seed}-{noise_name}-{name}-{long_runs}")
                outs = [one(rng, long_runs=long_runs, effect=effect, within=within, between=between)
                        for _ in range(args.trials)]
                sel = [o for o in outs if o["selected"]]
                acc = sum(o["accepted"] for o in outs) / len(outs)
                rate = (1 - acc) if meaning == "missed" else acc
                blocks = (f"{statistics.mean(o['blocks'][0] for o in sel):.1f}/"
                          f"{statistics.mean(o['blocks'][1] for o in sel):.1f}") if sel else "-"
                minutes = f"{statistics.mean(o['total_s'] for o in sel) / 60:.1f}" if sel else "-"
                passed = sum(o.get("long_passed", False) for o in sel)
                guard_failed = sum(o.get("long_passed", False) and not o["accepted"] for o in sel)
                print(f"  {name:32} {long_runs:>4} {len(sel) / len(outs):8.0%} {acc:8.1%} "
                      f"{meaning + f' {rate:.1%}':>26} {blocks:>17} {minutes:>17} "
                      f"{f'{passed}/{len(sel)} / {guard_failed}':>27}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
