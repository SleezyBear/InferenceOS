"""Error rates of the real fair comparison code, by simulation (no model needed).

Runs ``tune/fair.py``'s ``compare_fairly`` itself (not a copy of its statistics) many
times with a scripted machine whose two setups are truly equal (or truly 2% worse for
the "must not lose" check), under conditions the laptop shows: a different offset per
measurement, slow drift that carries over from one measurement to the next, heavy
tails, measurements that fail, and time limits that cut a comparison short. Prints
how often a change that does not help would be kept, and how often a 2% loss would
pass "not worse". The owner's limit is 5% (1 in 20). Review 2, 4 Oct 2026.

    python scripts/simulate_fair.py [--runs 2000] [--seed 1]
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferenceos.config import Settings  # noqa: E402
from inferenceos.contracts.experiment import RuntimeConfiguration  # noqa: E402
from inferenceos.optimization.keep import ModeRules  # noqa: E402
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask  # noqa: E402
from inferenceos.profiling.workloads import get_workload  # noqa: E402
from inferenceos.selftest import sample_model  # noqa: E402
from inferenceos.tune.fair import compare_fairly  # noqa: E402
from inferenceos.tune.measure import DisturbanceWatch  # noqa: E402
from tests.experiment_helpers import (  # noqa: E402
    SampleResult,
    ScriptedHarness,
    build_test_policy,
)

START = RuntimeConfiguration(context_length=2048, threads=4)
NEW = RuntimeConfiguration(context_length=2048, threads=5)


class Machine:
    """Scripted laptop: per-measurement offsets, drift, heavy tails, failures."""

    def __init__(
        self,
        rng: random.Random,
        *,
        true_change: float,
        noise: float,
        drift_rho: float,
        heavy_tails: bool,
        fail_share: float,
    ) -> None:
        self.rng = rng
        self.true_change = true_change
        self.noise = noise
        self.rho = drift_rho
        self.heavy = heavy_tails
        self.fail_share = fail_share
        self.drift = 0.0
        self.offset = 0.0
        self.fail = False

    def new_measurement(self) -> None:
        shock = self.rng.gauss(0.0, 1.0)
        if self.heavy:
            # Student-t with 3 degrees of freedom, scaled to unit spread.
            chi = sum(self.rng.gauss(0.0, 1.0) ** 2 for _ in range(3))
            shock = shock / math.sqrt(chi / 3.0) / math.sqrt(3.0)
        self.drift = self.rho * self.drift + math.sqrt(1 - self.rho**2) * self.noise * shock
        self.offset = self.drift
        self.fail = self.rng.random() < self.fail_share

    def respond(self, task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if self.fail:
            return SampleResult(fail=True)
        factor = 1.0 + self.offset + (self.true_change if options.threads == 5 else 0.0)
        # Both goal measures move together (writing faster, first word sooner).
        return SampleResult(decode_tps=20.0 * factor, total_latency_s=2.0 / factor)


class _Harness(ScriptedHarness):
    def __init__(self, machine: Machine) -> None:
        super().__init__(machine.respond)
        self.machine = machine

    def run_once(self, task: Any, **kwargs: Any) -> Any:
        if kwargs.get("warmup"):
            self.machine.new_measurement()
        return super().run_once(task, **kwargs)


def one(
    rng: random.Random,
    *,
    effort: str,
    mode: str,
    true_change: float,
    noise: float,
    drift_rho: float,
    heavy_tails: bool,
    fail_share: float,
    cut_share: float,
) -> bool:
    """One comparison; True when it passes (kept, or shown not worse)."""
    tmp = Path(tempfile.mkdtemp(prefix="simfair-"))
    settings = Settings(
        data_dir=tmp / "data", reports_dir=tmp / "reports", logs_dir=tmp / "logs",
        optimization_mode=effort,
    )
    machine = Machine(
        rng, true_change=true_change, noise=noise, drift_rho=drift_rho,
        heavy_tails=heavy_tails, fail_share=fail_share,
    )
    deadline = None
    if cut_share > 0 and rng.random() < cut_share:
        # A time limit somewhere inside the comparison (a few blocks in).
        deadline = time.perf_counter() + rng.uniform(0.0, 0.02)
    fair = compare_fairly(
        harness=_Harness(machine),
        model=sample_model(),
        workload=get_workload("short-short"),
        start=START,
        candidate=NEW,
        policy=build_test_policy(),
        settings=settings,
        rules=ModeRules.from_settings(settings),
        watch=DisturbanceWatch(logical_cores=12, limit_percent=1000.0, max_repeats=0),
        card_id="runtime.reduce_threads",
        phase="sim",
        mode=mode,  # type: ignore[arg-type]
        deadline=deadline,
    )
    if mode == "not_worse":
        return fair.not_worse
    return fair.kept


SCENARIOS: list[tuple[str, dict[str, Any]]] = [
    ("plain noise 3%", {}),
    ("drift carried over (0.7)", {"drift_rho": 0.7}),
    ("heavy tails", {"heavy_tails": True}),
    ("20% of measurements fail", {"fail_share": 0.2}),
    ("time limit cuts 50%", {"cut_share": 0.5}),
    ("all of the above", {"drift_rho": 0.7, "heavy_tails": True, "fail_share": 0.2, "cut_share": 0.5}),
]


def rates(
    runs: int, seed: int, *, scenarios: list[tuple[str, dict[str, Any]]] | None = None,
    efforts: tuple[str, ...] = ("balanced", "performance_first", "minimize_optimization_cost"),
    say: Callable[[str], None] = print,
) -> dict[str, float]:
    out: dict[str, float] = {}
    rng = random.Random(seed)
    margin = float(Settings().tune_guard_margin)
    for name, extra in scenarios or SCENARIOS:
        base = {"noise": 0.03, "drift_rho": 0.0, "heavy_tails": False, "fail_share": 0.0, "cut_share": 0.0}
        base.update(extra)
        for effort in efforts:
            # Equal setups: how often "better" and kept?
            kept = sum(
                one(rng, effort=effort, mode="better", true_change=0.0, **base) for _ in range(runs)
            ) / runs
            # The guard: a true loss of exactly the margin, how often "not worse"?
            # (Only the minimize-cost effort keeps the smallest gain at 15%, so the keep
            # rate there is near 0 anyway; the guard check is the same in every mode.)
            passed = sum(
                one(rng, effort=effort, mode="not_worse", true_change=-margin, **base)
                for _ in range(runs)
            ) / runs
            out[f"{name} | {effort} | kept"] = kept
            out[f"{name} | {effort} | 2% loss passes"] = passed
            say(f"{name:28} {effort:28} kept {kept:6.1%}   2% loss passes {passed:6.1%}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    found = rates(args.runs, args.seed)
    worst = max(found.values())
    print(f"worst rate: {worst:.1%} (limit 5%)")
    return 0 if worst <= 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
