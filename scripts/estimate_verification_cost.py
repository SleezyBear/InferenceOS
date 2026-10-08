"""What one final check costs, and what each candidate change might save (estimates only).

Handoff (6 Oct 2026): verification cost is a primary task. This separates **measured
inputs** (per-load, per-answer and noise figures read from the Unbounded campaign logs,
``claude/search-order-v2-2026-10-06.md`` section 6) from **derived estimates** (what a
check would cost under a changed procedure). Nothing here is a measurement of a changed
procedure; the per-answer overhead in particular is an *unconfirmed* attribution.

Procedures compared, per two-question final check (short "better" or "not worse", long
"better"), blocks per question fixed:

* ``today``: every measurement = 1 load + 1 warm-up + N runs, each question in its own
  loads; N from the tune's rule (5 at the floor on the long question).
* ``long runs from the certainty-per-minute rule``: the tune already picks runs per
  measurement as (within / between) x sqrt(load / run); the floor of 5 holds the long
  question above that optimum. With the floor at 2, the block spread grows a little, and
  the blocks needed are recomputed for the same power.
* ``both questions in one load``: one load per side per block, a warm-up per question,
  short then long (order alternated between blocks). Saves a load per side per block on
  the question with fewer blocks. Statistically: each question's blocks stay independent
  across blocks; the two questions' results share the load's offset (correlated), so
  they are never counted as independent evidence -- each must pass on its own, which
  the final check already requires. A Monte Carlo below checks the error rates.
* ``no per-answer monitor overhead`` (hypothesis): if the 1.2-3.4 s per answer is the
  telemetry monitor's start-up, a monitor kept running for the whole measurement would
  remove most of it. Shown only as an upper bound until the timing probe confirms it.

    python scripts/estimate_verification_cost.py [--mc 20000]
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Costs:
    """Measured from the campaign logs (per model; seconds and fractions)."""

    name: str
    load_s: float
    short_answer_s: float  # reading + writing, short question
    long_answer_s: float  # reading + writing, long question
    overhead_s: float  # per answer: answer time minus reading and writing (attribution unconfirmed)
    short_runs: int
    long_runs: int
    within_short: float
    within_long: float
    between: float


# Per-measurement figures behind section 6 of search-order-v2 (one 4-block check = 8
# measurements): loads, warm-ups, reading, writing and overhead divided back per item.
MODELS = [
    Costs("Llama 3.2 3B", 11.0, 2.0, 18.0, 1.4, 5, 5, 0.030, 0.010, 0.012),
    Costs("Granite 3.1 MoE", 12.0, 1.6, 21.0, 2.2, 5, 5, 0.035, 0.012, 0.012),
    Costs("Phi-4 mini", 16.0, 4.4, 18.0, 3.4, 16, 5, 0.040, 0.009, 0.012),
    Costs("Granite 4 H Tiny", 20.0, 4.2, 19.0, 3.4, 20, 5, 0.060, 0.015, 0.015),
    Costs("Qwen3.5 9B", 20.0, 6.0, 17.0, 0.2, 10, 5, 0.025, 0.019, 0.012),
]

Z = 1.645 + 0.842  # one-sided 5% test, 80% power


def block_spread(within: float, between: float, runs: int) -> float:
    return math.sqrt(2.0 * (between**2 + within**2 / runs))


def blocks_for(spread: float, effect: float, low: int = 4, high: int = 16) -> int:
    return int(min(max(math.ceil((Z * spread / effect) ** 2), low), high))


def measurement_s(load: float, answer: float, overhead: float, runs: int, *, monitor: bool = True) -> float:
    per = answer + (overhead if monitor else 0.0)
    return load + (1 + runs) * per


def check_cost(c: Costs, *, short_effect: float, long_effect: float, long_runs: int | None = None,
               shared: bool = False, monitor: bool = True) -> tuple[float, int, int]:
    lr = long_runs if long_runs is not None else c.long_runs
    n_short = blocks_for(block_spread(c.within_short, c.between, c.short_runs), short_effect)
    n_long = blocks_for(block_spread(c.within_long, c.between, lr), long_effect)
    m_short = measurement_s(c.load_s, c.short_answer_s, c.overhead_s, c.short_runs, monitor=monitor)
    m_long = measurement_s(c.load_s, c.long_answer_s, c.overhead_s, lr, monitor=monitor)
    total = 2 * n_short * m_short + 2 * n_long * m_long
    if shared:
        total -= 2 * min(n_short, n_long) * c.load_s
    return total, n_short, n_long


def optimum_runs(c: Costs, floor: int) -> int:
    """The tune's certainty-per-minute rule (``choose_runs``), with a different floor."""
    run = c.long_answer_s + c.overhead_s
    raw = (c.within_long / c.between) * math.sqrt(c.load_s / run)
    return max(floor, min(20, round(raw)))


def monte_carlo(trials: int, rho_questions: float, seed: int = 1) -> dict[str, float]:
    """False-pass rates of two null questions (no true change) at 6 blocks each, with a
    load offset shared by both questions in a shared load (correlation ``rho``)."""
    rng = random.Random(seed)
    n, t_crit = 6, 2.015  # one-sided 5%, 5 degrees of freedom
    passes = [0, 0]
    both = 0
    for _ in range(trials):
        results = []
        diffs: list[list[float]] = [[], []]
        for _block in range(n):
            shared = rng.gauss(0, 1)
            for q in range(2):
                own = rng.gauss(0, 1)
                diffs[q].append(math.sqrt(rho_questions) * shared + math.sqrt(1 - rho_questions) * own)
        for q in range(2):
            mean = statistics.mean(diffs[q])
            sd = statistics.stdev(diffs[q])
            results.append(mean - t_crit * sd / math.sqrt(n) > 0)
            passes[q] += results[-1]
        both += all(results)
    return {"q1": passes[0] / trials, "q2": passes[1] / trials, "both": both / trials}


def order_bias(order_effect_a: float, order_effect_b: float) -> float:
    """Bias of the long question's difference when it always runs second in the load:
    zero when the second-in-load slowdown is the same for both setups; otherwise the
    difference of the two slowdowns (unknown; needs the probe)."""
    return order_effect_b - order_effect_a


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=20000)
    args = parser.parse_args(argv)
    print("Two-question final check (short: better at +5%; long: better at +10%), minutes. Estimates.")
    print(f"{'model':18} {'today':>7} {'runs(2)':>8} {'shared':>7} {'both':>6} {'no-mon':>7} {'all':>6}   blocks today -> runs(2)")
    for c in MODELS:
        today, ns, nl = check_cost(c, short_effect=0.05, long_effect=0.10)
        lr = optimum_runs(c, 2)
        fewer, _, nl2 = check_cost(c, short_effect=0.05, long_effect=0.10, long_runs=lr)
        shared, _, _ = check_cost(c, short_effect=0.05, long_effect=0.10, shared=True)
        both, _, _ = check_cost(c, short_effect=0.05, long_effect=0.10, long_runs=lr, shared=True)
        nomon, _, _ = check_cost(c, short_effect=0.05, long_effect=0.10, monitor=False)
        allr, _, _ = check_cost(c, short_effect=0.05, long_effect=0.10, long_runs=lr, shared=True, monitor=False)
        print(f"{c.name:18} {today / 60:7.1f} {fewer / 60:8.1f} {shared / 60:7.1f} {both / 60:6.1f} "
              f"{nomon / 60:7.1f} {allr / 60:6.1f}   short {ns}, long {nl} (5 runs) -> long {nl2} ({lr} runs)")
    print()
    print("Long question only (a batch change guarded on short at 'not worse' 2%):")
    for c in MODELS:
        n_guard = blocks_for(block_spread(c.within_short, c.between, c.short_runs), 0.02)
        print(f"  {c.name:18} short guard needs {n_guard} blocks for 80% power at zero true change")
    print()
    mc0 = monte_carlo(args.mc, 0.0)
    mc7 = monte_carlo(args.mc, 0.7)
    print(f"Monte Carlo, two null questions, 6 blocks, 5% one-sided each ({args.mc} trials):")
    print(f"  separate loads (rho 0):  q1 {mc0['q1']:.3f}  q2 {mc0['q2']:.3f}  both {mc0['both']:.4f}")
    print(f"  shared load   (rho 0.7): q1 {mc7['q1']:.3f}  q2 {mc7['q2']:.3f}  both {mc7['both']:.4f}")
    print("  -> each question keeps its 5% rate; only the joint rate rises (still <= 5%),")
    print("     so requiring both to pass stays valid; pooling them as independent would not.")
    print(f"  order effect: bias = difference of the second-in-load slowdowns, e.g. {order_bias(0.02, 0.025):+.1%}"
          " if one setup slows 0.5 points more (unknown until measured)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
