"""The fair test's false "better" rate under persistent load-to-load drift (9 Oct 2026).

A focused Monte Carlo of the block statistic alone (no tuner): each measurement's score
is a load-level drift d plus independent measurement noise. Drift follows an AR(1)
process across consecutive loads with persistence ``rho`` and unit variance. Blocks
alternate their order (start then candidate, then candidate then start, ...), as
``compare_fairly`` does, so a block score is s_k * (x_second - x_first) with s_k = +1, -1,
+1, ... There is no true effect. The decision is the one ``tune/stats.py`` makes at a
single look: "better" when the one-sided ``alpha`` lower t-bound of the mean block score
is above zero.

What it shows: with rho = 0 the rate is alpha. With positive rho, consecutive block
scores become positively correlated (the alternation turns the negative correlation of
consecutive load differences positive), so the t-bound is too narrow and the rate rises
above alpha. ``--within`` adds independent per-measurement noise (as a share of the drift
standard deviation), which dilutes the effect.

Assumptions: Gaussian drift and noise, stationary AR(1), no true effect, no discarded
blocks, every block counted. Real load-to-load behaviour on the laptop is not known to
follow this model; the numbers say how sensitive the test is, not what the laptop does.

    python scripts/simulate_block_drift.py [--trials 20000] [--rho 0,0.3,0.6,0.85]
        [--blocks 4,7,11,16] [--within 0] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inferenceos.tune.stats import block_bounds


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def false_better(rho: float, blocks: int, trials: int, *, alpha: float, within: float, seed: int) -> int:
    rng = random.Random(seed)
    shock = math.sqrt(1 - rho * rho)
    bad = 0
    for _ in range(trials):
        drift = rng.gauss(0.0, 1.0)
        scores = []
        for k in range(blocks):
            first = drift + within * rng.gauss(0.0, 1.0)
            drift = rho * drift + shock * rng.gauss(0.0, 1.0)
            second = drift + within * rng.gauss(0.0, 1.0)
            drift = rho * drift + shock * rng.gauss(0.0, 1.0)
            # Even blocks: start first, candidate second; odd blocks: the reverse.
            scores.append(second - first if k % 2 == 0 else first - second)
        test = block_bounds(scores, alpha=alpha)
        if test.lower is not None and test.lower > 0:
            bad += 1
    return bad


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--rho", default="0,0.3,0.6,0.85")
    parser.add_argument("--blocks", default="4,7,11,16")
    parser.add_argument("--within", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    for rho in (float(x) for x in args.rho.split(",")):
        for n in (int(x) for x in args.blocks.split(",")):
            bad = false_better(rho, n, args.trials, alpha=args.alpha, within=args.within, seed=args.seed)
            lo, hi = wilson(bad, args.trials)
            rows.append({"rho": rho, "blocks": n, "within_share": args.within, "trials": args.trials,
                         "false_better": bad, "rate": bad / args.trials, "wilson_95": [lo, hi]})
            print(f"rho {rho:4.2f}  blocks {n:2d}  within {args.within:.2f}: {bad}/{args.trials} = "
                  f"{bad / args.trials:.4f}  (95% {lo:.4f}-{hi:.4f})", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"alpha": args.alpha, "assumptions": __doc__.split("Assumptions:")[1]
                                               .split("python")[0].strip(), "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
