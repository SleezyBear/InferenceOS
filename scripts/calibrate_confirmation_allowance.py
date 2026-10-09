"""Reproduce the confirmation planner's noise-allowance calibration (9 Oct 2026).

Input: ``q6_final_check_calibration.csv`` (jobs 090-107: one row per fair comparison and
question, with the screening noise width the tune had and the block scores the fair
comparison then produced). Output: per group of rows, an **estimate** of how the
realised block-to-block spread compares with the screening noise width, after removing
the sampling error of each comparison's own small number of blocks.

Assumptions (all of them matter; none is verified here):

1. Within one comparison the counted block scores are independent and normal, so the
   sample variance s^2 from n blocks satisfies (n - 1) s^2 / sigma^2 ~ chi^2(n - 1).
   (Serial dependence from persistent drift would break this; see
   ``scripts/simulate_block_drift.py``.)
2. Across comparisons, the ratio r = sigma / screening width is log-normal. The screening
   width is taken as fixed; its own uncertainty is part of the spread of r.
3. log(s / width) = log r + e, where e = log(s / sigma) is independent of r with the known
   mean and variance of log(sqrt(chi^2_k / k)), k = n - 1:
   E[e] = (digamma(k/2) - log(k/2)) / 2 and Var[e] = trigamma(k/2) / 4.
4. Method of moments: the mean of log r is the mean of (log(s / width) - E[e]); the
   variance of log r is the observed variance minus the mean sampling variance (floored
   at zero). Percentiles of r follow from the normal quantiles of log r.
5. Every row used the 1.2% fallback load-to-load spread, so the result describes
   fallback-based widths only.

Uncertainty: a row bootstrap (resampling comparisons with replacement) gives an interval
for each reported percentile. With 10 final-check rows these intervals are wide.

    python scripts/calibrate_confirmation_allowance.py CSV [--json OUT] [--bootstrap 2000]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

Z = {0.5: 0.0, 0.8: 0.8416212335729143, 0.9: 1.2815515655446004}


def digamma(x: float) -> float:
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return result + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


def trigamma(x: float) -> float:
    result = 0.0
    while x < 6.0:
        result += 1.0 / (x * x)
        x += 1.0
    f = 1.0 / (x * x)
    return result + 1.0 / x + f / 2.0 + f / x * (1 / 6 - f * (1 / 30 - f * (1 / 42 - f / 30)))


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Rows with a positive realised SD, a positive screening width and at least 3 blocks."""
    rows = []
    for raw in csv.DictReader(path.open(encoding="utf-8")):
        try:
            sd, width, n = float(raw["fair_sd"]), float(raw["screening_noise_width"]), int(raw["fair_blocks"])
        except (KeyError, ValueError):
            continue
        if sd <= 0 or width <= 0 or n < 3:
            continue
        k = n - 1
        rows.append({
            "job": raw["job"], "kind": raw["kind"], "question": raw["question"], "candidate": raw["candidate"],
            "blocks": n, "fair_sd": sd, "screening_noise_width": width, "observed_ratio": sd / width,
            "between_source": raw.get("between_source", ""),
            "log_ratio": math.log(sd / width),
            "sampling_mean": (digamma(k / 2) - math.log(k / 2)) / 2,
            "sampling_variance": trigamma(k / 2) / 4,
        })
    return rows


def estimate(rows: list[dict[str, Any]]) -> dict[str, float]:
    centred = [r["log_ratio"] - r["sampling_mean"] for r in rows]
    mean = statistics.fmean(centred)
    observed_var = statistics.pvariance(centred)
    sampling_var = statistics.fmean(r["sampling_variance"] for r in rows)
    true_var = max(observed_var - sampling_var, 0.0)
    out = {"mean_log_ratio": mean, "observed_log_variance": observed_var, "mean_sampling_variance": sampling_var,
           "estimated_log_sd": math.sqrt(true_var)}
    for p, z in Z.items():
        out[f"p{int(p * 100)}"] = math.exp(mean + z * math.sqrt(true_var))
    return out


def bootstrap(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {f"p{int(p * 100)}": [] for p in Z}
    for _ in range(draws):
        pick = [rows[rng.randrange(len(rows))] for _ in rows]
        est = estimate(pick)
        for key in samples:
            samples[key].append(est[key])
    out = {}
    for key, values in samples.items():
        values.sort()
        out[key] = [values[int(0.05 * len(values))], values[int(0.95 * len(values)) - 1]]
    return out


GROUPS = {
    "all comparisons": lambda r: True,
    "fresh final whole-setup checks (kind = final check)": lambda r: r["kind"] == "final check",
    "short question": lambda r: r["question"].startswith("short"),
    "long question": lambda r: r["question"].startswith("long"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--json", default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261009)
    args = parser.parse_args(argv)
    rows = load_rows(Path(args.csv))
    result: dict[str, Any] = {"source": str(args.csv), "rows_used": len(rows), "assumptions": __doc__.split(
        "Assumptions")[1].split("Uncertainty")[0].strip(), "groups": {}, "rows": rows}
    for name, keep in GROUPS.items():
        group = [r for r in rows if keep(r)]
        est: dict[str, Any] = dict(estimate(group))
        est["n_rows"] = len(group)
        est["bootstrap_90pct_intervals"] = bootstrap(group, args.bootstrap, args.seed)
        result["groups"][name] = est
        b = est["bootstrap_90pct_intervals"]
        print(f"{name:52s} n={len(group):2d}  median {est['p50']:.2f} [{b['p50'][0]:.2f}-{b['p50'][1]:.2f}]  "
              f"p80 {est['p80']:.2f} [{b['p80'][0]:.2f}-{b['p80'][1]:.2f}]  "
              f"p90 {est['p90']:.2f} [{b['p90'][0]:.2f}-{b['p90'][1]:.2f}]")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
