"""Deterministic aggregation statistics for profiling metrics.

Statistics are observational only — no thresholds are applied here except the
explicit, documented reproducibility tolerance bands used by acceptance.
"""

from __future__ import annotations

import statistics
from typing import Sequence

from inferenceos.contracts.profiling import MetricStats

# Agreed relative-variance bands (coefficient of variation) for the Phase 3
# reproducibility acceptance. These are deliberately differentiated: decode
# throughput is expected to be tight, while model load (OS page cache / driver
# state) is expected to vary more. They are validated against observed data.
REPRODUCIBILITY_TOLERANCE: dict[str, float] = {
    "decode_tokens_per_sec": 0.10,
    "prompt_tokens_per_sec": 0.15,
    "prefill_latency_s": 0.15,
    "decode_latency_s": 0.15,
    "total_latency_s": 0.15,
    "ttft_s": 0.20,
    "model_load_time_s": 0.50,
}


def summarize(metric: str, values: Sequence[float | None]) -> MetricStats:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return MetricStats(metric=metric, count=0)
    mean = statistics.fmean(cleaned)
    median = statistics.median(cleaned)
    stdev = statistics.stdev(cleaned) if len(cleaned) >= 2 else None
    cv = (stdev / mean) if stdev is not None and mean != 0.0 else None
    return MetricStats(
        metric=metric,
        count=len(cleaned),
        mean=mean,
        median=median,
        stdev=stdev,
        min=min(cleaned),
        max=max(cleaned),
        cv=cv,
    )


def within_tolerance(stats: MetricStats, tolerance: float) -> bool:
    """True if the metric's observed coefficient of variation is within the band.

    A metric with fewer than two samples or no spread (cv is None) is considered
    within tolerance because reproducibility cannot be assessed from one sample;
    acceptance separately requires enough repetitions.
    """
    if stats.cv is None:
        return True
    return stats.cv <= tolerance
