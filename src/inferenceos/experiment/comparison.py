"""Statistically transparent baseline vs candidate comparison.

The comparison is multi-dimensional: the diagnosed bottleneck, the overall workload
objective, and important resource constraints are all evaluated. No universal
ranking (throughput beats memory, or vice versa) is hardcoded — the diagnosis and
the candidate's expected metric determine which effect matters.
"""

from __future__ import annotations

import math

from inferenceos.contracts.experiment import (
    ExperimentComparison,
    ExperimentMeasurement,
    ExperimentPolicy,
    MeasurementDelta,
    QualityValidationResult,
)
from inferenceos.contracts.profiling import MetricStats

HIGHER = "higher_is_better"
LOWER = "lower_is_better"

_OBJECTIVE: dict[str, str] = {
    "decode_tokens_per_sec": HIGHER,
    "prompt_tokens_per_sec": HIGHER,
    "total_latency_s": LOWER,
    "ttft_s": LOWER,
}
_RESOURCE: dict[str, str] = {
    "peak_rss_bytes": LOWER,
    "process_cpu_percent": LOWER,
    "available_ram_bytes": HIGHER,
    "committed_bytes": LOWER,
}
_BOTTLENECK_EXTRA: dict[str, str] = {
    "hard_page_faults_per_sec": LOWER,
    "gpu_memory_used_bytes": LOWER,
    "gpu_utilization_percent": HIGHER,
    "process_cpu_percent": LOWER,
}

_DELTA_ORDER: tuple[str, ...] = (
    "decode_tokens_per_sec",
    "prompt_tokens_per_sec",
    "total_latency_s",
    "ttft_s",
    "peak_rss_bytes",
    "hard_page_faults_per_sec",
    "process_cpu_percent",
    "available_ram_bytes",
    "committed_bytes",
    "gpu_memory_used_bytes",
    "gpu_utilization_percent",
)


def compare(
    *,
    baseline: ExperimentMeasurement,
    candidate: ExperimentMeasurement,
    bottleneck_metric: str | None,
    bottleneck_direction: str | None,
    quality: QualityValidationResult,
    policy: ExperimentPolicy,
    diagnosed_bottleneck: str | None = None,
) -> ExperimentComparison:
    direction = _direction(bottleneck_direction)
    primary = _primary_metrics(bottleneck_metric, direction, diagnosed_bottleneck)
    metrics = list(_DELTA_ORDER)
    for extra in primary:
        if extra not in metrics:
            metrics.append(extra)

    deltas: dict[str, MeasurementDelta] = {}
    for metric in metrics:
        base_stats = baseline.metrics.get(metric)
        cand_stats = candidate.metrics.get(metric)
        role, metric_direction = _role(metric, primary)
        deltas[metric] = _delta(
            metric, role, metric_direction, base_stats, cand_stats, policy
        )

    primary_deltas = [d for d in deltas.values() if d.role == "bottleneck_primary"]
    secondary = [d for d in deltas.values() if d.role == "bottleneck_secondary"]
    objective_deltas = [d for d in deltas.values() if d.role == "objective"]
    resource_deltas = [d for d in deltas.values() if d.role == "resource"]

    measurable_primary = [
        d
        for d in primary_deltas
        if d.baseline_mean is not None and d.candidate_mean is not None
    ]
    decisive = measurable_primary if measurable_primary else secondary
    bottleneck_improved = any(d.improved is True for d in decisive)
    bottleneck_regressed = any(d.regressed is True for d in decisive)
    objective_improved = any(d.improved is True for d in objective_deltas)
    objective_regressed = any(d.regressed is True for d in objective_deltas)
    resource_regressed = any(d.regressed is True for d in resource_deltas)

    cv_values = [
        stats.cv
        for stats in (baseline.metrics.get(metric) for metric in metrics)
        if stats is not None and stats.cv is not None
    ]
    max_cv = max(cv_values) if cv_values else None
    uncertainty = _uncertainty(max_cv, min(baseline.valid_runs, candidate.valid_runs))

    rationale = [
        f"bottleneck metric={bottleneck_metric} direction={direction or 'unknown'}: "
        f"improved={bottleneck_improved} regressed={bottleneck_regressed}",
        f"objective: improved={objective_improved} regressed={objective_regressed}",
        f"resource regression={resource_regressed}",
        f"quality={quality.status.value}",
        f"max observed CV={max_cv if max_cv is not None else 'n/a'}",
    ]
    return ExperimentComparison(
        deltas=deltas,
        bottleneck_metric=bottleneck_metric,
        bottleneck_direction=direction,
        bottleneck_improved=bottleneck_improved,
        bottleneck_regressed=bottleneck_regressed,
        bottleneck_persistent=not bottleneck_improved and not bottleneck_regressed,
        objective_improved=objective_improved,
        objective_regressed=objective_regressed,
        resource_regressed=resource_regressed,
        quality_change=quality.status.value,
        sample_count=min(baseline.valid_runs, candidate.valid_runs),
        decision_thresholds={
            "min_effect": policy.min_effect,
            "noise_tolerance": policy.noise_tolerance,
        },
        uncertainty=uncertainty,
        rationale=rationale,
    )


def _direction(bottleneck_direction: str | None) -> str | None:
    if bottleneck_direction == "decrease":
        return LOWER
    if bottleneck_direction == "increase":
        return HIGHER
    return None


def _primary_metrics(
    bottleneck_metric: str | None, direction: str | None, diagnosed_bottleneck: str | None
) -> dict[str, str]:
    """Metric(s) that correspond to the diagnosed bottleneck.

    The candidate's expected metric is always primary; the diagnosis kind adds the
    mechanism metrics the Phase 4 finding is actually about, so an unrelated side
    effect (e.g. VRAM moving for a paging diagnosis) cannot be reported as success.
    """
    metrics: dict[str, str] = {}
    if bottleneck_metric and direction is not None:
        metrics[bottleneck_metric] = direction
    kind = diagnosed_bottleneck or ""
    if kind in {"paging", "memory_pressure"}:
        metrics.setdefault("hard_page_faults_per_sec", LOWER)
        metrics.setdefault("available_ram_bytes", HIGHER)
    elif kind == "model_footprint":
        metrics.setdefault("peak_rss_bytes", LOWER)
    elif kind == "cpu":
        metrics.setdefault("process_cpu_percent", LOWER)
    elif kind == "gpu_memory_limit":
        metrics.setdefault("gpu_memory_used_bytes", LOWER)
    return metrics


def _role(metric: str, primary: dict[str, str]) -> tuple[str, str]:
    if metric in primary:
        return "bottleneck_primary", primary[metric]
    if metric in _BOTTLENECK_EXTRA:
        return "bottleneck_secondary", _BOTTLENECK_EXTRA[metric]
    if metric in _OBJECTIVE:
        return "objective", _OBJECTIVE[metric]
    if metric in _RESOURCE:
        return "resource", _RESOURCE[metric]
    return "other", HIGHER


def _delta(
    metric: str,
    role: str,
    direction: str,
    base: MetricStats | None,
    cand: MetricStats | None,
    policy: ExperimentPolicy,
) -> MeasurementDelta:
    base_mean = base.mean if base is not None else None
    cand_mean = cand.mean if cand is not None else None
    base_stdev = base.stdev if base is not None else None
    cand_stdev = cand.stdev if cand is not None else None
    sample_count = min(
        base.count if base is not None else 0, cand.count if cand is not None else 0
    )
    if base_mean is None or cand_mean is None:
        return MeasurementDelta(
            metric=metric,
            role=role,
            direction=direction,
            baseline_mean=base_mean,
            candidate_mean=cand_mean,
            baseline_stdev=base_stdev,
            baseline_cv=base.cv if base is not None else None,
            candidate_cv=cand.cv if cand is not None else None,
            sample_count=sample_count,
        )
    absolute = cand_mean - base_mean
    relative = absolute / abs(base_mean) if base_mean != 0.0 else None
    pooled = _pooled_stdev(base_stdev, cand_stdev)
    noise = max(policy.min_effect, policy.noise_tolerance) * abs(base_mean)
    if pooled is not None:
        noise = max(noise, pooled)
    significant = abs(absolute) >= noise and absolute != 0.0
    improved = regressed = None
    if significant:
        improved = (direction == LOWER and absolute < 0) or (direction == HIGHER and absolute > 0)
        regressed = not improved
    return MeasurementDelta(
        metric=metric,
        role=role,
        direction=direction,
        baseline_mean=base_mean,
        candidate_mean=cand_mean,
        absolute_delta=absolute,
        relative_delta=relative,
        baseline_stdev=base_stdev,
        baseline_cv=base.cv if base is not None else None,
        candidate_cv=cand.cv if cand is not None else None,
        sample_count=sample_count,
        noise_band=noise,
        improved=improved,
        regressed=regressed,
        significant=significant,
    )


def _pooled_stdev(a: float | None, b: float | None) -> float | None:
    values = [value for value in (a, b) if value is not None]
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values) / len(values))


def _uncertainty(max_cv: float | None, samples: int) -> str:
    if samples <= 1:
        return "insufficient samples; effect cannot be distinguished from noise"
    if max_cv is None:
        return f"{samples} samples; variance not estimated"
    if max_cv <= 0.05:
        return f"{samples} samples; low measured variance (max CV {max_cv:.3f})"
    if max_cv <= 0.15:
        return f"{samples} samples; moderate measured variance (max CV {max_cv:.3f})"
    return f"{samples} samples; high measured variance (max CV {max_cv:.3f})"
