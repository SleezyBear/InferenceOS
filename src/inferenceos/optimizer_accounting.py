"""H4 optimizer self-overhead attribution and conservative net-benefit accounting.

This is a *derived* accounting layer over measurements that already exist
(``RuntimeOverhead``, ``LoopOverhead``, ``ResourceImpact`` and the experiment/baseline
metric maps). It introduces no timing in the inference hot path.

Boundary (see also ``contracts/accounting.py``):

* **optimizer overhead** — time InferenceOS spends observing, diagnosing, generating
  candidates, reasoning/ranking, checkpointing, validating and persisting *because of*
  optimization. It never includes target inference time.
* **target benefit** — improvement in a target metric (decode throughput, latency,
  TTFT, memory) measured the same way before and after.
* **measurement distortion** — whether the observation/optimization itself perturbed
  the target; this is reported separately and, when material, forces
  ``INSUFFICIENT_EVIDENCE`` rather than being subtracted from benefit.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from inferenceos.contracts.accounting import (
    BenefitRelationship,
    MeasurementDistortion,
    NetBenefitAssessment,
    NetBenefitStatus,
    OverheadAttribution,
)

_HIGHER = "higher_is_better"
_LOWER = "lower_is_better"


@dataclass(frozen=True)
class _MetricSpec:
    direction: str
    unit: str
    priority: int


# Known *target-outcome* benefit metrics, reusing the project's existing metric
# vocabulary. System-state signals (memory_pressure, available_ram_bytes, hard page
# faults) are deliberately excluded: they describe the host, not the target workload's
# optimization outcome, and must not be presented as a target benefit.
# Priority order is the documented auto-selection order: throughput, then latency/TTFT,
# then target memory. Adding a metric here is the only way to make it comparable.
_KNOWN_METRICS: dict[str, _MetricSpec] = {
    "decode_tokens_per_sec": _MetricSpec(_HIGHER, "tokens_per_sec", 0),
    "prompt_tokens_per_sec": _MetricSpec(_HIGHER, "tokens_per_sec", 1),
    "total_latency_s": _MetricSpec(_LOWER, "seconds", 2),
    "ttft_s": _MetricSpec(_LOWER, "seconds", 3),
    "peak_rss_bytes": _MetricSpec(_LOWER, "bytes", 4),
    "final_rss_bytes": _MetricSpec(_LOWER, "bytes", 5),
}


@dataclass(frozen=True)
class AccountingPolicy:
    """Explicit, conservative H4 thresholds.

    ``benefit_min_relative`` reuses the existing Phase 7 experiment minimum-effect
    threshold (default 5%) rather than inventing a new one. ``overhead_fraction_max``
    (default 20%) is the smallest defensible bound consistent with the existing
    overhead budgets (Phase 6/10 ``max_optimizer_overhead_s`` are ~120 s against
    multi-hundred-second target windows, i.e. well under 20%); above it the optimizer's
    own cost is treated as potentially dominant. ``distortion_tolerance`` reuses the
    runtime deviation threshold (default 25%).
    """

    benefit_min_relative: float = 0.05
    overhead_fraction_max: float = 0.20
    distortion_tolerance: float = 0.25

    @classmethod
    def from_settings(cls, settings: object) -> "AccountingPolicy":
        benefit = getattr(settings, "phase7_min_effect", 0.05)
        try:
            benefit_min = float(benefit)
        except (TypeError, ValueError):
            benefit_min = 0.05
        return cls(benefit_min_relative=benefit_min)


def build_overhead_attribution(
    *,
    inclusive_overhead_s: float,
    components: dict[str, float] | None = None,
    baseline_target_workload_s: float | None = None,
    decision_count: int = 0,
    notes: list[str] | None = None,
) -> OverheadAttribution:
    """Build the inclusive/exclusive overhead attribution without double counting.

    The authoritative value is ``inclusive_overhead_s``; ``components`` are recorded
    for attribution only. When their sum differs from the inclusive value, the record
    is flagged ``nested=True`` rather than adding overlapping timers together.
    """
    clean_components = {
        name: max(float(value), 0.0) for name, value in (components or {}).items()
    }
    components_sum = sum(clean_components.values())
    nested = abs(components_sum - inclusive_overhead_s) > 1e-9
    merged_notes = list(notes or [])
    if nested:
        merged_notes.append(
            "component timers are exclusive attribution, not a decomposition of "
            "inclusive_overhead_s (no double counting applied)"
        )
    fraction: float | None = None
    if baseline_target_workload_s is not None and baseline_target_workload_s > 0.0:
        fraction = inclusive_overhead_s / baseline_target_workload_s
    else:
        merged_notes.append(
            "baseline target-workload duration unavailable; overhead fraction not computed"
        )
    return OverheadAttribution(
        inclusive_overhead_s=max(float(inclusive_overhead_s), 0.0),
        components=clean_components,
        components_sum_s=components_sum,
        baseline_target_workload_s=baseline_target_workload_s,
        overhead_fraction=fraction,
        decision_count=max(int(decision_count), 0),
        nested=nested,
        notes=merged_notes,
    )


def distortion_from_stability(
    *,
    cv: float | None,
    within_tolerance: bool | None,
    tolerance: float,
    metric: str = "decode_tokens_per_sec",
    detail: str = "",
) -> MeasurementDistortion:
    """Represent the existing runtime throughput-stability evidence as distortion."""
    if cv is None:
        return MeasurementDistortion(
            available=False,
            metric=metric,
            direction=_HIGHER,
            tolerance=tolerance,
            within_tolerance=within_tolerance,
            material=None,
            detail=detail or "no throughput stability sample available",
        )
    return MeasurementDistortion(
        available=True,
        metric=metric,
        direction=_HIGHER,
        tolerance=tolerance,
        cv=cv,
        within_tolerance=within_tolerance,
        material=(not within_tolerance) if within_tolerance is not None else None,
        detail=detail or f"observed throughput cv={cv:.3f}",
    )


def _insufficient(
    *,
    overhead: OverheadAttribution | None,
    benefit: BenefitRelationship | None = None,
    distortion: MeasurementDistortion | None = None,
    policy: AccountingPolicy,
    reasons: list[str],
    notes: list[str] | None = None,
    comparable: bool = False,
) -> NetBenefitAssessment:
    return NetBenefitAssessment(
        assessment_id=f"nba-{uuid.uuid4().hex[:10]}",
        created_at=datetime.now(timezone.utc),
        status=NetBenefitStatus.INSUFFICIENT_EVIDENCE,
        benefit=benefit,
        overhead=overhead,
        distortion=distortion,
        benefit_min_relative=policy.benefit_min_relative,
        overhead_fraction_max=policy.overhead_fraction_max,
        benefit_ratio=benefit.relative_change if benefit is not None else None,
        overhead_fraction=overhead.overhead_fraction if overhead is not None else None,
        comparable=comparable and benefit is not None,
        reasons=reasons,
        notes=list(notes or []),
    )


def assess_net_benefit(
    *,
    baseline_metrics: dict[str, float],
    measured_metrics: dict[str, float],
    overhead: OverheadAttribution | None,
    distortion: MeasurementDistortion | None = None,
    policy: AccountingPolicy | None = None,
    preferred_metric: str | None = None,
) -> NetBenefitAssessment:
    """Conservatively assess whether an optimization benefit outweighed overhead.

    Returns ``INSUFFICIENT_EVIDENCE`` for: no valid baseline/measured benefit, missing
    overhead, an incomparable/unknown benefit metric, an unavailable overhead fraction,
    or material measurement distortion. It never converts uncertainty into a positive
    result.
    """
    policy = policy or AccountingPolicy()
    notes: list[str] = []

    # --- benefit metric selection -------------------------------------- #
    if preferred_metric is not None and preferred_metric not in _KNOWN_METRICS:
        benefit = BenefitRelationship(
            metric=preferred_metric,
            unit="unknown",
            comparable=False,
            available=False,
            detail="metric has no comparable target-benefit definition",
        )
        return _insufficient(
            overhead=overhead,
            benefit=benefit,
            distortion=distortion,
            policy=policy,
            reasons=[f"incomparable benefit metric: {preferred_metric}"],
        )

    metric = preferred_metric or _select_metric(baseline_metrics, measured_metrics)
    if metric is None:
        return _insufficient(
            overhead=overhead,
            distortion=distortion,
            policy=policy,
            reasons=["no comparable benefit metric present in both baseline and measurement"],
        )

    spec = _KNOWN_METRICS[metric]
    baseline_value = baseline_metrics.get(metric)
    measured_value = measured_metrics.get(metric)
    if baseline_value is None or measured_value is None:
        benefit = BenefitRelationship(
            metric=metric,
            direction=spec.direction,
            unit=spec.unit,
            baseline_value=baseline_value,
            measured_value=measured_value,
            comparable=True,
            available=False,
            detail="metric missing from baseline or measurement",
        )
        return _insufficient(
            overhead=overhead,
            benefit=benefit,
            distortion=distortion,
            policy=policy,
            reasons=[f"benefit metric unavailable: {metric}"],
            comparable=True,
        )
    if baseline_value == 0.0:
        benefit = BenefitRelationship(
            metric=metric,
            direction=spec.direction,
            unit=spec.unit,
            baseline_value=baseline_value,
            measured_value=measured_value,
            comparable=True,
            available=False,
            detail="baseline value is zero; relative benefit undefined",
        )
        return _insufficient(
            overhead=overhead,
            benefit=benefit,
            distortion=distortion,
            policy=policy,
            reasons=[f"baseline value is zero for {metric}"],
            comparable=True,
        )

    absolute = measured_value - baseline_value
    if spec.direction == _HIGHER:
        improvement = (measured_value - baseline_value) / abs(baseline_value)
    else:
        improvement = (baseline_value - measured_value) / abs(baseline_value)
    benefit = BenefitRelationship(
        metric=metric,
        direction=spec.direction,
        unit=spec.unit,
        baseline_value=baseline_value,
        measured_value=measured_value,
        absolute_change=absolute,
        relative_change=improvement,
        comparable=True,
        available=True,
        detail=(
            f"baseline {baseline_value:.6g} -> measured {measured_value:.6g} "
            f"({improvement:+.1%})"
        ),
    )

    # --- overhead required --------------------------------------------- #
    if overhead is None:
        return _insufficient(
            overhead=None,
            benefit=benefit,
            distortion=distortion,
            policy=policy,
            reasons=["optimizer overhead measurement unavailable"],
            comparable=True,
        )

    # --- measurement distortion is reported, never netted out ----------- #
    if distortion is not None and distortion.material is True:
        return _insufficient(
            overhead=overhead,
            benefit=benefit,
            distortion=distortion,
            policy=policy,
            reasons=["material measurement distortion; benefit/overhead not separable"],
            comparable=True,
        )

    # --- overhead fraction is required to judge dominance --------------- #
    if overhead.overhead_fraction is None:
        return _insufficient(
            overhead=overhead,
            benefit=benefit,
            distortion=distortion,
            policy=policy,
            reasons=[
                "baseline target-workload duration unavailable; overhead fraction "
                "not comparable to benefit"
            ],
            comparable=True,
        )

    ratio = improvement
    fraction = overhead.overhead_fraction
    if ratio >= policy.benefit_min_relative and fraction <= policy.overhead_fraction_max:
        status = NetBenefitStatus.BENEFICIAL
        reasons = [
            f"benefit {ratio:+.1%} >= {policy.benefit_min_relative:.0%} and overhead "
            f"fraction {fraction:.1%} <= {policy.overhead_fraction_max:.0%}"
        ]
    elif fraction > policy.overhead_fraction_max and (
        ratio < policy.benefit_min_relative or ratio < fraction
    ):
        status = NetBenefitStatus.OVERHEAD_DOMINANT
        reasons = [
            f"overhead fraction {fraction:.1%} exceeds {policy.overhead_fraction_max:.0%} "
            f"and is not outweighed by benefit {ratio:+.1%}"
        ]
    else:
        status = NetBenefitStatus.NEUTRAL
        reasons = [
            f"benefit {ratio:+.1%} and overhead fraction {fraction:.1%} are within "
            "bound but do not meet the beneficial criteria"
        ]

    notes.append(
        "benefit ratio and overhead fraction are dimensionless; no heterogeneous "
        "quantities are subtracted"
    )
    return NetBenefitAssessment(
        assessment_id=f"nba-{uuid.uuid4().hex[:10]}",
        created_at=datetime.now(timezone.utc),
        status=status,
        benefit=benefit,
        overhead=overhead,
        distortion=distortion,
        benefit_ratio=ratio,
        overhead_fraction=fraction,
        benefit_min_relative=policy.benefit_min_relative,
        overhead_fraction_max=policy.overhead_fraction_max,
        net_value=None,
        net_unit=None,
        comparable=True,
        enforceable=False,
        reasons=reasons,
        notes=notes,
    )


def _select_metric(
    baseline_metrics: dict[str, float], measured_metrics: dict[str, float]
) -> str | None:
    candidates = [
        (spec.priority, name)
        for name, spec in _KNOWN_METRICS.items()
        if baseline_metrics.get(name) is not None and measured_metrics.get(name) is not None
    ]
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def metric_direction(metric: str) -> str | None:
    spec = _KNOWN_METRICS.get(metric)
    return spec.direction if spec is not None else None


def metric_unit(metric: str) -> str | None:
    spec = _KNOWN_METRICS.get(metric)
    return spec.unit if spec is not None else None
