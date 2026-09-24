"""Transparent experiment decision policy.

A candidate is only successful when the intended bottleneck improves sufficiently,
the overall objective does not regress unacceptably, quality is preserved, the
effect is distinguishable from noise, and restoration succeeds. Hard safety/quality/
restoration failures dominate and are never overridden by an unrelated improvement.
"""

from __future__ import annotations

from inferenceos.contracts.enums import ExperimentDecisionKind, RestorationStatus
from inferenceos.contracts.experiment import (
    ApplicationResult,
    ExperimentComparison,
    ExperimentDecision,
    ExperimentPolicy,
    QualityValidationResult,
)


def decide(
    *,
    comparison: ExperimentComparison | None,
    quality: QualityValidationResult | None,
    application: ApplicationResult | None,
    restoration_status: RestorationStatus,
    measurement_valid: bool,
    precondition_ok: bool,
    precondition_kind: ExperimentDecisionKind | None,
    precondition_reason: str,
    policy: ExperimentPolicy,
    errors: list[str],
) -> ExperimentDecision:
    def build(
        kind: ExperimentDecisionKind,
        reason: str,
        *,
        confidence: float = 0.0,
        regression: bool = False,
        quality_regression: bool = False,
        bottleneck_improved: bool = False,
        evidence: list[str] | None = None,
    ) -> ExperimentDecision:
        return ExperimentDecision(
            kind=kind,
            keep=kind is ExperimentDecisionKind.IMPROVED,
            reason=reason,
            evidence=evidence or [],
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            regression=regression,
            quality_regression=quality_regression,
            bottleneck_improved=bottleneck_improved,
            restoration_status=restoration_status,
        )

    if not precondition_ok:
        kind = precondition_kind or ExperimentDecisionKind.PRECONDITION_FAILURE
        return build(
            kind,
            f"candidate not executed: {precondition_reason}",
            evidence=[precondition_reason],
        )

    if restoration_status is RestorationStatus.FAILED:
        return build(
            ExperimentDecisionKind.RESTORATION_FAILURE,
            "baseline restoration failed; experiment result is not trustworthy",
            evidence=["restoration.status=failed"],
        )

    if application is None or not application.applied:
        reason = application.reason if application is not None else "no application result"
        return build(
            ExperimentDecisionKind.APPLICATION_FAILURE,
            f"candidate could not be applied: {reason}",
            evidence=[reason],
        )

    if not measurement_valid:
        return build(
            ExperimentDecisionKind.MEASUREMENT_FAILURE,
            "insufficient valid measurements to compare; effect cannot be established",
            evidence=errors or ["measurement validity failed"],
        )

    assert comparison is not None and quality is not None  # guarded by measurement_valid

    evidence = _evidence(comparison, quality, policy)
    confidence = _confidence(comparison, quality, policy)

    if not quality.validated or quality.unacceptable_output:
        return build(
            ExperimentDecisionKind.QUALITY_REGRESSION,
            f"quality validation failed ({quality.status.value}); improvement is not acceptable",
            confidence=confidence,
            quality_regression=True,
            bottleneck_improved=comparison.bottleneck_improved,
            evidence=evidence,
        )

    if comparison.objective_regressed:
        return build(
            ExperimentDecisionKind.REGRESSION,
            "overall workload objective regressed beyond tolerance",
            confidence=confidence,
            regression=True,
            bottleneck_improved=comparison.bottleneck_improved,
            evidence=evidence,
        )

    if comparison.resource_regressed and not comparison.bottleneck_improved:
        return build(
            ExperimentDecisionKind.REGRESSION,
            "relevant resource pressure worsened and the bottleneck did not improve",
            confidence=confidence,
            regression=True,
            evidence=evidence,
        )

    if comparison.bottleneck_improved:
        return build(
            ExperimentDecisionKind.IMPROVED,
            "the diagnosed bottleneck improved beyond noise with no quality/objective regression",
            confidence=confidence,
            bottleneck_improved=True,
            evidence=evidence,
        )

    if comparison.bottleneck_regressed:
        return build(
            ExperimentDecisionKind.REGRESSION,
            "the diagnosed bottleneck worsened",
            confidence=confidence,
            regression=True,
            evidence=evidence,
        )

    return build(
        ExperimentDecisionKind.NO_MEANINGFUL_IMPROVEMENT,
        "measured effect is within noise/tolerance; the bottleneck persisted",
        confidence=confidence,
        evidence=evidence,
    )


def _evidence(
    comparison: ExperimentComparison,
    quality: QualityValidationResult,
    policy: ExperimentPolicy,
) -> list[str]:
    evidence: list[str] = [
        f"bottleneck={comparison.bottleneck_metric} "
        f"improved={comparison.bottleneck_improved} "
        f"regressed={comparison.bottleneck_regressed}",
        f"objective improved={comparison.objective_improved} "
        f"regressed={comparison.objective_regressed}",
        f"quality={quality.status.value} similarity={quality.similarity}",
        f"thresholds min_effect={policy.min_effect} noise_tolerance={policy.noise_tolerance}",
        f"uncertainty: {comparison.uncertainty}",
    ]
    for metric, delta in comparison.deltas.items():
        if delta.baseline_mean is None or delta.candidate_mean is None:
            continue
        relative = f"{delta.relative_delta:+.1%}" if delta.relative_delta is not None else "n/a"
        evidence.append(
            f"{metric} ({delta.role}): {delta.baseline_mean:.4g} -> "
            f"{delta.candidate_mean:.4g} ({relative}, significant={delta.significant})"
        )
    return evidence


def _confidence(
    comparison: ExperimentComparison,
    quality: QualityValidationResult,
    policy: ExperimentPolicy,
) -> float:
    sample_factor = min(1.0, comparison.sample_count / max(policy.measured_repetitions, 1))
    cv_values = [
        delta.baseline_cv
        for delta in comparison.deltas.values()
        if (delta.role.startswith("bottleneck") or delta.role == "objective")
        and delta.baseline_cv is not None
    ]
    max_cv = max(cv_values) if cv_values else 0.0
    quality_factor = 1.0 if quality.validated else 0.5
    return sample_factor * quality_factor * (1.0 - min(0.5, max_cv))
