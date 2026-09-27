"""Multi-objective comparison and exactly-one-value selection.

Selection reuses the existing Phase 7 comparison and decision contracts
(``experiment.comparison.compare`` / ``experiment.decision.decide``) so the same
effect/noise thresholds and regression rules govern every candidate. A candidate is
only acceptable when it:

* is safe and valid (no OOM/VRAM/paging/invalid measurement),
* improves the diagnosed bottleneck beyond noise, and
* does not regress the workload objective (latency/TTFT), quality, or resources.

The maximum feasible configuration is never preferred automatically: among acceptable
candidates whose improvement is within the configured diminishing-returns margin of the
best, the smallest (cheapest) value wins.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from inferenceos.contracts.enums import QualityStatus, SearchSelectionKind
from inferenceos.contracts.experiment import (
    ExperimentComparison,
    ExperimentDecision,
    ExperimentMeasurement,
    ExperimentPolicy,
    QualityValidationResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.search import GpuOffloadSelection
from inferenceos.experiment.comparison import compare
from inferenceos.experiment.decision import decide
from inferenceos.optimization.search.evaluation import PointMeasurement
from inferenceos.optimization.search.policy import SearchPolicy

BOTTLENECK_METRIC = "decode_tokens_per_sec"
BOTTLENECK_DIRECTION = "increase"


@dataclass(frozen=True)
class CandidateOutcome:
    point: PointMeasurement
    comparison: ExperimentComparison
    decision: ExperimentDecision
    quality: QualityValidationResult


@dataclass(frozen=True)
class SelectionOutcome:
    selection: GpuOffloadSelection
    outcomes: list[CandidateOutcome]


def neutral_quality(reason: str) -> QualityValidationResult:
    """A weight-preserving runtime change does not alter the model or sampling."""
    return QualityValidationResult(
        status=QualityStatus.OK,
        validated=True,
        execution_ok=True,
        structural_ok=True,
        reference_available=False,
        rationale=[reason],
    )


def _as_measurement(
    point: PointMeasurement, configuration: RuntimeConfiguration, phase: str
) -> ExperimentMeasurement:
    if point.measurement is not None:
        return point.measurement
    return ExperimentMeasurement(
        measurement_id=f"meas-{uuid.uuid4().hex[:10]}",
        phase=phase,
        configuration=configuration,
        repetitions=point.samples,
        valid_runs=point.samples,
        metrics=dict(point.metrics),
        generated_texts=[],
        notes=list(point.reasons),
    )


def _improvement(outcome: CandidateOutcome) -> float:
    delta = outcome.comparison.deltas.get(BOTTLENECK_METRIC)
    if delta is None or delta.relative_delta is None:
        return 0.0
    return float(delta.relative_delta)


def _relative(outcome: CandidateOutcome, metric: str) -> float | None:
    delta = outcome.comparison.deltas.get(metric)
    return None if delta is None or delta.relative_delta is None else float(delta.relative_delta)


def evaluate_point(
    baseline: PointMeasurement,
    point: PointMeasurement,
    *,
    baseline_configuration: RuntimeConfiguration,
    candidate_configuration: RuntimeConfiguration,
    experiment_policy: ExperimentPolicy,
    quality: QualityValidationResult | None,
) -> CandidateOutcome:
    from inferenceos.contracts.enums import RestorationStatus
    from inferenceos.contracts.experiment import ApplicationResult

    base_measurement = _as_measurement(baseline, baseline_configuration, "baseline")
    cand_measurement = _as_measurement(point, candidate_configuration, "candidate")
    quality_result = quality or neutral_quality(
        "GPU-offload layer change does not modify model weights or sampling"
    )
    comparison = compare(
        baseline=base_measurement,
        candidate=cand_measurement,
        bottleneck_metric=BOTTLENECK_METRIC,
        bottleneck_direction=BOTTLENECK_DIRECTION,
        quality=quality_result,
        policy=experiment_policy,
        diagnosed_bottleneck="gpu_underutilization",
    )
    # The evaluator applied the configuration; present a successful application so the
    # decision focuses on the measurement rather than the (already-completed) apply.
    application = ApplicationResult(
        applied=True,
        action_id="runtime.increase_gpu_offload_layers",
        supported=True,
        reason="candidate configuration applied for measurement",
    )
    decision = decide(
        comparison=comparison,
        quality=quality_result,
        application=application,
        restoration_status=RestorationStatus.NOT_REQUIRED,
        measurement_valid=point.valid,
        precondition_ok=True,
        precondition_kind=None,
        precondition_reason="",
        policy=experiment_policy,
        errors=[],
    )
    return CandidateOutcome(
        point=point, comparison=comparison, decision=decision, quality=quality_result
    )


def select_configuration(
    *,
    baseline: PointMeasurement,
    candidates: list[PointMeasurement],
    baseline_configuration: RuntimeConfiguration,
    candidate_configurations: dict[int, RuntimeConfiguration],
    experiment_policy: ExperimentPolicy,
    search_policy: SearchPolicy,
    quality_by_layers: dict[int, QualityValidationResult] | None = None,
) -> SelectionOutcome:
    quality_by_layers = quality_by_layers or {}
    outcomes: list[CandidateOutcome] = []
    for point in candidates:
        outcome = evaluate_point(
            baseline,
            point,
            baseline_configuration=baseline_configuration,
            candidate_configuration=candidate_configurations[point.layers],
            experiment_policy=experiment_policy,
            quality=quality_by_layers.get(point.layers),
        )
        outcomes.append(outcome)

    acceptable = [
        outcome
        for outcome in outcomes
        if outcome.point.ok and outcome.decision.keep and outcome.point.layers > baseline.layers
    ]

    comparisons = [_describe(outcome) for outcome in outcomes]
    baseline_layers = int(baseline.layers)

    if not acceptable:
        reason = (
            "no candidate configuration improved the diagnosed GPU-underutilization "
            "bottleneck beyond noise/tolerance without an objective, quality or resource "
            "regression; the current configuration is retained"
        )
        return SelectionOutcome(
            selection=GpuOffloadSelection(
                kind=SearchSelectionKind.NO_CHANGE,
                selected_layers=None,
                baseline_layers=baseline_layers,
                reason=reason,
                comparisons=comparisons,
            ),
            outcomes=outcomes,
        )

    best_improvement = max(_improvement(outcome) for outcome in acceptable)
    margin = search_policy.diminishing_returns_margin
    equivalent = [
        outcome
        for outcome in acceptable
        if best_improvement - _improvement(outcome) <= margin
    ]
    winner = min(equivalent, key=lambda outcome: outcome.point.layers)

    runner_up = max(
        (
            outcome
            for outcome in acceptable
            if outcome.point.layers != winner.point.layers
        ),
        key=lambda outcome: _improvement(outcome),
        default=None,
    )
    reason = (
        f"{winner.point.layers} layers produced the best safe measured tradeoff "
        f"({_improvement(winner) * 100:+.1f}% decode throughput vs the current "
        f"{baseline_layers}-layer baseline, no material objective/quality regression)"
    )
    if runner_up is not None:
        delta = best_improvement - _improvement(runner_up)
        if delta <= margin:
            reason += (
                f"; {runner_up.point.layers} layers provided no meaningful additional "
                f"benefit (within {margin:.0%})"
            )
        else:
            reason += f"; {runner_up.point.layers} layers was not measurably better"
    else:
        reason += "; no higher candidate improved further"

    return SelectionOutcome(
        selection=GpuOffloadSelection(
            kind=SearchSelectionKind.SELECT,
            selected_layers=int(winner.point.layers),
            baseline_layers=baseline_layers,
            reason=reason,
            comparisons=comparisons,
        ),
        outcomes=outcomes,
    )


def _describe(outcome: CandidateOutcome) -> str:
    throughput = _relative(outcome, "decode_tokens_per_sec")
    latency = _relative(outcome, "total_latency_s")
    vram = outcome.point.measured_vram_bytes

    def _fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:+.1f}%"

    return (
        f"layers={outcome.point.layers} status={outcome.point.status.value} "
        f"safe={outcome.point.safe} valid={outcome.point.valid} "
        f"decision={outcome.decision.kind.value} "
        f"decode={_fmt(throughput)} latency={_fmt(latency)} "
        f"vram={vram if vram is not None else 'n/a'}"
    )
