"""Resource-aware GPU-offload search runner.

Orchestrates the full pipeline:

    resource inspection
      -> feasible-range construction
      -> deterministic candidate sampling
      -> per-candidate empirical evaluation (warm-up + repeated measurements)
      -> multi-objective comparison against the current configuration
      -> exactly-one-value selection (or NO_CHANGE)

The runner never applies a change to a long-lived environment; each measurement is
performed by the evaluator via the controlled harness. It never returns the maximum
feasible value unless the measurements show that the maximum is genuinely the best
tradeoff.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from inferenceos.contracts.enums import SearchSelectionKind
from inferenceos.contracts.experiment import (
    ExperimentPolicy,
    QualityValidationResult,
)
from inferenceos.contracts.search import (
    CandidatePointEvaluation,
    GpuOffloadResourceEnvelope,
    GpuOffloadSearchResult,
    GpuOffloadSelection,
    MeasurementAttemptEvidence,
)
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.search.evaluation import (
    MappingPointEvaluator,
    PointEvaluator,
    PointMeasurement,
    RuntimeConfigurationPointEvaluator,
)
from inferenceos.optimization.search.policy import SearchPolicy
from inferenceos.optimization.search.resources import envelope_from_context
from inferenceos.optimization.search.sampling import planned_probe, sample_layers
from inferenceos.optimization.search.selection import select_configuration


@dataclass(frozen=True)
class GpuOffloadPlan:
    """Read-only plan produced for Phase 6 candidate generation / diagnostics."""

    envelope: GpuOffloadResourceEnvelope
    sampled_layers: list[int]
    planned_probe: int | None


def plan_gpu_offload(
    ctx: OptimizationContext, policy: SearchPolicy, *, now: object | None = None
) -> GpuOffloadPlan:
    import datetime as _datetime

    moment = now if isinstance(now, _datetime.datetime) else None
    envelope = envelope_from_context(ctx, policy, now=moment)
    current = envelope.current_layers
    feasible_max = envelope.feasible_max
    if not envelope.sufficient or current is None or feasible_max is None:
        return GpuOffloadPlan(envelope=envelope, sampled_layers=[], planned_probe=None)
    samples = sample_layers(current, feasible_max, policy)
    probe = planned_probe(current, feasible_max, policy)
    return GpuOffloadPlan(
        envelope=envelope,
        sampled_layers=samples,
        planned_probe=probe if probe > current else None,
    )


def _point_contract(
    point: PointMeasurement,
    *,
    decision: str | None,
    evaluated: bool = True,
    relative_improvement: float | None = None,
) -> CandidatePointEvaluation:
    return CandidatePointEvaluation(
        layers=point.layers,
        evaluated=evaluated,
        status=point.status,
        valid=point.valid,
        safe=point.safe,
        samples=point.samples,
        warmup_runs=point.warmup_runs,
        metrics=dict(point.metrics),
        decision=decision,
        reasons=list(point.reasons),
        measured_vram_bytes=point.measured_vram_bytes,
        relative_improvement=relative_improvement,
        attempts=point.attempts,
        valid_attempts=point.valid_attempts,
        invalid_attempts=point.invalid_attempts,
        attempt_records=[
            MeasurementAttemptEvidence(
                attempt=record.attempt,
                status=record.status,
                valid=record.valid,
                safe=record.safe,
                samples=record.samples,
                warmup_runs=record.warmup_runs,
                decode_tokens_per_sec=record.decode_tokens_per_sec,
                total_latency_s=record.total_latency_s,
                reasons=list(record.reasons),
            )
            for record in point.attempt_records
        ],
        stabilized=point.stabilized,
        contamination_reasons=list(point.contamination_reasons),
    )


def run_gpu_offload_search(
    *,
    envelope: GpuOffloadResourceEnvelope,
    evaluator: PointEvaluator,
    search_policy: SearchPolicy,
    experiment_policy: ExperimentPolicy,
    model_ref: str | None = None,
    quality_by_layers: dict[int, QualityValidationResult] | None = None,
    quality_evaluator: (
        Callable[[PointMeasurement, PointMeasurement], QualityValidationResult | None] | None
    ) = None,
    overhead_s: float = 0.0,
) -> GpuOffloadSearchResult:
    started = time.perf_counter()
    search_id = f"search-{uuid.uuid4().hex[:12]}"
    current = envelope.current_layers
    feasible_max = envelope.feasible_max
    limitations: list[str] = list(envelope.reasons)

    def _result(
        *,
        selection: GpuOffloadSelection,
        points: list[CandidatePointEvaluation],
        sampled: list[int],
        requested: list[int],
    ) -> GpuOffloadSearchResult:
        return GpuOffloadSearchResult(
            search_id=search_id,
            model_ref=model_ref,
            context_length=envelope.context_length,
            resource=envelope,
            sampled_layers=sampled,
            requested_layers=requested,
            points=points,
            selection=selection,
            overhead_s=overhead_s + (time.perf_counter() - started),
            environment_restored=True,
            environments_mutated=False,
            limitations=limitations,
            summary=selection.reason,
        )

    if current is None:
        return _result(
            selection=GpuOffloadSelection(
                kind=SearchSelectionKind.NO_CHANGE,
                baseline_layers=0,
                reason="current GPU-offload layer count unavailable; no search performed",
            ),
            points=[],
            sampled=[],
            requested=[],
        )

    if not envelope.sufficient or feasible_max is None or feasible_max <= current:
        if envelope.sufficient:
            reason = (
                f"no safe higher configuration exists above the current {current} layers "
                f"within the resource envelope; the current configuration is retained"
            )
        else:
            reason = (
                "resource envelope insufficient to establish a safe feasible range: "
                + "; ".join(envelope.reasons)
            )
        return _result(
            selection=GpuOffloadSelection(
                kind=SearchSelectionKind.NO_CHANGE,
                baseline_layers=int(current),
                reason=reason,
            ),
            points=[],
            sampled=[int(current)],
            requested=[],
        )

    sampled = sample_layers(current, feasible_max, search_policy)
    baseline = evaluator.measure(current)
    baseline_configuration = evaluator.configuration_for(current)

    if not baseline.ok:
        reason = (
            f"the current {current}-layer baseline could not be measured validly/safely "
            f"({baseline.status.value}); no candidate can be compared"
        )
        limitations.append(reason)
        return _result(
            selection=GpuOffloadSelection(
                kind=SearchSelectionKind.NO_CHANGE,
                baseline_layers=int(current),
                reason=reason,
            ),
            points=[_point_contract(baseline, decision="baseline")],
            sampled=sampled,
            requested=[],
        )

    requested = [value for value in sampled if value > current]
    candidates: list[PointMeasurement] = []
    for value in requested:
        point = evaluator.measure(value)
        candidates.append(point)
        if not point.ok:
            limitations.append(
                f"search stopped at {value} layers: {point.status.value} "
                f"({'; '.join(point.reasons[-2:])})"
            )
            break

    quality_map = dict(quality_by_layers or {})
    if quality_evaluator is not None:
        for point in candidates:
            if point.layers in quality_map:
                continue
            quality = quality_evaluator(baseline, point)
            if quality is not None:
                quality_map[point.layers] = quality

    candidate_configs = {
        point.layers: evaluator.configuration_for(point.layers) for point in candidates
    }
    outcome = select_configuration(
        baseline=baseline,
        candidates=candidates,
        baseline_configuration=baseline_configuration,
        candidate_configurations=candidate_configs,
        experiment_policy=experiment_policy,
        search_policy=search_policy,
        quality_by_layers=quality_map,
    )

    decision_by_layers = {
        item.point.layers: item.decision.kind.value for item in outcome.outcomes
    }
    improvement_by_layers: dict[int, float] = {}
    for item in outcome.outcomes:
        delta = item.comparison.deltas.get("decode_tokens_per_sec")
        if delta is not None and delta.relative_delta is not None:
            improvement_by_layers[item.point.layers] = float(delta.relative_delta)
    final_points: list[CandidatePointEvaluation] = [
        _point_contract(baseline, decision="baseline")
    ]
    for point in candidates:
        final_points.append(
            _point_contract(
                point,
                decision=decision_by_layers.get(point.layers),
                relative_improvement=improvement_by_layers.get(point.layers),
            )
        )

    selection = outcome.selection
    if selection.kind is SearchSelectionKind.SELECT:
        selected = selection.selected_layers
        assert selected is not None
        if selected > feasible_max:
            raise AssertionError(
                f"selected {selected} layers exceeds the safe feasible maximum {feasible_max}"
            )

    return _result(
        selection=selection,
        points=final_points,
        sampled=sampled,
        requested=requested,
    )


__all__ = [
    "GpuOffloadPlan",
    "MappingPointEvaluator",
    "PointEvaluator",
    "RuntimeConfigurationPointEvaluator",
    "plan_gpu_offload",
    "run_gpu_offload_search",
]
