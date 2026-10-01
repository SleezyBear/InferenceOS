"""Measuring for ``tune``: disturbed measurements are repeated, and the final setup is
confirmed against the starting one by alternating runs (step 5).

A measurement counts as *disturbed* when the whole machine's processor use during it
was clearly more than the model itself explains (the user or another program was
busy). Disturbed measurements are thrown away and repeated a bounded number of times.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from inferenceos.config import Settings
from inferenceos.contracts.enums import CandidateEvaluationStatus, QualityStatus, Situation
from inferenceos.contracts.experiment import (
    ExperimentComparison,
    ExperimentMeasurement,
    ExperimentPolicy,
    QualityValidationResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import MetricStats
from inferenceos.contracts.tune import ConfirmationResult
from inferenceos.experiment.comparison import compare
from inferenceos.experiment.measurement import (
    AGGREGATED_METRICS,
    MeasurementHarness,
    measure_configuration,
)
from inferenceos.experiment.quality import validate_quality
from inferenceos.optimization.goals import assess_comparison
from inferenceos.optimization.keep import ModeRules, judge_keep
from inferenceos.optimization.search.evaluation import PointEvaluator, PointMeasurement
from inferenceos.profiling.statistics import summarize
from inferenceos.profiling.workloads import BenchmarkWorkload

_BAD_QUALITY = {
    QualityStatus.QUALITY_REGRESSION,
    QualityStatus.EXECUTION_FAILURE,
    QualityStatus.STRUCTURAL_FAILURE,
}


def other_activity_percent(
    measurement: ExperimentMeasurement, logical_cores: int | None
) -> float | None:
    """Machine-wide processor use during a measurement that the model doesn't explain.

    ``system_cpu_percent`` is the whole machine (0-100); ``process_cpu_percent`` is the
    model's own peak use summed over cores, so dividing by the core count puts it on the
    same scale. Using the peak makes this err towards *not* calling a run disturbed.
    """
    system = measurement.metrics.get("system_cpu_percent")
    own = measurement.metrics.get("process_cpu_percent")
    if system is None or system.mean is None:
        return None
    cores = max(int(logical_cores or 1), 1)
    own_share = (own.mean or 0.0) / cores if own is not None and own.mean is not None else 0.0
    return max(float(system.mean) - float(own_share), 0.0)


def disturbance_reason(
    measurement: ExperimentMeasurement, *, logical_cores: int | None, limit_percent: float
) -> str | None:
    other = other_activity_percent(measurement, logical_cores)
    if other is not None and other > limit_percent:
        return (
            f"disturbed: other activity used about {other:.0f}% of the processor "
            f"(limit {limit_percent:.0f}%)"
        )
    return None


@dataclass
class DisturbanceWatch:
    """Settings and a running count of repeated (disturbed) measurements."""

    logical_cores: int | None
    limit_percent: float
    max_repeats: int
    repeats: int = 0
    on_repeat: Callable[[str], None] | None = None

    def note_repeat(self, reason: str) -> None:
        self.repeats += 1
        if self.on_repeat is not None:
            self.on_repeat(reason)


def measure_steady(
    *,
    harness: MeasurementHarness,
    model: ModelProfile,
    workload: BenchmarkWorkload,
    config: RuntimeConfiguration,
    policy: ExperimentPolicy,
    phase: str,
    watch: DisturbanceWatch,
    deadline: float | None = None,
) -> ExperimentMeasurement:
    """Measure ``config``; repeat a disturbed measurement up to ``watch.max_repeats``."""
    measurement = measure_configuration(
        profiler=harness,
        model=model,
        workload=workload,
        config=config,
        policy=policy,
        phase=phase,
        deadline=deadline,
    )
    for _ in range(max(watch.max_repeats, 0)):
        reason = disturbance_reason(
            measurement, logical_cores=watch.logical_cores, limit_percent=watch.limit_percent
        )
        if reason is None:
            break
        if deadline is not None and time.perf_counter() >= deadline:
            measurement = measurement.model_copy(
                update={"notes": [*measurement.notes, reason, "no time left to repeat it"]}
            )
            break
        watch.note_repeat(reason)
        measurement = measure_configuration(
            profiler=harness,
            model=model,
            workload=workload,
            config=config,
            policy=policy,
            phase=phase,
            deadline=deadline,
        )
        measurement = measurement.model_copy(
            update={"notes": [*measurement.notes, f"repeated after: {reason}"]}
        )
    return measurement


class DisturbanceAwareEvaluator:
    """Marks a disturbed layer-search measurement as not valid, so it is repeated.

    Wraps a point evaluator; the robust evaluator around it repeats invalid attempts.
    """

    def __init__(self, inner: PointEvaluator, watch: DisturbanceWatch) -> None:
        self._inner = inner
        self._watch = watch

    def configuration_for(self, layers: int) -> RuntimeConfiguration:
        return self._inner.configuration_for(layers)

    def measure(self, layers: int) -> PointMeasurement:
        point = self._inner.measure(layers)
        if not point.ok or point.measurement is None:
            return point
        reason = disturbance_reason(
            point.measurement,
            logical_cores=self._watch.logical_cores,
            limit_percent=self._watch.limit_percent,
        )
        if reason is None:
            return point
        self._watch.note_repeat(reason)
        return replace(
            point,
            status=CandidateEvaluationStatus.INVALID,
            valid=False,
            reasons=[*point.reasons, reason],
            contamination_reasons=[*point.contamination_reasons, reason],
            stabilized=False,
        )


def pool(measurements: Sequence[ExperimentMeasurement], *, phase: str) -> ExperimentMeasurement:
    """One measurement made of all the runs of several measurements of one setup."""
    if not measurements:
        raise ValueError("nothing to pool")
    runs = [run for item in measurements for run in item.runs]

    def value(run: object, name: str) -> float | None:
        metric = getattr(getattr(getattr(run, "observation"), "metrics"), name, None)
        return float(metric) if metric is not None else None

    metrics: dict[str, MetricStats] = {
        name: summarize(name, [value(run, name) for run in runs]) for name in AGGREGATED_METRICS
    }
    first = measurements[0]
    return ExperimentMeasurement(
        measurement_id=f"meas-{uuid.uuid4().hex[:10]}",
        created_at=datetime.now(timezone.utc),
        phase=phase,
        configuration=first.configuration,
        repetitions=len(runs),
        valid_runs=sum(item.valid_runs for item in measurements),
        invalid_runs=sum(item.invalid_runs for item in measurements),
        warmup_runs=sum(item.warmup_runs for item in measurements),
        total_runs=sum(item.total_runs for item in measurements),
        run_ids=[run_id for item in measurements for run_id in item.run_ids],
        metrics=metrics,
        run_metrics=[metric for item in measurements for metric in item.run_metrics],
        generated_texts=[text for item in measurements for text in item.generated_texts],
        metric_quality=dict(first.metric_quality),
        runs=runs,
        notes=[note for item in measurements for note in item.notes],
    )


def quality_ok(quality: QualityValidationResult) -> bool:
    return quality.status not in _BAD_QUALITY and not quality.quality_regression


@dataclass(frozen=True)
class Judged:
    """A before/after comparison scored against the goal."""

    comparison: ExperimentComparison
    quality: QualityValidationResult
    gain: float | None
    guard_breaches: list[str]
    technically_better: bool


def judge_pair(
    *,
    before: ExperimentMeasurement,
    after: ExperimentMeasurement,
    workload: BenchmarkWorkload,
    model: ModelProfile,
    policy: ExperimentPolicy,
    settings: Settings,
    rules: ModeRules,
    card_id: str,
) -> Judged:
    quality = validate_quality(
        baseline=before,
        candidate=after,
        workload=workload,
        policy=policy,
        settings=settings,
        action_id=card_id,
        model=model,
    )
    comparison = compare(
        baseline=before,
        candidate=after,
        bottleneck_metric="decode_tokens_per_sec",
        bottleneck_direction="increase",
        quality=quality,
        policy=policy,
    )
    assessment = assess_comparison(comparison, rules.goal)
    better = (
        quality_ok(quality)
        and comparison.objective_improved
        and not comparison.objective_regressed
    )
    return Judged(
        comparison=comparison,
        quality=quality,
        gain=assessment.gain,
        guard_breaches=list(assessment.guard_breaches),
        technically_better=better,
    )


def confirm_by_alternating(
    *,
    harness: MeasurementHarness,
    model: ModelProfile,
    workload: BenchmarkWorkload,
    starting: RuntimeConfiguration,
    final: RuntimeConfiguration,
    policy: ExperimentPolicy,
    settings: Settings,
    rules: ModeRules,
    rounds: int,
    watch: DisturbanceWatch,
    card_id: str,
    deadline: float | None = None,
    on_step: Callable[[str], None] | None = None,
) -> ConfirmationResult:
    """Measure starting, final, starting, final ... and judge the pooled results.

    Time already spent searching is never charged: the keep rule sees only what is
    still to come (nothing, before loading), so the payback check decides on the gain.
    """
    starting_runs: list[ExperimentMeasurement] = []
    final_runs: list[ExperimentMeasurement] = []
    order: list[str] = []
    notes: list[str] = []
    repeats_before = watch.repeats
    for index in range(max(rounds, 1)):
        if index > 0 and deadline is not None and time.perf_counter() >= deadline:
            notes.append(f"time limit reached after {index} round(s)")
            break
        for label, config, sink in (
            ("starting", starting, starting_runs),
            ("final", final, final_runs),
        ):
            if on_step is not None:
                on_step(f"round {index + 1}: {label} setup")
            sink.append(
                measure_steady(
                    harness=harness,
                    model=model,
                    workload=workload,
                    config=config,
                    policy=policy,
                    phase=f"tune-confirm-{label}-{index + 1}",
                    watch=watch,
                )
            )
            order.append(label)
    # A round is only complete when both setups were measured in it.
    complete = min(len(starting_runs), len(final_runs))
    starting_runs, final_runs = starting_runs[:complete], final_runs[:complete]
    if complete == 0:
        return ConfirmationResult(order=order, notes=[*notes, "no complete round"])
    before = pool(starting_runs, phase="tune-confirm-starting")
    after = pool(final_runs, phase="tune-confirm-final")
    judged = judge_pair(
        before=before,
        after=after,
        workload=workload,
        model=model,
        policy=policy,
        settings=settings,
        rules=rules,
        card_id=card_id,
    )
    judgement = judge_keep(
        judged.comparison,
        technically_better=judged.technically_better,
        effort=rules.effort,
        goal=rules.goal,
        situation=Situation.BEFORE_LOADING,
        settings=settings,
        remaining_cost_terms={},
    )
    return ConfirmationResult(
        rounds=complete,
        order=order,
        starting_samples=before.valid_runs,
        final_samples=after.valid_runs,
        per_measure=dict(judgement.goal.per_measure),
        quality_passed=quality_ok(judged.quality),
        judgement=judgement,
        disturbed_repeats=watch.repeats - repeats_before,
        notes=notes,
    )


__all__ = [
    "DisturbanceAwareEvaluator",
    "DisturbanceWatch",
    "Judged",
    "confirm_by_alternating",
    "disturbance_reason",
    "judge_pair",
    "measure_steady",
    "other_activity_percent",
    "pool",
    "quality_ok",
]
