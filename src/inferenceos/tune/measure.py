"""Measuring for ``tune``: disturbed measurements are repeated, and the final setup is
confirmed against the starting one by alternating runs (step 5).

A measurement counts as *disturbed* when the whole machine's processor use during it
was clearly more than the model itself explains (the user or another program was
busy). Disturbed measurements are thrown away and repeated a bounded number of times.
"""

from __future__ import annotations

import math
import statistics
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
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
from inferenceos.optimization.goals import assess_comparison, improvement
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
    """Measure ``config``; repeat a disturbed measurement up to ``watch.max_repeats``.

    Every measurement (and every repeat) starts from a fresh load of the model, also
    when the setup is the same as the one already running, so two measurements are
    always made under the same conditions. A measurement that is still disturbed after
    the last allowed repeat is marked as not usable (``valid_runs = 0``).
    """
    began = time.perf_counter()
    try:
        return _measure_steady(harness=harness, model=model, workload=workload, config=config,
                               policy=policy, phase=phase, watch=watch, deadline=deadline, began=began)
    finally:
        # Normal end, failure or cancellation: a measurement-scoped monitor is closed.
        end = getattr(harness, "end_measurement", None)
        if callable(end):
            end()


def _measure_steady(
    *,
    harness: MeasurementHarness,
    model: ModelProfile,
    workload: BenchmarkWorkload,
    config: RuntimeConfiguration,
    policy: ExperimentPolicy,
    phase: str,
    watch: DisturbanceWatch,
    deadline: float | None,
    began: float,
) -> ExperimentMeasurement:
    _fresh_start(harness)
    measurement = measure_configuration(
        profiler=harness,
        model=model,
        workload=workload,
        config=config,
        policy=policy,
        phase=phase,
        deadline=deadline,
    )
    failures = measurement.invalid_runs
    failure_notes = [note for note in measurement.notes if "failed" in note]
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
        _fresh_start(harness)
        measurement = measure_configuration(
            profiler=harness,
            model=model,
            workload=workload,
            config=config,
            policy=policy,
            phase=phase,
            deadline=deadline,
        )
        failures += measurement.invalid_runs
        failure_notes.extend(note for note in measurement.notes if "failed" in note)
        measurement = measurement.model_copy(
            update={"notes": [*measurement.notes, f"repeated after: {reason}"]}
        )
    if failures != measurement.invalid_runs:
        # Review 6: a run that failed in a disturbed attempt is still evidence that the
        # setup failed to run; repeating the attempt must not erase it.
        measurement = measurement.model_copy(
            update={
                "invalid_runs": failures,
                "notes": [
                    *measurement.notes,
                    *[f"earlier attempt: {note}" for note in failure_notes
                      if note not in measurement.notes],
                ],
            }
        )
    reason = disturbance_reason(
        measurement, logical_cores=watch.logical_cores, limit_percent=watch.limit_percent
    )
    if reason is not None:
        # Still disturbed after every allowed repeat (or no time to repeat): not usable.
        measurement = measurement.model_copy(
            update={
                "valid_runs": 0,
                "notes": [*measurement.notes, f"not used: still {reason}"],
            }
        )
    # The whole time this measurement took, fresh start and disturbed repeats included.
    return measurement.model_copy(update={"elapsed_s": round(time.perf_counter() - began, 4)})


def _fresh_start(harness: MeasurementHarness) -> None:
    fresh = getattr(harness, "request_fresh_start", None)
    if callable(fresh):
        fresh()


class DisturbanceAwareEvaluator:
    """Marks a disturbed layer-search measurement as not valid, so it is repeated.

    Wraps a point evaluator; the robust evaluator around it repeats invalid attempts.
    """

    def __init__(
        self,
        inner: PointEvaluator,
        watch: DisturbanceWatch,
        harness: MeasurementHarness | None = None,
    ) -> None:
        self._inner = inner
        self._watch = watch
        self._harness = harness

    def configuration_for(self, layers: int) -> RuntimeConfiguration:
        return self._inner.configuration_for(layers)

    def measure(self, layers: int) -> PointMeasurement:
        # Every screening measurement starts from a fresh load too, also when the same
        # count is measured again (review 2, 4 Oct 2026).
        if self._harness is not None:
            _fresh_start(self._harness)
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


def _signed_log_score(values: Mapping[str, float], weights: Mapping[str, float]) -> float | None:
    """Goal score of one reading in log form (a change of x in it is about x%)."""
    total = 0.0
    for metric, weight in weights.items():
        value = values.get(metric)
        if value is None or value <= 0:
            return None
        total += weight * (1.0 if improvement(metric, 1.0) > 0 else -1.0) * math.log(value)
    return total


def run_score_spread(
    measurement: ExperimentMeasurement, weights: Mapping[str, float]
) -> float | None:
    """Run-to-run spread of the goal score inside one measurement (one load).

    The goal score of each run (every goal measure of that run together), and the
    relative spread of those scores. Measures move together within a run (a slow run is
    often slow on both), so this replaces treating them as independent (review 4); runs
    missing a measure fall back to combining each measure's own spread. ``None`` with
    fewer than 3 runs or a measure that was not read.
    """
    scores = []
    for run in measurement.run_metrics:
        values = {
            metric: float(value)
            for metric in weights
            if (value := getattr(run, metric, None)) is not None
        }
        if len(values) == len(weights):
            score = _signed_log_score(values, weights)
            if score is not None:
                scores.append(score)
    if len(scores) >= 3:
        return float(statistics.stdev(scores))
    total = 0.0
    for metric, weight in weights.items():
        values_list = [
            float(value)
            for run in measurement.run_metrics
            if (value := getattr(run, metric, None)) is not None
        ]
        if len(values_list) < 3:
            return None
        middle = statistics.median(values_list)
        if middle == 0:
            return None
        total += (weight * statistics.stdev(values_list) / abs(middle)) ** 2
    return math.sqrt(total)


def answer_costs(
    measurement: ExperimentMeasurement, wall_s: float | None
) -> tuple[float | None, float | None]:
    """(time per extra answer, time per load) of a measurement that took ``wall_s``.

    The time per answer is the span from the first answer's start to the last one's end
    divided by the answers: it includes everything that recurs per answer (telemetry
    start-up and sampling, gaps), which more answers cannot spread out. The rest of the
    wall time is the one-off cost of a load. Review 4: counting everything outside the
    engine's own answer time as "load" made more runs look cheaper than they are.
    ``(None, None)`` without answer start and end times.
    """
    spans = [
        (run.identity.started_at, run.identity.finished_at)
        for run in measurement.runs
        if run.identity.finished_at is not None
    ]
    if wall_s is None or len(spans) < 2:
        return None, None
    first = min(start for start, _ in spans)
    last = max(end for _, end in spans)
    span = (last - first).total_seconds()
    if span <= 0:
        return None, None
    per_answer = span / len(spans)
    return per_answer, max(wall_s - span, 0.0)


#: A load-to-load spread is estimated only from this many fresh loads of one setup.
MIN_LOADS_FOR_SPREAD = 8
#: The smallest load-to-load spread an estimate may give (laptop measurements 0.5-2.2%).
MIN_LOAD_SPREAD = 0.005


def between_load_spread(
    measurements: Sequence[ExperimentMeasurement], weights: Mapping[str, float]
) -> float | None:
    """Load-to-load spread of the goal score, from measurements of one setup each made
    from a fresh load (e.g. the current setup's side of a fair comparison).

    The spread of their goal scores, minus the part the run-to-run spread explains.
    ``None`` with fewer than ``MIN_LOADS_FOR_SPREAD`` measurements; never below
    ``MIN_LOAD_SPREAD``.
    """
    scores: list[float] = []
    withins: list[float] = []
    runs: list[int] = []
    for item in measurements:
        values = {
            metric: value
            for metric in weights
            if (value := _median_of(item, metric)) is not None
        }
        score = _signed_log_score(values, weights) if len(values) == len(weights) else None
        if score is None:
            continue
        scores.append(score)
        within = run_score_spread(item, weights)
        if within is not None:
            withins.append(within)
        runs.append(max(item.valid_runs, 1))
    if len(scores) < MIN_LOADS_FOR_SPREAD:
        return None
    observed = statistics.stdev(scores)
    within = statistics.median(withins) if withins else 0.0
    count = statistics.median(runs)
    left = observed**2 - (within**2) / count
    # Laptop job 092: from 4 loads the subtraction often went to nothing and a floor of
    # 0.2% came out, which then asked for 11 runs per measurement. Few loads cannot show
    # a spread that small, so the estimate never goes below ``MIN_LOAD_SPREAD``.
    return max(math.sqrt(left) if left > 0 else 0.0, MIN_LOAD_SPREAD)


def _median_of(measurement: ExperimentMeasurement, metric: str) -> float | None:
    stats = measurement.metrics.get(metric)
    if stats is not None and stats.median is not None:
        return float(stats.median)
    return None


def choose_runs(
    *, within: float, between: float, load_s: float, run_s: float, low: int, high: int
) -> int:
    """Runs per measurement that give the most certainty per minute.

    A block's spread is about sqrt(between^2 + within^2 / runs) and it costs about
    2 x (load + runs x run time); the product is smallest at
    runs = (within / between) x sqrt(load / run time). Kept between ``low`` and ``high``.
    """
    if within <= 0 or between <= 0 or load_s <= 0 or run_s <= 0:
        return low
    best = (within / between) * math.sqrt(load_s / run_s)
    return int(min(max(math.ceil(best), low), high))


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
    """Only an affirmative pass counts: OK or acceptable variation. "Unknown" (nothing
    to compare with) is not a pass (review, 3 Oct 2026)."""
    return (
        quality.status in (QualityStatus.OK, QualityStatus.ACCEPTABLE_VARIATION)
        and not quality.quality_regression
    )


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
    "choose_runs",
    "confirm_by_alternating",
    "run_score_spread",
    "disturbance_reason",
    "judge_pair",
    "measure_steady",
    "other_activity_percent",
    "pool",
    "quality_ok",
]
