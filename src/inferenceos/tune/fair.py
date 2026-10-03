"""Fair comparison for ``tune``: measure the current setup and a candidate in turns.

Why this exists (found on the laptop, Oct 2026):

* The starting setup used to be measured once, at the very beginning. When the machine
  drifted (a fresh reading of the start mid-search was 11% slower on the 9B), every
  later candidate was compared with a stale, too-high number and real gains vanished.
* The general comparison counts any change smaller than 10% as "no change" (its noise
  margin is ``max(min_effect, noise_tolerance)``), so gains of 2-9% could never be
  kept, whatever the effort mode's smallest worthwhile gain was.

Here both setups are measured in turns, in the order A B, B A, A B, ... (so neither
always goes first), each a fresh load. For every turn the candidate is compared with
the start measured right next to it. Neighbouring turns are averaged so a steady drift
of the machine cancels out; a measure counts as changed only when every such average
agrees on the direction and the overall change is at least the noise floor.
The gain is then scored with the goal's own weighting and judged with the normal keep
rule (effort mode's smallest gain, payback). Answer quality is checked on all runs.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from inferenceos.config import Settings
from inferenceos.contracts.enums import Situation
from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentPolicy,
    RuntimeConfiguration,
)
from inferenceos.contracts.goal import GoalAssessment, KeepJudgement
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.tune import ConfirmationResult
from inferenceos.experiment.measurement import MeasurementHarness
from inferenceos.experiment.quality import validate_quality
from inferenceos.optimization.goals import assess_known_gains, improvement, plain_name
from inferenceos.optimization.keep import ModeRules, judge_assessment
from inferenceos.profiling.workloads import BenchmarkWorkload
from inferenceos.tune.measure import DisturbanceWatch, measure_steady, pool, quality_ok

#: Average processor speed during a measurement (% of rated speed; falls when the
#: laptop throttles from heat). Recorded per turn, never scored.
CLOCK_METRIC = "cpu_performance_percent"

#: Measures read from each measurement (the goal's first measures and the guarded ones
#: come from this list).
FAIR_METRICS: tuple[str, ...] = (
    "decode_tokens_per_sec",
    "ttft_s",
    "prompt_tokens_per_sec",
    "total_latency_s",
    "peak_rss_bytes",
    "hard_page_faults_per_sec",
)


def measurement_value(measurement: ExperimentMeasurement, metric: str) -> float | None:
    """The middle value of one measure over a measurement's runs.

    The median of the per-run values is used (one odd run does not move it); when no
    per-run values exist the measurement's own summary is used.
    """
    values: list[float] = []
    for run in measurement.run_metrics:
        value = getattr(run, metric, None)
        if value is not None:
            values.append(float(value))
    if values:
        return float(statistics.median(values))
    stats = measurement.metrics.get(metric)
    if stats is not None and stats.mean is not None:
        return float(stats.mean)
    return None


@dataclass(frozen=True)
class TurnReading:
    """One turn: the start and the candidate measured next to each other."""

    turn: int
    order: str
    start: dict[str, float | None]
    candidate: dict[str, float | None]
    improvements: dict[str, float]


@dataclass
class FairResult:
    turns: list[TurnReading] = field(default_factory=list)
    per_measure: dict[str, float] = field(default_factory=dict)
    raw_median: dict[str, float] = field(default_factory=dict)
    agreeing: dict[str, bool] = field(default_factory=dict)
    guard_breaches: list[str] = field(default_factory=list)
    quality_passed: bool = False
    assessment: GoalAssessment | None = None
    judgement: KeepJudgement | None = None
    notes: list[str] = field(default_factory=list)
    uneven_turns: list[int] = field(default_factory=list)
    #: What heat did to this comparison, in plain words ("" when nothing happened).
    heat_effect: str = ""
    #: True when leaving out the heat-hit turns changed the keep / don't-keep answer.
    heat_changed_decision: bool = False
    #: The verdict the comparison would have had with every turn counted.
    verdict_with_all_turns: str | None = None
    start_runs: list[ExperimentMeasurement] = field(default_factory=list)
    candidate_runs: list[ExperimentMeasurement] = field(default_factory=list)

    @property
    def kept(self) -> bool:
        return bool(self.judgement is not None and self.judgement.keep)

    @property
    def gain(self) -> float | None:
        return self.assessment.gain if self.assessment is not None else None

    def record(self) -> dict[str, object]:
        """Every reading, for the tune report."""
        return {
            "turns": [
                {
                    "turn": item.turn,
                    "order": item.order,
                    "start": item.start,
                    "candidate": item.candidate,
                    "improvements": item.improvements,
                }
                for item in self.turns
            ],
            "per_measure": self.per_measure,
            "raw_median": self.raw_median,
            "agreeing": self.agreeing,
            "guard_breaches": self.guard_breaches,
            "uneven_turns": self.uneven_turns,
            "heat_effect": self.heat_effect,
            "heat_changed_decision": self.heat_changed_decision,
            "verdict_with_all_turns": self.verdict_with_all_turns,
            "quality_passed": self.quality_passed,
            "gain": self.gain,
            "kept": self.kept,
            "verdict": self.judgement.verdict.value if self.judgement else None,
            "reasons": list(self.judgement.reasons) if self.judgement else [],
            "notes": self.notes,
        }

    def as_confirmation(self) -> ConfirmationResult:
        return ConfirmationResult(
            rounds=len(self.turns),
            order=[part for item in self.turns for part in item.order.split(",")],
            starting_samples=sum(item.valid_runs for item in self.start_runs),
            final_samples=sum(item.valid_runs for item in self.candidate_runs),
            per_measure=dict(self.per_measure),
            quality_passed=self.quality_passed,
            judgement=self.judgement,
            notes=list(self.notes),
        )


def split_by_clock(
    turns: Sequence[TurnReading], *, tolerance: float
) -> tuple[list[TurnReading], list[TurnReading]]:
    """Turns hit by a sudden heat or power event, and the rest.

    Measured on the laptop (Oct 2026): a setting can change the processor speed by
    itself (prompt batch 128 ran 37 points lower), and the speed drifts steadily as the
    laptop warms (117% -> 103% over a 9B run), which the alternating order cancels.
    Neither is a reason to leave a turn out.

    A turn is left out only when **both** are true:

    1. one of its two measurements ran more than ``tolerance`` points below the best
       speed the *same* setup reached in this comparison (that setup slowed down), and
    2. the speed gap between the two setups in that turn is far from the usual gap
       (``tolerance`` points, widened to three times the typical spread of the gaps
       when the readings are naturally noisy), so the slowdown hit one setup and not
       the other.

    Guarantee: if neither setup ever runs more than ``tolerance`` points below its own
    best speed (no throttling), nothing is ever left out and the comparison is exactly
    the one without the heat check. Turns without a reading are never left out.
    """
    start_speeds = [
        float(v) for v in (item.start.get(CLOCK_METRIC) for item in turns) if v is not None
    ]
    new_speeds = [
        float(v) for v in (item.candidate.get(CLOCK_METRIC) for item in turns) if v is not None
    ]
    if not start_speeds or not new_speeds:
        return list(turns), []
    start_best, new_best = max(start_speeds), max(new_speeds)
    gaps: dict[int, float] = {}
    for item in turns:
        start = item.start.get(CLOCK_METRIC)
        new = item.candidate.get(CLOCK_METRIC)
        if start is not None and new is not None:
            gaps[item.turn] = float(new) - float(start)
    if len(gaps) < 2:
        return list(turns), []
    middle = float(statistics.median(gaps.values()))
    spread = float(statistics.median(abs(value - middle) for value in gaps.values()))
    limit = max(float(tolerance), 3.0 * spread)
    even: list[TurnReading] = []
    uneven: list[TurnReading] = []
    for item in turns:
        start = item.start.get(CLOCK_METRIC)
        new = item.candidate.get(CLOCK_METRIC)
        gap = gaps.get(item.turn)
        slowed = (start is not None and float(start) < start_best - tolerance) or (
            new is not None and float(new) < new_best - tolerance
        )
        if gap is not None and slowed and abs(gap - middle) > limit:
            uneven.append(item)
        else:
            even.append(item)
    return even, uneven


def throttled_any(turns: Sequence[TurnReading], *, tolerance: float) -> bool:
    """Whether either setup ever ran more than ``tolerance`` points below its own best."""
    for side in ("start", "candidate"):
        speeds = [
            float(v)
            for v in (getattr(item, side).get(CLOCK_METRIC) for item in turns)
            if v is not None
        ]
        if speeds and min(speeds) < max(speeds) - tolerance:
            return True
    return False


def drift_balanced(values: Sequence[float]) -> tuple[float, bool]:
    """Combine per-turn changes so a steady drift of the machine cancels out.

    The turns alternate which setup goes first, so a machine that slowly speeds up or
    slows down biases odd and even turns in opposite directions. Averaging each turn
    with the next one cancels that; the estimate is the mean of those neighbour
    averages (weights 1, 2, ..., 2, 1), and the change counts as one direction only
    when every neighbour average agrees.
    """
    items = [float(value) for value in values]
    if not items:
        return 0.0, False
    if len(items) == 1:
        return items[0], items[0] != 0.0
    pairs = [(items[index] + items[index + 1]) / 2.0 for index in range(len(items) - 1)]
    estimate = float(sum(pairs) / len(pairs))
    agree = all(value > 0 for value in pairs) or all(value < 0 for value in pairs)
    return estimate, agree


def score_turns(
    turns: Sequence[TurnReading],
    *,
    rules: ModeRules,
    noise_floor: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, bool], list[str], GoalAssessment]:
    """Turn the per-turn improvements into per-measure gains and guard breaches."""
    goal = rules.goal
    raw_median: dict[str, float] = {}
    agreeing: dict[str, bool] = {}
    counted: dict[str, float] = {}
    for metric in FAIR_METRICS:
        values = [item.improvements[metric] for item in turns if metric in item.improvements]
        if not values:
            continue
        middle, same_direction = drift_balanced(values)
        raw_median[metric] = middle
        agreeing[metric] = same_direction
        counted[metric] = middle if same_direction and abs(middle) >= noise_floor else 0.0
    per_measure = {metric: value for metric, value in counted.items() if metric in goal.first}
    assessment = assess_known_gains(per_measure, goal)
    breaches: list[str] = []
    rationale = list(assessment.rationale)
    for metric in goal.guarded:
        value = counted.get(metric)
        if value is None or value >= 0.0:
            continue
        if -value > goal.guard_limit:
            breaches.append(metric)
            rationale.append(
                f"guarded {plain_name(metric)} got {-value:.1%} worse "
                f"(limit {goal.guard_limit:.0%})"
            )
    assessment = assessment.model_copy(
        update={"guard_breaches": breaches, "rationale": rationale}
    )
    return per_measure, raw_median, agreeing, breaches, assessment


def compare_fairly(
    *,
    harness: MeasurementHarness,
    model: ModelProfile,
    workload: BenchmarkWorkload,
    start: RuntimeConfiguration,
    candidate: RuntimeConfiguration,
    policy: ExperimentPolicy,
    settings: Settings,
    rules: ModeRules,
    watch: DisturbanceWatch,
    card_id: str,
    turns: int,
    phase: str,
    deadline: float | None = None,
    on_step: Callable[[str], None] | None = None,
) -> FairResult:
    """Measure ``start`` and ``candidate`` in turns and judge the candidate fairly."""
    result = FairResult()
    noise_floor = float(settings.tune_fair_noise_floor)
    for index in range(max(int(turns), 1)):
        if index > 0 and deadline is not None and time.perf_counter() >= deadline:
            result.notes.append(f"time limit reached after {index} turn(s)")
            break
        first_is_start = index % 2 == 0
        order = ("start", "candidate") if first_is_start else ("candidate", "start")
        measured: dict[str, ExperimentMeasurement] = {}
        for label in order:
            config = start if label == "start" else candidate
            if on_step is not None:
                on_step(f"turn {index + 1}: {'current' if label == 'start' else 'new'} setup")
            measured[label] = measure_steady(
                harness=harness,
                model=model,
                workload=workload,
                config=config,
                policy=policy,
                phase=f"{phase}-{label}-{index + 1}",
                watch=watch,
            )
        start_m, cand_m = measured["start"], measured["candidate"]
        if (
            start_m.valid_runs < policy.min_valid_samples
            or cand_m.valid_runs < policy.min_valid_samples
        ):
            result.notes.append(
                f"turn {index + 1} could not be measured "
                f"({start_m.valid_runs}/{cand_m.valid_runs} good runs)"
            )
            continue
        result.start_runs.append(start_m)
        result.candidate_runs.append(cand_m)
        start_values = {
            metric: measurement_value(start_m, metric) for metric in (*FAIR_METRICS, CLOCK_METRIC)
        }
        cand_values = {
            metric: measurement_value(cand_m, metric) for metric in (*FAIR_METRICS, CLOCK_METRIC)
        }
        improvements: dict[str, float] = {}
        for metric in FAIR_METRICS:
            base, new = start_values[metric], cand_values[metric]
            if base is None or new is None or base == 0.0:
                continue
            improvements[metric] = improvement(metric, (new - base) / abs(base))
        result.turns.append(
            TurnReading(
                turn=index + 1,
                order=",".join(order),
                start=start_values,
                candidate=cand_values,
                improvements=improvements,
            )
        )

    if not result.turns:
        result.notes.append("no turn could be measured")
        result.judgement = judge_assessment(
            GoalAssessment(rationale=["nothing measured"]),
            technically_better=False,
            effort=rules.effort,
            goal=rules.goal,
            situation=Situation.BEFORE_LOADING,
            settings=settings,
            remaining_cost_terms={},
        )
        return result

    tolerance = float(settings.tune_fair_clock_tolerance)
    even, uneven = split_by_clock(result.turns, tolerance=tolerance)
    result.uneven_turns = [item.turn for item in uneven]
    for item in uneven:
        result.notes.append(
            f"turn {item.turn} left out: the processor ran at "
            f"{item.start.get(CLOCK_METRIC):.0f}% (current setup) and "
            f"{item.candidate.get(CLOCK_METRIC):.0f}% (new setup); that gap is far from "
            "the gap in the other turns (a sudden heat or power change)"
        )
    scored = even if even else result.turns
    if not even:
        result.notes.append(
            "every turn ran at uneven processor speeds, so the comparison is not fair; "
            "nothing is kept from it"
        )
    per_measure, raw_median, agreeing, breaches, assessment = score_turns(
        scored, rules=rules, noise_floor=noise_floor
    )
    result.per_measure = per_measure
    result.raw_median = raw_median
    result.agreeing = agreeing
    result.guard_breaches = breaches
    result.assessment = assessment

    quality = validate_quality(
        baseline=pool(result.start_runs, phase=f"{phase}-start"),
        candidate=pool(result.candidate_runs, phase=f"{phase}-candidate"),
        workload=workload,
        policy=policy,
        settings=settings,
        action_id=card_id,
        model=model,
    )
    result.quality_passed = quality_ok(quality)
    if not result.quality_passed:
        result.notes.append(f"answer quality check: {quality.status.value}")
    gain = assessment.gain
    technically_better = (
        result.quality_passed and bool(even) and gain is not None and gain > 0.0
    )
    result.judgement = judge_assessment(
        assessment,
        technically_better=technically_better,
        effort=rules.effort,
        goal=rules.goal,
        situation=Situation.BEFORE_LOADING,
        settings=settings,
        remaining_cost_terms={},
    )
    if uneven:
        # Judge again with every turn counted, so the user can be told whether heat
        # changed the answer.
        _, _, _, _, all_assessment = score_turns(
            result.turns, rules=rules, noise_floor=noise_floor
        )
        all_gain = all_assessment.gain
        all_judgement = judge_assessment(
            all_assessment,
            technically_better=(
                result.quality_passed and all_gain is not None and all_gain > 0.0
            ),
            effort=rules.effort,
            goal=rules.goal,
            situation=Situation.BEFORE_LOADING,
            settings=settings,
            remaining_cost_terms={},
        )
        result.verdict_with_all_turns = all_judgement.verdict.value
        result.heat_changed_decision = all_judgement.keep != result.judgement.keep
        turns_text = ", ".join(str(turn) for turn in result.uneven_turns)
        all_text = f"{all_gain:+.1%}" if all_gain is not None else "unknown"
        if not even:
            result.heat_effect = (
                "the laptop slowed down from heat in every turn, so this could not be "
                f"judged fairly and was not kept (counting every turn it measured "
                f"{all_text})"
            )
        elif result.heat_changed_decision and not result.judgement.keep:
            result.heat_effect = (
                f"REJECTED BECAUSE OF HEAT: with every turn counted it looked {all_text} "
                f"and would have been kept, but turn(s) {turns_text} ran while the laptop "
                "was throttling; without them it was not clearly better"
            )
        elif result.heat_changed_decision:
            result.heat_effect = (
                f"kept only after leaving out heat-hit turn(s) {turns_text}: counting them "
                f"it looked {all_text} and would not have been kept"
            )
        else:
            result.heat_effect = (
                f"turn(s) {turns_text} ran while the laptop was throttling and were left "
                "out; the decision is the same either way"
            )
        result.notes.append(result.heat_effect)
    return result


def fair_words(result: FairResult) -> str:
    """One plain line about a fair comparison."""
    if not result.turns:
        return "could not be measured: " + "; ".join(result.notes or ["no turn"])
    parts = []
    for metric, value in result.raw_median.items():
        if metric in ("decode_tokens_per_sec", "ttft_s"):
            mark = "" if result.agreeing.get(metric) else " (turns disagree)"
            parts.append(f"{plain_name(metric)} {value:+.1%}{mark}")
    gain = result.gain
    verdict = result.judgement.verdict.value.replace("_", " ") if result.judgement else "?"
    gain_text = f"{gain:+.1%}" if gain is not None else "unknown"
    clocks = [
        (item.start.get(CLOCK_METRIC), item.candidate.get(CLOCK_METRIC)) for item in result.turns
    ]
    known = [(a, b) for a, b in clocks if a is not None and b is not None]
    if known:
        starts = [float(a) for a, _ in known]
        news = [float(b) for _, b in known]
        parts.append(
            f"processor speed {min(starts):.0f}-{max(starts):.0f}% (current) and "
            f"{min(news):.0f}-{max(news):.0f}% (new)"
        )
    if result.uneven_turns:
        parts.append(f"turns left out for heat: {result.uneven_turns}")
    return (
        f"{len(result.turns)} turn(s): " + ", ".join(parts)
        + f"; goal gain {gain_text} -> {verdict}"
    )


__all__ = [
    "CLOCK_METRIC",
    "FAIR_METRICS",
    "FairResult",
    "TurnReading",
    "compare_fairly",
    "drift_balanced",
    "fair_words",
    "measurement_value",
    "score_turns",
    "split_by_clock",
    "throttled_any",
]
