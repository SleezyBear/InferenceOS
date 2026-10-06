"""Fair comparison for ``tune``: the current setup and a candidate in independent blocks.

Each *block* measures both setups, each from a fresh load of the model, and the order
alternates between blocks (A B, B A, ...) so a steady drift of the machine cancels out.
One block gives one score per measure (how much better the new setup was in that block)
and one goal score (the goal's measures with the goal's fixed weights). A block counts
only when both measurements are good and every one of the goal's measures was read; a
score is never re-weighted over whatever happened to be measured (review, 4 Oct 2026).

The block scores are the evidence (``tune/stats.py``):

* mode ``better``: the change counts as better only when the lower confidence bound of
  the mean goal score is above 0, so a change that does not really help is kept at most
  ``tune_fair_alpha`` (5%) of the time (user decision, 3 Oct 2026). It also has to pass
  the normal keep rule (effort mode's smallest gain, payback) and the quality check.
* mode ``not_worse``: passes only when any loss is shown to be smaller than
  ``tune_guard_margin`` (2%, user decision 4 Oct 2026, the same in every effort mode):
  the lower bound is above ``-margin``. "Unsure" fails.

The looks are planned before the first block (``planned_looks``, user decision 4 Oct
2026): Minimize cost one look at 4 usable blocks; Balanced and Custom looks at 4 and 8;
Performance first at 4 and a preset maximum of up to 16. ``alpha`` is split between the
looks. A decision is made **only** at a planned look, on exactly the planned number of
usable blocks. Running out of time or attempts before a look decides gives "unsure"
(never an extra acceptance check).

Every block counts. Processor-speed readings are recorded and reported, but no block is
left out because of them (user decision 4 Oct 2026: a speed difference is not proof of
overheating, and heat a setting causes itself counts against that setting). A
measurement disturbed by another program is still repeated (``tune/measure.py``).
Answer quality is checked on all runs.
"""

from __future__ import annotations

import statistics
import time
from typing import Literal
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from inferenceos.config import Settings
from inferenceos.contracts.enums import OptimizationMode, Situation
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
from inferenceos.tune.stats import BlockTest, block_bounds
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

#: Count-like guarded measures are judged in absolute units, not as ratios (a healthy
#: run pages 0-20 times a second, so ratios turn noise into "+1500%"; laptop, Oct 2026).
#: For each: (typical tolerance per second, severe-episode level per second).
#:
#: * typical use: the middle (median) of the per-block paired differences (new minus
#:   current) must not exceed the tolerance, i.e. most blocks page about as much;
#: * severe episodes: a measurement above the episode level is reported with the speeds
#:   of its block; episodes count against the new setup when it has at least
#:   ``EPISODE_EXCESS`` more of them than the current setup (a repeated pattern, not
#:   one spike). Seen on the laptop: one Gemma measurement at 9,990 faults/s wrote 15%
#:   slower than its partner; one 9B measurement at 791/s showed no clear slowdown.
COUNT_GUARDS: dict[str, tuple[float, float]] = {"hard_page_faults_per_sec": (100.0, 1000.0)}
EPISODE_EXCESS = 2


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
    #: Processor-speed change during this comparison, in plain words ("" when the speed
    #: stayed within ``tune_fair_clock_tolerance`` points). Reported only: no block is
    #: ever left out because of it.
    heat_effect: str = ""
    #: Guarded measures that could not be read (reported as unavailable, not a pass).
    unavailable_guards: list[str] = field(default_factory=list)
    #: Severe episodes of count-like guarded measures, in plain words.
    episodes: list[str] = field(default_factory=list)
    #: Blocks attempted (usable or not).
    attempts: int = 0
    #: Why the comparison stopped before a planned look decided ("" when one decided).
    stopped_early: str = ""
    start_runs: list[ExperimentMeasurement] = field(default_factory=list)
    candidate_runs: list[ExperimentMeasurement] = field(default_factory=list)
    #: "better" or "not_worse": what this comparison had to show.
    mode: str = "better"
    #: The answer: "better", "not better", "not worse", "worse" or "unsure".
    decision: str = "unsure"
    #: Goal score of each counted block, in block order.
    block_scores: list[float] = field(default_factory=list)
    #: The statistics of the counted blocks at the last look.
    test: BlockTest | None = None
    #: Blocks counted at each look.
    looks: list[int] = field(default_factory=list)

    @property
    def not_worse(self) -> bool:
        """Shown not worse (beyond the margin), with good answers and no guard breach."""
        return bool(
            self.decision in ("not worse", "better")
            and self.quality_passed
            and not self.guard_breaches
        )

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
            "mode": self.mode,
            "decision": self.decision,
            "block_scores": self.block_scores,
            "looks": self.looks,
            "mean": self.test.mean if self.test else None,
            "lower_bound": self.test.lower if self.test else None,
            "upper_bound": self.test.upper if self.test else None,
            "alpha_per_look": self.test.alpha if self.test else None,
            "per_measure": self.per_measure,
            "means": self.raw_median,
            "guard_breaches": self.guard_breaches,
            "heat_effect": self.heat_effect,
            "unavailable_guards": self.unavailable_guards,
            "episodes": self.episodes,
            "attempts": self.attempts,
            "stopped_early": self.stopped_early,
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


def block_score(improvements: dict[str, float], rules: ModeRules) -> float | None:
    """One block's goal score: the goal's measures with the goal's fixed weights.

    ``None`` when any of the goal's measures is missing (the block is not usable).
    """
    if any(metric not in improvements for metric in rules.goal.first):
        return None
    known = {metric: improvements[metric] for metric in rules.goal.first}
    return assess_known_gains(known, rules.goal, require_all=True).gain


def _pairs(blocks: Sequence[TurnReading], metric: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for item in blocks:
        start, candidate = item.start.get(metric), item.candidate.get(metric)
        if start is not None and candidate is not None:
            pairs.append((float(start), float(candidate)))
    return pairs


def guard_change(blocks: Sequence[TurnReading], metric: str) -> float | None:
    """How much a guarded measure improved (negative = worse), over the counted blocks.

    Proportional measures (memory, total time): the mean of the per-block paired
    relative changes. Count-like measures (``COUNT_GUARDS``, e.g. paging): the median of
    the per-block paired differences, as a fraction of the typical tolerance (so -1.0
    means "most blocks page one tolerance more"). Severe episodes are judged separately
    (``guard_episodes``). ``None`` when the measure was not read.
    """
    pairs = _pairs(blocks, metric)
    if not pairs:
        return None
    if metric in COUNT_GUARDS:
        tolerance, _ = COUNT_GUARDS[metric]
        middle = float(statistics.median(new - start for start, new in pairs))
        return improvement(metric, middle / tolerance)
    ratios = [improvement(metric, (new - start) / abs(start)) for start, new in pairs if start]
    if not ratios:
        return None
    return float(statistics.fmean(ratios))


def guard_episodes(
    blocks: Sequence[TurnReading], metric: str
) -> tuple[bool, list[str]]:
    """Severe episodes of a count-like measure: (counts against the new setup, words)."""
    if metric not in COUNT_GUARDS:
        return False, []
    _, level = COUNT_GUARDS[metric]
    words: list[str] = []
    counts = {"start": 0, "candidate": 0}
    for item in blocks:
        for side, label in (("start", "current"), ("candidate", "new")):
            value = getattr(item, side).get(metric)
            if value is None or float(value) < level:
                continue
            counts[side] += 1
            other = item.candidate if side == "start" else item.start
            mine = getattr(item, side)
            speed = []
            for name in ("decode_tokens_per_sec", "ttft_s"):
                a, b = mine.get(name), other.get(name)
                if a is not None and b is not None and b:
                    change = improvement(name, (float(a) - float(b)) / abs(float(b)))
                    speed.append(f"{plain_name(name)} {change:+.0%}")
            words.append(
                f"block {item.turn}: {plain_name(metric)} {float(value):,.0f}/s in the {label} "
                f"setup" + (f" (vs its partner, + is better: {', '.join(speed)})" if speed else "")
            )
    against = counts["candidate"] - counts["start"] >= EPISODE_EXCESS
    return against, words


def _summarise(
    blocks: Sequence[TurnReading], rules: ModeRules
) -> tuple[dict[str, float], dict[str, float], list[str], GoalAssessment, list[float]]:
    """Per-measure mean improvements, guard breaches and the goal assessment."""
    goal = rules.goal
    means: dict[str, float] = {}
    for metric in FAIR_METRICS:
        values = [item.improvements[metric] for item in blocks if metric in item.improvements]
        if values:
            means[metric] = float(statistics.fmean(values))
    per_measure = {metric: value for metric, value in means.items() if metric in goal.first}
    scores = [
        score
        for score in (block_score(item.improvements, rules) for item in blocks)
        if score is not None
    ]
    assessment = assess_known_gains(per_measure, goal, require_all=True)
    if scores:
        assessment = assessment.model_copy(update={"gain": float(statistics.fmean(scores))})
    breaches: list[str] = []
    rationale = list(assessment.rationale)
    for metric in goal.guarded:
        value = guard_change(blocks, metric)
        # Count-like measures are in units of their tolerance (limit 1.0); the others
        # are relative changes (limit: the goal's guard limit, 10%).
        limit = 1.0 if metric in COUNT_GUARDS else goal.guard_limit
        if value is not None and value < 0.0 and -value > limit:
            breaches.append(metric)
            if metric in COUNT_GUARDS:
                tolerance, _ = COUNT_GUARDS[metric]
                rationale.append(
                    f"guarded {plain_name(metric)}: most blocks went up by about "
                    f"{-value * tolerance:,.0f}/s (limit {tolerance:,.0f}/s)"
                )
            else:
                rationale.append(
                    f"guarded {plain_name(metric)} got {-value:.1%} worse "
                    f"(limit {goal.guard_limit:.0%})"
                )
        against, _ = guard_episodes(blocks, metric)
        if against and metric not in breaches:
            breaches.append(metric)
            rationale.append(
                f"guarded {plain_name(metric)}: severe episodes in at least "
                f"{EPISODE_EXCESS} more blocks with the new setup"
            )
    assessment = assessment.model_copy(
        update={"guard_breaches": breaches, "rationale": rationale}
    )
    return per_measure, means, breaches, assessment, scores


def decide(
    test: BlockTest,
    *,
    mode: str,
    margin: float,
    smallest_gain: float,
    final_look: bool,
) -> str | None:
    """The answer from the block statistics, or ``None`` to keep measuring.

    ``better``: lower bound above 0 -> "better"; upper bound below 0 -> "worse"; upper
    bound below the smallest worthwhile gain -> "not better" (it cannot be worth it).
    ``not_worse``: lower bound above ``-margin`` -> "not worse"; upper bound below
    ``-margin`` -> "worse". Otherwise "unsure" at the last look.
    """
    if not test.known:
        return "unsure" if final_look else None
    assert test.lower is not None and test.upper is not None
    if mode == "not_worse":
        if test.lower > -margin:
            return "not worse"
        if test.upper < -margin:
            return "worse"
        return "unsure" if final_look else None
    if test.lower > 0.0:
        return "better"
    if test.upper < 0.0:
        return "worse"
    if test.upper < smallest_gain:
        return "not better"
    return "unsure" if final_look else None


def planned_looks(
    settings: Settings,
    rules: ModeRules,
    *,
    min_blocks: int | None = None,
    max_blocks: int | None = None,
) -> list[int]:
    """The usable-block counts at which the comparison is judged, fixed in advance.

    User decision (4 Oct 2026): Minimize cost one look at ``tune_fair_min_blocks`` (4);
    Balanced and Custom looks at 4 and ``tune_fair_max_blocks`` (8); Performance first
    at 4 and ``tune_fair_max_blocks_performance`` (a preset maximum of up to 16, chosen
    from the measured noise). ``min_blocks`` / ``max_blocks`` override (re-check scripts).
    """
    first = max(int(min_blocks or settings.tune_fair_min_blocks), 2)
    if min_blocks is not None or max_blocks is not None:
        last = int(max_blocks or first)
    elif rules.effort.mode is OptimizationMode.MINIMIZE_OPTIMIZATION_COST:
        last = first
    elif rules.effort.mode is OptimizationMode.PERFORMANCE_FIRST:
        last = int(settings.tune_fair_max_blocks_performance)
    else:
        last = int(settings.tune_fair_max_blocks)
    last = max(min(last, MAX_BLOCKS), first)
    return [first] if last == first else [first, last]


#: The largest planned look (user decision, 4 Oct 2026).
MAX_BLOCKS = 16


def _clock_words(turns: Sequence[TurnReading], tolerance: float) -> str:
    """The processor-speed change during a comparison, reported only."""
    readings = [
        float(v)
        for item in turns
        for v in (item.start.get(CLOCK_METRIC), item.candidate.get(CLOCK_METRIC))
        if v is not None
    ]
    if len(readings) < 2 or max(readings) - min(readings) <= tolerance:
        return ""
    first, last = readings[0], readings[-1]
    return (
        f"processor speed ranged {min(readings):.0f}-{max(readings):.0f}% during this "
        f"comparison (first reading {first:.0f}%, last {last:.0f}%); every block was "
        "counted. A speed change alone does not show overheating; heat a setting causes "
        "itself counts against it"
    )


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
    phase: str,
    mode: Literal["better", "not_worse"] = "better",
    min_blocks: int | None = None,
    max_blocks: int | None = None,
    deadline: float | None = None,
    on_step: Callable[[str], None] | None = None,
    alpha: float | None = None,
) -> FairResult:
    """Measure ``start`` and ``candidate`` in independent blocks and judge fairly.

    Judged only at the planned looks, on exactly the planned number of usable blocks.
    Unusable blocks (a measurement without enough good runs, or a goal measure that was
    not read) are attempted again up to ``tune_fair_extra_attempts`` times in total.
    Running out of time or attempts before a look decides gives "unsure".
    """
    result = FairResult(mode=mode)
    # ``alpha`` overrides the false-keep rate (a lighter per-path check whose results
    # only select what the fresh final check then tests).
    alpha = float(settings.tune_fair_alpha) if alpha is None else float(alpha)
    margin = float(settings.tune_guard_margin)
    looks = planned_looks(settings, rules, min_blocks=min_blocks, max_blocks=max_blocks)
    alpha_look = alpha / len(looks)
    smallest = float(rules.effort.minimum_improvement)
    tolerance = float(settings.tune_fair_clock_tolerance)
    max_attempts = looks[-1] + max(int(settings.tune_fair_extra_attempts), 0)

    scores: list[float] = []
    decided: str | None = None
    look_at = 0
    while result.attempts < max_attempts and decided is None:
        if deadline is not None and time.perf_counter() >= deadline:
            result.stopped_early = (
                f"time limit reached after {len(scores)} usable block(s); "
                f"the planned look at {looks[look_at]} was not reached"
            )
            break
        index = result.attempts
        first_is_start = index % 2 == 0
        order = ("start", "candidate") if first_is_start else ("candidate", "start")
        measured: dict[str, ExperimentMeasurement] = {}
        for label in order:
            config = start if label == "start" else candidate
            if on_step is not None:
                on_step(f"block {index + 1}: {'current' if label == 'start' else 'new'} setup")
            measured[label] = measure_steady(
                harness=harness,
                model=model,
                workload=workload,
                config=config,
                policy=policy,
                phase=f"{phase}-{label}-{index + 1}",
                watch=watch,
                deadline=deadline,
            )
        result.attempts += 1
        number = result.attempts
        start_m, cand_m = measured["start"], measured["candidate"]
        if (
            start_m.valid_runs < policy.min_valid_samples
            or cand_m.valid_runs < policy.min_valid_samples
        ):
            result.notes.append(
                f"block {number} could not be used "
                f"({start_m.valid_runs}/{cand_m.valid_runs} good runs"
                + "".join(f"; {note}" for note in (start_m.notes + cand_m.notes)[-2:])
                + ")"
            )
            continue
        start_values = {
            metric: measurement_value(start_m, metric)
            for metric in (*FAIR_METRICS, CLOCK_METRIC)
        }
        cand_values = {
            metric: measurement_value(cand_m, metric)
            for metric in (*FAIR_METRICS, CLOCK_METRIC)
        }
        improvements: dict[str, float] = {}
        for metric in FAIR_METRICS:
            base, new = start_values[metric], cand_values[metric]
            if base is None or new is None or base == 0.0:
                continue
            improvements[metric] = improvement(metric, (new - base) / abs(base))
        score = block_score(improvements, rules)
        if score is None:
            missing = [plain_name(m) for m in rules.goal.first if m not in improvements]
            result.notes.append(
                f"block {number} could not be used (not read: {', '.join(missing)})"
            )
            continue
        result.start_runs.append(start_m)
        result.candidate_runs.append(cand_m)
        result.turns.append(
            TurnReading(
                turn=number,
                order=",".join(order),
                start=start_values,
                candidate=cand_values,
                improvements=improvements,
            )
        )
        scores.append(score)
        if len(scores) == looks[look_at]:
            final = look_at == len(looks) - 1
            test = block_bounds(scores, alpha=alpha_look)
            result.looks.append(len(scores))
            result.test = test
            decided = decide(
                test, mode=mode, margin=margin, smallest_gain=smallest, final_look=final
            )
            look_at += 1
    if decided is None and not result.stopped_early:
        result.stopped_early = (
            f"only {len(scores)} usable block(s) in {result.attempts} attempt(s); the "
            f"planned look at {looks[min(look_at, len(looks) - 1)]} was not reached"
        )
    if result.stopped_early:
        result.notes.append(result.stopped_early + " -> unsure")
    result.decision = decided or "unsure"

    if not result.turns:
        result.notes.append("no block could be measured")
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

    per_measure, means, breaches, assessment, block_scores = _summarise(result.turns, rules)
    result.per_measure = per_measure
    result.raw_median = means
    result.guard_breaches = breaches
    result.assessment = assessment
    result.block_scores = block_scores
    for metric in rules.goal.guarded:
        if not _pairs(result.turns, metric):
            result.unavailable_guards.append(metric)
        _, words = guard_episodes(result.turns, metric)
        result.episodes.extend(words)
    if result.unavailable_guards:
        result.notes.append(
            "not measured (unavailable, not counted as a pass): "
            + ", ".join(plain_name(m) for m in result.unavailable_guards)
        )
    result.notes.extend(f"severe episode, {words}" for words in result.episodes)

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
    technically_better = (
        mode == "better" and result.quality_passed and result.decision == "better"
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
    result.heat_effect = _clock_words(result.turns, tolerance)
    if result.heat_effect:
        result.notes.append(result.heat_effect)
    return result


def fair_words(result: FairResult) -> str:
    """One plain line about a fair comparison."""
    if not result.turns:
        return "could not be measured: " + "; ".join(result.notes or ["no block"])
    parts = []
    for metric, value in result.raw_median.items():
        if metric in ("decode_tokens_per_sec", "ttft_s"):
            parts.append(f"{plain_name(metric)} {value:+.1%}")
    test = result.test
    if test is not None and test.known and test.lower is not None and test.upper is not None:
        # Each bound is one-sided at ``alpha``; together they cover 1 - 2*alpha.
        sure = 1.0 - 2.0 * test.alpha
        parts.append(
            f"score {test.mean:+.1%} ({sure:.0%} range {test.lower:+.1%} to "
            f"{test.upper:+.1%}, {len(result.block_scores)} usable block(s))"
        )
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
    if result.unavailable_guards:
        parts.append("not measured: " + ", ".join(plain_name(m) for m in result.unavailable_guards))
    if result.episodes:
        parts.append(f"{len(result.episodes)} severe episode(s)")
    if result.stopped_early:
        parts.append(result.stopped_early)
    if result.guard_breaches:
        parts.append("worse on: " + ", ".join(plain_name(m) for m in result.guard_breaches))
    if result.mode == "better":
        verdict = result.judgement.verdict.value.replace("_", " ") if result.judgement else "?"
        tail = f"{result.decision}; keep rule: {verdict}"
    else:
        tail = result.decision + ("" if result.quality_passed else " (answers changed)")
    if result.decision == "unsure" and result.mode == "better":
        tail += " (no improvement shown within this budget)"
    return f"{len(result.turns)} block(s): " + ", ".join(parts) + f" -> {tail}"


__all__ = [
    "CLOCK_METRIC",
    "COUNT_GUARDS",
    "FAIR_METRICS",
    "MAX_BLOCKS",
    "FairResult",
    "TurnReading",
    "block_score",
    "compare_fairly",
    "decide",
    "fair_words",
    "guard_change",
    "guard_episodes",
    "measurement_value",
    "planned_looks",
    "throttled_any",
]
