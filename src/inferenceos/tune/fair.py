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
from typing import Any, Literal
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from inferenceos.config import Settings
from inferenceos.contracts.enums import OptimizationMode, Situation
from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentPolicy,
    RuntimeConfiguration,
)
from inferenceos.contracts.goal import GoalAssessment, KeepJudgement, KeepVerdict
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

#: ``(lower, upper)``: what a count-like reading is known to lie between. An exact
#: reading is ``(v, v)``.
Bounds = tuple[float, float]


def run_bounds(run: object, metric: str) -> Bounds | None:
    """One run's reading of a count-like measure as bounds (reviewer, monitor round 3).

    A run may carry ``<metric>_lower`` / ``<metric>_upper`` (the experimental scoped
    monitor's peak-rate bounds; the exact value is then set only when they coincide);
    otherwise its exact value counts as ``(v, v)``. ``None`` when it has neither."""
    lower, upper = getattr(run, f"{metric}_lower", None), getattr(run, f"{metric}_upper", None)
    if lower is not None and upper is not None:
        return float(lower), float(upper)
    value = getattr(run, metric, None)
    return None if value is None else (float(value), float(value))


def measurement_bounds(measurement: ExperimentMeasurement, metric: str) -> Bounds | None:
    """Bounds on a measurement's middle value: the median of the runs' lower bounds and
    the median of their upper bounds (the median never decreases when any run's value
    increases, so the true median lies between them). Runs with no reading are left
    out, as ``measurement_value`` does; with no per-run reading the measurement's own
    summary counts as exact."""
    found = [b for b in (run_bounds(run, metric) for run in measurement.run_metrics) if b is not None]
    if found:
        return (float(statistics.median(b[0] for b in found)),
                float(statistics.median(b[1] for b in found)))
    stats = measurement.metrics.get(metric)
    if stats is not None and stats.mean is not None:
        return float(stats.mean), float(stats.mean)
    return None


def measurement_value(measurement: ExperimentMeasurement, metric: str) -> float | None:
    """The middle value of one measure over a measurement's runs.

    The median of the per-run values is used (one odd run does not move it); when no
    per-run values exist the measurement's own summary is used. A count-like measure
    known only within bounds has no single value (``None``): its guard is evaluated
    over the bounds (``count_guard_verdict``), never on an upper bound passed off as a
    reading.
    """
    if metric in COUNT_GUARDS:
        bounds = measurement_bounds(measurement, metric)
        if bounds is None or bounds[1] - bounds[0] > 1e-9:
            return None
        return bounds[0]
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
    #: Count-like guarded measures known only within bounds (``(lower, upper)``); a
    #: measure missing here is exact (its value above) or not read.
    start_bounds: dict[str, Bounds] = field(default_factory=dict)
    candidate_bounds: dict[str, Bounds] = field(default_factory=dict)


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
    #: Count-like guarded measures whose bounds leave the guard's decision open (it
    #: passes under some supported values and breaches under others): not a pass.
    undecided_guards: list[str] = field(default_factory=list)
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
    #: Runs that failed to execute (an error from the engine, a crash, a failed load),
    #: per setup, over every attempted block, also blocks that could not be counted
    #: (review 6: a later good block must not erase a crash).
    start_failures: int = 0
    candidate_failures: int = 0
    #: Every block that could not be counted, with why (kept as evidence).
    discarded_blocks: list[dict[str, object]] = field(default_factory=list)
    #: For each guarded measure: in how many counted blocks both setups had a reading.
    guard_coverage: dict[str, int] = field(default_factory=dict)
    #: "ok", "candidate failed to run" (the new setup failed and the current one did
    #: not) or "both failed to run" (cannot tell whose fault; inconclusive).
    reliability: str = "ok"

    def acceptance_blockers(self) -> list[str]:
        """Why this comparison cannot support accepting the new setup, whatever its
        speed (review 6): a required measure missing in any counted block, or runs of
        the new setup that failed to execute. Empty when nothing blocks."""
        blockers: list[str] = []
        if self.unavailable_guards:
            blockers.append(
                "not measured in every counted block: "
                + ", ".join(plain_name(m) for m in self.unavailable_guards)
            )
        if self.undecided_guards:
            blockers.append(
                "guard not settled by the readings' bounds: "
                + ", ".join(plain_name(m) for m in self.undecided_guards)
            )
        if self.reliability != "ok":
            blockers.append(
                f"{self.reliability} ({self.candidate_failures} failed run(s) of the new "
                f"setup, {self.start_failures} of the current one)"
            )
        return blockers

    @property
    def not_worse(self) -> bool:
        """Shown not worse (beyond the margin), with good answers, no guard breach, every
        guarded measure read in every counted block and no failed runs of the new setup."""
        return bool(
            self.decision in ("not worse", "better")
            and self.quality_passed
            and not self.guard_breaches
            and not self.acceptance_blockers()
        )

    @property
    def kept(self) -> bool:
        return bool(
            self.judgement is not None
            and self.judgement.keep
            and not self.acceptance_blockers()
        )

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
                    **({"start_bounds": item.start_bounds} if item.start_bounds else {}),
                    **({"candidate_bounds": item.candidate_bounds} if item.candidate_bounds else {}),
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
            "undecided_guards": self.undecided_guards,
            "episodes": self.episodes,
            "attempts": self.attempts,
            "start_failures": self.start_failures,
            "candidate_failures": self.candidate_failures,
            "discarded_blocks": self.discarded_blocks,
            "guard_coverage": self.guard_coverage,
            "reliability": self.reliability,
            "acceptance_blockers": self.acceptance_blockers(),
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


def _bound_pairs(blocks: Sequence[TurnReading], metric: str) -> list[tuple[Bounds, Bounds]]:
    """Per counted block: (current setup's bounds, new setup's bounds); an exact value
    counts as ``(v, v)``. Blocks where either side was not read are left out."""
    pairs: list[tuple[Bounds, Bounds]] = []
    for item in blocks:
        sides: list[Bounds] = []
        for values, bounds in ((item.start, item.start_bounds), (item.candidate, item.candidate_bounds)):
            known = bounds.get(metric)
            if known is None:
                value = values.get(metric)
                known = None if value is None else (float(value), float(value))
            if known is not None:
                sides.append(known)
        if len(sides) == 2:
            pairs.append((sides[0], sides[1]))
    return pairs


@dataclass(frozen=True)
class CountGuardVerdict:
    """A count-like guard decided over every value the readings support (reviewer,
    monitor round 3). ``typical`` judges the median per-block paired difference (new
    minus current) against the tolerance; ``episodes`` the severe-episode rule. Each is
    "pass" (passes under the most adverse supported values), "breach" (breaches under
    the most favourable ones too) or "unknown" (either is possible)."""

    state: str  # "pass" | "breach" | "unknown" | "unread"
    typical: str
    episodes: str
    #: Bounds on the median paired difference (per second).
    middle_low: float | None = None
    middle_high: float | None = None
    #: Severe-episode excess of the new setup: least and most the readings allow.
    excess_low: int = 0
    excess_high: int = 0
    words: tuple[str, ...] = ()


def count_guard_verdict(blocks: Sequence[TurnReading], metric: str) -> CountGuardVerdict:
    """The guard's existing rules and thresholds, over the readings' bounds.

    For each block the paired difference lies in ``[new_lower - current_upper,
    new_upper - current_lower]`` and the median is monotone, so the median difference
    lies between the medians of those ends. The typical-use rule passes only if the
    most adverse end is within the tolerance and breaches only if the most favourable
    end exceeds it. Severe episodes: a reading is certainly an episode when its lower
    bound reaches the level and possibly one when its upper bound does; the new setup's
    excess over the current one is bounded the same way against ``EPISODE_EXCESS``.
    Exact readings give exactly the previous decisions."""
    tolerance, level = COUNT_GUARDS[metric]
    pairs = _bound_pairs(blocks, metric)
    if not pairs:
        return CountGuardVerdict("unread", "unread", "unread")
    low = float(statistics.median(c[0] - s[1] for s, c in pairs))
    high = float(statistics.median(c[1] - s[0] for s, c in pairs))
    typical = "pass" if high <= tolerance else ("breach" if low > tolerance else "unknown")
    words: list[str] = []
    certain = {"start": 0, "candidate": 0}
    possible = {"start": 0, "candidate": 0}
    for item in blocks:
        for side, label in (("start", "current"), ("candidate", "new")):
            mine = getattr(item, side)
            known = getattr(item, f"{side}_bounds").get(metric)
            if known is None:
                value = mine.get(metric)
                if value is None:
                    continue
                known = (float(value), float(value))
            if known[1] < level:
                continue
            possible[side] += 1
            sure = known[0] >= level
            certain[side] += int(sure)
            other = item.candidate if side == "start" else item.start
            speed = []
            for name in ("decode_tokens_per_sec", "ttft_s"):
                a, b = mine.get(name), other.get(name)
                if a is not None and b is not None and b:
                    change = improvement(name, (float(a) - float(b)) / abs(float(b)))
                    speed.append(f"{plain_name(name)} {change:+.0%}")
            reading = (f"{known[0]:,.0f}/s" if known[0] == known[1] else
                       f"between {known[0]:,.0f}/s and {known[1]:,.0f}/s (bounds, not exact)")
            words.append(
                f"block {item.turn}: {plain_name(metric)} {reading} in the {label} setup"
                + ("" if sure else " -- possibly a severe episode")
                + (f" (vs its partner, + is better: {', '.join(speed)})" if speed else "")
            )
    excess_low = certain["candidate"] - possible["start"]
    excess_high = possible["candidate"] - certain["start"]
    episodes = ("breach" if excess_low >= EPISODE_EXCESS else
                "pass" if excess_high < EPISODE_EXCESS else "unknown")
    state = ("breach" if "breach" in (typical, episodes) else
             "unknown" if "unknown" in (typical, episodes) else "pass")
    return CountGuardVerdict(state, typical, episodes, low, high, excess_low, excess_high, tuple(words))


def guard_change(blocks: Sequence[TurnReading], metric: str) -> float | None:
    """How much a guarded measure improved (negative = worse), over the counted blocks.

    Proportional measures (memory, total time): the mean of the per-block paired
    relative changes. Count-like measures (``COUNT_GUARDS``, e.g. paging): the median of
    the per-block paired differences, as a fraction of the typical tolerance (so -1.0
    means "most blocks page one tolerance more"). Severe episodes are judged separately
    (``guard_episodes``). ``None`` when the measure was not read -- or, for a count-like
    measure known only within bounds, when the bounds leave the typical-use decision
    open (``count_guard_verdict``); otherwise the value on the deciding side (the most
    adverse when it passes, the most favourable when it breaches).
    """
    if metric in COUNT_GUARDS:
        tolerance, _ = COUNT_GUARDS[metric]
        verdict = count_guard_verdict(blocks, metric)
        if verdict.middle_low is None or verdict.middle_high is None:
            return None
        if verdict.typical == "pass":
            return improvement(metric, verdict.middle_high / tolerance)
        if verdict.typical == "breach":
            return improvement(metric, verdict.middle_low / tolerance)
        return None
    pairs = _pairs(blocks, metric)
    if not pairs:
        return None
    ratios = [improvement(metric, (new - start) / abs(start)) for start, new in pairs if start]
    if not ratios:
        return None
    return float(statistics.fmean(ratios))


def guard_episodes(
    blocks: Sequence[TurnReading], metric: str
) -> tuple[bool, list[str]]:
    """Severe episodes of a count-like measure: (certainly counts against the new
    setup, words). Readings known only within bounds that may be episodes are named as
    such; whether they leave the rule open is in ``count_guard_verdict``."""
    if metric not in COUNT_GUARDS:
        return False, []
    verdict = count_guard_verdict(blocks, metric)
    return verdict.episodes == "breach", list(verdict.words)


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
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> FairResult:
    """Measure ``start`` and ``candidate`` in independent blocks and judge fairly.

    Judged only at the planned looks, on exactly the planned number of usable blocks.
    Unusable blocks (a measurement without enough good runs, or a goal measure that was
    not read) are attempted again up to ``tune_fair_extra_attempts`` times in total.
    Running out of time or attempts before a look decides gives "unsure".

    ``progress`` (review 10) receives explicit progress events, so a caller can pace
    the comparison from completed work instead of guessing from warm-ups:

    * ``{"event": "boundary", ...}`` before each block is started (also before the
      first): the only point where the callback may stop the comparison (by raising);
      never sent after the last look has decided;
    * ``{"event": "measurement", "side", "wall_s", "retries", ...}`` after each side's
      measurement (``retries``: disturbed repeats inside it, which cost time but do not
      count as blocks);
    * ``{"event": "block", "counted": bool, "why", ...}`` once a block is judged; a
      counted block also carries its ``score`` (so a caller keeps the evidence of a
      comparison it stops before the result is returned).

    Each event carries ``workload`` (id), ``attempts``, ``usable``, ``discarded``,
    ``planned`` (blocks at the last look) and ``max_attempts``.
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

    def tell(event: str, **extra: Any) -> None:
        if progress is not None:
            progress({
                "event": event, "workload": workload.workload_id, "mode": mode,
                "attempts": result.attempts, "usable": len(scores),
                "discarded": len(result.discarded_blocks), "planned": looks[-1],
                "max_attempts": max_attempts, **extra,
            })

    while result.attempts < max_attempts and decided is None:
        tell("boundary")
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
            repeats_before, began_side = watch.repeats, time.perf_counter()
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
            tell("measurement", side=label, block=index + 1,
                 wall_s=time.perf_counter() - began_side, retries=watch.repeats - repeats_before)
        result.attempts += 1
        number = result.attempts
        start_m, cand_m = measured["start"], measured["candidate"]
        result.start_failures += start_m.invalid_runs
        result.candidate_failures += cand_m.invalid_runs
        if (
            start_m.valid_runs < policy.min_valid_samples
            or cand_m.valid_runs < policy.min_valid_samples
        ):
            result.discarded_blocks.append(
                {
                    "block": number,
                    "why": "not enough good runs",
                    "good_runs": [start_m.valid_runs, cand_m.valid_runs],
                    "failed_runs": [start_m.invalid_runs, cand_m.invalid_runs],
                    "notes": (start_m.notes + cand_m.notes)[-4:],
                }
            )
            result.notes.append(
                f"block {number} could not be used "
                f"({start_m.valid_runs}/{cand_m.valid_runs} good runs"
                + "".join(f"; {note}" for note in (start_m.notes + cand_m.notes)[-2:])
                + ")"
            )
            tell("block", counted=False, block=number, why="not enough good runs")
            if (
                result.candidate_failures
                and not result.start_failures
                and cand_m.invalid_runs
            ):
                # The new setup failed to run while the current one ran: nothing a
                # later good block shows can make it acceptable, so stop measuring.
                result.stopped_early = (
                    f"the new setup failed to run in block {number} "
                    "(the current setup ran)"
                )
                break
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
            result.discarded_blocks.append(
                {"block": number, "why": "not read: " + ", ".join(missing)}
            )
            result.notes.append(
                f"block {number} could not be used (not read: {', '.join(missing)})"
            )
            tell("block", counted=False, block=number, why="not read")
            continue
        result.start_runs.append(start_m)
        result.candidate_runs.append(cand_m)
        # Count-like measures known only within bounds keep them (no single value).
        start_bounds: dict[str, Bounds] = {}
        cand_bounds: dict[str, Bounds] = {}
        for metric in COUNT_GUARDS:
            for side_m, into in ((start_m, start_bounds), (cand_m, cand_bounds)):
                known = measurement_bounds(side_m, metric)
                if known is not None and known[1] - known[0] > 1e-9:
                    into[metric] = known
        result.turns.append(
            TurnReading(
                turn=number,
                order=",".join(order),
                start=start_values,
                candidate=cand_values,
                improvements=improvements,
                start_bounds=start_bounds,
                candidate_bounds=cand_bounds,
            )
        )
        scores.append(score)
        tell("block", counted=True, block=number, score=score)
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
    if result.candidate_failures:
        # Review 6: a failed run of the new setup is evidence it may crash. When the
        # current setup ran cleanly in the same comparison the failure is the new
        # setup's; when both failed it cannot be told apart from the machine (another
        # program, a driver) and the comparison is inconclusive. Either way it is not
        # accepted; disturbed measurements (other activity) are not failures.
        result.reliability = (
            "both failed to run" if result.start_failures else "candidate failed to run"
        )
        result.notes.append(
            f"{result.candidate_failures} run(s) of the new setup failed to execute "
            f"({result.start_failures} of the current setup) -> not accepted"
        )
    elif result.start_failures:
        result.notes.append(
            f"{result.start_failures} run(s) of the current setup failed to execute "
            "(the new setup ran every time)"
        )

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
        counted = metric in COUNT_GUARDS
        covered = len(_bound_pairs(result.turns, metric) if counted else _pairs(result.turns, metric))
        result.guard_coverage[metric] = covered
        if covered < len(result.turns):
            # Review 6: every counted block needs the reading, not just one of them.
            result.unavailable_guards.append(metric)
        _, words = guard_episodes(result.turns, metric)
        result.episodes.extend(words)
        if counted and covered:
            verdict = count_guard_verdict(result.turns, metric)
            if verdict.state == "unknown":
                # Reviewer, monitor round 3: the readings' bounds allow both a pass and
                # a breach -- neither is reported; it is not a pass.
                tolerance, _ = COUNT_GUARDS[metric]
                result.undecided_guards.append(metric)
                parts = []
                if verdict.typical == "unknown":
                    parts.append(f"the middle per-block change is between {verdict.middle_low:+,.0f}/s and "
                                 f"{verdict.middle_high:+,.0f}/s (limit {tolerance:,.0f}/s)")
                if verdict.episodes == "unknown":
                    parts.append(f"the new setup has {verdict.excess_low} to {verdict.excess_high} more severe "
                                 f"episodes (limit {EPISODE_EXCESS})")
                result.notes.append(f"guarded {plain_name(metric)} not settled by its bounds: "
                                    + "; ".join(parts) + " -> unknown, not a pass")
    if result.undecided_guards and result.decision in ("better", "not worse"):
        result.notes.append(
            f"decision '{result.decision}' cannot stand while a guard is unsettled -> unsure"
        )
        result.decision = "unsure"
    if result.unavailable_guards:
        result.notes.append(
            "not measured in every counted block (unavailable, not counted as a pass): "
            + ", ".join(
                f"{plain_name(m)} ({result.guard_coverage[m]} of {len(result.turns)})"
                for m in result.unavailable_guards
            )
        )
        if result.decision in ("better", "not worse"):
            result.notes.append(
                f"decision '{result.decision}' cannot stand without every guarded "
                "measure -> unsure"
            )
            result.decision = "unsure"
    if result.reliability != "ok" and result.decision in ("better", "not worse"):
        result.decision = "unsure"
    result.notes.extend(f"severe episode, {words}" for words in result.episodes)

    candidate_pool = pool(result.candidate_runs, phase=f"{phase}-candidate")
    # Execution failures from blocks that were not counted belong to the evidence too.
    candidate_pool = candidate_pool.model_copy(
        update={"invalid_runs": result.candidate_failures}
    )
    quality = validate_quality(
        baseline=pool(result.start_runs, phase=f"{phase}-start"),
        candidate=candidate_pool,
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
    judgement = judge_assessment(
        assessment,
        technically_better=technically_better,
        effort=rules.effort,
        goal=rules.goal,
        situation=Situation.BEFORE_LOADING,
        settings=settings,
        remaining_cost_terms={},
    )
    blockers = result.acceptance_blockers()
    if blockers and judgement.keep:
        judgement = judgement.model_copy(
            update={
                "keep": False,
                "verdict": KeepVerdict.NOT_BETTER,
                "reasons": [*judgement.reasons, *blockers],
            }
        )
    result.judgement = judgement
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
    if result.reliability != "ok":
        parts.append(
            f"{result.reliability}: {result.candidate_failures} failed run(s) of the new setup"
        )
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
    "Bounds",
    "CountGuardVerdict",
    "count_guard_verdict",
    "guard_change",
    "guard_episodes",
    "measurement_bounds",
    "run_bounds",
    "measurement_value",
    "planned_looks",
    "throttled_any",
]
