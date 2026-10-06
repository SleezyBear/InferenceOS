"""Goal mode: what counts as *better* (step 4).

The goal mode picks the measures that come first and the measures that are guarded.
It never touches cost or time (that is the effort mode) and never weakens a safety
gate. Answer quality is always checked separately and is never traded away.

Built-in goals (step 4):

* **balanced** - writing speed and time to first word, evenly; every other measure
  is guarded.
* **fastest_answers** - writing speed and time to first word; memory is guarded.
* **custom** - the user ranks the measures; every measure not ranked is guarded.

Goals that need readings added in step 6 (computer stays usable, quiet and cool,
save battery) are not available yet.
"""

from __future__ import annotations

from collections.abc import Mapping

from inferenceos.config import Settings
from inferenceos.contracts.enums import GoalMode
from inferenceos.contracts.experiment import ExperimentComparison
from inferenceos.contracts.goal import GoalAssessment, GoalPreference

#: Plain measure names -> (metric key, higher_is_better). These are the scorecard
#: measures available today.
MEASURES: dict[str, tuple[str, bool]] = {
    "writing_speed": ("decode_tokens_per_sec", True),
    "first_word": ("ttft_s", False),
    "prompt_reading": ("prompt_tokens_per_sec", True),
    "answer_time": ("total_latency_s", False),
    "memory": ("peak_rss_bytes", False),
    "paging": ("hard_page_faults_per_sec", False),
}
_HIGHER_IS_BETTER: dict[str, bool] = {metric: higher for metric, higher in MEASURES.values()}
_PLAIN_NAME: dict[str, str] = {metric: name for name, (metric, _) in MEASURES.items()}

_FIRST_SPEED = {"decode_tokens_per_sec": 0.5, "ttft_s": 0.5}


def _metric(name: str) -> str:
    return MEASURES[name][0]


def plain_name(metric: str) -> str:
    return _PLAIN_NAME.get(metric, metric)


def resolve_goal_mode(value: str) -> GoalMode:
    try:
        return GoalMode(value.strip().lower().replace("-", "_").replace(" ", "_"))
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in GoalMode)
        raise ValueError(
            f"invalid goal_mode {value!r}; expected one of: {valid} (the goals computer "
            "stays usable, quiet and cool and save battery arrive in step 6)"
        ) from exc


def build_goal_preference(settings: Settings) -> GoalPreference:
    """The resolved goal mode from the saved settings (raises on invalid values)."""
    mode = resolve_goal_mode(settings.goal_mode)
    limit_percent = float(settings.goal_guard_limit_percent)
    if not (0.0 <= limit_percent <= 100.0):
        raise ValueError("goal_guard_limit_percent must be between 0 and 100")
    guard_limit = limit_percent / 100.0
    everything = [metric for metric, _ in MEASURES.values()]

    if mode is GoalMode.BALANCED:
        first = dict(_FIRST_SPEED)
        guarded = [metric for metric in everything if metric not in first]
        notes = ["balanced: writing speed and first-word time evenly; the rest guarded"]
    elif mode is GoalMode.FASTEST_ANSWERS:
        first = dict(_FIRST_SPEED)
        guarded = [_metric("memory")]
        notes = ["fastest answers: writing speed and first-word time; memory guarded"]
    else:
        names = [
            item.strip().lower().replace("-", "_").replace(" ", "_")
            for item in settings.goal_custom_measures.split(",")
            if item.strip()
        ]
        if not names:
            raise ValueError(
                "goal_mode=custom requires goal_custom_measures, e.g. "
                "'writing_speed,first_word'"
            )
        unknown = [name for name in names if name not in MEASURES]
        if unknown:
            raise ValueError(
                f"unknown goal measures {unknown}; choose from: {', '.join(MEASURES)}"
            )
        if len(set(names)) != len(names):
            raise ValueError("goal_custom_measures lists a measure twice")
        # Ranked: the first measure weighs most (n, n-1, ..., 1), normalised to 1.
        total = sum(range(1, len(names) + 1))
        first = {
            _metric(name): (len(names) - index) / total for index, name in enumerate(names)
        }
        guarded = [metric for metric in everything if metric not in first]
        notes = [f"custom ranking: {', '.join(names)}"]

    return GoalPreference(
        mode=mode,
        first=first,
        guarded=guarded,
        guard_limit=guard_limit,
        source="settings.goal_mode",
        notes=notes,
    )


def _improvement(metric: str, relative_delta: float) -> float:
    """Relative change turned into an improvement (positive is better)."""
    higher = _HIGHER_IS_BETTER.get(metric, True)
    return relative_delta if higher else -relative_delta


def improvement(metric: str, relative_delta: float) -> float:
    """Public, signed improvement for a metric (positive is better).

    Used where a single run's change must be scored the same way the goal scores a
    full comparison.
    """
    return _improvement(metric, relative_delta)


def _weighted_gain(
    goal: GoalPreference, per_measure: Mapping[str, float], *, require_all: bool = False
) -> float | None:
    """The goal gain: the goal's first measures weighed with the goal's fixed weights.

    Every measured first measure counts with its weight, whether it changed or not
    (user decision, 3 Oct 2026, replacing "the biggest improvement counts in full when
    nothing got worse"), so the same changes always give the same score: under
    balanced, +8% writing speed alone is +4%, and +8% writing speed with an 8% slower
    first word is 0%. For a custom goal the weights come from the user's ranking.

    ``None`` when none of the goal's first measures was measured, or, with
    ``require_all``, unless **every** one was (review, 4 Oct 2026: scoring only what
    happened to be measured changes the weights, e.g. +8% writing speed alone counts
    +8% instead of +4% under balanced). ``tune``'s fair comparison always requires all
    (``tune/fair.py`` ``block_score``); the older general comparison and the run-time
    estimates of an expected gain do not.
    """
    if require_all and any(metric not in per_measure for metric in goal.first):
        return None
    measured = {metric: per_measure[metric] for metric in goal.first if metric in per_measure}
    if not measured:
        return None
    total_weight = sum(goal.first[metric] for metric in measured)
    if total_weight <= 0.0:
        return 0.0
    return sum(value * goal.first[metric] for metric, value in measured.items()) / total_weight


def assess_comparison(
    comparison: ExperimentComparison | None, goal: GoalPreference
) -> GoalAssessment:
    """Score a measured before/after comparison against the goal.

    Only changes the comparison marked as real (``improved`` or ``regressed``, i.e.
    beyond the measurement noise) count; a change inside the noise counts as zero.
    """
    if comparison is None:
        return GoalAssessment(rationale=["no measurement to judge"])
    per_measure: dict[str, float] = {}
    unmeasured: list[str] = []
    rationale: list[str] = []
    for metric in goal.first:
        delta = comparison.deltas.get(metric)
        if delta is None or delta.relative_delta is None:
            unmeasured.append(metric)
            continue
        if delta.improved or delta.regressed:
            value = _improvement(metric, float(delta.relative_delta))
        else:
            value = 0.0
        per_measure[metric] = value
        rationale.append(f"{plain_name(metric)}: {value:+.1%}")

    breaches: list[str] = []
    for metric in goal.guarded:
        delta = comparison.deltas.get(metric)
        if delta is None or delta.relative_delta is None or not delta.regressed:
            continue
        worsening = -_improvement(metric, float(delta.relative_delta))
        if worsening > goal.guard_limit:
            breaches.append(metric)
            rationale.append(
                f"guarded {plain_name(metric)} got {worsening:.1%} worse "
                f"(limit {goal.guard_limit:.0%})"
            )

    gain = _weighted_gain(goal, per_measure)
    rationale.append(
        "goal gain: unknown (none of the goal's measures was measured)"
        if gain is None
        else f"goal gain: {gain:+.1%}"
    )
    return GoalAssessment(
        gain=gain,
        measured=gain is not None,
        per_measure=per_measure,
        unmeasured=unmeasured,
        guard_breaches=breaches,
        rationale=rationale,
    )


def assess_known_gains(
    gains: Mapping[str, float], goal: GoalPreference, *, require_all: bool = False
) -> GoalAssessment:
    """Score gains already measured before a test (e.g. by the layer search).

    ``gains`` maps metric -> relative improvement (positive is better).
    """
    per_measure = {metric: float(value) for metric, value in gains.items() if metric in goal.first}
    unmeasured = [metric for metric in goal.first if metric not in per_measure]
    gain = _weighted_gain(goal, per_measure, require_all=require_all)
    rationale = [f"{plain_name(metric)}: {value:+.1%}" for metric, value in per_measure.items()]
    rationale.append(
        "goal gain: unknown (not every one of the goal's measures was measured)"
        if gain is None
        else f"goal gain (from measurements taken before the test): {gain:+.1%}"
    )
    return GoalAssessment(
        gain=gain,
        measured=gain is not None,
        per_measure=per_measure,
        unmeasured=unmeasured,
        rationale=rationale,
    )


__all__ = [
    "MEASURES",
    "assess_comparison",
    "assess_known_gains",
    "build_goal_preference",
    "improvement",
    "plain_name",
    "resolve_goal_mode",
]
