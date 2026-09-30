"""The shared keep rule (step 4), used by the watcher and the before-loading loop.

A measured change is kept only when all of these hold:

1. the experiment itself says it improved and is safe to keep (quality passed,
   nothing regressed, the undo works) - the technical decision is never overridden;
2. it improved the goal mode's first measures by at least the effort mode's smallest
   worthwhile gain;
3. no guarded measure got worse beyond the goal's limit;
4. it pays back: the time it saves over the expected answering time is more than the
   time still to be spent on it. Time already spent testing is sunk and never counted.

A change that passes 1 and 3 and measured better, but fails 2 or 4, is
``NOT_WORTH_IT``: remembered with its numbers, never counted as a bad result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from inferenceos.config import Settings
from inferenceos.contracts.enums import PerformanceState, Situation
from inferenceos.contracts.experiment import ExperimentComparison
from inferenceos.contracts.goal import (
    GoalAssessment,
    GoalPreference,
    KeepJudgement,
    KeepVerdict,
    PaybackCheck,
)
from inferenceos.contracts.preference import OptimizationPreference
from inferenceos.optimization.goals import assess_comparison, plain_name
from inferenceos.optimization.payback import payback_check


@dataclass(frozen=True)
class ModeRules:
    """The effort mode, the goal mode and the settings they were built from."""

    effort: OptimizationPreference
    goal: GoalPreference
    settings: Settings

    @classmethod
    def from_settings(cls, settings: Settings) -> ModeRules:
        from inferenceos.optimization.goals import build_goal_preference
        from inferenceos.optimization.preference import build_optimization_preference

        return cls(
            effort=build_optimization_preference(settings),
            goal=build_goal_preference(settings),
            settings=settings,
        )

    def describe(self) -> str:
        return (
            f"effort={self.effort.mode.value} "
            f"(smallest gain {self.effort.minimum_improvement:.1%}) "
            f"goal={self.goal.mode.value}"
        )


def judge_assessment(
    assessment: GoalAssessment,
    *,
    technically_better: bool,
    effort: OptimizationPreference,
    goal: GoalPreference,
    situation: Situation,
    settings: Settings,
    remaining_cost_terms: Mapping[str, float] | None = None,
    performance_state: PerformanceState = PerformanceState.UNKNOWN,
) -> KeepJudgement:
    """Apply the keep rule to an already-computed goal assessment."""
    required = effort.minimum_improvement_for(performance_state)
    reasons = list(assessment.rationale)

    def verdict(
        kind: KeepVerdict, why: str, payback: PaybackCheck | None = None
    ) -> KeepJudgement:
        return KeepJudgement(
            verdict=kind,
            keep=kind is KeepVerdict.KEPT,
            situation=situation,
            effort_mode=effort.mode.value,
            goal_mode=goal.mode.value,
            required_gain=required,
            goal=assessment,
            payback=payback,
            reasons=[*reasons, why],
        )

    if not technically_better:
        return verdict(KeepVerdict.NOT_BETTER, "the experiment did not show a safe improvement")
    if assessment.guard_breaches:
        names = ", ".join(plain_name(metric) for metric in assessment.guard_breaches)
        return verdict(KeepVerdict.NOT_BETTER, f"refused: made guarded {names} worse")
    gain = assessment.gain
    if gain is None or gain <= 0.0:
        return verdict(
            KeepVerdict.NOT_BETTER,
            f"no gain on the {goal.mode.value} goal's measures",
        )
    if gain < required:
        return verdict(
            KeepVerdict.NOT_WORTH_IT,
            f"gain {gain:.1%} is below the {effort.mode.value} effort mode's smallest "
            f"worthwhile gain {required:.1%}",
        )
    payback = payback_check(
        gain=gain,
        cost_terms=dict(remaining_cost_terms or {}),
        situation=situation,
        settings=settings,
    )
    reasons.extend(payback.rationale)
    if payback.pays_back is False:
        return verdict(KeepVerdict.NOT_WORTH_IT, "does not pay back", payback)
    return verdict(
        KeepVerdict.KEPT,
        f"kept: {gain:.1%} on the {goal.mode.value} goal, at least {required:.1%}, "
        "and it pays back",
        payback,
    )


def judge_keep(
    comparison: ExperimentComparison | None,
    *,
    technically_better: bool,
    effort: OptimizationPreference,
    goal: GoalPreference,
    situation: Situation,
    settings: Settings,
    remaining_cost_terms: Mapping[str, float] | None = None,
    performance_state: PerformanceState = PerformanceState.UNKNOWN,
) -> KeepJudgement:
    """Judge a measured change (see the module docstring)."""
    return judge_assessment(
        assess_comparison(comparison, goal),
        technically_better=technically_better,
        effort=effort,
        goal=goal,
        situation=situation,
        settings=settings,
        remaining_cost_terms=remaining_cost_terms,
        performance_state=performance_state,
    )


__all__ = ["ModeRules", "judge_assessment", "judge_keep"]
