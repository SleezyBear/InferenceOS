"""Goal mode, payback check and keep-decision contracts (step 4).

* The **effort mode** (:class:`~inferenceos.contracts.preference.OptimizationPreference`)
  decides only cost and time: the smallest gain worth keeping and how long to search.
* The **goal mode** (:class:`GoalPreference`) decides what counts as *better*: which
  measures come first, and which are guarded (must not get worse beyond a limit).
* The **payback check** (:class:`PaybackCheck`) asks whether the time a change saves,
  over the answering time the setup is expected to get, is worth more than the time
  still to be spent on it. Time already spent testing is never counted again.
* :class:`KeepJudgement` is the single keep decision shared by the watcher and the
  before-loading loop.

Everything here is inert data; the rules live in :mod:`inferenceos.optimization.goals`,
:mod:`inferenceos.optimization.payback` and :mod:`inferenceos.optimization.keep`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import GoalMode, Situation


class KeepVerdict(StrEnum):
    """The outcome of judging a measured change."""

    KEPT = "kept"
    #: Measured better, but not worth keeping (smallest-gain or payback check).
    NOT_WORTH_IT = "not_worth_it"
    #: Not better on the goal's measures, or it made a guarded measure worse.
    NOT_BETTER = "not_better"


class GoalPreference(ContractBase):
    """Resolved goal mode.

    ``first`` maps each measure that comes first to its weight (weights sum to 1).
    ``guarded`` lists measures that must not get worse by more than ``guard_limit``
    (a relative fraction, ``0.10`` == 10%).
    """

    mode: GoalMode = GoalMode.BALANCED
    first: dict[str, float] = Field(default_factory=dict)
    guarded: list[str] = Field(default_factory=list)
    guard_limit: float = Field(default=0.10, ge=0.0, le=1.0)
    source: str = "settings"
    notes: list[str] = Field(default_factory=list)


class GoalAssessment(ContractBase):
    """How a measured (or pre-measured) change scores against the goal.

    ``gain`` is the weighted relative improvement over the goal's first measures that
    were measured (``None`` when none of them was measured). A change within the
    measurement noise counts as zero, never as a gain or a loss.
    """

    gain: float | None = None
    measured: bool = False
    per_measure: dict[str, float] = Field(default_factory=dict)
    unmeasured: list[str] = Field(default_factory=list)
    guard_breaches: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


class PaybackCheck(ContractBase):
    """Does the time saved outweigh the time still to be spent?

    ``time_saved_s`` = ``answering_s`` x ``gain / (1 + gain)``: a change that makes
    answering ``gain`` faster saves that share of the answering time.
    """

    situation: Situation
    gain: float | None = None
    answering_s: float = Field(default=0.0, ge=0.0)
    time_saved_s: float | None = None
    cost_s: float = Field(default=0.0, ge=0.0)
    cost_terms: dict[str, float] = Field(default_factory=dict)
    pays_back: bool | None = None
    rationale: list[str] = Field(default_factory=list)


class KeepJudgement(ContractBase):
    """The shared keep decision for one measured change."""

    verdict: KeepVerdict
    keep: bool
    situation: Situation
    effort_mode: str
    goal_mode: str
    required_gain: float = Field(default=0.0, ge=0.0)
    goal: GoalAssessment = Field(default_factory=GoalAssessment)
    payback: PaybackCheck | None = None
    reasons: list[str] = Field(default_factory=list)


__all__ = [
    "GoalAssessment",
    "GoalPreference",
    "KeepJudgement",
    "KeepVerdict",
    "PaybackCheck",
]
