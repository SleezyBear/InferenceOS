"""Asking the user about each proposed change (step 5).

Each change is shown with its measured gain and gets its own yes or no. No answer in
time, or no console to answer on, is a no: a change is never kept without a yes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from inferenceos.contracts.tune import ChangeDecision, ProposedChange
from inferenceos.runtime.approval import ask_line


def describe_change(change: ProposedChange) -> str:
    gain = (
        f"measured {change.measured_gain:+.1%} on your goal"
        if change.measured_gain is not None
        else "gain not measured on its own"
    )
    return f"{change.description}: {change.old_value} -> {change.new_value} ({gain})"


class ChangeApprover(ABC):
    @abstractmethod
    def decide(self, change: ProposedChange, *, timeout_s: float) -> ChangeDecision: ...


class ConsoleChangeApprover(ChangeApprover):
    """Asks y/n on the console, waiting at most ``timeout_s``."""

    def decide(self, change: ProposedChange, *, timeout_s: float) -> ChangeDecision:
        answer = ask_line(
            f"Keep this change? {describe_change(change)} [y/N] "
            f"(waiting up to {timeout_s:.0f}s) ",
            timeout_s=timeout_s,
        )
        if answer is None:
            return ChangeDecision(
                change_id=change.change_id,
                approved=False,
                answer="no answer",
                detail="no answer in time (or no console); not kept",
            )
        text = answer.strip().lower()
        approved = text in ("y", "yes")
        return ChangeDecision(
            change_id=change.change_id, approved=approved, answer=text or "(empty)"
        )


class ApproveAllChanges(ChangeApprover):
    """Says yes to every change (only for checks the user runs on purpose)."""

    def decide(self, change: ProposedChange, *, timeout_s: float) -> ChangeDecision:
        return ChangeDecision(
            change_id=change.change_id, approved=True, answer="yes", detail="approved automatically"
        )


class ScriptedChangeApprover(ChangeApprover):
    """Answers from a list (tests). Runs out -> no."""

    def __init__(self, answers: Iterable[bool]) -> None:
        self._answers = list(answers)
        self.asked: list[ProposedChange] = []

    def decide(self, change: ProposedChange, *, timeout_s: float) -> ChangeDecision:
        self.asked.append(change)
        approved = self._answers.pop(0) if self._answers else False
        return ChangeDecision(
            change_id=change.change_id, approved=approved, answer="yes" if approved else "no"
        )


__all__ = [
    "ApproveAllChanges",
    "ChangeApprover",
    "ConsoleChangeApprover",
    "ScriptedChangeApprover",
    "describe_change",
]
