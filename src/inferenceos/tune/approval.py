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
    together = "".join(
        f"; together with {name}: {change.related_old.get(name, 'program default')} -> {value}"
        for name, value in change.related.items()
    )
    old = change.old_value if change.old_value not in (None, "") else "program default"
    new = change.new_value if change.new_value not in (None, "") else "program default"
    return f"{change.description}: {old} -> {new}{together} ({gain})"


class ChangeApprover(ABC):
    #: False when this approver can never say yes to a trade-off (``--yes``): tune then
    #: reports the trade-offs it found without spending time measuring them.
    can_accept_tradeoffs: bool = True

    @abstractmethod
    def decide(self, change: ProposedChange, *, timeout_s: float) -> ChangeDecision: ...

    def decide_tradeoff(
        self, change: ProposedChange, *, words: str, timeout_s: float
    ) -> ChangeDecision:
        """A trade-off (better on one question, a loss beyond the limit on another)
        needs an explicit yes (owner, 5 Oct 2026). Without a way to ask: no."""
        return ChangeDecision(
            change_id=change.change_id,
            approved=False,
            answer="not asked",
            detail="a trade-off needs your explicit yes; not kept",
        )


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

    def decide_tradeoff(
        self, change: ProposedChange, *, words: str, timeout_s: float
    ) -> ChangeDecision:
        return _console_tradeoff(change, words, timeout_s)


def _console_tradeoff(change: ProposedChange, words: str, timeout_s: float) -> ChangeDecision:
    answer = ask_line(
        f"TRADE-OFF (not kept unless you say yes): {words}. Keep it anyway? [y/N] "
        f"(waiting up to {timeout_s:.0f}s) ",
        timeout_s=timeout_s,
    )
    if answer is None:
        return ChangeDecision(
            change_id=change.change_id, approved=False, answer="no answer",
            detail="no answer in time (or no console); not kept",
        )
    text = answer.strip().lower()
    return ChangeDecision(
        change_id=change.change_id, approved=text in ("y", "yes"), answer=text or "(empty)"
    )


class ApproveAllChanges(ChangeApprover):
    """Says yes to every change (only for checks the user runs on purpose), but never to
    a trade-off: that always needs an explicit yes (owner, 5 Oct 2026)."""

    can_accept_tradeoffs = False

    def decide(self, change: ProposedChange, *, timeout_s: float) -> ChangeDecision:
        return ChangeDecision(
            change_id=change.change_id, approved=True, answer="yes", detail="approved automatically"
        )

    def decide_tradeoff(
        self, change: ProposedChange, *, words: str, timeout_s: float
    ) -> ChangeDecision:
        return ChangeDecision(
            change_id=change.change_id,
            approved=False,
            answer="not asked (--yes)",
            detail="trade-offs are never approved automatically, even with --yes",
        )


class ScriptedChangeApprover(ChangeApprover):
    """Answers from a list (tests). Runs out -> no."""

    def __init__(self, answers: Iterable[bool], tradeoff_answers: Iterable[bool] = ()) -> None:
        self._answers = list(answers)
        self._tradeoff_answers = list(tradeoff_answers)
        self.asked: list[ProposedChange] = []
        self.tradeoffs_asked: list[str] = []

    def decide_tradeoff(
        self, change: ProposedChange, *, words: str, timeout_s: float
    ) -> ChangeDecision:
        self.tradeoffs_asked.append(words)
        approved = self._tradeoff_answers.pop(0) if self._tradeoff_answers else False
        return ChangeDecision(
            change_id=change.change_id, approved=approved, answer="yes" if approved else "no"
        )

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
