"""Approval abstraction for disruptive active-runtime intervention.

Approval is always explicit and bound to a specific proposal + context. A future
desktop/UI notification system can replace these implementations; the runtime engine
only depends on the abstract interface.
"""

from __future__ import annotations

import sys
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone

from inferenceos.contracts.enums import ApprovalOutcome
from inferenceos.contracts.runtime import ApprovalRecord, RuntimeRecommendation


class ApprovalProvider(ABC):
    """Requests approval for a runtime recommendation."""

    @abstractmethod
    def request(
        self, recommendation: RuntimeRecommendation, *, timeout_s: float
    ) -> ApprovalRecord:
        ...


def _record(
    recommendation: RuntimeRecommendation,
    outcome: ApprovalOutcome,
    *,
    decided_by: str | None,
    detail: str = "",
    proposal_id: str | None = None,
    context_identity: str | None = None,
) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=f"appr-{uuid.uuid4().hex[:10]}",
        proposal_id=proposal_id if proposal_id is not None else recommendation.proposal_id,
        outcome=outcome,
        requested_at=datetime.now(timezone.utc),
        decided_at=datetime.now(timezone.utc) if outcome is not ApprovalOutcome.PENDING else None,
        context_identity=(
            context_identity if context_identity is not None else recommendation.context_identity
        ),
        decided_by=decided_by,
        detail=detail,
    )


class AutoRejectApprovalProvider(ApprovalProvider):
    """Never approves (safe default for verification / no-approval runs)."""

    def request(
        self, recommendation: RuntimeRecommendation, *, timeout_s: float
    ) -> ApprovalRecord:
        _ = timeout_s
        return _record(recommendation, ApprovalOutcome.REJECTED, decided_by="auto-reject")


class AutoApproveApprovalProvider(ApprovalProvider):
    """Approves (used only by explicit acceptance approval mode)."""

    def request(
        self, recommendation: RuntimeRecommendation, *, timeout_s: float
    ) -> ApprovalRecord:
        _ = timeout_s
        return _record(recommendation, ApprovalOutcome.APPROVED, decided_by="auto-approve")


class TimeoutApprovalProvider(ApprovalProvider):
    """Simulates no response within the timeout."""

    def request(
        self, recommendation: RuntimeRecommendation, *, timeout_s: float
    ) -> ApprovalRecord:
        _ = timeout_s
        return _record(
            recommendation,
            ApprovalOutcome.TIMEOUT,
            decided_by=None,
            detail="no response within the approval timeout",
        )


class ScriptedApprovalProvider(ApprovalProvider):
    """Deterministic approval outcomes for tests/verification.

    ``mismatch_proposal`` / ``mismatch_context`` deliberately return an APPROVED
    record bound to the *wrong* proposal/context so the monitor's staleness checks
    are exercised (a stale approval must never authorize a changed proposal).
    """

    def __init__(
        self,
        outcomes: list[ApprovalOutcome] | None = None,
        *,
        mismatch_proposal: bool = False,
        mismatch_context: bool = False,
        decided_by: str = "scripted",
    ) -> None:
        self._outcomes = list(outcomes or [ApprovalOutcome.TIMEOUT])
        self._index = 0
        self._mismatch_proposal = mismatch_proposal
        self._mismatch_context = mismatch_context
        self._decided_by = decided_by

    def request(
        self, recommendation: RuntimeRecommendation, *, timeout_s: float
    ) -> ApprovalRecord:
        _ = timeout_s
        if self._index < len(self._outcomes):
            outcome = self._outcomes[self._index]
        else:
            outcome = self._outcomes[-1] if self._outcomes else ApprovalOutcome.TIMEOUT
        self._index += 1
        return _record(
            recommendation,
            outcome,
            decided_by=self._decided_by if outcome is ApprovalOutcome.APPROVED else None,
            proposal_id=(
                f"stale-{recommendation.proposal_id}" if self._mismatch_proposal else None
            ),
            context_identity=(
                f"stale-context-{recommendation.context_identity[:8]}"
                if self._mismatch_context
                else None
            ),
        )


class CallbackApprovalProvider(ApprovalProvider):
    """A real, executable approval boundary backed by a caller-supplied decision.

    This is not a hardcoded "always approve" branch: the callback receives the actual
    recommendation and returns an explicit :class:`ApprovalOutcome`, so the decision
    boundary is a genuine function of the proposal (e.g. a CLI prompt, a control-plane
    callback, or an acceptance harness that inspects and records the proposal).
    """

    def __init__(
        self,
        callback: Callable[[RuntimeRecommendation], ApprovalOutcome],
        *,
        decided_by: str = "callback",
    ) -> None:
        self._callback = callback
        self._decided_by = decided_by

    def request(
        self, recommendation: RuntimeRecommendation, *, timeout_s: float
    ) -> ApprovalRecord:
        _ = timeout_s
        outcome = self._callback(recommendation)
        return _record(
            recommendation,
            outcome,
            decided_by=self._decided_by if outcome is ApprovalOutcome.APPROVED else None,
            detail=f"callback decision: {outcome.value}",
        )


class CliApprovalProvider(ApprovalProvider):
    """Interactive console approval (y/n) with a bounded wait (best effort)."""

    def __init__(self, *, assume_timeout: bool = False) -> None:
        self._assume_timeout = assume_timeout

    def request(
        self, recommendation: RuntimeRecommendation, *, timeout_s: float
    ) -> ApprovalRecord:
        if self._assume_timeout or not sys.stdin.isatty():
            return _record(
                recommendation,
                ApprovalOutcome.TIMEOUT,
                decided_by=None,
                detail="no interactive approval available",
            )
        prompt = (
            f"Approve runtime intervention {recommendation.proposal_id} "
            f"(cost ~{recommendation.expected_cost_s:.1f}s, reversible="
            f"{recommendation.reversible})? [y/N] "
        )
        try:
            answer = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return _record(
                recommendation, ApprovalOutcome.REJECTED, decided_by="cli", detail="interrupted"
            )
        if answer in ("y", "yes"):
            return _record(recommendation, ApprovalOutcome.APPROVED, decided_by="cli")
        return _record(recommendation, ApprovalOutcome.REJECTED, decided_by="cli")
