"""Approval abstraction for disruptive active-runtime intervention.

Approval is always explicit and bound to a specific proposal + context. A future
desktop/UI notification system can replace these implementations; the runtime engine
only depends on the abstract interface.
"""

from __future__ import annotations

import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone

from inferenceos.contracts.enums import ApprovalOutcome
from inferenceos.contracts.runtime import ApprovalRecord, RuntimeRecommendation

_TIMED_OUT_DETAIL = "no response within the approval timeout"


class _TimedOut(Exception):
    """Internal: the wait for an answer ran past its deadline."""


def _call_with_timeout(func: Callable[[], object], timeout_s: float) -> object:
    """Run a possibly-blocking ``func`` and wait at most ``timeout_s`` for it.

    The call runs on a daemon thread, so a late answer is simply discarded and can
    never approve anything. An exception raised by ``func`` in time is re-raised.
    """
    box: list[object] = []
    failure: list[BaseException] = []

    def _run() -> None:
        try:
            box.append(func())
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            failure.append(exc)

    worker = threading.Thread(target=_run, name="approval-wait", daemon=True)
    worker.start()
    worker.join(max(timeout_s, 0.0))
    if failure:
        raise failure[0]
    if not box:
        raise _TimedOut
    return box[0]


def _read_console_line_windows(deadline: float) -> str:  # pragma: no cover - Windows only
    """Read one typed line from the Windows console, giving up at ``deadline``."""
    import msvcrt

    chars: list[str] = []
    while time.monotonic() < deadline:
        if msvcrt.kbhit():
            char = msvcrt.getwche()
            if char in ("\r", "\n"):
                sys.stdout.write("\n")
                return "".join(chars)
            if char == "\x03":
                raise KeyboardInterrupt
            if char == "\x08":
                if chars:
                    chars.pop()
                continue
            chars.append(char)
        else:
            time.sleep(0.05)
    raise _TimedOut


def _read_line_with_timeout(timeout_s: float) -> str:
    """Read one line of user input, waiting at most ``timeout_s`` seconds.

    Raises :class:`_TimedOut` when nobody answers in time and ``EOFError`` when input
    is closed. On a real console nothing is left reading in the background after a
    timeout, so a later answer can never be picked up by the wrong prompt.
    """
    deadline = time.monotonic() + max(timeout_s, 0.0)
    stdin = sys.stdin
    try:
        fileno: int | None = stdin.fileno()
    except (AttributeError, OSError, ValueError):
        fileno = None
    if fileno is not None and sys.platform == "win32":  # pragma: no cover - Windows only
        return _read_console_line_windows(deadline)
    if fileno is not None and sys.platform != "win32":
        import select

        ready, _, _ = select.select([stdin], [], [], max(deadline - time.monotonic(), 0.0))
        if not ready:
            raise _TimedOut
        line = stdin.readline()
        if line == "":
            raise EOFError
        return line
    # Unusual input streams (no OS handle): wait on a helper thread instead.
    line = _call_with_timeout(stdin.readline, deadline - time.monotonic())
    if not isinstance(line, str) or line == "":
        raise EOFError
    return line


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
            detail=_TIMED_OUT_DETAIL,
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
        try:
            decided = _call_with_timeout(lambda: self._callback(recommendation), timeout_s)
        except _TimedOut:
            return _record(
                recommendation,
                ApprovalOutcome.TIMEOUT,
                decided_by=None,
                detail=_TIMED_OUT_DETAIL,
            )
        if not isinstance(decided, ApprovalOutcome):
            raise TypeError(f"approval callback returned {decided!r}, not an ApprovalOutcome")
        outcome = decided
        return _record(
            recommendation,
            outcome,
            decided_by=self._decided_by if outcome is ApprovalOutcome.APPROVED else None,
            detail=f"callback decision: {outcome.value}",
        )


class CliApprovalProvider(ApprovalProvider):
    """Interactive console approval (y/n) that waits at most ``timeout_s``.

    No answer in time is recorded as ``TIMEOUT``; it is never treated as approval.
    """

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
            f"(waiting up to {timeout_s:.0f}s) "
        )
        sys.stdout.write(prompt)
        sys.stdout.flush()
        try:
            answer = _read_line_with_timeout(timeout_s).strip().lower()
        except _TimedOut:
            sys.stdout.write("\nNo answer in time; the change was not approved.\n")
            sys.stdout.flush()
            return _record(
                recommendation,
                ApprovalOutcome.TIMEOUT,
                decided_by=None,
                detail=_TIMED_OUT_DETAIL,
            )
        except (EOFError, KeyboardInterrupt):
            return _record(
                recommendation, ApprovalOutcome.REJECTED, decided_by="cli", detail="interrupted"
            )
        if answer in ("y", "yes"):
            return _record(recommendation, ApprovalOutcome.APPROVED, decided_by="cli")
        return _record(recommendation, ApprovalOutcome.REJECTED, decided_by="cli")


def ask_line(prompt: str, *, timeout_s: float) -> str | None:
    """Ask one question on the console and wait at most ``timeout_s`` for the answer.

    Returns the typed line, or ``None`` when there is no interactive console, input is
    closed, or nobody answered in time. ``None`` must never be read as a yes.
    """
    if not sys.stdin.isatty():
        return None
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        return _read_line_with_timeout(timeout_s)
    except _TimedOut:
        sys.stdout.write("\nNo answer in time.\n")
        sys.stdout.flush()
        return None
    except EOFError:
        return None
