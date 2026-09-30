"""Step 2 bug: approval providers must honour their timeout instead of waiting forever."""

from __future__ import annotations

import sys
import threading
import time

import pytest

from inferenceos.contracts.enums import ApprovalOutcome
from inferenceos.contracts.runtime import RuntimeRecommendation
from inferenceos.runtime.approval import CallbackApprovalProvider, CliApprovalProvider


def _recommendation() -> RuntimeRecommendation:
    return RuntimeRecommendation(
        proposal_id="prop-1",
        opportunity_id="opp-1",
        condition="test",
        why_it_matters="x",
        analysis_summary="y",
    )


def test_slow_callback_times_out_instead_of_blocking() -> None:
    release = threading.Event()

    def slow_approve(_: RuntimeRecommendation) -> ApprovalOutcome:
        release.wait(3.0)  # an answer that arrives far too late
        return ApprovalOutcome.APPROVED

    provider = CallbackApprovalProvider(slow_approve)
    started = time.monotonic()
    record = provider.request(_recommendation(), timeout_s=0.2)
    elapsed = time.monotonic() - started
    release.set()
    assert record.outcome is ApprovalOutcome.TIMEOUT
    assert record.decided_by is None
    assert elapsed < 1.5


def test_fast_callback_still_decides() -> None:
    provider = CallbackApprovalProvider(lambda _: ApprovalOutcome.APPROVED)
    record = provider.request(_recommendation(), timeout_s=2.0)
    assert record.outcome is ApprovalOutcome.APPROVED


def test_callback_error_is_not_swallowed() -> None:
    def broken(_: RuntimeRecommendation) -> ApprovalOutcome:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        CallbackApprovalProvider(broken).request(_recommendation(), timeout_s=2.0)


class _SilentTerminal:
    """A terminal where nobody types: reading blocks until released."""

    def __init__(self, answer: str = "y\n") -> None:
        self.release = threading.Event()
        self._answer = answer

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        self.release.wait(3.0)
        return self._answer


class _TypingTerminal(_SilentTerminal):
    def readline(self) -> str:
        return self._answer


def test_cli_with_no_answer_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = _SilentTerminal()
    monkeypatch.setattr("sys.stdin", terminal)
    started = time.monotonic()
    record = CliApprovalProvider().request(_recommendation(), timeout_s=0.2)
    elapsed = time.monotonic() - started
    terminal.release.set()
    assert record.outcome is ApprovalOutcome.TIMEOUT
    assert elapsed < 1.5


def test_cli_answer_in_time_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", _TypingTerminal("y\n"))
    record = CliApprovalProvider().request(_recommendation(), timeout_s=2.0)
    assert record.outcome is ApprovalOutcome.APPROVED
    monkeypatch.setattr("sys.stdin", _TypingTerminal("n\n"))
    record = CliApprovalProvider().request(_recommendation(), timeout_s=2.0)
    assert record.outcome is ApprovalOutcome.REJECTED


def test_cli_real_terminal_times_out_then_reads_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform == "win32":
        pytest.skip("uses a POSIX pseudo-terminal")
    import os
    import pty

    main_fd, child_fd = pty.openpty()
    terminal = os.fdopen(child_fd, "r")
    try:
        monkeypatch.setattr("sys.stdin", terminal)
        started = time.monotonic()
        record = CliApprovalProvider().request(_recommendation(), timeout_s=0.2)
        assert record.outcome is ApprovalOutcome.TIMEOUT
        assert time.monotonic() - started < 1.5
        os.write(main_fd, b"y\n")
        record = CliApprovalProvider().request(_recommendation(), timeout_s=2.0)
        assert record.outcome is ApprovalOutcome.APPROVED
    finally:
        terminal.close()
        os.close(main_fd)
