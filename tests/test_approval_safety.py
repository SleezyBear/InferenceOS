"""B-approval safety gate: `monitor --approval approve` requires explicit opt-in."""

from __future__ import annotations

import argparse
import sys

import pytest

from inferenceos.cli import _approval_provider_from_args, _monitor_approval_from_args
from inferenceos.contracts.enums import ApprovalOutcome
from inferenceos.contracts.runtime import RuntimeBudget, RuntimeRecommendation
from inferenceos.runtime.approval import (
    AutoApproveApprovalProvider,
    AutoRejectApprovalProvider,
    CliApprovalProvider,
    TimeoutApprovalProvider,
)
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.verification import (
    ScriptedRuntimeSampler,
    ScriptedWorkload,
    healthy_metrics,
)
from inferenceos.selftest import sample_hardware
from tests.runtime_helpers import make_runtime_policy


def _args(*, approval: str, optin: bool) -> argparse.Namespace:
    return argparse.Namespace(approval=approval, i_accept_unattended_actions=optin)


def test_approve_without_optin_falls_back_and_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider, notes = _monitor_approval_from_args(_args(approval="approve", optin=False))
    assert isinstance(provider, CliApprovalProvider)
    assert not isinstance(provider, AutoApproveApprovalProvider)
    assert notes and "NOT authorized" in notes[0]
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "--i-accept-unattended-actions" in err
    assert "--approval approve" in err


def test_approve_with_optin_enables_auto_approval_and_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider, notes = _monitor_approval_from_args(_args(approval="approve", optin=True))
    assert isinstance(provider, AutoApproveApprovalProvider)
    assert notes and "explicitly accepted" in notes[0]
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "explicitly accepted" in err


def test_non_approve_modes_are_unchanged() -> None:
    reject, reject_notes = _monitor_approval_from_args(_args(approval="reject", optin=False))
    timeout, timeout_notes = _monitor_approval_from_args(
        _args(approval="timeout", optin=False)
    )
    interactive, interactive_notes = _monitor_approval_from_args(
        _args(approval="interactive", optin=False)
    )
    assert isinstance(reject, AutoRejectApprovalProvider)
    assert isinstance(timeout, TimeoutApprovalProvider)
    assert isinstance(interactive, CliApprovalProvider)
    assert reject_notes == [] and timeout_notes == [] and interactive_notes == []


def test_shared_helper_unchanged_for_non_monitor_callers() -> None:
    # The external command / acceptance harnesses still resolve `approve` directly;
    # only the monitor command applies the B-approval gate.
    provider = _approval_provider_from_args(argparse.Namespace(approval="approve"))
    assert isinstance(provider, AutoApproveApprovalProvider)


def test_safe_fallback_does_not_approve_non_interactive() -> None:
    # The fallback provider is the interactive CLI provider, which returns TIMEOUT
    # (never approval) when no interactive terminal is present.
    if sys.stdin.isatty():  # pragma: no cover - CI/non-interactive in practice
        pytest.skip("interactive terminal present; non-interactive fail-safe not exercised")
    recommendation = RuntimeRecommendation(
        proposal_id="prop-1",
        opportunity_id="opp-1",
        condition="test",
        why_it_matters="x",
        analysis_summary="y",
    )
    record = CliApprovalProvider().request(recommendation, timeout_s=1.0)
    assert record.outcome is ApprovalOutcome.TIMEOUT


def _budget() -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=2.0,
        telemetry_interval_s=0.5,
        observation_window_size=20,
        baseline_samples=2,
        max_deep_analyses=1,
        max_interventions=1,
        max_cumulative_intervention_s=60.0,
        max_optimizer_overhead_s=60.0,
        approval_timeout_s=5.0,
        max_alerts_per_condition=1,
    )


def test_unattended_provenance_persisted_in_session_notes() -> None:
    note = "unattended auto-approval explicitly accepted via test"
    monitor = RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=ScriptedRuntimeSampler(healthy_metrics),
        policy=make_runtime_policy(budget=_budget()),
        approval_provider=AutoRejectApprovalProvider(),
        hardware=sample_hardware(),
        run_id="b-approval-provenance",
        session_id="b-approval-provenance",
        session_notes=[note],
    )
    result = monitor.run()
    session = result.run.session
    assert session is not None
    assert note in session.notes
