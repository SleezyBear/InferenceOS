"""Phase 9 audit/report generation and CLI wiring tests."""

from __future__ import annotations

from inferenceos.cli import build_parser
from inferenceos.contracts.enums import LoopState, LoopTerminationReason
from inferenceos.loop.report import render_run, render_termination
from tests.loop_helpers import make_loop, no_improvement_responder, patched_paging_report


def test_run_report_is_machine_readable_and_consistent() -> None:
    report = patched_paging_report()
    stack = make_loop(responder=no_improvement_responder, report=report)
    result = stack.loop.run(report, report_path="<test-report>")

    # Idempotent/consistent terminal reporting.
    first = result.to_json()
    second = result.to_json()
    assert first == second
    assert result.termination.reason in {
        LoopTerminationReason.ALL_CANDIDATES_REJECTED,
        LoopTerminationReason.OPTIMIZATION_BUDGET_EXHAUSTED,
        LoopTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED,
        LoopTerminationReason.COMPLETED,
    }

    text = render_run(result)
    assert "Loop termination" in text
    assert result.termination.reason.value in text
    assert render_termination(result.run) == render_termination(result.run)


def test_state_transitions_are_auditable() -> None:
    report = patched_paging_report()
    stack = make_loop(responder=no_improvement_responder, report=report)
    result = stack.loop.run(report, report_path="<test-states>")
    stages = {event.stage for event in result.run.state_events}
    assert LoopState.INITIALIZED in stages
    assert LoopState.DIAGNOSING in stages
    assert LoopState.GENERATING in stages
    assert LoopState.KNOWLEDGE_QUERY in stages
    assert LoopState.SELECTING in stages
    assert LoopState.CHECKPOINTING in stages
    assert LoopState.EXPERIMENTING in stages
    assert LoopState.DECIDING in stages
    assert LoopState.PERSISTING in stages
    assert LoopState.TERMINATED in stages


def test_cli_registers_optimize_command() -> None:
    parser = build_parser()
    optimize = parser.parse_args(["optimize"])
    assert optimize.func.__name__ == "_cmd_optimize"
    assert optimize.workload == "short-short"
