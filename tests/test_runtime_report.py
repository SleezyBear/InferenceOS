"""Phase 10 runtime state-machine / reporting / CLI wiring tests."""

from __future__ import annotations

from inferenceos.cli import build_parser
from inferenceos.contracts.enums import ApprovalOutcome, RuntimeState
from inferenceos.runtime.approval import ScriptedApprovalProvider
from inferenceos.selftest import sample_hardware
from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy
from tests.runtime_helpers import (
    ScriptedSampler,
    ScriptedWorkload,
    baseline_and_pressured,
    make_runtime_policy,
)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


def _responder(task, options, call):  # type: ignore[no-untyped-def]
    return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def test_state_transitions_are_auditable() -> None:
    from inferenceos.runtime.monitor import RuntimeMonitor

    clock = _Clock()
    monitor = RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=ScriptedSampler(baseline_and_pressured()),
        harness=ScriptedHarness(_responder),
        policy=make_runtime_policy(),
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.REJECTED]),
        hardware=sample_hardware(),
        experiment_policy=build_test_policy(),
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    states = [event.state for event in result.run.states]
    assert RuntimeState.INITIALIZING in states
    assert RuntimeState.MONITORING in states
    assert RuntimeState.OPPORTUNITY_SUSPECTED in states
    assert RuntimeState.DIAGNOSING in states
    assert RuntimeState.DEEP_ANALYSIS_RECOMMENDED in states
    assert RuntimeState.TERMINAL_SAFE in states
    # The monitor samples repeatedly, not once.
    assert result.run.observations_collected > 4


def test_cli_registers_phase10_commands() -> None:
    parser = build_parser()
    monitor = parser.parse_args(["monitor"])
    assert monitor.func.__name__ == "_cmd_monitor"
    assert monitor.workload == "short-short"
    assert monitor.approval == "interactive"
    verify = parser.parse_args(["verify-phase10"])
    assert verify.func.__name__ == "_cmd_verify_phase10"
