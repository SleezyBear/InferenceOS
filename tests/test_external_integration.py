"""Phase 11 integration with the Phase 10 runtime monitor (no duplicated monitoring)."""

from __future__ import annotations

from inferenceos.contracts.enums import ApprovalOutcome, RuntimeState
from inferenceos.contracts.external import ExternalBudget, ExternalPolicy
from inferenceos.contracts.runtime import RuntimeBudget
from inferenceos.external.engine import ExternalRecommendationEngine
from inferenceos.external.integration import RuntimeExternalAdvisor
from inferenceos.optimization.actions import build_default_registry
from inferenceos.runtime.approval import AutoRejectApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.selftest import sample_hardware
from tests.experiment_helpers import ScriptedHarness, build_test_policy
from tests.external_helpers import (
    ScriptedProcessObserver,
    StubExternalActionExecutor,
    make_process_observation,
    make_report,
)
from tests.external_helpers import FakeClock
from tests.runtime_helpers import (
    ScriptedSampler,
    ScriptedWorkload,
    baseline_and_pressured,
    make_runtime_policy,
)


def _responder(task, options, call):  # type: ignore[no-untyped-def]
    from tests.experiment_helpers import SampleResult

    return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def _cpu_external_report():  # type: ignore[no-untyped-def]
    target = make_process_observation(pid=100, name="llama.exe", rss_bytes=1_300_000_000, cpu_percent=20.0, is_target=True)
    external = make_process_observation(pid=200, name="python.exe", rss_bytes=512 * 1024 * 1024, cpu_percent=180.0)
    return make_report(processes=[target, external], target_pid=100, memory_pressure=0.5)


def test_runtime_monitor_consults_external_advisor_without_bypassing_safety() -> None:
    clock = FakeClock()
    policy = ExternalPolicy.model_validate(
        {
            "budget": ExternalBudget(max_duration_s=300.0, max_recommendations=4, max_actions=1),
            "observation_interval_s": 1.0,
        }
    )
    engine = ExternalRecommendationEngine(
        observer=ScriptedProcessObserver(_cpu_external_report()),
        registry=build_default_registry(),
        policy=policy,
        approval_provider=AutoRejectApprovalProvider(),
        executor=StubExternalActionExecutor(),
        clock=clock,
        sleep=clock.sleep,
    )
    advisor = RuntimeExternalAdvisor(engine)

    monitor = RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=ScriptedSampler(baseline_and_pressured()),
        harness=ScriptedHarness(_responder),
        policy=make_runtime_policy(),
        approval_provider=AutoRejectApprovalProvider(),
        hardware=sample_hardware(),
        experiment_policy=build_test_policy(),
        clock=clock,
        sleep=clock.sleep,
        external_advisor=advisor,
    )
    result = monitor.run()
    run = result.run
    # External attribution/recommendation was consulted on the persistent condition.
    assert run.external_runs
    assert any(rec for ext in run.external_runs for rec in ext.recommendations)
    # No external action executed (rejected) and no internal intervention bypassed.
    assert run.interventions == []
    assert all(not ext.action_results for ext in run.external_runs)
    assert all(ext.termination.clean for ext in run.external_runs)
    assert result.final_state in (RuntimeState.TERMINAL_SAFE, RuntimeState.ERROR_SAFE)


def test_external_advisor_rejection_does_not_stop_monitoring() -> None:
    clock = FakeClock()
    policy = ExternalPolicy.model_validate(
        {"budget": ExternalBudget(max_duration_s=60.0), "observation_interval_s": 1.0}
    )
    engine = ExternalRecommendationEngine(
        observer=ScriptedProcessObserver(_cpu_external_report()),
        registry=build_default_registry(),
        policy=policy,
        approval_provider=AutoRejectApprovalProvider(),
        clock=clock,
        sleep=clock.sleep,
    )
    advisor = RuntimeExternalAdvisor(engine)
    monitor = RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=ScriptedSampler(baseline_and_pressured()),
        harness=ScriptedHarness(_responder),
        policy=make_runtime_policy(
            budget=RuntimeBudget(
                max_duration_s=6.0,
                telemetry_interval_s=0.5,
                observation_window_size=40,
                baseline_samples=4,
            )
        ),
        approval_provider=AutoRejectApprovalProvider(),
        hardware=sample_hardware(),
        external_advisor=advisor,
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    # The runtime run still terminates cleanly and the workload stays running.
    assert result.run.termination.clean is True
    assert result.run.external_runs
    assert all(a.outcome is not ApprovalOutcome.APPROVED for ext in result.run.external_runs for a in ext.approvals)
