"""``runtime.right_size_context`` never produces a candidate.

The optimizer must neither change nor suggest context changes: no executor, loop,
generator, or runtime monitor may surface a context candidate at all.
"""

from __future__ import annotations

from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import ApprovalOutcome
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.runtime import RuntimeBudget
from inferenceos.experiment.executor import (
    RuntimeConfigurationExecutor,
    supported_actions,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.context import build_optimization_context
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.strategies import prepare
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.runtime.approval import ScriptedApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.verification import ScriptedWorkload
from inferenceos.runtime.workload import RuntimeWorkloadExecutor
from inferenceos.selftest import sample_hardware
from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy
from tests.loop_helpers import make_loop, no_improvement_responder, patched_paging_report
from tests.optimization_helpers import synthetic_diagnosis
from tests.reasoning_helpers import paging_bundle
from tests.runtime_helpers import (
    ScriptedSampler,
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


def _small_budget() -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=4.0,
        telemetry_interval_s=0.5,
        observation_window_size=40,
        baseline_samples=4,
        max_deep_analyses=1,
        max_interventions=2,
        max_cumulative_intervention_s=300.0,
        max_optimizer_overhead_s=120.0,
        approval_timeout_s=10.0,
        max_alerts_per_condition=2,
    )


def _flat_metrics(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    return SampleResult()


def test_supported_actions_excludes_right_size_context() -> None:
    assert "runtime.right_size_context" not in supported_actions()


def test_configuration_executor_refuses_right_size_context() -> None:
    executor = RuntimeConfigurationExecutor()
    executor.set_configuration(RuntimeConfiguration(context_length=2048))
    action = build_default_registry().get("runtime.right_size_context")
    result = executor.apply(action, {"context_length": 1024})
    assert result.applied is False
    assert result.supported is False
    assert executor.current().context_length == 2048


def test_workload_executor_refuses_right_size_context() -> None:
    workload = ScriptedWorkload()
    executor = RuntimeWorkloadExecutor(workload)
    action = build_default_registry().get("runtime.right_size_context")
    result = executor.apply(action, {"context_length": 1024})
    assert result.applied is False
    assert result.supported is False
    assert workload.configuration().context_length == 2048


def test_every_supported_action_reports_a_change() -> None:
    params: dict[str, dict[str, object]] = {
        "runtime.enable_mmap": {"use_mmap": True},
        "runtime.reduce_gpu_offload_layers": {"gpu_offload_layers": 0},
        "runtime.increase_gpu_offload_layers": {"gpu_offload_layers": 120},
        "runtime.reduce_threads": {"threads": 1},
    }
    registry = build_default_registry()
    for action_id in sorted(supported_actions()):
        workload = ScriptedWorkload()
        executor = RuntimeWorkloadExecutor(workload)
        result = executor.apply(registry.get(action_id), params[action_id])
        assert result.applied is True, action_id
        assert result.changes, action_id


def test_prepare_returns_none_for_right_size_context() -> None:
    # The action has no preparation strategy, so the generator can never build a
    # candidate for it.
    action = build_default_registry().get("runtime.right_size_context")
    ctx = build_optimization_context(
        synthetic_diagnosis(BottleneckKind.PAGING),
        report=None,
        outcome=None,
        policy=OptimizationPolicy(),
    )
    assert prepare(action, ctx, OptimizationPolicy()) is None


def test_paging_plan_never_creates_a_context_candidate() -> None:
    _, diagnosis, report = paging_bundle()
    plan = generate_candidates(diagnosis, report=report)
    assert all(
        candidate.action_id != "runtime.right_size_context"
        for candidate in plan.candidates
    )


def test_loop_never_sees_a_context_candidate() -> None:
    report = patched_paging_report()
    stack = make_loop(responder=no_improvement_responder, report=report)
    result = stack.loop.run(report, report_path="<test-suggestion-only>")
    assert result.run.iterations
    assert all(
        iteration.selected_action_id != "runtime.right_size_context"
        for iteration in result.run.iterations
    )
    # No iteration plan contains a context candidate at all.
    assert all(
        candidate.action_id != "runtime.right_size_context"
        for iteration in result.run.iterations
        if iteration.plan is not None
        for candidate in iteration.plan.candidates
    )


def test_runtime_monitor_never_applies_right_size_context() -> None:
    # With mmap unknown, context right-sizing would be the only paging candidate, but it
    # is non-executable, so the monitor must make no intervention.
    clock = _Clock()
    workload = ScriptedWorkload(
        configuration=RuntimeConfiguration(context_length=2048, gpu_offload_layers=99)
    )
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=ScriptedSampler(baseline_and_pressured()),
        harness=ScriptedHarness(_flat_metrics),
        policy=make_runtime_policy(budget=_small_budget()),
        approval_provider=ScriptedApprovalProvider(
            [ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED]
        ),
        hardware=sample_hardware(),
        experiment_policy=build_test_policy(),
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    assert result.run.interventions == []
