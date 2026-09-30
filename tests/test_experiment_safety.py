"""Safety limits, restoration, and no side-effects / no recursion."""

from __future__ import annotations

from inferenceos.contracts.enums import ExperimentDecisionKind
from inferenceos.contracts.experiment import ExperimentRequest, ExperimentResult, RuntimeConfiguration
from inferenceos.experiment.engine import run_experiment
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.optimization.actions import build_default_registry
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from tests.optimization_helpers import registry_with_requantize
from tests.experiment_helpers import (
    SampleResult,
    ScriptedHarness,
    build_test_policy,
    make_request,
)


def _ok(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    if options.use_mmap:
        return SampleResult(rss_bytes=900_000_000)
    return SampleResult(rss_bytes=1_300_000_000)


def _run(
    request: ExperimentRequest,
    harness: ScriptedHarness,
    action_id: str = "runtime.enable_mmap",
) -> ExperimentResult:
    return run_experiment(
        request,
        executor=RuntimeConfigurationExecutor(),
        profiler=harness,
        workload=get_workload("short-short"),
        action=build_default_registry().get(action_id),
    )


def test_resource_limit_bounds_total_runs() -> None:
    harness = ScriptedHarness(_ok)
    policy = build_test_policy(
        warmup_repetitions=1, measured_repetitions=5, max_repetitions=5, max_total_runs=6
    )
    result = _run(make_request(policy=policy), harness)
    assert len(harness.calls) <= 6
    assert result.baseline_measurement is not None
    assert result.candidate_measurement is not None


def test_time_limit_fails_safely_and_restores() -> None:
    harness = ScriptedHarness(_ok)
    policy = build_test_policy(time_limit_s=1e-9)
    result = _run(make_request(policy=policy), harness)
    assert result.status is ExperimentDecisionKind.MEASUREMENT_FAILURE
    assert result.environment_restored is True
    assert len(harness.calls) == 0


def test_baseline_configuration_preserved_after_experiment() -> None:
    baseline = RuntimeConfiguration(
        context_length=2048, gpu_offload_layers=99, use_mmap=False, threads=None
    )
    request = make_request(baseline=baseline)
    executor = RuntimeConfigurationExecutor()
    run_experiment(
        request,
        executor=executor,
        profiler=ScriptedHarness(_ok),
        workload=get_workload("short-short"),
        action=build_default_registry().get("runtime.enable_mmap"),
    )
    assert executor.current() == baseline


def test_elevated_and_system_actions_have_no_runtime_surface() -> None:
    executor = RuntimeConfigurationExecutor()
    registry = registry_with_requantize()
    for action_id in ("system.increase_pagefile", "external.suspend_process", "model.requantize"):
        action = registry.get(action_id)
        assert executor.supports(action_id) is False
        result = executor.apply(action, {"x": 1})
        assert result.applied is False


def test_engine_does_not_invoke_the_candidate_generator() -> None:
    import inferenceos.experiment.engine as engine_module

    assert not hasattr(engine_module, "generate_candidates")
    assert not hasattr(engine_module, "build_optimization_context")


def test_no_restart_or_persistent_deployment() -> None:
    result = _run(make_request(), ScriptedHarness(_ok))
    assert result.executed is True
    assert result.environment_restored is True
    assert result.restoration is not None
    assert result.restoration.status.value == "restored"
    # The experiment artifact is in-memory; no deployment is persisted by Phase 7.
    assert result.events
