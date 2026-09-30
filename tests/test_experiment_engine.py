"""Experiment engine: improvement, no-op, regression, quality, failures, restoration."""

from __future__ import annotations

from inferenceos.contracts.enums import ExperimentDecisionKind, RestorationStatus
from inferenceos.contracts.experiment import (
    ApplicationResult,
    ExperimentRequest,
    ExperimentResult,
    RestorationResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.optimization import CandidateAction
from inferenceos.experiment.engine import run_experiment
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.optimization.actions import build_default_registry
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from tests.experiment_helpers import (
    Responder,
    SampleResult,
    ScriptedHarness,
    make_request,
    build_test_policy,
)


def _run(
    request: ExperimentRequest,
    responder: Responder,
    *,
    executor: RuntimeConfigurationExecutor | None = None,
    action_id: str = "runtime.enable_mmap",
    revalidation_ok: bool = True,
    revalidation_kind: ExperimentDecisionKind | None = None,
    revalidation_reason: str = "ok",
) -> ExperimentResult:
    registry = build_default_registry()
    return run_experiment(
        request,
        executor=executor or RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(responder),
        workload=get_workload("short-short"),
        action=registry.get(action_id),
        revalidation_ok=revalidation_ok,
        revalidation_kind=revalidation_kind,
        revalidation_reason=revalidation_reason,
    )


def _improve(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    if options.use_mmap:
        return SampleResult(rss_bytes=900_000_000, hard_faults_per_sec=1.0)
    return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0)


def _no_change(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0)


def _objective_regression(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    if options.use_mmap:
        return SampleResult(rss_bytes=1_200_000_000, decode_tps=8.0)
    return SampleResult(rss_bytes=1_300_000_000, decode_tps=20.0)


def _quality_regression(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    if options.use_mmap:
        return SampleResult(rss_bytes=900_000_000, text="zzzz qqqq")
    return SampleResult(rss_bytes=1_300_000_000)


def _candidate_fails(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    if options.use_mmap:
        return SampleResult(fail=True)
    return SampleResult()


def test_meaningful_improvement_is_kept_and_restored() -> None:
    result = _run(make_request(), _improve)
    assert result.status is ExperimentDecisionKind.IMPROVED
    assert result.decision_detail is not None and result.decision_detail.keep
    assert result.executed is True
    assert result.environment_restored is True
    assert result.restoration is not None
    assert result.restoration.status is RestorationStatus.RESTORED
    assert result.comparison is not None and result.comparison.bottleneck_improved
    assert result.baseline_measurement is not None
    assert result.candidate_measurement is not None


def test_no_meaningful_improvement() -> None:
    result = _run(make_request(), _no_change)
    assert result.status is ExperimentDecisionKind.NO_MEANINGFUL_IMPROVEMENT
    assert result.decision_detail is not None and not result.decision_detail.keep
    assert result.environment_restored is True


def test_objective_regression_is_rejected() -> None:
    result = _run(make_request(), _objective_regression)
    assert result.status is ExperimentDecisionKind.REGRESSION
    assert result.decision_detail is not None and result.decision_detail.regression
    assert result.environment_restored is True


def test_quality_regression_blocks_success() -> None:
    result = _run(make_request(), _quality_regression)
    assert result.status is ExperimentDecisionKind.QUALITY_REGRESSION
    assert result.decision_detail is not None and result.decision_detail.quality_regression
    assert result.environment_restored is True


def test_measurement_failure_restores() -> None:
    result = _run(make_request(), _candidate_fails)
    assert result.status is ExperimentDecisionKind.MEASUREMENT_FAILURE
    assert result.environment_restored is True
    assert result.restoration is not None and result.restoration.verified


def test_application_failure_is_reported_and_restored() -> None:
    result = _run(
        make_request(), _improve, action_id="system.increase_pagefile"
    )
    assert result.status is ExperimentDecisionKind.APPLICATION_FAILURE
    assert result.executed is False
    assert result.environment_restored is True


def test_restoration_failure_dominates() -> None:
    class FailingRestore(RuntimeConfigurationExecutor):
        def restore(self, expected: RuntimeConfiguration) -> RestorationResult:
            return RestorationResult(
                attempted=True,
                status=RestorationStatus.FAILED,
                expected_configuration=expected,
                restored_configuration=self.current(),
                verified=False,
                reason="forced restoration failure",
            )

    result = _run(make_request(), _improve, executor=FailingRestore())
    assert result.status is ExperimentDecisionKind.RESTORATION_FAILURE
    assert result.environment_restored is False
    assert result.restoration is not None and not result.restoration.verified


def test_unexpected_exception_still_restores() -> None:
    class ExplodingApply(RuntimeConfigurationExecutor):
        def apply(
            self, action: CandidateAction, parameters: dict[str, object]
        ) -> ApplicationResult:
            self.set_configuration(
                self.current().model_copy(update={"context_length": 1024})
            )
            raise RuntimeError("boom during apply")

    result = _run(make_request(), _improve, executor=ExplodingApply())
    assert result.status is ExperimentDecisionKind.APPLICATION_FAILURE
    assert result.environment_restored is True
    assert any("boom during apply" in error for error in result.errors)


def test_precondition_failure_does_not_execute() -> None:
    result = _run(
        make_request(),
        _improve,
        revalidation_ok=False,
        revalidation_kind=ExperimentDecisionKind.UNSAFE,
        revalidation_reason="approval required",
    )
    assert result.status is ExperimentDecisionKind.UNSAFE
    assert result.executed is False
    assert result.baseline_measurement is None
    assert result.environment_restored is True


def test_repetition_policy_bounds_runs_and_measures_overhead() -> None:
    harness = ScriptedHarness(_improve)
    policy = build_test_policy(measured_repetitions=5, max_repetitions=5, min_valid_samples=2)
    request = make_request(policy=policy)
    result = run_experiment(
        request,
        executor=RuntimeConfigurationExecutor(),
        profiler=harness,
        workload=get_workload("short-short"),
        action=build_default_registry().get("runtime.enable_mmap"),
    )
    # warm-up + early stop keeps the run count bounded well below 2*5.
    assert len(harness.calls) <= 6
    assert result.overhead is not None
    assert result.overhead.total_s > 0.0
    assert result.overhead.measurement_s >= 0.0
