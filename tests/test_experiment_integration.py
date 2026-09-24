"""Integration: a real Phase 6 candidate flows through the Phase 7 service."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inferenceos.config import Settings
from inferenceos.contracts.enums import ExperimentDecisionKind
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.experiment.service import run_candidate_experiment
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from tests.experiment_helpers import (
    SampleResult,
    ScriptedHarness,
    build_test_policy,
    paging_candidate_and_plan,
)


def _responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    if (task.context_length or 0) >= 2048:
        return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0)
    return SampleResult(rss_bytes=900_000_000, hard_faults_per_sec=1.0)


def test_real_phase6_candidate_runs_through_service() -> None:
    candidate, plan, report = paging_candidate_and_plan()
    result = run_candidate_experiment(
        candidate,
        plan=plan,
        diagnosis=plan.source_diagnosis,
        report=report,
        outcome=None,
        workload_id="short-short",
        registry=build_default_registry(),
        optimization_policy=OptimizationPolicy(),
        experiment_policy=build_test_policy(),
        executor=RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(_responder),
    )
    assert result.status is ExperimentDecisionKind.IMPROVED
    assert result.executed is True
    assert result.environment_restored is True
    assert result.request is not None
    assert result.request.baseline_configuration.context_length == 2048
    assert result.request.experiment_configuration.context_length == 1024
    assert result.application is not None and result.application.applied
    assert result.decision_detail is not None and result.decision_detail.keep


def test_approval_required_candidate_never_executes_through_service() -> None:
    _, plan, report = paging_candidate_and_plan()
    gated = next(c for c in plan.candidates if c.action_id == "system.increase_pagefile")
    result = run_candidate_experiment(
        gated,
        plan=plan,
        diagnosis=plan.source_diagnosis,
        report=report,
        outcome=None,
        workload_id="short-short",
        registry=build_default_registry(),
        optimization_policy=OptimizationPolicy(),
        experiment_policy=build_test_policy(),
        executor=RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(_responder),
    )
    assert result.executed is False
    assert result.baseline_measurement is None
    assert result.status is ExperimentDecisionKind.UNSAFE


def _real_model_available() -> bool:
    settings = Settings()
    cli = settings.llama_cpp_dir / "llama-cli.exe"
    models = list(Path(settings.models_dir).glob("*.gguf"))
    return cli.is_file() and bool(models)


@pytest.mark.skipif(
    os.environ.get("INFERENCEOS_PHASE7_REAL") != "1" or not _real_model_available(),
    reason="set INFERENCEOS_PHASE7_REAL=1 with a real llama.cpp build/model for the real path",
)
def test_real_candidate_applies_to_real_backend() -> None:
    """Optional real path; verify-phase7 owns the real acceptance gate."""
    from inferenceos.profiling.runner import BaselineProfiler

    candidate, plan, report = paging_candidate_and_plan()
    result = run_candidate_experiment(
        candidate,
        plan=plan,
        diagnosis=plan.source_diagnosis,
        report=report,
        outcome=None,
        workload_id="short-short",
        registry=build_default_registry(),
        optimization_policy=OptimizationPolicy(),
        experiment_policy=build_test_policy(measured_repetitions=1, min_valid_samples=1),
        executor=RuntimeConfigurationExecutor(),
        profiler=BaselineProfiler(),
    )
    assert result.environment_restored is True
    assert result.executed is True
