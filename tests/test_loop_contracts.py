"""Phase 9 loop-contract, budget, checkpoint, and configuration tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferenceos.config import Settings
from inferenceos.contracts.enums import CheckpointStatus, LoopTerminationReason
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.optimization_loop import (
    LoopPolicy,
    LoopResult,
    OptimizationBudget,
)
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.loop.checkpoint import (
    capture_checkpoint,
    restore_checkpoint,
    verify_checkpoint,
)
from inferenceos.loop.policy import build_budget, build_loop_policy
from inferenceos.selftest import sample_loop_result, sample_optimization_budget


def test_budget_exhaustion_is_deterministic() -> None:
    budget = OptimizationBudget(
        max_iterations=2,
        max_experiments=3,
        max_wall_clock_s=100.0,
        max_optimizer_overhead_s=50.0,
        max_deep_diagnoses=1,
    )
    assert budget.exhaustion(wall_clock_s=0.0, optimizer_overhead_s=0.0) is None
    assert (
        budget.exhaustion(wall_clock_s=100.0, optimizer_overhead_s=0.0)
        is LoopTerminationReason.TIME_BUDGET_EXHAUSTED
    )
    assert (
        budget.exhaustion(wall_clock_s=0.0, optimizer_overhead_s=50.0)
        is LoopTerminationReason.OVERHEAD_BUDGET_EXHAUSTED
    )
    assert (
        budget.exhaustion(wall_clock_s=0.0, optimizer_overhead_s=0.0)
        is None
    )
    used = budget.model_copy(update={"iterations_used": 2})
    assert (
        used.exhaustion(wall_clock_s=0.0, optimizer_overhead_s=0.0)
        is LoopTerminationReason.OPTIMIZATION_BUDGET_EXHAUSTED
    )
    experiments = budget.model_copy(update={"experiments_used": 3})
    assert (
        experiments.exhaustion(wall_clock_s=0.0, optimizer_overhead_s=0.0)
        is LoopTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED
    )


def test_invalid_budget_fails_before_any_mutation() -> None:
    with pytest.raises(ValidationError):
        OptimizationBudget(max_iterations=0)
    with pytest.raises(ValidationError):
        OptimizationBudget(max_wall_clock_s=0.0)
    with pytest.raises(ValidationError):
        build_loop_policy(Settings(phase9_max_iterations=0))


def test_build_loop_policy_from_settings() -> None:
    settings = Settings(
        phase9_max_iterations=2,
        phase9_max_experiments=2,
        phase9_max_deep_diagnoses=0,
        phase9_require_knowledge=True,
    )
    policy = build_loop_policy(settings)
    assert policy.budget.max_iterations == 2
    assert policy.budget.max_deep_diagnoses == 0
    assert policy.require_knowledge is True
    assert build_budget(settings).max_experiments == 2
    assert "budget" in policy.as_dict()


def test_checkpoint_capture_restore_and_verify() -> None:
    executor = RuntimeConfigurationExecutor()
    baseline = RuntimeConfiguration(context_length=2048, use_mmap=False)
    executor.set_configuration(baseline)
    checkpoint = capture_checkpoint(
        configuration=baseline,
        model_ref="sample",
        diagnosis_id="diag-1",
    )
    assert checkpoint.status is CheckpointStatus.CAPTURED
    assert verify_checkpoint(executor, checkpoint)

    executor.set_configuration(baseline.model_copy(update={"context_length": 1024}))
    assert not verify_checkpoint(executor, checkpoint)

    result = restore_checkpoint(executor, checkpoint)
    assert result.verified is True
    assert verify_checkpoint(executor, checkpoint)


def test_loop_policy_defaults_are_safe() -> None:
    policy = LoopPolicy()
    assert policy.autonomous_only is True
    assert policy.stop_on_restoration_failure is True
    assert policy.allow_revalidation is True


def test_sample_loop_result_roundtrips_and_defaults() -> None:
    result = sample_loop_result()
    restored = LoopResult.from_json(result.to_json())
    assert restored == result
    assert sample_optimization_budget().max_iterations >= 1
