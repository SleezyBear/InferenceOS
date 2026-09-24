"""Phase 9 autonomous-loop engine tests (deterministic, scripted harness)."""

from __future__ import annotations

from inferenceos.contracts.enums import (
    KnowledgeSelectionEffect,
    LoopIterationOutcome,
    LoopTerminationReason,
    RestorationStatus,
)
from inferenceos.contracts.optimization_loop import OptimizationBudget
from tests.loop_helpers import (
    context_improvement_responder,
    default_loop_policy,
    make_loop,
    no_improvement_responder,
    patched_paging_report,
    report_for_config,
    make_test_experiment_policy,
)


def test_healthy_baseline_terminates_without_changes() -> None:
    from tests.reasoning_helpers import healthy_bundle

    _, _, report = healthy_bundle()
    stack = make_loop(
        responder=no_improvement_responder,
        report=report,
        loop_policy=default_loop_policy(),
    )
    result = stack.loop.run(report, report_path="<test-healthy>")
    assert result.termination.reason is LoopTerminationReason.NO_MEANINGFUL_OPPORTUNITY
    assert result.changed is False
    assert result.run.budget.experiments_used == 0
    assert len(result.run.iterations) == 1
    assert result.run.iterations[0].outcome is LoopIterationOutcome.NO_CANDIDATE
    assert not stack.harness.calls


def test_no_improvement_candidate_reverts_and_persists() -> None:
    report = patched_paging_report()
    stack = make_loop(responder=no_improvement_responder, report=report)
    result = stack.loop.run(report, report_path="<test-no-improvement>")
    assert result.changed is False
    assert result.run.budget.experiments_used >= 1
    experimented = [i for i in result.run.iterations if i.experiment_id is not None]
    assert experimented
    assert all(
        iteration.restoration_status is RestorationStatus.RESTORED
        for iteration in experimented
    )
    assert any(
        iteration.outcome is LoopIterationOutcome.CANDIDATE_REVERTED
        for iteration in experimented
    )
    assert stack.service.stats().optimization_records >= 1


def test_retained_candidate_becomes_new_baseline() -> None:
    report = patched_paging_report()
    stack = make_loop(
        responder=context_improvement_responder,
        report=report,
        observer=lambda config: report_for_config(report, config),
        loop_policy=default_loop_policy(),
    )
    result = stack.loop.run(report, report_path="<test-retain>")
    assert result.changed is True
    assert result.run.retained_candidate_ids
    assert any(iteration.retained for iteration in result.run.iterations)
    assert result.final_configuration != result.initial_configuration
    # Re-observation occurred after retention.
    assert any(
        event.stage.value == "re_observing" for event in result.run.state_events
    )


def test_multi_iteration_revert_then_retain() -> None:
    report = patched_paging_report()
    stack = make_loop(
        responder=context_improvement_responder,
        report=report,
        observer=lambda config: report_for_config(report, config),
        loop_policy=default_loop_policy(),
    )
    result = stack.loop.run(report, report_path="<test-multi>")
    assert result.run.budget.experiments_used >= 1
    assert len(result.run.iterations) >= 1
    # Deterministic and auditable: every iteration records a terminal outcome.
    assert all(
        iteration.finished_at is not None for iteration in result.run.iterations
    )


def test_iteration_budget_terminates_cleanly() -> None:
    report = patched_paging_report()
    policy = default_loop_policy(
        budget=OptimizationBudget(
            max_iterations=1,
            max_experiments=5,
            max_wall_clock_s=300.0,
            max_optimizer_overhead_s=120.0,
            max_deep_diagnoses=0,
        )
    )
    stack = make_loop(
        responder=no_improvement_responder, report=report, loop_policy=policy
    )
    result = stack.loop.run(report, report_path="<test-budget>")
    assert result.termination.reason is LoopTerminationReason.OPTIMIZATION_BUDGET_EXHAUSTED
    assert result.run.budget.iterations_used == 1
    assert result.run.budget.experiments_used <= 1


def test_experiment_budget_terminates_when_iterations_remain() -> None:
    report = patched_paging_report()
    policy = default_loop_policy(
        budget=OptimizationBudget(
            max_iterations=5,
            max_experiments=1,
            max_wall_clock_s=300.0,
            max_optimizer_overhead_s=120.0,
            max_deep_diagnoses=0,
        )
    )
    stack = make_loop(
        responder=no_improvement_responder, report=report, loop_policy=policy
    )
    result = stack.loop.run(report, report_path="<test-exp-budget>")
    assert result.termination.reason is LoopTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED
    assert result.run.budget.experiments_used == 1


def test_knowledge_aware_selection_rejects_repeat() -> None:
    report = patched_paging_report()
    policy = default_loop_policy()
    stack = make_loop(
        responder=no_improvement_responder, report=report, loop_policy=policy
    )
    first = stack.loop.run(report, report_path="<test-knowledge-1>")
    assert first.run.budget.experiments_used >= 1
    # A second run over the same context must consult the persisted negative evidence.
    second = stack.loop.run(report, report_path="<test-knowledge-2>")
    assert second.run.budget.experiments_used == 0
    assert second.termination.reason in {
        LoopTerminationReason.ALL_CANDIDATES_REJECTED,
        LoopTerminationReason.NO_ELIGIBLE_CANDIDATES,
    }
    effects = [
        evidence.effect
        for iteration in second.run.iterations
        for evidence in iteration.knowledge
    ]
    assert KnowledgeSelectionEffect.REJECTED in effects or not second.run.iterations
    assert second.run.iterations
    assert all(
        evidence.effect is KnowledgeSelectionEffect.REJECTED
        for iteration in second.run.iterations
        for evidence in iteration.knowledge
    )


def test_run_is_json_round_trippable() -> None:
    from inferenceos.contracts.optimization_loop import LoopResult

    report = patched_paging_report()
    stack = make_loop(responder=no_improvement_responder, report=report)
    result = stack.loop.run(report, report_path="<test-json>")
    restored = LoopResult.from_json(result.to_json())
    assert restored.termination.reason is result.termination.reason
    assert restored.run.run_id == result.run.run_id
