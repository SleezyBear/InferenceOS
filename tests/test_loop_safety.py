"""Phase 9 safety / failure-containment tests."""

from __future__ import annotations

from inferenceos.contracts.enums import LoopTerminationReason
from inferenceos.contracts.knowledge import (
    KnowledgeHealth,
    KnowledgeQuery,
    OptimizationRecord,
    StorageStats,
)
from inferenceos.knowledge import KnowledgePolicy, KnowledgeService
from inferenceos.knowledge.store import (
    KnowledgeUnavailableError,
    SQLiteKnowledgeStore,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from tests.experiment_helpers import SampleResult
from tests.loop_helpers import (
    FaultyExecutor,
    default_loop_policy,
    make_loop,
    no_improvement_responder,
    patched_paging_report,
)
from tests.knowledge_helpers import temp_db_path
from tests.reasoning_helpers import insufficient_bundle


class _UnavailableStore(SQLiteKnowledgeStore):
    """Store that reports itself unavailable (injected knowledge outage)."""

    def health(self) -> KnowledgeHealth:
        raise KnowledgeUnavailableError("injected knowledge outage")

    def find_records(
        self, query: KnowledgeQuery, *, newest_first: bool = True
    ) -> list[OptimizationRecord]:
        raise KnowledgeUnavailableError("injected knowledge outage")

    def all_records(self, *, limit: int = 50) -> list[OptimizationRecord]:
        raise KnowledgeUnavailableError("injected knowledge outage")

    def stats(self) -> StorageStats:
        raise KnowledgeUnavailableError("injected knowledge outage")


def _unavailable_service() -> KnowledgeService:
    return KnowledgeService(
        store=_UnavailableStore(temp_db_path("unavailable.db")),
        registry=build_default_registry(),
        policy=KnowledgePolicy(),
    )


def test_restoration_failure_is_safety_terminal() -> None:
    report = patched_paging_report()
    stack = make_loop(
        responder=no_improvement_responder,
        report=report,
        executor=FaultyExecutor(),
    )
    result = stack.loop.run(report, report_path="<test-restoration-failure>")
    assert result.termination.reason is LoopTerminationReason.RESTORATION_FAILURE
    assert result.termination.safety is True
    assert result.termination.clean is False
    # The loop must not silently continue as though the baseline were intact.
    assert result.run.budget.experiments_used == 1
    assert len(result.run.iterations) == 1


def test_quality_regression_prevents_retention() -> None:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        candidate = (task.context_length or 0) <= 1024 or bool(options.use_mmap)
        if candidate:
            return SampleResult(
                rss_bytes=900_000_000,
                hard_faults_per_sec=1.0,
                decode_tps=30.0,
                text="!!!! 1234 5678",
            )
        return SampleResult(
            rss_bytes=1_300_000_000,
            hard_faults_per_sec=5.0,
            decode_tps=20.0,
            text="A key value cache stores attention state.",
        )

    report = patched_paging_report()
    stack = make_loop(responder=responder, report=report)
    result = stack.loop.run(report, report_path="<test-quality>")
    assert result.termination.reason is LoopTerminationReason.QUALITY_SAFETY_STOP
    assert result.changed is False
    experimented = [i for i in result.run.iterations if i.experiment_id is not None]
    assert experimented
    assert all(not iteration.retained for iteration in experimented)
    assert all(iteration.restoration_verified for iteration in experimented)


def test_reasoning_required_but_unavailable_terminates() -> None:
    _, _, report = insufficient_bundle()
    policy = default_loop_policy(require_reasoning_when_escalated=True)
    stack = make_loop(
        responder=no_improvement_responder, report=report, loop_policy=policy
    )
    result = stack.loop.run(report, report_path="<test-reasoning>")
    assert result.termination.reason is LoopTerminationReason.REASONING_UNAVAILABLE
    assert result.run.budget.experiments_used == 0


def test_reasoning_failure_falls_back_safely() -> None:
    from inferenceos.loop.verification import make_insufficient_report
    from inferenceos.reasoning.engine import LocalReasoningEngine
    from inferenceos.reasoning.providers.fake import FakeReasoningProvider

    report = make_insufficient_report()
    policy = default_loop_policy()
    engine = LocalReasoningEngine(FakeReasoningProvider(fail_with="injected failure"))
    stack = make_loop(
        responder=no_improvement_responder,
        report=report,
        loop_policy=policy,
        reasoning_engine=engine,
    )
    result = stack.loop.run(report, report_path="<test-reasoning-failure>")
    # The reasoner was invoked but failed; the deterministic diagnosis is retained.
    assert any(iteration.deep_diagnosis_used for iteration in result.run.iterations)
    assert result.termination.reason in {
        LoopTerminationReason.INSUFFICIENT_EVIDENCE,
        LoopTerminationReason.NO_MEANINGFUL_OPPORTUNITY,
    }
    assert result.run.budget.experiments_used == 0


def test_deep_diagnosis_budget_terminates() -> None:
    _, _, report = insufficient_bundle()
    from inferenceos.contracts.optimization_loop import OptimizationBudget

    policy = default_loop_policy(
        require_reasoning_when_escalated=True,
        budget=OptimizationBudget(
            max_iterations=3,
            max_experiments=3,
            max_wall_clock_s=300.0,
            max_optimizer_overhead_s=120.0,
            max_deep_diagnoses=0,
        ),
    )
    stack = make_loop(
        responder=no_improvement_responder, report=report, loop_policy=policy
    )
    result = stack.loop.run(report, report_path="<test-deep-budget>")
    assert (
        result.termination.reason
        is LoopTerminationReason.DEEP_DIAGNOSIS_BUDGET_EXHAUSTED
    )
    assert result.run.budget.experiments_used == 0


def test_execution_failure_terminates() -> None:
    def failing(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        return SampleResult(fail=True)

    report = patched_paging_report()
    stack = make_loop(responder=failing, report=report)
    result = stack.loop.run(report, report_path="<test-exec-failure>")
    assert result.termination.reason is LoopTerminationReason.EXECUTION_FAILURE
    assert result.changed is False


def test_knowledge_required_but_unavailable_terminates() -> None:
    report = patched_paging_report()
    policy = default_loop_policy(require_knowledge=True)
    stack = make_loop(
        responder=no_improvement_responder,
        report=report,
        loop_policy=policy,
        knowledge_service=_unavailable_service(),
    )
    result = stack.loop.run(report, report_path="<test-knowledge-unavailable>")
    assert result.termination.reason is LoopTerminationReason.KNOWLEDGE_UNAVAILABLE
    assert result.run.budget.experiments_used == 0
    assert result.run.knowledge_available is False


def test_knowledge_optional_degrades_gracefully() -> None:
    report = patched_paging_report()
    stack = make_loop(
        responder=no_improvement_responder,
        report=report,
        knowledge_service=_unavailable_service(),
    )
    result = stack.loop.run(report, report_path="<test-knowledge-degrade>")
    # Optimization still proceeds safely without durable knowledge.
    assert result.termination.reason in {
        LoopTerminationReason.ALL_CANDIDATES_REJECTED,
        LoopTerminationReason.OPTIMIZATION_BUDGET_EXHAUSTED,
        LoopTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED,
        LoopTerminationReason.COMPLETED,
    }
    assert result.run.knowledge_available is False


def test_no_observation_runs_is_operator_required() -> None:
    from tests.reasoning_helpers import paging_report

    empty = paging_report().model_copy(update={"runs": []})
    stack = make_loop(responder=no_improvement_responder, report=empty)
    result = stack.loop.run(empty, report_path="<test-empty>")
    assert result.termination.reason is LoopTerminationReason.OPERATOR_REQUIRED
