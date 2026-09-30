"""Phase 10 runtime safety, knowledge, reporting and adversarial-containment tests."""

from __future__ import annotations

from inferenceos.cli import _adversarial_proposal_payload
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import ApprovalOutcome, OptimizationOutcome, RuntimeState
from inferenceos.contracts.knowledge import OptimizationRecord
from inferenceos.contracts.profiling import ProfilingConfig
from inferenceos.contracts.knowledge import ApplicabilityContext
from inferenceos.contracts.runtime import RuntimeBudget, RuntimePolicy
from inferenceos.experiment.measurement import MeasurementHarness
from inferenceos.knowledge import (
    KnowledgePolicy,
    KnowledgeService,
    SQLiteKnowledgeStore,
    build_current_context,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from inferenceos.runtime.approval import ApprovalProvider, ScriptedApprovalProvider
from inferenceos.runtime.knowledge import query_knowledge
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.report import render_monitor_run
from inferenceos.runtime.telemetry import RuntimeTelemetrySampler
from inferenceos.selftest import sample_hardware, sample_model, sample_workload
from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy
from tests.knowledge_helpers import temp_db_path
from tests.optimization_helpers import make_candidate, synthetic_context
from tests.runtime_helpers import (
    ScriptedSampler,
    ScriptedWorkload,
    baseline_and_pressured,
    make_runtime_policy,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


def _improving_responder(task, options, call):  # type: ignore[no-untyped-def]
    if options.use_mmap:
        return SampleResult(rss_bytes=900_000_000, hard_faults_per_sec=1.0, decode_tps=28.0)
    return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def _budget(duration: float = 12.0) -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=duration,
        telemetry_interval_s=0.5,
        observation_window_size=40,
        baseline_samples=4,
        max_deep_analyses=2,
        max_interventions=1,
        max_cumulative_intervention_s=600.0,
        max_optimizer_overhead_s=120.0,
        approval_timeout_s=10.0,
        max_alerts_per_condition=2,
    )


def _monitor(
    workload: ScriptedWorkload,
    policy: RuntimePolicy,
    approvals: ApprovalProvider,
    harness: MeasurementHarness,
    clock: FakeClock,
    *,
    sampler: RuntimeTelemetrySampler | None = None,
    knowledge_service: KnowledgeService | None = None,
) -> RuntimeMonitor:
    return RuntimeMonitor(
        workload=workload,
        sampler=sampler or ScriptedSampler(baseline_and_pressured()),
        harness=harness,
        policy=policy,
        approval_provider=approvals,
        hardware=sample_hardware(),
        experiment_policy=build_test_policy(),
        clock=clock,
        sleep=clock.sleep,
        knowledge_service=knowledge_service,
    )


def test_checkpoint_failure_does_not_interrupt_workload() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload(checkpoint_supported=False)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED])
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget()), approvals,
        ScriptedHarness(_improving_responder), clock,
    )
    result = monitor.run()
    assert workload.pause_calls == 0  # never paused without a verified checkpoint
    assert result.run.interventions
    assert result.run.checkpoints and result.run.checkpoints[0].verified is False
    assert result.run.cooldowns


def test_pause_failure_enters_error_safe() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload(pause_supported=False)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED])
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget()), approvals,
        ScriptedHarness(_improving_responder), clock,
    )
    result = monitor.run()
    assert result.final_state is RuntimeState.ERROR_SAFE


def test_intervention_budget_limit_terminates() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload()
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED])
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget(30.0)), approvals,
        ScriptedHarness(_improving_responder), clock,
    )
    result = monitor.run()
    assert result.run.budget.interventions_used == 1
    assert result.run.budget.max_interventions == 1


def _service_with_directly_applicable_negative() -> tuple[KnowledgeService, ApplicabilityContext]:
    from inferenceos.contracts.experiment import RuntimeConfiguration

    service = KnowledgeService(
        store=SQLiteKnowledgeStore(temp_db_path()),
        registry=build_default_registry(),
        policy=KnowledgePolicy(),
    )
    context = build_current_context(
        hardware=sample_hardware(),
        model=sample_model(),
        workload=sample_workload(),
        backend=ProfilingConfig(backend_name="llama.cpp", context_length=2048),
        runtime=RuntimeConfiguration(context_length=2048, gpu_offload_layers=99),
        diagnosis="paging",
        conditions={},
    )
    service.store.record_optimization(
        OptimizationRecord(
            record_id="opt-runtime-1",
            experiment_id="exp-runtime-1",
            candidate_id="cand-1",
            action_id="runtime.right_size_context",
            parameters={"context_length": 1024},
            diagnosed_bottleneck="paging",
            outcome=OptimizationOutcome.NO_MEANINGFUL_IMPROVEMENT,
            context=context,
        )
    )
    return service, context


def test_knowledge_is_consulted_and_can_reject_candidate() -> None:
    service, context = _service_with_directly_applicable_negative()
    candidate = make_candidate(
        synthetic_context(BottleneckKind.PAGING), action_id="runtime.right_size_context"
    )
    evidence = query_knowledge(
        candidate, service=service, current_context=context, policy=make_runtime_policy()
    )
    assert evidence.records_considered >= 1
    assert evidence.effect.value == "rejected"


def test_monitor_records_knowledge_queries_when_configured() -> None:
    clock = FakeClock()
    service, _ = _service_with_directly_applicable_negative()
    workload = ScriptedWorkload()
    approvals = ScriptedApprovalProvider([ApprovalOutcome.REJECTED])
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget()), approvals,
        ScriptedHarness(_improving_responder), clock, knowledge_service=service,
    )
    result = monitor.run()
    assert result.run.knowledge_queries  # consulted and recorded


def test_adversarial_reasoning_cannot_create_executable_runtime_action() -> None:
    ctx = synthetic_context(BottleneckKind.PAGING)
    provider = FakeReasoningProvider(scripted=[_adversarial_proposal_payload()])
    plan = generate_candidates(
        ctx.diagnosis, report=None, policy=OptimizationPolicy(),
        provider=provider, enable_reasoning=True,
    )
    assert not any("shell" in c.action_id for c in plan.candidates)
    assert not any(c.executable for c in plan.candidates)


def test_monitor_report_is_structured() -> None:
    clock = FakeClock()
    workload = ScriptedWorkload(checkpoint_supported=False)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED])
    monitor = _monitor(
        workload, make_runtime_policy(budget=_budget()), approvals,
        ScriptedHarness(_improving_responder), clock,
    )
    result = monitor.run()
    text = render_monitor_run(result)
    for expected in (
        "Phase 10 runtime monitor",
        "Runtime state transitions",
        "overhead",
        "resource",
        "summary",
    ):
        assert expected in text
