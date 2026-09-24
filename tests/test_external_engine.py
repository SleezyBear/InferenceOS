"""Phase 11 external attribution / recommendation / action tests."""

from __future__ import annotations

from inferenceos.contracts.enums import (
    ApprovalOutcome,
    ExternalActionSafetyClass,
    ExternalProcessAccess,
    ExternalRecommendationClass,
    ExternalRunTerminationReason,
)
from inferenceos.contracts.external import (
    ExternalBudget,
    ExternalPolicy,
    ExternalProcessObservation,
)
from inferenceos.external.identity import resolve_identity
from inferenceos.external.attribution import assess_external_attribution
from inferenceos.external.engine import ExternalRecommendationEngine
from inferenceos.external.knowledge import ExternalKnowledgeStore
from inferenceos.external.recommender import build_recommendations
from inferenceos.optimization.actions import build_default_registry
from inferenceos.runtime.approval import (
    AutoApproveApprovalProvider,
    AutoRejectApprovalProvider,
    ScriptedApprovalProvider,
)
from tests.external_helpers import (
    ScriptedProcessObserver,
    StubExternalActionExecutor,
    launch_sleeper,
    make_process_observation,
    make_report,
)
from tests.knowledge_helpers import temp_db_path


def _report_with_real_external(pid: int, *, cpu_percent: float = 180.0, rss_bytes: int = 512 * 1024 * 1024):  # type: ignore[no-untyped-def]
    identity = resolve_identity(pid)
    assert identity is not None
    target = make_process_observation(pid=100, name="llama.exe", rss_bytes=1_300_000_000, cpu_percent=20.0, is_target=True)
    external = ExternalProcessObservation(
        observation_id="extproc-real",
        identity=identity,
        cpu_percent=cpu_percent,
        rss_bytes=rss_bytes,
        measurable=["rss_bytes", "cpu_percent"],
        access=ExternalProcessAccess.READABLE,
    )
    return make_report(processes=[target, external], target_pid=100, memory_pressure=0.5)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


def _policy(**overrides: object) -> ExternalPolicy:
    base: dict[str, object] = {
        "budget": ExternalBudget(
            max_duration_s=300.0,
            max_recommendations=4,
            max_actions=2,
            max_processes=40,
        ),
        "observation_interval_s": 1.0,
        "memory_pressure_threshold": 0.75,
        "memory_dominance_threshold": 0.6,
        "cpu_dominance_threshold": 0.5,
        "min_process_rss_bytes": 256 * 1024 * 1024,
        "min_process_cpu_percent": 25.0,
        "require_target_degradation_for_recommendation": True,
        "require_approval_for_actions": True,
        "restore_after_action": True,
        "cooldown_s": 300.0,
    }
    base.update(overrides)
    return ExternalPolicy.model_validate(base)


def _cpu_report(*, target_degraded_ok: bool = True):  # type: ignore[no-untyped-def]
    target = make_process_observation(pid=100, name="llama.exe", rss_bytes=1_300_000_000, cpu_percent=20.0, is_target=True)
    external = make_process_observation(pid=200, name="python.exe", rss_bytes=512 * 1024 * 1024, cpu_percent=180.0)
    return make_report(processes=[target, external], target_pid=100, memory_pressure=0.5)


def _memory_report(*, memory_pressure: float = 0.9):  # type: ignore[no-untyped-def]
    target = make_process_observation(pid=100, name="llama.exe", rss_bytes=1_300_000_000, cpu_percent=20.0, is_target=True)
    external = make_process_observation(pid=300, name="memory_holder.exe", rss_bytes=6 * 1024**3, cpu_percent=2.0)
    return make_report(processes=[target, external], target_pid=100, memory_pressure=memory_pressure)


def test_model_dominating_memory_is_not_flagged() -> None:
    target = make_process_observation(pid=100, name="llama.exe", rss_bytes=3 * 1024**3, is_target=True)
    minor = make_process_observation(pid=400, name="helper.exe", rss_bytes=200 * 1024 * 1024)
    report = make_report(processes=[target, minor], target_pid=100, memory_pressure=0.85)
    assessments = assess_external_attribution(report, _policy(), target_degraded=True)
    assert all(not assessment.sufficient_for_recommendation for assessment in assessments)


def test_dominant_memory_contributor_under_pressure_is_sufficient() -> None:
    report = _memory_report()
    assessments = assess_external_attribution(report, _policy(), target_degraded=True)
    holder = next(a for a in assessments if a.identity.pid == 300)
    assert holder.sufficient_for_recommendation is True
    assert any(impact.dominant for impact in holder.impacts)
    assert holder.causal is True  # pressure + target degradation + dominant


def test_correlation_only_is_not_causal() -> None:
    report = _cpu_report()
    assessments = assess_external_attribution(report, _policy(), target_degraded=False)
    external = next(a for a in assessments if a.identity.pid == 200)
    assert external.sufficient_for_recommendation is False
    assert external.causal is False
    assert external.correlation_only is True
    recommendations = build_recommendations(
        assessments, registry=build_default_registry(), policy=_policy(), context_identity="ctx"
    )
    informational = [r for r in recommendations if r.identity and r.identity.pid == 200]
    assert informational
    assert informational[0].classification is ExternalRecommendationClass.INFORMATIONAL
    assert informational[0].proposed_action_id is None


def test_supported_cpu_recommendation_is_approval_gated() -> None:
    report = _cpu_report()
    assessments = assess_external_attribution(report, _policy(), target_degraded=True)
    recommendations = build_recommendations(
        assessments, registry=build_default_registry(), policy=_policy(), context_identity="ctx"
    )
    rec = next(r for r in recommendations if r.identity and r.identity.pid == 200)
    assert rec.classification is ExternalRecommendationClass.APPROVAL_GATED_ACTION
    assert rec.proposed_action_id == "external.reduce_priority"
    assert rec.supported is True
    assert rec.safety_class is ExternalActionSafetyClass.SAFE_REVERSIBLE
    assert rec.requires_approval is True


def test_memory_recommendation_is_user_action_only() -> None:
    report = _memory_report()
    assessments = assess_external_attribution(report, _policy(), target_degraded=True)
    recommendations = build_recommendations(
        assessments, registry=build_default_registry(), policy=_policy(), context_identity="ctx"
    )
    rec = next(r for r in recommendations if r.identity and r.identity.pid == 300)
    assert rec.classification is ExternalRecommendationClass.USER_ACTION
    assert rec.supported is False
    assert rec.proposed_action_id == "external.request_close"


def test_rejection_records_and_cooldowns_without_action() -> None:
    clock = FakeClock()
    observer = ScriptedProcessObserver(_cpu_report())
    engine = ExternalRecommendationEngine(
        observer=observer,
        registry=build_default_registry(),
        policy=_policy(),
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.REJECTED]),
        knowledge_store=ExternalKnowledgeStore(temp_db_path("ext.db")),
        executor=StubExternalActionExecutor(),
        clock=clock,
        sleep=clock.sleep,
    )
    run = engine.run(target_pid=100, target_degraded=True, context_identity="ctx", max_cycles=1)
    assert run.action_results == []
    assert any(a.outcome is ApprovalOutcome.REJECTED for a in run.approvals)
    assert run.cooldowns
    assert run.termination.reason is ExternalRunTerminationReason.APPROVAL_DECLINED
    assert run.knowledge_writes


def test_approved_action_executes_and_verifies() -> None:
    clock = FakeClock()
    with launch_sleeper() as proc:
        observer = ScriptedProcessObserver(_report_with_real_external(proc.pid))
        executor = StubExternalActionExecutor(executed=True, restore_ok=True)
        engine = ExternalRecommendationEngine(
            observer=observer,
            registry=build_default_registry(),
            policy=_policy(),
            approval_provider=AutoApproveApprovalProvider(),
            knowledge_store=ExternalKnowledgeStore(temp_db_path("ext2.db")),
            executor=executor,
            clock=clock,
            sleep=clock.sleep,
        )
        run = engine.run(target_pid=100, target_degraded=True, context_identity="ctx", max_cycles=1)
    assert run.action_results and run.action_results[0].executed
    assert run.verifications and run.verifications[0].verified
    assert run.termination.reason is ExternalRunTerminationReason.ACTION_COMPLETED
    assert run.knowledge_writes and run.knowledge_writes[-1].verified


def test_verification_failure_is_not_reported_as_success() -> None:
    clock = FakeClock()
    with launch_sleeper() as proc:
        observer = ScriptedProcessObserver(_report_with_real_external(proc.pid))
        executor = StubExternalActionExecutor(executed=False, reason="injected failure")
        engine = ExternalRecommendationEngine(
            observer=observer,
            registry=build_default_registry(),
            policy=_policy(),
            approval_provider=AutoApproveApprovalProvider(),
            executor=executor,
            clock=clock,
            sleep=clock.sleep,
        )
        run = engine.run(target_pid=100, target_degraded=True, context_identity="ctx", max_cycles=1)
    assert run.verifications
    assert run.verifications[0].verified is False
    assert run.termination.reason is ExternalRunTerminationReason.VERIFICATION_FAILURE
    assert run.termination.safety is True


def test_no_recommendation_when_nothing_dominant() -> None:
    clock = FakeClock()
    target = make_process_observation(pid=100, name="llama.exe", rss_bytes=3 * 1024**3, is_target=True)
    minor = make_process_observation(pid=400, name="helper.exe", rss_bytes=64 * 1024 * 1024)
    observer = ScriptedProcessObserver(make_report(processes=[target, minor], target_pid=100, memory_pressure=0.4))
    engine = ExternalRecommendationEngine(
        observer=observer,
        registry=build_default_registry(),
        policy=_policy(),
        approval_provider=AutoRejectApprovalProvider(),
        clock=clock,
        sleep=clock.sleep,
    )
    run = engine.run(target_pid=100, target_degraded=False, context_identity="ctx", max_cycles=1)
    assert run.action_results == []
    assert run.termination.reason is ExternalRunTerminationReason.NO_RECOMMENDATION
    assert run.termination.clean is True
