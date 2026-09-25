"""Phase 11R focused tests: identity, diagnosis-awareness, provenance, staleness, safety."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from inferenceos.contracts.enums import (
    ApprovalOutcome,
    ExternalProcessAccess,
    ExternalRecommendationClass,
    ExternalRecommendationStatus,
    ExternalResource,
    ExternalRunTerminationReason,
)
from inferenceos.contracts.external import (
    ExternalBudget,
    ExternalPolicy,
    ExternalProcessObservation,
)
from inferenceos.external.attribution import assess_external_attribution
from inferenceos.external.engine import ExternalRecommendationEngine
from inferenceos.external.identity import build_identity, resolve_identity, verify_identity
from inferenceos.external.recommendation import (
    bottleneck_to_resource,
    revalidate_recommendation,
)
from inferenceos.external.recommender import build_recommendations
from inferenceos.optimization.actions import build_default_registry
from inferenceos.runtime.approval import (
    AutoApproveApprovalProvider,
    AutoRejectApprovalProvider,
    ScriptedApprovalProvider,
)
from tests.external_helpers import (
    FakeClock,
    ScriptedProcessObserver,
    launch_sleeper,
    make_process_observation,
    make_report,
)


def _policy(**overrides: object) -> ExternalPolicy:
    base: dict[str, object] = {
        "budget": ExternalBudget(max_duration_s=300.0, max_recommendations=4),
        "observation_interval_s": 1.0,
        "memory_pressure_threshold": 0.75,
        "memory_dominance_threshold": 0.6,
        "cpu_dominance_threshold": 0.5,
        "min_process_rss_bytes": 256 * 1024 * 1024,
        "min_process_cpu_percent": 25.0,
        "require_target_degradation_for_recommendation": True,
        "require_approval_for_actions": True,
        "cooldown_s": 300.0,
    }
    base.update(overrides)
    return ExternalPolicy.model_validate(base)


def _memory_report():  # type: ignore[no-untyped-def]
    target = make_process_observation(
        pid=100, name="llama.exe", rss_bytes=1_300_000_000, cpu_percent=20.0, is_target=True
    )
    external = make_process_observation(
        pid=300, name="memory_holder.exe", rss_bytes=6 * 1024**3, cpu_percent=2.0
    )
    return make_report(processes=[target, external], target_pid=100, memory_pressure=0.9)


def _cpu_report():  # type: ignore[no-untyped-def]
    target = make_process_observation(
        pid=100, name="llama.exe", rss_bytes=1_300_000_000, cpu_percent=20.0, is_target=True
    )
    external = make_process_observation(
        pid=200, name="python.exe", rss_bytes=512 * 1024 * 1024, cpu_percent=180.0
    )
    return make_report(processes=[target, external], target_pid=100, memory_pressure=0.5)


def _first_recommendation(report, resource):  # type: ignore[no-untyped-def]
    assessments = assess_external_attribution(report, _policy(), target_degraded=True)
    recs = build_recommendations(
        assessments,
        registry=build_default_registry(),
        policy=_policy(),
        context_identity="ctx",
        diagnosed_resource=resource,
        diagnosed_bottleneck=resource.value,
        run_id="run-A",
        session_id="session-A",
        diagnosis_id="diag-A",
        observation_report_id=report.report_id,
    )
    return next(r for r in recs if r.identity is not None and r.identity.pid == 300)


def test_process_identity_is_pid_reuse_safe() -> None:
    with launch_sleeper() as proc:
        identity = resolve_identity(proc.pid)
        assert identity is not None
        assert identity.create_time_epoch is not None
        assert identity.identity_key == f"{identity.name}#{identity.pid_reuse_guard}"
        assert verify_identity(identity) is True
        # Same PID, different creation time -> a different process (PID reuse).
        reused = identity.model_copy(
            update={
                "create_time_epoch": identity.create_time_epoch - 500.0,
                "pid_reuse_guard": "reused-guard",
            }
        )
        assert verify_identity(reused) is False
    # Terminated process identity no longer verifies.
    assert verify_identity(identity) is False


def test_identity_requires_start_time_and_executable_identity() -> None:
    identity = build_identity(
        pid=999999,
        name="ghost.exe",
        executable_path=None,
        owner=None,
        create_time_epoch=None,
        access=ExternalProcessAccess.DENIED,
    )
    assert identity.potentially_stale is True


def test_read_only_observation_does_not_mutate_process() -> None:
    from inferenceos.external.observer import WindowsProcessObserver
    import psutil

    with launch_sleeper() as proc:
        before = psutil.Process(proc.pid).nice()
        observer = WindowsProcessObserver()
        report = observer.observe(max_processes=50)
        after = psutil.Process(proc.pid).nice()
        assert before == after
        assert report.privacy_notes
        assert any("no process memory" in note for note in report.privacy_notes)
        for observation in report.processes:
            assert "cmdline" not in observation.measurable
            assert "environ" not in observation.measurable


def test_diagnosis_aware_recommendation_has_provenance() -> None:
    report = _memory_report()
    rec = _first_recommendation(report, ExternalResource.MEMORY)
    assert rec.bottleneck == "memory"
    assert rec.requires_user_approval is True
    assert rec.supported is False
    assert rec.provenance is not None
    assert rec.provenance.run_id == "run-A"
    assert rec.provenance.optimization_session_id == "session-A"
    assert rec.provenance.diagnosis_id == "diag-A"
    assert rec.provenance.observation_report_id == report.report_id
    assert rec.status is ExternalRecommendationStatus.GENERATED
    assert rec.rationale
    assert rec.confidence > 0.0


def test_irrelevant_resource_produces_no_recommendation() -> None:
    report = _cpu_report()
    assessments = assess_external_attribution(report, _policy(), target_degraded=True)
    recs = build_recommendations(
        assessments,
        registry=build_default_registry(),
        policy=_policy(),
        context_identity="ctx",
        diagnosed_resource=ExternalResource.MEMORY,
        diagnosed_bottleneck="memory_pressure",
    )
    assert recs == []


def test_no_bottleneck_produces_no_recommendation() -> None:
    assert bottleneck_to_resource("none") is None
    assert bottleneck_to_resource("gpu_memory_limit") is None
    report = _memory_report()
    assessments = assess_external_attribution(report, _policy(), target_degraded=True)
    recs = build_recommendations(
        assessments,
        registry=build_default_registry(),
        policy=_policy(),
        context_identity="ctx",
        diagnosed_resource=None,
    )
    assert recs == []


def test_uncertainty_is_explicit_for_correlation_only() -> None:
    report = _cpu_report()
    assessments = assess_external_attribution(report, _policy(), target_degraded=False)
    recs = build_recommendations(
        assessments,
        registry=build_default_registry(),
        policy=_policy(),
        context_identity="ctx",
        diagnosed_resource=ExternalResource.CPU,
        diagnosed_bottleneck="cpu",
    )
    rec = next(r for r in recs if r.identity and r.identity.pid == 200)
    assert rec.classification is ExternalRecommendationClass.INFORMATIONAL
    assert rec.expected_benefit_available is False
    assert "correlation" in rec.uncertainty


def test_stale_process_identity_is_detected() -> None:
    with launch_sleeper() as proc:
        identity = resolve_identity(proc.pid)
        assert identity is not None
        external = ExternalProcessObservation(
            observation_id="extproc-real",
            identity=identity,
            cpu_percent=180.0,
            rss_bytes=512 * 1024 * 1024,
            measurable=["rss_bytes", "cpu_percent"],
        )
        target = make_process_observation(pid=100, name="llama.exe", is_target=True)
        report = make_report(processes=[target, external], target_pid=100, memory_pressure=0.5)
        assessments = assess_external_attribution(report, _policy(), target_degraded=True)
        recs = build_recommendations(
            assessments,
            registry=build_default_registry(),
            policy=_policy(),
            context_identity="ctx",
            diagnosed_resource=ExternalResource.CPU,
            diagnosed_bottleneck="cpu",
            run_id="run-A",
            session_id="session-A",
        )
        rec = recs[0]
    stale = revalidate_recommendation(
        rec, current_run_id="run-A", current_session_id="session-A", revalidate_process=True
    )
    assert stale.status is ExternalRecommendationStatus.STALE
    assert stale.stale_reason is not None


def test_stale_provenance_is_detected() -> None:
    rec = _first_recommendation(_memory_report(), ExternalResource.MEMORY)
    stale = revalidate_recommendation(
        rec,
        current_run_id="run-B",
        current_session_id="session-B",
        revalidate_process=False,
    )
    assert stale.status is ExternalRecommendationStatus.STALE
    assert stale.stale_reason is not None
    assert "run-B" in stale.stale_reason or "session-B" in stale.stale_reason


def test_recommendation_expiration_is_detected() -> None:
    rec = _first_recommendation(_memory_report(), ExternalResource.MEMORY)
    future = (rec.expires_at or datetime.now(timezone.utc)) + timedelta(seconds=1)
    expired = revalidate_recommendation(
        rec, current_run_id="run-A", current_session_id="session-A", now=future
    )
    assert expired.status is ExternalRecommendationStatus.EXPIRED
    assert expired.stale_reason is not None


def test_user_rejection_records_decision_without_action() -> None:
    clock = FakeClock()
    engine = ExternalRecommendationEngine(
        observer=ScriptedProcessObserver(_memory_report()),
        registry=build_default_registry(),
        policy=_policy(),
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.REJECTED]),
        clock=clock,
        sleep=clock.sleep,
    )
    run = engine.run(
        target_pid=100,
        target_degraded=True,
        context_identity="ctx",
        diagnosed_bottleneck="memory_pressure",
        diagnosed_resource=ExternalResource.MEMORY,
        max_cycles=1,
    )
    assert run.user_decisions
    assert run.user_decisions[0].decision is ExternalRecommendationStatus.REJECTED
    assert run.action_results == []
    assert run.termination.reason is ExternalRunTerminationReason.APPROVAL_DECLINED


def test_user_acceptance_records_intent_without_execution() -> None:
    clock = FakeClock()
    engine = ExternalRecommendationEngine(
        observer=ScriptedProcessObserver(_memory_report()),
        registry=build_default_registry(),
        policy=_policy(),
        approval_provider=AutoApproveApprovalProvider(),
        clock=clock,
        sleep=clock.sleep,
    )
    run = engine.run(
        target_pid=100,
        target_degraded=True,
        context_identity="ctx",
        diagnosed_bottleneck="memory_pressure",
        diagnosed_resource=ExternalResource.MEMORY,
        max_cycles=1,
    )
    assert run.user_decisions
    assert run.user_decisions[0].decision is ExternalRecommendationStatus.ACCEPTED
    assert run.user_decisions[0].autonomous_execution is False
    assert run.action_results == []
    assert run.termination.reason is ExternalRunTerminationReason.RECOMMENDATION_ACCEPTED


def test_acceptance_does_not_execute_external_action_on_real_process() -> None:
    import psutil

    clock = FakeClock()
    with launch_sleeper() as proc:
        identity = resolve_identity(proc.pid)
        assert identity is not None
        external = ExternalProcessObservation(
            observation_id="extproc-real",
            identity=identity,
            cpu_percent=190.0,
            rss_bytes=512 * 1024 * 1024,
            measurable=["rss_bytes", "cpu_percent"],
        )
        target = make_process_observation(pid=100, name="llama.exe", is_target=True)
        report = make_report(processes=[target, external], target_pid=100, memory_pressure=0.5)
        nice_before = psutil.Process(proc.pid).nice()
        engine = ExternalRecommendationEngine(
            observer=ScriptedProcessObserver(report),
            registry=build_default_registry(),
            policy=_policy(),
            approval_provider=AutoApproveApprovalProvider(),
            clock=clock,
            sleep=clock.sleep,
        )
        run = engine.run(
            target_pid=100,
            target_degraded=True,
            context_identity="ctx",
            diagnosed_bottleneck="cpu",
            diagnosed_resource=ExternalResource.CPU,
            max_cycles=1,
        )
        assert run.user_decisions[0].decision is ExternalRecommendationStatus.ACCEPTED
        assert run.action_results == []
        assert psutil.Process(proc.pid).nice() == nice_before
        assert verify_identity(identity) is True


def test_no_meanful_bottleneck_termination() -> None:
    clock = FakeClock()
    engine = ExternalRecommendationEngine(
        observer=ScriptedProcessObserver(_memory_report()),
        registry=build_default_registry(),
        policy=_policy(),
        approval_provider=AutoRejectApprovalProvider(),
        clock=clock,
        sleep=clock.sleep,
    )
    run = engine.run(
        target_pid=100,
        target_degraded=False,
        context_identity="ctx",
        diagnosed_resource=None,
        max_cycles=1,
    )
    assert run.recommendations == []
    assert run.termination.reason is ExternalRunTerminationReason.NO_MEANINGFUL_BOTTLENECK
