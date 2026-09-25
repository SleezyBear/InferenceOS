"""Phase 11 adversarial reasoning / safety-bypass tests."""

from __future__ import annotations

from inferenceos.cli import _adversarial_proposal_payload
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import ApprovalOutcome, ExternalProcessAccess, ExternalResource
from inferenceos.contracts.external import ExternalBudget, ExternalPolicy, ExternalProcessObservation
from inferenceos.external.actions import evaluate_external_action_safety
from inferenceos.external.engine import ExternalRecommendationEngine
from inferenceos.external.identity import resolve_identity, verify_identity
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from inferenceos.runtime.approval import AutoApproveApprovalProvider, ScriptedApprovalProvider
from tests.external_helpers import (
    FakeClock,
    ScriptedProcessObserver,
    launch_sleeper,
    make_process_observation,
    make_report,
)
from tests.optimization_helpers import synthetic_context


def _policy() -> ExternalPolicy:
    return ExternalPolicy.model_validate(
        {
            "budget": ExternalBudget(max_duration_s=300.0, max_recommendations=4, max_actions=2),
            "observation_interval_s": 1.0,
            "min_process_rss_bytes": 256 * 1024 * 1024,
            "min_process_cpu_percent": 25.0,
        }
    )


def _cpu_report():  # type: ignore[no-untyped-def]
    target = make_process_observation(pid=100, name="llama.exe", is_target=True)
    external = make_process_observation(pid=200, name="python.exe", cpu_percent=180.0)
    return make_report(processes=[target, external], target_pid=100, memory_pressure=0.5)


def test_reasoner_cannot_invent_an_external_action() -> None:
    ctx = synthetic_context(BottleneckKind.PAGING)
    provider = FakeReasoningProvider(scripted=[_adversarial_proposal_payload()])
    plan = generate_candidates(
        ctx.diagnosis, report=None, policy=OptimizationPolicy(),
        provider=provider, enable_reasoning=True,
    )
    registry = build_default_registry()
    assert all(candidate.action_id in registry for candidate in plan.candidates)
    assert not any("external." in c.action_id and c.action_id not in registry for c in plan.candidates)
    assert not any("shell" in c.action_id for c in plan.candidates)


def test_engine_only_recommends_observed_identities() -> None:
    report = _cpu_report()
    engine = ExternalRecommendationEngine(
        observer=ScriptedProcessObserver(report),
        registry=build_default_registry(),
        policy=_policy(),
        approval_provider=AutoApproveApprovalProvider(),
    )
    run = engine.run(
        target_pid=100,
        target_degraded=True,
        context_identity="ctx",
        diagnosed_bottleneck="cpu",
        diagnosed_resource=ExternalResource.CPU,
        max_cycles=1,
    )
    observed_pids = {obs.identity.pid for r in run.observation_reports for obs in r.processes}
    for recommendation in run.recommendations:
        if recommendation.identity is not None:
            assert recommendation.identity.pid in observed_pids


def test_mismatched_approval_is_stale_and_never_acts() -> None:
    clock = FakeClock()
    with launch_sleeper() as proc:
        identity = resolve_identity(proc.pid)
        assert identity is not None
        target = make_process_observation(pid=100, name="llama.exe", is_target=True)
        external = ExternalProcessObservation(
            observation_id="extproc-real",
            identity=identity,
            cpu_percent=190.0,
            rss_bytes=512 * 1024 * 1024,
            measurable=["rss_bytes", "cpu_percent"],
            access=ExternalProcessAccess.READABLE,
        )
        report = make_report(processes=[target, external], target_pid=100, memory_pressure=0.5)
        engine = ExternalRecommendationEngine(
            observer=ScriptedProcessObserver(report),
            registry=build_default_registry(),
            policy=_policy(),
            approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVED], mismatch_proposal=True),
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
        assert run.action_results == []
        assert any(a.outcome is ApprovalOutcome.STALE for a in run.approvals)
        assert run.user_decisions
        assert run.user_decisions[0].autonomous_execution is False


def test_unregistered_and_unsupported_actions_are_never_allowed() -> None:
    registry = build_default_registry()
    policy = ExternalPolicy.model_validate({"budget": ExternalBudget()})
    for action_id in (
        "external.kill_process",
        "external.inject",
        "external.suspend_process",
        "external.request_close",
        "system.increase_pagefile",
    ):
        safety = evaluate_external_action_safety(
            action_id, registry=registry, policy=policy, identity=None
        )
        assert safety.allowed is False, action_id
    assert registry.get("external.suspend_process").implemented is False
    assert registry.get("external.request_close").implemented is False


def test_acceptance_does_not_mutate_the_real_process() -> None:
    clock = FakeClock()
    with launch_sleeper() as proc:
        identity = resolve_identity(proc.pid)
        assert identity is not None
        target = make_process_observation(pid=100, name="llama.exe", is_target=True)
        external = ExternalProcessObservation(
            observation_id="extproc-real",
            identity=identity,
            cpu_percent=190.0,
            rss_bytes=512 * 1024 * 1024,
            measurable=["rss_bytes", "cpu_percent"],
            access=ExternalProcessAccess.READABLE,
        )
        report = make_report(processes=[target, external], target_pid=100, memory_pressure=0.5)
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
        # Acceptance records intent; it never touches the real external process.
        assert run.user_decisions
        assert run.user_decisions[0].autonomous_execution is False
        assert run.action_results == []
        assert verify_identity(identity) is True
