"""Phase 10 runtime contract / policy / checkpoint / economics tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferenceos.config import Settings
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    AnomalyClass,
    EvidenceStrength,
    InterventionClass,
    RuntimeState,
    RuntimeTerminationReason,
)
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.runtime import (
    RuntimeBudget,
    RuntimeMonitorResult,
    RuntimeNotification,
    RuntimeOpportunity,
    RuntimePolicy,
    RuntimeRecommendation,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.runtime.approval import AutoRejectApprovalProvider
from inferenceos.runtime.checkpoint import (
    capture_runtime_checkpoint,
    restore_runtime_checkpoint,
    verify_runtime_checkpoint,
)
from inferenceos.runtime.economics import classify_action, evaluate_intervention_economics
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.notification import CollectingNotificationSink
from inferenceos.runtime.policy import build_runtime_budget, build_runtime_policy
from inferenceos.selftest import sample_hardware
from tests.optimization_helpers import make_candidate, synthetic_context
from tests.runtime_helpers import ScriptedSampler, ScriptedWorkload, make_runtime_policy


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


def _healthy(sequence: int, condition: str) -> dict[str, float]:
    return {
        "memory_pressure": 0.5,
        "hard_page_faults_per_sec": 0.5,
        "available_ram_bytes": float(8 * 1024**3),
        "decode_tokens_per_sec": 24.0,
        "total_latency_s": 1.4,
        "peak_rss_bytes": 1_300_000_000.0,
    }


def test_budget_exhaustion_is_deterministic() -> None:
    budget = RuntimeBudget(
        max_duration_s=100.0,
        max_optimizer_overhead_s=50.0,
        max_interventions=2,
    )
    assert budget.exhaustion(elapsed_s=1.0) is None
    assert budget.exhaustion(elapsed_s=100.0) is RuntimeTerminationReason.DURATION_REACHED
    overhead = budget.model_copy(update={"optimizer_overhead_used_s": 50.0})
    assert (
        overhead.exhaustion(elapsed_s=1.0)
        is RuntimeTerminationReason.MONITORING_BUDGET_EXHAUSTED
    )
    used = budget.model_copy(update={"interventions_used": 2})
    assert (
        used.exhaustion(elapsed_s=1.0)
        is RuntimeTerminationReason.INTERVENTION_LIMIT_REACHED
    )


def test_invalid_configuration_fails_before_monitoring() -> None:
    with pytest.raises(ValidationError):
        RuntimeBudget(max_duration_s=0.0)
    with pytest.raises(ValidationError):
        RuntimeBudget(telemetry_interval_s=0.0)
    with pytest.raises(ValidationError):
        RuntimePolicy(persistence_samples=1)
    with pytest.raises(ValidationError):
        build_runtime_policy(Settings(phase10_baseline_samples=1))
    with pytest.raises(ValidationError):
        build_runtime_budget(Settings(phase10_max_duration_s=-1.0))


def test_build_runtime_policy_from_settings() -> None:
    settings = Settings(
        phase10_max_duration_s=42.0,
        phase10_require_approval_for_disruptive=True,
        phase10_allow_non_disruptive_auto_adaptation=False,
    )
    policy = build_runtime_policy(settings)
    assert policy.budget.max_duration_s == 42.0
    assert policy.require_approval_for_disruptive is True
    assert policy.allow_non_disruptive_auto_adaptation is False
    assert "cost_model" in policy.as_dict()


def test_checkpoint_capture_verify_restore() -> None:
    workload = ScriptedWorkload()
    checkpoint = capture_runtime_checkpoint(workload, reason="test", context_identity="ctx-1")
    assert checkpoint.verified is True
    assert checkpoint.integrity is not None
    assert checkpoint.resume_supported is True

    workload.set_configuration(RuntimeConfiguration(context_length=1024))
    assert verify_runtime_checkpoint(workload, checkpoint) is False
    assert restore_runtime_checkpoint(workload, checkpoint) is True
    assert verify_runtime_checkpoint(workload, checkpoint) is True


def test_classify_action_safety() -> None:
    registry = build_default_registry()
    policy = RuntimePolicy()
    assert (
        classify_action(registry.get("runtime.enable_mmap"), policy)
        is InterventionClass.NON_DISRUPTIVE_ADAPTATION
    )
    # A card that asks before testing is approval-gated, hence disruptive.
    gated = registry.get("runtime.enable_mmap").model_copy(
        update={"requires_human_approval": True}
    )
    assert classify_action(gated, policy) is InterventionClass.DISRUPTIVE_ADAPTATION
    assert (
        classify_action(registry.get("system.increase_pagefile"), policy)
        is InterventionClass.EXTERNAL_SYSTEM_WIDE
    )


def test_economics_marks_external_actions_as_approval_required() -> None:
    registry = build_default_registry()
    policy = RuntimePolicy()
    opportunity = RuntimeOpportunity(
        opportunity_id="opp",
        anomaly=AnomalyClass.PERSISTENT,
        condition_key="k",
        kind="paging",
        persistence_samples=3,
        deviation_score=0.9,
        confidence=0.8,
        strength=EvidenceStrength.MEDIUM,
    )
    external = evaluate_intervention_economics(
        opportunity,
        candidate=None,
        action=registry.get("system.increase_pagefile"),
        policy=policy,
        reasoning_required=False,
    )
    assert external.intervention_class is InterventionClass.EXTERNAL_SYSTEM_WIDE
    assert external.requires_approval is True

    ctx = synthetic_context(BottleneckKind.PAGING)
    candidate = make_candidate(ctx, action_id="runtime.enable_mmap")
    safe = evaluate_intervention_economics(
        opportunity,
        candidate=candidate,
        action=registry.get("runtime.enable_mmap"),
        policy=policy,
        reasoning_required=False,
    )
    assert safe.intervention_class is InterventionClass.NON_DISRUPTIVE_ADAPTATION
    assert safe.rationale


def test_notification_payload_is_structured() -> None:
    recommendation = RuntimeRecommendation(
        proposal_id="prop-1",
        opportunity_id="opp-1",
        condition="paging sustained",
        why_it_matters="latency",
        analysis_summary="controlled experiment",
        proposed_action_id="runtime.enable_mmap",
        expected_benefit="peak_rss_bytes decrease",
        expected_cost_s=12.0,
        expected_disruption_s=3.0,
        reversible=True,
        context_identity="ctx-1",
    )
    notification = RuntimeNotification(
        notification_id="notif-1",
        proposal_id="prop-1",
        title="t",
        summary="s",
        recommendation=recommendation,
    )
    sink = CollectingNotificationSink()
    sink.notify(notification)
    assert sink.notifications[0].recommendation.proposed_action_id == "runtime.enable_mmap"
    assert sink.notifications[0].recommendation.requires_approval is True


def test_runtime_monitor_result_roundtrips() -> None:
    clock = _Clock()
    monitor = RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=ScriptedSampler(_healthy),
        policy=make_runtime_policy(
            budget=RuntimeBudget(
                max_duration_s=3.0,
                telemetry_interval_s=0.5,
                observation_window_size=20,
                baseline_samples=4,
            )
        ),
        approval_provider=AutoRejectApprovalProvider(),
        hardware=sample_hardware(),
        clock=clock,
        sleep=clock.sleep,
    )
    result = monitor.run()
    assert result.final_state is RuntimeState.TERMINAL_SAFE
    restored = RuntimeMonitorResult.from_json(result.to_json())
    assert restored.run.run_id == result.run.run_id
    assert restored.termination.reason is result.termination.reason
    assert restored.to_json() == result.to_json()
