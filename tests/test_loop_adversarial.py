"""Phase 9 adversarial-reasoning containment tests."""

from __future__ import annotations

from inferenceos.cli import _adversarial_proposal_payload
from inferenceos.contracts.optimization_loop import LoopTerminationReason
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from tests.loop_helpers import (
    default_loop_policy,
    make_loop,
    no_improvement_responder,
)
from tests.reasoning_helpers import insufficient_bundle


def test_adversarial_proposals_are_rejected_by_the_registry() -> None:
    _, diagnosis, report = insufficient_bundle()
    provider = FakeReasoningProvider(scripted=[_adversarial_proposal_payload()])
    plan = generate_candidates(
        diagnosis,
        report=report,
        policy=OptimizationPolicy(),
        provider=provider,
        enable_reasoning=True,
    )
    assert plan.reasoning is not None
    joined = " | ".join(plan.reasoning.rejections)
    assert "unregistered" in joined
    assert "hallucinated evidence" in joined
    assert "fabricated measurement" in joined
    assert not any("shell" in candidate.action_id for candidate in plan.candidates)
    assert not any(candidate.executable for candidate in plan.candidates)


def test_loop_never_executes_an_adversarial_proposal() -> None:
    _, _, report = insufficient_bundle()
    provider = FakeReasoningProvider(scripted=[_adversarial_proposal_payload()])
    policy = default_loop_policy(allow_deep_diagnosis=False)
    stack = make_loop(
        responder=no_improvement_responder,
        report=report,
        loop_policy=policy,
        proposal_provider=provider,
        enable_reasoning_proposals=True,
    )
    result = stack.loop.run(report, report_path="<test-adversarial>")
    assert result.run.budget.experiments_used == 0
    assert result.changed is False
    assert not stack.harness.calls
    assert result.termination.reason in {
        LoopTerminationReason.INSUFFICIENT_EVIDENCE,
        LoopTerminationReason.NO_ELIGIBLE_CANDIDATES,
        LoopTerminationReason.ALL_CANDIDATES_REJECTED,
        LoopTerminationReason.NO_MEANINGFUL_OPPORTUNITY,
    }


def test_registry_rejects_command_shaped_actions() -> None:
    registry = build_default_registry()
    assert "shell.run_command" not in registry
    assert "system.increase_pagefile" in registry
