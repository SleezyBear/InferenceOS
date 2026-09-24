"""Candidate revalidation immediately before execution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from inferenceos.contracts.enums import ExperimentDecisionKind
from inferenceos.contracts.optimization import CandidatePlan, OptimizationCandidate
from inferenceos.experiment.eligibility import RevalidationResult, revalidate_candidate
from inferenceos.experiment.executor import ActionExecutor, RuntimeConfigurationExecutor
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.context import OptimizationContext, build_optimization_context
from inferenceos.optimization.policy import OptimizationPolicy
from tests.experiment_helpers import paging_candidate_and_plan


def _context_and_plan() -> tuple[OptimizationCandidate, CandidatePlan, OptimizationContext]:
    candidate, plan, report = paging_candidate_and_plan()
    context = build_optimization_context(
        plan.source_diagnosis, report=report, outcome=None, policy=OptimizationPolicy()
    )
    return candidate, plan, context


def _revalidate(
    candidate: OptimizationCandidate,
    plan: CandidatePlan,
    context: OptimizationContext,
    *,
    executor: ActionExecutor | None = None,
    ttl_s: float = 3600.0,
    now: datetime | None = None,
) -> RevalidationResult:
    return revalidate_candidate(
        candidate,
        plan=plan,
        context=context,
        registry=build_default_registry(),
        policy=OptimizationPolicy(),
        executor=executor or RuntimeConfigurationExecutor(),
        ttl_s=ttl_s,
        now=now,
    )


def test_eligible_candidate_passes() -> None:
    candidate, plan, context = _context_and_plan()
    result = _revalidate(candidate, plan, context)
    assert result.ok is True
    assert result.action is not None
    assert result.action.action_id == "runtime.right_size_context"
    assert result.checks


def test_unregistered_action_rejected() -> None:
    candidate, plan, context = _context_and_plan()
    stale = candidate.model_copy(update={"action_id": "shell.run_command"})
    result = _revalidate(stale, plan, context)
    assert result.ok is False
    assert result.kind is ExperimentDecisionKind.UNSAFE


def test_approval_required_candidate_blocked() -> None:
    _, plan, context = _context_and_plan()
    gated = next(c for c in plan.candidates if c.action_id == "system.increase_pagefile")
    result = _revalidate(gated, plan, context)
    assert result.ok is False
    assert result.kind is ExperimentDecisionKind.UNSAFE


def test_candidate_not_in_plan_rejected() -> None:
    candidate, plan, context = _context_and_plan()
    outsider = candidate.model_copy(update={"candidate_id": "cand-not-in-plan"})
    result = _revalidate(outsider, plan, context)
    assert result.ok is False
    assert result.kind is ExperimentDecisionKind.UNSAFE


def test_unsupported_executor_rejected() -> None:
    candidate, plan, context = _context_and_plan()
    executor = RuntimeConfigurationExecutor(supported=set())
    result = _revalidate(candidate, plan, context, executor=executor)
    assert result.ok is False
    assert result.kind is ExperimentDecisionKind.PRECONDITION_FAILURE


def test_expired_candidate_rejected() -> None:
    candidate, plan, context = _context_and_plan()
    old_plan = plan.model_copy(update={"created_at": datetime.now(timezone.utc) - timedelta(days=10)})
    result = _revalidate(candidate, old_plan, context, ttl_s=60.0)
    assert result.ok is False
    assert result.kind is ExperimentDecisionKind.PRECONDITION_FAILURE


def test_environment_mismatch_rejected() -> None:
    candidate, plan, context = _context_and_plan()
    stale = candidate.model_copy(update={"workload_id": "some-other-workload"})
    result = _revalidate(stale, plan, context)
    assert result.ok is False
    assert result.kind is ExperimentDecisionKind.PRECONDITION_FAILURE


def test_bottleneck_mismatch_rejected() -> None:
    from inferenceos.contracts.enums import BottleneckKind

    candidate, plan, context = _context_and_plan()
    stale = candidate.model_copy(update={"diagnosis_kind": BottleneckKind.CPU})
    result = _revalidate(stale, plan, context)
    assert result.ok is False
    assert result.kind is ExperimentDecisionKind.PRECONDITION_FAILURE


def test_invalid_parameters_rejected() -> None:
    candidate, plan, context = _context_and_plan()
    bad = candidate.model_copy(update={"parameters": {"cmd": "powershell"}})
    result = _revalidate(bad, plan, context)
    assert result.ok is False
    assert result.kind is ExperimentDecisionKind.PRECONDITION_FAILURE
