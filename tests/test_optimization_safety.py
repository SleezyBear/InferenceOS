"""Safety gates, side-effect protection, and prior-phase invariants."""

from __future__ import annotations

import json

import psutil

from inferenceos.contracts.enums import (
    ApprovalRequirement,
    BottleneckKind,
    CandidateEligibility,
    PreconditionKind,
    RiskLevel,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.preconditions import evaluate_eligibility, make
from tests.optimization_helpers import paging_bundle

FORBIDDEN = ("powershell", "cmd.exe", "rmdir", "Remove-Item", "reg add", "bcdedit")


def test_missing_preconditions_is_not_eligible() -> None:
    action = build_default_registry().get("runtime.right_size_context")
    preconditions = [make("evidence", PreconditionKind.EVIDENCE, "supported", None)]
    eligibility, _, reason = evaluate_eligibility(action, preconditions, OptimizationPolicy())
    assert eligibility is CandidateEligibility.INSUFFICIENT_EVIDENCE
    assert reason is not None


def test_failed_capability_precondition_rejects() -> None:
    action = build_default_registry().get("runtime.right_size_context")
    preconditions = [
        make("evidence", PreconditionKind.EVIDENCE, "supported", True),
        make("capability", PreconditionKind.CAPABILITY, "backend supports it", False),
    ]
    eligibility, _, reason = evaluate_eligibility(action, preconditions, OptimizationPolicy())
    assert eligibility is CandidateEligibility.REJECTED
    assert reason is not None


def test_broad_scope_action_is_always_gated() -> None:
    registry = build_default_registry()
    preconditions = [
        make("evidence", PreconditionKind.EVIDENCE, "supported", True),
    ]
    for action_id in ("external.suspend_process", "system.increase_pagefile"):
        action = registry.get(action_id)
        eligibility, approval, _ = evaluate_eligibility(
            action, preconditions, OptimizationPolicy()
        )
        assert eligibility is CandidateEligibility.GATED
        assert approval is ApprovalRequirement.REQUIRED


def test_risk_above_autonomous_ceiling_is_gated() -> None:
    action = build_default_registry().get("runtime.right_size_context").model_copy(
        update={"risk_level": RiskLevel.HIGH}
    )
    preconditions = [make("evidence", PreconditionKind.EVIDENCE, "supported", True)]
    eligibility, approval, _ = evaluate_eligibility(action, preconditions, OptimizationPolicy())
    assert eligibility is CandidateEligibility.GATED
    assert approval is ApprovalRequirement.REQUIRED


def test_disabled_action_is_rejected() -> None:
    action = build_default_registry().get("runtime.right_size_context").model_copy(
        update={"enabled": False}
    )
    eligibility, _, reason = evaluate_eligibility(action, [], OptimizationPolicy())
    assert eligibility is CandidateEligibility.REJECTED
    assert reason == "action disabled"


def test_candidate_generation_has_no_side_effects() -> None:
    _, diagnosis, report = paging_bundle()
    try:
        before = sorted(child.pid for child in psutil.Process().children(recursive=True))
    except (psutil.Error, OSError):  # pragma: no cover - defensive
        before = []
    plan = generate_candidates(diagnosis, report=report)
    try:
        after = sorted(child.pid for child in psutil.Process().children(recursive=True))
    except (psutil.Error, OSError):  # pragma: no cover - defensive
        after = []

    assert before == after
    assert plan.executed is False
    assert plan.environment_mutated is False
    payload = plan.to_json().lower()
    for token in FORBIDDEN:
        assert token.lower() not in payload


def test_plan_json_roundtrips_without_execution() -> None:
    _, diagnosis, report = paging_bundle()
    plan = generate_candidates(diagnosis, report=report)
    restored = json.loads(plan.to_json())
    assert restored["executed"] is False
    assert restored["environment_mutated"] is False
    assert restored["selected_ids"] == plan.selected_ids


def test_prior_phase_contracts_still_import_and_roundtrip() -> None:
    from inferenceos.contracts.diagnosis import BottleneckDiagnosis
    from inferenceos.selftest import sample_diagnosis

    diagnosis = sample_diagnosis()
    assert BottleneckDiagnosis.from_json(diagnosis.to_json()) == diagnosis
    assert diagnosis.primary is not None
    assert diagnosis.primary.kind is BottleneckKind.MEMORY_PRESSURE
