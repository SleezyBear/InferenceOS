"""Phase 6 optimization contracts: existence, round-trip, strictness."""

from __future__ import annotations

import pytest

from inferenceos.contracts.optimization import (
    CandidateAction,
    CandidatePlan,
    OptimizationCandidate,
)
from inferenceos.selftest import (
    sample_candidate_action,
    sample_candidate_plan,
    sample_optimization_candidate,
    verify_contract_roundtrip,
)

REQUIRED = {
    "CandidateAction",
    "OptimizationCandidate",
    "CandidatePlan",
}


def test_phase6_contracts_roundtrip() -> None:
    passed = verify_contract_roundtrip()
    assert REQUIRED.issubset(set(passed))


def test_candidate_plan_roundtrips() -> None:
    plan = sample_candidate_plan()
    restored = CandidatePlan.from_json(plan.to_json())
    assert restored == plan
    assert restored.executed is False
    assert restored.environment_mutated is False


def test_contracts_reject_unknown_fields() -> None:
    action = sample_candidate_action()
    with pytest.raises(ValueError):
        CandidateAction.model_validate({**action.model_dump(), "unexpected": 1})
    candidate = sample_optimization_candidate()
    with pytest.raises(ValueError):
        OptimizationCandidate.model_validate({**candidate.model_dump(), "unexpected": 1})


def test_candidate_has_context_and_evidence() -> None:
    candidate = sample_optimization_candidate()
    assert candidate.hardware.cpu.name
    assert candidate.model_ref
    assert candidate.workload_id
    assert candidate.evidence_refs
    assert candidate.preconditions
