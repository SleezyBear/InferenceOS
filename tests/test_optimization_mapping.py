"""Diagnosis-to-action mapping and controlled no-op / insufficient-evidence paths."""

from __future__ import annotations

from inferenceos.contracts.enums import (
    ApprovalRequirement,
    BottleneckKind,
    CandidateEligibility,
    CandidatePlanOutcome,
    PreconditionKind,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.context import build_optimization_context
from inferenceos.optimization.generator import (
    _deduplicate,
    _resolve_conflicts,
    generate_candidates,
)
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.preconditions import evaluate_eligibility, make
from tests.optimization_helpers import (
    healthy_bundle,
    insufficient_bundle,
    make_candidate,
    paging_bundle,
    requantize_action,
    synthetic_diagnosis,
)


def test_paging_diagnosis_generates_relevant_candidate() -> None:
    _, diagnosis, report = paging_bundle()
    plan = generate_candidates(diagnosis, report=report)
    assert plan.outcome is CandidatePlanOutcome.CANDIDATES_GENERATED
    assert plan.effective_bottleneck is BottleneckKind.PAGING
    eligible = [c for c in plan.candidates if c.eligibility is CandidateEligibility.ELIGIBLE]
    assert eligible
    mmap_candidates = [c for c in eligible if c.action_id == "runtime.enable_mmap"]
    assert mmap_candidates
    candidate = mmap_candidates[0]
    assert candidate.executable
    assert candidate.evidence_refs
    assert candidate.preconditions
    assert candidate.hardware.cpu.name
    assert candidate.model_ref == "m"
    assert candidate.workload_id == "short-short"
    assert candidate.ranking is not None
    # Context right-sizing is never generated as a candidate at all.
    assert all(
        c.action_id != "runtime.right_size_context" for c in plan.candidates
    )
    # Non-causal GPU candidates must not appear for a paging diagnosis.
    assert all(c.action_id != "runtime.reduce_gpu_offload_layers" for c in plan.candidates)


def test_candidate_generation_executes_nothing() -> None:
    _, diagnosis, report = paging_bundle()
    plan = generate_candidates(diagnosis, report=report)
    assert plan.executed is False
    assert plan.environment_mutated is False
    assert all(not c.risk.requires_approval for c in plan.candidates if c.executable)


def test_high_gpu_utilization_no_opportunity_no_speculative_candidate() -> None:
    _, diagnosis, report = healthy_bundle()
    # The healthy fixture has high GPU utilization but no meaningful bottleneck.
    assert diagnosis.meaningful_opportunity is False
    plan = generate_candidates(diagnosis, report=report)
    assert plan.outcome in {
        CandidatePlanOutcome.NO_OPPORTUNITY,
        CandidatePlanOutcome.INSUFFICIENT_EVIDENCE,
    }
    assert plan.candidates == []
    assert all(c.action_id != "runtime.reduce_gpu_offload_layers" for c in plan.candidates)


def test_insufficient_evidence_produces_no_executable_candidate() -> None:
    _, diagnosis, report = insufficient_bundle()
    assert diagnosis.insufficient_evidence is True
    plan = generate_candidates(diagnosis, report=report)
    assert plan.outcome is CandidatePlanOutcome.INSUFFICIENT_EVIDENCE
    assert all(not c.executable for c in plan.candidates)


def test_missing_telemetry_is_not_fabricated() -> None:
    diagnosis = synthetic_diagnosis(BottleneckKind.PAGING)
    backend = diagnosis.context.backend.model_copy(
        update={"context_length": None, "use_mmap": None}
    )
    context = diagnosis.context.model_copy(update={"backend": backend})
    diagnosis = diagnosis.model_copy(update={"context": context})
    plan = generate_candidates(diagnosis, report=None)
    assert all(not c.executable for c in plan.candidates)
    assert all(not c.benefit.known for c in plan.candidates)
    assert all(not c.observed_metrics for c in plan.candidates)


def test_external_and_elevated_actions_are_gated() -> None:
    _, diagnosis, report = paging_bundle()
    plan = generate_candidates(diagnosis, report=report)
    gated = [c for c in plan.candidates if c.eligibility is CandidateEligibility.GATED]
    assert any(c.action_id == "system.increase_pagefile" for c in gated)
    for candidate in gated:
        assert candidate.approval is ApprovalRequirement.REQUIRED
        assert candidate.risk.requires_approval is True
        assert candidate.executable is False


def test_non_reversible_action_is_rejected_by_policy() -> None:
    action = requantize_action()
    preconditions = [
        make("evidence", PreconditionKind.EVIDENCE, "supported", True),
        make("available", PreconditionKind.APPLICABILITY, "smaller available", True),
        make("approval", PreconditionKind.SAFETY, "approval", None),
    ]
    eligibility, approval, reason = evaluate_eligibility(
        action, preconditions, OptimizationPolicy()
    )
    assert eligibility is CandidateEligibility.REJECTED
    assert approval is ApprovalRequirement.NOT_REQUIRED
    assert reason is not None and "non-reversible" in reason


def test_conflicting_candidates_resolved_deterministically() -> None:
    from tests.optimization_helpers import synthetic_context

    ctx = synthetic_context(BottleneckKind.PAGING)
    first = make_candidate(
        ctx, action_id="runtime.reduce_threads", magnitude=500_000_000.0
    ).model_copy(update={"candidate_id": "cand-a", "parameters": {"threads": 4}})
    second = make_candidate(
        ctx, action_id="runtime.reduce_threads", magnitude=100_000_000.0
    ).model_copy(update={"candidate_id": "cand-b", "parameters": {"threads": 2}})
    resolved = _resolve_conflicts([first, second])
    by_id = {c.candidate_id: c for c in resolved}
    assert by_id["cand-a"].eligibility is CandidateEligibility.ELIGIBLE
    assert by_id["cand-b"].eligibility is CandidateEligibility.REJECTED
    assert by_id["cand-b"].rejection_reason is not None


def test_equivalent_candidates_are_deduplicated() -> None:
    from tests.optimization_helpers import synthetic_context

    ctx = synthetic_context(BottleneckKind.PAGING)
    first = make_candidate(ctx).model_copy(update={"candidate_id": "cand-a"})
    second = make_candidate(ctx).model_copy(update={"candidate_id": "cand-b"})
    assert len(_deduplicate([first, second])) == 1


def test_effective_bottleneck_from_validated_reasoning() -> None:
    from inferenceos.contracts.enums import EvidenceStrength, ReasoningStatus
    from inferenceos.contracts.reasoning import (
        DeepDiagnosisOutcome,
        EscalationDecision,
        ReasoningLifecycle,
        ReasoningResponse,
        ReasoningResult,
    )

    diagnosis = synthetic_diagnosis(
        BottleneckKind.INSUFFICIENT_EVIDENCE, meaningful=False, insufficient=True
    )
    result = ReasoningResult(
        root_cause_class=BottleneckKind.MEMORY_PRESSURE,
        confidence=0.7,
        evidence_strength=EvidenceStrength.MEDIUM,
    )
    response = ReasoningResponse(
        request_id="r",
        status=ReasoningStatus.OK,
        conclusion="x",
        confidence=0.7,
        validation_ok=True,
        result=result,
    )
    outcome = DeepDiagnosisOutcome(
        outcome_id="o",
        escalation=EscalationDecision(
            should_escalate=True, reason_code="insufficient_evidence", rationale="x"
        ),
        deterministic=diagnosis,
        reasoning=response,
        lifecycle=ReasoningLifecycle(reasoning_performed=True),
        final_root_cause=BottleneckKind.MEMORY_PRESSURE,
        final_confidence=0.7,
        final_evidence_strength=EvidenceStrength.MEDIUM,
    )
    ctx = build_optimization_context(
        diagnosis, report=None, outcome=outcome, policy=OptimizationPolicy()
    )
    assert ctx.effective_kind is BottleneckKind.MEMORY_PRESSURE
    assert ctx.effective_basis == "reasoning"
