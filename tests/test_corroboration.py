"""Adversarial reasoning containment + corroboration tests."""

from __future__ import annotations

import json

from inferenceos.contracts.enums import EvidenceStrength
from inferenceos.contracts.reasoning import (
    ReasoningDiagnosisContext,
    ReasoningRequest,
    ReasoningResponse,
)
from inferenceos.reasoning.context import build_reasoning_context
from inferenceos.reasoning.corroboration import (
    assess_corroboration,
    cap_evidence_strength,
)
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from tests.reasoning_helpers import (
    insufficient_bundle,
    paging_bundle,
    valid_result_json,
)


def _run(payload: dict[str, object], context: ReasoningDiagnosisContext) -> ReasoningResponse:
    engine = LocalReasoningEngine(FakeReasoningProvider(scripted=[json.dumps(payload)]))
    request = ReasoningRequest(request_id="adv", reason="x", objective="x", context=context)
    return engine.reason(request)


def _base(context: ReasoningDiagnosisContext) -> dict[str, object]:
    evidence = context.evidence
    return {
        "schema_version": "1.0",
        "root_cause_class": "paging",
        "affected_stage": "end_to_end",
        "contributing_factors": ["x"],
        "supporting_evidence_ids": [evidence[0].evidence_id],
        "contradicting_evidence_ids": [],
        "confidence": 0.7,
        "evidence_strength": "medium",
        "uncertainty": "",
        "further_evidence_required": False,
        "summary": "grounded",
        "reasoning": "grounded",
    }


def test_fabricated_evidence_id_is_rejected() -> None:
    context, _, _ = insufficient_bundle()
    payload = _base(context)
    payload["supporting_evidence_ids"] = ["E999"]
    response = _run(payload, context)
    assert response.validation_ok is False
    assert response.status.value == "failed"


def test_fabricated_measurement_is_rejected() -> None:
    context, _, _ = insufficient_bundle()
    payload = _base(context)
    payload["summary"] = "latency collapsed to 12345.6 ms"
    response = _run(payload, context)
    assert response.validation_ok is False


def test_recommendation_is_rejected() -> None:
    context, _, _ = insufficient_bundle()
    payload = _base(context)
    payload["reasoning"] = "You should reduce the context length."
    response = _run(payload, context)
    assert response.validation_ok is False


def test_confidence_inflation_is_capped() -> None:
    context, diagnosis, _ = insufficient_bundle()
    payload = _base(context)
    payload["confidence"] = 0.99
    payload["evidence_strength"] = "high"
    response = _run(payload, context)
    assert response.validation_ok is True
    assessment = assess_corroboration(
        diagnosis, response, ceiling=0.25, evidence_ids={"E1", "E2", "E3"}
    )
    assert assessment.corroborated is True
    assert assessment.confidence_capped is True
    assert assessment.effective_confidence < 0.99
    assert assessment.effective_confidence <= assessment.deterministic_support + 0.25 + 1e-9
    assert assessment.llm_hypothesis_confidence == 0.99


def test_matching_deterministic_finding_allows_bounded_confidence() -> None:
    context, diagnosis, report = paging_bundle()
    engine = LocalReasoningEngine(
        FakeReasoningProvider(scripted=[valid_result_json(context, root_cause_class="paging", confidence=0.9)])
    )
    request = ReasoningRequest(request_id="adv", reason="x", objective="x", context=context)
    response = engine.reason(request)
    assessment = assess_corroboration(
        diagnosis, response, ceiling=0.25, evidence_ids=set()
    )
    assert assessment.corroborated is True
    assert assessment.basis == "matching_deterministic_finding"
    assert assessment.effective_confidence <= assessment.deterministic_support + 0.25


def test_uncorroborated_hypothesis_is_bounded_by_deterministic_support() -> None:
    context, diagnosis, _ = insufficient_bundle()
    payload = _base(context)
    payload["supporting_evidence_ids"] = []
    payload["confidence"] = 0.99
    response = _run(payload, context)
    assessment = assess_corroboration(diagnosis, response, ceiling=0.25, evidence_ids={"E1"})
    assert assessment.corroborated is False
    assert assessment.confidence_capped is True


def test_cap_evidence_strength_downgrades_uncorroborated() -> None:
    assert (
        cap_evidence_strength(
            EvidenceStrength.HIGH, EvidenceStrength.LOW, corroborated=False
        )
        is EvidenceStrength.LOW
    )
    assert (
        cap_evidence_strength(
            EvidenceStrength.HIGH, EvidenceStrength.LOW, corroborated=True
        )
        is EvidenceStrength.HIGH
    )


def test_deep_diagnosis_outcome_records_corroboration() -> None:
    from inferenceos.contracts.enums import BottleneckKind
    from inferenceos.reasoning.lifecycle import run_deep_diagnosis

    context, diagnosis, report = insufficient_bundle()
    engine = LocalReasoningEngine(
        FakeReasoningProvider(
            scripted=[valid_result_json(context, root_cause_class="memory_pressure", confidence=0.95)]
        )
    )
    outcome = run_deep_diagnosis(diagnosis, report=report, engine=engine)
    assert outcome.corroboration is not None
    assert outcome.llm_hypothesis_confidence == 0.95
    assert outcome.corroboration.confidence_capped is True
    assert outcome.final_confidence < 0.95
    assert outcome.final_confidence <= outcome.deterministic_supported_confidence + 0.25 + 1e-9
    assert outcome.confidence_basis == "reasoning_corroborated"
    assert outcome.final_root_cause is BottleneckKind.MEMORY_PRESSURE
