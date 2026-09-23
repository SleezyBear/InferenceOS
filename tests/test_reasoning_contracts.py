"""Reasoning contract construction and JSON round-trips."""

from __future__ import annotations

from inferenceos.contracts.enums import (
    BottleneckKind,
    EvidenceStrength,
    InferenceStage,
    ReasoningEngineKind,
    ReasoningStatus,
    ReasoningTaskType,
)
from inferenceos.contracts.reasoning import (
    DeepDiagnosisOutcome,
    ReasoningRequest,
    ReasoningResponse,
    ReasoningResult,
)
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.lifecycle import run_deep_diagnosis
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from tests.reasoning_helpers import insufficient_bundle, valid_result_json


def _engine(text: str) -> LocalReasoningEngine:
    return LocalReasoningEngine(FakeReasoningProvider(scripted=[text]))


def test_reasoning_result_roundtrip() -> None:
    result = ReasoningResult(
        root_cause_class=BottleneckKind.PAGING,
        affected_stage=InferenceStage.END_TO_END,
        confidence=0.7,
        evidence_strength=EvidenceStrength.MEDIUM,
    )
    restored = ReasoningResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_reasoning_request_roundtrip() -> None:
    context, _, _ = insufficient_bundle()
    request = ReasoningRequest(
        request_id="req-1",
        reason="insufficient_evidence",
        objective="root cause",
        task_type=ReasoningTaskType.ROOT_CAUSE_DIAGNOSIS,
        context=context,
    )
    restored = ReasoningRequest.model_validate_json(request.model_dump_json())
    assert restored == request
    assert restored.context is not None
    assert restored.context.evidence


def test_deep_diagnosis_outcome_roundtrip() -> None:
    context, diagnosis, report = insufficient_bundle()
    engine = _engine(valid_result_json(context, root_cause_class="memory_pressure"))
    outcome = run_deep_diagnosis(diagnosis, report=report, engine=engine)
    restored = DeepDiagnosisOutcome.from_json(outcome.to_json())
    assert restored == outcome
    assert restored.reasoning is not None
    assert restored.reasoning.engine is ReasoningEngineKind.LLM


def test_reasoning_response_has_no_recommendations() -> None:
    context, diagnosis, report = insufficient_bundle()
    engine = _engine(valid_result_json(context))
    outcome = run_deep_diagnosis(diagnosis, report=report, engine=engine)
    assert outcome.reasoning is not None
    response: ReasoningResponse = outcome.reasoning
    assert response.recommended_candidates == []
    assert response.bottleneck is not None
    assert response.bottleneck.recommended_surface == []
    assert response.status in (ReasoningStatus.OK, ReasoningStatus.INSUFFICIENT_EVIDENCE)
