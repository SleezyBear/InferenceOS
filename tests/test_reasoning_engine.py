"""Engine + lifecycle: success, malformed/hallucinated output, failures, fallback."""

from __future__ import annotations

import json

from inferenceos.contracts.enums import BottleneckKind, ReasoningStatus
from inferenceos.contracts.reasoning import (
    ReasoningDiagnosisContext,
    ReasoningRequest,
)
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.lifecycle import run_deep_diagnosis
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from inferenceos.reasoning.providers.llama_cpp import LlamaCppReasoningProvider
from tests.reasoning_helpers import (
    insufficient_bundle,
    paging_bundle,
    valid_result_json,
)


def _engine(provider: FakeReasoningProvider) -> LocalReasoningEngine:
    return LocalReasoningEngine(provider, max_tokens=256, temperature=0.0, seed=1)


def _request(context: ReasoningDiagnosisContext, request_id: str = "req-1") -> ReasoningRequest:
    return ReasoningRequest(
        request_id=request_id,
        reason="insufficient_evidence",
        objective="root cause",
        escalation_reason="insufficient_evidence",
        context=context,
    )


def test_engine_success() -> None:
    context, _, _ = paging_bundle()
    engine = _engine(FakeReasoningProvider(scripted=[valid_result_json(context)]))
    response = engine.reason(_request(context))
    assert response.status is ReasoningStatus.OK
    assert response.validation_ok is True
    assert response.result is not None
    assert response.bottleneck is not None
    assert response.bottleneck.kind is BottleneckKind.PAGING
    assert response.bottleneck.recommended_surface == []
    assert response.recommended_candidates == []


def test_engine_strips_recommendation_keys() -> None:
    context, _, _ = paging_bundle()
    payload = json.loads(valid_result_json(context))
    payload["recommendations"] = ["reduce the context size"]
    engine = _engine(FakeReasoningProvider(scripted=[json.dumps(payload)]))
    response = engine.reason(_request(context))
    assert response.validation_ok is True
    assert response.result is not None
    assert response.recommended_candidates == []


def test_engine_malformed_json_fails() -> None:
    context, _, _ = paging_bundle()
    engine = _engine(FakeReasoningProvider(scripted=["not json at all"]))
    response = engine.reason(_request(context))
    assert response.status is ReasoningStatus.FAILED
    assert response.validation_ok is False
    assert response.failure is not None


def test_engine_schema_invalid_fails() -> None:
    context, _, _ = paging_bundle()
    payload = json.loads(valid_result_json(context))
    payload["root_cause_class"] = "not_a_cause"
    engine = _engine(FakeReasoningProvider(scripted=[json.dumps(payload)]))
    response = engine.reason(_request(context))
    assert response.status is ReasoningStatus.FAILED


def test_engine_hallucinated_evidence_fails() -> None:
    context, _, _ = paging_bundle()
    engine = _engine(
        FakeReasoningProvider(scripted=[valid_result_json(context, supporting=["E999"])])
    )
    response = engine.reason(_request(context))
    assert response.status is ReasoningStatus.FAILED
    assert response.failure is not None and "evidence" in response.failure.lower()


def test_engine_fabricated_measurement_fails() -> None:
    context, _, _ = paging_bundle()
    engine = _engine(
        FakeReasoningProvider(
            scripted=[
                valid_result_json(
                    context, summary="GPU utilization was 98% during the run."
                )
            ]
        )
    )
    response = engine.reason(_request(context))
    assert response.status is ReasoningStatus.FAILED
    assert response.failure is not None and "ungrounded" in response.failure.lower()


def test_engine_provider_failure() -> None:
    context, _, _ = paging_bundle()
    engine = _engine(FakeReasoningProvider(fail_with="model timed out"))
    response = engine.reason(_request(context))
    assert response.status is ReasoningStatus.FAILED
    assert response.failure == "model timed out"


def test_engine_missing_context_fails() -> None:
    context, _, _ = paging_bundle()
    engine = _engine(FakeReasoningProvider(scripted=[valid_result_json(context)]))
    from inferenceos.contracts.reasoning import ReasoningRequest

    request = ReasoningRequest(request_id="req", reason="r", objective="o")
    response = engine.reason(request)
    assert response.status is ReasoningStatus.FAILED
    assert response.failure is not None and "context" in response.failure


def test_engine_provider_raises_is_contained() -> None:
    class ExplodingProvider:
        name = "exploding"
        model_id = "boom"

        def is_available(self) -> bool:
            return True

        def complete(self, **kwargs: object) -> object:
            raise RuntimeError("boom")

    context, _, _ = paging_bundle()
    engine = LocalReasoningEngine(ExplodingProvider())  # type: ignore[arg-type]
    response = engine.reason(_request(context))
    assert response.status is ReasoningStatus.FAILED
    assert response.failure is not None and "provider raised" in response.failure


def test_lifecycle_strong_deterministic_does_not_escalate() -> None:
    context, diagnosis, report = paging_bundle()
    engine = _engine(FakeReasoningProvider(scripted=[valid_result_json(context)]))
    outcome = run_deep_diagnosis(diagnosis, report=report, engine=engine)
    assert outcome.escalation.should_escalate is False
    assert outcome.reasoning is None
    assert outcome.lifecycle.reasoning_performed is False
    assert outcome.final_root_cause is BottleneckKind.PAGING


def test_lifecycle_escalates_and_uses_valid_reasoning() -> None:
    context, diagnosis, report = insufficient_bundle()
    engine = _engine(
        FakeReasoningProvider(scripted=[valid_result_json(context, root_cause_class="memory_pressure", confidence=0.7, evidence_strength="medium")])
    )
    outcome = run_deep_diagnosis(diagnosis, report=report, engine=engine)
    assert outcome.escalation.should_escalate is True
    assert outcome.lifecycle.reasoning_performed is True
    assert outcome.lifecycle.reasoner_cleanup_ok is True
    assert outcome.reasoning is not None and outcome.reasoning.validation_ok
    assert outcome.final_root_cause is BottleneckKind.MEMORY_PRESSURE
    assert outcome.fallback_used is False


def test_lifecycle_failure_falls_back_to_deterministic() -> None:
    context, diagnosis, report = insufficient_bundle()
    engine = _engine(FakeReasoningProvider(fail_with="provider unavailable"))
    outcome = run_deep_diagnosis(diagnosis, report=report, engine=engine)
    assert outcome.escalation.should_escalate is True
    assert outcome.fallback_used is True
    assert outcome.reasoning is not None and outcome.reasoning.status is ReasoningStatus.FAILED
    # Deterministic diagnosis is retained.
    assert outcome.deterministic == diagnosis


def test_lifecycle_unavailable_engine_falls_back() -> None:
    _, diagnosis, report = insufficient_bundle()
    broken = LocalReasoningEngine(
        LlamaCppReasoningProvider(model_path="<missing>.gguf")
    )
    outcome = run_deep_diagnosis(diagnosis, report=report, engine=broken)
    assert outcome.escalation.should_escalate is True
    assert outcome.fallback_used is True
    assert outcome.reasoning is not None and outcome.reasoning.status is ReasoningStatus.FAILED
