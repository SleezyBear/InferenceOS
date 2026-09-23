"""ReasoningEngine interface tests (stub round-trip; anti-triviality behaviour)."""

from __future__ import annotations

from inferenceos.contracts.enums import BottleneckKind, ReasoningEngineKind, ReasoningStatus
from inferenceos.contracts.reasoning import ReasoningRequest, ReasoningResponse
from inferenceos.reasoning.stub import StubReasoningEngine
from inferenceos.selftest import run_reasoning_selftest, sample_reasoning_request


def test_stub_roundtrip_ok() -> None:
    response = run_reasoning_selftest()
    assert isinstance(response, ReasoningResponse)
    assert response.status is ReasoningStatus.OK
    assert response.engine is ReasoningEngineKind.STUB


def test_stub_echoes_request_id_and_finding() -> None:
    request = sample_reasoning_request()
    response = StubReasoningEngine().reason(request)
    assert response.request_id == request.request_id
    assert response.bottleneck is not None
    assert response.bottleneck.bottleneck_id == request.deterministic_findings[0].bottleneck_id


def test_stub_asserts_no_meaningful_bottleneck_when_no_findings() -> None:
    request = ReasoningRequest(
        request_id="req-none",
        reason="test",
        objective="test",
        deterministic_findings=[],
    )
    response = StubReasoningEngine().reason(request)
    assert response.status is ReasoningStatus.INSUFFICIENT_EVIDENCE
    assert response.bottleneck is not None
    assert response.bottleneck.kind is BottleneckKind.NONE
    assert response.bottleneck.is_meaningful is False


def test_stub_is_available_and_named() -> None:
    engine = StubReasoningEngine(name="my-stub")
    assert engine.is_available() is True
    assert engine.name == "my-stub"
    assert engine.kind is ReasoningEngineKind.STUB
