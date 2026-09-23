"""A deterministic stub ReasoningEngine.

Phase 0's job is the *interface*, not the reasoning logic. The stub exists to prove
that a real ``ReasoningRequest`` round-trips to a schema-valid
``ReasoningResponse`` through the abstract interface. It never invokes an LLM.

Note the anti-triviality behaviour even here: with no real deterministic findings it
returns an explicit "no meaningful bottleneck" conclusion rather than inventing one.
"""

from __future__ import annotations

import time

from inferenceos.contracts.bottleneck import Bottleneck
from inferenceos.contracts.enums import BottleneckKind, ReasoningEngineKind, ReasoningStatus
from inferenceos.contracts.reasoning import ReasoningRequest, ReasoningResponse
from inferenceos.interfaces.reasoning_engine import ReasoningEngine


class StubReasoningEngine(ReasoningEngine):
    """Interface-conformance stub (no model, no external calls)."""

    def __init__(self, name: str = "stub-reasoner") -> None:
        self._name = name

    @property
    def kind(self) -> ReasoningEngineKind:
        return ReasoningEngineKind.STUB

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        started = time.perf_counter()

        if request.deterministic_findings:
            primary = max(
                request.deterministic_findings, key=lambda finding: finding.confidence
            )
            conclusion = (
                "Stub echo: highest-confidence deterministic finding is "
                f"'{primary.kind.value}' (confidence {primary.confidence:.2f})."
            )
            bottleneck: Bottleneck | None = primary
            status = ReasoningStatus.OK
        else:
            conclusion = (
                "Stub echo: no deterministic findings supplied; asserting "
                "'no meaningful bottleneck' rather than inventing one."
            )
            bottleneck = Bottleneck(
                bottleneck_id=f"{request.request_id}-none",
                kind=BottleneckKind.NONE,
                confidence=0.0,
                is_meaningful=False,
                summary="No meaningful optimization opportunity.",
                evidence=[],
                contributing_factors=[],
                recommended_surface=[],
                source=self._name,
            )
            status = ReasoningStatus.INSUFFICIENT_EVIDENCE

        duration = time.perf_counter() - started
        return ReasoningResponse(
            request_id=request.request_id,
            status=status,
            engine=self.kind,
            model_used=self._name,
            conclusion=conclusion,
            bottleneck=bottleneck,
            recommended_candidates=[],
            confidence=0.0,
            rationale=(
                "Phase 0 stub: interface round-trip only; no LLM inference performed."
            ),
            evidence_refs=[finding.bottleneck_id for finding in request.deterministic_findings],
            duration_s=duration,
        )
