"""Real local-provider integration (one real reasoner invocation).

Skipped unless a real report, the llama.cpp toolchain, and the configured
reasoning model are present. This is a real invocation, not a mock.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.config import Settings
from inferenceos.contracts.enums import ReasoningEngineKind, ReasoningStatus
from inferenceos.contracts.reasoning import ReasoningRequest
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.diagnosis.loader import load_baseline_report
from inferenceos.reasoning.context import build_reasoning_context
from inferenceos.reasoning.factory import build_reasoning_engine

_REPO = Path(__file__).resolve().parents[1]
_REPORTS = sorted(
    (path for path in (_REPO / "reports").glob("phase3_verification_*.json")),
    key=lambda item: item.stat().st_mtime,
)
_REPORT = _REPORTS[-1] if _REPORTS else None
_SETTINGS = Settings()
_ENGINE = build_reasoning_engine(_SETTINGS) if _REPORT is not None else None

pytestmark = pytest.mark.skipif(
    _REPORT is None or _ENGINE is None or not _ENGINE.is_available(),
    reason="real reasoning model/toolchain not available",
)


def test_real_reasoning_invocation_is_grounded_and_valid() -> None:
    assert _REPORT is not None and _ENGINE is not None
    report = load_baseline_report(_REPORT)
    diagnosis = BottleneckDetector().detect(report, source_report_path=str(_REPORT))
    context = build_reasoning_context(diagnosis, report)
    assert context.evidence, "expected grounded evidence in the real context"
    allowed = {item.evidence_id for item in context.evidence}

    response = _ENGINE.reason(
        ReasoningRequest(
            request_id="integration-1",
            reason="test",
            objective="identify the most likely root cause",
            context=context,
        )
    )

    assert _ENGINE.kind is ReasoningEngineKind.LLM
    assert _ENGINE.model_id
    assert response.status is not ReasoningStatus.FAILED, response.failure
    assert response.validation_ok is True
    assert response.result is not None
    assert (
        set(response.result.supporting_evidence_ids)
        | set(response.result.contradicting_evidence_ids)
    ) <= allowed
    assert response.recommended_candidates == []
