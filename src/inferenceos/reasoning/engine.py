"""Local LLM reasoning engine behind the Phase 0 ``ReasoningEngine`` interface.

The engine builds a compact structured prompt, asks a local provider for a
schema-constrained completion, then strictly parses, validates and grounds the
result. Any failure yields a ``FAILED`` response with the deterministic diagnosis
left untouched — the caller decides how to fall back.
"""

from __future__ import annotations

import time

from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.enums import (
    BottleneckKind,
    EvidenceStrength,
    ReasoningEngineKind,
    ReasoningStatus,
)
from inferenceos.contracts.reasoning import (
    REASONING_SCHEMA_VERSION,
    ReasoningDiagnosisContext,
    ReasoningEvidence,
    ReasoningRequest,
    ReasoningResponse,
)
from inferenceos.interfaces.reasoning_engine import ReasoningEngine
from inferenceos.reasoning.parse import ReasoningParseError, parse_reasoning_result
from inferenceos.reasoning.prompt import (
    build_json_schema,
    render_system_prompt,
    render_user_prompt,
)
from inferenceos.reasoning.providers.base import ReasoningProvider
from inferenceos.reasoning.validate import ReasoningValidationError, validate_reasoning_result

_INSUFFICIENT_KINDS = {
    BottleneckKind.NONE,
    BottleneckKind.AMBIGUOUS,
    BottleneckKind.INSUFFICIENT_EVIDENCE,
}


class LocalReasoningEngine(ReasoningEngine):
    """Provider-agnostic local reasoning engine."""

    def __init__(
        self,
        provider: ReasoningProvider,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: int = 1234,
        timeout_s: float = 300.0,
        meaningful_confidence_threshold: float = 0.5,
        raw_output_limit: int = 8000,
    ) -> None:
        self._provider = provider
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seed = seed
        self._timeout_s = timeout_s
        self._meaningful_threshold = meaningful_confidence_threshold
        self._raw_output_limit = raw_output_limit
        self._last_peak_rss_bytes: int | None = None

    @property
    def kind(self) -> ReasoningEngineKind:
        return ReasoningEngineKind.LLM

    @property
    def name(self) -> str:
        return f"local-reasoner({self._provider.name})"

    @property
    def provider_name(self) -> str:
        return self._provider.name

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    @property
    def last_peak_rss_bytes(self) -> int | None:
        return self._last_peak_rss_bytes

    def is_available(self) -> bool:
        return self._provider.is_available()

    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        context = request.context
        if context is None:
            return self._failed(
                request,
                "missing structured reasoning context",
                duration=None,
                raw=None,
                model_version=None,
            )

        evidence_ids = [item.evidence_id for item in context.evidence]
        schema = build_json_schema(evidence_ids)
        started = time.perf_counter()
        try:
            provider_result = self._provider.complete(
                system_prompt=render_system_prompt(),
                user_prompt=render_user_prompt(context),
                json_schema=schema,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                seed=self._seed,
                timeout_s=self._timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - provider failures must not crash diagnosis
            return self._failed(
                request,
                f"provider raised: {exc}",
                duration=time.perf_counter() - started,
                raw=None,
                model_version=None,
            )
        latency = provider_result.latency_s or (time.perf_counter() - started)
        self._last_peak_rss_bytes = provider_result.peak_rss_bytes

        if not provider_result.success:
            return self._failed(
                request,
                provider_result.error or "reasoning provider failure",
                duration=latency,
                raw=provider_result.text or None,
                model_version=provider_result.model_version,
            )

        try:
            parsed, _stripped = parse_reasoning_result(
                provider_result.text, allowed_evidence_ids=evidence_ids
            )
            validate_reasoning_result(parsed, context)
        except (ReasoningParseError, ReasoningValidationError) as exc:
            return self._failed(
                request,
                f"{type(exc).__name__}: {exc}",
                duration=latency,
                raw=provider_result.text,
                model_version=provider_result.model_version,
            )

        supporting = [_to_evidence_item(item) for item in _resolve(parsed.supporting_evidence_ids, context)]
        contradicting = [
            _to_evidence_item(item) for item in _resolve(parsed.contradicting_evidence_ids, context)
        ]
        is_meaningful = (
            parsed.root_cause_class not in _INSUFFICIENT_KINDS
            and parsed.confidence >= self._meaningful_threshold
            and parsed.evidence_strength in (EvidenceStrength.MEDIUM, EvidenceStrength.HIGH)
            and not parsed.further_evidence_required
        )
        bottleneck = Bottleneck(
            bottleneck_id=f"{request.request_id}-llm",
            kind=parsed.root_cause_class,
            confidence=parsed.confidence,
            is_meaningful=is_meaningful,
            summary=parsed.summary or parsed.uncertainty or "LLM root-cause hypothesis",
            evidence=supporting + contradicting,
            contributing_factors=list(parsed.contributing_factors),
            recommended_surface=[],
            source=f"reasoning.{self._provider.name}",
            stage=parsed.affected_stage,
            subject=context.workload.workload_id,
            evidence_strength=parsed.evidence_strength,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            unavailable_measurements=list(context.unavailable_metrics),
            rationale=[parsed.reasoning] if parsed.reasoning else [],
            detector_version=REASONING_SCHEMA_VERSION,
        )
        status = (
            ReasoningStatus.INSUFFICIENT_EVIDENCE
            if (parsed.further_evidence_required or parsed.root_cause_class in _INSUFFICIENT_KINDS)
            else ReasoningStatus.OK
        )
        return ReasoningResponse(
            request_id=request.request_id,
            status=status,
            engine=self.kind,
            provider=self._provider.name,
            model_used=provider_result.model_id or self._provider.model_id,
            model_version=provider_result.model_version,
            conclusion=parsed.summary or parsed.reasoning,
            bottleneck=bottleneck,
            recommended_candidates=[],
            confidence=parsed.confidence,
            rationale=parsed.reasoning,
            evidence_refs=list(parsed.supporting_evidence_ids) + list(parsed.contradicting_evidence_ids),
            duration_s=latency,
            task_type=request.task_type,
            stage=parsed.affected_stage,
            evidence_strength=parsed.evidence_strength,
            supporting_evidence_refs=list(parsed.supporting_evidence_ids),
            contradicting_evidence_refs=list(parsed.contradicting_evidence_ids),
            uncertainty=parsed.uncertainty,
            further_evidence_required=parsed.further_evidence_required,
            result=parsed,
            validation_ok=True,
            raw_output=_truncate(provider_result.text, self._raw_output_limit),
            escalation_reason=request.escalation_reason,
        )

    def _failed(
        self,
        request: ReasoningRequest,
        failure: str,
        *,
        duration: float | None,
        raw: str | None,
        model_version: str | None,
    ) -> ReasoningResponse:
        return ReasoningResponse(
            request_id=request.request_id,
            status=ReasoningStatus.FAILED,
            engine=self.kind,
            provider=self._provider.name,
            model_used=self._provider.model_id,
            model_version=model_version,
            conclusion="Reasoning unavailable or invalid; deterministic diagnosis retained.",
            bottleneck=None,
            recommended_candidates=[],
            confidence=0.0,
            rationale="",
            evidence_refs=[],
            duration_s=duration,
            task_type=request.task_type,
            validation_ok=False,
            failure=failure,
            raw_output=_truncate(raw, self._raw_output_limit),
            escalation_reason=request.escalation_reason,
        )


def _resolve(ids: list[str], context: ReasoningDiagnosisContext) -> list[ReasoningEvidence]:
    by_id = {item.evidence_id: item for item in context.evidence}
    return [by_id[evidence_id] for evidence_id in ids if evidence_id in by_id]


def _to_evidence_item(item: ReasoningEvidence) -> EvidenceItem:
    return EvidenceItem(
        metric=item.metric,
        observed_value=item.observed_value,
        reference_value=item.reference_value,
        description=item.description,
        polarity=item.polarity,
        stage=item.stage,
        source=item.source,
        evidence_id=item.evidence_id,
    )


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"
