"""Deep-diagnosis lifecycle: escalation -> reasoning -> validation -> fallback.

The target model is process-per-generation in this backend, so there is no resident
target to release; the lifecycle still makes the reasoning cost explicit and
preserves the deterministic diagnosis when reasoning is unavailable or invalid.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from inferenceos.contracts.bottleneck import Bottleneck
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    BottleneckKind,
    EvidenceStrength,
    InferenceStage,
    ReasoningEngineKind,
    ReasoningStatus,
)
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.reasoning import (
    DeepDiagnosisOutcome,
    ReasoningLifecycle,
    ReasoningRequest,
    ReasoningResponse,
    ReasoningTaskType,
)
from inferenceos.interfaces.reasoning_engine import ReasoningEngine
from inferenceos.reasoning.context import DEFAULT_OBJECTIVE, build_reasoning_context
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.escalation import decide_escalation

_INSUFFICIENT_KINDS = {
    BottleneckKind.NONE,
    BottleneckKind.AMBIGUOUS,
    BottleneckKind.INSUFFICIENT_EVIDENCE,
}

TARGET_STATE = "process_per_generation_no_resident_target"


def run_deep_diagnosis(
    diagnosis: BottleneckDiagnosis,
    *,
    report: BaselineReport | None = None,
    engine: ReasoningEngine | LocalReasoningEngine | None = None,
    enabled: bool = True,
    confidence_threshold: float = 0.6,
    objective: str = DEFAULT_OBJECTIVE,
    max_evidence: int = 12,
    target_resume_verified: bool | None = None,
) -> DeepDiagnosisOutcome:
    escalation = decide_escalation(
        diagnosis, enabled=enabled, confidence_threshold=confidence_threshold
    )
    lifecycle = ReasoningLifecycle(
        reasoning_performed=False,
        target_state=TARGET_STATE,
        target_released=False,
        target_resume_verified=target_resume_verified,
        notes=[
            "backend is process-per-generation: no resident target model to release "
            "while the reasoner runs",
        ],
    )
    reasoning: ReasoningResponse | None = None
    fallback_used = False

    if escalation.should_escalate:
        request = ReasoningRequest(
            request_id=f"reason-{uuid.uuid4().hex[:12]}",
            task_type=ReasoningTaskType.ROOT_CAUSE_DIAGNOSIS,
            reason=escalation.reason_code,
            objective=objective,
            provider=getattr(engine, "provider_name", None),
            model_ref=getattr(engine, "model_id", None),
            escalation_reason=escalation.reason_code,
            constraints=[
                "no optimization recommendations",
                "ground every claim in the supplied evidence ids",
                "do not invent unavailable measurements",
            ],
            output_schema_hint="ReasoningResult v1",
        )
        context = build_reasoning_context(
            diagnosis, report, objective=objective, max_evidence=max_evidence
        )
        request = request.model_copy(update={"context": context})
        if engine is None or not engine.is_available():
            reasoning = _unavailable_reasoning(request, engine, "reasoning engine unavailable")
            fallback_used = True
            lifecycle = lifecycle.model_copy(
                update={
                    "provider": getattr(engine, "provider_name", None),
                    "model": getattr(engine, "model_id", None),
                    "notes": [*lifecycle.notes, "reasoning engine unavailable; skipped invocation"],
                }
            )
        else:
            reasoning = engine.reason(request)
            peak_rss = getattr(engine, "last_peak_rss_bytes", None)
            lifecycle = lifecycle.model_copy(
                update={
                    "reasoning_performed": True,
                    "provider": reasoning.provider,
                    "model": reasoning.model_used,
                    "reasoning_latency_s": reasoning.duration_s,
                    "peak_reasoner_rss_bytes": peak_rss,
                    "reasoner_cleanup_ok": True,
                    "notes": [
                        *lifecycle.notes,
                        f"reasoning status={reasoning.status.value}",
                    ],
                }
            )
            if reasoning.status is ReasoningStatus.FAILED or not reasoning.validation_ok:
                fallback_used = True

    (
        final_root,
        final_stage,
        final_confidence,
        final_strength,
        uncertainty,
        used_reasoning,
    ) = _resolve_final(diagnosis, reasoning)

    limitations = list(diagnosis.limitations)
    if reasoning is not None and reasoning.status is ReasoningStatus.FAILED:
        limitations.append(f"reasoning failed: {reasoning.failure}")
    if not escalation.should_escalate:
        limitations.append(f"deep diagnosis not invoked ({escalation.reason_code})")

    summary = _summary(diagnosis, escalation.reason_code, reasoning, final_root, used_reasoning)

    return DeepDiagnosisOutcome(
        outcome_id=f"deep-{uuid.uuid4().hex[:12]}",
        created_at=datetime.now(timezone.utc),
        source_diagnosis_id=diagnosis.diagnosis_id,
        source_report_path=diagnosis.context.source_report_path,
        escalation=escalation,
        deterministic=diagnosis,
        reasoning=reasoning,
        lifecycle=lifecycle,
        fallback_used=fallback_used,
        final_root_cause=final_root,
        final_stage=final_stage,
        final_confidence=final_confidence,
        final_evidence_strength=final_strength,
        uncertainty=uncertainty,
        limitations=limitations,
        summary=summary,
    )


def _unavailable_reasoning(
    request: ReasoningRequest,
    engine: ReasoningEngine | LocalReasoningEngine | None,
    failure: str,
) -> ReasoningResponse:
    kind = engine.kind if engine is not None else ReasoningEngineKind.LLM
    return ReasoningResponse(
        request_id=request.request_id,
        status=ReasoningStatus.FAILED,
        engine=kind,
        provider=getattr(engine, "provider_name", None),
        model_used=getattr(engine, "model_id", None),
        conclusion="Reasoning unavailable; deterministic diagnosis retained.",
        recommended_candidates=[],
        confidence=0.0,
        duration_s=None,
        task_type=request.task_type,
        validation_ok=False,
        failure=failure,
        raw_output=None,
        escalation_reason=request.escalation_reason,
    )


def _resolve_final(
    diagnosis: BottleneckDiagnosis,
    reasoning: ReasoningResponse | None,
) -> tuple[
    BottleneckKind | None,
    InferenceStage | None,
    float,
    EvidenceStrength,
    str | None,
    bool,
]:
    if (
        reasoning is not None
        and reasoning.status is not ReasoningStatus.FAILED
        and reasoning.validation_ok
        and reasoning.result is not None
        and reasoning.result.root_cause_class not in _INSUFFICIENT_KINDS
        and not reasoning.result.further_evidence_required
    ):
        result = reasoning.result
        return (
            result.root_cause_class,
            result.affected_stage,
            result.confidence,
            result.evidence_strength,
            result.uncertainty or None,
            True,
        )

    primary: Bottleneck | None = diagnosis.primary
    if primary is not None:
        uncertainty = "deterministic diagnosis retained"
        if reasoning is not None and reasoning.failure:
            uncertainty = f"deterministic diagnosis retained; reasoning unavailable ({reasoning.failure})"
        return (
            primary.kind,
            primary.stage,
            primary.confidence,
            primary.evidence_strength,
            uncertainty,
            False,
        )
    return (
        None,
        None,
        0.0,
        EvidenceStrength.NONE,
        "no confident root cause; deterministic evidence insufficient",
        False,
    )


def _summary(
    diagnosis: BottleneckDiagnosis,
    escalation_code: str,
    reasoning: ReasoningResponse | None,
    final_root: BottleneckKind | None,
    used_reasoning: bool,
) -> str:
    if used_reasoning and reasoning is not None:
        root = final_root.value if final_root is not None else "unknown"
        return (
            f"Deep diagnosis via {reasoning.provider}: root_cause={root} "
            f"(confidence {reasoning.confidence:.2f}, {reasoning.evidence_strength.value})."
        )
    if reasoning is not None and reasoning.status is ReasoningStatus.FAILED:
        return (
            f"Deep diagnosis unavailable ({reasoning.failure}); deterministic diagnosis "
            f"retained: {diagnosis.summary}"
        )
    if final_root is not None:
        return (
            f"Deterministic diagnosis retained ({escalation_code}): {final_root.value}. "
            f"{diagnosis.summary}"
        )
    return f"No confident root cause ({escalation_code}); {diagnosis.summary}"
