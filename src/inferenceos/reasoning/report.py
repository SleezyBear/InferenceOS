"""Human-readable rendering of a deep-diagnosis outcome."""

from __future__ import annotations

from inferenceos.contracts.reasoning import (
    DeepDiagnosisOutcome,
    ReasoningResponse,
)


def render_deep_diagnosis(outcome: DeepDiagnosisOutcome) -> str:
    lines: list[str] = []
    lines.append("=== Deep diagnosis outcome ===")
    lines.append(f"outcome        : {outcome.outcome_id}")
    lines.append(f"source report  : {outcome.source_report_path}")
    lines.append(f"detector       : {outcome.deterministic.context.detector_version}")
    lines.append("")
    lines.append("--- Deterministic escalation decision ---")
    lines.append(f"escalate       : {outcome.escalation.should_escalate} ({outcome.escalation.reason_code})")
    lines.append(f"rationale      : {outcome.escalation.rationale}")
    lines.append(
        f"signals        : {outcome.escalation.candidate_signals} "
        f"competing_families={outcome.escalation.competing_families} "
        f"threshold={outcome.escalation.threshold:.2f}"
    )
    lines.append("")
    lines.append("--- Deterministic diagnosis ---")
    lines.append(f"{outcome.deterministic.summary}")
    lines.append("")
    if outcome.reasoning is not None:
        lines.append("--- Local reasoning ---")
        _render_response(lines, outcome.reasoning)
        lines.append("")
    lines.append("--- Lifecycle / cost ---")
    lifecycle = outcome.lifecycle
    lines.append(f"reasoning performed : {lifecycle.reasoning_performed}")
    lines.append(f"provider / model    : {lifecycle.provider} / {lifecycle.model}")
    lines.append(f"target state        : {lifecycle.target_state}")
    lines.append(f"reasoning latency   : {lifecycle.reasoning_latency_s}")
    lines.append(f"peak reasoner RSS   : {lifecycle.peak_reasoner_rss_bytes}")
    lines.append(f"cleanup ok          : {lifecycle.reasoner_cleanup_ok}")
    lines.append(f"target resume ok    : {lifecycle.target_resume_verified}")
    lines.append("")
    lines.append("--- Final result ---")
    lines.append(f"root cause      : {outcome.final_root_cause.value if outcome.final_root_cause else None}")
    lines.append(f"stage           : {outcome.final_stage.value if outcome.final_stage else None}")
    lines.append(f"confidence      : {outcome.final_confidence:.2f} ({outcome.final_evidence_strength.value})")
    lines.append(f"fallback used   : {outcome.fallback_used}")
    lines.append(f"uncertainty     : {outcome.uncertainty}")
    lines.append(f"summary         : {outcome.summary}")
    if outcome.limitations:
        lines.append("--- Limitations ---")
        for item in outcome.limitations:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def _render_response(lines: list[str], response: ReasoningResponse) -> None:
    lines.append(
        f"status={response.status.value} provider={response.provider} "
        f"model={response.model_used} version={response.model_version} "
        f"validation_ok={response.validation_ok}"
    )
    if response.failure:
        lines.append(f"failure         : {response.failure}")
    if response.result is not None:
        result = response.result
        lines.append(
            f"hypothesis      : {result.root_cause_class.value} "
            f"stage={result.affected_stage.value if result.affected_stage else None} "
            f"confidence={result.confidence:.2f} strength={result.evidence_strength.value}"
        )
        lines.append(f"summary         : {result.summary}")
        lines.append(f"reasoning       : {result.reasoning}")
        lines.append(f"uncertainty     : {result.uncertainty}")
        lines.append(
            f"evidence        : supporting={result.supporting_evidence_ids} "
            f"contradicting={result.contradicting_evidence_ids}"
        )
        lines.append(f"further evidence required: {result.further_evidence_required}")
    if response.raw_output:
        lines.append("--- Raw model output (audit) ---")
        lines.append(response.raw_output)
