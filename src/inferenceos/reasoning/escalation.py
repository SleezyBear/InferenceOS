"""Deterministic escalation gating.

The decision to invoke the local reasoning model is computed from the Phase 4
diagnosis structure — never by the LLM itself. The decision and its reason code are
returned for audit, whether or not escalation happens.
"""

from __future__ import annotations

from inferenceos.contracts.bottleneck import Bottleneck
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import BottleneckKind
from inferenceos.contracts.reasoning import EscalationDecision

_RESOURCE_FAMILY: dict[BottleneckKind, str] = {
    BottleneckKind.CPU: "cpu",
    BottleneckKind.COMPUTE_BOUND: "gpu",
    BottleneckKind.GPU_UNDERUTILIZATION: "gpu",
    BottleneckKind.GPU_MEMORY_LIMIT: "gpu",
    BottleneckKind.MEMORY_PRESSURE: "memory",
    BottleneckKind.MODEL_FOOTPRINT: "memory",
    BottleneckKind.PAGING: "memory",
    BottleneckKind.STORAGE: "storage",
    BottleneckKind.IO_BOUND: "storage",
    BottleneckKind.EXTERNAL_PROCESS: "external",
    BottleneckKind.LATENCY: "latency",
}


def _findings(diagnosis: BottleneckDiagnosis) -> list[Bottleneck]:
    findings: list[Bottleneck] = []
    if diagnosis.primary is not None:
        findings.append(diagnosis.primary)
    findings.extend(diagnosis.candidates)
    findings.extend(diagnosis.observations)
    return findings


def decide_escalation(
    diagnosis: BottleneckDiagnosis,
    *,
    enabled: bool = True,
    confidence_threshold: float = 0.6,
) -> EscalationDecision:
    findings = _findings(diagnosis)
    meaningful = [finding for finding in findings if finding.is_meaningful]
    signals = [finding for finding in findings if finding.supporting_evidence]
    families = {_RESOURCE_FAMILY.get(finding.kind, finding.kind.value) for finding in meaningful}
    best_confidence = max((finding.confidence for finding in findings), default=0.0)

    def decision(escalate: bool, code: str, rationale: str) -> EscalationDecision:
        return EscalationDecision(
            should_escalate=escalate,
            reason_code=code,
            rationale=rationale,
            deterministic_confidence=round(best_confidence, 4),
            candidate_signals=len(signals),
            competing_families=len(families),
            threshold=confidence_threshold,
        )

    if not enabled:
        return decision(False, "disabled", "deep diagnosis is disabled by configuration")

    if (
        diagnosis.meaningful_opportunity
        and diagnosis.primary is not None
        and not diagnosis.insufficient_evidence
        and len(families) <= 1
        and diagnosis.primary.confidence >= confidence_threshold
    ):
        return decision(
            False,
            "strong_deterministic",
            "deterministic evidence is sufficient and above the confidence threshold; "
            "the reasoning model must not override it",
        )

    if len(families) >= 2:
        return decision(
            True,
            "competing_hypotheses",
            f"{len(families)} resource families are comparably supported; the model may "
            "resolve the interaction",
        )

    if diagnosis.insufficient_evidence and signals:
        return decision(
            True,
            "insufficient_evidence",
            "deterministic evidence is insufficient and at least one candidate signal "
            "exists that reasoning can interpret",
        )

    if (
        diagnosis.primary is not None
        and diagnosis.primary.confidence < confidence_threshold
        and signals
    ):
        return decision(
            True,
            "confidence_gap",
            "deterministic confidence is below the configured threshold",
        )

    if not signals:
        return decision(
            False,
            "no_signal",
            "no candidate signal carries supporting evidence; reasoning cannot resolve "
            "missing telemetry",
        )

    return decision(
        False,
        "deterministic_sufficient",
        "existing deterministic evidence is sufficient; no deeper reasoning is warranted",
    )
