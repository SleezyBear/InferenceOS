"""Deterministic corroboration of LLM reasoning confidence.

The LLM may hypothesise a root cause, but it may not hand the optimizer a
*stronger* diagnosis than the deterministic evidence supports. This module computes
how much of the LLM's stated confidence is corroborated:

* a matching deterministic finding for the same bottleneck kind, or
* explicit citations of evidence ids that actually exist in the supplied context.

A hypothesis that neither matches deterministic findings nor cites supplied
evidence is bounded by the deterministic support (usually zero) plus a small
configurable ceiling, so it cannot be used to inflate an actionable confidence.
"""

from __future__ import annotations

from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import EvidenceStrength
from inferenceos.contracts.reasoning import (
    CorroborationAssessment,
    ReasoningResponse,
)

_STRENGTH_RANK: dict[EvidenceStrength, int] = {
    EvidenceStrength.NONE: 0,
    EvidenceStrength.LOW: 1,
    EvidenceStrength.MEDIUM: 2,
    EvidenceStrength.HIGH: 3,
}


def strength_rank(value: EvidenceStrength) -> int:
    return _STRENGTH_RANK.get(value, 0)


def min_strength(left: EvidenceStrength, right: EvidenceStrength) -> EvidenceStrength:
    return left if strength_rank(left) <= strength_rank(right) else right


def assess_corroboration(
    diagnosis: BottleneckDiagnosis,
    reasoning: ReasoningResponse | None,
    *,
    ceiling: float = 0.25,
    evidence_ids: set[str] | None = None,
) -> CorroborationAssessment:
    """Return the corroboration-limited effective confidence for a reasoning result."""
    findings = _findings(diagnosis)
    deterministic_support = _deterministic_support(findings, diagnosis)

    if reasoning is None or reasoning.result is None:
        return CorroborationAssessment(
            corroborated=False,
            basis="no_reasoning",
            deterministic_support=deterministic_support,
            llm_hypothesis_confidence=0.0,
            effective_confidence=deterministic_support,
            confidence_capped=False,
            notes=["no LLM result to corroborate; deterministic support retained"],
        )

    result = reasoning.result
    llm_confidence = float(result.confidence)
    supporting = list(result.supporting_evidence_ids)
    valid_ids = (
        [item for item in supporting if item in evidence_ids]
        if evidence_ids is not None
        else list(supporting)
    )
    matched_kinds = [
        finding.kind.value
        for finding in findings
        if finding.kind == result.root_cause_class
    ]
    support_for_class = max(
        (
            float(finding.confidence)
            for finding in findings
            if finding.kind == result.root_cause_class
        ),
        default=0.0,
    )

    if support_for_class > 0.0:
        allowed = min(1.0, support_for_class + ceiling)
        basis = "matching_deterministic_finding"
        corroborated = True
        notes = [
            f"LLM root cause {result.root_cause_class.value} matches a deterministic "
            f"finding (support {support_for_class:.2f}); confidence bounded by support + "
            f"ceiling {ceiling:.2f}",
        ]
    elif valid_ids:
        allowed = min(1.0, ceiling)
        basis = "evidence_citations_only"
        corroborated = True
        notes = [
            "LLM cites supplied evidence but does not match a deterministic finding; "
            f"confidence bounded by the corroboration ceiling {ceiling:.2f}",
        ]
    else:
        allowed = deterministic_support
        basis = "uncorroborated_hypothesis"
        corroborated = False
        notes = [
            "LLM hypothesis is neither supported by a deterministic finding nor by any "
            f"supplied evidence id; confidence bounded by deterministic support "
            f"{deterministic_support:.2f}",
        ]

    effective = min(llm_confidence, allowed)
    capped = effective < llm_confidence - 1e-9
    if capped:
        notes.append(
            f"LLM confidence {llm_confidence:.2f} capped to {effective:.2f} "
            "(no unbounded confidence inflation)"
        )
    return CorroborationAssessment(
        corroborated=corroborated,
        basis=basis,
        deterministic_support=support_for_class
        if support_for_class > 0.0
        else deterministic_support,
        llm_hypothesis_confidence=llm_confidence,
        effective_confidence=effective,
        confidence_capped=capped,
        supporting_evidence_ids=valid_ids,
        matched_deterministic_kinds=matched_kinds,
        notes=notes,
    )


def cap_evidence_strength(
    llm_strength: EvidenceStrength,
    deterministic_strength: EvidenceStrength,
    *,
    corroborated: bool,
) -> EvidenceStrength:
    """Prevent an uncorroborated hypothesis from claiming a stronger evidence tier."""
    if corroborated:
        return llm_strength
    return min_strength(llm_strength, deterministic_strength)


def _findings(diagnosis: BottleneckDiagnosis) -> list:
    findings = []
    if diagnosis.primary is not None:
        findings.append(diagnosis.primary)
    findings.extend(diagnosis.candidates)
    findings.extend(diagnosis.observations)
    return findings


def _deterministic_support(findings: list, diagnosis: BottleneckDiagnosis) -> float:
    if diagnosis.primary is not None and diagnosis.meaningful_opportunity:
        return float(diagnosis.primary.confidence)
    return float(diagnosis.confidence)
