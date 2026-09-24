"""Knowledge-aware candidate selection.

Historical Phase 8 records are *conditional evidence*, never commands. For each
candidate the loop classifies prior outcomes against the current context and derives an
explainable selection effect: preferred, deprioritized, requires-revalidation, rejected,
or unavailable. A stored action id can never make an unregistered action executable.
"""

from __future__ import annotations

from inferenceos.contracts.enums import (
    ApplicabilityClass,
    KnowledgeAvailability,
    KnowledgeSelectionEffect,
    OptimizationOutcome,
    RevalidationTrigger,
)
from inferenceos.contracts.knowledge import ApplicabilityContext
from inferenceos.contracts.optimization import OptimizationCandidate
from inferenceos.contracts.optimization_loop import KnowledgeEvidence, LoopPolicy
from inferenceos.knowledge.service import KnowledgeService
from inferenceos.knowledge.store import KnowledgeError

NEGATIVE_OUTCOMES = {
    OptimizationOutcome.NO_MEANINGFUL_IMPROVEMENT,
    OptimizationOutcome.REGRESSION,
    OptimizationOutcome.QUALITY_REGRESSION,
    OptimizationOutcome.INVALID,
    OptimizationOutcome.FAILED,
    OptimizationOutcome.BLOCKED,
    OptimizationOutcome.REVERTED,
}

DIRECT_APPLICABLE = {
    ApplicabilityClass.DIRECTLY_APPLICABLE,
    ApplicabilityClass.CONDITIONALLY_APPLICABLE,
}


def unavailable_evidence(candidate: OptimizationCandidate, detail: str) -> KnowledgeEvidence:
    return KnowledgeEvidence(
        candidate_id=candidate.candidate_id,
        action_id=candidate.action_id,
        effect=KnowledgeSelectionEffect.UNAVAILABLE,
        availability=KnowledgeAvailability.UNAVAILABLE,
        detail=detail,
    )


def gather_knowledge_evidence(
    candidate: OptimizationCandidate,
    *,
    service: KnowledgeService,
    current_context: ApplicabilityContext,
    policy: LoopPolicy,
) -> KnowledgeEvidence:
    """Consult the durable store for one candidate; never raises for store failures."""
    try:
        results = service.query_applicable(
            candidate.action_id, current_context, limit=policy.knowledge_lookup_limit
        )
    except KnowledgeError as exc:
        return unavailable_evidence(candidate, f"knowledge store unavailable: {exc}")

    classifications: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    record_ids: list[str] = []
    applicable_positive = 0
    applicable_negative = 0
    directly_negative = 0
    conditionally_negative = 0
    requires_revalidation = False
    triggers: list[RevalidationTrigger] = []
    scores: list[float] = []

    for result in results:
        classifications[result.classification.value] = (
            classifications.get(result.classification.value, 0) + 1
        )
        outcomes[result.record_outcome.value] = outcomes.get(result.record_outcome.value, 0) + 1
        record_ids.append(result.record_id)
        if result.score is not None:
            scores.append(result.score)
        negative = result.record_outcome in NEGATIVE_OUTCOMES
        if result.classification in DIRECT_APPLICABLE and not negative:
            applicable_positive += 1
        if result.classification in DIRECT_APPLICABLE and negative:
            applicable_negative += 1
        if result.classification is ApplicabilityClass.DIRECTLY_APPLICABLE and negative:
            directly_negative += 1
        if result.classification is ApplicabilityClass.CONDITIONALLY_APPLICABLE and negative:
            conditionally_negative += 1
        if result.classification is ApplicabilityClass.REQUIRES_REVALIDATION:
            requires_revalidation = True
        for trigger in result.revalidation_triggers:
            if trigger not in triggers:
                triggers.append(trigger)

    effect = _effect(
        policy=policy,
        records=len(results),
        applicable_positive=applicable_positive,
        directly_negative=directly_negative,
        conditionally_negative=conditionally_negative,
        requires_revalidation=requires_revalidation,
    )
    detail = _detail(effect, len(results), applicable_positive, applicable_negative, requires_revalidation)
    return KnowledgeEvidence(
        candidate_id=candidate.candidate_id,
        action_id=candidate.action_id,
        effect=effect,
        availability=KnowledgeAvailability.OK,
        records_considered=len(results),
        classifications=classifications,
        outcomes=outcomes,
        applicable_positive=applicable_positive,
        applicable_negative=applicable_negative,
        requires_revalidation=requires_revalidation,
        revalidation_triggers=triggers,
        best_score=round(max(scores), 4) if scores else None,
        record_ids=record_ids,
        detail=detail,
    )


def _effect(
    *,
    policy: LoopPolicy,
    records: int,
    applicable_positive: int,
    directly_negative: int,
    conditionally_negative: int,
    requires_revalidation: bool,
) -> KnowledgeSelectionEffect:
    if records == 0:
        return KnowledgeSelectionEffect.NEUTRAL
    if directly_negative:
        if policy.skip_directly_negative:
            return KnowledgeSelectionEffect.REJECTED
        return KnowledgeSelectionEffect.DEPRIORITIZED
    if requires_revalidation:
        if policy.allow_revalidation:
            return KnowledgeSelectionEffect.REVALIDATION_REQUIRED
        return KnowledgeSelectionEffect.DEPRIORITIZED
    if conditionally_negative:
        return KnowledgeSelectionEffect.DEPRIORITIZED
    if applicable_positive:
        return KnowledgeSelectionEffect.PREFERRED
    return KnowledgeSelectionEffect.NEUTRAL


def _detail(
    effect: KnowledgeSelectionEffect,
    record_count: int,
    applicable_positive: int,
    applicable_negative: int,
    requires_revalidation: bool,
) -> str:
    if record_count == 0:
        return "no prior records for this action"
    parts = [
        f"records={record_count}",
        f"applicable_positive={applicable_positive}",
        f"applicable_negative={applicable_negative}",
        f"requires_revalidation={requires_revalidation}",
        f"effect={effect.value}",
    ]
    return "; ".join(parts)


def selection_adjustment(evidence: KnowledgeEvidence, policy: LoopPolicy) -> float:
    """Deterministic, explainable score delta contributed by historical evidence."""
    if evidence.effect is KnowledgeSelectionEffect.PREFERRED:
        return policy.knowledge_positive_bonus
    if evidence.effect is KnowledgeSelectionEffect.DEPRIORITIZED:
        return -policy.knowledge_negative_penalty
    if evidence.effect is KnowledgeSelectionEffect.REVALIDATION_REQUIRED:
        return -policy.knowledge_negative_penalty * 0.5
    return 0.0
