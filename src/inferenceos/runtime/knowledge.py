"""Runtime reuse of Phase 8/9 knowledge (conditional evidence, never authority).

The runtime layer reuses the exact Phase 9 applicability/selection semantics so it
does not invent a parallel knowledge policy: prior results inform candidate handling
but never bypass the registry, preconditions, safety gates, approval, or quality
validation.
"""

from __future__ import annotations

from inferenceos.contracts.knowledge import ApplicabilityContext
from inferenceos.contracts.optimization import OptimizationCandidate
from inferenceos.contracts.optimization_loop import (
    KnowledgeEvidence,
    LoopPolicy,
    OptimizationBudget,
)
from inferenceos.contracts.runtime import RuntimeKnowledgeQuery, RuntimePolicy
from inferenceos.knowledge.service import KnowledgeService
from inferenceos.loop.knowledge import gather_knowledge_evidence, selection_adjustment


def _loop_policy(policy: RuntimePolicy) -> LoopPolicy:
    return LoopPolicy(
        budget=OptimizationBudget(),
        knowledge_lookup_limit=policy.knowledge_lookup_limit,
        skip_directly_negative=policy.skip_directly_negative,
        allow_revalidation=policy.allow_revalidation,
        knowledge_positive_bonus=policy.knowledge_positive_bonus,
        knowledge_negative_penalty=policy.knowledge_negative_penalty,
    )


def query_knowledge(
    candidate: OptimizationCandidate,
    *,
    service: KnowledgeService,
    current_context: ApplicabilityContext,
    policy: RuntimePolicy,
) -> KnowledgeEvidence:
    return gather_knowledge_evidence(
        candidate,
        service=service,
        current_context=current_context,
        policy=_loop_policy(policy),
    )


def knowledge_adjustment(evidence: KnowledgeEvidence, policy: RuntimePolicy) -> float:
    return selection_adjustment(evidence, _loop_policy(policy))


def to_record(evidence: KnowledgeEvidence) -> RuntimeKnowledgeQuery:
    return RuntimeKnowledgeQuery(
        candidate_id=evidence.candidate_id,
        action_id=evidence.action_id,
        effect=evidence.effect,
        records_considered=evidence.records_considered,
        detail=evidence.detail,
    )
