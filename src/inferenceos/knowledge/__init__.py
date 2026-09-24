"""Phase 8 persistent knowledge & rollback.

Turn Phase 7 ``ExperimentResult`` objects into durable, contextual
``OptimizationRecord`` / ``RollbackRecord`` entries; retrieve them deterministically;
and classify how well historical evidence applies to the *current* context rather than
treating past results as universal truth.
"""

from __future__ import annotations

from inferenceos.knowledge.applicability import evaluate_applicability
from inferenceos.knowledge.fingerprint import (
    build_context_from_request,
    build_current_context,
    fingerprint_context,
)
from inferenceos.knowledge.policy import KnowledgePolicy, build_knowledge_policy
from inferenceos.knowledge.report import (
    render_applicability,
    render_health,
    render_history,
    render_record,
    render_rollback,
)
from inferenceos.knowledge.rollback import build_rollback_record, verify_rollback
from inferenceos.knowledge.service import KnowledgeService, PersistResult, build_service
from inferenceos.knowledge.store import (
    KNOWLEDGE_SCHEMA_VERSION,
    KnowledgeConflictError,
    KnowledgeCorruptionError,
    KnowledgeError,
    KnowledgeSchemaError,
    KnowledgeStoreError,
    KnowledgeUnavailableError,
    SQLiteKnowledgeStore,
)

__all__ = [
    "KNOWLEDGE_SCHEMA_VERSION",
    "KnowledgeConflictError",
    "KnowledgeCorruptionError",
    "KnowledgeError",
    "KnowledgePolicy",
    "KnowledgeSchemaError",
    "KnowledgeService",
    "KnowledgeStoreError",
    "KnowledgeUnavailableError",
    "PersistResult",
    "SQLiteKnowledgeStore",
    "build_context_from_request",
    "build_current_context",
    "build_knowledge_policy",
    "build_rollback_record",
    "build_service",
    "evaluate_applicability",
    "fingerprint_context",
    "render_applicability",
    "render_health",
    "render_history",
    "render_record",
    "render_rollback",
    "verify_rollback",
]
