"""Audit-record schema.

Every accepted or rejected decision is persisted as an ``AuditRecord`` covering the
full chain: baseline -> diagnosis -> (optional reasoning) -> candidates ->
experiment -> decision, plus the exact prior values needed for rollback.

The schema is defined in Phase 0 even though it is populated from Phase 7 onward so
that all persisted artifacts share one stable shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.bottleneck import Bottleneck
from inferenceos.contracts.candidate import Candidate
from inferenceos.contracts.enums import Decision
from inferenceos.contracts.experiment import ExperimentResult
from inferenceos.contracts.observation import Observation
from inferenceos.contracts.reasoning import ReasoningRequest, ReasoningResponse
from inferenceos.signature import Signature

AUDIT_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AppliedChange(ContractBase):
    """A single runtime value that was changed, with its prior value for rollback."""

    parameter: str
    prior_value: Any = None
    applied_value: Any = None


class AuditRecord(ContractBase):
    schema_version: str = AUDIT_SCHEMA_VERSION
    record_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    signature: Signature

    baseline: Observation | None = None
    diagnosis: Bottleneck | None = None
    reasoning_request: ReasoningRequest | None = None
    reasoning_response: ReasoningResponse | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    experiment: ExperimentResult | None = None
    decision: Decision = Decision.PENDING
    applied_changes: list[AppliedChange] = Field(default_factory=list)
    notes: str | None = None

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str) -> "AuditRecord":
        """Reconstruct a record from persisted JSON alone (no in-memory state)."""
        return cls.model_validate_json(data)
