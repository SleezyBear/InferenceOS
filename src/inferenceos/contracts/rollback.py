"""Phase 8R real rollback-execution contracts.

These contracts record the difference between *recording* a rollback and *executing*
one: applicability against a live target, integrity, checkpoint/apply/reload,
behavioural and measurement verification, and the terminal outcome. They are additive
and do not replace the Phase 8 ``RollbackRecord``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import (
    RollbackApplicabilityClass,
    RollbackExecutionStatus,
)
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.lifecycle import ProcessIdentity

ROLLBACK_SCHEMA_VERSION = "8r.1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RollbackExecutionPolicy(ContractBase):
    """Explicit, inspectable policy for a real rollback attempt."""

    max_age_days: float = Field(default=30.0, ge=0.0)
    measurement_tolerance: float = Field(default=0.30, ge=0.0)
    latency_tolerance: float = Field(default=2.0, ge=0.0)
    probe_n_predict: int = Field(default=8, ge=1)
    probe_seed: int = 1234
    reload_timeout_s: float = Field(default=180.0, gt=0.0)
    # Controlled fault injection for exercising the containment path in acceptance.
    fault_stage: str | None = None


class RollbackApplicability(ContractBase):
    """Explainable verdict about whether a rollback may be executed against a target."""

    classification: RollbackApplicabilityClass
    rollback_id: str
    experiment_id: str
    integrity_ok: bool
    model_match: bool | None = None
    backend_match: bool | None = None
    hardware_match: bool | None = None
    target_in_resulting_state: bool | None = None
    already_restored: bool | None = None
    stale: bool = False
    reasons: list[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=_utcnow)

    @property
    def executable(self) -> bool:
        return self.classification is RollbackApplicabilityClass.APPLICABLE


class RollbackExecutionResult(ContractBase):
    """Full evidence record for one real rollback attempt."""

    schema_version: str = ROLLBACK_SCHEMA_VERSION
    execution_id: str
    run_id: str
    rollback_id: str
    experiment_id: str
    status: RollbackExecutionStatus
    applicability: RollbackApplicability
    integrity_ok: bool

    checkpoint_configuration: RuntimeConfiguration | None = None
    requested_configuration: RuntimeConfiguration | None = None
    applied_configuration: RuntimeConfiguration | None = None
    actual_configuration: RuntimeConfiguration | None = None

    pid_before: int | None = None
    pid_after: int | None = None
    identity_before: ProcessIdentity | None = None
    identity_after: ProcessIdentity | None = None
    process_reloaded: bool = False
    reload_s: float | None = None
    ready: bool | None = None

    configuration_restored: bool = False
    behavioural_probe_attempted: bool = False
    behavioural_probe_passed: bool = False
    baseline_probe_hash: str | None = None
    checkpoint_probe_hash: str | None = None
    restored_probe_hash: str | None = None
    measurement_verification_passed: bool | None = None
    measurement_detail: str = ""
    restored_metrics: dict[str, float] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)

    recovery_attempted: bool = False
    recovery_verified: bool = False

    event_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    duration_s: float | None = None
    summary: str = ""
