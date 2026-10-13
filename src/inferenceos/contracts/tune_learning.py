"""C4-1: observations, never acceptance evidence or executable instructions."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from inferenceos.contracts.base import ContractBase


class MeasurementObservation(ContractBase):
    operation_id: int = Field(ge=1)
    kind: Literal["screening", "final-side"]
    request: dict[str, Any]
    status: Literal["completed", "failed", "incomplete"]
    wall_s: float = Field(ge=0, allow_inf_nan=False)
    valid_runs: int | None = Field(default=None, ge=0)
    retries: int | None = Field(default=None, ge=0)
    phases: dict[str, Any] = Field(default_factory=dict)
    failure: dict[str, str] | None = None
    measurement_notes: tuple[str, ...] = ()
    # A completed measurement does not imply its settings/effect/quality passed.
    acceptance_evidence: Literal[False] = False


class DisturbanceControls(ContractBase):
    rule_version: Literal["disturbance-1"] = "disturbance-1"
    logical_cores: int = Field(ge=1, strict=True)
    limit_percent: float = Field(ge=0, allow_inf_nan=False, strict=True)
    max_repeats: int = Field(ge=0, strict=True)


class TuneHistoryRun(ContractBase):
    schema_version: Literal["tune-history-1"] = "tune-history-1"
    run_id: str = Field(min_length=1)
    started_at: datetime | None = None
    origin: Literal["hardware", "simulation", "unknown"] = "unknown"
    identity: dict[str, Any]
    observations: tuple[MeasurementObservation, ...]
    comparisons: tuple[dict[str, Any], ...]
    disposition: dict[str, Any]
    source_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostics: tuple[str, ...] = ()
    # C4-1 records only. No caller can mistake the journal for verified evidence.
    reusable_for_acceptance: Literal[False] = False
