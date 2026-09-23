"""Workload contracts (Phase 3 harness: prompt sets, seeds, measurement modes)."""

from __future__ import annotations

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import MeasurementMode, WorkloadKind


class WorkloadProfile(ContractBase):
    workload_id: str
    kind: WorkloadKind = WorkloadKind.UNKNOWN
    description: str = ""
    prompt_set_ref: str | None = None
    prompts: list[str] = Field(default_factory=list)
    seed: int = 0
    expected_context_length: int | None = None
    concurrency: int = Field(default=1, ge=1)
    modes: list[MeasurementMode] = Field(
        default_factory=lambda: [MeasurementMode.COLD, MeasurementMode.WARM]
    )
