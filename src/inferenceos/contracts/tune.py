"""Contracts for the everyday "before loading" commands (step 5).

* ``tune`` searches for the best setup for one model on this machine, confirms it by
  alternating runs against the starting setup, asks the user to approve each change,
  and saves the approved setup (:class:`SavedSetup`). It does not start the model.
* ``start`` runs the model with the saved setup after a quick check that the machine
  has not changed (:class:`StartCheck`), and keeps it running until the user stops it.

Saved setups are kept separately for plugged in and on battery. Context size is part
of a saved setup exactly as the user gave it; nothing here ever changes it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.goal import KeepJudgement

TUNE_SCHEMA_VERSION = "5.1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PowerPlan(ContractBase):
    """The operating system's active power plan (Windows), when it can be read."""

    plan_id: str | None = None
    name: str | None = None


class MachineCheck(ContractBase):
    """What the machine looks like with no model loaded.

    Only the parts something uses today (step 5): free graphics card memory, the power
    source and plan, the graphics driver and the program version. Speed tests of the
    drive, memory and cores come in a later step.
    """

    schema_version: str = TUNE_SCHEMA_VERSION
    checked_at: datetime = Field(default_factory=_utcnow)
    hostname: str | None = None
    hardware_signature: str | None = None
    graphics_drivers: list[str] = Field(default_factory=list)
    total_graphics_memory_bytes: int | None = None
    used_graphics_memory_bytes: int | None = None
    free_graphics_memory_bytes: int | None = None
    power_source: PowerState = PowerState.UNKNOWN
    power_plan: PowerPlan = Field(default_factory=PowerPlan)
    program: str = "llama.cpp"
    program_version: str | None = None
    notes: list[str] = Field(default_factory=list)


class ModelFileIdentity(ContractBase):
    """Which model file a saved setup belongs to."""

    path: str
    size_bytes: int = Field(ge=0)
    modified_ns: int
    sha256: str | None = None


class ProposedChange(ContractBase):
    """One change the search found, shown to the user with its measured gain."""

    change_id: str
    card_id: str
    setting: str
    description: str
    old_value: int | bool | str | None = None
    new_value: int | bool | str | None = None
    measured_gain: float | None = None
    evidence: list[str] = Field(default_factory=list)


class ChangeDecision(ContractBase):
    """The user's answer for one proposed change."""

    change_id: str
    approved: bool
    answer: str
    detail: str = ""


class ConfirmationResult(ContractBase):
    """Final setup measured against the starting setup, alternating between the two."""

    rounds: int = Field(default=0, ge=0)
    order: list[str] = Field(default_factory=list)
    starting_samples: int = Field(default=0, ge=0)
    final_samples: int = Field(default=0, ge=0)
    per_measure: dict[str, float] = Field(default_factory=dict)
    quality_passed: bool = False
    judgement: KeepJudgement | None = None
    disturbed_repeats: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return bool(self.judgement is not None and self.judgement.keep)


class SavedSetup(ContractBase):
    """An approved setup for one machine, model, program, power source and context."""

    schema_version: str = TUNE_SCHEMA_VERSION
    setup_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    model: ModelFileIdentity
    model_name: str
    context_length: int = Field(ge=1)
    power_source: PowerState
    program: str = "llama.cpp"
    machine: MachineCheck
    configuration: RuntimeConfiguration
    starting_configuration: RuntimeConfiguration
    approved_changes: list[ProposedChange] = Field(default_factory=list)
    declined_changes: list[ProposedChange] = Field(default_factory=list)
    confirmed_gain: float | None = None
    effort_mode: str | None = None
    goal_mode: str | None = None
    tune_report: str | None = None
    notes: list[str] = Field(default_factory=list)


class TuneResult(ContractBase):
    """Everything one ``tune`` run did, for the report file."""

    schema_version: str = TUNE_SCHEMA_VERSION
    run_id: str
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    model_path: str
    context_length: int
    effort_mode: str
    goal_mode: str
    time_limit_s: float
    machine: MachineCheck
    machine_changes: list[str] = Field(default_factory=list)
    starting_configuration: RuntimeConfiguration
    starting_point_source: str
    final_configuration: RuntimeConfiguration
    rounds: list[dict[str, object]] = Field(default_factory=list)
    proposed_changes: list[ProposedChange] = Field(default_factory=list)
    confirmation: ConfirmationResult | None = None
    decisions: list[ChangeDecision] = Field(default_factory=list)
    recheck: ConfirmationResult | None = None
    saved_setup: SavedSetup | None = None
    outcome: str = ""
    not_reached: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    #: What heat did during this run: every comparison it disturbed (and whether it
    #: changed the decision) and the heat check of the chosen setup.
    heat_report: dict[str, object] = Field(default_factory=dict)

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)


class StartCheck(ContractBase):
    """The quick check ``start`` does before running a saved setup."""

    setup: SavedSetup | None = None
    power_source: PowerState = PowerState.UNKNOWN
    matched_power_source: bool = False
    model_matches: bool = False
    changes: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def can_start(self) -> bool:
        return self.setup is not None and self.model_matches and not self.problems


__all__ = [
    "ChangeDecision",
    "ConfirmationResult",
    "MachineCheck",
    "ModelFileIdentity",
    "PowerPlan",
    "ProposedChange",
    "SavedSetup",
    "StartCheck",
    "TUNE_SCHEMA_VERSION",
    "TuneResult",
]
