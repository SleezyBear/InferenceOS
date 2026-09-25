"""Measurement-validity contracts (B1: power/thermal/frequency confounds).

A :class:`MeasurementValidityAssessment` explains whether a set of observations can
be trusted as an optimization signal. It records *directly observed* power state and
CPU clock performance, the reference used, the material deviation (if any), and the
deterministic reasons for the resulting status.

It never claims direct thermal detection: ``power_state`` and
``cpu_performance_ratio`` are observed; a "confound" is an inference from those
observations, stated as such.
"""

from __future__ import annotations

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import MeasurementValidity, PowerState

MEASUREMENT_VALIDITY_SCHEMA_VERSION = "b1.1"


class MeasurementValidityAssessment(ContractBase):
    schema_version: str = MEASUREMENT_VALIDITY_SCHEMA_VERSION

    validity: MeasurementValidity = MeasurementValidity.UNKNOWN
    # True only when a materially confounded state was observed and persisted.
    confounded: bool = False

    # Directly observed power signal (summary over the assessed runs).
    power_state: PowerState = PowerState.UNKNOWN

    # Directly observed CPU performance signal.
    cpu_performance_ratio: float | None = None
    reference_ratio: float | None = None
    deviation_fraction: float | None = None

    observed_samples: int = Field(default=0, ge=0)
    degraded_samples: int = Field(default=0, ge=0)
    max_consecutive_degraded: int = Field(default=0, ge=0)

    # Policy under which the assessment was made (for auditability).
    ratio_floor: float = 0.0
    deviation_threshold: float = 0.0
    persistence_samples: int = 1
    unknown_blocks: bool = False

    reasons: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
