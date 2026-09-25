"""B1 measurement-validity evaluation.

Determines whether a set of real observations can be trusted as an optimization
signal, given the laptop's observed power and CPU-performance state.

Design (deterministic, conservative):

* Power state and CPU performance are **directly observed** (``SystemState``); a
  "confound" is an explicit inference from those observations, never a claimed
  direct thermal measurement.
* A sample is *degraded* when the observed current/(ceiling) clock ratio falls below
  ``cpu_ratio_floor``. Comparing against the CPU's nominal maximum alone is not
  sufficient, so the floor is only treated as a confound when **corroborated** by
  either battery power or a material drop relative to the best ratio observed in the
  same session (``cpu_ratio_deviation``).
* Persistence (``persistence_samples`` consecutive degraded samples) prevents a
  single transient sample from invalidating a workload.
* Unknown telemetry is never treated as healthy/AC; ``unknown_blocks`` optionally
  makes unknown telemetry fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferenceos.config import Settings
from inferenceos.contracts.enums import MeasurementValidity, PowerState
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.validity import MEASUREMENT_VALIDITY_SCHEMA_VERSION, MeasurementValidityAssessment

MEASUREMENT_VALIDITY_POLICY_VERSION = "b1.1"


@dataclass(frozen=True)
class MeasurementValidityPolicy:
    """Explicit, configurable validity policy (see config ``phase_validity_*``)."""

    cpu_ratio_floor: float = 0.60
    cpu_ratio_deviation: float = 0.25
    persistence_samples: int = 2
    unknown_blocks: bool = False
    version: str = MEASUREMENT_VALIDITY_POLICY_VERSION


def build_validity_policy(settings: Settings) -> MeasurementValidityPolicy:
    return MeasurementValidityPolicy(
        cpu_ratio_floor=settings.phase_validity_cpu_ratio_floor,
        cpu_ratio_deviation=settings.phase_validity_cpu_ratio_deviation,
        persistence_samples=settings.phase_validity_persistence_samples,
        unknown_blocks=settings.phase_validity_unknown_blocks,
    )


def _longest_consecutive(flags: list[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def evaluate_report_validity(
    report: BaselineReport,
    policy: MeasurementValidityPolicy | None = None,
) -> MeasurementValidityAssessment:
    """Evaluate measurement validity over a baseline report's real observations."""
    policy = policy or MeasurementValidityPolicy()
    runs = [run for run in report.runs if not run.identity.warmup]

    power_states = [
        run.observation.system_state.power_state
        for run in runs
        if run.observation.system_state is not None
    ]
    battery_seen = any(state is PowerState.BATTERY for state in power_states)
    ac_seen = any(state is PowerState.AC for state in power_states)
    summary_power = (
        PowerState.BATTERY if battery_seen else PowerState.AC if ac_seen else PowerState.UNKNOWN
    )

    ratios = [
        float(run.observation.system_state.cpu_performance_ratio)
        for run in runs
        if run.observation.system_state is not None
        and run.observation.system_state.cpu_performance_ratio is not None
    ]

    base_reasons = [
        "power state and CPU clock performance are directly observed; a confound is "
        "inferred from those observations (no direct thermal sensor is read)",
    ]

    if not ratios:
        reasons = list(base_reasons)
        reasons.append(
            "CPU performance/frequency telemetry is unavailable; validity cannot be "
            "established (this is not treated as healthy)"
        )
        return MeasurementValidityAssessment(
            schema_version=MEASUREMENT_VALIDITY_SCHEMA_VERSION,
            validity=MeasurementValidity.UNKNOWN,
            confounded=False,
            power_state=summary_power,
            cpu_performance_ratio=None,
            reference_ratio=None,
            deviation_fraction=None,
            observed_samples=0,
            degraded_samples=0,
            max_consecutive_degraded=0,
            ratio_floor=policy.cpu_ratio_floor,
            deviation_threshold=policy.cpu_ratio_deviation,
            persistence_samples=policy.persistence_samples,
            unknown_blocks=policy.unknown_blocks,
            reasons=reasons,
            limitations=[
                "no direct thermal throttling sensor is read; performance state is "
                "inferred from the observed clock ratio and power state",
            ],
        )

    reference = max(ratios)
    minimum = min(ratios)
    deviation = (reference - minimum) / reference if reference > 0 else None
    degraded_flags = [ratio < policy.cpu_ratio_floor for ratio in ratios]
    degraded_count = sum(1 for flag in degraded_flags if flag)
    max_consecutive = _longest_consecutive(degraded_flags)
    persisted = max_consecutive >= max(policy.persistence_samples, 1)
    corroborated = battery_seen or (
        deviation is not None and deviation >= policy.cpu_ratio_deviation
    )
    confounded = bool(persisted and corroborated)

    reasons = list(base_reasons)
    if confounded:
        reasons.append(
            "CPU performance was materially below the observed ceiling for "
            f"{max_consecutive} consecutive sample(s) (floor {policy.cpu_ratio_floor:.2f}), "
            "corroborated by "
            + ("battery power" if battery_seen else "a material drop from the best observed state")
        )
        if battery_seen:
            reasons.append("external power source is battery")
        if deviation is not None and deviation >= policy.cpu_ratio_deviation:
            reasons.append(
                f"relative performance drop from the best observed state "
                f"{deviation:.2%} (threshold {policy.cpu_ratio_deviation:.0%})"
            )
    elif battery_seen and max_consecutive < max(policy.persistence_samples, 1):
        reasons.append(
            "battery power is present but observed CPU performance remained within "
            "bounds; battery alone does not invalidate a measurement"
        )
    elif battery_seen:
        reasons.append(
            "battery power is present but consecutive CPU performance stayed within "
            "hysteresis; not treated as a confound"
        )
    else:
        reasons.append("observed CPU performance and power state are within bounds")

    return MeasurementValidityAssessment(
        schema_version=MEASUREMENT_VALIDITY_SCHEMA_VERSION,
        validity=MeasurementValidity.DEGRADED if confounded else MeasurementValidity.VALID,
        confounded=confounded,
        power_state=summary_power,
        cpu_performance_ratio=round(sum(ratios) / len(ratios), 4),
        reference_ratio=round(reference, 4),
        deviation_fraction=round(deviation, 4) if deviation is not None else None,
        observed_samples=len(ratios),
        degraded_samples=degraded_count,
        max_consecutive_degraded=max_consecutive,
        ratio_floor=policy.cpu_ratio_floor,
        deviation_threshold=policy.cpu_ratio_deviation,
        persistence_samples=policy.persistence_samples,
        unknown_blocks=policy.unknown_blocks,
        reasons=reasons,
        limitations=[
            "no direct thermal throttling sensor is read; performance state is "
            "inferred from the observed clock ratio and power state",
        ],
    )


def validity_blocks(assessment: MeasurementValidityAssessment | None) -> bool:
    """Whether the assessment must prevent optimization from trusting the measurement."""
    if assessment is None:
        return False
    if assessment.confounded:
        return True
    return bool(
        assessment.validity is MeasurementValidity.UNKNOWN and assessment.unknown_blocks
    )
