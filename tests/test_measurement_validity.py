"""B1: deterministic power/performance measurement-validity policy and propagation."""

from __future__ import annotations

from inferenceos.config import Settings
from inferenceos.contracts.enums import MeasurementValidity, PowerState
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.loop.verification import make_paging_report
from inferenceos.measurement_validity import (
    MeasurementValidityPolicy,
    build_validity_policy,
    evaluate_report_validity,
    validity_blocks,
)
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.reasoning.escalation import decide_escalation
from inferenceos.validity_acceptance import _inject, _synthetic_report


def test_normal_ac_is_valid() -> None:
    assessment = evaluate_report_validity(
        _synthetic_report([(0.95, PowerState.AC), (0.97, PowerState.AC)])
    )
    assert assessment.validity is MeasurementValidity.VALID
    assert assessment.confounded is False
    assert assessment.power_state is PowerState.AC


def test_battery_without_degradation_is_valid() -> None:
    assessment = evaluate_report_validity(
        _synthetic_report([(0.90, PowerState.BATTERY), (0.92, PowerState.BATTERY)])
    )
    assert assessment.power_state is PowerState.BATTERY
    assert assessment.validity is MeasurementValidity.VALID
    assert not validity_blocks(assessment)


def test_persisted_battery_degradation_is_confounded() -> None:
    assessment = evaluate_report_validity(
        _synthetic_report(
            [(0.95, PowerState.AC), (0.45, PowerState.BATTERY), (0.50, PowerState.BATTERY)]
        )
    )
    assert assessment.validity is MeasurementValidity.DEGRADED
    assert assessment.confounded is True
    assert validity_blocks(assessment)


def test_ac_relative_drop_is_confounded() -> None:
    assessment = evaluate_report_validity(
        _synthetic_report(
            [
                (0.96, PowerState.AC),
                (0.95, PowerState.AC),
                (0.45, PowerState.AC),
                (0.46, PowerState.AC),
            ]
        )
    )
    assert assessment.confounded is True
    assert assessment.deviation_fraction is not None
    assert assessment.deviation_fraction >= 0.25


def test_unknown_telemetry_is_explicit_and_policy_controlled() -> None:
    report = _synthetic_report([(None, PowerState.UNKNOWN), (None, PowerState.UNKNOWN)])
    default = evaluate_report_validity(report)
    assert default.validity is MeasurementValidity.UNKNOWN
    assert validity_blocks(default) is False
    blocking = evaluate_report_validity(report, MeasurementValidityPolicy(unknown_blocks=True))
    assert validity_blocks(blocking) is True


def test_transient_deviation_does_not_invalidate() -> None:
    assessment = evaluate_report_validity(
        _synthetic_report([(0.95, PowerState.AC), (0.45, PowerState.AC), (0.96, PowerState.AC)])
    )
    assert assessment.validity is MeasurementValidity.VALID
    assert assessment.max_consecutive_degraded == 1


def test_single_bad_sample_respects_persistence() -> None:
    policy = MeasurementValidityPolicy(persistence_samples=2)
    assessment = evaluate_report_validity(
        _synthetic_report([(0.45, PowerState.BATTERY)]), policy
    )
    assert assessment.validity is MeasurementValidity.VALID


def test_boundary_around_floor_is_deterministic() -> None:
    above = evaluate_report_validity(
        _synthetic_report([(0.61, PowerState.BATTERY), (0.61, PowerState.BATTERY)])
    )
    below = evaluate_report_validity(
        _synthetic_report([(0.59, PowerState.BATTERY), (0.59, PowerState.BATTERY)])
    )
    assert above.validity is MeasurementValidity.VALID
    assert below.confounded is True


def test_policy_builder_reads_settings() -> None:
    policy = build_validity_policy(
        Settings(
            phase_validity_cpu_ratio_floor=0.75,
            phase_validity_cpu_ratio_deviation=0.10,
            phase_validity_persistence_samples=3,
            phase_validity_unknown_blocks=True,
        )
    )
    assert policy.cpu_ratio_floor == 0.75
    assert policy.cpu_ratio_deviation == 0.10
    assert policy.persistence_samples == 3
    assert policy.unknown_blocks is True


def test_confounded_diagnosis_is_suppressed_end_to_end() -> None:
    report = _inject(make_paging_report(), ratio=0.45, power=PowerState.BATTERY)
    diagnosis = BottleneckDetector().detect(report, source_report_path="<test>")
    assert diagnosis.measurement_validity is not None
    assert diagnosis.measurement_validity.confounded is True
    assert diagnosis.meaningful_opportunity is False
    assert diagnosis.primary is None
    assert diagnosis.escalation_required is False
    assert decide_escalation(diagnosis).should_escalate is False
    plan = generate_candidates(diagnosis, report=report, policy=OptimizationPolicy())
    assert not any(candidate.executable for candidate in plan.candidates)


def test_normal_measurement_behavior_is_unchanged() -> None:
    report = make_paging_report()
    diagnosis = BottleneckDetector().detect(report, source_report_path="<test>")
    assert diagnosis.primary is not None
    assert diagnosis.meaningful_opportunity is True
