"""H4 optimizer self-overhead / net-benefit accounting tests (controlled)."""

from __future__ import annotations

import time

from inferenceos.contracts.accounting import (
    MeasurementDistortion,
    NetBenefitAssessment,
    NetBenefitStatus,
    OverheadAttribution,
)
from inferenceos.optimizer_accounting import (
    AccountingPolicy,
    assess_net_benefit,
    build_overhead_attribution,
    distortion_from_stability,
)


def _overhead(*, inclusive: float, target_s: float) -> OverheadAttribution:
    return build_overhead_attribution(
        inclusive_overhead_s=inclusive,
        components={"diagnosis_s": inclusive / 2.0, "reasoning_s": inclusive / 2.0},
        baseline_target_workload_s=target_s,
        decision_count=1,
    )


def _assess(
    baseline: dict[str, float],
    measured: dict[str, float],
    overhead: OverheadAttribution | None,
    *,
    distortion: MeasurementDistortion | None = None,
    preferred: str | None = None,
) -> NetBenefitAssessment:
    return assess_net_benefit(
        baseline_metrics=baseline,
        measured_metrics=measured,
        overhead=overhead,
        distortion=distortion,
        policy=AccountingPolicy(),
        preferred_metric=preferred,
    )


def test_a_clearly_beneficial_case() -> None:
    assessment = _assess(
        {"decode_tokens_per_sec": 20.0},
        {"decode_tokens_per_sec": 26.0},
        _overhead(inclusive=1.0, target_s=20.0),
    )
    assert assessment.status == NetBenefitStatus.BENEFICIAL
    assert assessment.benefit is not None
    assert assessment.benefit.relative_change is not None
    assert assessment.benefit.relative_change > 0.29
    assert assessment.overhead is not None
    assert assessment.overhead.overhead_fraction == 0.05


def test_b_overhead_dominant_case() -> None:
    assessment = _assess(
        {"decode_tokens_per_sec": 20.0},
        {"decode_tokens_per_sec": 20.2},
        _overhead(inclusive=10.0, target_s=20.0),
    )
    assert assessment.status == NetBenefitStatus.OVERHEAD_DOMINANT
    assert assessment.overhead is not None
    assert assessment.overhead.overhead_fraction == 0.5


def test_c_insufficient_baseline() -> None:
    assessment = _assess({}, {"decode_tokens_per_sec": 26.0}, _overhead(inclusive=1.0, target_s=20.0))
    assert assessment.status == NetBenefitStatus.INSUFFICIENT_EVIDENCE
    assert any("no comparable benefit" in reason for reason in assessment.reasons)


def test_d_missing_overhead_is_insufficient() -> None:
    assessment = _assess(
        {"decode_tokens_per_sec": 20.0}, {"decode_tokens_per_sec": 26.0}, None
    )
    assert assessment.status == NetBenefitStatus.INSUFFICIENT_EVIDENCE
    assert assessment.overhead is None
    assert any("overhead measurement unavailable" in reason for reason in assessment.reasons)


def test_e_material_distortion_is_insufficient() -> None:
    distortion = distortion_from_stability(cv=0.9, within_tolerance=False, tolerance=0.25)
    assert distortion.material is True
    assessment = _assess(
        {"decode_tokens_per_sec": 20.0},
        {"decode_tokens_per_sec": 26.0},
        _overhead(inclusive=1.0, target_s=20.0),
        distortion=distortion,
    )
    assert assessment.status == NetBenefitStatus.INSUFFICIENT_EVIDENCE
    assert any("distortion" in reason for reason in assessment.reasons)


def test_f_incomparable_metric_is_structured_insufficient() -> None:
    assessment = _assess(
        {"quality_score": 0.5},
        {"quality_score": 0.9},
        _overhead(inclusive=1.0, target_s=20.0),
        preferred="quality_score",
    )
    assert assessment.status == NetBenefitStatus.INSUFFICIENT_EVIDENCE
    assert assessment.comparable is False
    assert assessment.net_value is None
    assert assessment.benefit is not None and assessment.benefit.comparable is False


def test_g_nested_timing_is_not_double_counted() -> None:
    # A parent "monitor" timer already includes the "diagnosis" child.
    flat = build_overhead_attribution(
        inclusive_overhead_s=1.0, components={"monitor_s": 0.6, "diagnosis_s": 0.4}
    )
    nested = build_overhead_attribution(
        inclusive_overhead_s=1.0, components={"monitor_s": 1.0, "diagnosis_s": 0.6}
    )
    assert flat.inclusive_overhead_s == 1.0
    assert flat.components_sum_s == 1.0
    assert flat.nested is False
    assert nested.inclusive_overhead_s == 1.0  # not 1.0 + 1.0 + 0.6
    assert nested.components_sum_s == 1.6
    assert nested.nested is True
    assert any("double counting" in note for note in nested.notes)


def test_lower_is_better_latency_benefit() -> None:
    assessment = _assess(
        {"total_latency_s": 1.0},
        {"total_latency_s": 0.7},
        _overhead(inclusive=1.0, target_s=20.0),
    )
    assert assessment.status == NetBenefitStatus.BENEFICIAL
    assert assessment.benefit is not None
    assert assessment.benefit.direction == "lower_is_better"
    assert assessment.benefit.relative_change == 0.30000000000000004 or abs(
        (assessment.benefit.relative_change or 0.0) - 0.3
    ) < 1e-9


def test_memory_benefit_is_represented_not_forced_into_throughput() -> None:
    assessment = _assess(
        {"peak_rss_bytes": 1_000_000_000.0},
        {"peak_rss_bytes": 800_000_000.0},
        _overhead(inclusive=1.0, target_s=20.0),
    )
    assert assessment.status == NetBenefitStatus.BENEFICIAL
    assert assessment.benefit is not None
    assert assessment.benefit.unit == "bytes"
    assert assessment.benefit.metric == "peak_rss_bytes"


def test_neutral_when_benefit_below_threshold_and_overhead_in_bound() -> None:
    assessment = _assess(
        {"decode_tokens_per_sec": 20.0},
        {"decode_tokens_per_sec": 20.2},
        _overhead(inclusive=1.0, target_s=20.0),
    )
    assert assessment.status == NetBenefitStatus.NEUTRAL


def test_zero_baseline_is_insufficient() -> None:
    assessment = _assess(
        {"decode_tokens_per_sec": 0.0},
        {"decode_tokens_per_sec": 4.0},
        _overhead(inclusive=1.0, target_s=20.0),
    )
    assert assessment.status == NetBenefitStatus.INSUFFICIENT_EVIDENCE


def test_missing_target_duration_is_insufficient() -> None:
    attribution = build_overhead_attribution(inclusive_overhead_s=1.0)
    assessment = _assess(
        {"decode_tokens_per_sec": 20.0}, {"decode_tokens_per_sec": 26.0}, attribution
    )
    assert assessment.status == NetBenefitStatus.INSUFFICIENT_EVIDENCE
    assert assessment.overhead is not None
    assert assessment.overhead.overhead_fraction is None


def test_metric_selection_prefers_throughput_then_latency() -> None:
    decode = _assess(
        {"decode_tokens_per_sec": 20.0, "ttft_s": 0.5},
        {"decode_tokens_per_sec": 22.0, "ttft_s": 0.6},
        _overhead(inclusive=1.0, target_s=20.0),
    )
    assert decode.benefit is not None and decode.benefit.metric == "decode_tokens_per_sec"
    latency = _assess(
        {"total_latency_s": 2.0, "ttft_s": 0.5},
        {"total_latency_s": 1.6, "ttft_s": 0.5},
        _overhead(inclusive=1.0, target_s=20.0),
    )
    assert latency.benefit is not None and latency.benefit.metric == "total_latency_s"


def test_assessment_roundtrips_json() -> None:
    assessment = _assess(
        {"decode_tokens_per_sec": 20.0},
        {"decode_tokens_per_sec": 26.0},
        _overhead(inclusive=1.0, target_s=20.0),
    )
    restored = NetBenefitAssessment.model_validate_json(assessment.model_dump_json())
    assert restored.status == assessment.status
    assert restored.benefit_ratio == assessment.benefit_ratio


def test_non_material_distortion_does_not_force_insufficient() -> None:
    distortion = distortion_from_stability(cv=0.05, within_tolerance=True, tolerance=0.25)
    assert distortion.material is False
    assessment = _assess(
        {"decode_tokens_per_sec": 20.0},
        {"decode_tokens_per_sec": 26.0},
        _overhead(inclusive=1.0, target_s=20.0),
        distortion=distortion,
    )
    assert assessment.status == NetBenefitStatus.BENEFICIAL


def test_runtime_monitor_emits_net_benefit_assessment() -> None:
    from inferenceos.contracts.enums import ApprovalOutcome
    from inferenceos.contracts.runtime import RuntimeBudget
    from inferenceos.runtime.approval import CallbackApprovalProvider
    from inferenceos.runtime.monitor import RuntimeMonitor
    from inferenceos.runtime.verification import (
        ScriptedRuntimeSampler,
        ScriptedWorkload,
        healthy_metrics,
    )
    from inferenceos.selftest import sample_hardware
    from tests.runtime_helpers import make_runtime_policy

    workload = ScriptedWorkload()
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=ScriptedRuntimeSampler(healthy_metrics),
        policy=make_runtime_policy(
            budget=RuntimeBudget(
                max_duration_s=2.0,
                telemetry_interval_s=0.5,
                observation_window_size=20,
                baseline_samples=2,
                max_deep_analyses=1,
                max_interventions=1,
                max_cumulative_intervention_s=60.0,
                max_optimizer_overhead_s=60.0,
                approval_timeout_s=5.0,
                max_alerts_per_condition=1,
            )
        ),
        approval_provider=CallbackApprovalProvider(lambda _rec: ApprovalOutcome.REJECTED),
        hardware=sample_hardware(),
    )
    result = monitor.run()
    assessment = result.run.net_benefit
    assert assessment is not None
    assert assessment.status in {
        NetBenefitStatus.BENEFICIAL,
        NetBenefitStatus.NEUTRAL,
        NetBenefitStatus.OVERHEAD_DOMINANT,
        NetBenefitStatus.INSUFFICIENT_EVIDENCE,
    }
    # Overhead is always measurable in the runtime monitor; benefit is same-baseline.
    assert assessment.overhead is not None
    assert assessment.overhead.inclusive_overhead_s >= 0.0


def test_accounting_is_cheap() -> None:
    attribution = _overhead(inclusive=1.0, target_s=20.0)
    iterations = 1000
    start = time.perf_counter()
    for _ in range(iterations):
        _assess(
            {"decode_tokens_per_sec": 20.0},
            {"decode_tokens_per_sec": 26.0},
            attribution,
        )
    mean_ms = (time.perf_counter() - start) / iterations * 1000.0
    assert mean_ms < 1.0
