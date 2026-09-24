"""Comparison + decision policy: improvement, no-op, regression, quality, failures."""

from __future__ import annotations

from inferenceos.contracts.enums import (
    ExperimentDecisionKind,
    QualityStatus,
    RestorationStatus,
)
from inferenceos.contracts.experiment import (
    ApplicationResult,
    ExperimentComparison,
    ExperimentDecision,
    ExperimentMeasurement,
    QualityValidationResult,
    RuntimeConfiguration,
)
from inferenceos.experiment.comparison import compare
from inferenceos.experiment.decision import decide
from inferenceos.profiling.statistics import summarize
from tests.experiment_helpers import build_test_policy


def _measurement(values: dict[str, list[float]]) -> ExperimentMeasurement:
    return ExperimentMeasurement(
        measurement_id="m",
        phase="x",
        configuration=RuntimeConfiguration(),
        repetitions=2,
        valid_runs=2,
        run_ids=["a", "b"],
        metrics={name: summarize(name, vals) for name, vals in values.items()},
    )


def _quality(status: QualityStatus = QualityStatus.OK, regression: bool = False) -> QualityValidationResult:
    return QualityValidationResult(
        status=status,
        validated=status in {QualityStatus.OK, QualityStatus.ACCEPTABLE_VARIATION},
        execution_ok=True,
        structural_ok=True,
        quality_regression=regression,
        unacceptable_output=regression,
    )


def test_bottleneck_improvement_detected() -> None:
    comparison = compare(
        baseline=_measurement({"peak_rss_bytes": [1.3e9, 1.3e9], "decode_tokens_per_sec": [20.0, 20.0]}),
        candidate=_measurement({"peak_rss_bytes": [0.9e9, 0.9e9], "decode_tokens_per_sec": [20.0, 20.0]}),
        bottleneck_metric="peak_rss_bytes",
        bottleneck_direction="decrease",
        quality=_quality(),
        policy=build_test_policy(),
    )
    assert comparison.bottleneck_improved is True
    assert comparison.bottleneck_persistent is False
    assert comparison.objective_regressed is False


def test_no_meaningful_change_detected() -> None:
    comparison = compare(
        baseline=_measurement({"peak_rss_bytes": [1.3e9, 1.3e9], "decode_tokens_per_sec": [20.0, 20.0]}),
        candidate=_measurement({"peak_rss_bytes": [1.3e9, 1.3e9], "decode_tokens_per_sec": [20.0, 20.0]}),
        bottleneck_metric="peak_rss_bytes",
        bottleneck_direction="decrease",
        quality=_quality(),
        policy=build_test_policy(),
    )
    assert comparison.bottleneck_persistent is True
    assert comparison.bottleneck_improved is False
    assert comparison.deltas["peak_rss_bytes"].significant is False


def test_objective_regression_detected() -> None:
    comparison = compare(
        baseline=_measurement({"peak_rss_bytes": [1.3e9, 1.3e9], "decode_tokens_per_sec": [20.0, 20.0]}),
        candidate=_measurement({"peak_rss_bytes": [0.9e9, 0.9e9], "decode_tokens_per_sec": [8.0, 8.0]}),
        bottleneck_metric="peak_rss_bytes",
        bottleneck_direction="decrease",
        quality=_quality(),
        policy=build_test_policy(),
    )
    assert comparison.objective_regressed is True
    assert comparison.deltas["decode_tokens_per_sec"].regressed is True


def test_resource_regression_detected() -> None:
    comparison = compare(
        baseline=_measurement({"peak_rss_bytes": [1.0e9, 1.0e9], "hard_page_faults_per_sec": [5.0, 5.0]}),
        candidate=_measurement({"peak_rss_bytes": [1.4e9, 1.4e9], "hard_page_faults_per_sec": [5.0, 5.0]}),
        bottleneck_metric="hard_page_faults_per_sec",
        bottleneck_direction="decrease",
        quality=_quality(),
        policy=build_test_policy(),
    )
    assert comparison.resource_regressed is True


def _decide(
    comparison: ExperimentComparison | None,
    quality: QualityValidationResult | None,
    *,
    application: ApplicationResult | None = None,
    restoration_status: RestorationStatus = RestorationStatus.RESTORED,
    measurement_valid: bool = True,
    precondition_ok: bool = True,
    precondition_kind: ExperimentDecisionKind | None = None,
    precondition_reason: str = "ok",
    errors: list[str] | None = None,
) -> ExperimentDecision:
    if application is None:
        application = ApplicationResult(
            applied=True, action_id="runtime.right_size_context", supported=True
        )
    return decide(
        comparison=comparison,
        quality=quality,
        application=application,
        restoration_status=restoration_status,
        measurement_valid=measurement_valid,
        precondition_ok=precondition_ok,
        precondition_kind=precondition_kind,
        precondition_reason=precondition_reason,
        policy=build_test_policy(),
        errors=errors or [],
    )


def test_decide_keep_on_improvement() -> None:
    comparison = compare(
        baseline=_measurement({"peak_rss_bytes": [1.3e9, 1.3e9]}),
        candidate=_measurement({"peak_rss_bytes": [0.9e9, 0.9e9]}),
        bottleneck_metric="peak_rss_bytes",
        bottleneck_direction="decrease",
        quality=_quality(),
        policy=build_test_policy(),
    )
    decision = _decide(comparison, _quality())
    assert decision.kind is ExperimentDecisionKind.IMPROVED
    assert decision.keep is True


def test_decide_quality_regression_blocks_success() -> None:
    comparison = compare(
        baseline=_measurement({"peak_rss_bytes": [1.3e9, 1.3e9]}),
        candidate=_measurement({"peak_rss_bytes": [0.9e9, 0.9e9]}),
        bottleneck_metric="peak_rss_bytes",
        bottleneck_direction="decrease",
        quality=_quality(QualityStatus.QUALITY_REGRESSION, regression=True),
        policy=build_test_policy(),
    )
    decision = _decide(comparison, _quality(QualityStatus.QUALITY_REGRESSION, regression=True))
    assert decision.kind is ExperimentDecisionKind.QUALITY_REGRESSION
    assert decision.keep is False


def test_decide_restoration_failure_dominates() -> None:
    comparison = compare(
        baseline=_measurement({"peak_rss_bytes": [1.3e9, 1.3e9]}),
        candidate=_measurement({"peak_rss_bytes": [0.9e9, 0.9e9]}),
        bottleneck_metric="peak_rss_bytes",
        bottleneck_direction="decrease",
        quality=_quality(),
        policy=build_test_policy(),
    )
    decision = _decide(
        comparison, _quality(), restoration_status=RestorationStatus.FAILED
    )
    assert decision.kind is ExperimentDecisionKind.RESTORATION_FAILURE


def test_decide_measurement_failure() -> None:
    decision = _decide(None, None, measurement_valid=False, errors=["no samples"])
    assert decision.kind is ExperimentDecisionKind.MEASUREMENT_FAILURE


def test_decide_application_failure() -> None:
    decision = _decide(
        None,
        None,
        application=ApplicationResult(
            applied=False, action_id="x", supported=False, reason="no surface"
        ),
    )
    assert decision.kind is ExperimentDecisionKind.APPLICATION_FAILURE


def test_decide_precondition_failure() -> None:
    decision = _decide(
        None,
        None,
        precondition_ok=False,
        precondition_kind=ExperimentDecisionKind.UNSAFE,
        precondition_reason="approval required",
    )
    assert decision.kind is ExperimentDecisionKind.UNSAFE
