"""Quality validation: execution, structure, truncation, reference, regression."""

from __future__ import annotations

from inferenceos.contracts.enums import MeasurementMode, QualityStatus
from inferenceos.contracts.experiment import ExperimentMeasurement, RuntimeConfiguration
from inferenceos.experiment.quality import validate_quality
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_model
from tests.experiment_helpers import SampleResult, build_test_policy, make_run

REFERENCE = "A key value cache stores attention state."


def _measurement(
    texts: list[str], *, tokens: int = 32, workload_id: str = "short-short"
) -> ExperimentMeasurement:
    workload = get_workload(workload_id)
    task = ProfileTask(model=sample_model(), workload=workload, context_length=2048)
    runs = [
        make_run(
            task,
            ProfilerOptions(),
            SampleResult(text=text, generated_tokens=tokens),
            mode=MeasurementMode.WARM,
            warmup=False,
            sequence=index,
            repetition=index,
        )
        for index, text in enumerate(texts)
    ]
    return ExperimentMeasurement(
        measurement_id="m",
        phase="candidate",
        configuration=RuntimeConfiguration(context_length=2048),
        repetitions=len(runs),
        valid_runs=len(runs),
        run_ids=[run.identity.run_id for run in runs],
        generated_texts=list(texts),
        runs=runs,
    )


def test_valid_output_is_accepted() -> None:
    result = validate_quality(
        baseline=_measurement([REFERENCE]),
        candidate=_measurement([REFERENCE]),
        workload=get_workload("short-short"),
        policy=build_test_policy(),
    )
    assert result.status is QualityStatus.OK
    assert result.validated is True
    assert result.deterministic_match is True
    assert result.unacceptable_output is False


def test_acceptable_variation() -> None:
    result = validate_quality(
        baseline=_measurement([REFERENCE]),
        candidate=_measurement(["A key value cache stores attention state for tokens."]),
        workload=get_workload("short-short"),
        policy=build_test_policy(),
    )
    assert result.status is QualityStatus.ACCEPTABLE_VARIATION
    assert result.quality_regression is False


def test_empty_output_is_execution_failure() -> None:
    result = validate_quality(
        baseline=_measurement([REFERENCE]),
        candidate=_measurement([""]),
        workload=get_workload("short-short"),
        policy=build_test_policy(),
    )
    assert result.status is QualityStatus.EXECUTION_FAILURE
    assert result.unacceptable_output is True


def test_structurally_invalid_output_is_rejected() -> None:
    result = validate_quality(
        baseline=_measurement([REFERENCE]),
        candidate=_measurement(["12345 67890 !!!"]),
        workload=get_workload("short-short"),
        policy=build_test_policy(),
    )
    assert result.status is QualityStatus.STRUCTURAL_FAILURE
    assert result.unacceptable_output is True


def test_truncated_output_is_rejected() -> None:
    workload = get_workload("short-long-decode")
    result = validate_quality(
        baseline=_measurement([REFERENCE], tokens=128, workload_id="short-long-decode"),
        candidate=_measurement([REFERENCE], tokens=40, workload_id="short-long-decode"),
        workload=workload,
        policy=build_test_policy(),
    )
    assert result.status is QualityStatus.STRUCTURAL_FAILURE
    assert result.unacceptable_output is True


def test_quality_regression_detected() -> None:
    result = validate_quality(
        baseline=_measurement([REFERENCE]),
        candidate=_measurement(["zzzz qqqq wwww"]),
        workload=get_workload("short-short"),
        policy=build_test_policy(),
    )
    assert result.status is QualityStatus.QUALITY_REGRESSION
    assert result.quality_regression is True
    assert result.unacceptable_output is True


def test_missing_reference_is_unknown_not_fabricated() -> None:
    result = validate_quality(
        baseline=_measurement([""]),
        candidate=_measurement([REFERENCE]),
        workload=get_workload("short-short"),
        policy=build_test_policy(),
    )
    assert result.status is QualityStatus.UNKNOWN
    assert result.reference_available is False
    assert result.quality_regression is False
