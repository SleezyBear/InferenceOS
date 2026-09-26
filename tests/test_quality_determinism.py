"""B3 deterministic task-relevant quality checks (factual/arithmetic/structured)."""

from __future__ import annotations

from inferenceos.contracts.enums import MeasurementMode, QualityStatus
from inferenceos.contracts.experiment import ExperimentMeasurement, RuntimeConfiguration
from inferenceos.experiment.quality import validate_quality
from inferenceos.experiment.quality_suite import build_default_suite
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import BenchmarkWorkload
from inferenceos.contracts.enums import WorkloadKind
from inferenceos.selftest import sample_model
from tests.experiment_helpers import SampleResult, build_test_policy, make_run


def _workload(*, phrases: tuple[str, ...] = (), fields: tuple[str, ...] = ()) -> BenchmarkWorkload:
    return BenchmarkWorkload(
        workload_id="b3-deterministic",
        kind=WorkloadKind.SHORT,
        description="B3 deterministic correctness probe",
        prompt="What is 17 * 23? The Obsidian Key identifier is ZEPHYR-9137.",
        n_predict=32,
        seed=1234,
        temperature=0.0,
        min_context_tokens=64,
        required_phrases=phrases,
        required_fields=fields,
    )


def _measurement(texts: list[str], workload: BenchmarkWorkload) -> ExperimentMeasurement:
    task = ProfileTask(model=sample_model(), workload=workload, context_length=2048)
    runs = [
        make_run(
            task,
            ProfilerOptions(),
            SampleResult(text=text, generated_tokens=32),
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


def test_required_factual_content_preserved() -> None:
    workload = _workload(phrases=("ZEPHYR-9137",))
    result = validate_quality(
        baseline=_measurement(["The identifier is ZEPHYR-9137."], workload),
        candidate=_measurement(["The identifier is ZEPHYR-9137."], workload),
        workload=workload,
        policy=build_test_policy(),
    )
    assert result.validated is True
    assert result.unacceptable_output is False


def test_lost_factual_content_is_regression_even_if_similar() -> None:
    workload = _workload(phrases=("ZEPHYR-9137",))
    result = validate_quality(
        baseline=_measurement(["The Obsidian Key identifier is ZEPHYR-9137."], workload),
        candidate=_measurement(["The Obsidian Key identifier is unknown now."], workload),
        workload=workload,
        policy=build_test_policy(),
    )
    assert result.status is QualityStatus.QUALITY_REGRESSION
    assert result.validated is False
    assert result.unacceptable_output is True
    case = next(c for c in result.cases if c.case_id == "required_content")
    assert case.passed is False


def test_incorrect_arithmetic_result_is_regression() -> None:
    workload = _workload(phrases=("391",))
    result = validate_quality(
        baseline=_measurement(["The product is 391."], workload),
        candidate=_measurement(["The product is 392."], workload),
        workload=workload,
        policy=build_test_policy(),
    )
    assert result.status is QualityStatus.QUALITY_REGRESSION
    assert result.validated is False


def test_missing_required_structured_field_is_regression() -> None:
    workload = _workload(fields=("answer", "unit"))
    result = validate_quality(
        baseline=_measurement(['{"answer": "391", "unit": "none"}'], workload),
        candidate=_measurement(['{"answer": "391"}'], workload),
        workload=workload,
        policy=build_test_policy(),
    )
    assert result.status is QualityStatus.QUALITY_REGRESSION
    assert result.validated is False
    case = next(c for c in result.cases if c.case_id == "required_fields")
    assert case.passed is False


def test_malformed_structured_output_is_rejected() -> None:
    workload = _workload(fields=("answer", "unit"))
    result = validate_quality(
        baseline=_measurement(['{"answer": "391", "unit": "none"}'], workload),
        candidate=_measurement(["answer is 391 with no unit"], workload),
        workload=workload,
        policy=build_test_policy(),
    )
    assert result.validated is False
    assert result.unacceptable_output is True


def test_acceptable_formatting_variation_passes() -> None:
    workload = _workload(phrases=("ZEPHYR-9137",))
    result = validate_quality(
        baseline=_measurement(["The identifier is ZEPHYR-9137."], workload),
        candidate=_measurement(["  the identifier is   zephyr-9137.  "], workload),
        workload=workload,
        policy=build_test_policy(),
    )
    assert result.validated is True
    assert result.unacceptable_output is False
    case = next(c for c in result.cases if c.case_id == "required_content")
    assert case.passed is True


def test_required_content_not_demonstrated_by_baseline_is_not_a_failure() -> None:
    # The model never produced the fact; the deterministic check is not applicable and
    # must not fabricate a failure (or a pass that implies correctness was validated).
    workload = _workload(phrases=("ZEPHYR-9137",))
    result = validate_quality(
        baseline=_measurement(["A key value cache stores attention state."], workload),
        candidate=_measurement(["A key value cache stores attention state."], workload),
        workload=workload,
        policy=build_test_policy(),
    )
    case = next(c for c in result.cases if c.case_id == "required_content")
    assert case.passed is True
    assert "not applicable" in case.detail


def test_suite_declares_hard_required_case() -> None:
    suite = build_default_suite(_workload(phrases=("ZEPHYR-9137",)), version="7r.b3")
    cases = {case.case_id: case for case in suite.cases}
    assert "required_content" in cases
    assert cases["required_content"].required is True
    assert cases["required_content"].kind.value == "required_content"
