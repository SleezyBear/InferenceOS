"""Quality check fixes after the 3 Oct 2026 review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inferenceos.config import Settings
from inferenceos.contracts.enums import QualityStatus
from inferenceos.experiment.quality_policy import build_adaptive_quality_policy
from inferenceos.experiment.quality_suite import build_default_suite, evaluate_suite
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_model
from inferenceos.tune.measure import quality_ok
from tests.experiment_helpers import SampleResult
from tests.test_tune import _settings, _tuner


def _evaluate(baseline: list[str], candidate: list[str]) -> Any:
    settings = Settings()
    workload = get_workload("short-short")
    return evaluate_suite(
        build_default_suite(workload, sample_model(), version=settings.phase7r_suite_version),
        policy=build_adaptive_quality_policy(settings, action_id="runtime.reduce_threads"),
        baseline_texts=baseline,
        candidate_texts=candidate,
        baseline_tokens=[32] * len(baseline),
        candidate_tokens=[32] * len(candidate),
        baseline_valid_runs=len(baseline),
        candidate_valid_runs=len(candidate),
        invalid_runs=0,
        ignore_eos=False,
        n_predict=32,
    )


def test_a_changed_number_is_a_quality_failure() -> None:
    """The reviewer's example: 2468 -> 9999 used to pass at 0.969 similarity."""
    good = "The total of the two orders is 2468 units, shipped on Monday morning."
    bad = "The total of the two orders is 9999 units, shipped on Monday morning."
    result = _evaluate([good, good], [bad, bad])
    assert result.status is QualityStatus.QUALITY_REGRESSION
    assert any("9999" not in reason and "2468" in reason for reason in result.reasons)
    assert _evaluate([good, good], [good, good]).status is QualityStatus.OK


def test_a_changed_sign_is_a_changed_number() -> None:
    """Review 2: +500 -> -500 used to pass. A range hyphen is not a sign."""
    from inferenceos.experiment.quality_suite import _numbers

    assert _numbers("The balance moved by +500 dollars.") == {"500"}
    assert _numbers("The balance moved by -500 dollars.") == {"-500"}
    assert _numbers("The balance moved by \u2212500 dollars.") == {"-500"}
    assert _numbers("Pages 3-4 and v1.2") == {"3", "4", "1.2"}
    good = "The balance moved by +500 dollars after the transfer on Monday."
    bad = "The balance moved by -500 dollars after the transfer on Monday."
    assert _evaluate([good, good], [bad, bad]).status is QualityStatus.QUALITY_REGRESSION
    assert _evaluate([good, good], [good, good]).status is QualityStatus.OK


def test_numbers_that_the_baseline_does_not_always_give_are_not_required() -> None:
    a = "Step 1 stores the keys."
    b = "It stores the keys and values."
    result = _evaluate([a, b], [b, b])
    assert result.status is not QualityStatus.QUALITY_REGRESSION


def test_unknown_quality_is_not_a_pass() -> None:
    from inferenceos.contracts.experiment import QualityValidationResult

    def result(status: QualityStatus) -> QualityValidationResult:
        return QualityValidationResult(
            status=status, validated=True, execution_ok=True, structural_ok=True
        )

    assert not quality_ok(result(QualityStatus.UNKNOWN))
    assert quality_ok(result(QualityStatus.OK))
    assert quality_ok(result(QualityStatus.ACCEPTABLE_VARIATION))


def test_speed_noise_does_not_loosen_the_answer_check() -> None:
    settings = Settings()
    calm = build_adaptive_quality_policy(settings, action_id="runtime.reduce_threads", baseline_cv=0.01)
    noisy = build_adaptive_quality_policy(settings, action_id="runtime.reduce_threads", baseline_cv=0.5)
    assert noisy.effective_similarity == calm.effective_similarity


def test_a_value_that_failed_to_measure_is_not_tried_again(tmp_path: Path) -> None:
    """Refining around the best value must never re-try a value that failed."""
    tried: list[int | None] = []

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        tried.append(options.threads)
        if options.threads == 7:
            return SampleResult(fail=True)
        if options.threads == 8:
            return SampleResult(decode_tps=22.0, total_latency_s=2.0)
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)

    settings = _settings(
        tmp_path, optimization_mode="performance_first", tune_paths="threads", tune_layer_path=False
    )
    tuner, _, _, _ = _tuner(tmp_path, responder, answers=[True], settings=settings)
    result = tuner.run()
    threads = next(item for item in result.rounds if item.get("path") == "threads")
    screened: Any = threads["screened"]
    seven = [entry for entry in screened if entry.get("value") == 7]
    assert len(seven) == 1 and seven[0]["result"] == "could not be measured"
    assert "why" in seven[0]
