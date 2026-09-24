"""Versioned quality suite + adaptive policy."""

from __future__ import annotations

from inferenceos.config import Settings
from inferenceos.contracts.enums import EvidenceStrength, QualityStatus, WorkloadKind
from inferenceos.contracts.quality import AdaptiveQualityPolicy, QualitySuiteResult
from inferenceos.experiment.quality_policy import build_adaptive_quality_policy
from inferenceos.experiment.quality_suite import build_default_suite, evaluate_suite
from inferenceos.profiling.workloads import get_workload

REFERENCE = "A key value cache stores attention state."
SECOND_REFERENCE = "The cache keeps attention state from previous tokens."


def _policy(**overrides: object) -> AdaptiveQualityPolicy:
    settings = Settings()
    kwargs: dict[str, object] = {
        "action_id": "runtime.right_size_context",
        "bottleneck_kind": "model_footprint",
        "bottleneck_confidence": 0.8,
        "evidence_strength": EvidenceStrength.HIGH,
        "workload_kind": WorkloadKind.SHORT,
    }
    kwargs.update(overrides)
    return build_adaptive_quality_policy(settings, **kwargs)  # type: ignore[arg-type]


def _evaluate(
    candidate_texts: list[str],
    *,
    baseline_texts: list[str] | None = None,
    ignore_eos: bool = False,
    candidate_tokens: list[int] | None = None,
    policy: AdaptiveQualityPolicy | None = None,
) -> QualitySuiteResult:
    workload = get_workload("short-long-decode" if ignore_eos else "short-short")
    suite = build_default_suite(workload, version="7r.test")
    baseline = baseline_texts if baseline_texts is not None else [REFERENCE]
    return evaluate_suite(
        suite,
        policy=policy or _policy(),
        baseline_texts=baseline,
        candidate_texts=candidate_texts,
        baseline_tokens=[32 for _ in baseline],
        candidate_tokens=candidate_tokens or [32 for _ in candidate_texts],
        baseline_valid_runs=len(baseline),
        candidate_valid_runs=len(candidate_texts),
        invalid_runs=0,
        ignore_eos=ignore_eos,
        n_predict=workload.n_predict,
    )


def test_suite_is_versioned_and_has_multiple_cases() -> None:
    suite = build_default_suite(get_workload("short-short"), version="7r.test")
    assert suite.version == "7r.test"
    assert suite.suite_id
    kinds = {case.kind.value for case in suite.cases}
    assert "execution" in kinds
    assert "structural" in kinds
    assert "multi_reference_similarity" in kinds
    assert "content_coverage" in kinds


def test_exact_match_is_ok() -> None:
    result = _evaluate([REFERENCE])
    assert result.status is QualityStatus.OK
    assert result.exact_match is True
    assert result.validated is True


def test_matches_second_reference_not_only_first() -> None:
    result = _evaluate(
        [SECOND_REFERENCE], baseline_texts=[REFERENCE, SECOND_REFERENCE]
    )
    assert result.status is QualityStatus.OK
    assert result.reference_count == 2


def test_acceptable_variation_passes() -> None:
    result = _evaluate(["A key value cache stores the attention state for tokens."])
    assert result.status is QualityStatus.ACCEPTABLE_VARIATION
    assert result.aggregate_score is not None


def test_empty_output_is_execution_failure() -> None:
    result = _evaluate([""])
    assert result.status is QualityStatus.EXECUTION_FAILURE


def test_malformed_output_is_structural_failure() -> None:
    result = _evaluate(["12345 67890 !!!"])
    assert result.status is QualityStatus.STRUCTURAL_FAILURE


def test_truncated_output_is_structural_failure() -> None:
    result = _evaluate(
        [REFERENCE], ignore_eos=True, candidate_tokens=[8]
    )
    assert result.status is QualityStatus.STRUCTURAL_FAILURE


def test_semantic_regression_detected() -> None:
    result = _evaluate(["zzzz qqqq wwww"])
    assert result.status is QualityStatus.QUALITY_REGRESSION


def test_missing_reference_is_unknown() -> None:
    result = _evaluate([REFERENCE], baseline_texts=[""])
    assert result.status is QualityStatus.UNKNOWN
    assert result.reference_count == 0


def test_content_coverage_regression_when_fact_lost() -> None:
    baseline = ["The Obsidian Key cache block identifier is ZEPHYR-9137 exactly."]
    candidate = ["Inference operating systems schedule model execution across hardware."]
    result = _evaluate(candidate, baseline_texts=baseline)
    assert result.status is QualityStatus.QUALITY_REGRESSION
    coverage = next(c for c in result.cases if c.case_id == "content_coverage")
    assert coverage.passed is False


def test_runtime_only_action_relaxes_tolerance() -> None:
    settings = Settings()
    runtime_only = build_adaptive_quality_policy(
        settings, action_id="runtime.reduce_threads", workload_kind=WorkloadKind.SHORT
    )
    semantic = build_adaptive_quality_policy(
        settings, action_id="runtime.right_size_context", workload_kind=WorkloadKind.SHORT
    )
    assert runtime_only.effective_similarity < semantic.effective_similarity
    assert runtime_only.affects_generation_semantics is False
    assert semantic.affects_generation_semantics is True
    assert runtime_only.tolerance_basis


def test_adaptive_policy_is_versioned_and_bounded() -> None:
    policy = _policy(baseline_cv=0.4)
    assert policy.policy_version
    assert 0.40 <= policy.effective_similarity <= 0.95
