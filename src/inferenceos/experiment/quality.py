"""Quality/correctness validation for an experiment (Phase 7R).

Performance improvement is never sufficient. The engine delegates to a *versioned,
task-relevant* suite (``experiment.quality_suite``) evaluated under an *adaptive*
policy (``experiment.quality_policy``) derived from the diagnosed bottleneck, the
optimization magnitude, the workload, and the baseline variability.

The public ``validate_quality`` function preserves the Phase 7 call signature while
accepting optional context used to derive the tolerance. It returns the existing
``QualityValidationResult`` extended with the full suite/policy evidence.
"""

from __future__ import annotations

import hashlib

from inferenceos.config import Settings
from inferenceos.contracts.enums import EvidenceStrength, QualityStatus
from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentPolicy,
    QualityCheck,
    QualityValidationResult,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.quality import QualitySuiteResult
from inferenceos.experiment.quality_policy import build_adaptive_quality_policy
from inferenceos.experiment.quality_suite import build_default_suite, evaluate_suite
from inferenceos.profiling.workloads import BenchmarkWorkload

_MAX_OUTPUT_CHARS = 200_000


def validate_quality(
    *,
    baseline: ExperimentMeasurement,
    candidate: ExperimentMeasurement,
    workload: BenchmarkWorkload,
    policy: ExperimentPolicy,
    settings: Settings | None = None,
    action_id: str | None = None,
    diagnosed_bottleneck: str | None = None,
    diagnosis_confidence: float = 0.0,
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE,
    optimization_magnitude: float = 0.0,
    model: ModelProfile | None = None,
) -> QualityValidationResult:
    settings = settings or Settings()
    baseline_texts = list(baseline.generated_texts)
    candidate_texts = list(candidate.generated_texts)
    baseline_tokens = [
        int(run.observation.metrics.generated_tokens)
        for run in baseline.runs
        if run.observation.metrics.generated_tokens is not None
    ]
    candidate_tokens = [
        int(run.observation.metrics.generated_tokens)
        for run in candidate.runs
        if run.observation.metrics.generated_tokens is not None
    ]

    baseline_cv = _max_cv(baseline)
    adaptive = build_adaptive_quality_policy(
        settings,
        action_id=action_id,
        bottleneck_kind=diagnosed_bottleneck,
        bottleneck_confidence=diagnosis_confidence,
        evidence_strength=evidence_strength,
        optimization_magnitude=optimization_magnitude,
        workload_kind=workload.kind,
        baseline_cv=baseline_cv,
    )
    suite = build_default_suite(workload, model, version=settings.phase7r_suite_version)
    suite_result = evaluate_suite(
        suite,
        policy=adaptive,
        baseline_texts=baseline_texts,
        candidate_texts=candidate_texts,
        baseline_tokens=baseline_tokens,
        candidate_tokens=candidate_tokens,
        baseline_valid_runs=baseline.valid_runs,
        candidate_valid_runs=candidate.valid_runs,
        invalid_runs=candidate.invalid_runs,
        ignore_eos=workload.ignore_eos,
        n_predict=workload.n_predict,
    )

    checks = [
        QualityCheck(
            check_id=case.case_id,
            description=kind_description(case.case_id),
            passed=case.passed,
            detail=case.detail,
            hard=case.required,
        )
        for case in suite_result.cases
    ]

    reference = _normalize(next((text for text in baseline_texts if text), ""))
    candidate_text = _normalize(next((text for text in candidate_texts if text), ""))
    execution_ok = _case_passed(suite_result, "execution_completed")
    structural_ok = _structural_ok(suite_result)
    return QualityValidationResult(
        status=suite_result.status,
        validated=suite_result.validated,
        execution_ok=execution_ok,
        structural_ok=structural_ok,
        reference_available=suite_result.reference_count > 0,
        deterministic_match=suite_result.exact_match,
        similarity=suite_result.similarity_score,
        unacceptable_output=(
            suite_result.status
            in {
                QualityStatus.EXECUTION_FAILURE,
                QualityStatus.STRUCTURAL_FAILURE,
                QualityStatus.QUALITY_REGRESSION,
            }
        ),
        quality_regression=suite_result.status is QualityStatus.QUALITY_REGRESSION,
        checks=checks,
        rationale=list(suite_result.reasons),
        baseline_text_hash=_hash(reference) if reference else None,
        candidate_text_hash=_hash(candidate_text) if candidate_text else None,
        suite_id=suite.suite_id,
        suite_version=suite.version,
        suite=suite_result,
        policy=adaptive,
        cases=suite_result.cases,
        aggregate_score=suite_result.aggregate_score,
        tolerance=adaptive.effective_similarity,
        tolerance_basis=list(adaptive.tolerance_basis),
    )


def kind_description(case_id: str) -> str:
    mapping = {
        "execution_completed": "every candidate run produced non-empty output",
        "structural_valid": "output is non-empty, alphabetic, and not corrupted",
        "not_truncated": "ignored-EOS generation reached the requested length",
        "deterministic_reference": "candidate output exactly matches a baseline reference",
        "multi_reference_similarity": "candidate is semantically close to the reference",
        "content_coverage": "candidate retains the reference's task-relevant content",
        "length_sanity": "candidate output length is comparable to the reference",
        "required_content": (
            "candidate preserves the exact factual/arithmetic content the baseline "
            "demonstrated"
        ),
        "required_fields": (
            "candidate preserves the required structured-output fields the baseline "
            "demonstrated"
        ),
    }
    return mapping.get(case_id, case_id)


def _structural_ok(suite_result: QualitySuiteResult) -> bool:
    for case in suite_result.cases:
        if case.case_id in {"structural_valid", "not_truncated"} and not case.passed:
            return False
    return True


def _case_passed(suite_result: QualitySuiteResult, case_id: str) -> bool:
    for case in suite_result.cases:
        if case.case_id == case_id:
            return case.passed
    return False


def _max_cv(measurement: ExperimentMeasurement) -> float | None:
    values = [
        stats.cv
        for stats in measurement.metrics.values()
        if stats.cv is not None
    ]
    return max(values) if values else None


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
