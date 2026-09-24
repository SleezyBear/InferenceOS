"""Quality/correctness validation for an experiment.

Performance improvement is never sufficient: an output that failed, is empty,
truncated, structurally invalid, or materially different from the deterministic
reference is treated as a quality regression and blocks success.

The current benchmark workloads are deterministic (fixed prompts, seeds, and
temperature 0), so the baseline output is a real reference. Similarity is reported
with a stable, dependency-free metric; no quality score is fabricated and no
semantic equivalence is claimed beyond what is actually measured.
"""

from __future__ import annotations

import difflib
import hashlib

from inferenceos.contracts.enums import QualityStatus
from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentPolicy,
    QualityCheck,
    QualityValidationResult,
)
from inferenceos.profiling.workloads import BenchmarkWorkload

_MAX_OUTPUT_CHARS = 200_000


def validate_quality(
    *,
    baseline: ExperimentMeasurement,
    candidate: ExperimentMeasurement,
    workload: BenchmarkWorkload,
    policy: ExperimentPolicy,
) -> QualityValidationResult:
    checks: list[QualityCheck] = []
    rationale: list[str] = []

    candidate_texts = [text for text in candidate.generated_texts if text]
    baseline_texts = [text for text in baseline.generated_texts if text]

    execution_ok = candidate.valid_runs > 0 and len(candidate_texts) == candidate.valid_runs
    checks.append(
        QualityCheck(
            check_id="execution_completed",
            description="every candidate run produced non-empty output",
            passed=execution_ok,
            detail=(
                f"valid_runs={candidate.valid_runs} "
                f"invalid_runs={candidate.invalid_runs} "
                f"non_empty={len(candidate_texts)}"
            ),
        )
    )
    if not execution_ok:
        rationale.append("candidate generation did not complete successfully")

    structural_ok = bool(candidate_texts) and all(_structurally_valid(text) for text in candidate_texts)
    checks.append(
        QualityCheck(
            check_id="structural_valid",
            description="output is non-empty, alphabetic, and not corrupted",
            passed=structural_ok,
            detail=f"checked {len(candidate_texts)} output(s)",
        )
    )
    if not structural_ok:
        rationale.append("candidate output was empty or structurally corrupt")

    if workload.ignore_eos:
        lengths = [
            int(run.observation.metrics.generated_tokens)
            for run in candidate.runs
            if run.observation.metrics.generated_tokens is not None
        ]
        truncation_ok = bool(lengths) and all(length >= workload.n_predict for length in lengths)
        checks.append(
            QualityCheck(
                check_id="not_truncated",
                description="ignored-EOS generation reached the requested length",
                passed=truncation_ok,
                detail=f"requested={workload.n_predict} observed={lengths}",
            )
        )
        if not truncation_ok:
            rationale.append("candidate output was truncated")
        structural_ok = structural_ok and truncation_ok

    reference = _normalize(baseline_texts[0]) if baseline_texts else None
    reference_available = reference is not None
    checks.append(
        QualityCheck(
            check_id="reference_available",
            description="a deterministic baseline reference output exists",
            passed=reference_available,
            detail=f"baseline outputs={len(baseline_texts)}",
            hard=False,
        )
    )

    deterministic_match: bool | None = None
    similarity: float | None = None
    quality_regression = False
    if reference is not None and candidate_texts:
        candidate_text = _normalize(candidate_texts[0])
        deterministic_match = candidate_text == reference
        similarity = difflib.SequenceMatcher(None, reference, candidate_text).ratio()
        if deterministic_match:
            rationale.append("candidate output exactly matches the deterministic reference")
        elif similarity >= policy.quality_similarity_min:
            rationale.append(
                f"candidate output differs but is acceptable (similarity {similarity:.2f})"
            )
        else:
            quality_regression = True
            rationale.append(
                f"candidate output diverges from the reference "
                f"(similarity {similarity:.2f} < {policy.quality_similarity_min:.2f})"
            )
        checks.append(
            QualityCheck(
                check_id="reference_match",
                description="candidate output matches the reference within tolerance",
                passed=not quality_regression,
                detail=f"exact={deterministic_match} similarity={similarity:.4f}",
            )
        )

    if not execution_ok:
        status = QualityStatus.EXECUTION_FAILURE
    elif not structural_ok:
        status = QualityStatus.STRUCTURAL_FAILURE
    elif quality_regression:
        status = QualityStatus.QUALITY_REGRESSION
    elif not reference_available:
        status = QualityStatus.UNKNOWN
    elif deterministic_match:
        status = QualityStatus.OK
    else:
        status = QualityStatus.ACCEPTABLE_VARIATION

    unacceptable = (not execution_ok) or (not structural_ok) or quality_regression
    if not reference_available:
        rationale.append("no baseline reference; quality impact is unknown, not assumed good")

    return QualityValidationResult(
        status=status,
        validated=execution_ok and structural_ok,
        execution_ok=execution_ok,
        structural_ok=structural_ok,
        reference_available=reference_available,
        deterministic_match=deterministic_match,
        similarity=similarity,
        unacceptable_output=unacceptable,
        quality_regression=quality_regression,
        checks=checks,
        rationale=rationale,
        baseline_text_hash=_hash(reference) if reference else None,
        candidate_text_hash=_hash(_normalize(candidate_texts[0])) if candidate_texts else None,
    )


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _structurally_valid(text: str) -> bool:
    if not text or len(text) > _MAX_OUTPUT_CHARS:
        return False
    if "\x00" in text:
        return False
    return any(character.isalpha() for character in text)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
