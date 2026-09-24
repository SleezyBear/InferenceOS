"""Versioned, task-relevant quality-validation suite.

The suite replaces the previous single-output ``difflib`` check with an explicit set
of deterministic cases evaluated against **all** reference outputs produced by the
baseline runs:

* execution completion (catastrophic generation failure vs. normal output)
* structural validity (malformed/corrupt output)
* truncation (when EOS is ignored)
* exact deterministic reference match
* multi-reference similarity (best match across every baseline output)
* task-relevant content coverage (shared content words)
* output-length sanity

All comparisons are deterministic and dependency-free; no external judge or API is
used. The suite version and every case threshold/score are recorded.
"""

from __future__ import annotations

import difflib
import re

from inferenceos.contracts.enums import QualityStatus
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.quality import (
    AdaptiveQualityPolicy,
    QualityCaseKind,
    QualityCaseResult,
    QualityCaseSpec,
    QualitySuiteResult,
    QualitySuiteSpec,
)
from inferenceos.profiling.workloads import BenchmarkWorkload

_SUITE_ID = "inferenceos.quality.task-relevant"
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "which", "what", "when", "where", "into", "than", "then", "them", "they",
        "its", "not", "but", "can", "does", "did", "has", "have", "had", "you",
        "your", "one", "two", "use", "used", "using", "each", "all", "any", "our",
        "their", "there", "these", "those", "such", "over", "under", "more",
    }
)


def build_default_suite(
    workload: BenchmarkWorkload,
    model: ModelProfile | None = None,
    *,
    version: str = "7r.1",
) -> QualitySuiteSpec:
    """Build the deterministic validation suite appropriate to the task/workload."""
    cases: list[QualityCaseSpec] = [
        QualityCaseSpec(
            case_id="execution_completed",
            kind=QualityCaseKind.EXECUTION,
            description="every candidate run produced non-empty output",
            required=True,
        ),
        QualityCaseSpec(
            case_id="structural_valid",
            kind=QualityCaseKind.STRUCTURAL,
            description="output is non-empty, alphabetic, and not corrupted",
            required=True,
        ),
        QualityCaseSpec(
            case_id="deterministic_reference",
            kind=QualityCaseKind.DETERMINISTIC_REFERENCE,
            description="candidate output exactly matches a baseline reference",
            required=False,
        ),
        QualityCaseSpec(
            case_id="multi_reference_similarity",
            kind=QualityCaseKind.MULTI_REFERENCE_SIMILARITY,
            description="candidate is semantically close to some baseline reference",
            required=True,
        ),
        QualityCaseSpec(
            case_id="content_coverage",
            kind=QualityCaseKind.CONTENT_COVERAGE,
            description="candidate retains the reference's task-relevant content words",
            required=True,
        ),
        QualityCaseSpec(
            case_id="length_sanity",
            kind=QualityCaseKind.LENGTH_SANITY,
            description="candidate output length is comparable to the reference",
            required=True,
        ),
    ]
    if workload.ignore_eos:
        cases.insert(
            2,
            QualityCaseSpec(
                case_id="not_truncated",
                kind=QualityCaseKind.TRUNCATION,
                description="ignored-EOS generation reached the requested length",
                required=True,
                params={"n_predict": workload.n_predict},
            ),
        )
    return QualitySuiteSpec(
        suite_id=_SUITE_ID,
        version=version,
        task_kind=workload.kind.value,
        description=(
            f"task-relevant quality validation for workload '{workload.workload_id}' "
            f"({workload.description})"
        ),
        cases=cases,
    )


def evaluate_suite(
    spec: QualitySuiteSpec,
    *,
    policy: AdaptiveQualityPolicy,
    baseline_texts: list[str],
    candidate_texts: list[str],
    baseline_tokens: list[int],
    candidate_tokens: list[int],
    baseline_valid_runs: int,
    candidate_valid_runs: int,
    invalid_runs: int,
    ignore_eos: bool,
    n_predict: int,
) -> QualitySuiteResult:
    baseline_refs = [text for text in baseline_texts if text]
    candidates = [text for text in candidate_texts if text]
    results: list[QualityCaseResult] = []
    reasons: list[str] = []

    execution_ok = candidate_valid_runs > 0 and len(candidates) == candidate_valid_runs
    results.append(
        QualityCaseResult(
            case_id="execution_completed",
            kind=QualityCaseKind.EXECUTION,
            required=True,
            passed=execution_ok,
            status=QualityStatus.EXECUTION_FAILURE if not execution_ok else None,
            score=(len(candidates) / candidate_valid_runs) if candidate_valid_runs else 0.0,
            detail=(
                f"valid_runs={candidate_valid_runs} invalid_runs={invalid_runs} "
                f"non_empty={len(candidates)}"
            ),
        )
    )
    if not execution_ok:
        reasons.append("candidate generation did not complete successfully")

    structural_ok = bool(candidates) and all(_structurally_valid(text) for text in candidates)
    truncation_ok = True
    if ignore_eos:
        lengths = [int(token) for token in candidate_tokens]
        truncation_ok = bool(lengths) and all(length >= n_predict for length in lengths)
    structural_fail = not structural_ok or not truncation_ok
    results.append(
        QualityCaseResult(
            case_id="structural_valid",
            kind=QualityCaseKind.STRUCTURAL,
            required=True,
            passed=structural_ok,
            status=QualityStatus.STRUCTURAL_FAILURE if not structural_ok else None,
            detail=f"checked {len(candidates)} output(s)",
        )
    )
    if ignore_eos:
        results.append(
            QualityCaseResult(
                case_id="not_truncated",
                kind=QualityCaseKind.TRUNCATION,
                required=True,
                passed=truncation_ok,
                status=QualityStatus.STRUCTURAL_FAILURE if not truncation_ok else None,
                threshold=float(n_predict),
                score=float(min(candidate_tokens)) if candidate_tokens else None,
                detail=f"requested={n_predict} observed={candidate_tokens}",
            )
        )
    if not structural_ok:
        reasons.append("candidate output was empty or structurally corrupt")
    if not truncation_ok:
        reasons.append("candidate output was truncated before the requested length")

    reference_available = bool(baseline_refs)
    exact_match = False
    similarity_score: float | None = None
    coverage_score: float | None = None
    length_ok = True
    semantic_ok = False

    if reference_available and candidates:
        per_candidate_best = [
            max(difflib.SequenceMatcher(None, _normalize(ref), _normalize(cand)).ratio()
                for ref in baseline_refs)
            for cand in candidates
        ]
        similarity_score = sum(per_candidate_best) / len(per_candidate_best)
        exact_match = all(
            _normalize(cand) in {_normalize(ref) for ref in baseline_refs}
            for cand in candidates
        )
        reference_words = _content_words(" ".join(baseline_refs))
        candidate_words = _content_words(" ".join(candidates))
        coverage_score = (
            len(reference_words & candidate_words) / len(reference_words)
            if reference_words
            else 1.0
        )
        base_len = _mean_length(baseline_refs)
        cand_len = _mean_length(candidates)
        ratio = (cand_len / base_len) if base_len else 1.0
        length_ok = policy.length_ratio_min <= ratio <= policy.length_ratio_max

        similarity_ok = exact_match or similarity_score >= policy.effective_similarity
        coverage_ok = coverage_score >= policy.content_overlap_min
        semantic_ok = similarity_ok and coverage_ok

        results.append(
            QualityCaseResult(
                case_id="deterministic_reference",
                kind=QualityCaseKind.DETERMINISTIC_REFERENCE,
                required=False,
                passed=exact_match,
                score=1.0 if exact_match else similarity_score,
                detail=f"exact={exact_match} best_mean_similarity={similarity_score:.4f}",
            )
        )
        results.append(
            QualityCaseResult(
                case_id="multi_reference_similarity",
                kind=QualityCaseKind.MULTI_REFERENCE_SIMILARITY,
                required=True,
                passed=similarity_ok,
                threshold=0.0 if exact_match else policy.effective_similarity,
                score=similarity_score,
                detail=(
                    f"mean best similarity over {len(baseline_refs)} reference(s)="
                    f"{similarity_score:.4f} (tolerance {policy.effective_similarity:.3f})"
                ),
            )
        )
        results.append(
            QualityCaseResult(
                case_id="content_coverage",
                kind=QualityCaseKind.CONTENT_COVERAGE,
                required=True,
                passed=coverage_ok,
                threshold=policy.content_overlap_min,
                score=coverage_score,
                detail=(
                    f"content-word coverage={coverage_score:.4f} "
                    f"(floor {policy.content_overlap_min:.3f})"
                ),
            )
        )
        results.append(
            QualityCaseResult(
                case_id="length_sanity",
                kind=QualityCaseKind.LENGTH_SANITY,
                required=True,
                passed=length_ok,
                threshold=policy.length_ratio_max,
                score=ratio,
                detail=(
                    f"length ratio={ratio:.3f} "
                    f"(band {policy.length_ratio_min:.2f}-{policy.length_ratio_max:.2f})"
                ),
            )
        )
        if exact_match:
            reasons.append("candidate output exactly matches a deterministic reference")
        elif similarity_ok and coverage_ok:
            reasons.append(
                f"candidate varies acceptably (similarity {similarity_score:.3f}, "
                f"coverage {coverage_score:.3f})"
            )
        else:
            reasons.append(
                f"candidate diverges from the reference (similarity "
                f"{similarity_score:.3f} vs {policy.effective_similarity:.3f}, coverage "
                f"{coverage_score:.3f} vs {policy.content_overlap_min:.3f})"
            )
    else:
        results.append(
            QualityCaseResult(
                case_id="deterministic_reference",
                kind=QualityCaseKind.DETERMINISTIC_REFERENCE,
                required=False,
                passed=False,
                detail=f"baseline references={len(baseline_refs)}",
            )
        )

    quality_regression = bool(reference_available and candidates and not semantic_ok)
    if not reference_available:
        status = QualityStatus.UNKNOWN
    elif not execution_ok:
        status = QualityStatus.EXECUTION_FAILURE
    elif structural_fail:
        status = QualityStatus.STRUCTURAL_FAILURE
    elif quality_regression:
        status = QualityStatus.QUALITY_REGRESSION
    elif exact_match:
        status = QualityStatus.OK
    else:
        status = QualityStatus.ACCEPTABLE_VARIATION

    if not reference_available:
        reasons.append("no baseline reference; quality impact is unknown, not assumed good")

    return QualitySuiteResult(
        suite_id=spec.suite_id,
        version=spec.version,
        status=status,
        validated=bool(execution_ok and not structural_fail),
        exact_match=exact_match,
        aggregate_score=(
            min(similarity_score, coverage_score)
            if similarity_score is not None and coverage_score is not None
            else None
        ),
        similarity_score=similarity_score,
        coverage_score=coverage_score,
        reference_count=len(baseline_refs),
        cases=results,
        reasons=reasons,
    )


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _content_words(text: str) -> set[str]:
    words = {match.group(0).lower() for match in _WORD_RE.finditer(text)}
    return {word for word in words if len(word) >= 3 and word not in _STOPWORDS}


def _mean_length(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(text) for text in texts) / len(texts)


def _structurally_valid(text: str) -> bool:
    if not text or len(text) > 200_000:
        return False
    if "\x00" in text:
        return False
    return any(character.isalpha() for character in text)
