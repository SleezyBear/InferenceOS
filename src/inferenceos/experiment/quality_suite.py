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
import json
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
    if workload.required_phrases:
        cases.append(
            QualityCaseSpec(
                case_id="required_content",
                kind=QualityCaseKind.REQUIRED_CONTENT,
                description=(
                    "candidate preserves the exact factual/arithmetic content the "
                    "baseline demonstrated"
                ),
                required=True,
                params={"phrases": list(workload.required_phrases)},
            )
        )
    if workload.required_fields:
        cases.append(
            QualityCaseSpec(
                case_id="required_fields",
                kind=QualityCaseKind.REQUIRED_FIELDS,
                description=(
                    "candidate preserves the required structured-output fields the "
                    "baseline demonstrated"
                ),
                required=True,
                params={"fields": list(workload.required_fields)},
            )
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

    # --- B3 deterministic required content/fields --------------------- #
    required_ok = True
    # Numbers every baseline answer gave must be in every candidate answer: a changed
    # number reads almost the same but is a wrong answer (review, 3 Oct 2026: 2468 ->
    # 9999 passed the similarity check at 0.969).
    numbers_passed, numbers_detail = _numbers_preserved(baseline_refs, candidates)
    results.append(
        QualityCaseResult(
            case_id="numbers_preserved",
            kind=QualityCaseKind.REQUIRED_CONTENT,
            required=True,
            passed=numbers_passed,
            status=None if numbers_passed else QualityStatus.QUALITY_REGRESSION,
            detail=numbers_detail,
        )
    )
    required_ok = required_ok and numbers_passed
    if not numbers_passed:
        reasons.append(numbers_detail)
    for case in spec.cases:
        if case.kind is QualityCaseKind.REQUIRED_CONTENT:
            raw_phrases = case.params.get("phrases")
            phrases = (
                [str(item) for item in raw_phrases]
                if isinstance(raw_phrases, (list, tuple))
                else []
            )
            passed, detail = _required_content(phrases, baseline_refs, candidates)
            results.append(
                QualityCaseResult(
                    case_id=case.case_id,
                    kind=case.kind,
                    required=True,
                    passed=passed,
                    status=None if passed else QualityStatus.QUALITY_REGRESSION,
                    detail=detail,
                )
            )
            required_ok = required_ok and passed
            if not passed:
                reasons.append(detail)
        elif case.kind is QualityCaseKind.REQUIRED_FIELDS:
            raw_fields = case.params.get("fields")
            fields = (
                [str(item) for item in raw_fields]
                if isinstance(raw_fields, (list, tuple))
                else []
            )
            passed, detail = _required_fields(fields, baseline_refs, candidates)
            results.append(
                QualityCaseResult(
                    case_id=case.case_id,
                    kind=case.kind,
                    required=True,
                    passed=passed,
                    status=None if passed else QualityStatus.QUALITY_REGRESSION,
                    detail=detail,
                )
            )
            required_ok = required_ok and passed
            if not passed:
                reasons.append(detail)

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

    # A deterministic required-content/fields failure is a quality regression even when
    # the candidate is semantically similar and performance improved.
    quality_regression = bool(
        reference_available and candidates and not (semantic_ok and required_ok)
    )
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
        validated=bool(execution_ok and not structural_fail and required_ok),
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


def _normalize_ci(text: str) -> str:
    """Case-insensitive whitespace normalization for deterministic content checks.

    Used only for required-content preservation; the exact-match/similarity
    comparison intentionally remains case-sensitive.
    """
    return " ".join(text.split()).lower()


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


def _required_content(
    phrases: list[str], baseline_refs: list[str], candidates: list[str]
) -> tuple[bool, str]:
    """Deterministic preservation of exact factual/arithmetic content.

    Only phrases the baseline itself demonstrated are required, so a model limitation
    is never a candidate failure; a candidate that loses a demonstrated fact is.
    Comparison is normalized (whitespace-collapsed, case-insensitive) so legitimate
    formatting variation does not fail.
    """
    if not phrases:
        return True, "no required content declared"
    demonstrated = [
        phrase
        for phrase in phrases
        if any(_normalize_ci(phrase) in _normalize_ci(ref) for ref in baseline_refs if ref)
    ]
    if not demonstrated:
        return True, "required content not demonstrated by the baseline; not applicable"
    if not candidates:
        return True, "no candidate output (execution case governs)"
    missing = [
        phrase
        for phrase in demonstrated
        if not all(_normalize_ci(phrase) in _normalize_ci(cand) for cand in candidates)
    ]
    if missing:
        return False, f"candidate lost required deterministic content: {missing}"
    return True, f"required deterministic content preserved: {demonstrated}"


#: A number with its sign: a minus or plus sign counts when it starts the number (not
#: the hyphen of a range like "3-4"), so "+500" -> "-500" is a changed number
#: (review, 4 Oct 2026). The Unicode minus is read as "-".
_NUMBER_RE = re.compile(r"(?:(?<![\w.,])[-+\u2212])?\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    return {item.replace("\u2212", "-").lstrip("+") for item in _NUMBER_RE.findall(text)}


def _numbers_preserved(baseline_refs: list[str], candidates: list[str]) -> tuple[bool, str]:
    """Numbers that appear in every baseline answer must appear in every candidate."""
    if not baseline_refs or not candidates:
        return True, "no answers to compare numbers"
    common: set[str] | None = None
    for ref in baseline_refs:
        found = _numbers(ref)
        common = found if common is None else common & found
    if not common:
        return True, "the baseline answers share no numbers"
    missing = sorted(
        number for number in common if not all(number in _numbers(c) for c in candidates)
    )
    if missing:
        return False, f"candidate lost or changed numbers the baseline always gave: {missing}"
    return True, f"numbers preserved: {sorted(common)}"


def _json_keys(text: str) -> set[str] | None:
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(payload, dict):
        return {str(key) for key in payload}
    return None


def _required_fields(
    fields: list[str], baseline_refs: list[str], candidates: list[str]
) -> tuple[bool, str]:
    """Deterministic preservation of required structured-output fields."""
    if not fields:
        return True, "no required fields declared"
    baseline_keys: set[str] = set()
    demonstrated = False
    for ref in baseline_refs:
        keys = _json_keys(ref)
        if keys:
            baseline_keys |= keys
            demonstrated = True
    required = [field for field in fields if field in baseline_keys]
    if not demonstrated or not required:
        return True, "required fields not demonstrated by the baseline; not applicable"
    if not candidates:
        return True, "no candidate output (execution case governs)"
    problems: list[str] = []
    for candidate in candidates:
        keys = _json_keys(candidate)
        if keys is None:
            problems.append("candidate output is not valid JSON")
            break
        missing = [field for field in required if field not in keys]
        if missing:
            problems.append(f"missing required field(s) {missing}")
    if problems:
        return False, "structured output invalid: " + "; ".join(problems)
    return True, f"required structured fields preserved: {required}"
