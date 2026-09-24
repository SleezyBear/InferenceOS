"""Phase 7R quality-validation contracts.

A *versioned* quality suite defines a set of deterministic, task-relevant checks
(multiple reference cases, structural validity, semantic comparison, length sanity),
and an *adaptive* policy derives the tolerance from the diagnosed/optimization
context rather than applying one universal string-similarity threshold.

These contracts are additive: the Phase 7 ``QualityValidationResult`` remains the
engine-facing result and now carries the suite evaluation for auditability.
"""

from __future__ import annotations

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import QualityStatus, StrEnum

QUALITY_SUITE_SCHEMA_VERSION = "7r.1"


class QualityCaseKind(StrEnum):
    EXECUTION = "execution"
    STRUCTURAL = "structural"
    TRUNCATION = "truncation"
    DETERMINISTIC_REFERENCE = "deterministic_reference"
    MULTI_REFERENCE_SIMILARITY = "multi_reference_similarity"
    CONTENT_COVERAGE = "content_coverage"
    LENGTH_SANITY = "length_sanity"


class QualityCaseSpec(ContractBase):
    """One explicit, inspectable validation case in a versioned suite."""

    case_id: str
    kind: QualityCaseKind
    description: str
    required: bool = True
    threshold: float | None = None
    params: dict[str, object] = Field(default_factory=dict)


class QualitySuiteSpec(ContractBase):
    """A deterministic, versioned quality-validation suite for a task/workload."""

    schema_version: str = QUALITY_SUITE_SCHEMA_VERSION
    suite_id: str
    version: str
    task_kind: str
    description: str
    cases: list[QualityCaseSpec] = Field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "version": self.version,
            "task_kind": self.task_kind,
            "description": self.description,
            "cases": [
                {
                    "case_id": case.case_id,
                    "kind": case.kind.value,
                    "required": case.required,
                    "threshold": case.threshold,
                }
                for case in self.cases
            ],
        }


class QualityCaseResult(ContractBase):
    case_id: str
    kind: QualityCaseKind
    required: bool
    passed: bool
    status: QualityStatus | None = None
    score: float | None = None
    threshold: float | None = None
    detail: str = ""


class QualitySuiteResult(ContractBase):
    suite_id: str
    version: str
    status: QualityStatus
    validated: bool
    exact_match: bool = False
    aggregate_score: float | None = None
    similarity_score: float | None = None
    coverage_score: float | None = None
    reference_count: int = 0
    cases: list[QualityCaseResult] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class AdaptiveQualityPolicy(ContractBase):
    """Tolerance derived from the diagnosed/optimization context (versioned)."""

    policy_version: str = QUALITY_SUITE_SCHEMA_VERSION
    base_similarity: float
    effective_similarity: float
    content_overlap_min: float
    length_ratio_min: float
    length_ratio_max: float
    affects_generation_semantics: bool
    changes_model_behavior: bool
    bottleneck_kind: str | None = None
    bottleneck_severity: float = 0.0
    optimization_magnitude: float = 0.0
    workload_kind: str = "unknown"
    baseline_cv: float | None = None
    tolerance_basis: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return self.model_dump()
