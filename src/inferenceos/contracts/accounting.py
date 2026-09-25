"""H4 optimizer self-overhead / net-benefit accounting contracts.

These records consolidate *already measured* optimizer overhead, target-workload
benefit and measurement distortion into one machine-readable assessment. They add no
new timing in the inference hot path; components are populated from existing
``RuntimeOverhead``/``LoopOverhead``/``ResourceImpact`` evidence.

Accounting model (important; avoids double counting):

* ``OverheadAttribution.inclusive_overhead_s`` is the authoritative optimizer overhead
  for the decision. It is **not** recomputed as the sum of components.
* ``components`` are the *exclusive* phase timers already recorded by the runtime/loop.
  If ``components_sum_s`` differs from ``inclusive_overhead_s``, the difference is
  declared via ``nested=True`` and explained in ``notes``; no component is added twice.
* Target inference time is never part of optimizer overhead. It enters only as the
  ``baseline_target_workload_s`` reference used to form the dimensionless
  ``overhead_fraction``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase

ACCOUNTING_SCHEMA_VERSION = "h4.1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NetBenefitStatus:
    """Conservative net-benefit classification (never a claim of profitability).

    ``INSUFFICIENT_EVIDENCE`` is a first-class terminal result: incomplete baseline,
    missing overhead, incomparable units, or material measurement distortion must not
    be converted into a positive result.
    """

    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"
    OVERHEAD_DOMINANT = "overhead_dominant"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class BenefitRelationship(ContractBase):
    """The target-metric relationship the optimization was evaluated against."""

    metric: str
    direction: str = "higher_is_better"  # higher_is_better | lower_is_better
    unit: str = "unknown"
    baseline_value: float | None = None
    measured_value: float | None = None
    absolute_change: float | None = None
    # Positive means improvement in the intended direction; negative is a regression.
    relative_change: float | None = None
    comparable: bool = True
    available: bool = True
    detail: str = ""


class OverheadAttribution(ContractBase):
    """Measured optimizer overhead with an explicit inclusive/exclusive model."""

    accounting_model: str = (
        "inclusive optimizer_overhead_s; exclusive non-overlapping components"
    )
    inclusive_overhead_s: float = Field(default=0.0, ge=0.0)
    components: dict[str, float] = Field(default_factory=dict)
    components_sum_s: float = Field(default=0.0, ge=0.0)
    baseline_target_workload_s: float | None = None
    # optimizer_overhead_s / baseline_target_workload_s (dimensionless).
    overhead_fraction: float | None = None
    decision_count: int = Field(default=0, ge=0)
    nested: bool = False
    notes: list[str] = Field(default_factory=list)


class MeasurementDistortion(ContractBase):
    """Whether observing/optimizing the target materially distorted it."""

    available: bool = False
    metric: str = "decode_tokens_per_sec"
    direction: str = "higher_is_better"
    tolerance: float = Field(default=0.0, ge=0.0)
    cv: float | None = None
    relative_change: float | None = None
    within_tolerance: bool | None = None
    material: bool | None = None
    detail: str = ""


class NetBenefitAssessment(ContractBase):
    """Consolidated "did the benefit outweigh the optimizer's own cost?" result.

    ``benefit_ratio`` and ``overhead_fraction`` are dimensionless and are the only
    quantities compared. No scalar subtraction of heterogeneous units is performed;
    ``net_value`` is populated only when the two quantities share a unit (currently
    never, because per-request benefit and total overhead time are not comparable).
    """

    schema_version: str = ACCOUNTING_SCHEMA_VERSION
    assessment_id: str
    created_at: datetime = Field(default_factory=_utcnow)
    status: str = NetBenefitStatus.INSUFFICIENT_EVIDENCE

    benefit: BenefitRelationship | None = None
    overhead: OverheadAttribution | None = None
    distortion: MeasurementDistortion | None = None

    benefit_ratio: float | None = None
    overhead_fraction: float | None = None
    benefit_min_relative: float = Field(default=0.05, ge=0.0)
    overhead_fraction_max: float = Field(default=0.20, ge=0.0)
    net_value: float | None = None
    net_unit: str | None = None
    comparable: bool = False
    enforceable: bool = False
    reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "NetBenefitAssessment":
        return cls.model_validate_json(data)
