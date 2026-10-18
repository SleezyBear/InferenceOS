"""Resource-aware optimization search contracts.

These contracts describe the *plan and evidence* for a bounded, empirical search over
one optimization variable (currently GPU-offload layer count). They are deliberately
separate from the Phase 6 candidate vocabulary and the Phase 7 single-candidate
experiment: a search inspects the resource envelope, constructs a safe feasible range,
samples deterministic candidate points, evaluates each against the current
configuration, and selects exactly one value only when the measurements justify it.

Nothing here fabricates a value. An unavailable metric is ``None``; an insufficient
envelope is an explicit ``sufficient=False`` with reasons; ``NO_CHANGE`` is a
first-class terminal outcome.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import (
    CandidateEvaluationStatus,
    SearchSelectionKind,
)
from inferenceos.contracts.profiling import MetricStats

SEARCH_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GpuOffloadResourceEnvelope(ContractBase):
    """The real resource accounting a GPU-offload search reasons over.

    All ``*_bytes`` values are estimates *unless* telemetry-backed; the
    ``estimated``/``basis`` fields state which. ``sufficient`` is False whenever a
    required input (total VRAM, used VRAM telemetry, model geometry, or the
    configured context length) is missing or stale, in which case no higher
    configuration may be recommended.
    """

    sufficient: bool
    reasons: list[str] = Field(default_factory=list)

    total_vram_bytes: int | None = None
    used_vram_bytes: int | None = None
    free_vram_bytes: int | None = None
    other_consumers_bytes: int | None = None
    safety_headroom_bytes: int | None = None
    runtime_reserve_bytes: int | None = None
    kv_cache_bytes: int | None = None
    per_layer_weight_bytes: int | None = None
    weight_bytes_estimate: int | None = None

    context_length: int | None = None
    current_layers: int | None = None
    # Transformer blocks, excluding the separate output step. The feasible range
    # may reach total_layers + 1; engine offload totals already include that step.
    total_layers: int | None = None
    feasible_min: int | None = None
    feasible_max: int | None = None

    telemetry_available: bool = False
    telemetry_fresh: bool = False
    telemetry_source: str | None = None
    telemetry_observed_at: datetime | None = None

    estimated: bool = True
    basis: list[str] = Field(default_factory=list)


class MeasurementAttemptEvidence(ContractBase):
    """One attempt at measuring a candidate (a retry is a new attempt).

    Every attempt is preserved, including invalid/contaminated ones, so the final
    candidate result is auditable and retries cannot hide a bad measurement.
    """

    attempt: int = Field(ge=1)
    status: CandidateEvaluationStatus
    valid: bool
    safe: bool
    samples: int = Field(default=0, ge=0)
    warmup_runs: int = Field(default=0, ge=0)
    decode_tokens_per_sec: float | None = None
    total_latency_s: float | None = None
    reasons: list[str] = Field(default_factory=list)


class CandidatePointEvaluation(ContractBase):
    """One empirically measured candidate configuration."""

    layers: int = Field(ge=0)
    evaluated: bool = False
    status: CandidateEvaluationStatus = CandidateEvaluationStatus.INVALID
    valid: bool = False
    safe: bool = False
    samples: int = Field(default=0, ge=0)
    warmup_runs: int = Field(default=0, ge=0)
    metrics: dict[str, MetricStats] = Field(default_factory=dict)
    decision: str | None = None
    reasons: list[str] = Field(default_factory=list)
    measured_vram_bytes: int | None = None
    # Measured relative change of the primary objective (decode throughput) vs baseline.
    relative_improvement: float | None = None
    # Measurement-robustness evidence (attempts/retries, stabilization, contamination).
    attempts: int = Field(default=1, ge=0)
    valid_attempts: int = Field(default=0, ge=0)
    invalid_attempts: int = Field(default=0, ge=0)
    attempt_records: list[MeasurementAttemptEvidence] = Field(default_factory=list)
    stabilized: bool = True
    contamination_reasons: list[str] = Field(default_factory=list)


class GpuOffloadSelection(ContractBase):
    """The exactly-one-value (or NO_CHANGE) terminal selection."""

    kind: SearchSelectionKind
    selected_layers: int | None = None
    baseline_layers: int
    winning_candidate_id: str | None = None
    reason: str
    comparisons: list[str] = Field(default_factory=list)


class GpuOffloadSearchResult(ContractBase):
    """Full, auditable record of a GPU-offload search."""

    schema_version: str = SEARCH_SCHEMA_VERSION
    search_id: str
    created_at: datetime = Field(default_factory=_utcnow)

    model_ref: str | None = None
    context_length: int | None = None
    resource: GpuOffloadResourceEnvelope

    sampled_layers: list[int] = Field(default_factory=list)
    requested_layers: list[int] = Field(default_factory=list)
    points: list[CandidatePointEvaluation] = Field(default_factory=list)
    selection: GpuOffloadSelection

    overhead_s: float = Field(default=0.0, ge=0.0)
    environment_restored: bool = True
    environments_mutated: bool = False
    limitations: list[str] = Field(default_factory=list)
    summary: str = ""

    # Search-cost / policy accounting (explicit; never hidden). ``overhead_s`` above is
    # the actual measured search time; ``search_budget_s`` is the configured maximum the
    # user policy allowed for this search.
    optimization_mode: str | None = None
    performance_state: str | None = None
    search_budget_s: float | None = Field(default=None, ge=0.0)
    estimated_search_s: float | None = Field(default=None, ge=0.0)
    candidates_evaluated: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "GpuOffloadSearchResult":
        return cls.model_validate_json(data)
