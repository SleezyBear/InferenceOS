"""User-controlled optimization preference & adaptive-economics contracts.

These contracts separate *product policy* (how much improvement is worth searching
for, and how much search time is allowed) from the *technical search* (is candidate X
actually better?) and from every hard safety gate. The policy can lower the required
worthwhile improvement and expand the bounded search budget, but it can never make an
unsafe/OOM/quality-regressing/invalid candidate acceptable.

Everything here is inert data; the decision functions live in
:mod:`inferenceos.optimization.preference`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import (
    DegradedPerformancePolicy,
    OptimizationMode,
    PerformanceState,
    ResumeVerificationOutcome,
)

PREFERENCE_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class OptimizationPreference(ContractBase):
    """Resolved program-level optimization policy.

    ``minimum_improvement*`` are relative fractions (``0.05`` == 5% measured decode-
    throughput improvement) the economics gate requires before an empirically
    supported candidate is considered worth applying. ``maximum_search_time_s`` is the
    hard, bounded search budget for the NORMAL performance state; per-state
    multipliers scale it when performance is degraded (never above
    ``maximum_search_time_hard_cap_s``).
    """

    schema_version: str = PREFERENCE_SCHEMA_VERSION
    mode: OptimizationMode = OptimizationMode.BALANCED
    minimum_improvement: float = Field(default=0.05, ge=0.0, le=1.0)
    maximum_search_time_s: float = Field(default=240.0, gt=0.0)
    degraded_performance_policy: DegradedPerformancePolicy = (
        DegradedPerformancePolicy.LOWER_REQUIREMENT
    )
    degraded_minimum_improvement: float | None = Field(default=None, ge=0.0, le=1.0)
    severe_minimum_improvement: float | None = Field(default=None, ge=0.0, le=1.0)
    degraded_search_time_multiplier: float = Field(default=1.0, ge=1.0)
    severe_search_time_multiplier: float = Field(default=1.0, ge=1.0)
    maximum_search_time_hard_cap_s: float = Field(default=1200.0, gt=0.0)
    degraded_threshold: float = Field(default=0.25, gt=0.0)
    severe_degraded_threshold: float = Field(default=0.50, gt=0.0)
    custom: bool = False
    source: str = "settings"
    notes: list[str] = Field(default_factory=list)

    def minimum_improvement_for(self, state: PerformanceState) -> float:
        """Required worthwhile relative improvement for a measured performance state."""
        base = float(self.minimum_improvement)
        if self.degraded_performance_policy is DegradedPerformancePolicy.STRICT:
            return base
        if state is PerformanceState.SEVERELY_DEGRADED:
            if self.severe_minimum_improvement is not None:
                return float(self.severe_minimum_improvement)
            return max(0.0, base * 0.25)
        if state is PerformanceState.DEGRADED:
            if self.degraded_minimum_improvement is not None:
                return float(self.degraded_minimum_improvement)
            return max(0.0, base * 0.5)
        return base

    def search_budget_for(self, state: PerformanceState) -> float:
        """Bounded search budget (seconds) for a measured performance state."""
        multiplier = 1.0
        if self.degraded_performance_policy is not DegradedPerformancePolicy.STRICT:
            if state is PerformanceState.SEVERELY_DEGRADED:
                multiplier = self.severe_search_time_multiplier
            elif state is PerformanceState.DEGRADED:
                multiplier = self.degraded_search_time_multiplier
        budget = float(self.maximum_search_time_s) * multiplier
        cap = float(self.maximum_search_time_hard_cap_s)
        return min(budget, cap)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "minimum_improvement": self.minimum_improvement,
            "maximum_search_time_s": self.maximum_search_time_s,
            "degraded_performance_policy": self.degraded_performance_policy.value,
            "degraded_minimum_improvement": self.degraded_minimum_improvement,
            "severe_minimum_improvement": self.severe_minimum_improvement,
            "degraded_search_time_multiplier": self.degraded_search_time_multiplier,
            "severe_search_time_multiplier": self.severe_search_time_multiplier,
            "maximum_search_time_hard_cap_s": self.maximum_search_time_hard_cap_s,
            "degraded_threshold": self.degraded_threshold,
            "severe_degraded_threshold": self.severe_degraded_threshold,
            "custom": self.custom,
            "source": self.source,
            "notes": list(self.notes),
        }


class PerformanceStateAssessment(ContractBase):
    """Explainable classification of the target's current performance state.

    ``degradation_fraction`` is the worst measured relative regression of a workload
    performance metric versus the established baseline. ``measured`` is False (and
    ``state`` is ``UNKNOWN``) when there is insufficient baseline evidence, so the
    policy is applied without guessing.
    """

    state: PerformanceState = PerformanceState.UNKNOWN
    measured: bool = False
    degradation_fraction: float | None = Field(default=None, ge=0.0)
    metric_degradation: dict[str, float] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    recent_metrics: dict[str, float] = Field(default_factory=dict)
    degraded_threshold: float = Field(default=0.25, gt=0.0)
    severe_degraded_threshold: float = Field(default=0.50, gt=0.0)
    rationale: list[str] = Field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "measured": self.measured,
            "degradation_fraction": self.degradation_fraction,
            "metric_degradation": dict(self.metric_degradation),
            "baseline_metrics": dict(self.baseline_metrics),
            "recent_metrics": dict(self.recent_metrics),
            "degraded_threshold": self.degraded_threshold,
            "severe_degraded_threshold": self.severe_degraded_threshold,
            "rationale": list(self.rationale),
        }


class ResumeVerification(ContractBase):
    """Explicit, bounded post-search/post-intervention resume verification.

    Captures the search duration, the configured/effective budget, the bounded timeout
    actually used, retry attempts and the terminal outcome, so a long search can be
    distinguished from a genuine resume failure.
    """

    verification_id: str
    phase: str
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: datetime | None = None
    search_duration_s: float = Field(default=0.0, ge=0.0)
    configured_search_budget_s: float = Field(default=0.0, ge=0.0)
    timeout_s: float = Field(default=0.0, ge=0.0)
    hard_max_s: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=0, ge=0)
    outcome: ResumeVerificationOutcome = ResumeVerificationOutcome.NOT_REQUIRED
    reason: str = ""
    terminal_checks: dict[str, object] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "verification_id": self.verification_id,
            "phase": self.phase,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "search_duration_s": self.search_duration_s,
            "configured_search_budget_s": self.configured_search_budget_s,
            "timeout_s": self.timeout_s,
            "hard_max_s": self.hard_max_s,
            "attempts": self.attempts,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "terminal_checks": dict(self.terminal_checks),
        }


__all__ = [
    "OptimizationPreference",
    "PerformanceStateAssessment",
    "ResumeVerification",
    "PREFERENCE_SCHEMA_VERSION",
]
