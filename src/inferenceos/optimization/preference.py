"""Resolve the user-controlled optimization preference and classify performance state.

The preference is stored at the application/program level through the existing
:class:`~inferenceos.config.Settings` surface (environment variables / ``.env``), so it
survives application, monitor and resident-server restarts. This module only turns that
configuration into an explicit, inspectable :class:`OptimizationPreference` and
classifies the target's current performance state from *existing* baseline/workload
metrics. It never invents a performance score and never weakens a hard safety gate.
"""

from __future__ import annotations

from collections.abc import Mapping

from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    DegradedPerformancePolicy,
    OptimizationMode,
    PerformanceState,
)
from inferenceos.contracts.optimization import OptimizationCandidate
from inferenceos.contracts.preference import (
    OptimizationPreference,
    PerformanceStateAssessment,
)

# Metrics carrying *workload performance*, reused from the existing runtime baseline.
# ``(higher_is_better, direction)``: a positive relative change against the "better"
# direction is a degradation.
_WORKLOAD_PERFORMANCE_METRICS: dict[str, bool] = {
    "decode_tokens_per_sec": True,
    "prompt_tokens_per_sec": True,
    "total_latency_s": False,
    "ttft_s": False,
}

# Per-mode economics defaults. ``minimum_improvement*`` are relative fractions; search
# multipliers scale the operator's configured search time limit.
_MODE_DEFAULTS: dict[OptimizationMode, dict[str, float]] = {
    OptimizationMode.BALANCED: {
        "minimum_improvement": 0.05,
        "degraded_minimum_improvement": 0.03,
        "severe_minimum_improvement": 0.01,
        "search_multiplier": 1.0,
        "degraded_search_time_multiplier": 1.25,
        "severe_search_time_multiplier": 1.5,
        "experiment_cost_multiplier": 1.0,
    },
    OptimizationMode.PERFORMANCE_FIRST: {
        "minimum_improvement": 0.015,
        "degraded_minimum_improvement": 0.005,
        "severe_minimum_improvement": 0.0,
        "search_multiplier": 1.5,
        "degraded_search_time_multiplier": 1.5,
        "severe_search_time_multiplier": 2.0,
        "experiment_cost_multiplier": 2.0,
    },
    OptimizationMode.MINIMIZE_OPTIMIZATION_COST: {
        "minimum_improvement": 0.15,
        "degraded_minimum_improvement": 0.10,
        "severe_minimum_improvement": 0.05,
        "search_multiplier": 0.5,
        "degraded_search_time_multiplier": 1.0,
        "severe_search_time_multiplier": 1.0,
        "experiment_cost_multiplier": 0.5,
    },
}


def _resolve_mode(value: str) -> OptimizationMode:
    try:
        return OptimizationMode(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in OptimizationMode)
        raise ValueError(
            f"invalid optimization_mode {value!r}; expected one of: {valid}"
        ) from exc


def build_optimization_preference(settings: Settings) -> OptimizationPreference:
    """Build the resolved preference from the persisted program-level settings.

    Raises ``ValueError`` for an unknown mode or invalid custom values so configuration
    errors fail before monitoring begins rather than silently falling back.
    """
    mode = _resolve_mode(settings.optimization_mode)
    hard_cap = float(settings.optimization_max_search_time_hard_cap_s)
    if hard_cap <= 0.0:
        raise ValueError("optimization_max_search_time_hard_cap_s must be positive")
    degraded_threshold = float(settings.optimization_degraded_threshold)
    severe_threshold = float(settings.optimization_severe_degraded_threshold)
    if not (0.0 < degraded_threshold < severe_threshold):
        raise ValueError(
            "optimization_degraded_threshold must be greater than 0 and less than "
            "optimization_severe_degraded_threshold"
        )
    policy = _resolve_degraded_policy(settings.optimization_degraded_performance_policy)
    configured_limit = max(1.0, float(settings.phase10_gpu_search_time_limit_s))
    base_default = _MODE_DEFAULTS.get(mode)
    notes: list[str] = []

    if mode is OptimizationMode.CUSTOM:
        return _build_custom(
            settings,
            hard_cap=hard_cap,
            degraded_threshold=degraded_threshold,
            severe_threshold=severe_threshold,
            policy=policy,
        )

    assert base_default is not None  # non-custom modes are all in the table
    maximum_search_time_s = min(
        configured_limit * base_default["search_multiplier"], hard_cap
    )
    maximum_experiment_cost_s = max(
        1.0,
        float(settings.optimization_experiment_cost_tolerance_s)
        * base_default["experiment_cost_multiplier"],
    )
    preference = OptimizationPreference(
        mode=mode,
        minimum_improvement=base_default["minimum_improvement"],
        maximum_search_time_s=maximum_search_time_s,
        maximum_experiment_cost_s=maximum_experiment_cost_s,
        degraded_performance_policy=policy,
        degraded_minimum_improvement=base_default["degraded_minimum_improvement"],
        severe_minimum_improvement=base_default["severe_minimum_improvement"],
        degraded_search_time_multiplier=base_default["degraded_search_time_multiplier"],
        severe_search_time_multiplier=base_default["severe_search_time_multiplier"],
        maximum_search_time_hard_cap_s=hard_cap,
        degraded_threshold=degraded_threshold,
        severe_degraded_threshold=severe_threshold,
        custom=False,
        source="settings.optimization_mode",
        notes=notes,
    )
    return preference


def _resolve_degraded_policy(value: str) -> DegradedPerformancePolicy:
    try:
        return DegradedPerformancePolicy(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in DegradedPerformancePolicy)
        raise ValueError(
            f"invalid optimization_degraded_performance_policy {value!r}; "
            f"expected one of: {valid}"
        ) from exc


def _build_custom(
    settings: Settings,
    *,
    hard_cap: float,
    degraded_threshold: float,
    severe_threshold: float,
    policy: DegradedPerformancePolicy,
) -> OptimizationPreference:
    minimum = settings.optimization_minimum_improvement_percent
    maximum = settings.optimization_maximum_search_time_s
    if minimum is None:
        raise ValueError(
            "optimization_mode=custom requires optimization_minimum_improvement_percent"
        )
    if maximum is None:
        raise ValueError(
            "optimization_mode=custom requires optimization_maximum_search_time_s"
        )
    if not (0.0 < float(minimum) <= 100.0):
        raise ValueError(
            "optimization_minimum_improvement_percent must be greater than 0 and at "
            "most 100"
        )
    if float(maximum) <= 0.0:
        raise ValueError("optimization_maximum_search_time_s must be positive")
    notes = [
        "custom policy: minimum_improvement_percent="
        f"{float(minimum):g}, maximum_search_time_s={float(maximum):g}"
    ]
    bounded = min(float(maximum), hard_cap)
    if bounded < float(maximum):
        notes.append(
            f"custom maximum_search_time_s capped at the configured hard maximum {hard_cap:g}s"
        )
    degraded = settings.optimization_custom_degraded_minimum_improvement_percent
    severe = settings.optimization_custom_severe_minimum_improvement_percent
    if degraded is not None and not (0.0 <= float(degraded) <= 100.0):
        raise ValueError(
            "optimization_custom_degraded_minimum_improvement_percent must be between 0 "
            "and 100"
        )
    if severe is not None and not (0.0 <= float(severe) <= 100.0):
        raise ValueError(
            "optimization_custom_severe_minimum_improvement_percent must be between 0 "
            "and 100"
        )
    return OptimizationPreference(
        mode=OptimizationMode.CUSTOM,
        minimum_improvement=float(minimum) / 100.0,
        maximum_search_time_s=bounded,
        maximum_experiment_cost_s=max(
            1.0, float(settings.optimization_experiment_cost_tolerance_s)
        ),
        degraded_performance_policy=policy,
        degraded_minimum_improvement=(
            None if degraded is None else float(degraded) / 100.0
        ),
        severe_minimum_improvement=None if severe is None else float(severe) / 100.0,
        degraded_search_time_multiplier=float(
            settings.optimization_custom_degraded_search_time_multiplier
        ),
        severe_search_time_multiplier=float(
            settings.optimization_custom_severe_search_time_multiplier
        ),
        maximum_search_time_hard_cap_s=hard_cap,
        degraded_threshold=degraded_threshold,
        severe_degraded_threshold=severe_threshold,
        custom=True,
        source="settings.optimization_mode=custom",
        notes=notes,
    )


def classify_performance_state(
    *,
    baseline_metrics: Mapping[str, float | None],
    recent_metrics: Mapping[str, float | None],
    preference: OptimizationPreference,
) -> PerformanceStateAssessment:
    """Classify NORMAL/DEGRADED/SEVERELY_DEGRADED/UNKNOWN from measured metrics.

    Only existing workload-performance metrics are used. When no workload-performance
    baseline/recent pair is available the result is ``UNKNOWN`` and the caller falls
    back to the selected user policy (no guessing).
    """
    baseline: dict[str, float] = {}
    recent: dict[str, float] = {}
    degradation: dict[str, float] = {}
    for metric in _WORKLOAD_PERFORMANCE_METRICS:
        base_value = _as_float(baseline_metrics.get(metric))
        recent_value = _as_float(recent_metrics.get(metric))
        if base_value is None or recent_value is None or base_value == 0.0:
            continue
        higher_is_better = _WORKLOAD_PERFORMANCE_METRICS[metric]
        raw = (recent_value - base_value) / abs(base_value)
        # Positive ``worse`` means a regression relative to the baseline: a throughput
        # drop or a latency rise.
        worse = -raw if higher_is_better else raw
        baseline[metric] = base_value
        recent[metric] = recent_value
        degradation[metric] = worse

    if not degradation:
        return PerformanceStateAssessment(
            state=PerformanceState.UNKNOWN,
            measured=False,
            degradation_fraction=None,
            metric_degradation={},
            baseline_metrics=baseline,
            recent_metrics=recent,
            degraded_threshold=preference.degraded_threshold,
            severe_degraded_threshold=preference.severe_degraded_threshold,
            rationale=[
                "insufficient measured workload-performance baseline evidence; the "
                "selected optimization policy is applied without guessing a state",
            ],
        )

    worst = max(degradation.values())
    worst_metric = max(degradation, key=lambda key: degradation[key])
    if worst >= preference.severe_degraded_threshold:
        state = PerformanceState.SEVERELY_DEGRADED
    elif worst >= preference.degraded_threshold:
        state = PerformanceState.DEGRADED
    else:
        state = PerformanceState.NORMAL
    return PerformanceStateAssessment(
        state=state,
        measured=True,
        degradation_fraction=max(0.0, worst),
        metric_degradation=degradation,
        baseline_metrics=baseline,
        recent_metrics=recent,
        degraded_threshold=preference.degraded_threshold,
        severe_degraded_threshold=preference.severe_degraded_threshold,
        rationale=[
            f"worst measured workload-performance change {worst:+.1%} on {worst_metric} "
            f"vs baseline; degraded>={preference.degraded_threshold:.0%}, "
            f"severe>={preference.severe_degraded_threshold:.0%}",
            f"classified performance_state={state.value}",
        ],
    )


def _as_float(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def candidate_measured_improvement(candidate: OptimizationCandidate | None) -> float | None:
    """The candidate's empirically-known relative *performance* improvement, if any.

    Only throughput-style benefits (``decode_tokens_per_sec`` /
    ``prompt_tokens_per_sec``, increasing) are reported as a relative fraction;
    byte-valued memory benefits are not a comparable relative-performance improvement,
    so ``None`` is returned and the economics gate keeps its existing behaviour instead
    of comparing incomparable units.
    """
    if candidate is None:
        return None
    benefit = candidate.benefit
    if not benefit.known:
        return None
    if benefit.metric not in {"decode_tokens_per_sec", "prompt_tokens_per_sec"}:
        return None
    if benefit.direction != "increase":
        return None
    values = [
        float(value)
        for value in (benefit.magnitude_low, benefit.magnitude_high)
        if value is not None
    ]
    if not values:
        return None
    return max(values)


__all__ = [
    "build_optimization_preference",
    "candidate_measured_improvement",
    "classify_performance_state",
]
