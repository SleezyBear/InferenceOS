"""Shared fixtures for Phase 10 runtime-monitoring tests.

The scripted workload/sampler doubles live in ``inferenceos.runtime.verification`` so
the runtime tests share a single implementation.
"""

from __future__ import annotations

from inferenceos.contracts.runtime import RuntimeBudget, RuntimePolicy
from inferenceos.runtime.verification import (
    ScriptedRuntimeSampler as ScriptedSampler,
    ScriptedWorkload,
    baseline_and_pressured_metrics as baseline_and_pressured,
    healthy_metrics,
)


def make_runtime_policy(**overrides: object) -> RuntimePolicy:
    base: dict[str, object] = {
        "budget": RuntimeBudget(
            max_duration_s=300.0,
            telemetry_interval_s=0.5,
            observation_window_size=40,
            baseline_samples=4,
            max_deep_analyses=2,
            max_interventions=2,
            max_cumulative_intervention_s=300.0,
            max_optimizer_overhead_s=120.0,
            approval_timeout_s=10.0,
            max_alerts_per_condition=2,
        ),
        "persistence_samples": 3,
        "condition_lookback_samples": 6,
        "deviation_threshold": 0.25,
        "hysteresis_ratio": 0.5,
        "diagnosis_cooldown_s": 0.0,
        "intervention_cooldown_s": 60.0,
        "require_approval_for_disruptive": True,
    }
    base.update(overrides)
    return RuntimePolicy.model_validate(base)


__all__ = [
    "ScriptedSampler",
    "ScriptedWorkload",
    "baseline_and_pressured",
    "healthy_metrics",
    "make_runtime_policy",
]
