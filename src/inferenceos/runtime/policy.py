"""Phase 10 runtime policy construction from validated configuration.

All runtime budgets/cooldowns/thresholds come from settings; invalid configuration
fails here (pydantic validation) before any monitoring starts.
"""

from __future__ import annotations

from inferenceos.config import Settings
from inferenceos.contracts.enums import EvidenceStrength
from inferenceos.contracts.runtime import (
    InterventionCostModel,
    RuntimeBudget,
    RuntimePolicy,
)


def build_runtime_budget(
    settings: Settings,
    *,
    max_duration_s: float | None = None,
    telemetry_interval_s: float | None = None,
) -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=(
            max_duration_s if max_duration_s is not None else settings.phase10_max_duration_s
        ),
        telemetry_interval_s=(
            telemetry_interval_s
            if telemetry_interval_s is not None
            else settings.phase10_telemetry_interval_s
        ),
        observation_window_size=settings.phase10_observation_window_size,
        baseline_samples=settings.phase10_baseline_samples,
        max_deep_analyses=settings.phase10_max_deep_analyses,
        max_interventions=settings.phase10_max_interventions,
        max_cumulative_intervention_s=settings.phase10_max_cumulative_intervention_s,
        max_optimizer_overhead_s=settings.phase10_max_optimizer_overhead_s,
        approval_timeout_s=settings.phase10_approval_timeout_s,
        max_alerts_per_condition=settings.phase10_max_alerts_per_condition,
    )


def build_runtime_policy(
    settings: Settings,
    *,
    budget: RuntimeBudget | None = None,
    require_approval_for_disruptive: bool | None = None,
    allow_non_disruptive_auto_adaptation: bool | None = None,
) -> RuntimePolicy:
    policy = RuntimePolicy(
        budget=budget if budget is not None else build_runtime_budget(settings),
        persistence_samples=settings.phase10_persistence_samples,
        condition_lookback_samples=settings.phase10_condition_lookback_samples,
        baseline_workload_samples=settings.phase10_baseline_workload_samples,
        baseline_max_wait_s=settings.phase10_baseline_max_wait_s,
        monitor_mode_overhead_fraction_high=settings.phase10_monitor_overhead_fraction_high,
        monitor_mode_overhead_fraction_low=settings.phase10_monitor_overhead_fraction_low,
        monitor_mode_active_window_s=settings.phase10_monitor_active_window_s,
        monitor_mode_dormant_after_s=settings.phase10_monitor_dormant_after_s,
        monitor_mode_low_cost_interval_s=settings.phase10_monitor_low_cost_interval_s,
        monitor_mode_dormant_interval_s=settings.phase10_monitor_dormant_interval_s,
        monitor_mode_expensive_scan_interval_s=settings.phase10_monitor_expensive_scan_interval_s,
        monitor_mode_low_cost_scan_interval_s=settings.phase10_monitor_low_cost_scan_interval_s,
        monitor_mode_dormant_scan_interval_s=settings.phase10_monitor_dormant_scan_interval_s,
        monitor_mode_gpu_scan_interval_s=settings.phase10_monitor_gpu_scan_interval_s,
        opportunity_resuppress_score_delta=settings.phase10_opportunity_resuppress_score_delta,
        opportunity_resuppress_cooldown_s=settings.phase10_opportunity_resuppress_cooldown_s,
        deviation_threshold=settings.phase10_deviation_threshold,
        hysteresis_ratio=settings.phase10_hysteresis_ratio,
        baseline_drift_tolerance=settings.phase10_baseline_drift_tolerance,
        diagnosis_cooldown_s=settings.phase10_diagnosis_cooldown_s,
        intervention_cooldown_s=settings.phase10_intervention_cooldown_s,
        min_evidence_strength=EvidenceStrength(settings.phase10_min_evidence_strength),
        require_approval_for_disruptive=(
            require_approval_for_disruptive
            if require_approval_for_disruptive is not None
            else settings.phase10_require_approval_for_disruptive
        ),
        allow_non_disruptive_auto_adaptation=(
            allow_non_disruptive_auto_adaptation
            if allow_non_disruptive_auto_adaptation is not None
            else settings.phase10_allow_non_disruptive_auto_adaptation
        ),
        max_alerts_per_condition=settings.phase10_max_alerts_per_condition,
        max_observation_processes=settings.phase10_max_observation_processes,
        cost_model=InterventionCostModel(
            benefit_scale_s=settings.phase10_benefit_scale_s,
            min_net_value=settings.phase10_min_net_value,
            deep_analysis_evidence_threshold=settings.phase10_deep_analysis_evidence_threshold,
        ),
        knowledge_lookup_limit=settings.phase9_knowledge_lookup_limit,
        skip_directly_negative=settings.phase9_skip_directly_negative,
        allow_revalidation=settings.phase9_allow_revalidation,
        knowledge_positive_bonus=settings.phase9_knowledge_positive_bonus,
        knowledge_negative_penalty=settings.phase9_knowledge_negative_penalty,
    )
    return policy
