"""Phase 11 external recommendation policy construction.

All budgets/thresholds/elevation policy come from validated settings; invalid
configuration fails here (pydantic validation) before any observation or action.
"""

from __future__ import annotations

from inferenceos.config import Settings
from inferenceos.contracts.enums import ExternalScope
from inferenceos.contracts.external import (
    ExternalBudget,
    ExternalCostModel,
    ExternalPolicy,
)


def build_external_budget(
    settings: Settings,
    *,
    max_duration_s: float | None = None,
    max_processes: int | None = None,
) -> ExternalBudget:
    return ExternalBudget(
        max_duration_s=(
            max_duration_s if max_duration_s is not None else settings.phase11_max_duration_s
        ),
        max_processes=(
            max_processes if max_processes is not None else settings.phase11_max_processes
        ),
        max_recommendations=settings.phase11_max_recommendations,
        max_actions=settings.phase11_max_actions,
        max_cumulative_action_s=settings.phase11_max_cumulative_action_s,
    )


def build_external_policy(
    settings: Settings,
    *,
    budget: ExternalBudget | None = None,
    require_approval_for_actions: bool | None = None,
    restore_after_action: bool | None = None,
) -> ExternalPolicy:
    return ExternalPolicy(
        budget=budget if budget is not None else build_external_budget(settings),
        observation_interval_s=settings.phase11_observation_interval_s,
        scope=ExternalScope(settings.phase11_scope),
        memory_pressure_threshold=settings.phase11_memory_pressure_threshold,
        memory_dominance_threshold=settings.phase11_memory_dominance_threshold,
        cpu_dominance_threshold=settings.phase11_cpu_dominance_threshold,
        min_relative_significance=settings.phase11_min_relative_significance,
        min_process_rss_bytes=settings.phase11_min_process_rss_bytes,
        min_process_cpu_percent=settings.phase11_min_process_cpu_percent,
        require_target_degradation_for_recommendation=(
            settings.phase11_require_target_degradation
        ),
        require_approval_for_actions=(
            require_approval_for_actions
            if require_approval_for_actions is not None
            else settings.phase11_require_approval_for_actions
        ),
        restore_after_action=(
            restore_after_action
            if restore_after_action is not None
            else settings.phase11_restore_after_action
        ),
        elevation_never_automatic=settings.phase11_elevation_never_automatic,
        allow_unsupported_actions=settings.phase11_allow_unsupported_actions,
        cooldown_s=settings.phase11_cooldown_s,
        approval_timeout_s=settings.phase11_approval_timeout_s,
        max_alerts_per_identity=settings.phase11_max_alerts_per_identity,
        cost_model=ExternalCostModel(
            benefit_scale_s=settings.phase11_benefit_scale_s,
            min_net_value=settings.phase11_min_net_value,
        ),
    )
