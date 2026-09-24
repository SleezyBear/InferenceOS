"""Phase 9 loop policy construction.

All budget ceilings and safety/decision gates come from validated configuration —
there are no magic numbers in the orchestrator. Invalid budgets fail here (pydantic
validation) before any mutation can occur.
"""

from __future__ import annotations

from inferenceos.config import Settings
from inferenceos.contracts.optimization_loop import OptimizationBudget, LoopPolicy


def build_budget(settings: Settings) -> OptimizationBudget:
    return OptimizationBudget(
        max_iterations=settings.phase9_max_iterations,
        max_experiments=settings.phase9_max_experiments,
        max_wall_clock_s=settings.phase9_max_wall_clock_s,
        max_optimizer_overhead_s=settings.phase9_max_optimizer_overhead_s,
        max_deep_diagnoses=settings.phase9_max_deep_diagnoses,
    )


def build_loop_policy(settings: Settings) -> LoopPolicy:
    return LoopPolicy(
        budget=build_budget(settings),
        allow_deep_diagnosis=settings.phase9_allow_deep_diagnosis,
        require_reasoning_when_escalated=settings.phase9_require_reasoning_when_escalated,
        require_knowledge=settings.phase9_require_knowledge,
        knowledge_lookup_limit=settings.phase9_knowledge_lookup_limit,
        skip_directly_negative=settings.phase9_skip_directly_negative,
        allow_revalidation=settings.phase9_allow_revalidation,
        max_candidates_per_iteration=settings.phase9_max_candidates_per_iteration,
        autonomous_only=settings.phase9_autonomous_only,
        stop_on_quality_regression=settings.phase9_stop_on_quality_regression,
        stop_on_restoration_failure=settings.phase9_stop_on_restoration_failure,
        reobserve_after_retention=settings.phase9_reobserve_after_retention,
        knowledge_positive_bonus=settings.phase9_knowledge_positive_bonus,
        knowledge_negative_penalty=settings.phase9_knowledge_negative_penalty,
        min_experiment_confidence=settings.phase9_min_experiment_confidence,
    )
