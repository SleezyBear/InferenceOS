"""Candidate cost estimation, including the optimizer's own overhead.

Optimization is not free: selection must account for intervention, reconfiguration,
model reload, the estimated Phase 7 experiment cost, the reasoning cost when Phase 5
was invoked, and the optimizer's own overhead.
"""

from __future__ import annotations

from inferenceos.contracts.optimization import CandidateAction, CostEstimate
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.policy import OptimizationPolicy

RECONFIGURATION_COST_S = 0.5


def estimate_cost(
    ctx: OptimizationContext,
    action: CandidateAction,
    policy: OptimizationPolicy,
) -> CostEstimate:
    notes: list[str] = []
    execution = action.execution_cost_estimate
    reconfiguration = RECONFIGURATION_COST_S

    reload_s = ctx.metric("model_load_time_s")
    if reload_s is None:
        reload_s = 0.0
        notes.append("model reload time unavailable; assumed 0 (Phase 7 must measure)")
    else:
        notes.append(f"model reload estimated from measured load time ({reload_s:.2f}s)")

    latency = ctx.metric("total_latency_s")
    if latency is None:
        experiment = 0.0
        notes.append("experiment cost unavailable; estimated 0")
    else:
        experiment = latency * policy.experiment_repetitions + reload_s * policy.experiment_repetitions
        notes.append(
            f"experiment estimated at {policy.experiment_repetitions} runs "
            f"(Phase 3 harness cost estimate)"
        )

    reasoning = ctx.reasoning_cost_s if ctx.reasoning_performed and ctx.reasoning_cost_s else 0.0
    if reasoning:
        notes.append("apportioned Phase 5 reasoning latency")

    overhead = action.optimizer_overhead_estimate
    total = execution + reconfiguration + reload_s + experiment + reasoning + overhead
    return CostEstimate(
        execution_cost_s=execution,
        reconfiguration_cost_s=reconfiguration,
        model_reload_cost_s=reload_s,
        experiment_cost_s=experiment,
        reasoning_cost_s=reasoning,
        optimizer_overhead_s=overhead,
        total_cost_s=total,
        notes=notes,
    )
