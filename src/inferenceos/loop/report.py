"""Human-readable rendering of a Phase 9 autonomous run."""

from __future__ import annotations

from inferenceos.contracts.enums import KnowledgeSelectionEffect
from inferenceos.contracts.optimization_loop import LoopResult, OptimizationRun


def render_termination(run: OptimizationRun) -> str:
    termination = run.termination
    lines = [
        "=== Loop termination ===",
        f"reason    : {termination.reason.value}",
        f"clean     : {termination.clean} (safety={termination.safety})",
        f"iteration : {termination.at_iteration}",
        f"detail    : {termination.detail}",
    ]
    return "\n".join(lines)


def render_run(result: LoopResult) -> str:
    run = result.run
    lines: list[str] = []
    lines.append("=== InferenceOS Phase 9 autonomous run ===")
    lines.append(f"run          : {run.run_id}")
    lines.append(
        f"changed      : {run.changed} (retained {len(run.retained_candidate_ids)})"
    )
    lines.append(f"iterations   : {len(run.iterations)}")
    lines.append(
        f"budget       : iterations {run.budget.iterations_used}/{run.budget.max_iterations}, "
        f"experiments {run.budget.experiments_used}/{run.budget.max_experiments}, "
        f"deep {run.budget.deep_diagnoses_used}/{run.budget.max_deep_diagnoses}"
    )
    lines.append(
        f"wall clock   : {run.budget.wall_clock_used_s:.2f}s "
        f"(max {run.budget.max_wall_clock_s:.0f}s)"
    )
    lines.append(
        f"optimizer    : {run.overhead.optimizer_overhead_s:.3f}s "
        f"(experiment {run.overhead.experiment_s:.3f}s)"
    )
    lines.append(f"knowledge    : {run.knowledge_health.value} (available={run.knowledge_available})")
    lines.append(f"reasoning    : available={run.reasoning_available}")
    lines.append(f"initial cfg  : {run.initial_configuration.as_dict()}")
    lines.append(f"final cfg    : {run.final_configuration.as_dict()}")
    lines.append(f"summary      : {run.summary}")
    lines.append("")
    lines.append("--- Iterations ---")
    for iteration in run.iterations:
        lines.append(
            f"[{iteration.iteration}] outcome={iteration.outcome.value} "
            f"diagnosis={iteration.diagnosis_kind} "
            f"plan={iteration.plan_outcome.value if iteration.plan_outcome else None} "
            f"candidate={iteration.selected_action_id} "
            f"experiment={iteration.experiment_id} "
            f"status={iteration.experiment_status.value if iteration.experiment_status else None} "
            f"retained={iteration.retained} "
            f"restoration={iteration.restoration_status.value}"
        )
        if iteration.checkpoint is not None:
            lines.append(
                f"    checkpoint={iteration.checkpoint.checkpoint_id} "
                f"config={iteration.checkpoint.runtime_configuration.as_dict()}"
            )
        for evidence in iteration.knowledge:
            if evidence.effect is KnowledgeSelectionEffect.NEUTRAL and evidence.records_considered == 0:
                continue
            lines.append(
                f"    knowledge[{evidence.action_id}]: effect={evidence.effect.value} "
                f"records={evidence.records_considered} {evidence.detail}"
            )
        for note in iteration.notes:
            lines.append(f"    note: {note}")
        for error in iteration.errors:
            lines.append(f"    error: {error}")
    lines.append("")
    lines.append(render_termination(run))
    if run.errors:
        lines.append("")
        lines.append("--- Run errors ---")
        for error in run.errors:
            lines.append(f"  - {error}")
    return "\n".join(lines)
