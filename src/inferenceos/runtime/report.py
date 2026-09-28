"""Human-readable rendering of a Phase 10 runtime monitoring run."""

from __future__ import annotations

from inferenceos.contracts.runtime import RuntimeMonitorResult, RuntimeMonitorRun


def render_state(run: RuntimeMonitorRun) -> str:
    lines = [
        "=== Runtime state transitions ===",
    ]
    for event in run.states:
        lines.append(
            f"[obs {event.observation:>4}] {event.state.value:<24} {event.detail}"
        )
    return "\n".join(lines)


def render_monitor_run(result: RuntimeMonitorResult) -> str:
    run = result.run
    lines: list[str] = []
    lines.append("=== InferenceOS Phase 10 runtime monitor ===")
    lines.append(f"run           : {run.run_id}")
    lines.append(f"workload      : {run.workload.workload_id} ({run.workload.model_ref})")
    lines.append(f"final state   : {run.final_state.value}")
    lines.append(
        f"observations  : {run.observations_collected} "
        f"(diagnoses {run.diagnosis_events}, deep {run.deep_analysis_events})"
    )
    lines.append(
        f"baselines     : {len(run.baselines)} "
        f"(current {run.baseline.baseline_id if run.baseline else None})"
    )
    lines.append(
        f"opportunities : {len(run.opportunities)} "
        f"recommendations {len(run.recommendations)} approvals {len(run.approvals)}"
    )
    lines.append(
        f"checkpoints   : {len(run.checkpoints)} interventions {len(run.interventions)}"
    )
    lines.append(f"cooldowns     : {len(run.cooldowns)}")
    if run.optimization_preference is not None:
        preference = run.optimization_preference
        state = run.performance_state.state.value if run.performance_state else "unknown"
        lines.append(
            f"optimization  : mode={preference.mode.value} "
            f"min_improvement={preference.minimum_improvement:.2%} "
            f"max_search={preference.maximum_search_time_s:.0f}s "
            f"performance_state={state}"
        )
    if run.resume_verifications:
        for verification in run.resume_verifications:
            lines.append(
                f"resume        : {verification.phase} {verification.outcome.value} "
                f"search={verification.search_duration_s:.1f}s "
                f"timeout={verification.timeout_s:.1f}s attempts={verification.attempts}"
            )
    lines.append(
        f"overhead      : monitoring {run.overhead.optimizer_overhead_s:.3f}s / "
        f"intervention {run.overhead.experiment_s + run.overhead.pause_s + run.overhead.resume_s:.3f}s"
    )
    lines.append(
        f"resource      : decode_tps mean={run.resource_impact.decode_tps_mean} "
        f"cv={run.resource_impact.decode_tps_cv} "
        f"within_tolerance={run.resource_impact.throughput_within_tolerance}"
    )
    lines.append(f"summary       : {run.summary}")
    lines.append("")

    if run.opportunities:
        lines.append("--- Opportunities ---")
        for opportunity in run.opportunities:
            lines.append(
                f"  {opportunity.opportunity_id}: {opportunity.kind} "
                f"anomaly={opportunity.anomaly.value} persistence={opportunity.persistence_samples} "
                f"meaningful={opportunity.meaningful} rule={opportunity.diagnosis_kind}"
            )
        lines.append("")

    if run.recommendations:
        lines.append("--- Recommendations ---")
        for rec in run.recommendations:
            lines.append(
                f"  {rec.proposal_id}: action={rec.proposed_action_id} "
                f"cost={rec.expected_cost_s:.1f}s approval={rec.requires_approval} "
                f"class={rec.intervention_class.value}"
            )
        lines.append("")

    if run.approvals:
        lines.append("--- Approvals ---")
        for approval in run.approvals:
            lines.append(
                f"  {approval.proposal_id}: {approval.outcome.value} "
                f"(by {approval.decided_by}) {approval.detail}"
            )
        lines.append("")

    if run.target_recoveries:
        lines.append("--- Resident target recoveries (B2) ---")
        for recovery in run.target_recoveries:
            lines.append(
                f"  {recovery.recovery_id}: reason={recovery.reason} "
                f"attempted={recovery.attempted} attempt_count={recovery.attempt_count} "
                f"recovered={recovery.recovered} unrecoverable={recovery.unrecoverable} "
                f"pid {recovery.old_pid} -> {recovery.new_pid}"
            )
            lines.append(
                f"      identity={recovery.identity_verified} "
                f"readiness={recovery.readiness_verified} "
                f"configuration={recovery.configuration_verified} "
                f"model={recovery.model_verified} backend={recovery.backend_verified} "
                f"cleanup={recovery.cleanup_complete} orphan_free={recovery.orphan_free}"
            )
            if recovery.failure_reason:
                lines.append(f"      failure: {recovery.failure_reason}")
        lines.append("")

    if run.interventions:
        lines.append("--- Interventions ---")
        for intervention in run.interventions:
            lines.append(
                f"  {intervention.intervention_id}: candidate={intervention.candidate_id} "
                f"action={intervention.action_id} decision={intervention.decision.value if intervention.decision else None} "
                f"retained={intervention.retained} quality={intervention.quality_status.value if intervention.quality_status else None} "
                f"resume_verified={intervention.resume_verified}"
            )
            for note in intervention.notes:
                lines.append(f"      note: {note}")
            for error in intervention.errors:
                lines.append(f"      error: {error}")
        lines.append("")

    lines.append(render_state(run))
    if run.errors:
        lines.append("")
        lines.append("--- Errors ---")
        for error in run.errors:
            lines.append(f"  - {error}")
    return "\n".join(lines)
