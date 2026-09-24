"""Human-readable rendering of a Phase 11 external recommendation run."""

from __future__ import annotations

from inferenceos.contracts.external import ExternalRecommendationRun


def render_external_run(run: ExternalRecommendationRun) -> str:
    lines: list[str] = []
    lines.append("=== InferenceOS Phase 11 external recommendation run ===")
    lines.append(f"run          : {run.run_id}")
    lines.append(f"scope        : {run.scope.value}")
    lines.append(f"target pid   : {run.target_pid}")
    lines.append(
        f"observations : {len(run.observation_reports)} "
        f"assessments {len(run.assessments)} recommendations {len(run.recommendations)}"
    )
    lines.append(
        f"safety       : {len(run.safety_checks)} approvals {len(run.approvals)} "
        f"actions {len(run.action_results)} verifications {len(run.verifications)}"
    )
    lines.append(f"knowledge    : queried {len(run.knowledge_queries)} written {len(run.knowledge_writes)}")
    lines.append(f"overhead     : {run.overhead.total_s:.3f}s")
    lines.append(f"summary      : {run.summary}")
    lines.append("")

    if run.assessments:
        lines.append("--- Attribution assessments ---")
        for assessment in run.assessments:
            lines.append(
                f"  {assessment.identity.name} (pid {assessment.identity.pid}) "
                f"resource={assessment.resource.value} dominant={assessment.impacts[0].dominant if assessment.impacts else None} "
                f"causal={assessment.causal} sufficient={assessment.sufficient_for_recommendation} "
                f"strength={assessment.strength.value}"
            )
            lines.append(f"      identity={assessment.identity.identity_key}")
            for item in assessment.impacts:
                lines.append(
                    f"      {item.resource.value}: process={item.process_contribution} "
                    f"target={item.target_contribution} relative={item.relative_significance} "
                    f"dominant={item.dominant}"
                )
        lines.append("")

    if run.recommendations:
        lines.append("--- Recommendations ---")
        for recommendation in run.recommendations:
            lines.append(
                f"  {recommendation.recommendation_id}: class={recommendation.classification.value} "
                f"resource={recommendation.resource.value} action={recommendation.proposed_action_id} "
                f"supported={recommendation.supported} approval={recommendation.requires_approval}"
            )
            lines.append(f"      {recommendation.neutral_description}")
            lines.append(f"      benefit: {recommendation.expected_benefit}")
        lines.append("")

    if run.safety_checks:
        lines.append("--- Safety checks ---")
        for safety in run.safety_checks:
            lines.append(
                f"  {safety.action_id}: allowed={safety.allowed} requires_approval={safety.requires_approval} "
                f"elevation_required={safety.elevation_required} implemented={safety.implemented}"
            )
            for reason in safety.reasons:
                lines.append(f"      reason: {reason}")
        lines.append("")

    if run.approvals:
        lines.append("--- Approvals ---")
        for approval in run.approvals:
            lines.append(
                f"  {approval.recommendation_id}: {approval.outcome.value} by {approval.decided_by} "
                f"({approval.detail})"
            )
        lines.append("")

    if run.action_results:
        lines.append("--- External actions ---")
        for result in run.action_results:
            lines.append(
                f"  {result.action_result_id}: action={result.action_id} executed={result.executed} "
                f"reason={result.reason}"
            )
        lines.append("")

    if run.verifications:
        lines.append("--- Verifications ---")
        for verification in run.verifications:
            lines.append(
                f"  {verification.verification_id}: verified={verification.verified} "
                f"changed={verification.external_state_changed} "
                f"restored={verification.restoration_verified} "
                f"failure={verification.failure_reason}"
            )
        lines.append("")

    lines.append(f"termination  : {run.termination.reason.value} — {run.termination.detail}")
    if run.errors:
        lines.append("--- Errors ---")
        for error in run.errors:
            lines.append(f"  - {error}")
    return "\n".join(lines)
