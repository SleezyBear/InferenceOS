"""Human-readable rendering of a Phase 11R external recommendation run."""

from __future__ import annotations

from inferenceos.contracts.external import (
    ExternalRecommendation,
    ExternalRecommendationRun,
)


def render_recommendation(recommendation: ExternalRecommendation) -> str:
    """Render one recommendation exactly as a human decision aid.

    InferenceOS recommends; the user decides. The output never instructs the system to
    execute the action.
    """

    identity = recommendation.identity
    metrics = ", ".join(
        f"{key}={value:.4g}" for key, value in recommendation.observed_metrics.items()
    ) or "unavailable"
    lines = [
        "External process recommendation",
        "",
        f"  bottleneck        : {recommendation.bottleneck or 'unknown'}",
        f"  resource          : {recommendation.resource.value}",
        f"  inferred action   : {recommendation.proposed_action_id or 'none (observation only)'}",
        "",
        "Observed process:",
        f"  name              : {identity.name if identity else 'unknown'}",
        f"  pid               : {identity.pid if identity else 'unknown'}",
        f"  identity          : {identity.identity_key if identity else 'unknown'}",
        f"  observed metrics  : {metrics}",
        f"  observed at       : {recommendation.observed_at.isoformat() if recommendation.observed_at else 'unknown'}",
        "",
        "What was observed:",
        f"  {recommendation.neutral_description}",
        "",
        "Why it may matter:",
        f"  {recommendation.why_it_matters}",
        "",
        "Suggested human action:",
        f"  {recommendation.proposed_action_id or 'No action proposed.'}",
        "",
        "Expected effect:",
        f"  {recommendation.expected_benefit}",
        f"  grounded estimate : {recommendation.expected_benefit_available}",
        "",
        "Uncertainty:",
        f"  {recommendation.uncertainty}",
        "",
        "Risk:",
        f"  {recommendation.disruption_risk}",
        "",
        "Provenance:",
        f"  run               : {recommendation.run_id}",
        f"  session           : {recommendation.session_id}",
        f"  diagnosis         : {recommendation.provenance.diagnosis_id if recommendation.provenance else None}",
        f"  status            : {recommendation.status.value}",
        "",
        "Action:",
        "  User decision required. InferenceOS will not execute this action.",
    ]
    return "\n".join(lines)


def render_external_run(run: ExternalRecommendationRun) -> str:
    lines: list[str] = []
    lines.append("=== InferenceOS Phase 11R external recommendation run ===")
    lines.append(f"run          : {run.run_id}")
    lines.append(f"session      : {run.session_id}")
    lines.append(f"scope        : {run.scope.value}")
    lines.append(f"target pid   : {run.target_pid}")
    lines.append(f"bottleneck   : {run.diagnosed_bottleneck or 'unknown'}")
    lines.append(
        f"observations : {len(run.observation_reports)} "
        f"assessments {len(run.assessments)} recommendations {len(run.recommendations)}"
    )
    lines.append(
        f"decisions    : {len(run.user_decisions)} approvals {len(run.approvals)} "
        f"external mutations 0"
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
                f"status={recommendation.status.value} user_approval={recommendation.requires_user_approval}"
            )
            lines.append(f"      {recommendation.neutral_description}")
            lines.append(f"      benefit: {recommendation.expected_benefit}")
            lines.append(f"      uncertainty: {recommendation.uncertainty}")
        lines.append("")

    if run.safety_checks:
        lines.append("--- Safety checks (all external actions are non-executable) ---")
        for safety in run.safety_checks:
            lines.append(
                f"  {safety.action_id}: allowed={safety.allowed} requires_approval={safety.requires_approval} "
                f"elevation_required={safety.elevation_required} implemented={safety.implemented}"
            )
            for reason in safety.reasons:
                lines.append(f"      reason: {reason}")
        lines.append("")

    if run.user_decisions:
        lines.append("--- User decisions (no autonomous execution) ---")
        for decision in run.user_decisions:
            lines.append(
                f"  {decision.recommendation_id}: {decision.decision.value} by "
                f"{decision.decided_by} autonomous_execution={decision.autonomous_execution}"
            )
        lines.append("")

    lines.append(f"termination  : {run.termination.reason.value} — {run.termination.detail}")
    if run.errors:
        lines.append("--- Errors ---")
        for error in run.errors:
            lines.append(f"  - {error}")
    return "\n".join(lines)
