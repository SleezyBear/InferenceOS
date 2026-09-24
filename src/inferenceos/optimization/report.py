"""Human-readable rendering of a Phase 6 candidate plan."""

from __future__ import annotations

from inferenceos.contracts.optimization import (
    CandidatePlan,
    OptimizationCandidate,
)


def render_plan(plan: CandidatePlan) -> str:
    lines: list[str] = []
    lines.append("=== Candidate generation plan ===")
    lines.append(f"plan            : {plan.plan_id}")
    lines.append(f"source diagnosis: {plan.source_diagnosis_id}")
    lines.append(f"effective       : {plan.effective_bottleneck.value if plan.effective_bottleneck else None}"
                 f" ({plan.effective_basis}, confidence={plan.effective_confidence:.2f})")
    lines.append(f"outcome         : {plan.outcome.value}")
    lines.append(f"reason          : {plan.terminal_reason}")
    lines.append(f"executed        : {plan.executed} (environment mutated: {plan.environment_mutated})")
    lines.append("")
    lines.append("--- Diagnosis ---")
    lines.append(f"{plan.source_diagnosis.summary}")
    lines.append("")

    if plan.reasoning is not None:
        lines.append("--- Reasoning proposal (registry-enforced) ---")
        r = plan.reasoning
        lines.append(
            f"used={r.used} provider={r.provider} model={r.model} status={r.status} "
            f"validation_ok={r.validation_ok}"
        )
        lines.append(f"proposed={r.proposed_action_ids} accepted={r.accepted_action_ids} "
                     f"rejected={r.rejected_action_ids}")
        for rejection in r.rejections:
            lines.append(f"  - {rejection}")
        if r.failure:
            lines.append(f"failure: {r.failure}")
        lines.append("")

    lines.append("--- Candidates (in generation order) ---")
    if not plan.candidates:
        lines.append("  (none)")
    for candidate in plan.candidates:
        _render_candidate(lines, candidate)
    lines.append("")

    lines.append("--- Ranking (eligible only) ---")
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    for rank, candidate_id in enumerate(plan.ranking, start=1):
        candidate = by_id[candidate_id]
        score = candidate.ranking.composite_score if candidate.ranking else 0.0
        lines.append(f"  {rank}. {candidate.action_id} score={score:.4f} params={candidate.parameters}")
    lines.append("")
    lines.append("--- Selected for Phase 7 consideration (NOT executed) ---")
    for candidate_id in plan.selected_ids:
        lines.append(f"  - {by_id[candidate_id].action_id}")
    lines.append("")

    lines.append("--- Optimizer overhead ---")
    o = plan.overhead
    lines.append(f"generation latency : {o.generation_latency_s:.4f}s")
    lines.append(f"reasoning latency  : {o.reasoning_latency_s}")
    lines.append(
        f"candidates         : {o.candidates_generated} (eligible={o.eligible}, "
        f"gated={o.gated}, rejected={o.rejected}, insufficient={o.insufficient_evidence})"
    )
    lines.append(f"peak optimizer RSS : {o.peak_optimizer_rss_bytes}")

    if plan.limitations:
        lines.append("--- Limitations ---")
        for item in plan.limitations:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def _render_candidate(lines: list[str], candidate: OptimizationCandidate) -> None:
    lines.append(
        f"  [{candidate.eligibility.value}] {candidate.action_id} "
        f"(generated_by={candidate.generated_by}, executable={candidate.executable})"
    )
    lines.append(
        f"      surface={candidate.surface.value} subsystem={candidate.subsystem} "
        f"params={candidate.parameters}"
    )
    lines.append(
        f"      diagnosis={candidate.diagnosis_kind.value} "
        f"confidence={candidate.diagnosis_confidence:.2f} "
        f"strength={candidate.diagnosis_strength.value}"
    )
    lines.append(f"      rationale: {candidate.causal_rationale}")
    lines.append(f"      evidence : {candidate.evidence_refs}")
    if candidate.rejection_reason:
        lines.append(f"      status   : {candidate.rejection_reason}")
    lines.append(
        f"      benefit  : {candidate.benefit.metric} {candidate.benefit.direction} "
        f"known={candidate.benefit.known} range=[{candidate.benefit.magnitude_low}, "
        f"{candidate.benefit.magnitude_high}] confidence={candidate.benefit.confidence:.2f}"
    )
    lines.append(
        f"      cost     : total={candidate.cost.total_cost_s:.2f}s "
        f"(exec={candidate.cost.execution_cost_s:.2f}, reload={candidate.cost.model_reload_cost_s:.2f}, "
        f"experiment={candidate.cost.experiment_cost_s:.2f}, overhead={candidate.cost.optimizer_overhead_s:.2f})"
    )
    lines.append(
        f"      risk     : {candidate.risk.risk_level.value} "
        f"reversibility={candidate.risk.reversibility.value} scope={candidate.risk.scope.value} "
        f"privilege={candidate.risk.privilege.value} approval={candidate.approval.value}"
    )
    for item in candidate.preconditions:
        mark = {True: "yes", False: "no", None: "unknown"}[item.satisfied]
        lines.append(f"      precondition[{mark}] {item.description}: {item.detail}")
    if candidate.ranking is not None:
        for step in candidate.ranking.explanation:
            lines.append(f"      rank: {step}")
