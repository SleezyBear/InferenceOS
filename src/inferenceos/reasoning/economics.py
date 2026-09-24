"""Phase 5R entry economics and reasoner feasibility (pure, testable).

Units are explicitly *seconds-equivalent utility* — never money. The model is
deliberately simple and every input is recorded so a human can audit the decision.
"""

from __future__ import annotations

from inferenceos.config import Settings
from inferenceos.contracts.lifecycle import (
    EntryEconomics,
    FeasibilityAssessment,
    LifecyclePolicy,
    ResourceSnapshot,
)
from inferenceos.contracts.reasoning import EscalationDecision


def build_lifecycle_policy(settings: Settings) -> LifecyclePolicy:
    return LifecyclePolicy(
        benefit_scale_s=settings.phase5r_benefit_scale_s,
        benefit_uncertainty_weight=settings.phase5r_benefit_uncertainty_weight,
        benefit_competing_weight=settings.phase5r_benefit_competing_weight,
        cost_weight=settings.phase5r_cost_weight,
        min_net_utility=settings.phase5r_min_net_utility,
        diagnosis_estimate_s=settings.phase5r_diagnosis_estimate_s,
        reasoner_load_estimate_s=settings.phase5r_reasoner_load_estimate_s,
        reasoner_unload_estimate_s=settings.phase5r_reasoner_unload_estimate_s,
        feasibility_headroom_fraction=settings.phase5r_feasibility_headroom_fraction,
        reasoner_reclaim_fraction=settings.phase5r_reasoner_reclaim_fraction,
        corroboration_ceiling=settings.phase5r_corroboration_ceiling,
        transition_log_limit=settings.phase5r_transition_log_limit,
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def evaluate_entry_economics(
    escalation: EscalationDecision,
    *,
    policy: LifecyclePolicy,
    target_unload_s: float,
    reasoner_load_s: float,
    reasoner_unload_s: float,
    target_reload_s: float,
    diagnosis_s: float | None = None,
    interruption_cost_s: float = 0.0,
    optimizer_overhead_s: float = 0.0,
) -> EntryEconomics:
    """Decide whether entering deep diagnosis is worth its measured/estimated cost."""
    diagnosis = policy.diagnosis_estimate_s if diagnosis_s is None else diagnosis_s
    uncertainty = _clamp(1.0 - escalation.deterministic_confidence)
    competing = max(0, escalation.competing_families - 1)
    severity_factor = 1.0 + policy.benefit_competing_weight * competing

    benefit_utility = (
        policy.benefit_uncertainty_weight * uncertainty * severity_factor
    ) * policy.benefit_scale_s
    if not escalation.should_escalate:
        benefit_utility = 0.0

    costs = {
        "target_unload_s": round(target_unload_s, 4),
        "reasoner_load_s": round(reasoner_load_s, 4),
        "diagnosis_s": round(diagnosis, 4),
        "reasoner_unload_s": round(reasoner_unload_s, 4),
        "target_reload_s": round(target_reload_s, 4),
        "interruption_cost_s": round(interruption_cost_s, 4),
        "optimizer_overhead_s": round(optimizer_overhead_s, 4),
    }
    total_cost_s = sum(costs.values())
    cost_utility = policy.cost_weight * total_cost_s
    net_utility = benefit_utility - cost_utility
    enter = bool(escalation.should_escalate and net_utility > policy.min_net_utility)

    rationale = [
        f"uncertainty={uncertainty:.3f} (1 - deterministic confidence "
        f"{escalation.deterministic_confidence:.3f})",
        f"severity_factor={severity_factor:.3f} from {escalation.competing_families} "
        "competing resource families",
        f"benefit_utility={benefit_utility:.3f}s-equivalent",
        f"total lifecycle/interruption cost={total_cost_s:.3f}s",
        f"net_utility={net_utility:.3f} (threshold {policy.min_net_utility:.3f})",
    ]
    if not escalation.should_escalate:
        rationale.append("escalation gate did not fire; deep diagnosis not entered")
    elif not enter:
        rationale.append(
            "estimated cost exceeds estimated benefit; refusing deep diagnosis while "
            "leaving the target resident and running"
        )
    else:
        rationale.append("estimated benefit exceeds cost; deep diagnosis justified")

    return EntryEconomics(
        enter=enter,
        benefit_utility=round(benefit_utility, 4),
        cost_utility=round(cost_utility, 4),
        net_utility=round(net_utility, 4),
        deterministic_confidence=escalation.deterministic_confidence,
        uncertainty=uncertainty,
        competing_families=escalation.competing_families,
        benefits={
            "uncertainty": round(uncertainty, 4),
            "severity_factor": round(severity_factor, 4),
            "benefit_scale_s": policy.benefit_scale_s,
        },
        costs=costs,
        assumptions=[
            "units are seconds-equivalent utility, not currency",
            "benefit = uncertainty * severity_factor * benefit_scale_s",
            "cost = sum of measured/estimated lifecycle seconds + interruption cost",
            "reasoner load/unload before a first measurement use configured estimates",
        ],
        rationale=rationale,
    )


def assess_feasibility(
    *,
    snapshot: ResourceSnapshot,
    target_resident_bytes: int | None,
    reasoner_required_bytes: int | None,
    policy: LifecyclePolicy,
    reasoner_footprint_basis: str,
) -> FeasibilityAssessment:
    """Check that the reasoner can actually fit after the target is unloaded."""
    available = snapshot.system_available_bytes
    total = snapshot.system_total_bytes
    reasons: list[str] = []
    if available is None:
        return FeasibilityAssessment(
            feasible=False,
            available_bytes=None,
            target_resident_bytes=target_resident_bytes,
            reasoner_required_bytes=reasoner_required_bytes,
            reasoner_footprint_basis=reasoner_footprint_basis,
            reasons=["system available memory could not be measured"],
        )
    if reasoner_required_bytes is None or reasoner_required_bytes <= 0:
        return FeasibilityAssessment(
            feasible=False,
            available_bytes=available,
            target_resident_bytes=target_resident_bytes,
            reasoner_required_bytes=reasoner_required_bytes,
            reasoner_footprint_basis=reasoner_footprint_basis,
            reasons=["reasoner memory requirement could not be established"],
        )
    reclaim = int((target_resident_bytes or 0) * policy.reasoner_reclaim_fraction)
    projected_free = available + reclaim
    basis_total = total if total is not None else projected_free
    headroom = int(basis_total * policy.feasibility_headroom_fraction)
    feasible = projected_free - reasoner_required_bytes >= headroom
    reasons.append(
        f"available={available} B + reclaimed target={reclaim} B "
        f"(reclaim fraction {policy.reasoner_reclaim_fraction:.2f}) = {projected_free} B"
    )
    reasons.append(
        f"reasoner requires {reasoner_required_bytes} B; headroom "
        f"{policy.feasibility_headroom_fraction:.2f} = {headroom} B"
    )
    if feasible:
        reasons.append("reasoner can be loaded after the target is unloaded")
    else:
        reasons.append(
            "reasoner cannot be loaded safely; target must remain resident and running"
        )
    if snapshot.gpu_memory_bytes is None:
        reasons.append(
            "GPU/VRAM feasibility is not measured on this machine; the gate uses "
            "system physical memory only"
        )
    return FeasibilityAssessment(
        feasible=feasible,
        available_bytes=available,
        target_resident_bytes=target_resident_bytes,
        reasoner_required_bytes=reasoner_required_bytes,
        projected_free_bytes=projected_free,
        headroom_bytes=headroom,
        reasoner_footprint_basis=reasoner_footprint_basis,
        reasons=reasons,
    )
