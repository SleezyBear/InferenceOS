"""Card (suggest only): make the Windows page file bigger."""

from __future__ import annotations

from inferenceos.cards.base import ActsOn, ApprovalRule, Card, When
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    OptimizationScope,
    OptimizationSurface,
    ParameterType,
    PrivilegeRequirement,
    Reversibility,
    RiskLevel,
)
from inferenceos.contracts.optimization import CandidateAction, ParameterSpec
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.cost import estimate_cost
from inferenceos.optimization.estimate import unknown_benefit
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.preconditions import check_evidence_support, privilege_precondition
from inferenceos.optimization.strategies import PreparedAction

_MIN_BYTES = 268435456
_MAX_BYTES = 68719476736


def prepare(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    total = ctx.metric("total_ram_bytes")
    available = ctx.metric("available_ram_bytes")
    if total is not None and available is not None:
        shortfall = max(0.0, total - available)
    else:
        shortfall = float(_MIN_BYTES)
    increase = int(min(max(shortfall, float(_MIN_BYTES)), float(_MAX_BYTES)))
    preconditions = [
        check_evidence_support(ctx, action, policy),
        privilege_precondition(PrivilegeRequirement.ELEVATION_REQUIRED, available=False),
    ]
    return PreparedAction(
        parameters={"increase_bytes": increase},
        preconditions=preconditions,
        benefit=unknown_benefit(
            "hard_page_faults_per_sec",
            "decrease",
            confidence=ctx.effective_confidence,
            evidence_basis=list(ctx.evidence_ids),
            notes="pagefile effect depends on the commit workload; magnitude unknown",
        ),
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "A system-wide, elevated change to virtual memory; requires a restart and "
            "human approval, never autonomous."
        ),
        rollback_available=True,
    )


CARD = Card(
    card_id="system.increase_pagefile",
    name="Increase system pagefile size",
    what_it_changes=(
        "Suggests making the Windows page file bigger, which gives memory more room to "
        "spill onto disk. Needs admin rights and a computer restart."
    ),
    acts_on=ActsOn.OPERATING_SYSTEM,
    programs=(),
    when=When.BEFORE_LOADING,
    needs_restart=True,
    applies_when=(
        "commit pressure is the diagnosed mechanism",
        "administrator elevation is available and approved",
    ),
    values_to_try="Enough extra space to cover the memory shortfall",
    should_improve=("hard_page_faults_per_sec",),
    might_worsen=("free disk space",),
    undo="Set the page file size back and restart the computer",
    approval=ApprovalRule.SUGGEST_ONLY,
    admin_rights=True,
    test_cost_estimate_s=3.0,
    answers=(BottleneckKind.PAGING, BottleneckKind.MEMORY_PRESSURE),
    parameters=(
        ParameterSpec(
            name="increase_bytes",
            type=ParameterType.INTEGER,
            description="Additional pagefile capacity in bytes.",
            minimum=_MIN_BYTES,
            maximum=_MAX_BYTES,
            unit="bytes",
        ),
    ),
    surface=OptimizationSurface.SYSTEM_CONFIGURATION,
    subsystem="virtual_memory",
    causal_rationale=(
        "A larger pagefile raises the commit limit and can reduce hard page-fault "
        "activity when commit pressure is the diagnosed mechanism."
    ),
    expected_direction="decrease",
    reversibility=Reversibility.REVERSIBLE_WITH_RESTART,
    risk_level=RiskLevel.HIGH,
    scope=OptimizationScope.SYSTEM_WIDE,
    prepare=prepare,
)
