"""Card: map the model file into memory instead of copying it in."""

from __future__ import annotations

from inferenceos.cards.base import LLAMA_CPP, ActsOn, ApprovalRule, Card, ProgramSetting, When
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    CandidateKind,
    OptimizationSurface,
    ParameterType,
    PreconditionKind,
)
from inferenceos.contracts.optimization import CandidateAction, ParameterSpec
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.cost import estimate_cost
from inferenceos.optimization.estimate import unknown_benefit
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.preconditions import check_evidence_support, make
from inferenceos.optimization.strategies import PreparedAction


def prepare(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    configured = ctx.parameter("use_mmap")
    effective = ctx.effective_mmap
    # Effective runtime evidence (``mmap_effective``) is authoritative when known; the
    # configured value is only a fallback. Unknown never becomes "disabled".
    if effective is not None:
        known = True
        disabled = effective is False
        state_detail = f"effective mmap={effective} (backend evidence)"
    else:
        known = configured is not None
        disabled = configured is False
        state_detail = (
            f"configured use_mmap={configured} (effective state unavailable)"
            if configured is not None
            else "mmap state unavailable (effective and configured both unknown)"
        )
    preconditions = [
        check_evidence_support(ctx, action, policy),
        make(
            "mmap_config_known",
            PreconditionKind.EVIDENCE,
            "current mmap state is known (effective backend evidence or configured)",
            None if not known else True,
            state_detail,
        ),
        make(
            "mmap_currently_disabled",
            PreconditionKind.APPLICABILITY,
            "mmap is currently disabled (change would apply)",
            disabled if known else None,
            "already enabled (no-op)" if (known and not disabled) else "",
        ),
    ]
    return PreparedAction(
        parameters={"use_mmap": True},
        preconditions=preconditions,
        benefit=unknown_benefit(
            "peak_rss_bytes",
            "decrease",
            confidence=ctx.effective_confidence,
            evidence_basis=list(ctx.evidence_ids),
            notes="mmap effect on memory/paging is not responsibly estimable before measurement",
        ),
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "Enabling mmap changes weight-loading behaviour only; it does not alter the "
            "model. Reversible by restoring the prior setting."
        ),
        rollback_available=True,
    )


CARD = Card(
    card_id="runtime.enable_mmap",
    name="Enable memory-mapped weights",
    what_it_changes=(
        "Reads the model file straight from disk into memory as needed, so the computer "
        "can free those pages under memory pressure instead of swapping."
    ),
    acts_on=ActsOn.MODEL_SETTINGS,
    programs=(
        ProgramSetting(
            program=LLAMA_CPP,
            setting="use_mmap",
            how_set="memory mapping is on by default; --no-mmap turns it off",
        ),
    ),
    when=When.BOTH,
    needs_restart=True,
    applies_when=(
        "current mmap configuration is known and is disabled",
        "diagnosis evidence supports memory pressure/paging",
    ),
    values_to_try="On (only tried when it is currently off)",
    should_improve=("hard_page_faults_per_sec", "peak_rss_bytes"),
    might_worsen=("model_load_time_s",),
    undo="Restart with memory mapping set back to its old value",
    approval=ApprovalRule.KEPT_CHANGE,
    admin_rights=False,
    answers=(
        BottleneckKind.PAGING,
        BottleneckKind.MEMORY_PRESSURE,
        BottleneckKind.MODEL_FOOTPRINT,
    ),
    parameters=(
        ParameterSpec(
            name="use_mmap",
            type=ParameterType.BOOLEAN,
            description="Whether mmap-backed weight loading is enabled.",
        ),
    ),
    kind=CandidateKind.MMAP,
    surface=OptimizationSurface.MEMORY_MANAGEMENT,
    subsystem="model_mapping",
    causal_rationale=(
        "File-backed mappings let the OS reclaim clean weight pages, reducing "
        "anonymous commit pressure that drives hard page faults."
    ),
    expected_direction="decrease",
    prepare=prepare,
)
