"""Card (suggest only): pause another program that is hogging memory."""

from __future__ import annotations

from inferenceos.cards.base import ActsOn, ApprovalRule, Card, When
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    OptimizationScope,
    OptimizationSurface,
    ParameterType,
    PreconditionKind,
    RiskLevel,
)
from inferenceos.contracts.optimization import CandidateAction, ParameterSpec
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.cost import estimate_cost
from inferenceos.optimization.estimate import external_process_benefit
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.preconditions import check_evidence_support, make
from inferenceos.optimization.strategies import PreparedAction


def _dominant_external_process(ctx: OptimizationContext) -> tuple[str | None, float | None]:
    if ctx.group is None:
        return None, None
    model_rss = ctx.metric("peak_rss_bytes")
    best_name: str | None = None
    best_rss: float | None = None
    for run in ctx.group.runs:
        state = run.observation.system_state
        if state is None or not state.processes:
            continue
        dominant = max(state.processes, key=lambda process: process.rss_bytes)
        if model_rss is not None and float(dominant.rss_bytes) <= model_rss:
            continue
        if best_rss is None or float(dominant.rss_bytes) > best_rss:
            best_name = dominant.name
            best_rss = float(dominant.rss_bytes)
    return best_name, best_rss


def prepare(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    name, rss = _dominant_external_process(ctx)
    allowed_choices = next(
        (list(spec.choices) for spec in action.parameters if spec.name == "process_name"),
        [],
    )
    named = name if name in allowed_choices else None
    preconditions = [
        check_evidence_support(ctx, action, policy),
        make(
            "external_process_identified",
            PreconditionKind.EVIDENCE,
            "an external memory-dominant process is identified and supported",
            None if name is None else (name in allowed_choices),
            f"dominant external process={name!r}" if name else "none observed",
        ),
        make(
            "human_approval",
            PreconditionKind.SAFETY,
            "human approval is required before any external process is touched",
            None,
            "phase 6 never approves or applies external actions",
        ),
    ]
    params: dict[str, object] = {"process_name": named} if named is not None else {}
    return PreparedAction(
        parameters=params,
        preconditions=preconditions,
        benefit=external_process_benefit(
            process_rss_bytes=rss,
            confidence=ctx.effective_confidence,
            evidence_basis=list(ctx.evidence_ids),
        ),
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "Suspending an unrelated process affects the user's other work and must be "
            "human-approved and reversible."
        ),
        rollback_available=True,
    )


CARD = Card(
    card_id="external.suspend_process",
    name="Suspend an external memory-dominant process",
    what_it_changes=(
        "Suggests pausing another program that is holding more memory than the model, "
        "without closing it. Never done by InferenceOS itself."
    ),
    acts_on=ActsOn.OTHER_PROGRAMS,
    programs=(),
    when=When.WHILE_RUNNING,
    needs_restart=False,
    applies_when=(
        "an external process is identified as memory-dominant",
        "human approval is granted",
    ),
    values_to_try="The single program holding the most memory",
    should_improve=("available_ram_bytes",),
    might_worsen=("the other program's work",),
    undo="The user resumes the paused program",
    approval=ApprovalRule.SUGGEST_ONLY,
    admin_rights=False,
    answers=(BottleneckKind.EXTERNAL_PROCESS, BottleneckKind.MEMORY_PRESSURE),
    parameters=(
        ParameterSpec(
            name="process_name",
            type=ParameterType.STRING,
            description="Name of the external process to suspend (explicit list).",
            choices=["memory_holder.exe", "python.exe"],
        ),
    ),
    surface=OptimizationSurface.EXTERNAL_PROCESS,
    subsystem="memory_contention",
    causal_rationale=(
        "If an external process holds more resident memory than the model under "
        "pressure, pausing it can free memory for the target workload."
    ),
    expected_direction="increase",
    risk_level=RiskLevel.MEDIUM,
    scope=OptimizationScope.EXTERNAL_PROCESS,
    external_recommendation=True,
    prepare=prepare,
)
