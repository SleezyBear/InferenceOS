"""Card: bring the processor thread count down to the number of real cores."""

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
from inferenceos.optimization.strategies import ALREADY_AT_DESIRED_STATE, PreparedAction


def prepare(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    """Bind the oversubscription fix to the *observed* runtime threads.

    The authoritative current value is the observed ``effective_threads`` (never the
    requested value, and never inferred from CPU topology). The only target is the
    physical-core count; there is no search. Status semantics:

    * effective unknown          -> not eligible (``thread_count_known`` unestablished);
    * effective <= physical       -> valid NO_CHANGE (``ALREADY_AT_DESIRED_STATE``);
    * effective >  physical       -> prepare ``target_threads = physical_cores``.
    """
    physical = ctx.hardware.cpu.physical_cores
    effective = ctx.effective_threads
    oversubscribed = effective is not None and effective > physical

    if effective is None:
        status_detail = "effective runtime thread count unavailable; not inferred from CPU topology"
    else:
        status_detail = f"effective={effective} physical={physical}"

    preconditions = [
        check_evidence_support(ctx, action, policy),
        make(
            "thread_count_known",
            PreconditionKind.EVIDENCE,
            "effective runtime thread count is known",
            None if effective is None else True,
            status_detail,
        ),
    ]

    no_change_reason: str | None = None
    if effective is None:
        parameters: dict[str, object] = {}
    elif not oversubscribed:
        # Already at (or below) the desired state: a valid no-op, not a failure.
        preconditions.append(
            make(
                "threads_exceed_physical_cores",
                PreconditionKind.APPLICABILITY,
                "effective thread count exceeds physical core count",
                False,
                f"{ALREADY_AT_DESIRED_STATE} effective={effective} physical={physical}",
            )
        )
        parameters = {}
        no_change_reason = ALREADY_AT_DESIRED_STATE
    else:
        preconditions.append(
            make(
                "threads_exceed_physical_cores",
                PreconditionKind.APPLICABILITY,
                "effective thread count exceeds physical core count",
                True,
                f"effective={effective} physical={physical}",
            )
        )
        parameters = {"threads": physical}

    return PreparedAction(
        parameters=parameters,
        preconditions=preconditions,
        benefit=unknown_benefit(
            "decode_tokens_per_sec",
            "increase",
            confidence=ctx.effective_confidence,
            evidence_basis=list(ctx.evidence_ids),
            notes=(
                "oversubscription remediation: the causal objective is throughput and "
                "CPU-seconds/token (with context-switch reduction as supporting evidence); "
                "raw process_cpu_percent is secondary and magnitude is workload-dependent"
            ),
        ),
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "Reducing threads can lower throughput; reversible by restoring the prior count."
        ),
        rollback_available=True,
        no_change_reason=no_change_reason,
    )


CARD = Card(
    card_id="runtime.reduce_threads",
    name="Right-size CPU thread count",
    what_it_changes=(
        "Lowers how many processor threads the model uses to the number of real cores, "
        "so the threads stop fighting each other."
    ),
    acts_on=ActsOn.MODEL_SETTINGS,
    programs=(
        ProgramSetting(program=LLAMA_CPP, setting="threads", how_set="--threads N"),
    ),
    when=When.BOTH,
    needs_restart=True,
    applies_when=(
        "current thread count is known",
        "target thread count is below the current value",
        "diagnosis evidence supports a CPU execution limit",
    ),
    values_to_try="The number of physical processor cores (only when more threads are in use)",
    should_improve=(
        "decode_tokens_per_sec",
        "cpu_seconds_per_token",
        "context_switches_per_token",
    ),
    might_worsen=("decode_tokens_per_sec",),
    undo="Restart with the old thread count",
    approval=ApprovalRule.KEPT_CHANGE,
    admin_rights=False,
    works_with=("runtime.increase_gpu_offload_layers", "runtime.reduce_gpu_offload_layers"),
    answers=(BottleneckKind.CPU,),
    parameters=(
        ParameterSpec(
            name="threads",
            type=ParameterType.INTEGER,
            description="Thread count at or below the current value.",
            minimum=1,
            maximum=4096,
            unit="threads",
        ),
    ),
    kind=CandidateKind.THREADS,
    surface=OptimizationSurface.RUNTIME_PARAMS,
    subsystem="cpu_execution",
    causal_rationale=(
        "When the target process oversubscribes physical cores, matching the thread "
        "count to physical cores reduces contention for the diagnosed CPU limit; the "
        "causal objective is throughput and CPU-seconds per token, with a reduction in "
        "context switches as supporting evidence (raw CPU utilization is secondary)."
    ),
    expected_direction="increase",
    prepare=prepare,
)
