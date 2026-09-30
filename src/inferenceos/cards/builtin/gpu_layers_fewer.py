"""Card: move some model layers off the graphics card when its memory is the limit."""

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
    hardware = ctx.hardware
    gpu_available = any(gpu.vram_bytes for gpu in hardware.gpus) or bool(
        hardware.backends_available
    )
    current_raw = ctx.parameter("gpu_layers_offloaded")
    total_raw = ctx.parameter("gpu_layers_total")
    current = int(current_raw) if isinstance(current_raw, (int, float)) else None
    total = int(total_raw) if isinstance(total_raw, (int, float)) else None
    target = (
        max(0, current - max(1, (total or current) // 4)) if current is not None else None
    )
    preconditions = [
        check_evidence_support(ctx, action, policy),
        make(
            "gpu_available",
            PreconditionKind.CAPABILITY,
            "a usable GPU/accelerator backend exists",
            gpu_available,
            "no GPU with VRAM and no backend enumerated",
        ),
        make(
            "layers_currently_offloaded",
            PreconditionKind.APPLICABILITY,
            "layers are currently offloaded",
            (current is not None and current > 0) if current is not None else None,
            f"gpu_layers_offloaded={current}",
        ),
        make(
            "target_below_current",
            PreconditionKind.PARAMETER,
            "target offload is below the current value",
            (target is not None and target < current) if current is not None else None,
            "",
        ),
    ]
    return PreparedAction(
        parameters={"gpu_offload_layers": target} if target is not None else {},
        preconditions=preconditions,
        benefit=unknown_benefit(
            "gpu_memory_used_bytes",
            "decrease",
            confidence=ctx.effective_confidence,
            evidence_basis=list(ctx.evidence_ids),
            notes="per-layer VRAM geometry unavailable; magnitude unknown",
        ),
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "Moving layers to CPU reduces VRAM but typically lowers throughput; "
            "reversible by restoring the prior offload count."
        ),
        rollback_available=True,
    )


CARD = Card(
    card_id="runtime.reduce_gpu_offload_layers",
    name="Reduce GPU offload layers",
    what_it_changes=(
        "Moves some parts of the model from the graphics card back to the processor, "
        "to free graphics card memory when that memory is the limit."
    ),
    acts_on=ActsOn.MODEL_SETTINGS,
    programs=(
        ProgramSetting(
            program=LLAMA_CPP, setting="gpu_offload_layers", how_set="--n-gpu-layers N"
        ),
    ),
    when=When.BOTH,
    needs_restart=True,
    applies_when=(
        "a usable GPU backend exists",
        "layers are currently offloaded",
        "diagnosis evidence supports a GPU memory limit",
    ),
    values_to_try="About a quarter fewer layers than now",
    should_improve=("gpu_memory_used_bytes",),
    might_worsen=("decode_tokens_per_sec", "ttft_s"),
    undo="Restart with the old layer count",
    approval=ApprovalRule.KEPT_CHANGE,
    admin_rights=False,
    works_with=("runtime.reduce_threads",),
    clashes_with=("runtime.increase_gpu_offload_layers",),
    answers=(BottleneckKind.GPU_MEMORY_LIMIT,),
    parameters=(
        ParameterSpec(
            name="gpu_offload_layers",
            type=ParameterType.INTEGER,
            description="Number of layers to offload (lower than current).",
            minimum=0,
            maximum=1024,
            unit="layers",
        ),
    ),
    kind=CandidateKind.GPU_OFFLOAD_LAYERS,
    surface=OptimizationSurface.GPU_OFFLOAD,
    subsystem="gpu_memory",
    causal_rationale=(
        "VRAM footprint scales with the number of offloaded layers; reducing the "
        "layer count lowers VRAM demand when VRAM is the diagnosed limit."
    ),
    expected_direction="decrease",
    prepare=prepare,
)
