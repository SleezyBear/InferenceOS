"""Card: put more of the model on the graphics card when it is sitting idle."""

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
from inferenceos.tune.declarations import declaration

#: Kept as a name other modules can import (the layer search is specific to this card).
CARD_ID = "runtime.increase_gpu_offload_layers"


def prepare(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    """Resource-aware GPU-offload increase planning.

    The target is *never* the model's full layer count. The planner constructs a safe
    feasible range from real VRAM telemetry, the configured KV requirement, runtime
    reserve, other GPU consumers and an explicit safety headroom, then proposes a
    bounded single exploratory step within that range. Empirical selection of the
    final value is performed by the resource-aware search runner, not here.
    """
    from inferenceos.optimization.search.runner import plan_gpu_offload

    hardware = ctx.hardware
    gpu_available = any(gpu.vram_bytes for gpu in hardware.gpus) or bool(
        hardware.backends_available
    )
    current_raw = ctx.parameter("gpu_layers_offloaded")
    total_raw = ctx.parameter("gpu_layers_total")
    current = int(current_raw) if isinstance(current_raw, (int, float)) else None
    total = int(total_raw) if isinstance(total_raw, (int, float)) else None
    if total is None and ctx.model is not None:
        total = ctx.model.n_layers

    plan = plan_gpu_offload(ctx, policy.search)
    envelope = plan.envelope
    probe = plan.planned_probe
    feasible_max = envelope.feasible_max
    range_text = (
        f"feasible range {envelope.feasible_min}-{feasible_max}"
        if envelope.feasible_min is not None and feasible_max is not None
        else "feasible range unavailable"
    )
    telemetry_ready = bool(
        envelope.telemetry_available
        and envelope.telemetry_fresh
        and envelope.used_vram_bytes is not None
    )
    geometry_ready = bool(ctx.model is not None and envelope.per_layer_weight_bytes)
    context_ready = bool(ctx.config.context_length)
    range_ready = bool(
        envelope.sufficient and feasible_max is not None and feasible_max > (current or 0)
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
            "offload_configuration_known",
            PreconditionKind.EVIDENCE,
            "current offload count and layer total are known",
            (current is not None and total is not None),
            f"gpu_layers_offloaded={current} gpu_layers_total={total}",
        ),
        make(
            "vram_telemetry_available",
            PreconditionKind.EVIDENCE,
            "fresh GPU VRAM telemetry is available",
            telemetry_ready,
            (
                f"used_vram={envelope.used_vram_bytes} "
                f"source={envelope.telemetry_source!r} fresh={envelope.telemetry_fresh}"
            ),
        ),
        make(
            "model_geometry_known",
            PreconditionKind.EVIDENCE,
            "per-layer weight geometry is available",
            geometry_ready,
            f"per_layer_weight_bytes={envelope.per_layer_weight_bytes}",
        ),
        make(
            "context_known",
            PreconditionKind.EVIDENCE,
            "the configured context length is known",
            context_ready,
            f"context_length={envelope.context_length}",
        ),
        make(
            "safe_feasible_range_established",
            PreconditionKind.APPLICABILITY,
            "a safe higher configuration exists within the resource envelope",
            range_ready if envelope.sufficient else None,
            (
                f"{range_text}; reasons={envelope.reasons}"
                if not envelope.sufficient
                else range_text
            ),
        ),
        make(
            "target_above_current",
            PreconditionKind.PARAMETER,
            "target offload is above the current value",
            (probe is not None and probe > current) if current is not None else None,
            "" if (probe is not None and current is not None and probe > current)
            else "no safe higher configuration (no-op)",
        ),
        make(
            "within_layer_total",
            PreconditionKind.APPLICABILITY,
            "target offload does not exceed the layer total",
            (probe <= total) if (probe is not None and total is not None) else None,
            "layer total unknown" if total is None else f"total={total}",
        ),
        make(
            "within_feasible_range",
            PreconditionKind.APPLICABILITY,
            "target offload is within the safe feasible range",
            (probe <= feasible_max) if (probe is not None and feasible_max is not None) else None,
            f"target={probe} feasible_max={feasible_max}",
        ),
    ]
    parameters: dict[str, object] = (
        {"gpu_offload_layers": probe}
        if probe is not None and current is not None and probe > current
        else {}
    )
    return PreparedAction(
        parameters=parameters,
        preconditions=preconditions,
        benefit=unknown_benefit(
            "gpu_utilization_percent",
            "increase",
            confidence=ctx.effective_confidence,
            evidence_basis=list(ctx.evidence_ids),
            notes=(
                "resource-aware search: " + range_text + "; the final value must be "
                "selected empirically from measured throughput/latency/VRAM tradeoffs"
            ),
        ),
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "Offloading more layers raises VRAM demand; a candidate outside the safe "
            "feasible range may fail to load or trigger paging. Reversible by restoring "
            "the prior offload count."
        ),
        rollback_available=True,
    )


CARD = Card(
    card_id=CARD_ID,
    name="Increase GPU offload layers",
    what_it_changes=(
        "Moves more parts of the model onto the graphics card when the card is sitting "
        "idle and the processor is doing all the work."
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
        "current offload count and layer total are known",
        "target offload is above the current value and does not exceed the layer total",
    ),
    values_to_try="From the current number up to what fits in graphics card memory",
    should_improve=("gpu_utilization_percent", "decode_tokens_per_sec"),
    might_worsen=("gpu_memory_used_bytes", "peak_rss_bytes"),
    undo="Restart with the old layer count",
    approval=ApprovalRule.KEPT_CHANGE,
    admin_rights=False,
    works_with=("runtime.reduce_threads",),
    clashes_with=("runtime.reduce_gpu_offload_layers",),
    answers=(BottleneckKind.GPU_UNDERUTILIZATION,),
    parameters=(
        ParameterSpec(
            name="gpu_offload_layers",
            type=ParameterType.INTEGER,
            description=(
                "Number of layers to offload (higher than current, up to the layer total)."
            ),
            minimum=0,
            maximum=1024,
            unit="layers",
        ),
    ),
    kind=CandidateKind.GPU_OFFLOAD_LAYERS,
    surface=OptimizationSurface.GPU_OFFLOAD,
    subsystem="gpu_offload",
    causal_rationale=(
        "When offload is configured but the GPU is idle while process CPU is "
        "saturated, shifting more layers onto the accelerator can raise GPU "
        "utilization and throughput; this is an inference-level runtime parameter, "
        "not a model edit."
    ),
    expected_direction="increase",
    prepare=prepare,
    tune=declaration("layers"),
)
