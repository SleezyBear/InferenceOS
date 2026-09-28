"""Per-action candidate preparation.

Each registered action binds concrete parameters from the real context, evaluates its
explicit preconditions, and produces bounded benefit/cost estimates. Preparation does
not decide eligibility or ranking (the safety policy and ranker do), and it never
executes anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from inferenceos.contracts.enums import (
    PreconditionKind,
    PrivilegeRequirement,
)
from inferenceos.contracts.optimization import (
    BenefitEstimate,
    CandidateAction,
    CandidatePrecondition,
    CostEstimate,
)
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.cost import estimate_cost
from inferenceos.optimization.estimate import (
    context_reduction_benefit,
    external_process_benefit,
    unknown_benefit,
)
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.preconditions import (
    check_evidence_support,
    make,
    privilege_precondition,
)


@dataclass(frozen=True)
class PreparedAction:
    parameters: dict[str, object]
    preconditions: list[CandidatePrecondition]
    benefit: BenefitEstimate
    cost: CostEstimate
    causal_rationale: str
    failure_consequences: str
    rollback_available: bool
    # A distinct, machine-readable reason when the action is a valid no-op rather than a
    # failure (e.g. the runtime is already at the desired state). ``None`` otherwise.
    no_change_reason: str | None = None


#: Distinct NO_CHANGE reason: the observed runtime is already at (or below) the desired
#: state, so there is no oversubscription remediation to perform.
ALREADY_AT_DESIRED_STATE = "ALREADY_AT_DESIRED_STATE"


def _spec(action: CandidateAction, name: str) -> tuple[float | None, float | None]:
    for spec in action.parameters:
        if spec.name == name:
            return spec.minimum, spec.maximum
    return None, None


def _evidence_refs(ctx: OptimizationContext) -> list[str]:
    return list(ctx.evidence_ids)


def prepare(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction | None:
    handler = _HANDLERS.get(action.action_id)
    if handler is None:
        return None
    return handler(action, ctx, policy)


def _prepare_context_length(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    current_raw = ctx.parameter("context_length")
    current = int(current_raw) if isinstance(current_raw, (int, float)) else None
    trained = ctx.model.trained_context_length if ctx.model is not None else None
    minimum, maximum = _spec(action, "context_length")
    floor = max(policy.min_context_length, int(minimum or 1))

    target: int | None
    if current is None:
        target = None
    else:
        target = max(floor, int(current * (1.0 - policy.context_reduction_fraction)))
        if maximum is not None:
            target = min(target, int(maximum))

    preconditions = [
        check_evidence_support(ctx, action, policy),
        make(
            "current_context_known",
            PreconditionKind.EVIDENCE,
            "current context length is known",
            None if current is None else True,
            "context_length unavailable from the profiling config" if current is None else f"{current}",
        ),
        make(
            "target_below_current",
            PreconditionKind.PARAMETER,
            "target context length is below the current value",
            (target is not None and target < current) if current is not None else None,
            "no-op: target is not smaller" if (current is not None and target is not None and target >= current) else "",
        ),
        make(
            "within_trained_context",
            PreconditionKind.APPLICABILITY,
            "target does not exceed the model's trained context",
            (target <= trained) if (target is not None and trained is not None) else None,
            "trained context unknown" if trained is None else f"trained={trained}",
        ),
    ]
    parameters: dict[str, object] = {"context_length": target} if target is not None else {}
    if current is not None and target is not None:
        benefit = context_reduction_benefit(
            ctx,
            current_context=current,
            proposed_context=target,
            confidence=ctx.effective_confidence,
            evidence_basis=_evidence_refs(ctx),
        )
    else:
        benefit = unknown_benefit(
            "peak_rss_bytes",
            "decrease",
            confidence=ctx.effective_confidence,
            evidence_basis=_evidence_refs(ctx),
            notes="current context length unavailable; magnitude unknown",
        )
    return PreparedAction(
        parameters=parameters,
        preconditions=preconditions,
        benefit=benefit,
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "A smaller context reduces the usable conversation/task length; it does not "
            "modify the model. Reversible by restoring the prior value."
        ),
        rollback_available=True,
    )


def _prepare_mmap(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    current = ctx.parameter("use_mmap")
    enabled = current is True
    disabled = current is False
    preconditions = [
        check_evidence_support(ctx, action, policy),
        make(
            "mmap_config_known",
            PreconditionKind.EVIDENCE,
            "current mmap configuration is known",
            None if current is None else True,
            "use_mmap unavailable from the profiling config" if current is None else f"use_mmap={current}",
        ),
        make(
            "mmap_currently_disabled",
            PreconditionKind.APPLICABILITY,
            "mmap is currently disabled (change would apply)",
            disabled if current is not None else None,
            "already enabled (no-op)" if enabled else "",
        ),
    ]
    return PreparedAction(
        parameters={"use_mmap": True},
        preconditions=preconditions,
        benefit=unknown_benefit(
            "hard_page_faults_per_sec",
            "decrease",
            confidence=ctx.effective_confidence,
            evidence_basis=_evidence_refs(ctx),
            notes="mmap effect on paging is not responsibly estimable before measurement",
        ),
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "Enabling mmap changes weight-loading behaviour only; it does not alter the "
            "model. Reversible by restoring the prior setting."
        ),
        rollback_available=True,
    )


def _prepare_gpu_offload(
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
            evidence_basis=_evidence_refs(ctx),
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


def _prepare_increase_gpu_offload(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    """Resource-aware GPU-offload increase planning.

    The target is *never* the model's full layer count. The planner constructs a safe
    feasible range from real VRAM telemetry, the configured context/KV requirement,
    runtime reserve, other GPU consumers and an explicit safety headroom, then proposes
    a bounded single exploratory step within that range. Empirical selection of the
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
            evidence_basis=_evidence_refs(ctx),
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


def _prepare_threads(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    """Bind the existing oversubscription remediation to the *observed* runtime threads.

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
            "process_cpu_percent",
            "decrease",
            confidence=ctx.effective_confidence,
            evidence_basis=_evidence_refs(ctx),
            notes="thread-count effect on throughput is workload-dependent; magnitude unknown",
        ),
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "Reducing threads can lower throughput; reversible by restoring the prior count."
        ),
        rollback_available=True,
        no_change_reason=no_change_reason,
    )


def _prepare_external_process(
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
            evidence_basis=_evidence_refs(ctx),
        ),
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "Suspending an unrelated process affects the user's other work and must be "
            "human-approved and reversible."
        ),
        rollback_available=True,
    )


def _prepare_pagefile(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    minimum, maximum = _spec(action, "increase_bytes")
    total = ctx.metric("total_ram_bytes")
    available = ctx.metric("available_ram_bytes")
    if total is not None and available is not None:
        shortfall = max(0.0, total - available)
    else:
        shortfall = float(minimum or 268435456)
    lower = float(minimum or 268435456)
    upper = float(maximum or lower)
    increase = int(min(max(shortfall, lower), upper))
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
            evidence_basis=_evidence_refs(ctx),
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


def _prepare_requantize(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction:
    current = ctx.model.quantization.value if ctx.model is not None else "unknown"
    target = "q4_k_m"
    preconditions = [
        check_evidence_support(ctx, action, policy),
        make(
            "smaller_quantization_available",
            PreconditionKind.APPLICABILITY,
            "a smaller quantization is available",
            None if current == "unknown" else (current != target),
            f"current quantization={current}",
        ),
    ]
    return PreparedAction(
        parameters={"target_quantization": target},
        preconditions=preconditions,
        benefit=unknown_benefit(
            "peak_rss_bytes",
            "decrease",
            confidence=ctx.effective_confidence,
            evidence_basis=_evidence_refs(ctx),
            notes="requantization changes model identity and quality; not estimated here",
        ),
        cost=estimate_cost(ctx, action, policy),
        causal_rationale=action.causal_rationale,
        failure_consequences=(
            "Destroys and rewrites model weights; not reversible in place and may degrade "
            "quality. Rejected by policy."
        ),
        rollback_available=False,
    )


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


_HANDLERS: dict[
    str, Callable[[CandidateAction, OptimizationContext, OptimizationPolicy], PreparedAction]
] = {
    "runtime.right_size_context": _prepare_context_length,
    "runtime.enable_mmap": _prepare_mmap,
    "runtime.reduce_gpu_offload_layers": _prepare_gpu_offload,
    "runtime.increase_gpu_offload_layers": _prepare_increase_gpu_offload,
    "runtime.reduce_threads": _prepare_threads,
    "external.suspend_process": _prepare_external_process,
    "system.increase_pagefile": _prepare_pagefile,
    "model.requantize": _prepare_requantize,
}
