"""Resource inspection and feasible-range construction for GPU offloading.

The model is deliberately explicit and conservative:

    total VRAM
      - safety headroom            (fraction of total; never allocated to weights)
      - runtime/workspace reserve  (fraction of total)
      - other GPU consumers        (telemetry-derived, excluding the target estimate)
      - KV cache for the configured context
      = budget for GPU-resident model weights
    max_layers = floor(budget / per_layer_weight_bytes)

Per-layer weight and KV sizes are *analytical estimates* (``estimated=True``); the
used-VRAM term is real telemetry. When telemetry is missing or stale, or the model
geometry / configured context is unknown, the envelope is ``sufficient=False`` and no
higher configuration may be recommended. The configured context is a fixed
constraint: it is never reduced to make more layers fit.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.search import GpuOffloadResourceEnvelope
from inferenceos.contracts.system_state import SystemState
from inferenceos.hardware.gpu import planning_vram_bytes
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.search.policy import SearchPolicy


def total_vram_from_hardware(hardware: HardwareProfile) -> int | None:
    """Inventory capacity only when scalar usage can belong to one device."""
    return planning_vram_bytes(hardware)


def build_resource_envelope(
    *,
    total_vram_bytes: int | None,
    used_vram_bytes: int | None,
    telemetry_available: bool,
    telemetry_observed_at: datetime | None,
    telemetry_source: str | None,
    now: datetime,
    model: ModelProfile | None,
    context_length: int | None,
    current_layers: int | None,
    total_layers: int | None,
    policy: SearchPolicy,
) -> GpuOffloadResourceEnvelope:
    reasons: list[str] = []
    basis: list[str] = []

    if model is None:
        reasons.append("model profile unavailable; per-layer weight geometry unknown")
    if context_length is None or context_length < 1:
        reasons.append("configured context length unavailable; KV requirement unknown")
    if current_layers is None:
        reasons.append("current GPU-offload layer count unavailable")
    if total_layers is None or total_layers < 1:
        reasons.append("model total layer count unavailable")
    if total_vram_bytes is None or total_vram_bytes <= 0:
        reasons.append("total VRAM unavailable from hardware/backend inventory")
    else:
        basis.append(f"total_vram_bytes={total_vram_bytes} (backend device-local memory)")

    telemetry_fresh = False
    if not telemetry_available:
        reasons.append("GPU VRAM telemetry unavailable; free VRAM cannot be established")
    elif used_vram_bytes is None:
        reasons.append("GPU VRAM telemetry present but used VRAM is unavailable")
    elif telemetry_observed_at is None:
        reasons.append("GPU VRAM telemetry has no observation timestamp")
    else:
        age = (now - telemetry_observed_at).total_seconds()
        telemetry_fresh = 0.0 <= age <= policy.telemetry_max_age_s
        if not telemetry_fresh:
            reasons.append(
                f"GPU VRAM telemetry stale ({age:.1f}s > {policy.telemetry_max_age_s:.1f}s)"
            )
        else:
            basis.append(
                f"used_vram_bytes={used_vram_bytes} (telemetry source={telemetry_source!r}, "
                f"age={age:.1f}s)"
            )

    sufficient = not reasons

    free_vram_bytes: int | None = None
    other_consumers_bytes: int | None = None
    safety_headroom_bytes: int | None = None
    runtime_reserve_bytes: int | None = None
    kv_cache_bytes: int | None = None
    per_layer_weight_bytes: int | None = None
    weight_bytes_estimate: int | None = None
    feasible_min: int | None = current_layers
    feasible_max: int | None = current_layers

    if total_vram_bytes is not None and total_vram_bytes > 0:
        safety_headroom_bytes = int(round(policy.safety_headroom_fraction * total_vram_bytes))
        runtime_reserve_bytes = int(round(policy.runtime_reserve_fraction * total_vram_bytes))
        if used_vram_bytes is not None:
            free_vram_bytes = max(total_vram_bytes - used_vram_bytes, 0)

    if model is not None and context_length is not None and context_length >= 1:
        kv_cache_bytes = model.estimate_kv_cache_bytes(
            context_length, kv_dtype_bytes=policy.kv_element_bytes
        )
        basis.append(
            f"kv_cache_bytes={kv_cache_bytes} (architecture-derived estimate at "
            f"context_length={context_length})"
        )

    can_compute_range = (
        sufficient
        and model is not None
        and model.weight_bytes > 0
        and total_layers is not None
        and total_layers >= 1
        and current_layers is not None
        and total_vram_bytes is not None
        and used_vram_bytes is not None
        and kv_cache_bytes is not None
        and safety_headroom_bytes is not None
        and runtime_reserve_bytes is not None
    )

    if can_compute_range:
        assert model is not None  # narrowed by can_compute_range
        assert total_layers is not None
        assert current_layers is not None
        assert total_vram_bytes is not None
        assert used_vram_bytes is not None
        assert kv_cache_bytes is not None
        assert safety_headroom_bytes is not None
        assert runtime_reserve_bytes is not None
        weight_bytes_estimate = model.weight_bytes
        # llama.cpp b11139 counts go up to L+1 (output step first, then the blocks
        # from the last one backwards; ``tune/placement.py``), so the weights are split
        # over L+1 pieces here (an estimate; review 2, 4 Oct 2026).
        pieces = int(total_layers) + 1
        per_layer_weight_bytes = max(math.ceil(model.weight_bytes / pieces), 1)
        target_current_estimate = kv_cache_bytes + current_layers * per_layer_weight_bytes
        other_consumers_bytes = max(used_vram_bytes - target_current_estimate, 0)
        weight_budget = (
            total_vram_bytes
            - other_consumers_bytes
            - safety_headroom_bytes
            - runtime_reserve_bytes
            - kv_cache_bytes
        )
        if weight_budget <= 0:
            computed_max = 0
        else:
            computed_max = weight_budget // per_layer_weight_bytes
        bounded_max = min(pieces, int(computed_max))
        feasible_min = int(current_layers)
        feasible_max = max(int(current_layers), bounded_max)
        basis.append(
            f"per_layer_weight_bytes={per_layer_weight_bytes} "
            f"(weight_bytes={model.weight_bytes} / (total_layers+1)={pieces}; estimate)"
        )
        basis.append(
            f"other_consumers_bytes={other_consumers_bytes} "
            f"(= used - (kv + current*per_layer), floored at 0; estimate)"
        )
        basis.append(
            f"safety_headroom_bytes={safety_headroom_bytes} "
            f"({policy.safety_headroom_fraction:.2%} of total)"
        )
        basis.append(
            f"runtime_reserve_bytes={runtime_reserve_bytes} "
            f"({policy.runtime_reserve_fraction:.2%} of total)"
        )
        basis.append(
            f"feasible_range={feasible_min}-{feasible_max} "
            f"(current preserved as the lower bound)"
        )

    return GpuOffloadResourceEnvelope(
        sufficient=sufficient,
        reasons=reasons,
        total_vram_bytes=total_vram_bytes,
        used_vram_bytes=used_vram_bytes,
        free_vram_bytes=free_vram_bytes,
        other_consumers_bytes=other_consumers_bytes,
        safety_headroom_bytes=safety_headroom_bytes,
        runtime_reserve_bytes=runtime_reserve_bytes,
        kv_cache_bytes=kv_cache_bytes,
        per_layer_weight_bytes=per_layer_weight_bytes,
        weight_bytes_estimate=weight_bytes_estimate,
        context_length=context_length,
        current_layers=current_layers,
        total_layers=total_layers,
        feasible_min=feasible_min,
        feasible_max=feasible_max,
        telemetry_available=telemetry_available,
        telemetry_fresh=telemetry_fresh,
        telemetry_source=telemetry_source,
        telemetry_observed_at=telemetry_observed_at,
        estimated=True,
        basis=basis,
    )


def _latest_gpu_system_state(ctx: OptimizationContext) -> SystemState | None:
    runs = []
    if ctx.group is not None:
        runs = list(ctx.group.runs)
    if not runs and ctx.report is not None:
        runs = list(ctx.report.runs)
    latest: SystemState | None = None
    for run in runs:
        state = run.observation.system_state
        if state is None:
            continue
        if latest is None or state.captured_at >= latest.captured_at:
            latest = state
    return latest


def envelope_from_context(
    ctx: OptimizationContext, policy: SearchPolicy, *, now: datetime | None = None
) -> GpuOffloadResourceEnvelope:
    """Build the resource envelope from a real optimization context."""
    now = now or datetime.now(timezone.utc)
    total_vram = total_vram_from_hardware(ctx.hardware)
    used_vram = ctx.metric("gpu_memory_used_bytes")
    telemetry_available = used_vram is not None
    telemetry_observed_at: datetime | None = None
    telemetry_source: str | None = None

    state = _latest_gpu_system_state(ctx)
    if state is not None:
        available_flag = bool(state.gpu_telemetry_available)
        telemetry_observed_at = state.gpu_observed_at
        telemetry_source = state.gpu_source
        if len(state.gpu_memory) > 1:
            available_flag, used_vram = False, None
            telemetry_source = "multiple GPU adapters without selected-device attribution"
        if available_flag and used_vram is None:
            memory_used = [
                int(m.memory_used_bytes)
                for m in state.gpu_memory
                if m.memory_used_bytes is not None
            ]
            if memory_used:
                used_vram = float(max(memory_used))
                telemetry_available = True
        elif not available_flag:
            telemetry_available = False

    model = ctx.model
    context_length = ctx.config.context_length
    current = ctx.config.gpu_layers_offloaded
    # This calculator adds the output step itself. Engine telemetry's offload
    # total already includes that step; it is not a transformer-block count.
    total = model.n_layers if model is not None else None

    return build_resource_envelope(
        total_vram_bytes=total_vram,
        used_vram_bytes=int(used_vram) if used_vram is not None else None,
        telemetry_available=telemetry_available,
        telemetry_observed_at=telemetry_observed_at,
        telemetry_source=telemetry_source,
        now=now,
        model=model,
        context_length=context_length,
        current_layers=int(current) if current is not None else None,
        total_layers=int(total) if total is not None else None,
        policy=policy,
    )
