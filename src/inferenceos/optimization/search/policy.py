"""Explicit, inspectable policy for the resource-aware optimization search.

Every safety-critical number (VRAM safety headroom, runtime reserve, telemetry
freshness ceiling, sampling granularity, diminishing-returns margin) is a named,
validated field rather than a magic constant inside the range or selection code.
"""

from __future__ import annotations

from pydantic import Field

from inferenceos.config import Settings
from inferenceos.contracts.base import ContractBase


class SearchPolicy(ContractBase):
    """Bounds a search so it is safe, deterministic and economical."""

    # Fraction of total VRAM reserved and never allocated to model weights.
    safety_headroom_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    # Fraction of total VRAM reserved for runtime/workspace allocations.
    runtime_reserve_fraction: float = Field(default=0.05, ge=0.0, lt=1.0)
    # VRAM telemetry older than this is stale and cannot justify an increase.
    telemetry_max_age_s: float = Field(default=30.0, gt=0.0)
    # Deterministic sampling granularity across the feasible range.
    sample_step_layers: int = Field(default=2, ge=1)
    max_samples: int = Field(default=6, ge=2)
    # Planned Phase 6 probe: how far above the current value a single exploratory
    # candidate may move (the empirical search may move further).
    probe_step_layers: int = Field(default=2, ge=1)
    # Two candidates whose relative throughput is within this margin are treated as
    # equivalent, and the smaller (cheaper) value wins. Reuses the Phase 7
    # min-effect scale by default.
    diminishing_returns_margin: float = Field(default=0.05, ge=0.0, lt=1.0)
    kv_element_bytes: int = Field(default=2, ge=1)

    def as_dict(self) -> dict[str, object]:
        return {
            "safety_headroom_fraction": self.safety_headroom_fraction,
            "runtime_reserve_fraction": self.runtime_reserve_fraction,
            "telemetry_max_age_s": self.telemetry_max_age_s,
            "sample_step_layers": self.sample_step_layers,
            "max_samples": self.max_samples,
            "probe_step_layers": self.probe_step_layers,
            "diminishing_returns_margin": self.diminishing_returns_margin,
            "kv_element_bytes": self.kv_element_bytes,
        }


def build_search_policy(settings: Settings) -> SearchPolicy:
    return SearchPolicy(
        safety_headroom_fraction=settings.phase6_gpu_offload_safety_headroom_fraction,
        runtime_reserve_fraction=settings.phase6_gpu_offload_runtime_reserve_fraction,
        telemetry_max_age_s=settings.phase6_gpu_offload_telemetry_max_age_s,
        sample_step_layers=settings.phase6_gpu_offload_sample_step_layers,
        max_samples=settings.phase6_gpu_offload_max_samples,
        probe_step_layers=settings.phase6_gpu_offload_probe_step_layers,
        diminishing_returns_margin=settings.phase6_gpu_offload_diminishing_returns_margin,
        kv_element_bytes=settings.phase6_gpu_offload_kv_element_bytes,
    )
