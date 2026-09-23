"""Telemetry providers (real OS/GPU counters)."""

from __future__ import annotations

from inferenceos.telemetry.gpu import (
    GpuSample,
    GpuUtilizationSampler,
    GpuUtilizationSummary,
)

__all__ = ["GpuSample", "GpuUtilizationSampler", "GpuUtilizationSummary"]
