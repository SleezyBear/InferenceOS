"""Telemetry providers (real OS/GPU counters)."""

from __future__ import annotations

from inferenceos.telemetry.gpu import (
    GpuSample,
    GpuUtilizationSampler,
    GpuUtilizationSummary,
)
from inferenceos.telemetry.system import (
    aggregate_gpu_samples,
    compute_memory_pressure,
    hard_faults_from_samples,
    parse_gpu_adapter_instance,
    parse_gpu_engine_instance,
    sample_system_state,
)

__all__ = [
    "GpuSample",
    "GpuUtilizationSampler",
    "GpuUtilizationSummary",
    "aggregate_gpu_samples",
    "compute_memory_pressure",
    "hard_faults_from_samples",
    "parse_gpu_adapter_instance",
    "parse_gpu_engine_instance",
    "sample_system_state",
]
