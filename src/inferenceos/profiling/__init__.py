"""Baseline profiling & telemetry harness (Phase 3).

Measurement only: deterministic workloads, real repeated runs through the backend
adapter, real system telemetry sampling, and persistent baseline reports.

Explicitly NOT here: bottleneck detection, diagnosis, candidate generation,
optimization, or any adaptive behaviour (Phase 4+).
"""

from __future__ import annotations

from inferenceos.profiling.report import group_lookup, render_summary
from inferenceos.profiling.runner import (
    COLD_WARM_SEMANTICS,
    COUNTER_PATHS,
    PROFILER_VERSION,
    BaselineProfiler,
    ProfileTask,
    ProfilerOptions,
)
from inferenceos.profiling.statistics import (
    REPRODUCIBILITY_TOLERANCE,
    summarize,
    within_tolerance,
)
from inferenceos.profiling.workloads import (
    BENCHMARK_SUITE,
    BENCHMARK_SUITE_VERSION,
    BenchmarkWorkload,
    choose_context_length,
    get_workload,
    select_workloads,
    workload_profiles,
)

__all__ = [
    "BENCHMARK_SUITE",
    "BENCHMARK_SUITE_VERSION",
    "COLD_WARM_SEMANTICS",
    "COUNTER_PATHS",
    "PROFILER_VERSION",
    "REPRODUCIBILITY_TOLERANCE",
    "BaselineProfiler",
    "BenchmarkWorkload",
    "ProfileTask",
    "ProfilerOptions",
    "choose_context_length",
    "get_workload",
    "group_lookup",
    "render_summary",
    "select_workloads",
    "summarize",
    "within_tolerance",
    "workload_profiles",
]
