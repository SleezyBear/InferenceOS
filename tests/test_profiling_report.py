"""Human-readable summary rendering."""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.contracts.enums import MeasurementMode
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    RunGroupStats,
)
from inferenceos.profiling.report import group_lookup, render_summary
from inferenceos.profiling.statistics import summarize
from inferenceos.selftest import sample_hardware


def _report() -> BaselineReport:
    stats = summarize("decode_tokens_per_sec", [24.0, 24.5, 23.5])
    return BaselineReport(
        profiler_version="0.1.0+phase3",
        suite_version="1.0",
        generated_at=datetime.now(timezone.utc),
        hardware=sample_hardware(),
        backend=ProfilingConfig(
            backend_name="llama.cpp",
            backend_version="b11139",
            device="Radeon Pro 560X",
            gpu_layers_offloaded=28,
            gpu_layers_total=28,
        ),
        groups=[
            RunGroupStats(
                model_ref="sample",
                workload_id="short-short",
                mode=MeasurementMode.WARM,
                repetitions=3,
                metrics={"decode_tokens_per_sec": stats},
            )
        ],
        cold_warm_semantics="documented semantics",
    )


def test_render_summary_lists_context_and_metrics() -> None:
    text = render_summary(_report())
    assert "Baseline report" in text
    assert "llama.cpp" in text
    assert "sample / short-short [warm]" in text
    assert "decode_tokens_per_sec" in text
    assert "cv=" in text


def test_group_lookup_indexes_by_model_workload_mode() -> None:
    lookup = group_lookup(_report())
    assert ("sample", "short-short", "warm") in lookup
