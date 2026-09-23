"""Real-machine Phase 3 integration: real profiling through the backend adapter.

Skipped unless two real GGUF models and the llama.cpp toolchain are present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.backends.llama_cpp_toolchain import find_cli
from inferenceos.config import Settings
from inferenceos.contracts.enums import BackendType, MeasurementMode, MetricQuality
from inferenceos.models.loader import load_model_profile
from inferenceos.profiling.runner import (
    PROFILER_VERSION,
    BaselineProfiler,
    ProfilerOptions,
    ProfileTask,
)
from inferenceos.profiling.workloads import choose_context_length, get_workload

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATHS = sorted(MODELS_DIR.glob("*.gguf"))


def _llama_available() -> bool:
    try:
        find_cli(Settings().llama_cpp_dir)
    except FileNotFoundError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    len(MODEL_PATHS) < 2 or not _llama_available(),
    reason="Phase 3 integration needs two real GGUF models and the llama.cpp toolchain",
)


@pytest.fixture(scope="module")
def profiler() -> BaselineProfiler:
    return BaselineProfiler()


def _qwen_task() -> ProfileTask:
    model = max((load_model_profile(path) for path in MODEL_PATHS), key=lambda m: m.file_size_bytes)
    return ProfileTask(
        model=model,
        workload=get_workload("short-short"),
        context_length=choose_context_length(model),
    )


def test_run_once_captures_real_metrics(profiler: BaselineProfiler) -> None:
    run = profiler.run_once(
        _qwen_task(),
        mode=MeasurementMode.WARM,
        sequence=1,
        repetition=0,
        warmup=False,
        options=ProfilerOptions(),
    )
    metrics = run.observation.metrics
    assert run.observation.generated_text
    assert run.observation.backend == BackendType.VULKAN
    assert "radeon" in (run.context.config.device or "").lower()
    assert run.context.config.gpu_layers_offloaded == run.context.config.gpu_layers_total
    assert run.context.config.gpu_layers_offloaded
    for name in (
        "model_load_time_s",
        "ttft_s",
        "prefill_latency_s",
        "decode_latency_s",
        "prompt_tokens_per_sec",
        "decode_tokens_per_sec",
        "total_latency_s",
        "prompt_tokens",
        "generated_tokens",
        "peak_rss_bytes",
        "system_cpu_percent",
        "gpu_utilization_percent",
        "total_ram_bytes",
    ):
        assert getattr(metrics, name) is not None, name
    assert metrics.gpu_memory_total_bytes is None
    assert run.metric_quality["ttft_s"] is MetricQuality.DERIVED
    assert run.metric_quality["gpu_memory_total_bytes"] is MetricQuality.UNAVAILABLE
    assert run.observation.system_state is not None
    assert run.context.hardware.hostname
    assert run.context.model.architecture
    assert run.context.workload.prompt_set_ref


def test_profile_repeated_runs_are_aggregated(profiler: BaselineProfiler) -> None:
    tiny = min((load_model_profile(path) for path in MODEL_PATHS), key=lambda m: m.file_size_bytes)
    task = ProfileTask(
        model=tiny,
        workload=get_workload("short-short"),
        context_length=choose_context_length(tiny),
    )
    report = profiler.profile(
        [task], modes=[MeasurementMode.WARM], repetitions=2, options=ProfilerOptions()
    )
    assert report.profiler_version == PROFILER_VERSION
    assert report.resident_warm_supported is False
    assert report.cold_warm_semantics
    measured = [run for run in report.runs if not run.identity.warmup]
    assert len(measured) == 2
    assert any(run.identity.warmup for run in report.runs)
    assert len(report.groups) == 1
    group = report.groups[0]
    assert group.repetitions == 2
    stats = group.metrics["decode_tokens_per_sec"]
    assert stats.count == 2
    assert stats.mean is not None and stats.mean > 0
