"""Profiling contract construction and JSON round-trips."""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.contracts.enums import (
    BackendType,
    MeasurementMode,
    MetricQuality,
    ObservationSource,
)
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunGroupStats,
    RunIdentity,
)
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.profiling.statistics import summarize
from inferenceos.selftest import sample_hardware, sample_model


def _workload() -> WorkloadProfile:
    return WorkloadProfile(
        workload_id="short-short",
        description="short prompt, short generation",
        prompt_set_ref="short-short@1.0#abc",
        prompts=["hello"],
        seed=1234,
        expected_context_length=64,
    )


def _run() -> ProfilingRun:
    now = datetime.now(timezone.utc)
    metrics = RunMetrics(
        model_load_time_s=2.0,
        ttft_s=0.5,
        prefill_latency_s=0.45,
        decode_latency_s=0.9,
        prompt_tokens_per_sec=90.0,
        decode_tokens_per_sec=24.0,
        total_latency_s=1.35,
        process_wall_s=4.0,
        prompt_tokens=40,
        generated_tokens=24,
        total_tokens=64,
        peak_rss_bytes=1_300_000_000,
        process_cpu_percent=110.0,
        system_cpu_percent=25.0,
        gpu_utilization_percent=65.0,
        gpu_memory_total_bytes=None,
        hard_page_faults_per_sec=0.0,
        total_ram_bytes=16 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )
    observation = Observation(
        observation_id="run-1",
        captured_at=now,
        source=ObservationSource.HARNESS,
        mode=MeasurementMode.WARM,
        backend=BackendType.VULKAN,
        device="Radeon Pro 560X",
        model_ref="sample",
        workload_ref="short-short",
        metrics=metrics,
        generated_text="hello world",
    )
    return ProfilingRun(
        identity=RunIdentity(
            run_id="run-1",
            profiler_version="0.1.0+phase3",
            sequence=1,
            repetition=0,
            mode=MeasurementMode.WARM,
            warmup=False,
            started_at=now,
            finished_at=now,
        ),
        context=RunContext(
            hardware=sample_hardware(),
            model=sample_model(),
            workload=_workload(),
            config=ProfilingConfig(
                backend_name="llama.cpp",
                backend_type=BackendType.VULKAN,
                device="Radeon Pro 560X",
                gpu_layers_offloaded=28,
                gpu_layers_total=28,
                context_length=2048,
                n_predict=32,
                seed=1234,
                temperature=0.0,
            ),
        ),
        observation=observation,
        metric_quality={
            "decode_tokens_per_sec": MetricQuality.MEASURED,
            "ttft_s": MetricQuality.DERIVED,
            "gpu_memory_total_bytes": MetricQuality.UNAVAILABLE,
        },
    )


def test_profiling_run_roundtrip() -> None:
    run = _run()
    restored = ProfilingRun.model_validate_json(run.model_dump_json())
    assert restored == run
    assert restored.context.hardware.hostname
    assert restored.context.model.architecture
    assert restored.metric_quality["ttft_s"] is MetricQuality.DERIVED


def test_baseline_report_roundtrip() -> None:
    run = _run()
    stats = summarize("decode_tokens_per_sec", [24.0, 24.5, 23.5])
    report = BaselineReport(
        profiler_version="0.1.0+phase3",
        suite_version="1.0",
        generated_at=datetime.now(timezone.utc),
        hardware=sample_hardware(),
        backend=run.context.config,
        workloads=[_workload()],
        runs=[run],
        groups=[
            RunGroupStats(
                model_ref="sample",
                workload_id="short-short",
                mode=MeasurementMode.WARM,
                repetitions=3,
                metrics={"decode_tokens_per_sec": stats},
            )
        ],
        cold_warm_semantics="documented",
    )
    restored = BaselineReport.from_json(report.to_json())
    assert restored == report
    assert restored.resident_warm_supported is False
    assert restored.groups[0].metrics["decode_tokens_per_sec"].count == 3
