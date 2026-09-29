"""GPU-memory-limit diagnosis → reduce-offload action (end-to-end path).

The ``runtime.reduce_gpu_offload_layers`` action was registered and executor-supported
but had no diagnosis rule that could ever emit ``GPU_MEMORY_LIMIT``, so it was
unreachable. These tests prove the condition is diagnosed from real VRAM telemetry plus
comparative evidence, and that the existing action is then generated, applied, and
exactly restored.
"""

from __future__ import annotations

from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import BackendType, BottleneckKind, CandidateEligibility
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.optimization import CandidatePlan
from inferenceos.contracts.profiling import BaselineReport, ProfilingRun
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from tests.diagnosis_fixtures import make_report, make_run

_VRAM_4GIB = 4 * 1024**3


def _run(*, condition: str, used_bytes: int, latency: float, offload: int = 24, total: int = 28) -> ProfilingRun:
    return make_run(
        condition=condition,
        backend=BackendType.VULKAN,
        gpu_offload_layers=offload,
        gpu_offload_total=total,
        gpu_memory_used_bytes=used_bytes,
        gpu_telemetry_available=True,
        metrics={
            "decode_tokens_per_sec": 20.0,
            "total_latency_s": latency,
            "gpu_utilization_percent": 92.0,
            "process_cpu_percent": 120.0,
            "peak_rss_bytes": 1_300_000_000,
            "available_ram_bytes": float(8 * 1024**3),
            "total_ram_bytes": float(16 * 1024**3),
            "hard_page_faults_per_sec": 0.5,
        },
    )


def _report(runs: list[ProfilingRun]) -> BaselineReport:
    return make_report(runs)


def _detect(runs: list[ProfilingRun]) -> BottleneckDiagnosis:
    return BottleneckDetector().detect(_report(runs), source_report_path="<test>")


# --------------------------------------------------------------------------- #
# Positive: high VRAM + comparative evidence -> meaningful GPU memory limit     #
# --------------------------------------------------------------------------- #
def test_high_vram_with_comparative_evidence_is_meaningful() -> None:
    runs = [
        _run(condition="full", used_bytes=int(3.9 * 1024**3), latency=2.6),
        _run(condition="full", used_bytes=int(3.9 * 1024**3), latency=2.6),
        _run(condition="partial", used_bytes=int(2.0 * 1024**3), latency=1.4, offload=12),
        _run(condition="partial", used_bytes=int(2.0 * 1024**3), latency=1.4, offload=12),
    ]
    diag = _detect(runs)
    assert diag.primary is not None
    assert diag.primary.kind is BottleneckKind.GPU_MEMORY_LIMIT
    assert diag.primary.is_meaningful is True
    assert any("gpu_memory_used" in item.metric for item in diag.primary.supporting_evidence)


# --------------------------------------------------------------------------- #
# Negative: high VRAM alone is only an observation (anti-triviality)            #
# --------------------------------------------------------------------------- #
def test_high_vram_without_comparative_evidence_is_not_meaningful() -> None:
    runs = [
        _run(condition="full", used_bytes=int(3.9 * 1024**3), latency=2.6),
        _run(condition="full", used_bytes=int(3.9 * 1024**3), latency=2.6),
    ]
    diag = _detect(runs)
    gpu = [o for o in diag.observations if o.kind is BottleneckKind.GPU_MEMORY_LIMIT]
    assert gpu
    assert all(not item.is_meaningful for item in gpu)
    assert diag.primary is None or diag.primary.kind is not BottleneckKind.GPU_MEMORY_LIMIT


def test_low_vram_usage_is_not_flagged() -> None:
    runs = [
        _run(condition="full", used_bytes=int(1.0 * 1024**3), latency=1.4),
        _run(condition="full", used_bytes=int(1.0 * 1024**3), latency=1.4),
    ]
    diag = _detect(runs)
    assert all(
        item.kind is not BottleneckKind.GPU_MEMORY_LIMIT
        for item in list(diag.observations) + list(diag.candidates)
        + ([diag.primary] if diag.primary else [])
    )


def test_no_finding_without_offload_configured() -> None:
    runs = [
        _run(condition="full", used_bytes=int(3.9 * 1024**3), latency=2.6, offload=0),
        _run(condition="full", used_bytes=int(3.9 * 1024**3), latency=2.6, offload=0),
    ]
    diag = _detect(runs)
    assert all(
        item.kind is not BottleneckKind.GPU_MEMORY_LIMIT
        for item in list(diag.observations) + list(diag.candidates)
        + ([diag.primary] if diag.primary else [])
    )


def test_no_finding_without_total_vram() -> None:
    runs = [
        _run(condition="full", used_bytes=int(3.9 * 1024**3), latency=2.6),
        _run(condition="full", used_bytes=int(3.9 * 1024**3), latency=2.6),
    ]
    report = _report(runs)
    hardware = report.hardware.model_copy(update={"gpus": [], "backend_availability": []})
    report = report.model_copy(update={"hardware": hardware})
    diag = BottleneckDetector().detect(report, source_report_path="<test>")
    assert all(
        item.kind is not BottleneckKind.GPU_MEMORY_LIMIT
        for item in list(diag.observations) + list(diag.candidates)
        + ([diag.primary] if diag.primary else [])
    )


# --------------------------------------------------------------------------- #
# Action chain: generate → eligibility → apply → verify → restore               #
# --------------------------------------------------------------------------- #
def _positive_plan() -> tuple[BottleneckDiagnosis, BaselineReport, CandidatePlan]:
    runs = [
        _run(condition="full", used_bytes=int(3.9 * 1024**3), latency=2.6),
        _run(condition="full", used_bytes=int(3.9 * 1024**3), latency=2.6),
        _run(condition="partial", used_bytes=int(2.0 * 1024**3), latency=1.4, offload=12),
        _run(condition="partial", used_bytes=int(2.0 * 1024**3), latency=1.4, offload=12),
    ]
    report = _report(runs)
    diag = BottleneckDetector().detect(report, source_report_path="<test>")
    return diag, report, generate_candidates(diag, report=report)


def test_reduce_offload_candidate_is_generated_and_executable() -> None:
    _, _, plan = _positive_plan()
    candidate = next(
        (c for c in plan.candidates if c.action_id == "runtime.reduce_gpu_offload_layers"),
        None,
    )
    assert candidate is not None
    assert candidate.eligibility is CandidateEligibility.ELIGIBLE
    assert candidate.executable is True
    target = candidate.parameters["gpu_offload_layers"]
    assert isinstance(target, int) and 0 <= target < 24


def test_executor_applies_and_exactly_restores_offload() -> None:
    _, _, plan = _positive_plan()
    candidate = next(
        c for c in plan.candidates if c.action_id == "runtime.reduce_gpu_offload_layers"
    )
    action = build_default_registry().get(candidate.action_id)
    executor = RuntimeConfigurationExecutor()
    baseline = RuntimeConfiguration(context_length=2048, gpu_offload_layers=24, threads=12)
    executor.set_configuration(baseline)
    applied = executor.apply(action, candidate.parameters)
    assert applied.applied is True
    assert executor.current().gpu_offload_layers == candidate.parameters["gpu_offload_layers"]
    restored = executor.restore(baseline)
    assert restored.verified is True
    assert executor.current() == baseline


# --------------------------------------------------------------------------- #
# Regression: unrelated GPU/CPU diagnoses unchanged                             #
# --------------------------------------------------------------------------- #
def test_gpu_underutilization_still_diagnosed() -> None:
    runs = [
        _run(condition="control", used_bytes=int(1.0 * 1024**3), latency=1.4, offload=8),
        _run(condition="control", used_bytes=int(1.0 * 1024**3), latency=1.4, offload=8),
    ]
    runs += [
        make_run(
            condition="contended",
            backend=BackendType.VULKAN,
            gpu_offload_layers=8,
            gpu_offload_total=28,
            gpu_memory_used_bytes=int(1.0 * 1024**3),
            gpu_telemetry_available=True,
            metrics={
                "decode_tokens_per_sec": 6.0,
                "total_latency_s": 2.6,
                "gpu_utilization_percent": 5.0,
                "process_cpu_percent": 520.0,
                "peak_rss_bytes": 1_300_000_000,
                "available_ram_bytes": float(8 * 1024**3),
                "total_ram_bytes": float(16 * 1024**3),
                "hard_page_faults_per_sec": 0.5,
            },
        )
        for _ in range(2)
    ]
    diag = _detect(runs)
    assert diag.primary is not None and diag.primary.kind is BottleneckKind.GPU_UNDERUTILIZATION
