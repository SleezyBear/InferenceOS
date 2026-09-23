"""Deterministic Phase 4 rule tests over synthetic causal fixtures.

Covers positive, negative (false-positive avoidance), ambiguous and stage-specific
cases, including the plan's model-dominates / external-process-dominates /
no-pressure / ambiguous fixtures.
"""

from __future__ import annotations

from inferenceos.contracts.bottleneck import Bottleneck
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    BackendType,
    BottleneckKind,
    EvidenceStrength,
    InferenceStage,
    MeasurementMode,
)
from inferenceos.contracts.profiling import ProfilingRun
from inferenceos.contracts.system_state import ProcessState
from inferenceos.diagnosis.detector import BottleneckDetector
from tests.diagnosis_fixtures import make_report, make_run

_DETECTOR = BottleneckDetector()


def _detect(runs: list[ProfilingRun]) -> BottleneckDiagnosis:
    report = make_report(runs)
    return _DETECTOR.detect(report, source_report_path="<test>")


def _all_findings(diag: BottleneckDiagnosis) -> list[Bottleneck]:
    findings: list[Bottleneck] = list(diag.observations) + list(diag.candidates)
    if diag.primary is not None:
        findings.append(diag.primary)
    return findings


def _find(diag: BottleneckDiagnosis, kind: BottleneckKind) -> list[Bottleneck]:
    return [item for item in _all_findings(diag) if item.kind is kind]


def _healthy_metrics(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "decode_tokens_per_sec": 24.0,
        "prompt_tokens_per_sec": 90.0,
        "total_latency_s": 1.4,
        "prefill_latency_s": 0.45,
        "decode_latency_s": 0.9,
        "model_load_time_s": 2.0,
        "process_cpu_percent": 120.0,
        "system_cpu_percent": 25.0,
        "gpu_utilization_percent": 70.0,
        "peak_rss_bytes": 1_300_000_000,
        "available_ram_bytes": 8 * 1024**3,
        "total_ram_bytes": 16 * 1024**3,
        "hard_page_faults_per_sec": 0.0,
        "prompt_tokens": 40,
        "generated_tokens": 24,
    }
    base.update(overrides)
    return base


def _pair(
    model: str,
    *,
    condition: str,
    metrics: dict[str, object],
    memory_pressure: float = 0.5,
    mode: MeasurementMode | None = None,
    processes: list[ProcessState] | None = None,
    count: int = 2,
    workload: str = "short-short",
    gpu_offload: int = 99,
) -> list[ProfilingRun]:
    actual_mode = mode or MeasurementMode.WARM
    return [
        make_run(
            model_name=model,
            workload_id=workload,
            condition=condition,
            metrics=metrics,
            memory_pressure=memory_pressure,
            mode=actual_mode,
            processes=processes,
            gpu_offload_layers=gpu_offload,
        )
        for _ in range(count)
    ]


# --------------------------------------------------------------------------- #
# Positive evidence cases                                                      #
# --------------------------------------------------------------------------- #
def test_cpu_limited_scenario() -> None:
    runs = _pair("m", condition="gpu", metrics=_healthy_metrics())
    runs += _pair(
        "m",
        condition="cpu",
        metrics=_healthy_metrics(
            gpu_utilization_percent=5.0,
            process_cpu_percent=520.0,
            decode_tokens_per_sec=6.0,
        ),
        gpu_offload=0,
    )
    diag = _detect(runs)
    assert diag.meaningful_opportunity is True
    assert diag.primary is not None
    assert diag.primary.kind is BottleneckKind.CPU
    assert diag.primary.stage is InferenceStage.DECODE
    assert diag.primary.evidence_strength in (EvidenceStrength.MEDIUM, EvidenceStrength.HIGH)
    assert any("decode_tokens_per_sec" in item.metric for item in diag.primary.supporting_evidence)


def test_paging_scenario() -> None:
    runs = _pair("m", condition="control", metrics=_healthy_metrics(hard_page_faults_per_sec=0.5), memory_pressure=0.6)
    runs += _pair(
        "m",
        condition="pressured",
        metrics=_healthy_metrics(
            hard_page_faults_per_sec=18000.0,
            decode_tokens_per_sec=19.0,
            total_latency_s=2.5,
        ),
        memory_pressure=0.84,
    )
    diag = _detect(runs)
    assert diag.meaningful_opportunity is True
    assert diag.primary is not None and diag.primary.kind is BottleneckKind.PAGING
    assert any("hard_page_faults" in item.metric for item in diag.primary.supporting_evidence)


def test_storage_startup_scenario() -> None:
    runs = _pair(
        "m",
        condition="cold",
        metrics=_healthy_metrics(model_load_time_s=3.0, total_latency_s=2.0),
        mode=MeasurementMode.COLD,
    )
    runs += _pair(
        "m",
        condition="warm",
        metrics=_healthy_metrics(model_load_time_s=0.8, total_latency_s=2.0),
        mode=MeasurementMode.WARM,
    )
    diag = _detect(runs)
    storage = _find(diag, BottleneckKind.STORAGE)
    assert storage
    assert any(item.is_meaningful for item in storage)
    meaningful = [item for item in storage if item.is_meaningful][0]
    assert meaningful.stage is InferenceStage.LOAD
    assert any("cold_vs_warm" in item.metric for item in meaningful.supporting_evidence)


def test_model_dominates_paging_not_external_process() -> None:
    processes = [ProcessState(pid=9, name="minor", cpu_percent=1.0, rss_bytes=200_000_000)]
    runs = _pair("m", condition="control", metrics=_healthy_metrics(hard_page_faults_per_sec=0.2), memory_pressure=0.5)
    runs += _pair(
        "m",
        condition="pressured",
        metrics=_healthy_metrics(hard_page_faults_per_sec=15000.0, total_latency_s=2.6),
        memory_pressure=0.9,
        processes=processes,
    )
    diag = _detect(runs)
    assert diag.primary is not None and diag.primary.kind is BottleneckKind.PAGING
    assert not _find(diag, BottleneckKind.EXTERNAL_PROCESS)


def test_external_process_dominates() -> None:
    processes = [ProcessState(pid=7, name="memoryhog", cpu_percent=2.0, rss_bytes=12 * 1024**3)]
    runs = _pair(
        "m",
        condition="pressured",
        metrics=_healthy_metrics(peak_rss_bytes=1_000_000_000, hard_page_faults_per_sec=0.1),
        memory_pressure=0.9,
        processes=processes,
    )
    diag = _detect(runs)
    external = _find(diag, BottleneckKind.EXTERNAL_PROCESS)
    assert external
    assert any(item.is_meaningful for item in external)


# --------------------------------------------------------------------------- #
# Negative / false-positive-avoidance cases                                    #
# --------------------------------------------------------------------------- #
def test_high_gpu_saturation_is_not_a_bottleneck() -> None:
    metrics = _healthy_metrics(
        gpu_utilization_percent=96.0,
        decode_latency_s=5.0,
        total_latency_s=5.4,
        decode_tokens_per_sec=23.5,
        generated_tokens=128,
    )
    diag = _detect(_pair("m", condition="", metrics=metrics))
    assert diag.meaningful_opportunity is False
    assert all(item.kind is not BottleneckKind.COMPUTE_BOUND for item in diag.candidates)
    compute = _find(diag, BottleneckKind.COMPUTE_BOUND)
    assert compute and all(not item.is_meaningful for item in compute)
    assert compute[0].stage is InferenceStage.DECODE


def test_high_rss_without_pressure_is_not_a_bottleneck() -> None:
    metrics = _healthy_metrics(
        peak_rss_bytes=10 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        hard_page_faults_per_sec=0.1,
    )
    diag = _detect(_pair("m", condition="", metrics=metrics, memory_pressure=0.35))
    assert diag.meaningful_opportunity is False
    assert all(not item.is_meaningful for item in _all_findings(diag))


def test_high_cpu_without_degradation_is_not_a_bottleneck() -> None:
    metrics = _healthy_metrics(gpu_utilization_percent=5.0, process_cpu_percent=560.0)
    diag = _detect(_pair("m", condition="", metrics=metrics, gpu_offload=0))
    assert diag.meaningful_opportunity is False
    cpu = _find(diag, BottleneckKind.CPU)
    if cpu:
        assert all(not item.is_meaningful for item in cpu)


def test_transient_gpu_spike_is_not_saturation() -> None:
    runs = [
        make_run(model_name="m", metrics=_healthy_metrics(gpu_utilization_percent=96.0, decode_latency_s=5.0, total_latency_s=5.4, generated_tokens=128)),
        make_run(model_name="m", metrics=_healthy_metrics(gpu_utilization_percent=15.0, decode_latency_s=1.0, total_latency_s=1.4, generated_tokens=128)),
    ]
    diag = _detect(runs)
    assert not _find(diag, BottleneckKind.COMPUTE_BOUND)
    assert diag.meaningful_opportunity is False


def test_single_run_yields_insufficient_evidence() -> None:
    runs = _pair("m", condition="", metrics=_healthy_metrics(), count=1)
    diag = _detect(runs)
    assert diag.meaningful_opportunity is False
    assert diag.insufficient_evidence is True
    assert diag.escalation_required is True


def test_no_opportunity_when_healthy() -> None:
    diag = _detect(_pair("m", condition="", metrics=_healthy_metrics()))
    assert diag.meaningful_opportunity is False
    assert diag.insufficient_evidence is False
    assert diag.escalation_required is False
    assert diag.no_opportunity_reason


# --------------------------------------------------------------------------- #
# Ambiguous / conflicting cases                                                #
# --------------------------------------------------------------------------- #
def test_ambiguous_cpu_and_paging() -> None:
    runs = _pair("cpu-model", condition="gpu", metrics=_healthy_metrics())
    runs += _pair(
        "cpu-model",
        condition="cpu",
        metrics=_healthy_metrics(gpu_utilization_percent=5.0, process_cpu_percent=520.0, decode_tokens_per_sec=6.0),
        gpu_offload=0,
    )
    runs += _pair("io-model", condition="control", metrics=_healthy_metrics(hard_page_faults_per_sec=0.5), memory_pressure=0.6)
    runs += _pair(
        "io-model",
        condition="pressured",
        metrics=_healthy_metrics(hard_page_faults_per_sec=16000.0),
        memory_pressure=0.84,
    )
    diag = _detect(runs)
    assert diag.meaningful_opportunity is False
    assert diag.insufficient_evidence is True
    assert diag.escalation_required is True
    assert diag.primary is None
    kinds = {item.kind for item in _all_findings(diag)}
    assert BottleneckKind.CPU in kinds and BottleneckKind.PAGING in kinds


# --------------------------------------------------------------------------- #
# Stage-specific cases                                                         #
# --------------------------------------------------------------------------- #
def test_prefill_stage_attribution() -> None:
    metrics = _healthy_metrics(
        prefill_latency_s=2.1,
        decode_latency_s=0.5,
        total_latency_s=2.6,
        gpu_utilization_percent=63.0,
        generated_tokens=16,
    )
    diag = _detect(_pair("m", condition="", metrics=metrics))
    latency = _find(diag, BottleneckKind.LATENCY)
    assert latency
    assert any(item.stage is InferenceStage.PREFILL for item in latency)
    assert all(not item.is_meaningful for item in latency)
