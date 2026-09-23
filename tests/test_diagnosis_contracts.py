"""Diagnosis contracts: construction, evidence preservation, JSON round-trips."""

from __future__ import annotations

from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import BottleneckKind, EvidencePolarity, InferenceStage
from inferenceos.diagnosis.detector import BottleneckDetector
from tests.diagnosis_fixtures import make_report, make_run


def _healthy(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "decode_tokens_per_sec": 24.0,
        "total_latency_s": 1.4,
        "prefill_latency_s": 0.45,
        "decode_latency_s": 0.9,
        "model_load_time_s": 2.0,
        "process_cpu_percent": 120.0,
        "gpu_utilization_percent": 70.0,
        "peak_rss_bytes": 1_300_000_000,
        "available_ram_bytes": 8 * 1024**3,
        "hard_page_faults_per_sec": 0.0,
        "generated_tokens": 24,
    }
    base.update(overrides)
    return base


def _paging_diagnosis() -> BottleneckDiagnosis:
    runs = [
        make_run(model_name="m", workload_id="short-short", condition="control", metrics=_healthy(hard_page_faults_per_sec=0.5), memory_pressure=0.6),
        make_run(model_name="m", workload_id="short-short", condition="control", metrics=_healthy(hard_page_faults_per_sec=0.6), memory_pressure=0.6),
        make_run(model_name="m", workload_id="short-short", condition="pressured", metrics=_healthy(hard_page_faults_per_sec=17000.0, total_latency_s=2.5), memory_pressure=0.84),
        make_run(model_name="m", workload_id="short-short", condition="pressured", metrics=_healthy(hard_page_faults_per_sec=18000.0, total_latency_s=2.6), memory_pressure=0.85),
    ]
    return BottleneckDetector().detect(make_report(runs), source_report_path="<test>")


def test_diagnosis_roundtrip() -> None:
    diagnosis = _paging_diagnosis()
    restored = BottleneckDiagnosis.from_json(diagnosis.to_json())
    assert restored == diagnosis


def test_primary_finding_preserves_evidence_and_reasoning() -> None:
    diagnosis = _paging_diagnosis()
    assert diagnosis.primary is not None
    primary = diagnosis.primary
    assert primary.kind is BottleneckKind.PAGING
    assert primary.stage is InferenceStage.END_TO_END
    assert primary.supporting_evidence
    assert all(
        item.polarity is EvidencePolarity.SUPPORTING for item in primary.supporting_evidence
    )
    assert primary.rationale
    assert primary.detector_version == BottleneckDetector.version
    assert diagnosis.reasoning


def test_observation_only_diagnosis_roundtrips() -> None:
    runs = [
        make_run(model_name="m", workload_id="short-short", metrics=_healthy(gpu_utilization_percent=96.0, decode_latency_s=5.0, total_latency_s=5.4, generated_tokens=128)),
        make_run(model_name="m", workload_id="short-short", metrics=_healthy(gpu_utilization_percent=96.5, decode_latency_s=5.1, total_latency_s=5.5, generated_tokens=128)),
    ]
    diagnosis = BottleneckDetector().detect(make_report(runs), source_report_path="<test>")
    assert diagnosis.observations
    assert not diagnosis.meaningful_opportunity
    restored = BottleneckDiagnosis.from_json(diagnosis.to_json())
    assert restored == diagnosis
