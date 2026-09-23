"""Human-readable diagnosis rendering."""

from __future__ import annotations

from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.diagnosis.report import render_diagnosis
from tests.diagnosis_fixtures import make_report, make_run


def _healthy(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "decode_tokens_per_sec": 24.0,
        "total_latency_s": 1.4,
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


def test_render_diagnosis_contains_context_and_decision() -> None:
    runs = [
        make_run(model_name="m", workload_id="short-short", condition="control", metrics=_healthy(hard_page_faults_per_sec=0.5), memory_pressure=0.6),
        make_run(model_name="m", workload_id="short-short", condition="pressured", metrics=_healthy(hard_page_faults_per_sec=17000.0, total_latency_s=2.5), memory_pressure=0.84),
    ]
    diagnosis = BottleneckDetector().detect(make_report(runs), source_report_path="report.json")
    text = render_diagnosis(diagnosis)
    assert "Deterministic bottleneck diagnosis" in text
    assert "report.json" in text
    assert "meaningful opportunity" in text
    assert "Primary finding" in text
    assert "paging" in text
    assert "Reasoning chain" in text


def test_render_diagnosis_for_no_opportunity() -> None:
    runs = [
        make_run(model_name="m", workload_id="short-short", metrics=_healthy()),
        make_run(model_name="m", workload_id="short-short", metrics=_healthy()),
    ]
    diagnosis = BottleneckDetector().detect(make_report(runs), source_report_path="r.json")
    text = render_diagnosis(diagnosis)
    assert "No meaningful optimization opportunity" in text
