"""Shared fixtures for Phase 5 reasoning tests."""

from __future__ import annotations

import json

from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.reasoning import ReasoningDiagnosisContext
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.reasoning.context import build_reasoning_context
from tests.diagnosis_fixtures import make_report, make_run


def _healthy(**overrides: object) -> dict[str, object]:
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
        "hard_page_faults_per_sec": 0.5,
        "prompt_tokens": 40,
        "generated_tokens": 24,
    }
    base.update(overrides)
    return base


def paging_report() -> BaselineReport:
    # A known, disabled mmap baseline makes ``runtime.enable_mmap`` the applicable
    # paging action (context size is never changed, so no card exists for it).
    runs = [
        make_run(model_name="m", workload_id="short-short", condition="control", metrics=_healthy(hard_page_faults_per_sec=0.5), memory_pressure=0.6, use_mmap=False),
        make_run(model_name="m", workload_id="short-short", condition="control", metrics=_healthy(hard_page_faults_per_sec=0.6), memory_pressure=0.6, use_mmap=False),
        make_run(model_name="m", workload_id="short-short", condition="pressured", metrics=_healthy(hard_page_faults_per_sec=17000.0, total_latency_s=2.5), memory_pressure=0.84, use_mmap=False),
        make_run(model_name="m", workload_id="short-short", condition="pressured", metrics=_healthy(hard_page_faults_per_sec=18000.0, total_latency_s=2.6), memory_pressure=0.85, use_mmap=False),
    ]
    return make_report(runs)


def paging_bundle() -> tuple[ReasoningDiagnosisContext, BottleneckDiagnosis, BaselineReport]:
    report = paging_report()
    diagnosis = BottleneckDetector().detect(report, source_report_path="<test>")
    context = build_reasoning_context(diagnosis, report)
    return context, diagnosis, report


def healthy_bundle() -> tuple[ReasoningDiagnosisContext, BottleneckDiagnosis, BaselineReport]:
    runs = [
        make_run(model_name="m", workload_id="short-short", metrics=_healthy()),
        make_run(model_name="m", workload_id="short-short", metrics=_healthy()),
    ]
    report = make_report(runs)
    diagnosis = BottleneckDetector().detect(report, source_report_path="<test>")
    context = build_reasoning_context(diagnosis, report)
    return context, diagnosis, report


def insufficient_report() -> BaselineReport:
    """Elevated pressure without corroboration: a below-threshold candidate signal."""
    runs = [
        make_run(model_name="m", workload_id="short-short", condition="control", metrics=_healthy(hard_page_faults_per_sec=0.1), memory_pressure=0.6),
        make_run(model_name="m", workload_id="short-short", condition="control", metrics=_healthy(hard_page_faults_per_sec=0.1), memory_pressure=0.6),
        make_run(model_name="m", workload_id="short-short", condition="pressured", metrics=_healthy(hard_page_faults_per_sec=0.1), memory_pressure=0.85),
        make_run(model_name="m", workload_id="short-short", condition="pressured", metrics=_healthy(hard_page_faults_per_sec=0.1), memory_pressure=0.86),
    ]
    return make_report(runs)


def insufficient_bundle() -> tuple[ReasoningDiagnosisContext, BottleneckDiagnosis, BaselineReport]:
    report = insufficient_report()
    diagnosis = BottleneckDetector().detect(report, source_report_path="<test>")
    context = build_reasoning_context(diagnosis, report)
    return context, diagnosis, report


def valid_result_json(
    context: ReasoningDiagnosisContext,
    *,
    root_cause_class: str = "paging",
    affected_stage: str = "end_to_end",
    supporting: list[str] | None = None,
    contradicting: list[str] | None = None,
    confidence: float = 0.7,
    evidence_strength: str = "medium",
    summary: str = "Page-fault activity indicates paging during the run.",
    reasoning: str = "Evidence E1 shows elevated hard page faults.",
    **overrides: object,
) -> str:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "root_cause_class": root_cause_class,
        "affected_stage": affected_stage,
        "contributing_factors": ["elevated page faults"],
        "supporting_evidence_ids": supporting if supporting is not None else [context.evidence[0].evidence_id],
        "contradicting_evidence_ids": contradicting or [],
        "confidence": confidence,
        "evidence_strength": evidence_strength,
        "uncertainty": "moderate",
        "further_evidence_required": False,
        "summary": summary,
        "reasoning": reasoning,
    }
    payload.update(overrides)
    return json.dumps(payload)
