"""Integration: real captured Phase 4/5 artifacts drive inert candidate generation.

This supplements the unit tests with real subsystem output where it is present in
``reports/``. ``inferenceos candidates`` is the authoritative real acceptance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.config import Settings
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.diagnosis.loader import BaselineReportError, load_baseline_report
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates


def _latest(pattern: str) -> Path | None:
    settings = Settings()
    paths = sorted(
        settings.reports_dir.glob(pattern), key=lambda item: item.stat().st_mtime
    )
    return paths[-1] if paths else None


def test_real_pressure_diagnosis_generates_inert_plan() -> None:
    path = _latest("phase4_diagnosis_pressure_*.json")
    if path is None:
        pytest.skip("no real pressure diagnosis artifact available")
    diagnosis = BottleneckDiagnosis.from_json(path.read_text(encoding="utf-8"))
    report = None
    try:
        report = load_baseline_report(diagnosis.context.source_report_path)
    except (BaselineReportError, OSError):
        report = None

    plan = generate_candidates(diagnosis, report=report)
    assert plan.executed is False
    assert plan.environment_mutated is False
    assert plan.source_diagnosis_id == diagnosis.diagnosis_id
    registry_ids = set(build_default_registry().ids())
    assert all(candidate.action_id in registry_ids for candidate in plan.candidates)
    for candidate in plan.candidates:
        assert candidate.preconditions
        if candidate.executable:
            assert candidate.evidence_refs
            assert not candidate.risk.requires_approval


def test_real_baseline_report_generates_no_speculative_candidates() -> None:
    settings = Settings()
    paths = sorted(
        list(settings.reports_dir.glob("baseline_*.json")),
        key=lambda item: item.stat().st_mtime,
    )
    if not paths:
        pytest.skip("no real baseline report available")
    report = load_baseline_report(paths[-1])
    diagnosis = BottleneckDetector().detect(report, source_report_path=str(paths[-1]))
    plan = generate_candidates(diagnosis, report=report)
    assert plan.executed is False
    # A healthy/below-threshold baseline must never manufacture candidates.
    if not diagnosis.meaningful_opportunity:
        assert plan.candidates == []
        assert plan.outcome.value in {"no_opportunity", "insufficient_evidence"}
