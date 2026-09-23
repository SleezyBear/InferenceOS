"""Real Phase 3 report ingestion -> real deterministic diagnosis (no inference)."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import BottleneckKind
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.diagnosis.loader import load_baseline_report

_REPO = Path(__file__).resolve().parents[1]
_REPORTS = sorted(
    (path for path in (_REPO / "reports").glob("phase3_verification_*.json")),
    key=lambda item: item.stat().st_mtime,
)
_REPORT = _REPORTS[-1] if _REPORTS else None

pytestmark = pytest.mark.skipif(
    _REPORT is None, reason="no real Phase 3 verification report available"
)

_GPU_FAMILY = {
    BottleneckKind.COMPUTE_BOUND,
    BottleneckKind.GPU_UNDERUTILIZATION,
    BottleneckKind.GPU_MEMORY_LIMIT,
}


def test_real_report_ingestion_and_diagnosis() -> None:
    assert _REPORT is not None
    report = load_baseline_report(_REPORT)
    diagnosis = BottleneckDetector().detect(report, source_report_path=str(_REPORT))

    assert diagnosis.context.detector_version == BottleneckDetector.version
    assert diagnosis.context.model_refs
    assert diagnosis.context.hardware.hostname
    assert diagnosis.reasoning
    restored = BottleneckDiagnosis.from_json(diagnosis.to_json())
    assert restored == diagnosis


def test_real_high_gpu_is_not_a_false_positive_bottleneck() -> None:
    assert _REPORT is not None
    report = load_baseline_report(_REPORT)
    diagnosis = BottleneckDetector().detect(report, source_report_path=str(_REPORT))

    meaningful = ([diagnosis.primary] if diagnosis.primary else []) + list(diagnosis.candidates)
    assert all(item.kind not in _GPU_FAMILY for item in meaningful)
    # The sustained-GPU decode workload should be preserved as an explained observation.
    observations = [item for item in diagnosis.observations if item.kind in _GPU_FAMILY]
    if observations:
        assert all(not item.is_meaningful for item in observations)


def test_real_baseline_does_not_manufacture_a_bottleneck() -> None:
    assert _REPORT is not None
    report = load_baseline_report(_REPORT)
    diagnosis = BottleneckDetector().detect(report, source_report_path=str(_REPORT))
    assert (not diagnosis.meaningful_opportunity) or diagnosis.insufficient_evidence
