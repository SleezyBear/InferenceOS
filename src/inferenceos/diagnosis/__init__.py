"""Deterministic bottleneck detection (Phase 4).

Turns a Phase 3 :class:`BaselineReport` into a structured, evidence-preserving
:class:`BottleneckDiagnosis`. No LLM, no candidate generation, no recommendations.
"""

from __future__ import annotations

from inferenceos.diagnosis.detector import DETECTOR_VERSION, BottleneckDetector
from inferenceos.diagnosis.groups import (
    DIAGNOSTIC_METRICS,
    GroupObservation,
    extract_groups,
)
from inferenceos.diagnosis.loader import (
    BaselineReportError,
    find_latest_report,
    load_baseline_report,
)
from inferenceos.diagnosis.report import render_diagnosis
from inferenceos.diagnosis.rules import (
    RULES,
    CandidateFinding,
    RuleContext,
    run_rules,
)

__all__ = [
    "DETECTOR_VERSION",
    "DIAGNOSTIC_METRICS",
    "RULES",
    "BaselineReportError",
    "BottleneckDetector",
    "CandidateFinding",
    "GroupObservation",
    "RuleContext",
    "extract_groups",
    "find_latest_report",
    "load_baseline_report",
    "render_diagnosis",
    "run_rules",
]
