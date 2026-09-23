"""Load Phase 3 baseline reports (tolerant of verification-report extras)."""

from __future__ import annotations

import json
from pathlib import Path

from inferenceos.contracts.profiling import BaselineReport

# Phase 3 `verify-phase3` reports embed a full BaselineReport plus these extra keys;
# plain `baseline` reports do not. Ignore extras so both are consumable.
_VERIFICATION_EXTRA_KEYS = {
    "acceptance",
    "overhead",
    "reproducibility_exceptions",
    "result",
}


class BaselineReportError(ValueError):
    """Raised when a report file cannot be interpreted as a BaselineReport."""


def load_baseline_report(path: Path | str) -> BaselineReport:
    report_path = Path(path)
    if not report_path.is_file():
        raise BaselineReportError(f"report not found: {report_path}")
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineReportError(f"invalid JSON in {report_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BaselineReportError(f"unexpected report structure in {report_path}")
    allowed = set(BaselineReport.model_fields)
    filtered = {key: value for key, value in raw.items() if key in allowed}
    if "runs" not in filtered or "hardware" not in filtered:
        raise BaselineReportError(
            f"{report_path} does not look like a Phase 3 BaselineReport"
        )
    try:
        return BaselineReport.model_validate(filtered)
    except ValueError as exc:
        raise BaselineReportError(f"could not validate {report_path}: {exc}") from exc


def find_latest_report(reports_dir: Path) -> Path | None:
    candidates = sorted(
        list(reports_dir.glob("baseline_*.json")) + list(reports_dir.glob("phase3_verification_*.json")),
        key=lambda item: item.stat().st_mtime,
    )
    return candidates[-1] if candidates else None
