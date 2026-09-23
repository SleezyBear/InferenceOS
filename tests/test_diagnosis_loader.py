"""BaselineReport loader: plain and verification-style reports, error handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferenceos.diagnosis.loader import (
    BaselineReportError,
    find_latest_report,
    load_baseline_report,
)
from tests.diagnosis_fixtures import make_report, make_run


def _write_report(path: Path, *, extra: dict[str, object] | None = None) -> Path:
    runs = [
        make_run(model_name="m", workload_id="short-short", metrics={"decode_tokens_per_sec": 24.0}),
        make_run(model_name="m", workload_id="short-short", metrics={"decode_tokens_per_sec": 24.1}),
    ]
    payload = json.loads(make_report(runs).to_json())
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_plain_report(tmp_path: Path) -> None:
    path = _write_report(tmp_path / "baseline_x.json")
    report = load_baseline_report(path)
    assert len(report.runs) == 2
    assert report.hardware.hostname


def test_loads_verification_report_with_extra_keys(tmp_path: Path) -> None:
    path = _write_report(
        tmp_path / "phase3_verification_x.json",
        extra={"acceptance": {"ok": True}, "overhead": {}, "result": "PASS"},
    )
    report = load_baseline_report(path)
    assert len(report.runs) == 2


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(BaselineReportError):
        load_baseline_report(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(BaselineReportError):
        load_baseline_report(path)


def test_non_report_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(BaselineReportError):
        load_baseline_report(path)


def test_find_latest_report_prefers_newest(tmp_path: Path) -> None:
    _write_report(tmp_path / "baseline_old.json")
    newer = _write_report(tmp_path / "phase3_verification_new.json")
    assert find_latest_report(tmp_path) == newer
    assert find_latest_report(tmp_path / "empty") is None
