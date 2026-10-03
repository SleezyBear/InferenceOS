"""Tests for the layer-curve study orchestration (no real models needed)."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.study_layer_curves import (
    ModelConditions,
    ModelRecord,
    ModelSpec,
    StudySettings,
    build_plan,
    curves_rows,
    run_study,
    summary_lines,
)


def _spec(label: str, path: Path, design: str = "standard") -> ModelSpec:
    return ModelSpec(label, str(path), design)


def _conditions(on_battery: bool = False) -> ModelConditions:
    return ModelConditions(
        checked_at="2026-01-01T00:00:00+00:00",
        power_source="battery" if on_battery else "ac",
        on_battery=on_battery,
        free_graphics_memory_bytes=5_000_000_000,
        total_graphics_memory_bytes=8_000_000_000,
    )


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def _quiet(_message: str) -> None:
    return None


def _write_report(
    path: Path, label: str, pairs: Sequence[tuple[int, float | None, bool]]
) -> None:
    """Write a small report shaped like ``check_layer_range.py`` output."""
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for layers, speed, does_not_fit in pairs:
        rows.append(
            {
                "layers": layers,
                "ok": not does_not_fit,
                "writing_speed": speed,
                "writing_speed_runs": [] if speed is None else [speed * 0.95, speed * 1.05],
                "first_word_s": 0.25,
                "graphics_memory_bytes": 1_000_000_000,
                "notes": ["out of memory"] if does_not_fit else [],
                "pass": 1,
            }
        )
        summary.append(
            {
                "layers": layers,
                "measured": 0 if speed is None else 1,
                "writing_speed": speed,
                "first_word_s": 0.25,
                "slowest_run": None if speed is None else speed * 0.95,
                "fastest_run": None if speed is None else speed * 1.05,
                "runs_measured": 0 if speed is None else 2,
                "above_estimated_fit": False,
                "does_not_fit": does_not_fit,
            }
        )
    path.write_text(
        json.dumps({"model": label, "runs": rows, "summary": summary}), encoding="utf-8"
    )


class FakeRunner:
    """A stand-in for ``run_one`` that writes a small report instead of measuring."""

    def __init__(
        self,
        reports: Mapping[str, Sequence[tuple[int, float | None, bool]]],
        fail: Sequence[str] = (),
    ) -> None:
        self.calls: list[str] = []
        self._reports = reports
        self._fail = set(fail)

    def __call__(self, model: ModelSpec, report_path: Path, log_path: Path) -> int:
        self.calls.append(model.label)
        log_path.write_text("fake log\n", encoding="utf-8")
        if model.label in self._fail:
            return 1
        _write_report(report_path, model.label, self._reports[model.label])
        return 0


class FakeSleep:
    """Records every requested sleep length instead of waiting."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _models(tmp_path: Path, labels: Sequence[str]) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    for label in labels:
        path = tmp_path / f"{label}.gguf"
        path.write_bytes(b"x")
        models.append(_spec(label, path))
    return models


def test_missing_model_file_stops_before_running(tmp_path: Path) -> None:
    present = tmp_path / "present.gguf"
    present.write_bytes(b"x")
    absent = tmp_path / "absent.gguf"
    runner = FakeRunner({})
    sleep = FakeSleep()
    settings = StudySettings(output_dir=tmp_path / "out")

    result = run_study(
        [_spec("a", present), _spec("b", absent)],
        settings,
        run_one=runner,
        sleep=sleep,
        now=_now,
        emit=_quiet,
        read_conditions=lambda: _conditions(),
    )

    assert result.missing_files == [str(absent)]
    assert runner.calls == []
    assert not settings.output_dir.exists()


def test_cooldown_after_every_model_except_last(tmp_path: Path) -> None:
    models = _models(tmp_path, ["m0", "m1", "m2"])
    reports = {model.label: [(0, 20.0, False), (4, 25.0, False)] for model in models}
    runner = FakeRunner(reports)
    sleep = FakeSleep()
    settings = StudySettings(output_dir=tmp_path / "out", cooldown_min=10.0)

    run_study(
        models,
        settings,
        run_one=runner,
        sleep=sleep,
        now=_now,
        emit=_quiet,
        read_conditions=lambda: _conditions(),
    )

    assert runner.calls == ["m0", "m1", "m2"]
    # Two gaps of ten one-minute waits; no wait after the last model.
    assert len(sleep.calls) == 20
    assert sum(sleep.calls) == pytest.approx(600.0 * 2)


def test_failing_model_is_recorded_and_next_still_runs(tmp_path: Path) -> None:
    models = _models(tmp_path, ["a", "b"])
    runner = FakeRunner({"b": [(0, 20.0, False)]}, fail=["a"])
    settings = StudySettings(output_dir=tmp_path / "out")

    result = run_study(
        models,
        settings,
        run_one=runner,
        sleep=FakeSleep(),
        now=_now,
        emit=_quiet,
        read_conditions=lambda: _conditions(),
    )

    assert [record.label for record in result.records] == ["a", "b"]
    assert runner.calls == ["a", "b"]
    assert result.records[0].status == "failed"
    assert result.records[0].exit_code == 1
    assert result.records[1].status == "finished"


def test_resume_skips_finished_models(tmp_path: Path) -> None:
    out = tmp_path / "study"
    out.mkdir()
    models = _models(tmp_path, ["a", "b"])
    a_report = out / "a.json"
    a_report.write_text(json.dumps({"rows": [], "summary": []}), encoding="utf-8")
    (out / "study.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "label": "a",
                        "exit_code": 0,
                        "report_path": str(a_report),
                        "status": "finished",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = FakeRunner({"b": [(0, 20.0, False)]})
    settings = StudySettings(output_dir=out, resume=out)

    result = run_study(
        models,
        settings,
        run_one=runner,
        sleep=FakeSleep(),
        now=_now,
        emit=_quiet,
        read_conditions=lambda: _conditions(),
    )

    assert runner.calls == ["b"]
    statuses = {record.label: record.status for record in result.records}
    assert statuses["a"] == "skipped"
    assert statuses["b"] == "finished"


def test_curves_and_summary_files_have_expected_rows(tmp_path: Path) -> None:
    model = _models(tmp_path, ["a"])[0]
    runner = FakeRunner({"a": [(0, 20.0, False), (4, 25.0, False), (6, 24.0, False), (8, None, True)]})
    settings = StudySettings(output_dir=tmp_path / "out")

    run_study(
        [model],
        settings,
        run_one=runner,
        sleep=FakeSleep(),
        now=_now,
        emit=_quiet,
        read_conditions=lambda: _conditions(),
    )

    with (settings.output_dir / "curves.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[0]["label"] == "a"
    assert rows[0]["layers"] == "0"
    assert rows[0]["pass"] == "1"
    assert rows[0]["status"] == "measured"
    assert float(rows[0]["fastest_run"]) == pytest.approx(21.0)
    assert rows[-1]["status"] == "does not fit"

    summary = (settings.output_dir / "summary.txt").read_text(encoding="utf-8")
    assert "a [standard]" in summary
    assert "fastest 4 @" in summary
    assert "0 layers @ 20.00" in summary
    assert "first that doesn't fit: 8" in summary
    assert "on battery: no" in summary


def test_build_plan_filters_and_keeps_order() -> None:
    models = [ModelSpec(f"m{i}", f"m{i}.gguf", "standard") for i in range(4)]
    plan = build_plan(models, ("m3", "m1"))
    assert [model.label for model in plan] == ["m1", "m3"]
    assert [model.label for model in build_plan(models)] == ["m0", "m1", "m2", "m3"]


def test_curves_rows_and_summary_helpers(tmp_path: Path) -> None:
    report = tmp_path / "r.json"
    _write_report(report, "x", [(0, 10.0, False), (2, None, True)])
    record = ModelRecord(
        label="x",
        design="standard",
        path="x.gguf",
        report_path=str(report),
        log_path=str(tmp_path / "x.log"),
    )
    rows = curves_rows([record])
    assert [row["status"] for row in rows] == ["measured", "does not fit"]
    assert any("first that doesn't fit: 2" in line for line in summary_lines([record]))


def test_rebuild_csv_reads_the_runs_key(tmp_path: Path) -> None:
    # Regression: the per-model JSON stores its rows under "runs", not "rows".
    out = tmp_path / "study"
    out.mkdir()
    _write_report(out / "m.json", "m", [(0, 20.0, False), (2, 10.0, False)])
    (out / "study.json").write_text(
        json.dumps({"models": [{"label": "m", "design": "standard", "path": "m.gguf"}]}),
        encoding="utf-8",
    )
    from scripts.study_layer_curves import rebuild_outputs

    assert rebuild_outputs(out) == (2, 1)
    with (out / "curves.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["label"] == "m"
    assert (out / "summary.txt").read_text(encoding="utf-8").strip() != ""
