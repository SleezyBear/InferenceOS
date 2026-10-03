"""The layer-search replay: curves, variants, truth, seeding and the report."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from inferenceos.tune.layer_replay import (
    best_truth,
    build_curves,
    curve_truth,
    is_jumpy,
    real_curve_from_report,
    render_report,
    run_replay,
    write_outputs,
)


def _report(model: str, counts: dict[int, list[float]]) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for count, speeds in counts.items():
        for pass_no in (1, 2):
            runs.append(
                {
                    "layers": count,
                    "pass": pass_no,
                    "ok": True,
                    "writing_speed_runs": list(speeds),
                }
            )
    return {"model": model, "runs": runs}


def _report_with_passes(
    model: str, passes_by_count: dict[int, list[list[float]]]
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for count, passes in passes_by_count.items():
        for pass_no, speeds in enumerate(passes, start=1):
            runs.append(
                {
                    "layers": count,
                    "pass": pass_no,
                    "ok": True,
                    "writing_speed_runs": list(speeds),
                }
            )
    return {"model": model, "runs": runs}


def _study(folder: Path) -> Path:
    folder.mkdir()
    (folder / "a.json").write_text(
        json.dumps(_report("missing-a.gguf", {0: [10.0] * 5, 1: [20.0] * 5, 2: [15.0] * 5, 3: [12.0] * 5, 4: [8.0] * 5})),
        encoding="utf-8",
    )
    (folder / "b.json").write_text(
        json.dumps(_report("missing-b.gguf", {0: [5.0] * 5, 1: [6.0] * 5, 2: [9.0] * 5, 3: [7.0] * 5, 4: [4.0] * 5})),
        encoding="utf-8",
    )
    return folder


def test_real_curve_and_truth(tmp_path: Path) -> None:
    report = _report("missing.gguf", {0: [10.0] * 5, 1: [20.0] * 5})
    counts = real_curve_from_report(report)
    assert set(counts) == {0, 1}
    assert curve_truth(counts, 1) == pytest.approx(20.0)
    assert best_truth(counts) == pytest.approx(20.0)


def test_variants_are_built(tmp_path: Path) -> None:
    report = _report("missing.gguf", {count: [float(count + 10)] * 5 for count in range(5)})
    curves = build_curves("a", report, random.Random(1))
    names = {curve.variant for curve in curves}
    assert {"real", "reversed", "doubled"} <= names
    assert any(name.startswith("hill-") for name in names)

    reversed_curve = next(curve for curve in curves if curve.variant == "reversed")
    # The original best count 4 becomes 0 in the reversed curve.
    assert curve_truth(reversed_curve.counts, 0) == pytest.approx(14.0)
    doubled = next(curve for curve in curves if curve.variant == "doubled")
    assert max(doubled.counts) == 9


def test_quick_keeps_only_real_curves(tmp_path: Path) -> None:
    folder = _study(tmp_path / "study")
    result = run_replay(folder, quick=True)
    assert result.repeats == 50
    assert result.curves
    assert all(curve.variant == "real" for curve in result.curves)


def test_same_seed_gives_the_same_result(tmp_path: Path) -> None:
    folder = _study(tmp_path / "study")
    first = run_replay(folder, repeats=4, seed=7)
    second = run_replay(folder, repeats=4, seed=7)
    assert [(c.label, c.variant, c.smart, c.every) for c in first.curves] == [
        (c.label, c.variant, c.smart, c.every) for c in second.curves
    ]
    assert first.totals == second.totals


def test_report_and_outputs(tmp_path: Path) -> None:
    folder = _study(tmp_path / "study")
    result = run_replay(folder, repeats=3, seed=3)
    text = render_report(result)
    assert "Averages over all curves" in text
    assert "Pass marks" in text
    assert "Layer kinds" in text

    json_path = folder / "replay.json"
    txt_path = folder / "replay.txt"
    write_outputs(result, json_path, txt_path)
    assert json_path.is_file() and txt_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["curves"] and data["passes"]


def test_curve_truth_uses_lowest_pass_when_jumpy() -> None:
    report = _report_with_passes("missing.gguf", {0: [[20.0] * 5, [10.0] * 5]})
    counts = real_curve_from_report(report)
    assert is_jumpy(counts, 0)
    assert curve_truth(counts, 0) == pytest.approx(10.0)


def _stable_curves(top: int) -> dict[int, list[list[float]]]:
    return {count: [[float(count + 10)] * 5, [float(count + 10)] * 5] for count in range(top + 1)}


def test_hills_avoid_jumpy_counts() -> None:
    passes = _stable_curves(12)
    passes[3] = [[20.0] * 5, [10.0] * 5]  # count 3 is jumpy
    report = _report_with_passes("missing.gguf", passes)
    counts = real_curve_from_report(report)
    assert is_jumpy(counts, 3)
    top = max(counts)

    hills = [c for c in build_curves("m", report, random.Random(1)) if c.variant.startswith("hill-")]
    assert hills
    for curve in hills:
        width = int(curve.variant.split("-")[1])
        half = width // 2
        centre = max(curve.counts, key=lambda k: curve_truth(curve.counts, k) or 0.0)
        assert not any(
            is_jumpy(counts, k)
            for k in range(max(0, centre - half), min(top, centre + half) + 1)
        )


def test_no_hill_when_every_centre_is_jumpy() -> None:
    passes = {count: [[20.0] * 5, [10.0] * 5] for count in range(9)}
    report = _report_with_passes("missing.gguf", passes)
    curves = build_curves("m", report, random.Random(1))
    assert not any(curve.variant.startswith("hill-") for curve in curves)
