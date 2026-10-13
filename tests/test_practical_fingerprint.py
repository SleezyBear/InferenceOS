"""The practical parity tool (``scripts/practical_fingerprint.py``): what it sets aside and
what it never sets aside (reviewer, 9 Oct 2026: scratch paths must normalise on Windows
too, while a different model, a different decision or a float change stay visible)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _tool() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "practical_fingerprint.py"
    spec = importlib.util.spec_from_file_location("practical_fingerprint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["practical_fingerprint"] = module
    spec.loader.exec_module(module)
    return module


POSIX = "/tmp/simconf-k2j3h4/models/sample.gguf"
WINDOWS = r"C:\Users\madha\AppData\Local\Temp\simconf-aaa_9\models\sample.gguf"


def _result(model_path: str = POSIX, *, ops: tuple[int, ...] = (27, 0, 1), setup: tuple[int, ...] = (27, 6, 512),
            blocks: int = 11, score: float = 0.0337) -> dict[str, Any]:
    return {
        "run_id": "tune-practical-1791550917", "started_at": "2026-10-09T16:00:00Z", "model_path": model_path,
        "outcome": "saved",
        "saved_setup": {"model": {"path": model_path, "modified_ns": 1}, "setup_id": "setup-abc",
                        "tune_report": "tune-practical-1791550917"},
        "rounds": [
            {"round": "time budget", "operations": [{"kind": "measure", "settings": {"gpu_offload_layers": v}}
                                                    for v in ops]},
            {"round": "final check", "setup": list(setup), "blocks_per_question": {"short": blocks},
             "checks": [{"fair": {"verdict": "better", "score": score}}]},
        ],
    }


def _unexplained(a: dict[str, Any], b: dict[str, Any], numeric: bool = False) -> list[tuple[str, Any, Any]]:
    unexplained, _, _ = _tool().classify(a, b, set(), numeric=numeric)
    return list(unexplained)


def test_posix_and_windows_scratch_roots_normalise_to_the_same_model_relative_path() -> None:
    tool = _tool()
    assert tool.scratch_relative(POSIX) == tool.scratch_relative(WINDOWS) == "<tmp>/models/sample.gguf"
    assert _unexplained(_result(POSIX), _result(WINDOWS)) == []
    assert _unexplained(_result(r"C:\Temp\simorder-x1\models\sample.gguf"), _result("/tmp/simorder-y2/models/sample.gguf")) == []


def test_a_recorded_temp_root_must_match() -> None:
    tool = _tool()
    root = r"C:\Users\madha\AppData\Local\Temp"
    assert tool.scratch_relative(WINDOWS, root) == "<tmp>/models/sample.gguf"
    assert tool.scratch_relative(r"D:\Elsewhere\simconf-aaa_9\models\sample.gguf", root) is None


def test_uuid_run_ids_normalise_only_in_existing_identity_fields() -> None:
    a, b = _result(), _result()
    generated = "tune-practical-1791550917-0123456789abcdef0123456789abcdef"
    b["run_id"] = generated
    b["saved_setup"]["tune_report"] = generated
    assert _unexplained(a, b) == []
    a["rounds"][0]["note"] = a["run_id"]
    b["rounds"][0]["note"] = generated
    assert [p for p, _, _ in _unexplained(a, b)] == ["rounds[time budget].note"]


def test_a_different_model_file_is_still_a_difference() -> None:
    other = POSIX.replace("sample.gguf", "other.gguf")
    assert [p for p, _, _ in _unexplained(_result(POSIX), _result(other))] == ["model_path", "saved_setup.model.path"]
    assert _unexplained(_result("/home/u/models/sample.gguf"), _result("/home/v/models/sample.gguf"))


def test_other_paths_are_never_reduced() -> None:
    a, b = _result(), _result()
    b["rounds"][0]["note"] = "/tmp/simconf-zzz/models/sample.gguf"
    a["rounds"][0]["note"] = "/tmp/simconf-aaa/models/sample.gguf"
    assert [p for p, _, _ in _unexplained(a, b)] == ["rounds[time budget].note"]


@pytest.mark.parametrize("changed", [
    {"ops": (0, 27, 1)},          # operation order
    {"setup": (0, 10, 512)},      # selected setup
    {"blocks": 10},               # confirmation count
])
def test_decision_changes_are_differences(changed: dict[str, Any]) -> None:
    assert _unexplained(_result(), _result(**changed))
    assert _unexplained(_result(), _result(**changed), numeric=True)  # never hidden as numeric


def test_float_differences_fail_strictly_and_are_reported_apart_across_platforms() -> None:
    tool = _tool()
    a, b = _result(score=0.0337), _result(score=0.0337 + 9e-13)
    assert [p for p, _, _ in _unexplained(a, b)] == ["rounds[final check].checks[].fair.score"]
    unexplained, _, floats = tool.classify(a, b, set(), numeric=True)
    assert unexplained == [] and floats == [pytest.approx(9e-13)]


def test_compare_reports_numeric_only_tunes_without_calling_them_identical(tmp_path: Path,
                                                                           capsys: pytest.CaptureFixture[str]) -> None:
    tool = _tool()
    before, after = tmp_path / "a.json", tmp_path / "b.json"
    before.write_text(json.dumps({"t1": _result(POSIX), "t2": _result(POSIX, score=0.1),
                                  "_meta": {"temp_root": "/tmp", "platform": "linux"}}))
    after.write_text(json.dumps({"t1": _result(WINDOWS), "t2": _result(WINDOWS, score=0.1 + 1e-13),
                                 "_meta": {"temp_root": r"C:\Users\madha\AppData\Local\Temp", "platform": "win32"}}))
    assert tool.compare(before, after, set()) == 1
    assert "1 of 2 tunes identical" in capsys.readouterr().out
    assert tool.compare(before, after, set(), cross_platform=True) == 0
    out = capsys.readouterr().out
    assert "1 of 2 tunes identical" in out and "1 of 2 tunes differ ONLY in floating-point values" in out


def test_only_the_added_plan_format_version_can_be_set_aside() -> None:
    tool = _tool()
    a, b = _result(), _result()
    a["rounds"][1]["confirmation_plan"] = {"rule": "r"}
    b["rounds"][1]["confirmation_plan"] = {"rule": "r", "format_version": "confirmation-plan-3"}
    assert tool.classify(a, b, set())[0]
    assert tool.classify(a, b, {"plan-format-version"})[0] == []
    b["rounds"][1]["confirmation_plan"]["rule"] = "changed"
    assert [p for p, _, _ in tool.classify(a, b, {"plan-format-version"})[0]] == [
        "rounds[final check].confirmation_plan.rule"]
