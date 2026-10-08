"""Reviewer question 7: the hardware probe never passes a phase it did not exercise.

Pure tests of ``scripts/check_verify_cost.py``'s classification and exit logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_verify_cost.py"
spec = importlib.util.spec_from_file_location("check_verify_cost", SCRIPT)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def _ok(phase: str, round_: int) -> dict[str, Any]:
    return {"label": f"C-{phase}-{round_}", "phase": phase, "classification": "interrupted",
            "stop_error": None, "stop_call_s": 0.2, "worker_returned": True, "process_gone": True,
            "recovered": True}


def test_all_phases_exercised_and_passed_exits_0() -> None:
    rows = [_ok(p, r) for r in (1, 2, 3) for p in probe.PHASES]
    code, words = probe.interruption_verdict(rows, rounds=3)
    assert code == 0, words


def test_every_case_not_exercised_exits_2() -> None:
    rows = [{"label": f"C-{p}-{r}", "phase": p, "classification": "phase not exercised", "recovered": True}
            for r in (1, 2, 3) for p in probe.PHASES]
    code, words = probe.interruption_verdict(rows, rounds=3)
    assert code == 2
    for phase in probe.PHASES:
        assert f"{phase}: 0 of 3 rounds exercised and passed (3 not exercised)" in words


def test_one_round_not_exercised_fails_that_phase() -> None:
    rows = [_ok(p, r) for r in (1, 2, 3) for p in probe.PHASES]
    rows[4] = {**rows[4], "classification": "phase not exercised"}  # dispatched/pre-first-chunk, round 2
    code, words = probe.interruption_verdict(rows, rounds=3)
    assert code == 2 and words.startswith("dispatched_pre_first_chunk: 2 of 3")


def test_completed_despite_stop_or_survivor_fails() -> None:
    rows = [_ok(p, r) for r in (1, 2, 3) for p in probe.PHASES]
    rows[0] = {**rows[0], "classification": "completed despite the stop"}
    assert probe.interruption_verdict(rows, rounds=3)[0] == 2
    rows = [_ok(p, r) for r in (1, 2, 3) for p in probe.PHASES]
    rows[2] = {**rows[2], "process_gone": False, "blocked": "C-decode-1: still running"}
    assert probe.interruption_verdict(rows, rounds=3)[0] == 2
    rows = [_ok(p, r) for r in (1, 2, 3) for p in probe.PHASES]
    rows[1] = {**rows[1], "recovered": False}
    assert probe.interruption_verdict(rows, rounds=3)[0] == 2


def test_classification_requires_the_phase_at_the_stop() -> None:
    c = probe.classify_case
    # loading: only when the backend was starting AND the stop cancelled that start
    assert c(phase="loading", completed=False, returned_before_stop=False, start_cancelled=True,
             at_stop={"backend_state": "starting"}) == "interrupted"
    assert c(phase="loading", completed=False, returned_before_stop=False, start_cancelled=False,
             at_stop={"backend_state": "ready"}) == "phase not exercised"
    # request dispatched / pre-first-chunk: POST sent, no chunk yet
    assert c(phase="dispatched_pre_first_chunk", completed=False, returned_before_stop=False, start_cancelled=False,
             at_stop={"dispatched": -2.1, "first": None}) == "interrupted"
    assert c(phase="dispatched_pre_first_chunk", completed=False, returned_before_stop=False, start_cancelled=False,
             at_stop={"dispatched": -2.1, "first": -0.1}) == "phase not exercised"
    # decode: tokens streaming, stream not finished
    assert c(phase="decode", completed=False, returned_before_stop=False, start_cancelled=False,
             at_stop={"first": -0.6, "done": None}) == "interrupted"
    assert c(phase="decode", completed=False, returned_before_stop=False, start_cancelled=False,
             at_stop={"first": None, "done": None}) == "phase not exercised"
    # finished before the stop: not exercised; finished after it: a failure
    assert c(phase="decode", completed=True, returned_before_stop=True, start_cancelled=False,
             at_stop={}) == "phase not exercised"
    assert c(phase="decode", completed=True, returned_before_stop=False, start_cancelled=False,
             at_stop={"first": -0.6, "done": None}) == "completed despite the stop"
