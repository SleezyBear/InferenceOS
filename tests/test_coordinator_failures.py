"""C3-7: diagnostics retain the original failure and never drive execution."""
from pathlib import Path
from typing import Any
import json

import pytest

from inferenceos.tune import coordinator as co
from inferenceos.telemetry.counters import MonitorCleanupUnresolved
from tests.test_coordinator import ScriptedPolicy, ScriptedTuner, _policy, _run_tune, _kept


@pytest.mark.parametrize("stage", ["next_action", "advance"])
def test_transition_failure_is_terminal_without_changing_completed_actions(stage: str) -> None:
    error = RuntimeError("transition broke")
    policy = ScriptedPolicy([co.Outcome(co.CHANGED, "screening choice")])
    def fail(*args: Any) -> Any:
        raise error
    setattr(policy, stage, fail)
    trace = co.SearchTrace(co.UNBOUNDED)
    with pytest.raises(RuntimeError) as caught:
        co.coordinate(policy, trace)
    assert caught.value is error
    assert trace.termination == {"stage": stage, "error_type": "RuntimeError",
                                 "message": "transition broke", "action_id": 1 if stage == "advance" else None}
    assert trace.finished is None
    assert [e["status"] for e in trace.entries] == ([co.CHANGED] if stage == "advance" else [])
    assert policy.log.count("next") <= 1


def test_nested_revisit_failure_keeps_completed_parent_and_stops_queue() -> None:
    error = KeyboardInterrupt()
    def layer(*args: Any) -> Any:
        raise error
    tuner = ScriptedTuner("threads,kv_cache", path_round=_kept("kv_cache_type_k", "q8_0"), layer_round=layer)
    policy, _, _ = _policy(tuner)
    trace = co.SearchTrace(policy.mode)
    policy.start(trace)
    with pytest.raises(KeyboardInterrupt) as caught:
        co.coordinate(policy, trace)
    assert caught.value is error
    assert [e["status"] for e in trace.entries] == [co.NO_CHANGE, co.CHANGED, co.INTERRUPTED]
    assert trace.termination and trace.termination["stage"] == "advance"
    assert [c[1] for c in tuner.calls if c[0] == "path"] == ["threads", "kv_cache"]


def test_broken_exception_string_does_not_mask_cause() -> None:
    class Broken(RuntimeError):
        def __str__(self) -> str:
            raise ValueError("bad diagnostic")
    error = Broken()
    with pytest.raises(Broken) as caught:
        co.coordinate(ScriptedPolicy([error]), co.SearchTrace(co.UNBOUNDED))
    assert caught.value is error


def test_partial_trace_is_in_the_saved_monitor_failure_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tuner, store, calls, out = _run_tune(tmp_path, monkeypatch, MonitorCleanupUnresolved("sampler alive"))
    assert tuner._run_tune() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert calls == ["stop"] and report["saved_setup"] is None
    traces = [r for r in report["rounds"] if r["round"] == co.TRACE_ROUND]
    assert len(traces) == 1
    assert traces[0]["termination"]["stage"] == "execute"
    assert traces[0]["actions"][-1]["status"] == co.INTERRUPTED


def test_reused_tuner_clears_trace_before_an_early_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tuner, _, _, _ = _run_tune(tmp_path, monkeypatch, MonitorCleanupUnresolved("sampler alive"))
    tuner._run_tune()
    assert tuner._search_trace is not None
    error = RuntimeError("before exploration")
    def fail() -> Any:
        raise error
    tuner._run_search = fail
    with pytest.raises(RuntimeError) as caught:
        tuner.run()
    assert caught.value is error and tuner._search_trace is None


def test_command_monitor_report_survives_trace_export_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tuner, _, calls, out = _run_tune(tmp_path, monkeypatch, MonitorCleanupUnresolved("sampler alive"))
    def broken(*args: Any) -> Any:
        raise RuntimeError("broken trace export")
    monkeypatch.setattr(co.SearchTrace, "round", broken)
    assert tuner._run_tune() == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert calls == ["stop"] and report["saved_setup"] is None
    assert report["error"]["type"] == "MonitorCleanupUnresolved"
    assert report["rounds"][0]["trace_export_error"] == "RuntimeError"
