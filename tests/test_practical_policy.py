"""C3-7 dispatch contracts plus real-tune failure/report integration."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.tune import coordinator as co
from inferenceos.tune.practical_policy import PracticalPolicy
from inferenceos.tune.planner import Step, THREADS, BATCH
from inferenceos.tune.budgeted import Cancelled
from inferenceos.telemetry.counters import MonitorCleanupUnresolved
from tests.test_budgeted_tune import _practical, _world
from scripts.practical_fingerprint import classify


class Driver:
    def __init__(self, steps: list[Any], *, repeats: list[str] | None = None, reopen: bool = False) -> None:
        self.calls: list[Any] = []
        self.items = list(steps)
        self.repeats = list(repeats or ["none"])
        self.setup = (33, 6, 1024)
        self.stuck: list[str] = []
        self.steps: list[dict[str, Any]] = []
        self.screenings: list[dict[str, Any]] = []
        self._batch_left_out = reopen
        self._batch_retried = False
        self._long: Any = SimpleNamespace(workload_id="long-test")
        self.waiting_for_length = False
        self.log: list[str] = []
        self.why = ""
        self.error: BaseException | None = None

    def current_setup(self) -> tuple[int, int, int]:
        return self.setup

    def _config(self, setup: tuple[int, int, int]) -> RuntimeConfiguration:
        return RuntimeConfiguration(gpu_offload_layers=setup[0], threads=setup[1], ubatch_size=setup[2])

    def _now(self) -> float:
        return 0.0

    def check_user(self) -> None:
        self.calls.append("check_user")

    def next_step(self) -> Any:
        self.calls.append("next_step")
        value = self.items.pop(0) if self.items else None
        if value == "size":
            self.waiting_for_length = True
            return None
        return value

    def _size_long(self, planner: Any, *, reopen: bool = False) -> bool:
        self.calls.append("reopen" if reopen else "size")
        self.waiting_for_length = False
        return not reopen

    def _repeat_weak(self, planner: Any) -> str:
        self.calls.append("weak")
        return self.repeats.pop(0) if self.repeats else "none"

    def _screen(self, planner: Any, step: Step, value: int) -> str:
        self.calls.append(("screen", step.base, step.setup_for(value)))
        if self.error is not None:
            raise self.error
        self.setup = step.setup_for(value)
        return self.why


def drive(driver: Driver) -> tuple[PracticalPolicy, co.SearchTrace]:
    policy = PracticalPolicy(driver, driver)  # type: ignore[arg-type]
    trace = co.coordinate(policy, co.SearchTrace(co.PRACTICAL))
    return policy, trace


def test_captured_step_base_survives_a_first_candidate_choice_change() -> None:
    step = Step(THREADS, (10, 9), "threads", (33, 8, 1024))
    driver = Driver([step])
    _, trace = drive(driver)
    screens = [c for c in driver.calls if isinstance(c, tuple)]
    assert screens == [("screen", (33, 8, 1024), (33, 10, 1024)),
                       ("screen", (33, 8, 1024), (33, 9, 1024))]
    assert driver.calls[:4] == ["check_user", "next_step", *screens]
    parent, first, second = trace.entries[:3]
    assert parent["reference"]["threads"] == first["reference"]["threads"] == second["reference"]["threads"] == 8
    assert first["carried"]["threads"] == 6 and second["carried"]["threads"] == 10
    assert parent["status"] == co.CHANGED


def test_temporary_exhaustion_sizes_then_repeats_then_reopens_only_once() -> None:
    driver = Driver(["size", None, None], repeats=["repeated", "none"], reopen=True)
    policy, trace = drive(driver)
    assert driver.calls == ["check_user", "next_step", "size", "check_user", "next_step", "weak",
                            "check_user", "next_step", "weak", "reopen"]
    assert [e["kind"] for e in trace.entries] == [co.SIZE_LONG, co.WEAK_REPEAT, co.WEAK_REPEAT, co.REOPEN_BATCH]
    assert driver._batch_retried and policy.stop_reason == ""


def test_successful_reopening_resumes_at_the_old_loop_boundary() -> None:
    driver = Driver([None, Step(BATCH, (2048,), "batch", (33, 8, 512))], reopen=True)
    original = driver._size_long
    def size(planner: Any, *, reopen: bool = False) -> bool:
        original(planner, reopen=reopen)
        return True
    driver._size_long = size  # type: ignore[method-assign]
    _, trace = drive(driver)
    assert driver.calls[:5] == ["check_user", "next_step", "weak", "reopen", "check_user"]
    assert driver.calls.count("reopen") == 1
    assert any(e["path"] == "prompt_batch" and e["kind"] == co.SCREEN for e in trace.entries)


def test_missing_long_workload_skips_without_screening_or_false_completion() -> None:
    driver = Driver([Step(BATCH, (2048,), "batch"), Step(THREADS, (8,), "threads")])
    driver._long = None
    _, trace = drive(driver)
    assert trace.entries[0]["status"] == co.NOT_APPLICABLE
    assert len(driver.steps) == 1 and driver.steps[0]["category"] == THREADS


def test_screen_stop_preserves_reason_and_does_not_dispatch_rest_of_step() -> None:
    driver = Driver([Step(THREADS, (8, 9), "threads")])
    driver.why = "target: final check would not fit"
    policy, trace = drive(driver)
    assert policy.stop_reason == driver.why == trace.finished
    assert len([c for c in driver.calls if isinstance(c, tuple)]) == 1
    assert "weak" not in driver.calls


def test_stuck_operation_stops_before_asking_for_a_step() -> None:
    driver = Driver([])
    driver.stuck.append("answer")
    policy, trace = drive(driver)
    assert driver.calls == ["check_user"] and trace.entries == []
    assert policy.stop_reason == "an operation appeared stuck: answer"


@pytest.mark.parametrize("error", [Cancelled("stop"), MonitorCleanupUnresolved("sampler alive"), RuntimeError("failed")])
def test_failed_candidate_is_not_advanced_or_retried(error: BaseException) -> None:
    driver = Driver([Step(THREADS, (8, 9), "threads")])
    driver.error = error
    policy = PracticalPolicy(driver, driver)  # type: ignore[arg-type]
    trace = co.SearchTrace(co.PRACTICAL)
    with pytest.raises(type(error)) as caught:
        co.coordinate(policy, trace)
    assert caught.value is error
    assert [e["status"] for e in trace.entries] == [co.INTERRUPTED, co.INTERRUPTED]
    assert len([c for c in driver.calls if isinstance(c, tuple)]) == 1


@pytest.mark.parametrize("error", [Cancelled("stop"), MonitorCleanupUnresolved("sampler alive")])
def test_real_practical_failure_result_serializes_partial_trace(tmp_path: Path, error: BaseException) -> None:
    tuner, store, tune, _ = _practical(tmp_path, _world(None), 900)
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error
    tune._screen = fail  # type: ignore[method-assign]
    result = tune.run()
    traces = [r for r in result.rounds if r.get("round") == co.TRACE_ROUND]
    assert len(traces) == 1 and result.saved_setup is None
    assert traces[0]["termination"]["stage"] == "execute"  # type: ignore[index]
    assert "interrupted" in result.to_json()
    assert result.outcome.startswith("cancelled" if isinstance(error, Cancelled) else "failed")


def test_practical_parity_allowance_is_only_the_new_trace_round() -> None:
    before = {"rounds": [], "outcome": "saved"}
    after = {"rounds": [{"round": co.TRACE_ROUND, "mode": co.PRACTICAL}], "outcome": "saved"}
    assert not classify(before, after, {"coordinator-trace"})[0]
    after["outcome"] = "no change"
    assert classify(before, after, {"coordinator-trace"})[0]
    assert classify(before, {"rounds": [{"round": "final check"}], "outcome": "saved"}, {"coordinator-trace"})[0]


@pytest.mark.parametrize("names", [
    ["time budget", "final check", "practical search"],
    ["final check", "practical search", "time budget"],
    ["practical search", "practical search", "final check", "time budget"],
    ["practical search", "time budget"],
])
def test_practical_parity_rejects_round_order_and_multiplicity_changes(names: list[str]) -> None:
    before = {"rounds": [{"round": n} for n in ["practical search", "final check", "time budget"]]}
    after = {"rounds": [{"round": n} for n in names]}
    assert classify(before, after, {"coordinator-trace"})[0]


def test_allowed_trace_insertion_does_not_allow_other_round_reordering() -> None:
    a = {"rounds": [{"round": "layer top"}, {"round": "final check"}]}
    trace = {"round": co.TRACE_ROUND, "mode": co.PRACTICAL}
    b = {"rounds": [a["rounds"][0], trace, a["rounds"][1]]}
    assert not classify(a, b, {"coordinator-trace"})[0]
    b["rounds"] = [a["rounds"][1], trace, a["rounds"][0]]
    assert classify(a, b, {"coordinator-trace"})[0]


@pytest.mark.parametrize("cause", [Cancelled("user stop"), MonitorCleanupUnresolved("sampler alive"),
                                  RuntimeError("original failure"), KeyboardInterrupt()])
def test_export_failure_never_masks_active_exception(cause: BaseException, monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.tune.budgeted import BudgetedTune
    driver = Driver([Step(THREADS, (8, 9), "threads")])
    driver.error = cause
    driver.tuner = SimpleNamespace(_search_trace=None)  # type: ignore[attr-defined]
    driver._rounds = []  # type: ignore[attr-defined]
    def broken(*args: Any) -> Any:
        raise RuntimeError("broken trace export")
    monkeypatch.setattr(co.SearchTrace, "round", broken)
    with pytest.raises(type(cause)) as caught:
        BudgetedTune._explore(driver, driver)  # type: ignore[arg-type]
    assert caught.value is cause
    assert driver._rounds[0]["actions_unavailable"] is True  # type: ignore[attr-defined]


def test_real_practical_monitor_failure_survives_broken_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, tune, _ = _practical(tmp_path, _world(None), 900)
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise MonitorCleanupUnresolved("original sampler alive")
    def broken(*args: Any) -> Any:
        raise RuntimeError("broken trace export")
    monkeypatch.setattr(tune, "_screen", fail)
    monkeypatch.setattr(co.SearchTrace, "round", broken)
    result = tune.run()
    assert result.outcome == "failed: original sampler alive; nothing saved"
    assert result.saved_setup is None
    traces = [r for r in result.rounds if r.get("round") == co.TRACE_ROUND]
    assert len(traces) == 1 and traces[0]["trace_export_error"] == "RuntimeError"


def test_successful_search_export_error_stops_before_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, tune, _ = _practical(tmp_path, _world(None), 900)
    cause = RuntimeError("broken trace export")
    def broken(*args: Any) -> Any:
        raise cause
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("confirmation must not follow a failed successful-search export")
    monkeypatch.setattr(co.SearchTrace, "round", broken)
    monkeypatch.setattr(tune, "_complete_readings", forbidden)
    with pytest.raises(RuntimeError) as caught:
        tune.run()
    assert caught.value is cause
