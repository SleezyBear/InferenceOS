"""Reviewer follow-up 3, items 2, 3 and 5: the probe's overall verdict, the raw-counter
sampler's cleanup and load evidence from recorded load events.

Runs the real ``scripts/check_verify_cost.py`` ``main()`` with scripted dependencies (a
fake engine, backend, resident target and profiler). Part C is disabled with
``--interrupt-rounds 0`` (Part C has its own tests); that is reported as "disabled",
never as "passed".
"""

from __future__ import annotations

import importlib.util
import io
import itertools
import json
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from inferenceos.contracts.lifecycle import ProcessIdentity

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_verify_cost.py"


def _load_probe() -> Any:
    spec = importlib.util.spec_from_file_location("check_verify_cost_overall", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pids = itertools.count(4000)


class FakeBackend:
    def __init__(self, start_s: float = 0.0, identity: bool = True) -> None:
        self.state = "idle"
        self._identity: ProcessIdentity | None = None
        self._owned = None
        self.last_unload = None
        self.start_s = start_s
        self.with_identity = identity
        self.starts = 0

    def start(self, model: Any, params: Any) -> None:
        if self.state != "idle":
            raise RuntimeError("a resident server is already running")
        time.sleep(self.start_s)
        self.starts += 1
        self._identity = ProcessIdentity(pid=next(_pids), create_time=time.time())
        self.state = "ready"

    def stop(self) -> None:
        self.state, self._identity = "idle", None

    @property
    def identity(self) -> Any:
        if self._identity is None:
            return None
        return self._identity if self.with_identity else SimpleNamespace(pid=self._identity.pid, create_time=None)


class FakeEngine:
    backend_factory: Callable[[], FakeBackend] = FakeBackend

    def __init__(self, settings: Any = None) -> None:
        self._backend = FakeEngine.backend_factory()

    @property
    def pid(self) -> int | None:
        ident = self._backend.identity
        return None if ident is None else ident.pid

    @property
    def identity(self) -> Any:
        return self._backend.identity


class FakeResident:
    cancel_error: BaseException | None = None

    def __init__(self, *, settings: Any, model: Any, adapter: FakeEngine, workload: Any, configuration: Any) -> None:
        self.engine = adapter
        self.cancels = 0

    def cancel(self) -> None:
        self.cancels += 1
        if FakeResident.cancel_error is not None:
            raise FakeResident.cancel_error
        self.engine._backend.stop()

    def allow_restart(self) -> None:
        pass


class FakeProfiler:
    """Answers instantly. ``fail(warmup, profiler)`` decides whether an answer raises;
    ``reuse_load`` makes "fresh" requests reuse the running process while reporting a
    slow ``load_s`` phase (a slow call that is not a load)."""

    fail: Callable[[bool, Any], bool] = staticmethod(lambda warmup, profiler: False)
    reuse_load = False
    #: What a scoped monitor's close reports (None: not recorded). False behaves as the
    #: real profiler does: the profiler is blocked and every later run_once raises.
    close_resolved: bool | None = None
    instances: list[FakeProfiler] = []

    def __init__(self, *, settings: Any = None, adapter: Any = None) -> None:
        self.measurement_monitor = False
        self.monitor_events: deque[dict[str, Any]] = deque(maxlen=500)
        self._fresh = True
        self._scope_open = False
        self.engine: FakeEngine | None = None
        self.monitor_blocked: str | None = None
        self.started_after_block = 0
        FakeProfiler.instances.append(self)

    def request_fresh_start(self) -> None:
        self._fresh = True

    def end_measurement(self) -> None:
        if self._scope_open:
            resolved = FakeProfiler.close_resolved
            self.monitor_events.append({"event": "close", "s": 0.01, "answers": 1, "resolved": resolved,
                                        "errors": [] if resolved is not False else ["kill: TimeoutExpired"]})
            self._scope_open = False
            if resolved is False:
                self.monitor_blocked = "a counter sampler's cleanup was not confirmed (kill: TimeoutExpired)"

    def run_once(self, task: Any, *, mode: Any, sequence: int, repetition: int, warmup: bool, options: Any) -> Any:
        assert self.engine is not None
        if self.monitor_blocked is not None:
            from inferenceos.telemetry.counters import MonitorCleanupUnresolved

            raise MonitorCleanupUnresolved(f"no further measurement: {self.monitor_blocked}")
        backend = self.engine._backend
        load_s = 0.0
        if self._fresh and not FakeProfiler.reuse_load:
            if backend.state != "idle":
                backend.stop()
            began = time.perf_counter()
            backend.start(task.model, None)  # through the instance attribute (the probe's load log)
            load_s = time.perf_counter() - began
        elif self._fresh and FakeProfiler.reuse_load:
            if backend.state == "idle":
                backend.start(task.model, None)
            load_s = 9.0  # slow, but no new load happened
        self._fresh = False
        if FakeProfiler.fail(warmup, self):
            raise RuntimeError("answer failed (scripted)")
        if options.telemetry_sampling and self.measurement_monitor and not self._scope_open:
            self.monitor_events.append({"event": "open", "s": 0.02})
            self._scope_open = True
        metrics = SimpleNamespace(ttft_s=0.2, decode_tokens_per_sec=20.0, prompt_tokens=50, hard_page_faults_per_sec=0.0)
        return SimpleNamespace(observation=SimpleNamespace(metrics=metrics), phase_timings_s={"load_s": load_s})


def _good_capture(paths: Any, seconds: float, during: Any = None, marks: Any = None) -> dict[str, Any]:
    now = time.perf_counter()
    return {"lines": [{"perf": now, "wall": time.time(), "line": "a|b|c"}], "during_error": None,
            "exit_confirmed": True, "reader_joined": True, "cleanup_errors": []}


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    import inferenceos.backends.llama_cpp_resident_adapter as lra
    import inferenceos.models.loader as loader
    import inferenceos.profiling.runner as runner
    import inferenceos.runtime.gpu_search as gpu_search
    import inferenceos.runtime.resident_workload as rw
    import inferenceos.tune.engine_probe as engine_probe
    from tests.resident_helpers import synthetic_model

    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"\x00")
    monkeypatch.setattr(loader, "load_model_profile", lambda p: synthetic_model(str(model_file)))
    monkeypatch.setattr(lra, "LlamaCppResidentAdapter", FakeEngine)
    monkeypatch.setattr(rw, "ResidentActiveWorkload", FakeResident)
    monkeypatch.setattr(runner, "BaselineProfiler", FakeProfiler)
    gone = {"value": True}
    monkeypatch.setattr(engine_probe, "wait_until_gone", lambda identity, timeout_s=20.0: gone["value"])

    def adapter(resident: FakeResident, explicit_defaults: bool = True) -> Any:
        return resident  # the fake profiler reaches the engine through it

    monkeypatch.setattr(gpu_search, "ResidentMeasurementAdapter", adapter)
    original_init = FakeProfiler.__init__

    def init(self: FakeProfiler, *, settings: Any = None, adapter: Any = None) -> None:
        original_init(self, settings=settings, adapter=adapter)
        self.engine = adapter.engine

    monkeypatch.setattr(FakeProfiler, "__init__", init)
    monkeypatch.setattr(FakeProfiler, "fail", staticmethod(lambda warmup, profiler: False))
    monkeypatch.setattr(FakeProfiler, "reuse_load", False)
    monkeypatch.setattr(FakeProfiler, "close_resolved", None)
    monkeypatch.setattr(FakeResident, "cancel_error", None)
    monkeypatch.setattr(FakeEngine, "backend_factory", FakeBackend)
    probe = _load_probe()
    monkeypatch.setattr(probe, "capture_raw_counters", _good_capture)

    def run(*extra: str) -> tuple[int, dict[str, Any]]:
        out = tmp_path / "probe.json"
        code = probe.main(["--model", str(model_file), "--layers", "1", "--interrupt-rounds", "0",
                           "--rounds", "1", "--short-runs", "2", "--long-runs", "1", "--out", str(out), *extra])
        data = json.loads(out.read_text())
        print(f"\nexit {code} | status: {data['status']}")
        print("parts:", json.dumps(data["parts"]))
        return code, data

    return SimpleNamespace(run=run, probe=probe, gone=gone, monkeypatch=monkeypatch)


def test_everything_passing_exits_0_and_part_c_is_disabled_not_passed(world: Any) -> None:
    code, data = world.run()
    assert code == 0 and data["status"] == "done"
    assert data["parts"] == {"C": "disabled", "A0": "passed", "A": "passed", "B": "passed", "cleanup": "passed"}
    shared = [m for m in data["measurements"] if m["label"] == "B-shared"]
    assert shared and all(m["sharing"]["verdict"] == "shown" and m["loads"] == 1 for m in shared)


def test_every_part_a_measurement_failing_is_not_success(world: Any) -> None:
    # Every scored answer raises (warm-ups pass), so every Part A measurement fails.
    world.monkeypatch.setattr(FakeProfiler, "fail", staticmethod(lambda warmup, profiler: not warmup))
    code, data = world.run("--skip-part-b")
    assert code == 2 and data["status"].startswith("failed:")
    assert data["parts"]["A"].startswith("failed:") and "RuntimeError" in data["parts"]["A"]
    assert data["parts"]["B"] == "disabled"  # intentionally not requested: distinct from failed
    assert len(data["measurements"]) == 3 and all(m["error"] for m in data["measurements"])  # kept


def test_empty_required_counter_capture_is_not_success(world: Any) -> None:
    world.monkeypatch.setattr(world.probe, "capture_raw_counters", lambda *a, **k: {
        "lines": [], "during_error": None, "exit_confirmed": True, "reader_joined": True, "cleanup_errors": []})
    code, data = world.run()
    assert code == 2
    assert data["parts"]["A0"] == "failed: no counter line was captured"
    assert data["parts"]["A"] == "passed" and data["parts"]["B"] == "passed"  # partial results kept


def test_failed_final_cleanup_is_not_success(world: Any) -> None:
    world.monkeypatch.setattr(FakeResident, "cancel_error", RuntimeError("stop failed (scripted)"))
    code, data = world.run()
    assert code == 2 and data["status"].startswith("failed:")
    assert data["parts"]["cleanup"].startswith("failed: the final stop failed (RuntimeError: stop failed")
    assert all(data["parts"][p] == "passed" for p in ("A0", "A", "B"))


def test_engine_process_not_gone_after_cleanup_is_not_success(world: Any) -> None:
    world.gone["value"] = False
    code, data = world.run()
    assert code == 2 and "was not confirmed gone" in data["parts"]["cleanup"]


def test_sampler_whose_exit_is_unconfirmed_stops_later_measurements(world: Any) -> None:
    world.monkeypatch.setattr(world.probe, "capture_raw_counters", lambda *a, **k: {
        "lines": [{"perf": 0.0, "wall": 0.0, "line": "x"}], "during_error": None, "exit_confirmed": False,
        "reader_joined": False, "cleanup_errors": ["kill: TimeoutExpired"]})
    code, data = world.run()
    assert code == 2
    assert "exit was not confirmed" in data["parts"]["A0"]
    assert data["parts"]["A"] == "not run: the counter sampler's exit was not confirmed"
    assert data["parts"]["B"] == "not run: the counter sampler's exit was not confirmed"
    assert data["measurements"] == []


def test_measurement_problems_flags_a_failed_close() -> None:
    """Reviewer, monitor round 2, item 4: a close event with resolved False used to
    give an empty problems list."""
    probe = _load_probe()
    row: dict[str, Any] = {"label": "A-short", "monitor": "measurement", "error": None, "runs": [{}, {}, {}],
           "monitor_events": [{"event": "open", "s": 1.8},
                              {"event": "close", "s": 10.0, "resolved": False, "errors": ["kill: TimeoutExpired"]}]}
    problems = probe.measurement_problems(row, expected_runs=3, require_sharing=False)
    print("\nproblems:", problems)
    assert problems == ["A-short: sampler cleanup unresolved (kill: TimeoutExpired)"]
    row["monitor_events"][1]["resolved"] = True
    assert probe.measurement_problems(row, expected_runs=3, require_sharing=False) == []


def test_unresolved_scoped_cleanup_stops_the_probe_and_fails_it(world: Any) -> None:
    """Part A's measurement-scoped measurement closes with an unresolved cleanup: it
    fails, no further measurement starts (Part A's rest and Part B are "not run"), and
    the exit status is 2."""
    world.monkeypatch.setattr(FakeProfiler, "close_resolved", False)
    code, data = world.run()
    rows = [(m["label"], m["monitor"], m["error"]) for m in data["measurements"]]
    print("measurements:", rows)
    assert code == 2 and data["status"].startswith("failed:")
    # Round 1 order is answer, measurement, off: the scoped one is second and is the last to run.
    assert [m["monitor"] for m in data["measurements"]] == ["answer", "measurement"]
    assert "sampler cleanup unresolved (kill: TimeoutExpired)" in data["parts"]["A"]
    assert "1 measurement(s) not run: sampler cleanup unresolved" in data["parts"]["A"]
    assert data["parts"]["B"].startswith("not run: sampler cleanup unresolved")
    assert data["parts"]["cleanup"] == "passed"  # the final stop still runs (it starts nothing)
    profiler = FakeProfiler.instances[-1]
    assert profiler.monitor_blocked is not None


def test_a_blocked_profiler_mid_measurement_is_recorded_not_retried(world: Any) -> None:
    """The block is raised by run_once (as the real profiler does) inside a measurement:
    the probe records the error -- it is not swallowed as a generic failure and retried --
    and nothing further runs."""
    def fail(warmup: bool, profiler: Any) -> bool:
        if profiler.measurement_monitor and not warmup and profiler.monitor_blocked is None:
            profiler.monitor_blocked = "a counter sampler's cleanup was not confirmed (scripted)"
        return False

    world.monkeypatch.setattr(FakeProfiler, "fail", staticmethod(fail))
    code, data = world.run("--skip-part-b")
    scoped = [m for m in data["measurements"] if m["monitor"] == "measurement"]
    print("scoped error:", scoped[0]["error"])
    assert code == 2 and len(data["measurements"]) == 2
    assert scoped[0]["error"].startswith("MonitorCleanupUnresolved: no further measurement")
    assert "1 measurement(s) not run" in data["parts"]["A"]


# --------------------------------------------------------------------------- item 3


class FakeProc:
    def __init__(self, *, exits: bool = True) -> None:
        self.stdout = io.StringIO("line one\nline two\n")
        self.calls: list[str] = []
        self.exits = exits

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append("wait")
        if not self.exits:
            raise subprocess.TimeoutExpired("powershell", timeout or 0)
        return 0


def test_sampler_is_cleaned_up_when_an_answer_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe()
    proc = FakeProc()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    readers: list[threading.Thread] = []
    real_thread = threading.Thread

    def thread(*a: Any, **k: Any) -> threading.Thread:
        made = real_thread(*a, **k)
        readers.append(made)
        return made

    monkeypatch.setattr(probe.threading, "Thread", thread)

    def during() -> None:
        time.sleep(0.05)  # the reader has the lines by now
        raise RuntimeError("answer failed (injected)")

    record = probe.capture_raw_counters(("\\Memory\\Pages Input/sec",), 0.3, during=during, marks={})
    print("\nrecord:", {k: v for k, v in record.items() if k != "lines"}, "| lines:", [ln["line"] for ln in record["lines"]],
          "| process calls:", proc.calls, "| pipe closed:", proc.stdout.closed)
    assert record["during_error"] == "RuntimeError: answer failed (injected)"
    assert proc.calls[:2] == ["terminate", "wait"] and record["exit_confirmed"] is True
    assert proc.stdout.closed and record["reader_joined"] is True and not readers[0].is_alive()
    assert [ln["line"] for ln in record["lines"]] == ["line one", "line two"]  # kept


def test_sampler_that_will_not_exit_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe()
    proc = FakeProc(exits=False)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    record = probe.capture_raw_counters(("x",), 0.2)
    print("\nprocess calls:", proc.calls, "| exit_confirmed:", record["exit_confirmed"], "|", record["cleanup_errors"])
    assert proc.calls == ["terminate", "wait", "kill", "wait"]
    assert record["exit_confirmed"] is False and record["cleanup_errors"]
    assert "the counter sampler's exit was not confirmed" in probe.counter_problems(record)


def test_sampler_cleanup_runs_even_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = _load_probe()
    proc = FakeProc()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)

    def during() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        probe.capture_raw_counters(("x",), 0.2, during=during)
    assert proc.calls[:2] == ["terminate", "wait"] and proc.stdout.closed


# --------------------------------------------------------------------------- item 5


def test_a_fast_genuine_load_is_counted() -> None:
    probe = _load_probe()
    backend = FakeBackend(start_s=0.0)
    log = probe.LoadLog()
    detach = log.attach(backend)
    began = time.perf_counter()
    backend.start(None, None)
    took = time.perf_counter() - began
    detach()
    print(f"\nload took {took * 1000:.2f} ms; events: {log.events}")
    assert took < 0.5  # under the old 0.5 s threshold, which would have missed it
    assert len(log.events) == 1 and log.events[0]["load_id"] == 1
    assert log.events[0]["pid"] == backend.identity.pid and log.events[0]["create_time"] == backend.identity.create_time
    assert "start" not in vars(backend)  # detached


def test_a_failed_start_is_not_a_load() -> None:
    probe = _load_probe()
    backend = FakeBackend()
    log = probe.LoadLog()
    log.attach(backend)
    backend.start(None, None)
    with pytest.raises(RuntimeError):
        backend.start(None, None)  # already running: refused
    assert [e["load_id"] for e in log.events] == [1]


def test_a_slow_call_that_reuses_the_load_is_not_counted(world: Any) -> None:
    # "Fresh" requests reuse the running process but report a 9 s load phase: the old
    # rule (load_s > 0.5) would have counted a load for every measurement.
    world.monkeypatch.setattr(FakeProfiler, "reuse_load", True)
    code, data = world.run()
    rows = {m["label"]: m for m in data["measurements"]}
    for label in ("B-separate-short", "B-separate-long", "B-shared"):
        row = rows[label]
        print(label, "| loads:", row["loads"], "| load phase s:", [r["phases"]["load_s"] for r in row["runs"]][:1],
              "| sharing:", row["sharing"])
        assert row["loads"] == 0 and row["sharing"]["verdict"] == "not shown"
        assert all(not r["loaded_now"] for r in row["runs"])
    assert code == 2 and data["parts"]["B"].startswith("failed:")


def test_sharing_is_unverified_without_process_creation_time(world: Any) -> None:
    world.monkeypatch.setattr(FakeEngine, "backend_factory", lambda: FakeBackend(identity=False))
    code, data = world.run()
    shared = [m for m in data["measurements"] if m["label"] == "B-shared"]
    print("\nB-shared:", [m["sharing"] for m in shared])
    assert shared and all(m["sharing"]["verdict"] == "unverified" for m in shared)
    assert code == 2 and "unverified" in data["parts"]["B"]


def test_load_evidence_rules() -> None:
    probe = _load_probe()
    one = [{"load_id": 3, "pid": 10, "create_time": 1.5}]
    runs = [{"pid": 10, "create_time": 1.5}, {"pid": 10, "create_time": 1.5}]
    assert probe.load_evidence(one, runs)["verdict"] == "shown"
    assert probe.load_evidence(one, [*runs, {"pid": 10, "create_time": 2.5}])["verdict"] == "not shown"  # pid reused
    assert probe.load_evidence(one * 2, runs)["verdict"] == "not shown"
    assert probe.load_evidence([], runs)["verdict"] == "not shown"
    assert probe.load_evidence(one, [{"pid": 10, "create_time": None}])["verdict"] == "unverified"


def test_overall_verdict_keeps_disabled_apart_from_failed() -> None:
    probe = _load_probe()
    assert probe.overall_verdict({"C": "passed", "B": "disabled", "cleanup": "passed"}) == (
        0, "every requested part passed (B disabled)")
    code, words = probe.overall_verdict({"C": "passed", "A": "not run: Part C did not pass", "cleanup": "passed"})
    assert code == 2 and words == "A not run: Part C did not pass"


# --------------------------------------------------------------------------- follow-up 4, item 1


def test_sampler_cleanup_returns_while_a_real_pipe_writer_stays_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real OS pipe whose write end stays open (as if a child of the sampler still held
    it) and a sampler that will not exit: the reader stays blocked in read(). Cleanup must
    still return within its bounds, keep the lines, and report itself unresolved. The
    old cleanup closed the pipe from this thread and blocked until the writer let go."""
    import os

    probe = _load_probe()
    monkeypatch.setattr(probe, "COUNTER_EXIT_S", 0.2)
    monkeypatch.setattr(probe, "READER_JOIN_S", 0.5)
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"line one\nline two\n")
    stream = os.fdopen(read_fd, "r")

    class StuckProc:
        stdout = stream

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired("powershell", timeout or 0)

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: StuckProc())
    readers: list[threading.Thread] = []
    real_thread = threading.Thread

    def thread(*a: Any, **k: Any) -> threading.Thread:
        made = real_thread(*a, **k)
        readers.append(made)
        return made

    monkeypatch.setattr(probe.threading, "Thread", thread)
    box: dict[str, Any] = {}
    caller = real_thread(target=lambda: box.setdefault("record", probe.capture_raw_counters(("x",), 0.2)), daemon=True)
    began = time.perf_counter()
    try:
        caller.start()
        caller.join(10.0)
        took = time.perf_counter() - began
        returned = not caller.is_alive()
        reader_alive = readers[0].is_alive()
    finally:
        os.close(write_fd)  # release the writer: the reader sees end-of-file and ends
    caller.join(5.0)
    readers[0].join(5.0)
    stream.close()
    record = box.get("record", {})
    print(f"\ncleanup returned: {returned} after {took:.2f} s (writer still open); reader blocked then: {reader_alive}")
    print("record:", {k: v for k, v in record.items() if k != "lines"}, "| lines:", [ln["line"] for ln in record.get("lines", [])])
    assert returned and took < 3.0
    assert reader_alive  # the reader really was blocked on the open pipe
    assert record["exit_confirmed"] is False and record["reader_joined"] is False
    assert record["pipe_closed"] is False and record["cleanup_resolved"] is False
    assert [ln["line"] for ln in record["lines"]] == ["line one", "line two"]
    assert "reader: still blocked on the pipe" in " ".join(record["cleanup_errors"])
    problems = probe.counter_problems(record)
    assert "the counter reader was still blocked on the pipe" in problems
    assert not readers[0].is_alive() and not caller.is_alive()  # nothing left running


def test_unresolved_reader_cleanup_stops_later_measurements(world: Any) -> None:
    # The sampler exited but its reader is still blocked: Parts A and B must not run.
    world.monkeypatch.setattr(world.probe, "capture_raw_counters", lambda *a, **k: {
        "lines": [{"perf": 0.0, "wall": 0.0, "line": "x"}], "during_error": None, "exit_confirmed": True,
        "reader_joined": False, "pipe_closed": False, "cleanup_resolved": False,
        "cleanup_errors": ["reader: still blocked on the pipe after 5 s (pipe left open)"]})
    code, data = world.run()
    assert code == 2 and data["status"].startswith("failed:")
    assert "the counter reader was still blocked on the pipe" in data["parts"]["A0"]
    assert data["parts"]["A"] == "not run: the counter sampler's cleanup was unresolved"
    assert data["parts"]["B"] == "not run: the counter sampler's cleanup was unresolved"
    assert data["measurements"] == [] and data["counter_raw"]["lines"]  # partial result written
