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
        return SimpleNamespace(observation=SimpleNamespace(metrics=metrics), phase_timings_s={"load_s": load_s},
                               identity=SimpleNamespace(run_id=f"run-{sequence:04d}"))


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
        print("parts:", json.dumps(data.get("parts")))
        return code, data

    return SimpleNamespace(run=run, probe=probe, gone=gone, monkeypatch=monkeypatch)


def test_everything_passing_exits_0_and_part_c_is_disabled_not_passed(world: Any) -> None:
    code, data = world.run()
    assert code == 0 and data["status"] == "done"
    assert data["parts"] == {"C": "disabled", "A0": "passed", "A": "passed", "R": "disabled", "B": "passed",
                             "cleanup": "passed"}  # R: the reference sampler was not requested
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


# ------------------------- reviewer, after the boundary-bounds patch: consistency check
def _paging_reading(t: float, raw: int) -> Any:
    from inferenceos.telemetry.attribution import PAGING_SPEC, RawReading

    return RawReading(spec=PAGING_SPEC, read_before=t, read_after=t + 0.002, raw=raw, base=0,
                      ts100=int(round(t * 1e7)), counter_type=272696320, calculated=None,
                      arrival_wall=t + 0.004, arrival_perf=0.0)


def _sampler(events: list[tuple[float, int]], step: float, t0: float = 1000.0, stop: float = 1006.0) -> list[Any]:
    out, k = [], 0
    while t0 + k * step <= stop:
        t = t0 + k * step
        out.append(_paging_reading(t, 500 + sum(p for at, p in events if at < t)))
        k += 1
    return out


def _saved(record: dict[str, Any]) -> dict[str, Any]:
    """As the evidence is saved and loaded again (JSON)."""
    return json.loads(json.dumps(record))  # type: ignore[no-any-return]


def _measurement_row(events: list[tuple[float, int]], answers: list[tuple[float, float]],
                     monitor_step: float = 0.1, warmups: int = 0) -> dict[str, Any]:
    """One scoped measurement whose answers the monitor judged from its own readings."""
    from inferenceos.telemetry.attribution import paging_verdict

    readings = _sampler(events, monitor_step)
    records = [{"run_id": f"run-{k}", "start_wall": a, "end_wall": b,
                "paging": _saved(paging_verdict(readings, a, b).record())} for k, (a, b) in enumerate(answers)]
    # Each run carries its own ID, as the probe records it; records are matched by ID.
    runs = [{"run_id": f"run-{k}", "warmup": k < warmups} for k in range(len(answers))]
    return {"label": "A-short", "monitor": "measurement", "runs": runs,
            "monitor_events": [{"event": "finalize", "answers": records}]}


def _check(rows: list[dict[str, Any]], second: list[Any], **kw: Any) -> tuple[str, dict[str, Any]]:
    probe = _load_probe()
    state, report = probe.consistency_check(rows, second, sampler_started=kw.pop("started", True),
                                            finalize_warmups=kw.pop("warmups", False))
    print(f"\nR: {state} | checked {report['answers_checked']} of {report['answers_expected']}, "
          f"paging exercised {report['paging_exercised']}")
    return state, report


def test_consistency_check_passes_only_with_full_evidence_and_real_paging() -> None:
    events = [(1002.45, 300), (1003.33, 40)]  # 300 inside answer 1; 40 inside answer 2
    row = _measurement_row(events, [(1002.0, 1002.9), (1003.0, 1004.0)])
    state, report = _check([row], _sampler(events, 0.13))
    assert state == "passed" and report["answers_checked"] == 2 and report["paging_exercised"]
    assert report["measurement_totals"][0]["second_sampler_pages"] == 340


def test_no_second_sampler_readings_is_inconclusive_not_passed() -> None:
    """The reviewer's case: one recorded answer and zero readings used to pass."""
    row = _measurement_row([(1002.45, 300)], [(1002.0, 1002.9)])
    state, report = _check([row], [])
    assert state.startswith("inconclusive:") and "recorded no paging reading" in state
    assert report["answers_checked"] == 0


@pytest.mark.parametrize("damage", ["sampler did not start", "no finalize event", "an answer missing",
                                    "monitor evidence unknown", "malformed bounds", "no answer timing"])
def test_missing_or_malformed_evidence_is_inconclusive(damage: str) -> None:
    events = [(1002.45, 300)]
    row = _measurement_row(events, [(1002.0, 1002.9), (1003.0, 1004.0)])
    kw: dict[str, Any] = {}
    if damage == "sampler did not start":
        kw["started"] = False
    elif damage == "no finalize event":
        row["monitor_events"] = []
    elif damage == "an answer missing":
        row["monitor_events"][0]["answers"].pop()
    elif damage == "monitor evidence unknown":
        row["monitor_events"][0]["answers"][1]["paging"] = {"status": "unknown", "pages_lower": None}
    elif damage == "malformed bounds":
        row["monitor_events"][0]["answers"][1]["paging"]["pages_upper"] = "78"
    else:
        del row["monitor_events"][0]["answers"][1]["end_wall"]
    state, _ = _check([row], _sampler(events, 0.13), **kw)
    assert state.startswith("inconclusive:"), (damage, state)


def test_warm_ups_are_expected_when_they_are_judged() -> None:
    events = [(1002.45, 300)]
    row = _measurement_row(events, [(1001.0, 1001.8), (1002.0, 1002.9)], warmups=1)
    assert _check([row], _sampler(events, 0.13), warmups=True)[0] == "passed"
    row["monitor_events"][0]["answers"].pop(0)  # the warm-up's record is missing
    state, _ = _check([row], _sampler(events, 0.13), warmups=True)
    assert state.startswith("inconclusive:") and "no finalize record for run-0" in state


def test_a_quiet_run_is_consistent_but_inconclusive() -> None:
    row = _measurement_row([], [(1002.0, 1002.9), (1003.0, 1004.0)])
    state, report = _check([row], _sampler([], 0.13))
    assert state.startswith("inconclusive: consistent, but paging was not exercised")
    assert report["answers_checked"] == 2 and not report["paging_exercised"]


def test_an_exact_fractional_rate_agrees_with_its_own_saved_evidence() -> None:
    """Reviewer: 333.333 (rounded record) against 333.3333333333417 was reported as a
    disagreement. Records are full precision now; a verdict checked against the very
    readings it came from, after saving and loading, agrees."""
    from inferenceos.telemetry.attribution import paging_verdict

    events = [(1002.45, 300)]
    readings = _sampler(events, 0.1)
    record = _saved(paging_verdict(readings, 1002.0, 1002.9).record())
    assert record["evidence"] == "exact" and record["upper_bound_rate"] == pytest.approx(300 / 0.9)
    assert record["upper_bound_rate"] != round(record["upper_bound_rate"], 3)  # really fractional, unrounded
    row = {"label": "A-short", "monitor": "measurement", "runs": [{"run_id": "r", "warmup": False}],
           "monitor_events": [{"event": "finalize", "answers": [
               {"run_id": "r", "start_wall": 1002.0, "end_wall": 1002.9, "paging": record}]}]}
    state, report = _check([row], readings)
    assert state == "passed" and not report["failures"]


def test_disagreeing_bounds_and_impossible_totals_fail() -> None:
    events = [(1002.45, 300)]
    row = _measurement_row(events, [(1002.0, 1002.9)])
    hidden = json.loads(json.dumps(row))
    hidden["monitor_events"][0]["answers"][0]["paging"].update(
        pages_lower=0, pages_upper=0, lower_bound_rate=0.0, upper_bound_rate=0.0)
    state, _ = _check([hidden], _sampler(events, 0.13))
    assert state.startswith("failed:") and "pages, 1 s peak bounds disagree" in state
    too_many = _measurement_row(events + [(1003.4, 200)], [(1002.0, 1002.9), (1003.0, 1004.0)])
    state, _ = _check([too_many], _sampler(events, 0.13))  # the second sampler never saw the 200
    assert state.startswith("failed:")


class FakeReference:
    resolved = True
    starts = True
    made: list[FakeReference] = []

    def __init__(self, specs: Any, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.errors: list[str] = []
        FakeReference.made.append(self)

    def start(self) -> bool:
        return FakeReference.starts

    def has_reading(self, spec: str) -> bool:
        return FakeReference.starts

    def readings(self) -> list[Any]:
        return []

    def stop(self) -> dict[str, object]:
        return {"resolved": FakeReference.resolved, "errors": [] if FakeReference.resolved else ["kill: TimeoutExpired"]}


def test_consistency_part_is_never_passed_without_evidence_and_its_cleanup_gates_the_probe(world: Any) -> None:
    """Through the probe: the fake profiler records no finalize evidence and the second
    sampler no readings -- Part R is inconclusive (it used to pass), the status says so,
    and the exit status is not success. An unresolved cleanup still fails it."""
    import inferenceos.telemetry.counters as counters

    world.monkeypatch.setattr(counters, "RawCounterSampler", FakeReference)
    world.monkeypatch.setattr(FakeReference, "resolved", True)
    flags = ("--monitors", "measurement", "--skip-part-a0", "--finalize-warmups", "--consistency-sampler")
    code, data = world.run(*flags)
    assert code == 2 and data["parts"]["R"].startswith("inconclusive:") and data["parts"]["A0"] == "disabled"
    assert data["status"].startswith("inconclusive:")
    assert [m["monitor"] for m in data["measurements"] if m["label"] == "A-short"] == ["measurement"]
    assert FakeReference.made[-1].interval_s == world.probe.REFERENCE_INTERVAL_S != 0.1
    assert "not an independent proof" in data["consistency_check"]["kind"]
    world.monkeypatch.setattr(FakeReference, "starts", False)
    code, data = world.run(*flags)
    assert code == 2 and "the second sampler did not start" in data["parts"]["R"]
    world.monkeypatch.setattr(FakeReference, "starts", True)
    world.monkeypatch.setattr(FakeReference, "resolved", False)
    code, data = world.run(*flags)
    print("parts with an unresolved second-sampler cleanup:", data["parts"])
    assert code == 2 and data["parts"]["R"].startswith("failed: the second sampler's cleanup was unresolved")
    assert data["status"].startswith("failed:") and data["parts"]["B"].startswith("not run")
    # An unresolved second-sampler cleanup is never described as a successful cleanup.
    assert data["parts"]["cleanup"].startswith("failed:") and "second sampler cleanup UNRESOLVED" in data["parts"]["cleanup"]


def test_every_run_carries_its_own_id_into_the_measurement_record(world: Any) -> None:
    code, data = world.run("--monitors", "measurement", "--skip-part-a0", "--skip-part-b")
    runs = data["measurements"][0]["runs"]
    print("\nrun IDs:", [r["run_id"] for r in runs])
    assert all(isinstance(r["run_id"], str) and r["run_id"].startswith("run-") for r in runs)
    assert len({r["run_id"] for r in runs}) == len(runs)


# --------------- reviewer, after the validation fixes: the second sampler's lifecycle
class LifecycleReference(FakeReference):
    """A second sampler whose startup or readiness check misbehaves on cue."""

    fault: str = ""
    resolved_stop = True
    calls: list[str] = []

    def start(self) -> bool:
        LifecycleReference.calls.append("start")
        if LifecycleReference.fault == "start raises":  # e.g. the process began, then the reader failed
            raise RuntimeError("reader thread failed to start (scripted)")
        return True

    def has_reading(self, spec: str) -> bool:
        LifecycleReference.calls.append("has_reading")
        if LifecycleReference.fault == "readiness raises":
            raise RuntimeError("readiness check failed (scripted)")
        if LifecycleReference.fault == "user interrupt":
            raise KeyboardInterrupt("user interrupt during the readiness wait (scripted)")
        return True

    def readings(self) -> list[Any]:
        return [_paging_reading(1000.0, 500), _paging_reading(1000.13, 502)]

    def stop(self) -> dict[str, object]:
        LifecycleReference.calls.append("stop")
        if LifecycleReference.fault == "stop raises":
            raise OSError("terminate failed (scripted)")
        ok = LifecycleReference.resolved_stop
        return {"resolved": ok, "errors": [] if ok else ["kill: TimeoutExpired"]}


@pytest.mark.parametrize("resolved", [True, False])
@pytest.mark.parametrize("fault", ["readiness raises", "user interrupt", "start raises"])
def test_second_sampler_is_stopped_after_any_startup_attempt(world: Any, fault: str, resolved: bool) -> None:
    """The reviewer's case: an interruption during the readiness wait used to skip
    ``stop()``, leave no cleanup record, and report the cleanup as passed."""
    import inferenceos.telemetry.counters as counters

    world.monkeypatch.setattr(counters, "RawCounterSampler", LifecycleReference)
    world.monkeypatch.setattr(LifecycleReference, "fault", fault)
    world.monkeypatch.setattr(LifecycleReference, "resolved_stop", resolved)
    world.monkeypatch.setattr(LifecycleReference, "calls", [])
    code, data = world.run("--monitors", "measurement", "--skip-part-a0", "--consistency-sampler")
    record = data["consistency_sampler"]
    print(f"{fault}, stop resolved {resolved}: exit {code} | calls {LifecycleReference.calls} | "
          f"stop {record['stop']} | readings kept {len(record['readings'])} | "
          f"interrupted_by {record.get('interrupted_by')!r} | cleanup {data['parts']['cleanup']!r} | "
          f"R {data['parts']['R']!r} | measurements {len(data['measurements'])}")
    assert code == 3 and data["status"].startswith("error:")
    assert LifecycleReference.calls.count("stop") == 1 and LifecycleReference.calls[-1] == "stop"
    assert record["created"] is True and record["started"] is False
    assert record["stop"]["resolved"] is resolved and len(record["readings"]) == 2  # evidence retained
    assert record["interrupted_by"].startswith("KeyboardInterrupt" if fault == "user interrupt" else "RuntimeError")
    assert data["measurements"] == []  # no measurement started afterwards
    assert data["parts"]["A"].startswith("not run") and data["parts"]["B"].startswith("not run")
    assert data["parts"]["R"].startswith("not run: Part A did not complete")
    if resolved:
        assert data["parts"]["cleanup"] == "passed" and "cleanup resolved" in data["parts"]["R"]
    else:
        assert data["parts"]["cleanup"].startswith("failed:")
        assert "second sampler cleanup UNRESOLVED (kill: TimeoutExpired)" in data["parts"]["cleanup"]
        assert "UNRESOLVED" in data["parts"]["R"]


def test_a_stop_that_raises_is_recorded_as_unresolved(world: Any) -> None:
    import inferenceos.telemetry.counters as counters

    world.monkeypatch.setattr(counters, "RawCounterSampler", LifecycleReference)
    world.monkeypatch.setattr(LifecycleReference, "fault", "stop raises")
    world.monkeypatch.setattr(LifecycleReference, "calls", [])
    code, data = world.run("--monitors", "measurement", "--skip-part-a0", "--consistency-sampler")
    print("stop raises:", data["consistency_sampler"]["stop"], "| cleanup:", data["parts"]["cleanup"])
    assert code == 2 and data["consistency_sampler"]["stop"]["resolved"] is False
    assert data["parts"]["R"].startswith("failed: the second sampler's cleanup was unresolved (stop raised OSError")
    assert data["parts"]["cleanup"].startswith("failed:") and data["parts"]["B"].startswith("not run")


def test_an_interrupt_during_a_measurement_still_stops_the_second_sampler(world: Any) -> None:
    import inferenceos.telemetry.counters as counters

    world.monkeypatch.setattr(counters, "RawCounterSampler", LifecycleReference)
    world.monkeypatch.setattr(LifecycleReference, "fault", "")
    world.monkeypatch.setattr(LifecycleReference, "calls", [])

    def interrupt(warmup: bool, profiler: Any) -> bool:
        raise KeyboardInterrupt("user interrupt during an answer (scripted)")

    world.monkeypatch.setattr(FakeProfiler, "fail", staticmethod(interrupt))
    code, data = world.run("--monitors", "measurement", "--skip-part-a0", "--consistency-sampler")
    assert code == 3 and LifecycleReference.calls.count("stop") == 1
    assert data["consistency_sampler"]["stop"]["resolved"] is True
    assert data["consistency_sampler"]["interrupted_by"].startswith("KeyboardInterrupt")


# -------------- reviewer, after the validation fixes: answer identities and bad numbers
def _three() -> tuple[dict[str, Any], list[Any]]:
    events = [(1003.33, 40), (1004.4, 60)]  # run-0 quiet; run-1 and run-2 paged
    row = _measurement_row(events, [(1002.0, 1002.9), (1003.0, 1004.0), (1004.1, 1004.9)])
    return row, _sampler(events, 0.13)


def test_three_answers_pass_in_any_record_order() -> None:
    row, second = _three()
    assert _check([row], second)[0] == "passed"
    row["monitor_events"][0]["answers"].reverse()  # position is never relied on
    assert _check([row], second)[0] == "passed"


@pytest.mark.parametrize("damage, expect", [
    ("duplicate substitution", "more than one finalize record for run-0"),
    ("missing", "no finalize record for run-1"),
    ("unexpected", "finalize record(s) for unexpected answer(s) run-9"),
    ("record without ID", "1 finalize record(s) without a run ID"),
    ("expected run without ID", "1 expected answer(s) carry no run ID"),
    ("expected IDs repeated", "repeated among the expected answers"),
])
def test_answer_identities_must_match_exactly_once(damage: str, expect: str) -> None:
    """The reviewer's case first: the second record replaced by a copy of the first
    (run-0, run-0, run-2 for run-0, run-1, run-2) used to pass on the count alone."""
    row, second = _three()
    records = row["monitor_events"][0]["answers"]
    if damage == "duplicate substitution":
        records[1] = json.loads(json.dumps(records[0]))
    elif damage == "missing":
        records.pop(1)
    elif damage == "unexpected":
        records[1]["run_id"] = "run-9"
    elif damage == "record without ID":
        del records[1]["run_id"]
    elif damage == "expected run without ID":
        row["runs"][1]["run_id"] = None
    else:
        row["runs"][1]["run_id"] = "run-0"
    state, report = _check([row], second)
    print(f"{damage}: records {[a.get('run_id') for a in records]} -> {state}")
    assert state.startswith("inconclusive:") and expect in state
    assert report["answers_checked"] < 3 and report["identities"][0]["problems"]


@pytest.mark.parametrize("field, value, expect", [
    ("upper_bound_rate", float("inf"), "upper_bound_rate inf is not a finite number"),
    ("lower_bound_rate", float("nan"), "lower_bound_rate nan is not a finite number"),
    ("lower_bound_rate", -1.0, "lower_bound_rate -1.0 is negative"),
    ("pages_lower", -3, "pages_lower -3 is negative"),
    ("pages_upper", 40.5, "pages_upper 40.5 is not a whole page count"),
    ("pages_upper", 40.0, "pages_upper 40.0 is not a whole page count"),
    ("pages_lower", True, "pages_lower True is not a whole page count"),
    ("lower_bound_rate", 1e9, "exceeds upper_bound_rate"),
    ("start_wall", float("-inf"), "no usable answer timing"),
    ("end_wall", 1003.0, "no usable answer timing"),  # start == end
    ("end_wall", True, "no usable answer timing"),
])
def test_malformed_numbers_are_explicitly_inconclusive(field: str, value: Any, expect: str) -> None:
    row, second = _three()
    target = row["monitor_events"][0]["answers"][1]
    (target if field.endswith("_wall") else target["paging"])[field] = value
    state, report = _check([row], second)
    print(f"{field} = {value!r}: {state}")
    assert state.startswith("inconclusive:") and expect in state
    assert report["answers_checked"] == 2  # the other two answers were still checked


# ---------------------------------------------------------------- the cold model copy
@pytest.fixture()
def cold(world: Any, tmp_path: Path) -> Any:
    """The probe's temporary directory under the test's control; the fake copy writes
    part of the file first, so a failure leaves a partial copy behind to clean up."""
    import shutil

    made: list[Path] = []
    temp_root = tmp_path / "temp"
    temp_root.mkdir()

    def mkdtemp(prefix: str = "") -> str:
        path = temp_root / f"{prefix}{len(made)}"
        path.mkdir()
        made.append(path)
        return str(path)

    def fake_copy(src: Path, dst: Path) -> None:
        shutil.copyfile(src, dst)

    world.monkeypatch.setattr(world.probe.tempfile, "mkdtemp", mkdtemp)
    world.monkeypatch.setattr(world.probe, "copy_uncached", fake_copy)
    return SimpleNamespace(made=made, root=temp_root)


def _leftovers(cold: Any) -> list[str]:
    return sorted(str(p.relative_to(cold.root)) for p in cold.root.rglob("*"))


def test_cold_model_copy_is_used_and_removed(world: Any, cold: Any) -> None:
    code, data = world.run("--cold-model-copy", "--skip-part-b")
    assert code == 0 and data["cold_model_copy"]["path"].startswith(str(cold.made[0]))
    assert data["cold_model_copy"]["cleanup"] == "deleted" and _leftovers(cold) == []


def test_cold_copy_failure_leaves_nothing_behind(world: Any, cold: Any) -> None:
    def failing_copy(src: Path, dst: Path) -> None:
        dst.write_bytes(b"partial")
        raise OSError("disk full (scripted)")

    world.monkeypatch.setattr(world.probe, "copy_uncached", failing_copy)
    code, data = world.run("--cold-model-copy", "--skip-part-b")
    print("copy failure:", data["status"], "|", data["cold_model_copy"])
    assert code == 3 and "disk full" in data["status"]
    assert data["cold_model_copy"]["cleanup"] == "deleted" and _leftovers(cold) == []


def test_metadata_failure_on_the_copy_leaves_nothing_behind(world: Any, cold: Any) -> None:
    """The reviewer's case: an invalid GGUF fails while its facts are read (exit 3) and
    used to leave the temporary copy behind."""
    import inferenceos.models.loader as loader

    def bad_model(path: Path) -> Any:
        raise ValueError(f"not a GGUF file: {path}")

    world.monkeypatch.setattr(loader, "load_model_profile", bad_model)
    code, data = world.run("--cold-model-copy", "--skip-part-b")
    print("metadata failure:", data["status"], "|", data["cold_model_copy"])
    assert code == 3 and "not a GGUF file" in data["status"]
    assert data["cold_model_copy"]["cleanup"] == "deleted" and _leftovers(cold) == []


def test_engine_setup_failure_leaves_nothing_behind(world: Any, cold: Any) -> None:
    import inferenceos.backends.llama_cpp_resident_adapter as lra

    def broken(settings: Any = None) -> Any:
        raise RuntimeError("engine setup failed (scripted)")

    world.monkeypatch.setattr(lra, "LlamaCppResidentAdapter", broken)
    code, data = world.run("--cold-model-copy", "--skip-part-b")
    assert code == 3 and "engine setup failed" in data["status"]
    assert data["cold_model_copy"]["cleanup"] == "deleted" and _leftovers(cold) == []


def test_cold_copy_is_kept_while_an_engine_may_still_map_it(world: Any, cold: Any) -> None:
    """Once an engine was started, the copy is deleted only after the last engine
    process is confirmed gone; otherwise it is kept and the reason recorded."""
    world.gone["value"] = False
    code, data = world.run("--cold-model-copy", "--skip-part-b")
    print("engine not confirmed gone:", data["cold_model_copy"]["cleanup"])
    assert code == 2 and data["cold_model_copy"]["cleanup"].startswith("not deleted: an engine process")
    assert _leftovers(cold) != []
