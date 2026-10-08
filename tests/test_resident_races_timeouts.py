"""Review 10 follow-up (reviewer questions 1 and 2): user-stop races and timeout states.

Drives the real ``LlamaServerResidentBackend``, ``LlamaCppResidentAdapter``,
``ResidentActiveWorkload`` and ``BudgetedTune.request_cancel`` with a scripted engine
process. Where a test needs termination to stay *unconfirmed*, the scripted process
reports a real, still-running child process (``sleep``-like Python child) as its pid,
and its ``stop()`` deliberately does not kill it; the child is killed at the end of the
test. Bounded waits are shortened through the backend/workload attributes
(``unwind_timeout_s``, ``exit_timeout_s``, ``previous_gone_timeout_s``,
``join_timeout_s``); the code paths are the production ones.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

from inferenceos.backends import llama_server_toolchain as srv
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.backends.resident_server import (
    LlamaServerResidentBackend,
    ResidentBackendError,
    ResidentStartCancelled,
)
from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
from inferenceos.runtime.resident_workload import ResidentActiveWorkload, ResidentTargetCancelled
from inferenceos.tune.budgeted import BudgetedTune
from tests.resident_helpers import synthetic_model

GONE_PID = 2_000_000_001


class Engine:
    """Shared controls for one test."""

    def __init__(self) -> None:
        self.pid = GONE_PID
        self.kill_on_stop = True
        self.started = 0
        self.stopped = 0
        self.block_ready = threading.Event()  # set = readiness blocks
        self.ignore_should_stop = False       # True = readiness ignores the stop
        self.release_ready = threading.Event()
        self.ready_entered = threading.Event()
        self.stream_entered = threading.Event()
        self.stream_release = threading.Event()
        self.stream_blocks = False


E = Engine()


class Proc:
    def __init__(self, command: list[str], environment: Any = None) -> None:
        self.alive = False

    @property
    def pid(self) -> int:
        return E.pid

    def start(self) -> int:
        E.started += 1
        self.alive = True
        return E.pid

    def wait_ready(self, client: Any, *, timeout_s: float, poll_s: float = 0.25,
                   should_stop: Any = None) -> float:
        E.ready_entered.set()
        if E.block_ready.is_set():
            deadline = time.monotonic() + 20.0
            while not E.release_ready.is_set():
                if not E.ignore_should_stop and should_stop is not None and should_stop():
                    raise srv.ServerStartError("llama-server startup was cancelled")
                assert time.monotonic() < deadline, "never released"
                time.sleep(0.005)
        return 0.01

    def identity(self) -> None:
        return None

    def log_text(self) -> str:
        return ""

    def is_alive(self) -> bool:
        return self.alive

    def stop(self, *, timeout_s: float = 10.0) -> tuple[int | None, str, bool]:
        E.stopped += 1
        self.alive = False
        return 0, "process_tree", True  # claims success; the OS decides if it is gone


class Client:
    def __init__(self, **_: Any) -> None:
        pass

    def stream_completion(self, **_: Any) -> Iterator[dict[str, Any]]:
        E.stream_entered.set()
        if E.stream_blocks:  # a read that does not return (no chunk, no epoch check)
            assert E.stream_release.wait(20.0)
        for _index in range(400):
            yield {"content": "x"}
            time.sleep(0.005)
        yield {"stop": True, "timings": {"predicted_n": 1}}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    global E
    E = Engine()
    monkeypatch.setattr(srv, "LlamaServerClient", Client)
    yield
    E.release_ready.set()
    E.stream_release.set()


@pytest.fixture()
def live_child() -> Iterator[int]:
    """A real process that stays alive until the test ends (termination unconfirmed)."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield child.pid
    finally:
        child.kill()
        child.wait(10)


def _model(tmp: Path) -> Any:
    path = tmp / "m.gguf"
    path.write_bytes(b"\x00")
    return synthetic_model(str(path))


def _stack(tmp: Path, *, short: bool = True) -> tuple[LlamaServerResidentBackend, LlamaCppResidentAdapter, ResidentActiveWorkload]:
    backend = LlamaServerResidentBackend(server=Path("unused"), process_factory=Proc)  # type: ignore[arg-type]
    if short:
        backend.unwind_timeout_s = 0.5
        backend.exit_timeout_s = 0.3
        backend.previous_gone_timeout_s = 0.3
    adapter = LlamaCppResidentAdapter(settings=Settings(), server=Path("unused"), backend=backend)
    workload = ResidentActiveWorkload(settings=Settings(), model=_model(tmp), adapter=adapter,
                                      configuration=RuntimeConfiguration(context_length=4096))
    workload.join_timeout_s = 0.5
    return backend, adapter, workload


def _thread(fn: Any) -> tuple[threading.Thread, dict[str, Any]]:
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - inspected
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t, box


def _fields(backend: LlamaServerResidentBackend, workload: ResidentActiveWorkload | None = None) -> dict[str, Any]:
    out = {
        "_state": backend._state, "_idle": backend._idle.is_set(), "_owned": backend._owned is not None,  # noqa: SLF001
        "_process": backend._process is not None, "_unconfirmed_pid": backend._unconfirmed_pid,  # noqa: SLF001
        "_halted": backend._halted,  # noqa: SLF001
    }
    if workload is not None:
        thread = workload._thread  # noqa: SLF001
        out["_cancelled"] = workload._cancelled  # noqa: SLF001
        out["serving_thread_alive"] = bool(thread is not None and thread.is_alive())
    return out


def _driver(workload: ResidentActiveWorkload) -> BudgetedTune:
    """The real request_cancel, wired as tune/commands.py wires it."""
    tuner = SimpleNamespace(user_stop_hook=workload.cancel, cancel_hook=workload.stop)
    return BudgetedTune(tuner, 60.0)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Question 1: sticky user cancellation races
# --------------------------------------------------------------------------- #
def test_user_cancel_racing_a_non_sticky_stop_is_not_undone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stuck-operation stop (non-sticky) is about to lift the engine's halt when the
    user's request_cancel lands. The user stop must win: still halted, still cancelled,
    and no load can start. (With the first review-10 version, the stop's resume cleared
    both flags.)"""
    backend, adapter, workload = _stack(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    inside_resume, go_on = threading.Event(), threading.Event()
    real_wait = backend._wait_previous_gone  # noqa: SLF001

    def paused_wait(timeout_s: float | None = None) -> None:
        real_wait(timeout_s)
        if threading.current_thread().name == "stuck-stop":
            inside_resume.set()
            assert go_on.wait(5.0)

    monkeypatch.setattr(backend, "_wait_previous_gone", paused_wait)
    stopper = threading.Thread(target=workload.stop, name="stuck-stop", daemon=True)
    stopper.start()
    assert inside_resume.wait(5.0)  # the stop has decided to resume and is about to
    driver = _driver(workload)
    canceller = threading.Thread(target=driver.request_cancel, args=("user pressed stop",), daemon=True)
    canceller.start()
    deadline = time.monotonic() + 5.0
    while not workload._cancelled:  # noqa: SLF001 - the flag is set before any lock
        assert time.monotonic() < deadline
        time.sleep(0.002)
    go_on.set()
    stopper.join(5.0)
    canceller.join(5.0)
    assert driver.user_cancelled == "user pressed stop"
    assert workload._cancelled and backend.halted  # noqa: SLF001
    with pytest.raises(ResidentStartCancelled):
        adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    with pytest.raises(ResidentTargetCancelled):
        workload.start()
    assert E.started == 1


def test_user_cancel_racing_a_recovery_start(tmp_path: Path) -> None:
    """A recovery's controlled reload is in its readiness wait when request_cancel
    lands: the reload is cancelled, the recovery fails closed, and later recovery or
    start attempts never start the engine."""
    backend, adapter, workload = _stack(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    workload._capture_target()  # noqa: SLF001
    backend.stop()  # the target "died"
    E.block_ready.set()
    E.ready_entered.clear()
    recovery, box = _thread(lambda: workload.recover_target(reason="process_exited"))
    assert E.ready_entered.wait(5.0) and backend.state == "starting"
    _driver(workload).request_cancel("user pressed stop")
    recovery.join(5.0)
    record = box["value"]
    assert not record.recovered
    assert backend.state == "idle" and not adapter.is_loaded and backend.halted
    started = E.started
    again = workload.recover_target(reason="process_exited")
    assert not again.recovered and "stopped by the user" in (again.failure_reason or "")
    with pytest.raises(ResidentTargetCancelled):
        workload.start()
    with pytest.raises(ResidentStartCancelled):
        adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    assert E.started == started
    # Only the explicit allow_restart lifts it.
    E.block_ready.clear()
    workload.allow_restart()
    adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    assert backend.state == "ready"
    workload.stop()


def test_start_never_lifts_a_user_cancel(tmp_path: Path) -> None:
    backend, adapter, workload = _stack(tmp_path)
    workload.cancel()
    with pytest.raises(ResidentTargetCancelled):
        workload.start()
    assert backend.halted and E.started == 0


# --------------------------------------------------------------------------- #
# Question 2: timeout states
# --------------------------------------------------------------------------- #
def test_a_start_that_does_not_unwind_in_time(tmp_path: Path) -> None:
    """(a) The readiness wait ignores the stop (a hung health check): stop() returns
    after unwind_timeout_s with cleanup incomplete; the backend stays 'starting' and
    refuses new starts until the start thread finally unwinds."""
    backend, adapter, workload = _stack(tmp_path)
    E.block_ready.set()
    E.ignore_should_stop = True
    starter, box = _thread(lambda: backend.start(_model(tmp_path), RuntimeParams(context_length=4096)))
    assert E.ready_entered.wait(5.0)
    record = backend.halt()
    assert record is not None and record.cleanup_complete is False
    assert "had not unwound" in record.notes[0]
    during = _fields(backend)
    assert during == {"_state": "starting", "_idle": False, "_owned": True, "_process": False,
                      "_unconfirmed_pid": None, "_halted": True}
    assert starter.is_alive()
    with pytest.raises(ResidentBackendError, match="still stopping"):
        backend.resume()
    E.release_ready.set()  # the hung wait returns at last
    starter.join(5.0)
    assert isinstance(box.get("error"), ResidentStartCancelled)
    assert _fields(backend) == {"_state": "idle", "_idle": True, "_owned": False, "_process": False,
                                "_unconfirmed_pid": None, "_halted": True}


def test_termination_not_confirmed_blocks_new_starts(tmp_path: Path, live_child: int) -> None:
    """(b) A ready server whose process does not exit: stop() records the pid as
    unconfirmed and every new start is refused while that process lives."""
    backend, adapter, workload = _stack(tmp_path)
    E.pid = live_child
    adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    workload.stop()
    unload = backend.last_unload
    assert unload is not None and unload.cleanup_complete is False and unload.exited is False
    assert _fields(backend, workload) == {
        "_state": "idle", "_idle": True, "_owned": False, "_process": False, "_unconfirmed_pid": live_child,
        "_halted": True, "_cancelled": False, "serving_thread_alive": False}
    with pytest.raises(ResidentBackendError, match="has not exited"):
        adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    assert E.started == 1


def test_termination_not_confirmed_during_a_cancelled_start(tmp_path: Path, live_child: int) -> None:
    """(b) The owned process of a start being cancelled does not exit."""
    backend, adapter, workload = _stack(tmp_path)
    E.pid = live_child
    E.block_ready.set()
    starter, box = _thread(lambda: backend.start(_model(tmp_path), RuntimeParams(context_length=4096)))
    assert E.ready_entered.wait(5.0)
    record = backend.halt()
    starter.join(5.0)
    assert record is not None and record.cleanup_complete is False
    assert isinstance(box.get("error"), ResidentStartCancelled)
    assert _fields(backend) == {"_state": "idle", "_idle": True, "_owned": False, "_process": False,
                                "_unconfirmed_pid": live_child, "_halted": True}
    with pytest.raises(ResidentBackendError, match="has not exited"):
        backend.resume()


def test_serving_thread_that_does_not_return(tmp_path: Path) -> None:
    """(c) The serving thread is stuck in a read that never yields: cancel() returns
    after join_timeout_s; the thread is still recorded, restarts are refused, and the
    engine stays halted until the thread returns and allow_restart() is called."""
    backend, adapter, workload = _stack(tmp_path)
    E.stream_blocks = True
    workload.start()
    assert E.stream_entered.wait(5.0)
    workload.cancel()
    assert _fields(backend, workload) == {
        "_state": "idle", "_idle": True, "_owned": False, "_process": False, "_unconfirmed_pid": None,
        "_halted": True, "_cancelled": True, "serving_thread_alive": True}
    with pytest.raises(RuntimeError, match="has not returned"):
        workload.allow_restart()
    E.stream_release.set()
    thread = workload._thread  # noqa: SLF001
    assert thread is not None
    thread.join(5.0)
    assert not thread.is_alive()
    workload.allow_restart()
    assert not backend.halted and not workload._cancelled  # noqa: SLF001


def test_overlapping_stops_while_termination_is_unconfirmed(tmp_path: Path, live_child: int) -> None:
    """(d) Two stop callers and a cleanup overlap while the process does not exit: no
    caller raises, all return within the bounds, and the end state is consistent."""
    backend, adapter, workload = _stack(tmp_path)
    E.pid = live_child
    adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    callers = [_thread(backend.halt), _thread(backend.halt), _thread(backend.cleanup), _thread(workload.stop)]
    began = time.monotonic()
    for t, _ in callers:
        t.join(10.0)
        assert not t.is_alive()
    assert time.monotonic() - began < 6.0
    assert not any("error" in box for _, box in callers), [box.get("error") for _, box in callers]
    assert _fields(backend, workload) == {
        "_state": "idle", "_idle": True, "_owned": False, "_process": False, "_unconfirmed_pid": live_child,
        "_halted": True, "_cancelled": False, "serving_thread_alive": False}
    with pytest.raises(ResidentBackendError, match="has not exited"):
        backend.start(_model(tmp_path), RuntimeParams(context_length=4096))


def test_overlapping_stop_and_cancel_while_a_start_hangs(tmp_path: Path) -> None:
    """(d) A non-sticky stop and a user cancel overlap while the start ignores the stop
    (timeout in the unwind): the user stop's stickiness survives."""
    backend, adapter, workload = _stack(tmp_path)
    E.block_ready.set()
    E.ignore_should_stop = True
    starter, box = _thread(lambda: adapter.load(_model(tmp_path), RuntimeParams(context_length=4096)))
    assert E.ready_entered.wait(5.0)
    stopper, _ = _thread(workload.stop)
    canceller, _ = _thread(workload.cancel)
    stopper.join(10.0)
    canceller.join(10.0)
    assert workload._cancelled and backend.halted  # noqa: SLF001
    E.release_ready.set()
    starter.join(5.0)
    assert "error" in box
    assert _fields(backend, workload)["_state"] == "idle"
    with pytest.raises(ResidentStartCancelled):
        adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))


def test_generation_refused_after_cancel(tmp_path: Path) -> None:
    backend, adapter, workload = _stack(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    workload.cancel()
    with pytest.raises(Exception):
        adapter.generate(GenerationRequest(prompt="p", n_predict=1))
    assert Path(tempfile.gettempdir()).exists()


# --------------------------------------------------------------------------- #
# Reviewer follow-up, item 1: start() vs cancel() at the publication point.
def test_cancel_between_load_and_publication_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """start() begins with _cancelled False; its load completes; it is paused right
    before publishing _running/_accepting/_thread; cancel() runs to completion in
    another thread; start() is released. Nothing may be published."""
    backend, adapter, workload = _stack(tmp_path)
    loaded, release = threading.Event(), threading.Event()
    real_capture = workload._capture_target  # noqa: SLF001 - the last step before publication

    def paused_capture() -> None:
        real_capture()
        loaded.set()
        assert release.wait(10.0)

    monkeypatch.setattr(workload, "_capture_target", paused_capture)
    assert workload._cancelled is False  # noqa: SLF001
    starter, box = _thread(workload.start)
    assert loaded.wait(10.0)
    assert backend.state == "ready" and E.started == 1  # the load completed
    canceller, cbox = _thread(workload.cancel)
    canceller.join(10.0)
    assert not canceller.is_alive() and "error" not in cbox  # cancel() returned
    release.set()
    starter.join(10.0)
    assert not starter.is_alive()
    print("\nstart() ended with:", repr(box.get("error")))
    print("after release:", _fields(backend, workload))
    assert isinstance(box.get("error"), ResidentTargetCancelled)
    assert workload._thread is None and not workload._running and not workload._accepting  # noqa: SLF001
    assert backend.halted and workload._cancelled  # noqa: SLF001
    assert backend.state == "idle" and not adapter.is_loaded
    # No recovery and no load until an explicit allow_restart().
    started = E.started
    assert not workload.recover_target(reason="process_exited").recovered
    with pytest.raises(ResidentStartCancelled):
        adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    with pytest.raises(ResidentTargetCancelled):
        workload.start()
    assert E.started == started
    monkeypatch.setattr(workload, "_capture_target", real_capture)
    workload.allow_restart()
    workload.start()
    assert workload._thread is not None and workload._thread.is_alive()  # noqa: SLF001
    workload.stop()


def test_cancel_during_the_load_of_start_wins(tmp_path: Path) -> None:
    """The same, with the cancel landing during start()'s load (readiness wait)."""
    backend, adapter, workload = _stack(tmp_path)
    E.block_ready.set()
    starter, box = _thread(workload.start)
    assert E.ready_entered.wait(5.0)
    workload.cancel()
    starter.join(10.0)
    assert isinstance(box.get("error"), ResidentTargetCancelled)
    assert workload._thread is None and backend.halted and workload._cancelled  # noqa: SLF001


# --------------------------------------------------------------------------- #
# Reviewer, monitor round, item 5: a reload cancelled on purpose reads as a stop,
# not as a failure; genuine reload failures keep their failure reporting.
# --------------------------------------------------------------------------- #
def _reload_cancelled_by(tmp_path: Path, stop: str) -> tuple[Any, Any, Any, int]:
    backend, adapter, workload = _stack(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(threads=4, context_length=4096))
    started = E.started
    E.block_ready.set()
    E.ready_entered.clear()
    reload, box = _thread(lambda: adapter.set_runtime_params(RuntimeParams(threads=2)))
    assert E.ready_entered.wait(5.0) and backend.state == "starting"
    (workload.cancel if stop == "user" else workload.stop)()
    reload.join(5.0)
    assert not reload.is_alive()
    return backend, adapter, box.get("error"), E.started - started


def test_a_user_stop_during_a_reload_reads_as_stopped_by_the_user(tmp_path: Path) -> None:
    backend, adapter, error, starts = _reload_cancelled_by(tmp_path, "user")
    application = adapter.last_param_application
    print("\nerror:", repr(error), "| reason:", application.reason)
    assert isinstance(error, ResidentStartCancelled)
    assert str(error) == "reload stopped by the user; engine stopped"
    assert application.reason == "reload stopped by the user; engine stopped" and application.reloaded is False
    assert "NOT restored" not in str(error) and "failed" not in application.reason
    assert starts == 1 and backend.halted and backend.state == "idle"  # no restore attempt; sticky
    assert adapter.params.threads == 4  # the new settings are never reported as current
    with pytest.raises(ResidentStartCancelled):
        adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))  # no automatic restart


def test_a_timed_stop_during_a_reload_says_so(tmp_path: Path) -> None:
    backend, adapter, error, starts = _reload_cancelled_by(tmp_path, "timed")
    assert isinstance(error, ResidentStartCancelled)
    assert str(error) == "reload stopped by a timed stop (stuck operation or cleanup); engine stopped"
    assert starts == 1 and backend.state == "idle"
