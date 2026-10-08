"""Review 10, item 1: a stop interrupts a load that has not finished.

These tests drive the real lifecycle code -- ``LlamaServerResidentBackend``,
``LlamaCppResidentAdapter`` and ``ResidentActiveWorkload`` -- with a scripted engine
process (no real server). The process's pid does not exist, so "confirmed gone" checks
succeed at once; every wait is bounded by an Event, so the tests are deterministic.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Any

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
from inferenceos.runtime.resident_workload import ResidentActiveWorkload

GONE_PID = 2_000_000_000  # never a live process


class Script:
    """Shared controls for one test: where the scripted process blocks, and counts."""

    def __init__(self) -> None:
        self.created = 0
        self.started = 0
        self.stopped = 0
        self.processes: list[ScriptedProcess] = []
        self.block_ready = False
        self.ready_entered = threading.Event()
        self.release_ready = threading.Event()
        #: Called after readiness (from ``identity()``), before publication.
        self.after_ready: Any = None
        self.stop_called = threading.Event()


SCRIPT = Script()


class ScriptedProcess:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.alive = False
        SCRIPT.created += 1
        SCRIPT.processes.append(self)

    @property
    def pid(self) -> int:
        return GONE_PID

    def start(self) -> int:
        SCRIPT.started += 1
        self.alive = True
        return GONE_PID

    def wait_ready(self, client: Any, *, timeout_s: float, poll_s: float = 0.25,
                   should_stop: Any = None) -> float:
        SCRIPT.ready_entered.set()
        if SCRIPT.block_ready:
            deadline = time.monotonic() + 10.0
            while not SCRIPT.release_ready.is_set():
                if should_stop is not None and should_stop():
                    raise srv.ServerStartError("llama-server startup was cancelled")
                if time.monotonic() > deadline:
                    raise AssertionError("readiness wait was never interrupted")
                time.sleep(0.005)
        return 0.01

    def identity(self) -> None:
        if SCRIPT.after_ready is not None:
            SCRIPT.after_ready()
        return None

    def log_text(self) -> str:
        return ""

    def is_alive(self) -> bool:
        return self.alive

    def stop(self, *, timeout_s: float = 10.0) -> tuple[int | None, str, bool]:
        SCRIPT.stopped += 1
        self.alive = False
        SCRIPT.stop_called.set()
        return 0, "process_tree", True


class ScriptedClient:
    """A stream that keeps producing until told otherwise (the generation case)."""

    entered = threading.Event()
    chunks = 0

    def __init__(self, **_: Any) -> None:
        pass

    def stream_completion(self, **_: Any) -> Any:
        ScriptedClient.entered.set()
        for _index in range(2000):
            ScriptedClient.chunks += 1
            yield {"content": "x"}
            time.sleep(0.005)
        yield {"stop": True, "timings": {"predicted_n": 1}}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    global SCRIPT
    SCRIPT = Script()
    ScriptedClient.entered = threading.Event()
    ScriptedClient.chunks = 0
    monkeypatch.setattr(srv, "LlamaServerClient", ScriptedClient)
    yield


def _model(tmp: Path) -> Any:
    from tests.resident_helpers import synthetic_model

    path = tmp / "m.gguf"
    path.write_bytes(b"\x00")
    return synthetic_model(str(path))


def _backend() -> LlamaServerResidentBackend:
    return LlamaServerResidentBackend(server=Path("unused"), process_factory=ScriptedProcess)  # type: ignore[arg-type]


def _key_files() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob("inferenceos-key-*.txt")}


def _in_thread(fn: Any) -> tuple[threading.Thread, dict[str, Any]]:
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - inspected by the test
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def test_stop_before_the_process_is_created_starts_nothing(tmp_path: Path) -> None:
    """A stop that lands before the engine exists (e.g. between the two halves of a
    reload) is sticky: the next start is refused and no process is created."""
    backend = _backend()
    adapter = LlamaCppResidentAdapter(settings=Settings(), server=Path("unused"), backend=backend)
    workload = ResidentActiveWorkload(settings=Settings(), model=_model(tmp_path), adapter=adapter,
                                      configuration=RuntimeConfiguration(context_length=4096))
    workload.cancel()  # a user stop
    with pytest.raises(ResidentStartCancelled):
        adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    assert SCRIPT.created == 0 and SCRIPT.started == 0
    assert backend.state == "idle" and not backend.is_running()
    # Recovery is explicit and only once the old process is confirmed gone.
    workload.allow_restart()
    adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    assert backend.state == "ready" and SCRIPT.started == 1
    workload.stop()


def test_stop_during_start_before_process_creation_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stop arrives while the start is preparing (after it began, before the
    process object exists): the process is never started."""
    backend = _backend()
    reached, go = threading.Event(), threading.Event()
    real = srv.pick_free_port

    def slow_port(host: str) -> int:
        reached.set()
        assert go.wait(5.0)
        return int(real(host))

    monkeypatch.setattr(srv, "pick_free_port", slow_port)
    keys_before = _key_files()
    starter, box = _in_thread(lambda: backend.start(_model(tmp_path), RuntimeParams(context_length=4096)))
    assert reached.wait(5.0) and backend.state == "starting"
    stopper, stop_box = _in_thread(backend.stop)
    deadline = time.monotonic() + 5.0
    while not backend._cancel.is_set():  # noqa: SLF001 - wait for the stop to register
        assert time.monotonic() < deadline
        time.sleep(0.002)
    go.set()
    starter.join(5.0)
    stopper.join(5.0)
    assert isinstance(box.get("error"), ResidentStartCancelled)
    assert SCRIPT.started == 0, "the engine was started after the stop"
    assert "value" in stop_box and "start cancelled" in stop_box["value"].notes[0]
    assert backend.state == "idle" and not backend.is_running() and backend.pid is None
    assert _key_files() <= keys_before, "the temporary credential was left behind"


def test_stop_during_readiness_wait_interrupts_it(tmp_path: Path) -> None:
    backend = _backend()
    adapter = LlamaCppResidentAdapter(settings=Settings(), server=Path("unused"), backend=backend)
    workload = ResidentActiveWorkload(settings=Settings(), model=_model(tmp_path), adapter=adapter,
                                      configuration=RuntimeConfiguration(context_length=4096))
    SCRIPT.block_ready = True
    keys_before = _key_files()
    loader, box = _in_thread(lambda: adapter.load(_model(tmp_path), RuntimeParams(context_length=4096)))
    assert SCRIPT.ready_entered.wait(5.0)
    assert backend.state == "starting" and not adapter.is_loaded and adapter.is_active
    began = time.monotonic()
    workload.stop()  # not gated on a finished load
    took = time.monotonic() - began
    loader.join(5.0)
    assert not loader.is_alive() and "error" in box
    assert took < 5.0
    assert SCRIPT.started == 1 and SCRIPT.stopped >= 1, "the owned process was not terminated"
    assert backend.state == "idle" and not adapter.is_loaded and not adapter.is_active
    assert _key_files() <= keys_before
    # A cancelled start never publishes a ready server, and generation is refused.
    with pytest.raises(Exception):
        adapter.generate(GenerationRequest(prompt="p", n_predict=1))


def test_stop_immediately_after_readiness_publishes_nothing(tmp_path: Path) -> None:
    backend = _backend()
    stopper_box: dict[str, Any] = {}

    def stop_now() -> None:
        # Readiness passed; the stop arrives before publication. Called from another
        # thread, as a user stop would be.
        thread, box = _in_thread(backend.stop)
        stopper_box["thread"], stopper_box["box"] = thread, box
        assert SCRIPT.stop_called.wait(5.0)

    SCRIPT.after_ready = stop_now
    with pytest.raises(ResidentStartCancelled, match="immediately after readiness"):
        backend.start(_model(tmp_path), RuntimeParams(context_length=4096))
    stopper_box["thread"].join(5.0)
    assert "value" in stopper_box["box"]
    assert backend.state == "idle" and backend.pid is None and backend.last_load is None
    with pytest.raises(ResidentBackendError):
        backend.generate(GenerationRequest(prompt="p", n_predict=1))


def test_stop_during_generation_ends_the_answer(tmp_path: Path) -> None:
    backend = _backend()
    backend.start(_model(tmp_path), RuntimeParams(context_length=4096))
    assert backend.state == "ready"
    answer, box = _in_thread(lambda: backend.generate(GenerationRequest(prompt="p", n_predict=4000)))
    assert ScriptedClient.entered.wait(5.0)
    backend.stop()
    answer.join(5.0)
    assert not answer.is_alive()
    assert isinstance(box.get("error"), ResidentBackendError)
    assert ScriptedClient.chunks < 2000
    assert backend.state == "idle" and not backend.is_running()


def test_concurrent_stops_are_safe_and_idempotent(tmp_path: Path) -> None:
    backend = _backend()
    SCRIPT.block_ready = True
    starter, box = _in_thread(lambda: backend.start(_model(tmp_path), RuntimeParams(context_length=4096)))
    assert SCRIPT.ready_entered.wait(5.0)
    stops = [_in_thread(backend.halt) for _ in range(4)] + [_in_thread(backend.cleanup)]
    for thread, _ in stops:
        thread.join(10.0)
        assert not thread.is_alive()
    starter.join(5.0)
    assert isinstance(box.get("error"), BaseException)
    assert not any("error" in b for _, b in stops), [b.get("error") for _, b in stops]
    assert backend.state == "idle" and not backend.is_running()
    assert backend.halt() is None  # once more, with nothing running
    # Still halted: no start until resumed.
    with pytest.raises(ResidentStartCancelled):
        backend.start(_model(tmp_path), RuntimeParams(context_length=4096))
    SCRIPT.block_ready = False
    backend.resume()
    backend.start(_model(tmp_path), RuntimeParams(context_length=4096))
    assert backend.state == "ready"
    backend.stop()


def test_startup_failure_releases_process_and_credentials(tmp_path: Path) -> None:
    backend = _backend()

    def boom() -> None:
        raise RuntimeError("engine log unreadable")

    SCRIPT.after_ready = boom
    keys_before = _key_files()
    with pytest.raises(RuntimeError, match="unreadable"):
        backend.start(_model(tmp_path), RuntimeParams(context_length=4096))
    assert SCRIPT.stopped == 1 and backend.state == "idle" and backend.pid is None
    assert _key_files() <= keys_before
    SCRIPT.after_ready = None
    backend.start(_model(tmp_path), RuntimeParams(context_length=4096))  # recovery works
    backend.stop()


def test_no_new_start_while_the_previous_process_is_not_confirmed_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inferenceos.backends.resident_server as rs

    backend = _backend()
    backend.start(_model(tmp_path), RuntimeParams(context_length=4096))
    monkeypatch.setattr(rs, "wait_process_exit", lambda pid, timeout_s=10.0: False)
    backend.stop()
    with pytest.raises(ResidentBackendError, match="has not exited"):
        backend._wait_previous_gone(timeout_s=0.01)  # noqa: SLF001
    monkeypatch.setattr(rs, "wait_process_exit", lambda pid, timeout_s=10.0: True)
    backend.start(_model(tmp_path), RuntimeParams(context_length=4096))
    backend.stop()


def test_reload_cancelled_by_a_stop_does_not_restore(tmp_path: Path) -> None:
    """A stop during the start half of a controlled reload must not be undone by the
    adapter's "restore the prior configuration" fallback."""
    backend = _backend()
    adapter = LlamaCppResidentAdapter(settings=Settings(), server=Path("unused"), backend=backend)
    adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    SCRIPT.block_ready = True
    SCRIPT.ready_entered.clear()
    reload, box = _in_thread(adapter.restart)
    assert SCRIPT.ready_entered.wait(5.0)
    adapter.halt()
    reload.join(5.0)
    assert "error" in box
    assert SCRIPT.started == 2, "the prior configuration was restarted after the stop"
    assert backend.state == "idle" and not adapter.is_loaded
    application = adapter.last_param_application
    assert application is not None and "not restarted" in (application.error or "")


def test_plain_stop_interrupts_a_load_but_does_not_block_the_next_measurement(tmp_path: Path) -> None:
    """``stop()`` (a stuck operation, a timed probe, the cleanup) ends a load in progress
    too, but the next measurement may start the engine again once the old process is
    gone; only ``cancel()`` (a user stop) is sticky."""
    backend = _backend()
    adapter = LlamaCppResidentAdapter(settings=Settings(), server=Path("unused"), backend=backend)
    workload = ResidentActiveWorkload(settings=Settings(), model=_model(tmp_path), adapter=adapter,
                                      configuration=RuntimeConfiguration(context_length=4096))
    SCRIPT.block_ready = True
    loader, box = _in_thread(lambda: adapter.load(_model(tmp_path), RuntimeParams(context_length=4096)))
    assert SCRIPT.ready_entered.wait(5.0)
    workload.stop()
    loader.join(5.0)
    assert "error" in box and backend.state == "idle"
    SCRIPT.block_ready = False
    adapter.load(_model(tmp_path), RuntimeParams(context_length=4096))
    assert backend.state == "ready"
    workload.stop()
