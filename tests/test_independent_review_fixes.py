"""Independent review of 9e29ba8: public lifecycle, judgement and recovery routes."""
from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from inferenceos.backends import llama_server_toolchain as srv
from inferenceos.backends import resident_server as rs
from inferenceos.backends.release import release_evidence
from inferenceos.backends.process_control import process_identity
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.config import Settings
from inferenceos.contracts.enums import RollbackExecutionStatus
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.interfaces.backend_adapter import BackendTelemetry, RuntimeParams
from inferenceos.tune.prompt_reuse import REUSE, TrialResult, evaluate
from inferenceos.tune.prompt_reuse_sim import SimulatedReuseEngine
from inferenceos.tune.reuse_correctness import (
    DEFAULT, SWA_FULL, EVIDENCE_INCONCLUSIVE, UNSCORABLE, RIGHT,
    answer_oracle, build_pairs, chat_summary, correctness_summary, judge, run_chat_exposure,
)
from tests.resident_helpers import synthetic_model
from tests.test_resident_recovery import RecoverableFakeAdapter, _provision, _workload
from tests.test_rollback_executor import _setup


@pytest.mark.parametrize("first_route", ["stop", "cleanup"])
@pytest.mark.parametrize("second_route", ["stop", "cleanup"])
def test_cleanup_cannot_publish_idle_until_the_captured_stop_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_route: str, second_route: str,
) -> None:
    """Own real Python sleepers, never models. Always reap them, including on failure."""
    entered, proceed = threading.Event(), threading.Event()
    children: list[subprocess.Popen[bytes]] = []
    owners: list[Any] = []
    errors: list[BaseException] = []

    class Process:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.child: subprocess.Popen[bytes] | None = None
            self.index = len(owners)
            self.stops = 0
            owners.append(self)

        @property
        def pid(self) -> int | None:
            return self.child.pid if self.child else None

        def start(self) -> int:
            self.child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(60)"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            children.append(self.child)
            return self.child.pid

        def wait_ready(self, *args: Any, **kwargs: Any) -> float:
            return 0.0

        def identity(self) -> Any:
            return process_identity(self.pid) if self.pid is not None else None

        def log_text(self) -> str:
            return ""

        def is_alive(self) -> bool:
            return self.child is not None and self.child.poll() is None

        def stop(self, **kwargs: Any) -> tuple[int | None, str, bool]:
            self.stops += 1
            if self.index == 0 and self.stops == 1:
                entered.set()
                assert proceed.wait(5)
            assert self.child is not None
            if self.child.poll() is None:
                self.child.kill()
            return self.child.wait(5), "owned-python-child", True

    backend = rs.LlamaServerResidentBackend(server=Path("never-executed"), process_factory=Process)  # type: ignore[arg-type]
    backend.unwind_timeout_s = backend.previous_gone_timeout_s = 0.03
    monkeypatch.setattr(srv, "LlamaServerClient", lambda **kwargs: object())
    monkeypatch.setattr(rs, "sweep_stale_key_files", lambda: None)
    model_path = tmp_path / "placeholder.gguf"
    model_path.write_bytes(b"not a model; the factory launches only Python")
    model = synthetic_model(str(model_path))

    def first() -> None:
        try:
            getattr(backend, first_route)()
        except BaseException as exc:
            errors.append(exc)

    worker: threading.Thread | None = None
    try:
        backend.start(model, RuntimeParams(context_length=2048))
        key = backend.api_key_file
        worker = threading.Thread(target=first, daemon=True)
        worker.start()
        assert entered.wait(5)
        try:
            getattr(backend, second_route)()
        except rs.ResidentBackendError:
            assert second_route == "stop"
        assert backend.state == "stopping" and not backend._idle.is_set()
        assert not release_evidence(backend).clean
        assert key is not None and key.exists()
        with pytest.raises(rs.ResidentBackendError, match="still stopping"):
            backend.start(model, RuntimeParams(context_length=2048))
        assert len(owners) == 1 and owners[0].stops == 1
        proceed.set()
        worker.join(5)
        assert not worker.is_alive() and not errors
        assert not key.exists() and release_evidence(backend).clean
        backend.start(model, RuntimeParams(context_length=2048))
        replacement_key, replacement_pid = backend.api_key_file, backend.pid
        assert owners[1].is_alive() and backend.pid == replacement_pid
        assert replacement_key is not None and replacement_key.exists()
        assert not release_evidence(backend).clean
        backend.cleanup()
        assert not owners[1].is_alive() and not replacement_key.exists()
        assert release_evidence(backend).clean
    finally:
        proceed.set()
        if worker is not None:
            worker.join(5)
        backend.cleanup()
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(5)


@pytest.mark.parametrize("interrupt", [False, True])
def test_a_failed_stop_retains_its_process_owner_for_cleanup_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool,
) -> None:
    import tests.test_resident_cancellation as script
    monkeypatch.setattr(script, "SCRIPT", script.Script())
    monkeypatch.setattr(srv, "LlamaServerClient", script.ScriptedClient)
    failure = KeyboardInterrupt("injected interrupt") if interrupt else RuntimeError("injected stop failure")

    class Process(script.ScriptedProcess):
        calls = 0

        def stop(self, **kwargs: Any) -> tuple[int | None, str, bool]:
            self.calls += 1
            if self.calls == 1:
                raise failure
            return super().stop(**kwargs)

    backend = rs.LlamaServerResidentBackend(server=Path("unused"), process_factory=Process)  # type: ignore[arg-type]
    monkeypatch.setattr(rs, "wait_process_exit", lambda *a, **k: not script.SCRIPT.processes[0].alive)
    backend.start(script._model(tmp_path), RuntimeParams())
    key = backend.api_key_file
    with pytest.raises(type(failure)) as caught:
        backend.stop()
    assert caught.value is failure
    assert backend.state == "idle" and backend._idle.is_set()
    assert backend.unconfirmed_pid == script.GONE_PID
    assert key is not None and not key.exists() and not release_evidence(backend).clean
    backend.cleanup()
    assert not script.SCRIPT.processes[0].alive
    assert backend.unconfirmed_pid is None and release_evidence(backend).clean


@pytest.mark.parametrize("route", ["unload", "cleanup"])
def test_adapter_late_release_cannot_clear_a_replacements_requested_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    import tests.test_resident_cancellation as script
    monkeypatch.setattr(script, "SCRIPT", script.Script())
    monkeypatch.setattr(srv, "LlamaServerClient", script.ScriptedClient)
    backend = rs.LlamaServerResidentBackend(server=Path("unused"), process_factory=script.ScriptedProcess)  # type: ignore[arg-type]
    adapter = LlamaCppResidentAdapter(settings=Settings(), server=Path("unused"), backend=backend)
    model = script._model(tmp_path)
    adapter.load(model, RuntimeParams(threads=3))
    entered, proceed = threading.Event(), threading.Event()
    release_method = "stop" if route == "unload" else "cleanup"
    release = getattr(backend, release_method)

    def delayed_release() -> Any:
        result = release()
        entered.set()  # backend release completed; adapter bookkeeping has not
        assert proceed.wait(5)
        return result

    monkeypatch.setattr(backend, release_method, delayed_release)
    worker, outcome = script._in_thread(getattr(adapter, route))
    try:
        assert entered.wait(5)
        adapter.load(model, RuntimeParams(threads=5))
        proceed.set()
        worker.join(5)
        assert not worker.is_alive() and "error" not in outcome
        assert adapter.is_loaded and adapter.params.threads == 5 and adapter._model == model
    finally:
        proceed.set()
        worker.join(5)
        monkeypatch.setattr(backend, release_method, release)
        adapter.cleanup()


@pytest.mark.parametrize("route,stage", [
    (route, stage) for route in ("load", "reload", "restart")
    for stage in ("success", "failure", "restore_success", "restore_failure")
    if route != "load" or not stage.startswith("restore")
])
@pytest.mark.parametrize("release_route", ["unload", "cleanup"])
def test_superseded_adapter_completion_cannot_publish_or_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, stage: str,
    release_route: str,
) -> None:
    """Pause the backend return, not readiness; a replacement has already won."""
    from tests.resident_helpers import FakeResidentBackend, synthetic_model
    from inferenceos.backends.resident_server import ResidentBackendError
    import tests.test_resident_cancellation as script

    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0")
    model = synthetic_model(str(path))
    backend = FakeResidentBackend()
    adapter = LlamaCppResidentAdapter(server=Path("unused"), backend=backend)
    if route != "load":
        adapter.load(model, RuntimeParams(threads=2))
    entered, proceed = threading.Event(), threading.Event()
    original_start = backend.start
    failure = RuntimeError("delayed start failure")
    worker_id: int | None = None
    attempts = 0

    def delayed_start(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        if threading.get_ident() != worker_id:
            return original_start(*args, **kwargs)
        attempts += 1
        if stage.startswith("restore") and attempts == 1:
            raise failure
        if stage in {"success", "restore_success"}:
            result = original_start(*args, **kwargs)
        else:
            result = None
        entered.set()
        assert proceed.wait(5)
        if stage in {"failure", "restore_failure"}:
            raise failure
        return result

    monkeypatch.setattr(backend, "start", delayed_start)

    def operation() -> None:
        nonlocal worker_id
        worker_id = threading.get_ident()
        if route == "load":
            adapter.load(model, RuntimeParams(threads=3))
        elif route == "reload":
            adapter.set_runtime_params(RuntimeParams(threads=3))
        else:
            adapter.restart()

    worker, result = script._in_thread(operation)
    try:
        assert entered.wait(5)
        getattr(adapter, release_route)()
        adapter.load(model, RuntimeParams(threads=5))
        replacement_pid = backend.pid
        # A newer application record must survive failure paths as well as success.
        adapter.set_runtime_params(RuntimeParams(threads=5))
        application = adapter.last_param_application
        proceed.set()
        worker.join(5)
        assert not worker.is_alive()
        assert isinstance(result.get("error"), (RuntimeError, ResidentBackendError))
        if stage == "failure":
            assert result["error"] is failure  # stale failure never triggers rollback
        assert attempts == (2 if stage.startswith("restore") else 1)
        assert adapter.params.threads == 5 and adapter._model == model
        assert adapter.last_param_application is application
        assert backend.pid == replacement_pid and backend.is_running()
        adapter.restart()
        assert adapter.params.threads == 5
        adapter.set_runtime_params(RuntimeParams(threads=3))
        assert adapter.last_param_application is not None
        assert adapter.last_param_application.changed and adapter.last_param_application.reloaded
        assert adapter.params.threads == 3
    finally:
        proceed.set()
        worker.join(5)
        adapter.cleanup()


@pytest.mark.parametrize("route", ["load", "reload", "restart"])
def test_pending_adapter_request_refuses_competing_configuration_operations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    from tests.resident_helpers import FakeResidentBackend, synthetic_model
    from inferenceos.backends.resident_server import ResidentBackendError
    import tests.test_resident_cancellation as script

    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0")
    model = synthetic_model(str(path))
    backend = FakeResidentBackend()
    adapter = LlamaCppResidentAdapter(server=Path("unused"), backend=backend)
    if route != "load":
        adapter.load(model, RuntimeParams(threads=2))
    start = backend.start
    entered, proceed = threading.Event(), threading.Event()

    def delayed(*args: Any, **kwargs: Any) -> Any:
        result = start(*args, **kwargs)
        entered.set()
        assert proceed.wait(5)
        return result

    monkeypatch.setattr(backend, "start", delayed)
    action = (lambda: adapter.load(model, RuntimeParams(threads=3))) if route == "load" else (
        (lambda: adapter.set_runtime_params(RuntimeParams(threads=3))) if route == "reload"
        else adapter.restart)
    worker, result = script._in_thread(action)
    try:
        assert entered.wait(5)
        for competing in (lambda: adapter.load(model, RuntimeParams(threads=5)),
                          lambda: adapter.set_runtime_params(RuntimeParams(threads=2)),
                          lambda: adapter.set_runtime_params(RuntimeParams()), adapter.restart):
            with pytest.raises(ResidentBackendError, match="in progress"):
                competing()
        proceed.set()
        worker.join(5)
        assert not worker.is_alive() and "error" not in result
        assert adapter.params.threads == (2 if route == "restart" else 3)
    finally:
        proceed.set()
        worker.join(5)
        adapter.cleanup()


@pytest.mark.parametrize("route", ["load", "reload", "restart"])
@pytest.mark.parametrize("failed", [False, True])
def test_backend_load_evidence_is_required_even_if_replacement_bypassed_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, failed: bool,
) -> None:
    from tests.resident_helpers import FakeResidentBackend, synthetic_model
    import tests.test_resident_cancellation as script

    path = tmp_path / "model.gguf"
    path.write_bytes(b"\0")
    model = synthetic_model(str(path))
    backend = FakeResidentBackend()
    adapter = LlamaCppResidentAdapter(server=Path("unused"), backend=backend)
    if route != "load":
        adapter.load(model, RuntimeParams(threads=2))
    start = backend.start
    entered, proceed = threading.Event(), threading.Event()
    failure = RuntimeError("old backend call failed")

    def delayed(*args: Any, **kwargs: Any) -> Any:
        measurement = None if failed else start(*args, **kwargs)
        entered.set()
        assert proceed.wait(5)
        if failed:
            raise failure
        return measurement

    monkeypatch.setattr(backend, "start", delayed)
    action = (lambda: adapter.load(model, RuntimeParams(threads=3))) if route == "load" else (
        (lambda: adapter.set_runtime_params(RuntimeParams(threads=3))) if route == "reload"
        else adapter.restart)
    worker, result = script._in_thread(action)
    try:
        assert entered.wait(5)
        backend.cleanup()
        replacement = start(model, RuntimeParams(threads=5))
        proceed.set()
        worker.join(5)
        assert not worker.is_alive() and "error" in result
        if failed:
            assert result["error"] is failure
        assert adapter.params.threads != 3
        assert adapter.last_param_application is None
        assert backend.last_load is replacement and backend.pid == replacement.pid
    finally:
        proceed.set()
        worker.join(5)
        adapter.cleanup()


@pytest.mark.parametrize("operation", ["configuration", "restart"])
@pytest.mark.parametrize("route", ["stop", "cancel", "cancel_then_allow_restart"])
@pytest.mark.parametrize("phase", [
    "failed_start", "start_admission", "restore_admission", "stop_admission", "failed_restore",
])
@pytest.mark.parametrize("replacement", [False, True])
def test_workload_stop_cancels_old_dispatch_and_rollback_permanently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, route: str,
    phase: str, replacement: bool,
) -> None:
    """Real caller/thread/lifecycle; delay a return or admission, never a timeout."""
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.runtime.resident_workload import ResidentActiveWorkload
    import tests.test_resident_cancellation as script

    monkeypatch.setattr(script, "SCRIPT", script.Script())
    requests = 0

    class Client(script.ScriptedClient):
        def stream_completion(self, **kwargs: Any) -> Any:
            nonlocal requests
            requests += 1
            yield {"content": "ok"}
            yield {"stop": True, "timings": {"predicted_n": 1, "predicted_ms": 1,
                                             "prompt_n": 1, "prompt_ms": 1}}

    monkeypatch.setattr(srv, "LlamaServerClient", Client)
    monkeypatch.setattr(rs, "sweep_stale_key_files", lambda: None)
    entered, proceed = threading.Event(), threading.Event()
    failure = RuntimeError("ordinary readiness failure")
    worker_id: int | None = None
    attempts = 0

    class Process(script.ScriptedProcess):
        def wait_ready(self, *args: Any, **kwargs: Any) -> float:
            if threading.get_ident() == worker_id:
                if (phase in {"failed_start", "restore_admission", "failed_restore"}
                        and attempts == 1) or (phase == "failed_restore" and attempts == 2):
                    raise failure
            return super().wait_ready(*args, **kwargs)

    backend = rs.LlamaServerResidentBackend(server=Path("unused"), process_factory=Process)  # type: ignore[arg-type]
    adapter = LlamaCppResidentAdapter(server=Path("unused"), backend=backend)
    config = RuntimeConfiguration(context_length=2048, gpu_offload_layers=0, threads=2)
    workload = ResidentActiveWorkload(settings=Settings(), model=script._model(tmp_path),
                                      adapter=adapter, configuration=config)
    start, stop = backend.start, backend.stop

    def wait() -> None:
        entered.set()
        assert proceed.wait(10)

    def delayed_start(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        own = threading.get_ident() == worker_id
        if own:
            attempts += 1
            if (phase == "start_admission" and attempts == 1
                    or phase == "restore_admission" and attempts == 2):
                wait()  # before the backend claims the start, including rollback
        try:
            return start(*args, **kwargs)
        except Exception:
            if own and (phase == "failed_start" and attempts == 1
                        or phase == "failed_restore" and attempts == 2):
                assert backend.state == "idle"
                wait()  # failure cleanup is complete; Python exception is delayed
            raise

    def delayed_stop(*args: Any, **kwargs: Any) -> Any:
        if threading.get_ident() == worker_id and phase == "stop_admission":
            wait()
        return stop(*args, **kwargs)

    monkeypatch.setattr(backend, "start", delayed_start)
    monkeypatch.setattr(backend, "stop", delayed_stop)
    worker: threading.Thread | None = None
    try:
        workload.start()
        assert workload.snapshot().running and workload.pause()
        old_serve_thread = workload._thread

        def change() -> None:
            nonlocal worker_id
            worker_id = threading.get_ident()
            if operation == "configuration":
                workload.set_configuration(config.model_copy(update={"threads": 3}))
            else:
                workload.restart()

        worker, result = script._in_thread(change)
        assert entered.wait(10)
        if route == "stop":
            workload.stop()
        else:
            workload.cancel()
            if route == "cancel_then_allow_restart":
                workload.allow_restart()
        assert not workload.snapshot().running and workload.target_pid() is None
        assert old_serve_thread is not None and not old_serve_thread.is_alive()
        assert backend.state == "idle" and not backend.is_running()
        assert release_evidence(backend).clean
        if replacement:
            if route == "cancel":
                workload.allow_restart()
            workload.set_configuration(config.model_copy(update={"threads": 5}))
            workload.start()
            assert workload.pause() and workload.snapshot().running
            application = adapter.last_param_application
            load = backend.last_load
        before = (script.SCRIPT.started, requests)
        proceed.set()
        worker.join(10)
        assert not worker.is_alive() and isinstance(result.get("error"), Exception)
        assert (script.SCRIPT.started, requests) == before
        assert sum(p.alive for p in script.SCRIPT.processes) == int(replacement)
        if replacement:
            assert backend.is_running() and workload.snapshot().running
            assert workload.target_pid() == backend.pid
            assert backend.last_load is load and adapter.last_param_application is application
            assert workload.configuration().threads == adapter.params.threads == backend._params.threads == 5
        else:
            assert backend.state == "idle" and not workload.snapshot().running
            assert workload.target_pid() is None and release_evidence(backend).clean
            if adapter.last_param_application is not None:
                assert not adapter.last_param_application.verified
            # Cancellation bars old work; explicit future starts still work.
            if route == "cancel":
                workload.allow_restart()
            workload.start()
            assert workload.pause() and workload.snapshot().running
    finally:
        proceed.set()
        if worker is not None:
            worker.join(10)
        workload.stop()
        adapter.cleanup()
        assert not any(p.alive for p in script.SCRIPT.processes)


@pytest.mark.parametrize("by_user", [False, True])
def test_an_older_resume_cannot_undo_a_new_adapter_halt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, by_user: bool,
) -> None:
    import tests.test_resident_cancellation as script
    monkeypatch.setattr(script, "SCRIPT", script.Script())
    monkeypatch.setattr(srv, "LlamaServerClient", script.ScriptedClient)
    backend = rs.LlamaServerResidentBackend(server=Path("unused"), process_factory=script.ScriptedProcess)  # type: ignore[arg-type]
    adapter = LlamaCppResidentAdapter(server=Path("unused"), backend=backend)
    model = script._model(tmp_path)
    adapter.load(model, RuntimeParams(threads=2))
    adapter.halt()
    entered, proceed = threading.Event(), threading.Event()
    resume = backend.resume

    def delayed_resume(**kwargs: Any) -> bool:
        entered.set()
        assert proceed.wait(10)
        return resume(**kwargs)

    monkeypatch.setattr(backend, "resume", delayed_resume)
    worker, result = script._in_thread(adapter.resume)
    try:
        assert entered.wait(10)
        adapter.halt(by_user=by_user)
        proceed.set()
        worker.join(10)
        assert not worker.is_alive() and result.get("value") is False
        # The old backend resume cleared its flag after the new halt. Adapter
        # admission must nevertheless remain barred until a new explicit resume.
        assert not backend.halted
        with pytest.raises(rs.ResidentStartCancelled, match="stopped until resumed"):
            adapter.load(model, RuntimeParams(threads=3))
        for params in (RuntimeParams(), RuntimeParams(threads=3)):
            with pytest.raises(rs.ResidentStartCancelled):
                adapter.set_runtime_params(params)
        monkeypatch.setattr(backend, "resume", resume)
        assert adapter.resume()
        adapter.load(model, RuntimeParams(threads=3))
        assert adapter.is_loaded and adapter.params.threads == 3
    finally:
        proceed.set()
        worker.join(10)
        adapter.cleanup()


@pytest.mark.parametrize("by_user", [False, True])
@pytest.mark.parametrize("newer_halt", [False, True])
@pytest.mark.parametrize("release_route", ["unload", "cleanup", "restoration_verify", "restoration_capture"])
def test_release_preserves_halt_permission_independently_of_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, by_user: bool,
    newer_halt: bool, release_route: str,
) -> None:
    from inferenceos.experiment.restoration import ResidentRestorationVerifier
    import tests.test_resident_cancellation as script

    monkeypatch.setattr(script, "SCRIPT", script.Script())
    monkeypatch.setattr(srv, "LlamaServerClient", script.ScriptedClient)
    monkeypatch.setattr(rs, "sweep_stale_key_files", lambda: None)
    backend = rs.LlamaServerResidentBackend(server=Path("unused"), process_factory=script.ScriptedProcess)  # type: ignore[arg-type]
    adapter = LlamaCppResidentAdapter(server=Path("unused"), backend=backend)
    model = script._model(tmp_path)
    adapter.load(model, RuntimeParams(threads=2))
    key = backend.api_key_file
    adapter.halt(by_user=by_user)
    entered, proceed = threading.Event(), threading.Event()
    resume = backend.resume

    def delayed_resume(**kwargs: Any) -> bool:
        entered.set()
        assert proceed.wait(10)
        return resume(**kwargs)

    monkeypatch.setattr(backend, "resume", delayed_resume)
    worker, result = script._in_thread(adapter.resume)
    try:
        assert entered.wait(10)
        if newer_halt:
            adapter.halt(by_user=by_user)
        if release_route.startswith("restoration_"):
            verifier = ResidentRestorationVerifier(adapter=adapter, model=model)
            configuration = RuntimeConfiguration(context_length=2048, gpu_offload_layers=0, threads=2)
            if release_route == "restoration_verify":
                verification = verifier.verify(configuration)
                assert not verification.verified
                assert "ResidentStartCancelled" in verification.reason
            else:
                verifier.capture(configuration)  # logs the refusal, finally unloads
        else:
            getattr(adapter, release_route)()
        # Release still owns publication clearing and credential/process cleanup.
        assert adapter.params == RuntimeParams() and adapter.last_param_application is None
        assert key is not None and not key.exists() and adapter.release_evidence().clean
        assert script.SCRIPT.started == 1 and not backend.is_running()
        proceed.set()
        worker.join(10)
        assert not worker.is_alive() and result == {"value": not newer_halt}
        assert not backend.halted  # the old native resume really did clear its flag
        monkeypatch.setattr(backend, "resume", resume)
        if newer_halt:
            with pytest.raises(rs.ResidentStartCancelled, match="stopped until resumed"):
                adapter.load(model, RuntimeParams(threads=3))
            for params in (RuntimeParams(), RuntimeParams(threads=3)):
                with pytest.raises(rs.ResidentStartCancelled):
                    adapter.set_runtime_params(params)
            assert script.SCRIPT.started == 1 and not adapter.is_loaded
            assert adapter.resume()  # only a current explicit resume grants permission
        adapter.load(model, RuntimeParams(threads=3))
        assert adapter.is_loaded and adapter.params.threads == 3
        adapter.cleanup()
        adapter.load(model, RuntimeParams(threads=4))  # cleanup does not invent a halt
        assert adapter.is_loaded and adapter.params.threads == 4
    finally:
        proceed.set()
        worker.join(10)
        adapter.cleanup()
        assert not any(p.alive for p in script.SCRIPT.processes)


@pytest.mark.parametrize("release_route", ["unload", "cleanup"])
@pytest.mark.parametrize("refusal", ["false", "error", "interrupt", "unless"])
def test_release_preserves_halt_after_an_unsuccessful_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_route: str, refusal: str,
) -> None:
    import tests.test_resident_cancellation as script

    monkeypatch.setattr(script, "SCRIPT", script.Script())
    monkeypatch.setattr(srv, "LlamaServerClient", script.ScriptedClient)
    monkeypatch.setattr(rs, "sweep_stale_key_files", lambda: None)
    backend = rs.LlamaServerResidentBackend(server=Path("unused"), process_factory=script.ScriptedProcess)  # type: ignore[arg-type]
    adapter = LlamaCppResidentAdapter(server=Path("unused"), backend=backend)
    model = script._model(tmp_path)
    adapter.load(model, RuntimeParams(threads=2))
    adapter.halt(by_user=True)
    resume = backend.resume
    failure = KeyboardInterrupt("resume interrupted") if refusal == "interrupt" else rs.ResidentBackendError("exit still pending")

    def refused(**kwargs: Any) -> bool:
        if refusal == "false":
            return False
        raise failure

    try:
        if refusal == "unless":
            assert not adapter.resume(unless=lambda: True)
        else:
            monkeypatch.setattr(backend, "resume", refused)
            if refusal == "false":
                assert not adapter.resume()
            else:
                with pytest.raises(type(failure)) as caught:
                    adapter.resume()
                assert caught.value is failure
        getattr(adapter, release_route)()
        # Model a later native completion clearing the backend flag: it is not a
        # successful adapter resume and cannot erase the adapter's permission gate.
        assert resume() and not backend.halted
        expected = rs.ResidentBackendError if refusal == "error" else rs.ResidentStartCancelled
        message = "exit still pending" if refusal == "error" else "stopped until resumed"
        with pytest.raises(expected, match=message):
            adapter.load(model, RuntimeParams(threads=3))
        assert script.SCRIPT.started == 1 and adapter.release_evidence().clean
        monkeypatch.setattr(backend, "resume", resume)
        assert adapter.resume()
        adapter.load(model, RuntimeParams(threads=3))
        assert adapter.is_loaded and adapter.params.threads == 3
    finally:
        adapter.cleanup()


@pytest.mark.parametrize("release_route", ["unload", "cleanup"])
def test_release_inside_resume_predicate_cannot_erase_a_new_halt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, release_route: str,
) -> None:
    import tests.test_resident_cancellation as script

    monkeypatch.setattr(script, "SCRIPT", script.Script())
    monkeypatch.setattr(srv, "LlamaServerClient", script.ScriptedClient)
    monkeypatch.setattr(rs, "sweep_stale_key_files", lambda: None)
    backend = rs.LlamaServerResidentBackend(server=Path("unused"), process_factory=script.ScriptedProcess)  # type: ignore[arg-type]
    adapter = LlamaCppResidentAdapter(server=Path("unused"), backend=backend)
    model = script._model(tmp_path)
    adapter.halt()
    calls = 0

    def newer_intent() -> bool:
        nonlocal calls
        calls += 1
        if calls == 2:  # adapter's last check, after the native resume returned
            adapter.halt(by_user=True)
            getattr(adapter, release_route)()
        return False

    try:
        assert not adapter.resume(unless=newer_intent) and calls == 2
        backend.resume()  # an older native completion must not authorize a load
        with pytest.raises(rs.ResidentStartCancelled, match="stopped until resumed"):
            adapter.load(model, RuntimeParams())
        assert script.SCRIPT.started == 0
        assert adapter.resume()
        adapter.load(model, RuntimeParams())
        assert script.SCRIPT.started == 1
    finally:
        adapter.cleanup()


@pytest.mark.parametrize("intent", ["pause", "stop", "cancel"])
def test_serving_request_preparation_cannot_cross_pause_or_stop_admission(
    monkeypatch: pytest.MonkeyPatch, intent: str,
) -> None:
    from inferenceos.runtime import resident_workload as module
    adapter = RecoverableFakeAdapter()
    workload = _workload(adapter)
    workload.join_timeout_s = 0.03
    entered, proceed, finished = threading.Event(), threading.Event(), threading.Event()
    uuid4, serve = module.uuid.uuid4, workload._serve_one

    def delayed_id() -> Any:
        if threading.current_thread() is workload._thread and not entered.is_set():
            entered.set()
            assert proceed.wait(5)
        return uuid4()

    def one() -> None:
        try:
            serve()
        finally:
            finished.set()

    monkeypatch.setattr(module.uuid, "uuid4", delayed_id)
    monkeypatch.setattr(workload, "_serve_one", one)
    try:
        workload.start()
        assert entered.wait(5)
        serve_thread = workload._thread
        if intent == "pause":
            assert workload.pause()
        else:
            getattr(workload, intent)()
        assert workload.snapshot().safe_to_pause
        assert workload.request_records() == []
        loads = adapter.load_calls
        proceed.set()
        assert finished.wait(5)
        assert workload.request_records() == [] and adapter.load_calls == loads
        assert adapter.last_generation is None
        assert not workload.snapshot().generating
        if intent == "pause":
            assert workload.snapshot().running and adapter.is_loaded
            # Deliberate synchronous calls still use the existing helper contract.
            serve()
            assert len(workload.request_records()) == 1
        else:
            assert serve_thread is not None
            serve_thread.join(5)
            assert not serve_thread.is_alive()
            assert not workload.snapshot().running and workload.target_pid() is None
    finally:
        proceed.set()
        workload.stop()
        adapter.cleanup()


def test_cancellation_does_not_wait_forever_for_another_startup_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.test_resident_cancellation as script
    monkeypatch.setattr(script, "SCRIPT", script.Script())
    monkeypatch.setattr(srv, "LlamaServerClient", script.ScriptedClient)
    entered, proceed = threading.Event(), threading.Event()

    class Process(script.ScriptedProcess):
        calls = 0

        def stop(self, **kwargs: Any) -> tuple[int | None, str, bool]:
            self.calls += 1
            if self.calls == 1:
                entered.set()
                assert proceed.wait(5)
            return super().stop(**kwargs)

    backend = rs.LlamaServerResidentBackend(server=Path("unused"), process_factory=Process)  # type: ignore[arg-type]
    backend.exit_timeout_s = backend.unwind_timeout_s = 0.03
    script.SCRIPT.block_ready = True
    model = script._model(tmp_path)
    start, outcome = script._in_thread(lambda: backend.start(model, RuntimeParams()))
    cleanup: threading.Thread | None = None
    try:
        assert script.SCRIPT.ready_entered.wait(5)
        cleanup = threading.Thread(target=backend.cleanup, daemon=True)
        cleanup.start()
        assert entered.wait(5)
        second, stopped = script._in_thread(backend.stop)
        second.join(1)
        assert not second.is_alive() and "error" not in stopped
        assert not stopped["value"].cleanup_complete
        assert backend.state == "starting" and not backend._idle.is_set()
    finally:
        proceed.set()
        start.join(5)
        if cleanup is not None:
            cleanup.join(5)
        backend.cleanup()
    assert not start.is_alive() and isinstance(outcome.get("error"), rs.ResidentStartCancelled)
    assert backend.state == "idle" and release_evidence(backend).clean


@pytest.mark.parametrize("pair", build_pairs(), ids=lambda pair: pair.pair_id)
def test_answer_selection_preserves_detached_tokens_at_both_line_edges(pair: Any) -> None:
    for value in (pair.expected, str(int(pair.expected) - 1)):
        for operator in ("*", "**", "_", "__", "(*)", "**;", "***"):
            for raw in (f"{value} {operator}", f"{operator} {value}",
                        f"Answer: {value} {operator}", f"**Answer:** {value} {operator}",
                        f"Answer: {operator} {value}", f"**Answer:** {operator} {value}"):
                assert judge(raw, pair)[0] == UNSCORABLE, raw
    for raw in (f"**{pair.expected}**", f"`{pair.expected}`", f"__{pair.expected}__",
                f"**Here is the answer:**\n**{pair.expected}**"):
        assert judge(raw, pair)[0] == RIGHT, raw


@pytest.mark.parametrize("wrong", [False, True])
def test_edge_operators_cannot_become_native_chat_or_swa_correctness_evidence(wrong: bool) -> None:
    pairs = build_pairs()
    for operator in ("*", "**", "_", "(*)"):
        answers = {p.case.measured: f"{int(p.expected) - int(wrong)} {operator}" for p in pairs}

        class Native(SimulatedReuseEngine):
            def generate(self, prompt: str, **kwargs: Any) -> Any:
                reading = super().generate(prompt, **kwargs)
                if kwargs["cache_prompt"] and self._requests == 2 and self._variant == DEFAULT:
                    reading.text = answers[prompt]
                return reading

        record = evaluate(Native(oracle=answer_oracle(pairs), sliding_window=True),
            cases=tuple(p.case for p in pairs), variants=(DEFAULT, SWA_FULL), blocks=2,
            context_length=4096, configuration={}, run_id="operator-regression",
            power_check=False, cancellation=False)
        summary = correctness_summary([TrialResult(**t) for t in record["trials"]], pairs)
        assert summary["swa_full_comparison"]["interpretable"] is False
        assert summary["swa_full_comparison"]["comparable"] == {}
        for pair in pairs:
            assert summary["pairs"][f"default/{pair.pair_id}"]["partial_reuse_evidence"] == EVIDENCE_INCONCLUSIVE

        class Chat(SimulatedReuseEngine):
            def chat(self, messages: Any, **kwargs: Any) -> Any:
                reply = super().chat(messages, **kwargs)
                if kwargs.get("cache_prompt") is None and self._requests == 2:
                    # Read the question from the normal simulator answer, then
                    # replace the measured client only; matched references stay intact.
                    return replace(reply, text=f"{56 if wrong else 57} {operator}")
                return reply

        trials, stop, _ = run_chat_exposure(Chat(oracle=answer_oracle(pairs)), pairs,
            blocks=2, power=lambda: "ac", power_check=True)
        assert stop is None
        for item in chat_summary(trials, pairs)["pairs"].values():
            assert item["outcome"] == "unscorable for the client"


_DEFAULT_CHANGES = [
    ("effective_threads", 8), ("effective_threads", None), ("gpu_layers_offloaded", None),
    ("mmap_effective", False), ("flash_attention", True), ("cache_type_k", "q8_0"),
    ("cache_type_v", None), ("context_length", 8192), ("unchanged", None),
]


def _shown(base: BackendTelemetry, field: str, value: Any, changed: bool) -> BackendTelemetry:
    base = base.model_copy(update={"effective_threads": 6, "mmap_effective": True,
        "observed": {**base.observed, "flash_attention": False,
                     "cache_type_k": "f16", "cache_type_v": "f16"}})
    if not changed or field == "unchanged":
        return base
    if field in ("effective_threads", "mmap_effective", "gpu_layers_offloaded"):
        return base.model_copy(update={field: value})
    return base.model_copy(update={"observed": {**base.observed, field: value}})


@pytest.mark.parametrize("field,value", _DEFAULT_CHANGES)
def test_resident_recovery_preserves_every_previously_observed_setting(field: str, value: Any) -> None:
    class Adapter(RecoverableFakeAdapter):
        def telemetry(self) -> BackendTelemetry:
            return _shown(super().telemetry(), field, value, self.load_calls > 1)

    adapter = Adapter()
    workload = _workload(adapter)
    _provision(workload)
    config = workload.configuration().model_copy(deep=True)
    adapter.kill()
    record = workload.recover_target(reason="process_exited")
    assert record.recovered is (field == "unchanged")
    assert record.configuration_verified is (field == "unchanged")
    assert adapter.is_loaded is (field == "unchanged")
    assert workload.configuration() == config and config.threads is None
    if field != "unchanged":
        again = workload.recover_target(reason="process_exited")
        assert not again.attempted and not again.recovered and adapter.load_calls == 2
    adapter.cleanup()


@pytest.mark.parametrize("before", [None, 8])
def test_a_later_matching_request_does_not_repair_unverified_initial_recovery_evidence(before: int | None) -> None:
    class Adapter(RecoverableFakeAdapter):
        def telemetry(self) -> BackendTelemetry:
            return super().telemetry().model_copy(update={
                "effective_threads": before if self.load_calls == 1 else 6})

    adapter = Adapter()
    workload = _workload(adapter)
    workload.set_configuration(workload.configuration().model_copy(update={"threads": 6}))
    _provision(workload)
    adapter.kill()
    result = workload.recover_target(reason="process_exited")
    assert not result.recovered and not result.configuration_verified and not adapter.is_loaded


@pytest.mark.parametrize("field,value", _DEFAULT_CHANGES)
@pytest.mark.parametrize("failure", ["probe", "exception"])
def test_rollback_checkpoint_recovery_preserves_observed_defaults(
    field: str, value: Any, failure: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor, adapter, record, service = _setup()
    initial_pid = adapter.pid
    telemetry = adapter.telemetry
    monkeypatch.setattr(adapter, "telemetry", lambda: _shown(telemetry(), field, value, adapter.pid != initial_pid))
    if failure == "probe":
        adapter.text = "different answer"  # triggers the verification-failure recovery route
    else:
        original_probe = executor._probe
        calls = 0

        def probe() -> Any:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("post-apply probe failed")
            return original_probe()

        monkeypatch.setattr(executor, "_probe", probe)
    result = executor.execute_record(record, run_id="independent-review-defaults")
    assert result.recovery_verified is (field == "unchanged")
    assert result.status is (RollbackExecutionStatus.FAILED_RESTORED if field == "unchanged"
                             else RollbackExecutionStatus.FAILED_UNRECOVERED)
    assert result.checkpoint_configuration is not None and result.checkpoint_configuration.threads is None
    assert adapter.params.threads is None
