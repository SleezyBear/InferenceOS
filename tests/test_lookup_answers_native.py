"""Lifecycle composition and bounded observations; never create a real backend."""
from __future__ import annotations

from pathlib import Path
import importlib
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from inferenceos.tune import lookup_answers_native as native
from inferenceos.tune.lookup_answer_transport import AnswerCancelled, AnswerTimeout, bounded_operation
from inferenceos.tune.lookup_answers import request_body
from inferenceos.tune.lookup_corpus import build_corpus
from inferenceos.tune.lookup_fit import FitRefused
from inferenceos.tune.lookup_fit_sim import fixture_manifest


class FitDouble:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.cancelled = threading.Event()
        self.owed = False
        self.calls: list[Any] = []
        self.backend = SimpleNamespace(is_active=lambda: self.owed, unconfirmed_pid=None,
            api_key_file=None, undeleted_key_files=[], base_url="http://127.0.0.1:17001")

    def start(self, model: Any, arm: str, *, deadline: float | None = None) -> None:
        assert deadline is not None
        self.calls.append(("start", model.label, arm))
        self.owed = True
        key = self.folder / "fake-native-password.txt"
        key.write_text("fake-key", encoding="ascii")
        self.backend.api_key_file = key

    def observe(self) -> dict[str, Any]:
        self.calls.append(("observe",))
        return {"fake": True}

    def tokenize(self, text: str, *, add_special: bool) -> list[int]:
        self.calls.append(("tokenize", text, add_special))
        return [1, 2]

    def release(self, *, abort: bool) -> dict[str, Any]:
        self.calls.append(("release", abort))
        if self.backend.api_key_file is not None:
            self.backend.api_key_file.unlink(missing_ok=True)
        self.backend.api_key_file = None
        self.owed = False
        return dict(process_gone=True, credential_removed=True, problem=None)

    def retry_held(self) -> dict[str, Any] | None:
        self.calls.append(("retry_held",))
        if self.owed:
            return self.release(abort=True)
        return None


def engine(tmp_path: Path, **kwargs: Any) -> tuple[native.NativeAnswerEngine, FitDouble]:
    fit = FitDouble(tmp_path)
    return native.NativeAnswerEngine(fit, fixture_manifest().models, **kwargs), fit


def test_full_adapter_route_one_request_no_priming_and_existing_owner_only(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, fit = engine(tmp_path)
    requests: list[Any] = []

    def answer(url: str, key: str, body: Any, chunk: Any, **kwargs: Any) -> None:
        requests.append((url, key, body, kwargs))
        chunk({"stop": True})

    monkeypatch.setattr(native, "stream_completion", answer)
    adapter.start("llama3b", "control")
    assert adapter.observe() == {"fake": True}
    assert adapter.tokenize("prompt", add_special=True) == [1, 2]
    replies: list[Any] = []
    body = request_body(build_corpus().cases[0])
    adapter.answer(body, replies.append)
    adapter.observe()
    assert adapter.release(abort=False) == dict(process_gone=True, credential_removed=True, problem=None)
    assert [call[0] for call in fit.calls] == ["start", "observe", "tokenize", "observe", "release"]
    assert len(requests) == 1 and requests[0][2] == body
    assert requests[0][3]["cancelled"] is fit.cancelled
    assert 0 < requests[0][3]["timeout_s"] <= 120
    assert replies == [{"stop": True}] and not fit.cancelled.is_set()
    adapter.start("gemma1b", "lookup")
    adapter.release(abort=False)
    assert len(requests) == 1


@pytest.mark.parametrize("leftover", ["owed", "active", "pid", "key", "undeleted"])
def test_any_unreleased_resource_blocks_next_start(tmp_path: Path, leftover: str) -> None:
    adapter, fit = engine(tmp_path)
    if leftover == "owed": fit.owed = True
    if leftover == "active": fit.backend.is_active = lambda: True
    if leftover == "pid": fit.backend.unconfirmed_pid = 123
    if leftover == "key": fit.backend.api_key_file = tmp_path / "held.txt"
    if leftover == "undeleted": fit.backend.undeleted_key_files = [tmp_path / "held.txt"]
    with pytest.raises(FitRefused, match="remains owned"):
        adapter.start("llama3b", "control")
    assert fit.calls == [] and fit.cancelled.is_set()


@pytest.mark.parametrize("operation", ["observe", "tokenize", "answer"])
def test_no_native_operation_without_current_load(tmp_path: Path, operation: str) -> None:
    adapter, fit = engine(tmp_path)
    with pytest.raises(AnswerCancelled):
        if operation == "observe": adapter.observe()
        elif operation == "tokenize": adapter.tokenize("prompt", add_special=True)
        else: adapter.answer(request_body(build_corpus().cases[0]), lambda chunk: None)
    assert fit.calls == []


@pytest.mark.parametrize("operation", ["observe", "tokenize"])
def test_total_observation_deadline_rejects_late_result_and_next_start(tmp_path: Path, operation: str) -> None:
    adapter, fit = engine(tmp_path, observe_timeout_s=.06, tokenize_timeout_s=.06)
    adapter.start("llama3b", "control")
    entered, proceed, exited = threading.Event(), threading.Event(), threading.Event()

    def delayed(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        proceed.wait(2)
        exited.set()
        return {} if operation == "observe" else [1]

    setattr(fit, operation, delayed)
    began = time.monotonic()
    try:
        with pytest.raises(AnswerTimeout):
            if operation == "observe": adapter.observe()
            else: adapter.tokenize("prompt", add_special=True)
        assert entered.is_set() and time.monotonic() - began < .8
        assert fit.cancelled.is_set()
        adapter.release(abort=True)
        with pytest.raises(FitRefused):
            adapter.start("gemma1b", "lookup")
        assert sum(call[0] == "start" for call in fit.calls) == 1
    finally:
        proceed.set()
    assert exited.wait(1)


@pytest.mark.parametrize("operation", ["start", "observe", "tokenize", "stream", "release", "retry"])
@pytest.mark.parametrize("kind", ["error", "interrupt"])
def test_original_exception_and_single_owner_are_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           operation: str, kind: str) -> None:
    adapter, fit = engine(tmp_path)
    original: BaseException = RuntimeError("same failure") if kind == "error" else KeyboardInterrupt("same interrupt")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise original

    if operation != "start": adapter.start("llama3b", "control")
    if operation == "stream": monkeypatch.setattr(native, "stream_completion", fail)
    else: setattr(fit, "retry_held" if operation == "retry" else operation, fail)
    with pytest.raises(type(original)) as caught:
        if operation == "start": adapter.start("llama3b", "control")
        elif operation == "observe": adapter.observe()
        elif operation == "tokenize": adapter.tokenize("prompt", add_special=True)
        elif operation == "stream": adapter.answer(request_body(build_corpus().cases[0]), lambda chunk: None)
        elif operation == "release": adapter.release(abort=False)
        else: adapter.retry_held()
    assert caught.value is original and fit.cancelled.is_set()
    # The adapter never invents a recovery/kill/cleanup owner on failure.
    assert not any(call[0] in ("release", "retry_held") for call in fit.calls)


@pytest.mark.parametrize("operation", ["answer", "tokenize"])
def test_no_second_measurement_or_later_tokenization_within_same_load(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    adapter, fit = engine(tmp_path)
    seen: list[Any] = []
    monkeypatch.setattr(native, "stream_completion", lambda *args, **kwargs: seen.append(args))
    adapter.start("llama3b", "control")
    adapter.answer(request_body(build_corpus().cases[0]), lambda chunk: None)
    with pytest.raises(FitRefused):
        if operation == "answer": adapter.answer(request_body(build_corpus().cases[0]), lambda chunk: None)
        else: adapter.tokenize("again", add_special=True)
    assert len(seen) == 1 and fit.cancelled.is_set()


@pytest.mark.parametrize("receipt", [
    dict(process_gone=False, credential_removed=True, problem=None),
    dict(process_gone=True, credential_removed=False, problem=None),
    dict(process_gone=True, credential_removed=True, problem="held"), {}, None])
def test_unconfirmed_release_permanently_closes_admission(tmp_path: Path, receipt: Any) -> None:
    adapter, fit = engine(tmp_path)
    adapter.start("llama3b", "control")
    fit.release = lambda **kwargs: receipt  # type: ignore[method-assign]
    assert adapter.release(abort=False) is receipt
    assert fit.cancelled.is_set()
    with pytest.raises(FitRefused):
        adapter.start("gemma1b", "lookup")


def test_cancel_and_retry_are_forwarded_to_existing_owner(tmp_path: Path) -> None:
    adapter, fit = engine(tmp_path)
    adapter.start("llama3b", "control")
    adapter.cancel()
    assert adapter.retry_held() == dict(process_gone=True, credential_removed=True, problem=None)
    assert fit.calls == [("start", "llama3b", "control"), ("retry_held",), ("release", True)]
    with pytest.raises(FitRefused):
        adapter.start("llama3b", "control")


def test_concurrent_release_cancels_pending_read_and_late_native_dispatch(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, fit = engine(tmp_path)
    adapter.start("llama3b", "control")
    entered, proceed, stopped = threading.Event(), threading.Event(), threading.Event()
    original: list[BaseException] = []
    path = fit.backend.api_key_file

    class DelayedPath:
        def open(self, mode: str) -> Any:
            entered.set()
            proceed.wait(2)
            return path.open(mode)

        def unlink(self, *, missing_ok: bool) -> None:
            path.unlink(missing_ok=missing_ok)

    fit.backend.api_key_file = DelayedPath()
    requests: list[Any] = []
    monkeypatch.setattr(native, "stream_completion", lambda *args, **kwargs: requests.append(args))

    def answer() -> None:
        try:
            adapter.answer(request_body(build_corpus().cases[0]), lambda chunk: None)
        except BaseException as exc:
            original.append(exc)
        finally:
            stopped.set()

    worker = threading.Thread(target=answer)
    worker.start()
    assert entered.wait(1)
    try:
        assert adapter.release(abort=False)["process_gone"] is True
        assert stopped.wait(1) and fit.cancelled.is_set()
        assert fit.calls[-1] == ("release", True)
        assert len(original) == 1 and isinstance(original[0], AnswerCancelled)
    finally:
        proceed.set()
        worker.join(2)
    assert requests == []


@pytest.mark.parametrize("value", [[], [True], [-1], [1.5], "tokens", None])
def test_malformed_native_tokenizer_reply_refused(tmp_path: Path, value: Any) -> None:
    adapter, fit = engine(tmp_path)
    adapter.start("llama3b", "control")
    fit.tokenize = lambda *args, **kwargs: value  # type: ignore[method-assign]
    with pytest.raises(FitRefused):
        adapter.tokenize("prompt", add_special=True)
    assert fit.cancelled.is_set()


@pytest.mark.parametrize("key", [b"", b"a" * 4097, b"\xff"])
def test_invalid_password_never_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: bytes) -> None:
    adapter, fit = engine(tmp_path)
    adapter.start("llama3b", "control")
    fit.backend.api_key_file.write_bytes(key)
    sent: list[Any] = []

    def stream(*args: Any, **kwargs: Any) -> None:
        # Retain the production transport's credential validation without opening
        # a socket. An invalid key is not a generation request.
        if not args[1]: raise FitRefused("empty key")
        sent.append(args)

    monkeypatch.setattr(native, "stream_completion", stream)
    with pytest.raises((FitRefused, UnicodeError)):
        adapter.answer(request_body(build_corpus().cases[0]), lambda chunk: None)
    assert sent == [] and fit.cancelled.is_set()


@pytest.mark.parametrize("timeout", [0, -1, 31, float("inf"), float("nan"), True])
def test_registered_time_limits_cannot_be_extended(tmp_path: Path, timeout: Any) -> None:
    with pytest.raises(ValueError):
        engine(tmp_path, observe_timeout_s=timeout)


def test_bounded_read_control_keeps_late_interruption_as_original_object() -> None:
    cancelled = threading.Event()
    original = KeyboardInterrupt("same")

    def operation() -> None:
        raise original

    with pytest.raises(KeyboardInterrupt) as caught:
        bounded_operation(operation, timeout_s=1, cancelled=cancelled)
    assert caught.value is original and cancelled.is_set()


@pytest.mark.parametrize("late", [False, True])
def test_real_fit_observer_composition_has_one_total_deadline(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, late: bool) -> None:
    # Invoke the actual composed observer, whose individual old 30 s subreads
    # must not multiply Stage B's single ready-observation allowance.
    from inferenceos.tune.lookup_fit_sim import FitSimulator, fixture_machine
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    owner_module = importlib.import_module("collect_lookup_fit")
    manifest = fixture_manifest()
    sim = FitSimulator(manifest)
    sim.start(manifest.models[0], "control")
    snapshot = sim.observe()
    password = tmp_path / "fake-key.txt"
    password.write_text("key", encoding="ascii")
    delays_finished = threading.Event()

    def startup() -> Any:
        if late: time.sleep(.035)
        return dict(snapshot["startup"])

    backend = SimpleNamespace(identity=SimpleNamespace(pid=10001, create_time=1001), pid=10001,
        startup_log_snapshot=startup, base_url="http://127.0.0.1:17001", api_key_file=password,
        last_command=snapshot["command"], engine_settings=lambda: {}, is_active=lambda: True,
        unconfirmed_pid=None, undeleted_key_files=[])

    def inventory() -> Any:
        if late: time.sleep(.035)
        delays_finished.set()
        return fixture_machine()

    preflight = SimpleNamespace(selected=Path(snapshot["selected_executable"]))
    owner = owner_module.LiveFitEngine(preflight, backend_factory=lambda: backend, inventory=inventory)
    owner.owed = True
    monkeypatch.setattr(owner_module, "process_image", lambda pid: dict(snapshot["image"]))
    monkeypatch.setattr(owner_module, "inspect_json", lambda *args, **kwargs: snapshot["slots"])
    adapter = native.NativeAnswerEngine(owner, manifest.models, observe_timeout_s=.05 if late else 2)
    if late:
        with pytest.raises(AnswerTimeout):
            adapter.observe()
        assert owner.cancelled.is_set()
        assert delays_finished.wait(1)
    else:
        observed = adapter.observe()
        assert observed["process_current"] is True
        assert observed["command"][0] == snapshot["command"][0]
        assert not owner.cancelled.is_set()


class StartClock:
    def __init__(self) -> None:
        self.seconds = 1.0

    def monotonic(self) -> float:
        return self.seconds

    def ns(self) -> int:
        return round(self.seconds*1_000_000_000)

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class StartBackend:
    """No process creation: exercise the production owner's atomic predicate."""
    def __init__(self) -> None:
        self.active = False
        self.unconfirmed_pid: int | None = None
        self.api_key_file: Path | None = None
        self.undeleted_key_files: list[Path] = []
        self.calls: list[str] = []
        self.delay: Any = lambda: None
        self.current: Any = None

    def is_active(self) -> bool:
        return self.active

    def start_if_current(self, model: Any, params: Any, *, current: Any) -> None:
        self.calls.append("start-enter")
        self.current = current
        self.delay()
        if not current():
            self.calls.append("admission-refused")
            raise FitRefused("expired atomic native admission")
        self.calls.append("process-start")
        self.active = True

    def halt(self) -> None:
        self.calls.append("halt")
        self.active = False

    def stop(self) -> None:
        self.calls.append("stop")
        self.active = False


def actual_start_owner(monkeypatch: pytest.MonkeyPatch, backend: StartBackend,
                       executable: Any = lambda *args: None, check: Any = lambda: None) -> Any:
    scripts = Path(__file__).resolve().parents[1]/"scripts"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("collect_lookup_fit")
    preflight = SimpleNamespace(selected=Path("fake-only-server.exe"),
        inputs=SimpleNamespace(executable=executable), settings=SimpleNamespace(llama_cpp_dir="fake"),
        check=check, models={model.label: object() for model in fixture_manifest().models})
    return module.LiveFitEngine(preflight, backend_factory=lambda: backend)


@pytest.mark.parametrize("arm", ["control", "lookup"])
@pytest.mark.parametrize("delay,passes", [(0, True), (61, True), (62, False), (64, False), (119, False)])
def test_production_start_checks_and_atomic_admission_share_absolute_deadline(
        monkeypatch: pytest.MonkeyPatch, arm: str, delay: float, passes: bool) -> None:
    # The independent reproduction's 29+29+119 must never return successfully.
    # In particular, a process delayed to second 122 must not be admitted under
    # the old, newly issued second-178 deadline.
    clock = StartClock()
    backend = StartBackend()
    backend.delay = lambda: clock.advance(delay)
    owner = actual_start_owner(monkeypatch, backend,
        executable=lambda *args: clock.advance(29), check=lambda: clock.advance(29))
    module = importlib.import_module("collect_lookup_fit")
    real_read = module.bounded_read
    allowances: list[float] = []

    def measured(call: Any, *, timeout_s: float) -> Any:
        allowances.append(timeout_s)
        return real_read(call, timeout_s=timeout_s)

    monkeypatch.setattr(module, "bounded_read", measured)
    monkeypatch.setattr(native.time, "monotonic", clock.monotonic)
    adapter = native.NativeAnswerEngine(owner, fixture_manifest().models)
    if passes:
        adapter.start("llama3b", arm)
        assert backend.calls == ["start-enter", "process-start"]
        assert not owner.cancelled.is_set()
        assert adapter.release(abort=False)["process_gone"] is True
    else:
        with pytest.raises((TimeoutError, FitRefused)):
            adapter.start("llama3b", arm)
        assert backend.calls == ["start-enter", "admission-refused"]
        assert owner.cancelled.is_set() and owner.owed is True
        assert adapter.release(abort=True)["process_gone"] is True
        with pytest.raises(FitRefused):
            adapter.start("gemma1b", arm)
    assert allowances == [30, 30, 62]


@pytest.mark.parametrize("boundary", ["before-checks", "between-checks", "before-start"])
def test_exhaustion_at_each_start_boundary_prevents_backend_dispatch(
        monkeypatch: pytest.MonkeyPatch, boundary: str) -> None:
    clock = StartClock()
    backend = StartBackend()
    checks: list[str] = []

    def executable(*args: Any) -> None:
        checks.append("executable")
        clock.advance(120 if boundary == "between-checks" else 29)

    def check() -> None:
        checks.append("inputs")
        clock.advance(91)

    owner = actual_start_owner(monkeypatch, backend, executable, check)
    module = importlib.import_module("collect_lookup_fit")
    # Isolate the owner's cross-stage checks from bounded_read's own per-read
    # defense. This deliberately faulty read double permits late returns.
    monkeypatch.setattr(module, "bounded_read", lambda call, **kwargs: call())
    monkeypatch.setattr(native.time, "monotonic", clock.monotonic)
    with pytest.raises(TimeoutError):
        owner.start(fixture_manifest().models[0], "control", deadline=clock.seconds if boundary == "before-checks" else clock.seconds+120)
    assert owner.cancelled.is_set() and not owner.owed and backend.calls == []
    assert checks == ([] if boundary == "before-checks" else ["executable"] if boundary == "between-checks" else ["executable", "inputs"])


@pytest.mark.parametrize("kind", ["error", "interrupt"])
@pytest.mark.parametrize("stage", ["executable", "inputs", "startup"])
def test_complete_start_preserves_original_failure_and_ownership(
        monkeypatch: pytest.MonkeyPatch, kind: str, stage: str) -> None:
    backend = StartBackend()
    original = RuntimeError("same original") if kind == "error" else KeyboardInterrupt("same original")

    def fail(*args: Any) -> None:
        raise original

    owner = actual_start_owner(monkeypatch, backend,
        executable=fail if stage == "executable" else lambda *args: None,
        check=fail if stage == "inputs" else lambda: None)
    if stage == "startup":
        backend.delay = fail
    adapter = native.NativeAnswerEngine(owner, fixture_manifest().models)
    with pytest.raises(type(original)) as caught:
        adapter.start("llama3b", "control")
    assert caught.value is original and owner.cancelled.is_set()
    assert owner.owed is (stage == "startup")
    assert not backend.active and "halt" not in backend.calls
    adapter.release(abort=True)
    with pytest.raises(FitRefused):
        adapter.start("gemma1b", "control")


@pytest.mark.parametrize("stage", ["executable", "inputs", "startup"])
def test_blocked_start_is_cancelled_and_existing_owner_refuses_late_process_admission(
        monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    backend = StartBackend()
    entered, proceed, exited = threading.Event(), threading.Event(), threading.Event()

    def blocked(*args: Any) -> None:
        entered.set()
        try:
            proceed.wait(2)
        finally:
            exited.set()

    owner = actual_start_owner(monkeypatch, backend,
        executable=blocked if stage == "executable" else lambda *args: None,
        check=blocked if stage == "inputs" else lambda: None)
    if stage == "startup":
        backend.delay = blocked
    adapter = native.NativeAnswerEngine(owner, fixture_manifest().models, start_timeout_s=.06)
    began = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            adapter.start("llama3b", "control")
        assert entered.is_set() and time.monotonic()-began < .8
        assert owner.cancelled.is_set()
        assert adapter.release(abort=True)["process_gone"] is True
        with pytest.raises(FitRefused):
            adapter.start("gemma1b", "lookup")
    finally:
        proceed.set()
    assert exited.wait(1)
    if backend.current is not None:
        assert backend.current() is False
    assert not backend.active and "process-start" not in backend.calls


@pytest.mark.parametrize("timeout", [0, -1, 120.001, True, float("nan"), float("inf")])
def test_native_complete_start_bound_cannot_be_extended(tmp_path: Path, timeout: Any) -> None:
    with pytest.raises(ValueError):
        engine(tmp_path, start_timeout_s=timeout)


@pytest.mark.parametrize("delay", [64, 119])
def test_actual_owner_adapter_and_collector_stop_late_start_before_any_answer(
        monkeypatch: pytest.MonkeyPatch, delay: float) -> None:
    from inferenceos.tune.lookup_answers_collect import collect_answers
    from inferenceos.tune.lookup_answers_sim import fixture_scope
    clock = StartClock()
    backend = StartBackend()
    backend.delay = lambda: clock.advance(delay)
    owner = actual_start_owner(monkeypatch, backend,
        executable=lambda *args: clock.advance(29), check=lambda: clock.advance(29))
    monkeypatch.setattr(native.time, "monotonic", clock.monotonic)
    adapter = native.NativeAnswerEngine(owner, fixture_manifest().models)
    observed: list[Any] = []
    monkeypatch.setattr(adapter, "observe", lambda: observed.append("observe"))
    record = collect_answers(adapter, fixture_scope(), run_id="late-start-regression", clock=clock.ns,
        power=lambda: "ac", guard=lambda: None, rest=lambda seconds: clock.advance(seconds),
        checkpoint=lambda record: None, simulated=True)
    assert record["status"] == "stopped" and len(record["rows"]) == 1
    assert record["rows"][0]["errors"] and record["rows"][0]["cleanup"]["process_gone"] is True
    assert owner.cancelled.is_set() and observed == []
    assert backend.calls == ["start-enter", "admission-refused", "halt"]


def test_resource_admission_inspection_consumes_the_same_owner_start_budget(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = StartClock()
    backend = StartBackend()
    reads: list[float] = []

    def inspected_active() -> bool:
        reads.append(clock.seconds)
        clock.advance(10)
        return backend.active

    backend.is_active = inspected_active  # type: ignore[method-assign]
    backend.delay = lambda: clock.advance(52)
    owner = actual_start_owner(monkeypatch, backend,
        executable=lambda *args: clock.advance(29), check=lambda: clock.advance(29))
    monkeypatch.setattr(native.time, "monotonic", clock.monotonic)
    adapter = native.NativeAnswerEngine(owner, fixture_manifest().models)
    with pytest.raises((TimeoutError, FitRefused)):
        adapter.start("llama3b", "control")
    assert reads == [1.0] and clock.seconds == 121
    assert backend.calls == ["start-enter", "admission-refused"]
    assert owner.cancelled.is_set() and not backend.active


def test_blocked_resource_admission_inspection_times_out_without_starting_owner(
        monkeypatch: pytest.MonkeyPatch) -> None:
    backend = StartBackend()
    entered, proceed, exited = threading.Event(), threading.Event(), threading.Event()

    def blocked_active() -> bool:
        entered.set()
        try:
            proceed.wait(2)
            return False
        finally:
            exited.set()

    backend.is_active = blocked_active  # type: ignore[method-assign]
    owner = actual_start_owner(monkeypatch, backend)
    adapter = native.NativeAnswerEngine(owner, fixture_manifest().models, start_timeout_s=.06)
    began = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            adapter.start("llama3b", "control")
        assert entered.is_set() and time.monotonic()-began < .8
        assert owner.cancelled.is_set() and not owner.owed and not backend.calls
    finally:
        proceed.set()
    assert exited.wait(1) and not backend.calls
