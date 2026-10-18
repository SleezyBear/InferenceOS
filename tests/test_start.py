"""Step 5: the everyday ``start`` command, and tune -> start end to end."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from inferenceos.cli import build_parser
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.tune import PowerPlan, TuneResult
from inferenceos.interfaces.backend_adapter import RuntimeParams
from inferenceos.selftest import sample_model
from inferenceos.tune.pipeline import Tuner
from inferenceos.tune.profiles import SetupStore
from inferenceos.tune.start import port_is_free, quick_check, serve_until_stopped
from tests.test_tune import CONTEXT, _machine, _settings, _tuner, layer_gain_responder


class FakeBackend:
    def __init__(self, *, fail: bool = False, dies_after: int | None = None) -> None:
        self.fail = fail
        self.dies_after = dies_after
        self.started_with: RuntimeParams | None = None
        self.running = False
        self.checks = 0
        self.stops = 0
        self.key_file: Path | None = None

    def start(self, model: ModelProfile, params: RuntimeParams) -> object:
        if self.fail:
            raise RuntimeError("llama-server failed to load")
        self.started_with = params
        self.running = True
        self.key_file = Path("/tmp/inferenceos-key-test.txt")
        return None

    def stop(self) -> object:
        self.stops += 1
        self.running = False
        self.key_file = None
        return None

    def is_running(self) -> bool:
        self.checks += 1
        if self.dies_after is not None and self.checks > self.dies_after:
            self.running = False
        return self.running

    @property
    def base_url(self) -> str | None:
        return "http://127.0.0.1:8080"

    @property
    def api_key_file(self) -> Path | None:
        return self.key_file


def _tuned(
    tmp_path: Path, power: PowerState = PowerState.AC
) -> tuple[Tuner, TuneResult, SetupStore]:
    tuner, _, _, store = _tuner(tmp_path, layer_gain_responder, answers=[True], power=power)
    result = tuner.run()
    return tuner, result, store


def test_tune_then_start_runs_the_approved_setup(tmp_path: Path) -> None:
    """Done when: the commands take a new model to a running, approved setup."""
    tuner, result, store = _tuned(tmp_path)
    assert result.outcome == "saved"
    check = quick_check(
        store=store,
        model_path=tuner.model_path,
        context_length=None,
        machine=_machine(_settings(tmp_path)),
    )
    assert check.can_start and check.setup is not None
    assert check.matched_power_source and check.changes == []
    backend = FakeBackend()
    lines: list[str] = []
    polls = iter([True, True, False])
    outcome = serve_until_stopped(
        setup=check.setup,
        model=sample_model(),
        backend=backend,
        say=lines.append,
        keep_running=lambda: next(polls),
        poll_s=0.0,
    )
    assert outcome.started and outcome.stopped_cleanly
    assert backend.started_with is not None
    assert backend.started_with.gpu_offload_layers == result.final_configuration.gpu_offload_layers
    assert backend.started_with.context_length == CONTEXT
    assert backend.stops == 1
    assert any("this computer only" in line for line in lines)
    assert any("Password for this run is in" in line for line in lines)


def test_start_reports_changes_and_starts_anyway(tmp_path: Path) -> None:
    tuner, _, store = _tuned(tmp_path)
    now = _machine(_settings(tmp_path)).model_copy(
        update={
            "program_version": "b12000",
            "power_plan": PowerPlan(plan_id="8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c", name="High"),
        }
    )
    check = quick_check(
        store=store, model_path=tuner.model_path, context_length=CONTEXT, machine=now
    )
    assert check.can_start
    assert any("llama.cpp version" in change for change in check.changes)
    assert any("power plan" in change for change in check.changes)
    assert any("tune" in note for note in check.notes)


def test_start_uses_the_other_power_setup_and_says_so(tmp_path: Path) -> None:
    tuner, _, store = _tuned(tmp_path, power=PowerState.AC)
    on_battery = _machine(_settings(tmp_path), PowerState.BATTERY)
    check = quick_check(
        store=store, model_path=tuner.model_path, context_length=None, machine=on_battery
    )
    assert check.can_start and not check.matched_power_source
    assert any("no saved setup for on battery" in note for note in check.notes)


def test_start_refuses_a_replaced_model_file(tmp_path: Path) -> None:
    tuner, _, store = _tuned(tmp_path)
    tuner.model_path.write_bytes(b"GGUF" + b"\2" * 64)  # same size, other contents
    check = quick_check(
        store=store,
        model_path=tuner.model_path,
        context_length=None,
        machine=_machine(_settings(tmp_path)),
    )
    assert not check.can_start
    assert any("contents changed" in problem for problem in check.problems)


def test_start_without_a_saved_setup_says_run_tune(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    model = tmp_path / "other.gguf"
    model.write_bytes(b"GGUF")
    check = quick_check(
        store=SetupStore(settings), model_path=model, context_length=None, machine=_machine(settings)
    )
    assert not check.can_start
    assert "run `tune` first" in check.problems[0]


def test_start_asks_which_context_when_there_are_several(tmp_path: Path) -> None:
    tuner, _, store = _tuned(tmp_path)
    saved = store.all()[0]
    store.save(
        saved.model_copy(
            update={
                "setup_id": "other",
                "context_length": 4096,
                "configuration": saved.configuration.model_copy(update={"context_length": 4096}),
            }
        )
    )
    check = quick_check(
        store=store,
        model_path=tuner.model_path,
        context_length=None,
        machine=_machine(_settings(tmp_path)),
    )
    assert not check.can_start and "--context" in check.problems[0]
    chosen = quick_check(
        store=store,
        model_path=tuner.model_path,
        context_length=4096,
        machine=_machine(_settings(tmp_path)),
    )
    assert chosen.setup is not None and chosen.setup.context_length == 4096


def test_start_reports_a_model_that_will_not_load(tmp_path: Path) -> None:
    _, _, store = _tuned(tmp_path)
    outcome = serve_until_stopped(
        setup=store.all()[0], model=sample_model(), backend=FakeBackend(fail=True), say=lambda _: None
    )
    assert not outcome.started and "did not start" in outcome.reason


def test_start_notices_the_model_stopping_on_its_own(tmp_path: Path) -> None:
    _, _, store = _tuned(tmp_path)
    backend = FakeBackend(dies_after=2)
    outcome = serve_until_stopped(
        setup=store.all()[0], model=sample_model(), backend=backend, say=lambda _: None, poll_s=0.0
    )
    assert outcome.started and outcome.reason == "the model stopped on its own"


def test_port_check_sees_a_busy_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        assert not port_is_free(port)


def test_tune_needs_a_context_size_and_start_does_not() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["tune", "--model", "m.gguf"])
    args = parser.parse_args(["tune", "--model", "m.gguf", "--context", "4096", "--effort", "minimize_cost"])
    assert args.context == 4096 and args.effort == "minimize_cost" and not args.yes
    start = parser.parse_args(["start", "--model", "m.gguf"])
    assert start.context is None and start.port is None
