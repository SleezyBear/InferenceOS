"""Caller-level Ctrl+C during checkpoint serialization, entirely fake engine IO."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from inferenceos.tune import lookup_answer_publication as publication
from inferenceos.tune import lookup_answers_runner as runner
from inferenceos.tune.lookup_answers_sim import AnswerSimulator, FakeClock, fixture_scope
from inferenceos.tune.lookup_fit_sim import fixture_manifest


@pytest.mark.parametrize("stage", ("first-checkpoint", "final-checkpoint"))
def test_interrupted_checkpoint_retains_raw_diagnostics_and_same_interrupt_without_receipt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    clock = FakeClock()
    scope = fixture_scope()
    engine = AnswerSimulator(clock)
    retry = Mock(return_value=None)
    # The simulator's FitSimulator remains the sole lifecycle owner. Futility
    # closes the six cells quickly, retaining all 48 scheduled fabricated rests.
    answer = engine.answer
    def wrong(body: Any, receive: Any) -> None:
        answer(body, lambda chunk: receive(dict(chunk, content="wrong") if chunk.get("content") else chunk))
    monkeypatch.setattr(engine, "answer", wrong)
    monkeypatch.setattr(engine, "retry_held", retry, raising=False)
    preflight = SimpleNamespace(check=lambda **kwargs: None, preservation=lambda: dict(fixture_only=True))
    inputs = SimpleNamespace(manifest=SimpleNamespace(models=fixture_manifest().models),
                             derived_scope=lambda: scope, provenance=lambda: dict(fixture_only=True))
    monkeypatch.setattr(runner, "AnswerPreflight", Mock(return_value=preflight))
    monkeypatch.setattr(runner, "NativeAnswerEngine", Mock(return_value=engine))
    original = KeyboardInterrupt("original " + stage)
    encode = publication.encoded
    interrupted: list[str] = []
    def interrupt_packet(value: Any) -> bytes:
        # Throw from INSIDE the real checkpoint, after it owns the raw core and
        # before journal bytes have been written. Do not replace the callback.
        if isinstance(value, dict) and set(value) == {"sequence", "previous_sha256", "event"}:
            meta = value["event"]["meta"]
            target = value["sequence"] == 0 if stage == "first-checkpoint" else meta["status"] == "done"
            if target:
                interrupted.append(meta["status"])
                raise original
        return encode(value)
    monkeypatch.setattr(publication, "encoded", interrupt_packet)
    output = tmp_path / "interrupted.json"
    construct = Mock(return_value=None)
    with patch("subprocess.Popen", side_effect=AssertionError("zero native processes")):
        with pytest.raises(KeyboardInterrupt) as caught:
            runner.run_native(root=tmp_path, settings=object(), rules=tmp_path / "never-read.json",
                              rules_sha="0"*64, out=output, owner_authorized=True,
                              preflight_factory=Mock(side_effect=AssertionError("fake caller preflight only")),
                              fit_engine_factory=construct, power=lambda: "ac", clock=clock,
                              rest=lambda seconds: clock.advance(seconds*1_000_000_000),
                              input_loader=lambda *args: inputs, execution_id=stage)
    assert caught.value is original and interrupted
    count = 1 if stage == "first-checkpoint" else 6
    record = json.loads(output.read_bytes())
    assert record["status"] == "interrupted"
    assert len(record["core"]["rows"]) == count
    assert engine.answers == engine.sim.starts == engine.sim.releases == count
    assert not engine.sim.active
    if stage == "final-checkpoint":
        assert len(record["core"]["rests"]) == 48
    assert all(r["cleanup"] == dict(process_gone=True, credential_removed=True, problem=None)
               for r in record["core"]["rows"])
    construct.assert_called_once()
    retry.assert_called_once()  # delegated owner only, no new release attempt
    assert not output.with_name(output.name + ".closure.json").exists()
    assert record["actual_qualification"] is record["automatic_admission"] is record["deployment_allowed"] is False


@pytest.mark.parametrize("stage", ("first-checkpoint", "final-checkpoint"))
def test_journal_close_error_never_replaces_original_checkpoint_interrupt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    close = publication.OwnedAnswerOutput.close
    def fail_close(owned: publication.OwnedAnswerOutput) -> None:
        close(owned)
        raise OSError("ordinary journal close failure")
    monkeypatch.setattr(publication.OwnedAnswerOutput, "close", fail_close)
    test_interrupted_checkpoint_retains_raw_diagnostics_and_same_interrupt_without_receipt(tmp_path, monkeypatch, stage)


@pytest.mark.parametrize("close_interrupt", (False, True))
def test_final_journal_close_error_uses_the_existing_failure_owner(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close_interrupt: bool) -> None:
    primary = RuntimeError("initial power observer failed")
    later = KeyboardInterrupt("new interrupt during close")
    close = publication.OwnedAnswerOutput.close
    def fail_close(owned: publication.OwnedAnswerOutput) -> None:
        close(owned)
        if close_interrupt:raise later
        raise OSError("ordinary journal close failure")
    def power() -> str:raise primary
    monkeypatch.setattr(publication.OwnedAnswerOutput, "close", fail_close)
    construct = Mock(side_effect=AssertionError("no backend construction"))
    def invoke() -> int:
        return runner.run_native(root=tmp_path, settings=object(), rules=tmp_path/"never-read.json",
                                 rules_sha="0"*64, out=tmp_path/"failed.json", owner_authorized=True,
                                 preflight_factory=construct, fit_engine_factory=construct,
                                 power=power, input_loader=lambda *args: None)
    with patch("subprocess.Popen", side_effect=AssertionError("zero native processes")):
        if close_interrupt:
            with pytest.raises(KeyboardInterrupt) as caught:invoke()
            assert caught.value is later and later.__context__ is primary
        else:assert invoke()==7
    construct.assert_not_called()
    record=json.loads((tmp_path/"failed.json").read_bytes())
    assert record["error"]==dict(type="RuntimeError", message=str(primary))
    assert not record["actual_qualification"]
