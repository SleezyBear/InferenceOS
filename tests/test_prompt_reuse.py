"""Prompt reuse (16 Oct 2026, revised after review): the development evaluation, the
engine plumbing, and the separation of fresh and cached readings in tune's request
identity.

Simulated engines establish correctness only. Nothing here measures hardware gains.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from inferenceos.tune.prompt_reuse import (
    CACHE_REUSE_DISABLED_WARNINGS,
    CUT_BY_CLIENT,
    DEFAULT_VARIANTS,
    DONE,
    ENDED_EARLY,
    FRESH,
    INCOMPLETE,
    INTERRUPTED,
    REUSE,
    STOPPED_CLEANUP,
    STOPPED_POWER,
    STOPPED_VALIDITY,
    Abandoned,
    EngineVariant,
    TrialResult,
    build_cases,
    cache_reuse_activation,
    cancellation_check,
    common_prefix,
    evaluate,
    judge_block,
    plan_trials,
    reuse_bound,
    run_trial,
    shifted_reuse,
    summarize,
    work_evidence,
)
from inferenceos.tune.prompt_reuse_sim import SimFaults, SimulatedReuseEngine

CASES = build_cases(4)
CASE = {c.case_id: c for c in CASES}
DEFAULT = DEFAULT_VARIANTS[0]
#: The short test cases' shared ends are shorter than 256 pieces: a smaller chunk here.
SMALL_CHUNK = EngineVariant("cache-reuse-16", cache_reuse=16, cases=("edited_middle", "deleted_middle"))
VARIANTS = (DEFAULT, SMALL_CHUNK)


def _run(faults: SimFaults | None = None, *, blocks: int = 2, engine: Any = None, **kwargs: Any) -> dict[str, Any]:
    sim = engine or SimulatedReuseEngine(faults=faults or SimFaults())
    return evaluate(sim, cases=CASES, variants=VARIANTS, blocks=blocks, context_length=4096,
                    configuration={"context_length": 4096}, run_id="test", power_check=False, **kwargs)


def _expected(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in summary.values() if item["role"] == "reuse"]


def _pair(case: str = "shared_prefix", faults: SimFaults | None = None) -> tuple[TrialResult, TrialResult]:
    sim = SimulatedReuseEngine(faults=faults or SimFaults())
    fresh = run_trial(sim, CASE[case], DEFAULT, FRESH, block=0, context_length=4096, configuration_sha256="c")
    reuse = run_trial(sim, CASE[case], DEFAULT, REUSE, block=0, context_length=4096, configuration_sha256="c")
    return fresh, reuse


# --------------------------------------------------------------------------- #
# The evaluation on a well-behaved simulated engine                           #
# --------------------------------------------------------------------------- #
def test_a_clean_engine_gives_usable_reuse_evidence_and_a_quiet_control() -> None:
    record = _run()
    assert record["status"] == DONE and record["problems"] == [] and record["stop_reason"] == ""
    assert record["planned_trials"] == len(record["trials"])
    for item in _expected(record["summary"]):
        if item["case"] == "edited_middle" and item["variant"] == SMALL_CHUNK.variant_id:
            continue
        assert item["clean"] and item["fresh_arm_stayed_fresh"] is True
        assert item["mean_relative_ttft_gain"] > 0.0
        assert all(item["answers_identical"])
    control = record["summary"]["default/unrelated"]
    assert control["role"] == "isolation control" and control["clean"]
    assert abs(control["mean_relative_ttft_gain"]) < 0.05  # nothing to reuse: no gain
    assert record["cancellation"][0]["verdict"] == "passed"


def test_shifted_reuse_follows_the_engines_chunk_search() -> None:
    # b11139: the new-prompt cursor stays at the first mismatch while the old-work cursor
    # scans forward. A deletion lines up again; a replacement never does.
    head, removed, tail = [1, 2, 3], [40, 41, 42, 43], [9, 8, 7, 6, 5, 4]
    assert shifted_reuse(head + removed + tail, head + tail, 3) == len(tail)
    assert shifted_reuse(head + removed + tail, head + [50, 51, 52, 53] + tail, 3) == 0
    assert shifted_reuse(head + removed + tail, head + tail, 7) == 0  # shorter than the chunk


def test_the_deleted_middle_control_shows_shifted_reuse_and_the_replacement_does_not() -> None:
    summary = _run()["summary"]
    on, off = summary["cache-reuse-16/deleted_middle"], summary["default/deleted_middle"]
    assert on["clean"] and off["clean"]
    assert min(on["cached_pieces"]) > max(off["cached_pieces"])
    edited_on, edited_off = summary["cache-reuse-16/edited_middle"], summary["default/edited_middle"]
    assert edited_on["cached_pieces"] == edited_off["cached_pieces"]  # nothing shiftable by the engine's rule


def test_the_shifted_control_is_shiftable_at_the_real_chunk_size() -> None:
    # On the full-length cases (simulated word pieces), 256-piece chunks are shiftable for
    # the deletion and not for the replacement. Real tokenizers are checked on hardware.
    full = {c.case_id: c for c in build_cases()}
    sim = SimulatedReuseEngine()
    sim.start(DEFAULT)
    deleted = full["deleted_middle"]
    edited = full["edited_middle"]
    assert shifted_reuse(sim.tokenize(deleted.prime), sim.tokenize(deleted.measured), 256) >= 256
    assert shifted_reuse(sim.tokenize(edited.prime), sim.tokenize(edited.measured), 256) == 0
    sim.stop()


def test_every_trial_runs_on_its_own_load_and_is_stopped() -> None:
    sim = SimulatedReuseEngine()
    record = _run(engine=sim)
    assert sim.starts == sim.stops == record["planned_trials"] + 1  # + the cancellation check
    assert not sim.running


def test_arm_order_alternates_between_blocks() -> None:
    plan = plan_trials(CASES[:1], [DEFAULT], 4)
    assert [arm for *_x, arm in plan] == [FRESH, REUSE, REUSE, FRESH, FRESH, REUSE, REUSE, FRESH]


def test_fresh_and_cached_readings_never_share_an_identity() -> None:
    fresh, reuse = _pair()
    assert fresh.identity["cache_prompt"] is False and reuse.identity["cache_prompt"] is True
    assert fresh.identity_sha256 != reuse.identity_sha256
    sim = SimulatedReuseEngine()
    other = run_trial(sim, CASE["shared_prefix"], SMALL_CHUNK, REUSE, block=0, context_length=4096,
                      configuration_sha256="c")
    assert other.identity_sha256 != reuse.identity_sha256  # the engine reuse option is part of it


def test_reuse_bound_is_the_shared_start_plus_one_written_piece_plus_shiftable_chunks() -> None:
    prime, measured = [1, 2, 3, 4, 9, 9], [1, 2, 3, 5, 6, 7]
    assert common_prefix(prime, measured) == 3
    assert reuse_bound(CASE["shared_prefix"], prime, measured, DEFAULT) == 4
    assert reuse_bound(CASE["identical"], [1, 2, 3], [1, 2, 3], DEFAULT) == 2  # last piece always processed
    deleted = reuse_bound(CASE["deleted_middle"], [1, 8, 8, 5, 6, 7, 3], [1, 5, 6, 7, 3], SMALL_CHUNK)
    assert deleted == 2  # chunk 16 is longer than the 4 shared pieces: prefix 1 + written 1
    variant = EngineVariant("c3", cache_reuse=3)
    assert reuse_bound(CASE["deleted_middle"], [1, 8, 8, 5, 6, 7, 3], [1, 5, 6, 7, 3, 2], variant) == 5


# --------------------------------------------------------------------------- #
# Finding 2: shorter or unverified work is never counted                      #
# --------------------------------------------------------------------------- #
def _modify(trial: Any, **values: Any) -> Any:
    return replace(trial, measured={**trial.measured, **values})


def test_missing_whole_prompt_counts_make_the_block_unusable() -> None:
    fresh, reuse = _pair()
    assert judge_block(fresh, reuse).usable
    verdict = judge_block(_modify(fresh, total=None), _modify(reuse, total=None))
    # The reuse arm still reports processed + cached; the fresh arm's freshness is unknown.
    assert not verdict.usable and verdict.gain_s is None
    assert verdict.fresh_stayed_fresh is None


def test_shorter_unreported_work_is_never_a_gain() -> None:
    fresh, reuse = _pair()
    verdict = judge_block(_modify(fresh, total=None),
                          _modify(reuse, total=None, processed=1, cached=2, ttft_s=0.001))
    assert not verdict.usable and verdict.relative_gain is None
    assert any("engine counted" in r or "whole-prompt" in r for r in verdict.reasons)


def test_counts_that_disagree_with_the_tokenizer_are_unusable() -> None:
    fresh, reuse = _pair()
    verdict = judge_block(_modify(fresh, total=10, processed=10, cached=0),
                          _modify(reuse, total=10, processed=1, cached=9))
    assert not verdict.usable
    assert any("its tokenizer" in r for r in verdict.reasons)


@pytest.mark.parametrize("faults,reason", [
    (SimFaults(short_generation=True), "pieces, asked for"),
    (SimFaults(incomplete_response=True), "did not complete"),
    (SimFaults(inconsistent_counters=True), "do not add up"),
    (SimFaults(wrong_total=5), "its tokenizer"),
    (SimFaults(hide_counters=True, warming_s=0.03), "whole-prompt count missing"),
    (SimFaults(other_slot=True), "not the priming slot"),
])
def test_incomplete_work_evidence_is_caught(faults: SimFaults, reason: str) -> None:
    fresh, reuse = _pair(faults=faults)
    verdict = judge_block(fresh, reuse)
    assert not verdict.usable and verdict.gain_s is None
    assert any(reason in r for r in verdict.reasons), verdict.reasons


def test_freshness_is_unknown_not_true_when_counters_are_missing() -> None:
    record = _run(SimFaults(hide_counters=True))
    assert record["status"] == INCOMPLETE  # nothing usable; no validity gate can fire without counters
    for item in _expected(record["summary"]):
        assert item["fresh_arm_stayed_fresh"] is None and item["usable_blocks"] == 0


def test_work_evidence_rejects_negative_and_non_integer_counts() -> None:
    good = {"processed": 5, "cached": 5, "total": 10, "generated": 3, "completed": True, "ttft_s": 0.2}
    assert work_evidence(good, expected_total=10, expected_generated=3) == (5, [])
    for bad in ({"processed": -1}, {"cached": 2.5}, {"total": True}, {"ttft_s": float("nan")}):
        cached, problems = work_evidence({**good, **bad}, expected_total=10, expected_generated=3)
        assert problems


def test_answers_that_change_with_reuse_are_not_counted() -> None:
    record = _run(SimFaults(corrupt_after_reuse=True))
    for item in _expected(record["summary"]):
        assert item["usable_blocks"] == 0 and False in item["answers_identical"]
    assert record["cancellation"][0]["verdict"] == "failed"


def test_warming_is_equal_in_both_arms_by_construction() -> None:
    record = _run(SimFaults(warming_s=0.03))
    assert abs(record["summary"]["default/unrelated"]["mean_relative_ttft_gain"]) < 0.05


# --------------------------------------------------------------------------- #
# Finding 4: the pre-registered validity gates stop the run at once           #
# --------------------------------------------------------------------------- #
def test_e1_a_fresh_reading_that_reused_stops_the_run() -> None:
    sim = SimulatedReuseEngine(faults=SimFaults(reuse_when_fresh=True))
    record = _run(engine=sim)
    assert record["status"] == STOPPED_VALIDITY and record["stop_reason"].startswith("E1")
    assert len(record["trials"]) < record["planned_trials"] and record["cancellation"] == []
    assert sim.starts == sim.stops == len(record["trials"])


def test_e4_impossible_reuse_stops_the_run() -> None:
    record = _run(SimFaults(over_reuse=400))
    assert record["status"] == STOPPED_VALIDITY and record["stop_reason"].startswith("E4")
    assert record["cancellation"] == []


def test_no_reuse_at_all_is_a_completed_negative_result() -> None:
    class NoReuse(SimulatedReuseEngine):
        def generate(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int) -> Any:
            return super().generate(prompt, cache_prompt=False, n_predict=n_predict, seed=seed)

    record = _run(engine=NoReuse())
    assert record["status"] == DONE and record["stop_reason"] == ""
    for item in _expected(record["summary"]):
        assert item["usable_blocks"] == 0 and item["fresh_arm_stayed_fresh"] is True
    assert not any(r.startswith(("E1", "E4")) for r in [record["stop_reason"]])


# --------------------------------------------------------------------------- #
# Finding 1: an unconfirmed cleanup stops everything                          #
# --------------------------------------------------------------------------- #
def test_a_password_file_left_behind_stops_the_run_at_once() -> None:
    sim = SimulatedReuseEngine(faults=SimFaults(leave_credential=True))
    record = _run(engine=sim)
    assert record["status"] == STOPPED_CLEANUP
    assert len(record["trials"]) == 1 and record["cancellation"] == []
    assert sim.starts == sim.stops == 1  # no later trial, no cancellation check


def test_a_failed_start_that_cleaned_up_is_a_failed_trial_not_a_cleanup_stop() -> None:
    sim = SimulatedReuseEngine(faults=SimFaults(fail_start=True))
    record = _run(engine=sim, blocks=1)
    assert record["status"] == INCOMPLETE
    assert all(any("did not start" in p for p in t["problems"]) for t in record["trials"])


def test_an_engine_that_fails_mid_trial_is_still_stopped() -> None:
    class Failing(SimulatedReuseEngine):
        def generate(self, prompt: str, **kwargs: Any) -> Any:
            raise RuntimeError("engine fell over")

    sim = Failing()
    trial = run_trial(sim, CASE["identical"], DEFAULT, REUSE, block=0, context_length=4096, configuration_sha256="c")
    assert sim.stops == 1 and not sim.running
    assert any("engine fell over" in p for p in trial.problems)


def _real_engine(factory: Any) -> Any:
    from inferenceos.contracts.experiment import RuntimeConfiguration

    return _script().RealReuseEngine(model=object(), configuration=RuntimeConfiguration(context_length=4096),
                                     backend_factory=factory)


def _failing_start_backend(made: list[Any], *, exits: bool) -> Any:
    """The real resident backend whose start fails after creating (and owning) a
    process double; the double's exit is or is not confirmed (no process is created)."""
    from types import SimpleNamespace

    from inferenceos.backends import resident_server as rs

    backend = rs.LlamaServerResidentBackend(server=Path("not-executed.exe"))
    backend.exit_timeout_s = backend.previous_gone_timeout_s = 0.01

    def failed_start(model: Any, params: Any) -> Any:
        backend._owned = SimpleNamespace(pid=424242, stop=lambda **kw: (None, "terminated", exits))  # type: ignore[assignment]
        raise RuntimeError("the engine did not start")

    backend._start = failed_start  # type: ignore[method-assign]
    made.append(backend)
    return backend


def test_review3_1_a_failed_start_with_an_unconfirmed_exit_stops_the_run_and_keeps_its_identity() -> None:
    """Review round 3, finding 1: the real backend's failed start leaves an unconfirmed
    process identity; that is not a clean cleanup, the run stops at once, the same
    backend is kept, no new start is allowed, and the retry reports it honestly."""
    from inferenceos.backends import resident_server as rs

    made: list[Any] = []
    engine = _real_engine(lambda: _failing_start_backend(made, exits=False))
    with patch.object(rs, "wait_process_exit", return_value=False):
        record = evaluate(engine, cases=(CASE["shared_prefix"],), variants=(DEFAULT,), blocks=2,
                          context_length=4096, configuration={}, run_id="r", power_check=False)
        assert record["status"] == STOPPED_CLEANUP and len(record["trials"]) == 1
        assert "424242" in record["stop_reason"] and record["trials"][0]["cleanup"]["process_gone"] is False
        assert len(made) == 1 and made[0].unconfirmed_pid == 424242 and engine.held is made[0]
        with pytest.raises(RuntimeError, match="not confirmed"):
            engine.start(DEFAULT)
        retry = engine.retry_held()
    assert retry["released"] is False and retry["unconfirmed_pid"] == 424242
    assert any("424242" in item for item in retry["unresolved"])
    with patch.object(rs, "wait_process_exit", return_value=True):
        retry = engine.retry_held()  # the process has exited now: confirmed, released
    assert retry["released"] is True and made[0].unconfirmed_pid is None and engine.held is None


def test_review3_1_a_failed_start_that_confirmed_its_exit_is_a_failed_trial_on_the_same_backend() -> None:
    from inferenceos.backends import resident_server as rs

    made: list[Any] = []
    engine = _real_engine(lambda: _failing_start_backend(made, exits=True))
    with patch.object(rs, "wait_process_exit", return_value=True):
        record = evaluate(engine, cases=(CASE["shared_prefix"],), variants=(DEFAULT,), blocks=1,
                          context_length=4096, configuration={}, run_id="r", power_check=False, cancellation=False)
    assert record["status"] == INCOMPLETE and len(record["trials"]) == 2
    assert all(t["cleanup"]["problem"] is None for t in record["trials"])
    assert len(made) == 1  # one backend, one cleanup owner, for the whole run
    assert engine.retry_held() is None


def test_review3_1_an_interrupted_stop_is_held_and_retried() -> None:
    """A Ctrl+C inside the real backend's stop: the engine is kept as unreleased, the
    interruption propagates, and the final retry covers it."""
    from types import SimpleNamespace

    from inferenceos.backends import resident_server as rs
    from inferenceos.contracts.lifecycle import ResourceSnapshot

    backend = rs.LlamaServerResidentBackend(server=Path("not-executed.exe"))
    backend.exit_timeout_s = backend.previous_gone_timeout_s = 0.01

    def interrupted(**kwargs: Any) -> Any:
        raise KeyboardInterrupt

    def fake_start(model: Any, params: Any) -> None:
        backend._process = SimpleNamespace(pid=515151, is_alive=lambda: True, stop=interrupted)  # type: ignore[assignment]
        backend._state = "ready"

    backend.start = fake_start  # type: ignore[method-assign,assignment]
    engine = _real_engine(lambda: backend)
    engine.start(DEFAULT)
    with patch.object(rs, "resource_snapshot", lambda pid: ResourceSnapshot(exists=False)), \
            patch.object(rs, "wait_process_exit", return_value=False):
        with pytest.raises(KeyboardInterrupt):
            engine.stop()
        assert engine.held is backend and engine.unreleased
        with pytest.raises(RuntimeError, match="not confirmed"):
            engine.start(DEFAULT)
        retry = engine.retry_held()
    assert retry["released"] is False and retry["unconfirmed_pid"] == 515151
    with patch.object(rs, "wait_process_exit", return_value=True):
        assert engine.retry_held()["released"] is True


def test_the_real_adapter_keeps_a_backend_whose_password_file_was_not_deleted(tmp_path: Path) -> None:
    from types import SimpleNamespace

    class Backend:
        undeleted_key_files = [tmp_path / "key.txt"]
        unconfirmed_pid = None
        cleanups = 0

        def start(self, model: Any, params: Any) -> None:
            pass

        def is_active(self) -> bool:
            return True

        def stop(self) -> Any:
            return SimpleNamespace(cleanup_complete=True, credential_removed=False)

        def cleanup(self) -> None:
            Backend.cleanups += 1

        def confirm_previous_gone(self, timeout_s: Any = None) -> bool:
            return True

    engine = _real_engine(Backend)
    engine.start(DEFAULT)
    engine._backend.is_active = lambda: False  # after the stop
    cleanup = engine.stop()
    assert not cleanup.clean and cleanup.credential_removed is False and engine.held is not None
    with pytest.raises(RuntimeError, match="not confirmed"):
        engine.start(DEFAULT)
    retry = engine.retry_held()
    assert retry is not None and Backend.cleanups == 1 and retry["password_files_left"] and not retry["released"]


# --------------------------------------------------------------------------- #
# Finding 5: cancellation must really cancel, against a real reference        #
# --------------------------------------------------------------------------- #
def _reference() -> dict[str, Any]:
    fresh, _ = _pair()
    return dict(fresh.measured)


def test_a_real_cancellation_with_a_valid_reference_passes() -> None:
    check = cancellation_check(SimulatedReuseEngine(), CASE["shared_prefix"], DEFAULT, reference=_reference())
    assert check["verdict"] == "passed" and check["abandoned"]["pieces"] >= 1


class _Counting(SimulatedReuseEngine):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.generated = 0
        self.abandoned = 0

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        self.generated += 1
        return super().generate(prompt, **kwargs)

    def abandon(self, prompt: str, **kwargs: Any) -> Any:
        self.abandoned += 1
        return super().abandon(prompt, **kwargs)


@pytest.mark.parametrize("faults,missing", [
    (SimFaults(abandon_after_finish=True), "engine finished"),
    (SimFaults(abandon_early_eof=True), "stream ended early"),
    (SimFaults(no_slot_info=True), "could not confirm"),
])
def test_a_cancellation_that_proves_nothing_is_inconclusive_and_sends_no_repeat(faults: SimFaults,
                                                                                 missing: str) -> None:
    engine = _Counting(faults=faults)
    check = cancellation_check(engine, CASE["shared_prefix"], DEFAULT, reference=_reference())
    assert check["verdict"] == "inconclusive" and not check["passed"]
    assert any(missing in m for m in check["missing"]), check["missing"]
    assert engine.generated == 1 and check["repeat"].startswith("not sent")  # priming only


@pytest.mark.parametrize("faults,problem", [
    (SimFaults(busy_after_abandon=True), "not confirmed ended"),
])
def test_review3_3_a_busy_engine_fails_the_check_and_gets_no_more_work(faults: SimFaults, problem: str) -> None:
    engine = _Counting(faults=faults)
    check = cancellation_check(engine, CASE["shared_prefix"], DEFAULT, reference=_reference())
    assert check["verdict"] == "failed" and any(problem in p for p in check["problems"])
    assert engine.generated == 1 and "healthy_after_abandon" not in check  # straight to cleanup


def test_review3_3_an_unhealthy_engine_fails_the_check_and_gets_no_more_work() -> None:
    class Unhealthy(_Counting):
        def healthy(self) -> bool:
            return False

    engine = Unhealthy()
    check = cancellation_check(engine, CASE["shared_prefix"], DEFAULT, reference=_reference())
    assert check["verdict"] == "failed" and engine.generated == 1 and check["repeat"].startswith("not sent")


def test_review3_3_an_early_end_from_the_real_backend_is_not_a_cancellation() -> None:
    """The reviewer's reproduction: one piece, then the stream ends (cutoff two), idle
    slots. The backend now reports it as an early end and the check is inconclusive."""
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    backend = _abandon_backend([{"content": "one"}], [[{"id": 0, "is_processing": False}]])
    evidence = backend.abandon_generation(GenerationRequest(prompt="p", n_predict=24), after_chunks=2)
    assert evidence["ending"] == ENDED_EARLY and evidence["pieces"] == 1

    class EarlyEnd(SimulatedReuseEngine):
        def abandon(self, *args: Any, **kwargs: Any) -> Abandoned:
            return Abandoned(**evidence)

    check = cancellation_check(EarlyEnd(), CASE["shared_prefix"], DEFAULT, reference=_reference())
    assert check["verdict"] == "inconclusive"


def test_no_fresh_reference_makes_the_check_inconclusive() -> None:
    engine = _Counting()
    check = cancellation_check(engine, CASE["shared_prefix"], DEFAULT, reference=None)
    assert check["verdict"] == "inconclusive" and engine.generated == 1  # a pass is impossible: no repeat


def test_an_engine_killed_by_cancelling_fails_the_check_and_stops_the_run() -> None:
    record = _run(SimFaults(die_on_abandon=True))
    check = record["cancellation"][0]
    assert check["verdict"] == "failed" and record["status"] == STOPPED_CLEANUP


# --------------------------------------------------------------------------- #
# Finding 6: an interruption keeps every completed trial                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("where", ["start", "generate", "between"])
def test_an_interruption_keeps_completed_trials_and_cleans_up(where: str) -> None:
    saved: list[dict[str, Any]] = []

    class Interrupted(SimulatedReuseEngine):
        def start(self, variant: EngineVariant) -> None:
            if where == "start" and self.starts == 2:
                self.starts += 1
                raise KeyboardInterrupt
            super().start(variant)

        def generate(self, prompt: str, **kwargs: Any) -> Any:
            if where == "generate" and self.starts == 3 and self._requests == 1:
                raise KeyboardInterrupt
            return super().generate(prompt, **kwargs)

    lines: list[str] = []

    def progress(line: str) -> None:
        lines.append(line)
        if where == "between" and len(lines) == 3:
            raise KeyboardInterrupt

    sim = Interrupted()
    with pytest.raises(KeyboardInterrupt):
        _run(engine=sim, checkpoint=lambda record: saved.append(record), progress=progress)
    last = saved[-1]
    assert last["status"] == INTERRUPTED and last["stop_reason"]
    assert len(last["trials"]) >= 2  # the completed trials survive
    assert not sim.running  # the interrupted trial's engine was stopped


# --------------------------------------------------------------------------- #
# Finding 8 (evaluation side): mains power before and after every trial       #
# --------------------------------------------------------------------------- #
def test_losing_mains_power_stops_the_run_and_marks_the_trial() -> None:
    states = iter(["ac", "ac", "ac", "battery"] + ["battery"] * 50)
    sim = SimulatedReuseEngine()
    record = evaluate(sim, cases=CASES, variants=VARIANTS, blocks=1, context_length=4096, configuration={},
                      run_id="p", power=lambda: next(states), power_check=True)
    assert record["status"] == STOPPED_POWER and len(record["trials"]) == 2
    assert record["trials"][-1]["power"] == ["ac", "battery"]


@pytest.mark.parametrize("state", ["battery", "unknown", "raise"])
def test_no_trial_starts_without_verified_mains_power(state: str) -> None:
    def probe() -> str:
        if state == "raise":
            raise OSError("no power information")
        return state

    sim = SimulatedReuseEngine()
    record = evaluate(sim, cases=CASES, variants=VARIANTS, blocks=1, context_length=4096, configuration={},
                      run_id="p", power=probe, power_check=True)
    assert record["status"] == STOPPED_POWER and record["trials"] == [] and sim.starts == 0


def test_an_unchecked_power_state_is_recorded() -> None:
    record = _run(blocks=1)
    assert record["power_check"].startswith("off")
    assert record["trials"][0]["power"] == ["not checked", "not checked"]


def test_an_incomplete_block_is_reported_not_dropped() -> None:
    fresh, _ = _pair("identical")
    item = summarize([fresh])["default/identical"]
    assert item["blocks"] == 1 and item["usable_blocks"] == 0 and not item["clean"]


# --------------------------------------------------------------------------- #
# Engine plumbing: fresh by default, reuse only when asked                    #
# --------------------------------------------------------------------------- #
class _Captured:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self) -> Any:
        return iter(self._lines)

    def read(self) -> bytes:
        return b"".join(self._lines)

    def close(self) -> None:
        pass

    def __enter__(self) -> "_Captured":
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def _client(monkeypatch: pytest.MonkeyPatch, lines: list[bytes]) -> tuple[Any, list[dict[str, Any]]]:
    from inferenceos.backends.llama_server_toolchain import LlamaServerClient

    sent: list[dict[str, Any]] = []
    client = LlamaServerClient(host="127.0.0.1", port=1, api_key="k")

    def fake_open(path: str, *, body: Any = None, method: str = "GET", timeout_s: Any = None) -> Any:
        sent.append({"path": path, "body": body})
        return _Captured(lines)

    monkeypatch.setattr(client, "_open", fake_open)
    return client, sent


def test_the_client_asks_for_a_fresh_reading_unless_told_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    client, sent = _client(monkeypatch, [b'data: {"content": "a"}\n'])
    list(client.stream_completion(prompt="p", n_predict=1, seed=1, temperature=0.0, stop=[], json_schema=None))
    list(client.stream_completion(prompt="p", n_predict=1, seed=1, temperature=0.0, stop=[], json_schema=None,
                                  cache_prompt=True))
    assert sent[0]["body"]["cache_prompt"] is False and sent[1]["body"]["cache_prompt"] is True


def test_tokenize_returns_the_engines_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    client, sent = _client(monkeypatch, [b'{"tokens": [1, 15, 27]}'])
    assert client.tokenize("hello") == [1, 15, 27]
    assert sent[0]["path"] == "/tokenize" and sent[0]["body"] == {"content": "hello", "add_special": True}


def test_the_reuse_option_reaches_the_command_only_when_set(tmp_path: Path) -> None:
    from inferenceos.backends.llama_server_toolchain import ServerOptions, build_server_command
    from inferenceos.backends.resident_server import _options_from
    from inferenceos.interfaces.backend_adapter import RuntimeParams

    model = tmp_path / "m.gguf"
    plain = build_server_command(Path("s"), _options_from(model, RuntimeParams(context_length=4096)),
                                 host="h", port=1, api_key_file=tmp_path / "k")
    assert "--cache-reuse" not in plain
    shifted = build_server_command(Path("s"), _options_from(model, RuntimeParams(context_length=4096, cache_reuse=256)),
                                   host="h", port=1, api_key_file=tmp_path / "k")
    assert shifted[shifted.index("--cache-reuse") + 1] == "256"
    assert ServerOptions(model_path=model).cache_reuse is None


def _backend_with(chunks: list[dict[str, Any]]) -> Any:
    from types import SimpleNamespace

    from inferenceos.backends import resident_server as rs

    class Client:
        def __init__(self) -> None:
            self.asked: list[bool] = []

        def stream_completion(self, **kwargs: Any) -> Any:
            self.asked.append(kwargs["cache_prompt"])
            return iter(chunks)

    backend = rs.LlamaServerResidentBackend(server=Path("not-run.exe"))
    backend._client = Client()  # type: ignore[assignment]
    backend._process = SimpleNamespace(pid=4242, is_alive=lambda: True)  # type: ignore[assignment]
    backend._state = "ready"
    return backend


def test_the_backend_reports_the_engines_reuse_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.backends import resident_server as rs
    from inferenceos.contracts.lifecycle import ResourceSnapshot
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    _quiet_backend(monkeypatch, rs, ResourceSnapshot)
    final = {"content": "", "stop": True, "tokens_evaluated": 1200,
             "timings": {"prompt_n": 30, "cache_n": 1170, "prompt_ms": 50.0, "predicted_n": 4}}
    backend = _backend_with([{"content": "hi"}, final])
    result = backend.generate(GenerationRequest(prompt="p", n_predict=4, cache_prompt=True))
    assert backend._client.asked == [True]
    assert (result.prompt_tokens, result.prompt_tokens_cached, result.prompt_tokens_total) == (30, 1170, 1200)
    assert result.cache_prompt is True and backend.last_generation.cache_prompt is True
    fresh = _backend_with([{"content": "hi"}, {"content": "", "stop": True, "tokens_evaluated": 1200,
                                               "timings": {"prompt_n": 1200, "cache_n": 0}}])
    plain = fresh.generate(GenerationRequest(prompt="p", n_predict=4))
    assert fresh._client.asked == [False] and plain.cache_prompt is False
    assert (plain.prompt_tokens, plain.prompt_tokens_cached) == (1200, 0)
    assert backend.last_response_counters["stop_seen"] is True


def test_slot_occupancy_is_never_read_as_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review finding 3: in b11139 ``tokens_cached`` is the slot's size after the answer."""
    from inferenceos.backends import resident_server as rs
    from inferenceos.contracts.lifecycle import ResourceSnapshot
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    _quiet_backend(monkeypatch, rs, ResourceSnapshot)
    backend = _backend_with([{"content": "a", "stop": True, "tokens_evaluated": 10, "tokens_predicted": 2,
                              "tokens_cached": 11, "id_slot": 0, "timings": {"prompt_n": 10, "predicted_n": 2}}])
    result = backend.generate(GenerationRequest(prompt="p", n_predict=2))
    assert result.prompt_tokens_cached is None  # unknown, not 11
    assert backend.last_response_counters["tokens_cached_occupancy"] == 11
    assert backend.last_response_counters["id_slot"] == 0


def _quiet_backend(monkeypatch: pytest.MonkeyPatch, rs: Any, snapshot: Any) -> None:
    monkeypatch.setattr(rs, "PeakRssAccumulator", lambda: type("P", (), {
        "observe": lambda self, pid: None, "peak_rss_bytes": None, "final_rss_bytes": None, "sample_count": 0,
        "os_peak_available": False, "peak_process_cpu_percent": None, "cpu_seconds": None,
        "context_switches_voluntary": None, "context_switches_involuntary": None, "context_switches_total": None})())
    monkeypatch.setattr(rs, "resource_snapshot", lambda pid: snapshot(exists=True))


def _abandon_backend(chunks: list[dict[str, Any]], slots: list[Any]) -> Any:
    backend = _backend_with(chunks)
    closed: list[bool] = []

    class Stream:
        def __init__(self) -> None:
            self._items = iter(chunks)

        def __iter__(self) -> Any:
            return self._items

        def close(self) -> None:
            closed.append(True)

    replies = iter(slots)
    limits: list[float] = []

    def slots_call(*, timeout_s: float = 10.0) -> Any:
        limits.append(timeout_s)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    backend._client.stream_completion = lambda **kwargs: Stream()
    backend._client.slots = slots_call
    backend.closed = closed
    backend.slot_limits = limits
    return backend


def test_abandoning_reports_an_interruption_and_waits_for_idle_slots() -> None:
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    busy, idle = [{"id": 0, "is_processing": True}], [{"id": 0, "is_processing": False}]
    backend = _abandon_backend([{"content": "a"}, {"content": "b"}, {"content": "c"}], [busy, idle])
    evidence = backend.abandon_generation(GenerationRequest(prompt="p", n_predict=9), after_chunks=2,
                                          idle_timeout_s=5)
    assert evidence == {"pieces": 2, "ending": CUT_BY_CLIENT, "error": None, "idle_confirmed": True,
                        "idle_detail": None} and backend.closed
    assert all(0 < limit <= 5 for limit in backend.slot_limits)  # each request limited to the time left


def test_abandoning_a_finished_answer_is_not_an_interruption() -> None:
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    backend = _abandon_backend([{"content": "a"}, {"content": "", "stop": True}],
                               [srv.ServerRequestError("GET /slots failed")])
    evidence = backend.abandon_generation(GenerationRequest(prompt="p", n_predict=9), after_chunks=5)
    assert evidence["ending"] == "engine finished" and evidence["pieces"] == 1
    assert evidence["idle_confirmed"] is None and "could not be read" in evidence["idle_detail"]


def test_review3_3_a_slot_reply_after_the_limit_is_not_accepted() -> None:
    from inferenceos.backends import resident_server as rs
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    backend = _abandon_backend([{"content": "a"}, {"content": "b"}], [])
    clock = [0.0]

    def late_slots(*, timeout_s: float = 10.0) -> Any:
        clock[0] = 35.0  # the reply arrives after the 30 s limit
        return [{"id": 0, "is_processing": False}]

    backend._client.slots = late_slots
    with patch.object(rs.time, "monotonic", side_effect=lambda: clock[0]):
        evidence = backend.abandon_generation(GenerationRequest(prompt="p"), after_chunks=2, idle_timeout_s=30)
    assert evidence["idle_confirmed"] is False and "after the 30 s limit" in evidence["idle_detail"]


def test_review3_3_still_busy_at_the_limit_is_reported_as_not_ended() -> None:
    from inferenceos.backends import resident_server as rs
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    busy = [{"id": 0, "is_processing": True}]
    backend = _abandon_backend([{"content": "a"}, {"content": "b"}], [busy] * 50)
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += 4.0  # each wait between polls takes a while

    with patch.object(rs.time, "monotonic", side_effect=lambda: clock[0]), patch.object(rs.time, "sleep", sleep):
        evidence = backend.abandon_generation(GenerationRequest(prompt="p"), after_chunks=2, idle_timeout_s=30)
    assert evidence["idle_confirmed"] is False and "still processing" in evidence["idle_detail"]


def test_review3_3_a_failed_abandon_request_is_reported_not_raised() -> None:
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    backend = _abandon_backend([], [[{"id": 0, "is_processing": False}]])

    def failing(**kwargs: Any) -> Any:
        raise srv.ServerRequestError("POST /completion failed: reset")

    backend._client.stream_completion = failing
    evidence = backend.abandon_generation(GenerationRequest(prompt="p"), after_chunks=2)
    assert evidence["ending"] == "request failed" and "reset" in evidence["error"] and evidence["pieces"] == 0


def test_the_client_reads_the_slot_list(monkeypatch: pytest.MonkeyPatch) -> None:
    client, sent = _client(monkeypatch, [b'[{"id": 0, "is_processing": false}]'])
    assert client.slots() == [{"id": 0, "is_processing": False}] and sent[0]["path"] == "/slots"


def test_only_the_reuse_evaluation_ever_asks_for_reuse() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "inferenceos"
    allowed = {"tune/prompt_reuse.py", "tune/prompt_reuse_sim.py", "backends/resident_server.py",
               "backends/llama_server_toolchain.py", "interfaces/backend_adapter.py"}
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if rel not in allowed and ("cache_prompt=True" in text or '"cache_prompt": True' in text
                                   or "cache_reuse=" in text):
            offenders.append(rel)
    assert offenders == []


# --------------------------------------------------------------------------- #
# Tune's request identity: fresh is explicit; cached never matches history     #
# --------------------------------------------------------------------------- #
def test_tune_requests_say_they_are_fresh_and_still_match_older_records() -> None:
    from inferenceos.tune.history_cost import compatible_request
    from inferenceos.tune.learning import FRESH_PROMPT_REUSE, comparable_request, request_identity_problems
    from tests.test_history_cost import request

    new = request()
    assert new["format_version"] == "tune-request-4" and new["prompt_reuse"] == FRESH_PROMPT_REUSE
    assert request_identity_problems(new) == []
    old = {k: v for k, v in deepcopy(new).items() if k != "prompt_reuse"}
    old["format_version"] = "tune-request-3"  # how every request was recorded before 16 Oct
    assert request_identity_problems(old) == []
    assert comparable_request(new) == old
    assert compatible_request(old, new, 10.0) and compatible_request(new, old, 10.0)


def test_a_cached_request_is_never_comparable_with_a_fresh_one() -> None:
    from inferenceos.tune.history_cost import compatible_request
    from inferenceos.tune.learning import comparable_request, request_identity_problems
    from tests.test_history_cost import request

    fresh = request()
    cached = deepcopy(fresh)
    cached["prompt_reuse"] = {"rule": "prompt-reuse-1", "cache_prompt": True, "server_cache_reuse": None}
    assert "prompt reuse: not a fresh reading of the whole prompt" in request_identity_problems(cached)
    assert comparable_request(cached) is None
    assert not compatible_request(fresh, cached, 10.0) and not compatible_request(cached, cached, 10.0)
    shifted = deepcopy(fresh)
    shifted["prompt_reuse"] = {"rule": "prompt-reuse-1", "cache_prompt": False, "server_cache_reuse": 256}
    assert comparable_request(shifted) is None and request_identity_problems(shifted)


def test_cached_history_never_feeds_a_cost_forecast() -> None:
    from inferenceos.contracts.tune_learning import MeasurementObservation
    from tests.test_history_cost import forecaster, history, request

    runs = history()
    tainted = []
    for run in runs:
        obs = run.observations[0]
        req = deepcopy(obs.request)
        req["prompt_reuse"] = {"rule": "prompt-reuse-1", "cache_prompt": True, "server_cache_reuse": None}
        tainted.append(run.model_copy(update={"observations": (MeasurementObservation.model_validate(
            {**obs.model_dump(), "request": req}),)}))
    assert forecaster(runs).estimate(request()) is not None
    assert forecaster(tuple(tainted)).estimate(request()) is None


# --------------------------------------------------------------------------- #
# The script: isolation, checkpoints, refusals                                 #
# --------------------------------------------------------------------------- #
def _script() -> Any:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_prompt_reuse.py"
    spec = importlib.util.spec_from_file_location("evaluate_prompt_reuse", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_prompt_reuse"] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_runs_the_simulation_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "sim.json"
    code = _script().main(["--simulate", "--blocks", "2", "--out", str(out), "--run-id", "rid"])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and record["status"] == DONE and record["run_id"] == "rid" and record["simulated"]
    assert record["power_check"].startswith("off") and "isolation control" in capsys.readouterr().out


def test_the_script_refuses_without_a_model(tmp_path: Path) -> None:
    assert _script().main(["--model", str(tmp_path / "missing.gguf"), "--out", str(tmp_path / "o.json")]) == 4


def test_the_script_never_guesses_a_setup(tmp_path: Path) -> None:
    from inferenceos.config import Settings

    script = _script()
    settings = Settings().model_copy(update={"data_dir": tmp_path / "data"})
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    with pytest.raises(LookupError):
        script._setup(settings, model, 4096, None)
    config, source = script._setup(settings, model, 4096, 12)
    assert config.context_length == 4096 and config.gpu_offload_layers == 12 and "--layers" in source


def _real_mode(tmp_path: Path, out: Path, *, engine: Any = None) -> tuple[int, Path]:
    """The real-mode orchestration with every hardware piece replaced: temporary data
    folder, mocked model loading, the simulated engine behind the real adapter's name."""
    from inferenceos.config import Settings
    from inferenceos.contracts.experiment import RuntimeConfiguration

    script = _script()
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    saved = data / "saved_setups.json"
    if not saved.exists():
        saved.write_text('{"sentinel": "owner data"}', encoding="utf-8")
    model = tmp_path / "dummy.gguf"
    model.write_bytes(b"not a model")
    settings = Settings().model_copy(update={"data_dir": data})
    with patch("inferenceos.config.Settings", return_value=settings), \
            patch("inferenceos.models.loader.load_model_profile", return_value=object()), \
            patch.object(script, "_setup", return_value=(RuntimeConfiguration(context_length=4096), "mocked")), \
            patch.object(script, "RealReuseEngine", return_value=engine or SimulatedReuseEngine()), \
            patch.object(script, "power_state", return_value="ac"):
        code = script.main(["--model", str(model), "--blocks", "1", "--no-variants", "--cases", "identical,unrelated",
                            "--out", str(out)])
    return code, saved


def test_a_results_path_inside_the_data_folder_is_refused_before_anything_starts(tmp_path: Path) -> None:
    out = tmp_path / "data" / "saved_setups.json"
    code, saved = _real_mode(tmp_path, out)
    assert code == 4 and saved.read_text(encoding="utf-8") == '{"sentinel": "owner data"}'


def test_a_results_path_reaching_the_data_folder_through_a_link_is_refused(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    link = tmp_path / "elsewhere"
    try:
        link.symlink_to(tmp_path / "data", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot create a folder link")
    code, saved = _real_mode(tmp_path, link / "saved_setups.json")
    assert code == 4 and "sentinel" in saved.read_text(encoding="utf-8")


def test_real_mode_reports_an_unchanged_data_folder_truthfully(tmp_path: Path) -> None:
    out = tmp_path / "out" / "r.json"
    code, saved = _real_mode(tmp_path, out)
    record = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and record["data_folder_unchanged"] is True and record["status"] == DONE
    assert "sentinel" in saved.read_text(encoding="utf-8")


def test_a_cleanup_stop_exits_5_with_the_partial_record(tmp_path: Path) -> None:
    out = tmp_path / "out" / "r.json"
    code, _ = _real_mode(tmp_path, out, engine=SimulatedReuseEngine(faults=SimFaults(leave_credential=True)))
    record = json.loads(out.read_text(encoding="utf-8"))
    assert code == 5 and record["status"] == STOPPED_CLEANUP and len(record["trials"]) == 1


def test_an_interrupted_run_keeps_its_trials_in_the_results_file(tmp_path: Path) -> None:
    class Interrupted(SimulatedReuseEngine):
        def start(self, variant: EngineVariant) -> None:
            if self.starts == 1:
                self.starts += 1
                raise KeyboardInterrupt
            super().start(variant)

    out = tmp_path / "out" / "r.json"
    with pytest.raises(KeyboardInterrupt):
        _real_mode(tmp_path, out, engine=Interrupted())
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["status"] == INTERRUPTED and len(record["trials"]) == 2  # one done, one interrupted
    assert record["trials"][0]["problems"] == [] and record["trials"][1]["interrupted"] is True
    assert record.get("data_folder_unchanged") is True and record.get("finished_at")


def test_power_states_from_the_battery_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import psutil

    script = _script()
    for battery, state in ((None, "no battery"), (SimpleNamespace(power_plugged=True), "ac"),
                           (SimpleNamespace(power_plugged=False), "battery"),
                           (SimpleNamespace(power_plugged=None), "unknown")):
        monkeypatch.setattr(psutil, "sensors_battery", lambda b=battery: b)
        # The psutil branch is the non-Windows one; Windows reads its own AC line status.
        assert script.power_state(platform="linux").startswith(state)


def test_case_prompts_fit_well_inside_the_context() -> None:
    for case in build_cases():
        words = max(len(case.prime.split()), len(case.measured.split()))
        assert words * 2 < 4096 // 2  # generous: about two pieces per word, half the context
    assert {c.case_id for c in build_cases()} == {"identical", "shared_prefix", "continuation", "edited_middle",
                                                   "deleted_middle", "unrelated"}
    assert Abandoned(pieces=1, ending=CUT_BY_CLIENT, idle_confirmed=True).pieces == 1


# --------------------------------------------------------------------------- #
# Review round 3                                                              #
# --------------------------------------------------------------------------- #
def test_review3_5_missing_slot_identity_makes_the_block_unusable() -> None:
    fresh, reuse = _pair()
    no_slots = judge_block(replace(fresh, prime={**fresh.prime, "slot": None}, measured={**fresh.measured, "slot": None}),
                           replace(reuse, prime={**reuse.prime, "slot": None}, measured={**reuse.measured, "slot": None}))
    assert not no_slots.usable and no_slots.relative_gain is None
    assert any("slot identity not reported" in r for r in no_slots.reasons)
    one_missing = judge_block(fresh, replace(reuse, measured={**reuse.measured, "slot": None}))
    assert not one_missing.usable
    fresh_only = judge_block(replace(fresh, measured={**fresh.measured, "slot": None}), reuse)
    assert fresh_only.usable and fresh_only.slots[FRESH] == [0, None]  # recorded, not required


def test_review3_5_an_engine_without_slot_ids_counts_no_gain() -> None:
    record = _run(SimFaults(no_slot_ids=True), blocks=1)
    for item in _expected(record["summary"]):
        assert item["usable_blocks"] == 0


def test_review3_6_a_priming_reading_that_reused_on_a_fresh_load_stops_before_the_measured_request() -> None:
    engine = _Counting(faults=SimFaults(prime_reuse=1))
    record = _run(engine=engine)
    assert record["status"] == STOPPED_VALIDITY and record["stop_reason"].startswith("E4: the priming request")
    assert len(record["trials"]) == 1 and record["cancellation"] == []
    trial = record["trials"][0]
    assert trial["measured"] == {} and trial["breach"] and engine.generated == 1  # measured request never sent
    assert engine.starts == engine.stops == 1 and trial["cleanup"]["problem"] is None


def test_review3_6_the_cancellation_checks_priming_is_gated_too() -> None:
    engine = _Counting(faults=SimFaults(prime_reuse=1))
    check = cancellation_check(engine, CASE["shared_prefix"], DEFAULT, reference=_reference())
    assert check["verdict"] == "failed" and check["breach"].startswith("E4")
    assert engine.abandoned == 0 and engine.stops == 1


def test_review3_6_a_breach_in_the_cancellation_check_stops_the_run() -> None:
    class LateBreach(SimulatedReuseEngine):
        planned = len(plan_trials(CASES, VARIANTS, 1))

        def generate(self, prompt: str, *, cache_prompt: bool, n_predict: int, seed: int) -> Any:
            reading = super().generate(prompt, cache_prompt=cache_prompt, n_predict=n_predict, seed=seed)
            if self.starts > self.planned and self._requests == 2:  # the repeat after cancelling
                reading.cached, reading.processed = reading.total, 0
            return reading

    record = _run(engine=LateBreach(), blocks=1)
    assert record["status"] == STOPPED_VALIDITY and "repeat after a cancelled answer" in record["stop_reason"]
    assert record["cancellation"][0]["verdict"] == "failed"


def test_review3_6_validity_breach_checks_every_reading_of_a_saved_trial() -> None:
    from inferenceos.tune.prompt_reuse import validity_breach

    fresh, reuse = _pair()
    assert validity_breach(fresh) is None and validity_breach(reuse) is None
    tainted = replace(fresh, prime={**fresh.prime, "cached": 1, "processed": fresh.prime["processed"] - 1})
    assert (validity_breach(tainted) or "").startswith("E4: the priming request")


def test_review3_2_unknown_windows_power_is_never_mains(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    script = _script()

    def not_used() -> Any:
        raise AssertionError("psutil must not decide power on Windows")

    monkeypatch.setattr(psutil, "sensors_battery", not_used)
    assert script.power_state(platform="win32", ac_line=lambda: 1) == "ac"
    assert script.power_state(platform="win32", ac_line=lambda: 0) == "battery"
    assert script.power_state(platform="win32", ac_line=lambda: 255).startswith("unknown")

    def unreadable() -> int:
        raise OSError("no status")

    assert script.power_state(platform="win32", ac_line=unreadable).startswith("unknown")


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="Windows power status")
def test_review3_2_installed_psutil_hides_unknown_and_the_evaluation_does_not() -> None:
    import psutil._pswindows as winpower  # type: ignore[import-not-found,unused-ignore]

    script = _script()
    with patch.object(getattr(winpower, "cext"), "sensors_battery", return_value=(255, 255, 255, -1)):
        assert winpower.sensors_battery() is None  # psutil's conversion: "no battery"
    assert script.power_state(ac_line=lambda: 255).startswith("unknown")
    assert script._windows_ac_line() in (0, 1, 255)  # the real system call answers


def test_review3_4_an_existing_results_file_is_never_overwritten_or_merged(tmp_path: Path) -> None:
    """The reviewer's stale-checkpoint reproduction: refused before anything, bytes kept."""
    out = tmp_path / "out" / "old.json"
    out.parent.mkdir()
    old = json.dumps({"run_id": "old-run", "trials": [{"old_evidence": True}], "summary": {"old": {"gain": 999}}})
    out.write_text(old, encoding="utf-8")
    code, _ = _real_mode(tmp_path, out)
    assert code == 4 and out.read_text(encoding="utf-8") == old
    code = _script().main(["--simulate", "--blocks", "1", "--out", str(out)])
    assert code == 4 and out.read_text(encoding="utf-8") == old


def test_review3_4_a_refusal_after_the_checks_is_recorded_as_refused_with_no_other_evidence(tmp_path: Path) -> None:
    from inferenceos.config import Settings

    script = _script()
    data = tmp_path / "data"
    data.mkdir()
    model = tmp_path / "dummy.gguf"
    model.write_bytes(b"x")
    out = tmp_path / "out" / "r.json"
    settings = Settings().model_copy(update={"data_dir": data})
    with patch("inferenceos.config.Settings", return_value=settings), \
            patch.object(script, "_setup", side_effect=LookupError("no setup")):
        code = script.main(["--model", str(model), "--out", str(out), "--run-id", "new-run"])
    record = json.loads(out.read_text(encoding="utf-8"))
    assert code == 4 and record["status"] == "refused" and record["refusal"] == "no setup"
    assert record["run_id"] == "new-run" and "trials" not in record and "summary" not in record


def test_review3_4_simulation_never_writes_into_the_data_folder(tmp_path: Path) -> None:
    from inferenceos.config import Settings

    script = _script()
    data = tmp_path / "data"
    data.mkdir()
    protected = data / "saved_setups.json"
    protected.write_text("owner sentinel", encoding="utf-8")
    settings = Settings().model_copy(update={"data_dir": data})
    with patch("inferenceos.config.Settings", return_value=settings):
        code = script.main(["--simulate", "--blocks", "1", "--cases", "identical", "--no-variants",
                            "--out", str(protected)])
        assert script.main(["--simulate", "--blocks", "1", "--out", str(data / "new.json")]) == 4
    assert code == 4 and protected.read_text(encoding="utf-8") == "owner sentinel"
    assert not (data / "new.json").exists()


def test_review3_7_the_engines_own_log_decides_whether_cache_reuse_was_turned_off() -> None:
    on = EngineVariant("c", cache_reuse=256)
    loaded = "srv    load_model: loading\nsrv    operator(): model loaded\n"
    assert cache_reuse_activation(DEFAULT, loaded) == "not requested"
    assert cache_reuse_activation(on, None).startswith("unverified")
    assert cache_reuse_activation(on, "srv    load_model: loading\n").startswith("unverified")
    for warning in CACHE_REUSE_DISABLED_WARNINGS:
        text = f"srv    load_model: {warning}\n" + loaded
        assert cache_reuse_activation(on, text).startswith("disabled by the engine")
    assert cache_reuse_activation(on, loaded).startswith("not disabled")


@pytest.mark.parametrize("faults,state", [
    (SimFaults(), "testable: shifted reuse observed in every reuse-arm load"),
    (SimFaults(disable_cache_reuse=True), "not testable: the engine turned --cache-reuse off"),
    (SimFaults(disable_cache_reuse=True, no_startup_log=True), "not testable: activation unverified in 2 of 2"),
    # The reuse-arm load shows shifted reuse, but the fresh-arm load has no evidence of its own.
    (SimFaults(no_startup_log=True), "not testable: activation unverified in 1 of 2"),
])
def test_review3_7_e7_needs_eligible_tokens_and_an_engine_that_kept_the_option(faults: SimFaults,
                                                                               state: str) -> None:
    summary = _run(faults, blocks=1)["summary"]
    deleted = summary["cache-reuse-16/deleted_middle"]["shifted_reuse"]
    assert deleted["e7"].startswith(state), deleted
    assert all(e and e > 0 for e in deleted["eligible_pieces"])
    edited = summary["cache-reuse-16/edited_middle"]["shifted_reuse"]
    assert edited["e7"].startswith("not testable: no shiftable chunk")  # a replacement never shifts
    assert "shifted_reuse" not in summary["default/deleted_middle"]


def test_review3_7_the_real_adapter_gives_no_log_when_warnings_are_not_logged() -> None:
    from types import SimpleNamespace

    class Backend:
        def start(self, model: Any, params: Any) -> None:
            pass

        def server_startup_log(self) -> str:
            return "srv    operator(): model loaded"

    engine = _real_engine(Backend)
    engine.start(EngineVariant("c", cache_reuse=256))
    assert engine.startup_log() == "srv    operator(): model loaded"
    engine._params = SimpleNamespace(log_verbosity=1)  # errors only: a missing warning proves nothing
    assert engine.startup_log() is None


# --------------------------------------------------------------------------- #
# Review round 4                                                              #
# --------------------------------------------------------------------------- #
def _ready_backend(tmp_path: Path, made: list[Any]) -> Any:
    """The real resident backend, 'started' with a process double already gone and a
    real temporary password file (no process is created)."""
    from types import SimpleNamespace

    from inferenceos.backends import resident_server as rs

    backend = rs.LlamaServerResidentBackend(server=Path("not-executed.exe"))
    backend.exit_timeout_s = backend.previous_gone_timeout_s = 0.01
    key = tmp_path / "inferenceos-key-test.txt"

    def fake_start(model: Any, params: Any) -> None:
        key.write_text("secret\n", encoding="utf-8")
        backend._key_file = key
        backend._process = SimpleNamespace(pid=626262, is_alive=lambda: False,  # type: ignore[assignment]
                                           stop=lambda **kw: (0, "terminate", True))
        backend._state = "ready"

    backend.start = fake_start  # type: ignore[method-assign,assignment]
    made.append((backend, key))
    return backend


def test_review4_1_an_interrupted_password_file_deletion_stays_on_record_and_is_retried(tmp_path: Path) -> None:
    """The reviewer's credential_interrupt: Ctrl+C inside the deletion of the password
    file. The path stays on the backend's record, the release stays unconfirmed, and the
    final retry deletes the real file."""
    from inferenceos.backends import resident_server as rs
    from inferenceos.contracts.lifecycle import ResourceSnapshot

    made: list[Any] = []
    engine = _real_engine(lambda: _ready_backend(tmp_path, made))
    engine.start(DEFAULT)
    backend, key = made[0]
    real_unlink = Path.unlink

    def interrupted_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == key:
            raise KeyboardInterrupt
        real_unlink(self, missing_ok=missing_ok)

    with patch.object(rs, "resource_snapshot", lambda pid: ResourceSnapshot(exists=False)), \
            patch.object(rs, "wait_process_exit", return_value=True):
        with patch.object(Path, "unlink", interrupted_unlink):
            with pytest.raises(KeyboardInterrupt):
                engine.stop()
        assert key.exists() and backend.undeleted_key_files == [key]  # still on record
        assert engine.held is backend
        with pytest.raises(RuntimeError, match="not confirmed"):
            engine.start(DEFAULT)
        retry = engine.retry_held()
    assert retry["released"] is True and retry["password_files_left"] == []
    assert not key.exists() and engine.held is None


def test_review4_1_a_password_file_that_still_cannot_be_deleted_stays_unresolved(tmp_path: Path) -> None:
    from inferenceos.backends import resident_server as rs
    from inferenceos.contracts.lifecycle import ResourceSnapshot

    made: list[Any] = []
    engine = _real_engine(lambda: _ready_backend(tmp_path, made))
    engine.start(DEFAULT)
    backend, key = made[0]
    real_unlink = Path.unlink

    def locked(self: Path, missing_ok: bool = False) -> None:
        if self == key:
            raise PermissionError("held open by another program")
        real_unlink(self, missing_ok=missing_ok)

    with patch.object(rs, "resource_snapshot", lambda pid: ResourceSnapshot(exists=False)), \
            patch.object(rs, "wait_process_exit", return_value=True), patch.object(Path, "unlink", locked):
        cleanup = engine.stop()
        retry = engine.retry_held()
    assert not cleanup.clean and cleanup.credential_removed is False
    assert retry["released"] is False and retry["password_files_left"] == [str(key)] and key.exists()


def test_review4_1_an_interruption_after_the_process_stop_keeps_its_identity(tmp_path: Path) -> None:
    """Ctrl+C after the process was told to stop but before its exit was confirmed: the
    identity was recorded before acting, so it is never lost."""
    from inferenceos.backends import resident_server as rs
    from inferenceos.contracts.lifecycle import ResourceSnapshot

    made: list[Any] = []
    engine = _real_engine(lambda: _ready_backend(tmp_path, made))
    engine.start(DEFAULT)
    backend, key = made[0]
    calls = [0]

    def snapshot(pid: Any) -> Any:
        calls[0] += 1
        if calls[0] == 2:  # the snapshot after the stop
            raise KeyboardInterrupt
        return ResourceSnapshot(exists=False)

    with patch.object(rs, "resource_snapshot", snapshot), patch.object(rs, "wait_process_exit", return_value=False):
        with pytest.raises(KeyboardInterrupt):
            engine.stop()
        assert backend.unconfirmed_pid == 626262 and engine.held is backend
        assert engine.retry_held()["released"] is False
    with patch.object(rs, "resource_snapshot", lambda pid: ResourceSnapshot(exists=False)), \
            patch.object(rs, "wait_process_exit", return_value=True):
        retry = engine.retry_held()
    assert retry["released"] is True and backend.unconfirmed_pid is None and not key.exists()


def test_review4_1_a_clean_stop_leaves_no_identity_or_file_on_record(tmp_path: Path) -> None:
    from inferenceos.backends import resident_server as rs
    from inferenceos.contracts.lifecycle import ResourceSnapshot

    made: list[Any] = []
    engine = _real_engine(lambda: _ready_backend(tmp_path, made))
    engine.start(DEFAULT)
    backend, key = made[0]
    with patch.object(rs, "resource_snapshot", lambda pid: ResourceSnapshot(exists=False)), \
            patch.object(rs, "wait_process_exit", return_value=True):
        cleanup = engine.stop()
    assert cleanup.clean and backend.unconfirmed_pid is None and backend.undeleted_key_files == []
    assert not key.exists() and engine.retry_held() is None


@pytest.mark.parametrize("state", [None, 0, "", "no", 1])
def test_review4_2_a_slot_state_that_is_not_true_or_false_is_never_idle(state: Any) -> None:
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    backend = _abandon_backend([{"content": "a"}, {"content": "b"}], [[{"id": 0, "is_processing": state}]])
    evidence = backend.abandon_generation(GenerationRequest(prompt="p"), after_chunks=2)
    assert evidence["ending"] == CUT_BY_CLIENT and evidence["idle_confirmed"] is None

    class Malformed(_Counting):
        def abandon(self, *args: Any, **kwargs: Any) -> Abandoned:
            return Abandoned(**evidence)

    engine = Malformed()
    check = cancellation_check(engine, CASE["shared_prefix"], DEFAULT, reference=_reference())
    assert check["verdict"] == "inconclusive" and engine.generated == 1  # no repeat


def test_review4_2_a_slot_list_with_an_unreadable_entry_is_rejected_not_filtered(monkeypatch: pytest.MonkeyPatch) -> None:
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.interfaces.backend_adapter import GenerationRequest

    client, _ = _client(monkeypatch, [b'[{"id": 0, "is_processing": false}, null]'])
    with pytest.raises(srv.ServerRequestError, match="not a slot object"):
        client.slots()
    backend = _abandon_backend([{"content": "a"}, {"content": "b"}], [])
    backend._client.slots = lambda timeout_s=10.0: client.slots(timeout_s=timeout_s)
    evidence = backend.abandon_generation(GenerationRequest(prompt="p"), after_chunks=2)
    assert evidence["idle_confirmed"] is None and "not a slot object" in evidence["idle_detail"]


def test_review4_2_a_valid_b11139_slot_reply_is_still_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _client(monkeypatch, [b'[{"id": 0, "is_processing": false}, {"id": 1, "is_processing": false}]'])
    assert [s["is_processing"] for s in client.slots()] == [False, False]


def test_review4_3_one_loads_shifted_reuse_never_vouches_for_another_load() -> None:
    """The reviewer's mixed_activation: two blocks, every load eligible, every startup
    log unverified, shifted reuse in only one reuse-arm load: not testable."""
    from inferenceos.tune.prompt_reuse import shifted_reuse_status

    sim = SimulatedReuseEngine(faults=SimFaults(no_startup_log=True))
    trials = [run_trial(sim, CASE["deleted_middle"], SMALL_CHUNK, arm, block=b, context_length=4096,
                        configuration_sha256="c") for b in range(2) for arm in (FRESH, REUSE)]
    no_shift = SimulatedReuseEngine(faults=SimFaults(no_startup_log=True, disable_cache_reuse=True))
    trials[3] = run_trial(no_shift, CASE["deleted_middle"], SMALL_CHUNK, REUSE, block=1, context_length=4096,
                          configuration_sha256="c")
    status = shifted_reuse_status(trials)
    beyond = status["reuse_beyond_shared_start"]
    assert beyond[0] > 0 and beyond[1] <= 0
    assert status["e7"] == "not testable: activation unverified in 3 of 4 loads"
    assert [load["evidence"] for load in status["loads"]] == [
        "unverified", "on: shifted reuse in this load's counters", "unverified", "unverified"]


def test_review4_3_an_enabled_option_with_no_shift_stays_distinct_from_unverified() -> None:
    from inferenceos.tune.prompt_reuse import shifted_reuse_status

    class KeptOnNoShift(SimulatedReuseEngine):
        def startup_log(self) -> str | None:
            return "srv    load_model: loading\nsrv    operator(): model loaded"

    sim = KeptOnNoShift(faults=SimFaults(disable_cache_reuse=True))  # no shifting, and no warning in its log
    trials = [run_trial(sim, CASE["deleted_middle"], SMALL_CHUNK, arm, block=0, context_length=4096,
                        configuration_sha256="c") for arm in (FRESH, REUSE)]
    assert shifted_reuse_status(trials)["e7"] == (
        "testable: kept on in every load (startup log), no shifted reuse observed")


# --------------------------------------------------------------------------- #
# Review round 5                                                              #
# --------------------------------------------------------------------------- #
def test_review5_1_overlapping_cleanups_keep_the_original_error_and_return_to_idle(tmp_path: Path) -> None:
    """The reviewer's concurrent_cleanup: a failed start is deleting its password file
    when a public cleanup() arrives. The two never work on the record at once; the
    original exception object propagates; the backend ends idle and can start again."""
    import threading

    from inferenceos.backends import resident_server as rs

    backend = rs.LlamaServerResidentBackend(server=Path("not-executed.exe"))
    backend.exit_timeout_s = backend.previous_gone_timeout_s = 0.01
    key = tmp_path / "inferenceos-key-overlap.txt"
    original = RuntimeError("the specific start failure")

    def failing_start(model: Any, params: Any) -> Any:
        key.write_text("secret\n", encoding="utf-8")
        backend._key_file = key
        raise original

    backend._start = failing_start  # type: ignore[method-assign]
    entered, release = threading.Event(), threading.Event()
    starter: list[int] = []
    real_unlink = Path.unlink

    def paused_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == key and threading.get_ident() in starter:
            entered.set()
            assert release.wait(5)
        real_unlink(self, missing_ok=missing_ok)

    caught: list[BaseException] = []

    def run_start() -> None:
        starter.append(threading.get_ident())
        try:
            backend.start(object(), object())  # type: ignore[arg-type]
        except BaseException as exc:  # noqa: BLE001
            caught.append(exc)

    with patch.object(Path, "unlink", paused_unlink):
        first = threading.Thread(target=run_start)
        first.start()
        assert entered.wait(5)
        second = threading.Thread(target=backend.cleanup)
        second.start()
        second.join(0.3)
        assert second.is_alive()  # waits for the failed start's bookkeeping, never overlaps it
        release.set()
        first.join(5)
        second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert caught == [original] and caught[0] is original
    assert backend.state == "idle" and backend._idle.is_set()
    assert not key.exists() and backend.undeleted_key_files == [] and backend.api_key_file is None
    assert backend.confirm_previous_gone(0.01) is True
    backend._start = lambda model, params: "started"  # type: ignore[method-assign,assignment,return-value]
    assert backend.start(object(), object()) == "started"  # type: ignore[arg-type]


def test_review5_1_a_record_already_cleared_by_another_caller_never_raises(tmp_path: Path) -> None:
    from inferenceos.backends import resident_server as rs

    backend = rs.LlamaServerResidentBackend(server=Path("not-executed.exe"))
    key = tmp_path / "inferenceos-key-gone.txt"
    key.write_text("secret\n", encoding="utf-8")
    backend._key_file = key
    real_unlink = Path.unlink

    def unlink_and_clear(self: Path, missing_ok: bool = False) -> None:
        real_unlink(self, missing_ok=missing_ok)
        backend._undeleted_keys.clear()  # another bookkeeper already took it off the record

    with patch.object(Path, "unlink", unlink_and_clear):
        assert backend._cleanup_key_file() is True
    assert not key.exists()


def _start_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[int], list[Path]]:
    """The real backend start up to the password file, its temporary file redirected to
    ``tmp_path``; no process is ever created."""
    import tempfile

    from inferenceos.backends import resident_server as rs

    made_fds: list[int] = []
    made_paths: list[Path] = []
    real_mkstemp = tempfile.mkstemp

    def mkstemp(*args: Any, **kwargs: Any) -> Any:
        handle, name = real_mkstemp(*args, **{**kwargs, "dir": str(tmp_path)})
        made_fds.append(handle)
        made_paths.append(Path(name))
        return handle, name

    monkeypatch.setattr(rs.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(rs, "sweep_stale_key_files", lambda *a, **k: None)

    def no_process(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no process may be created")

    model = tmp_path / "model.gguf"
    model.write_bytes(b"x")
    backend = rs.LlamaServerResidentBackend(server=Path("not-executed.exe"),
                                            process_factory=no_process)  # type: ignore[arg-type]
    backend.exit_timeout_s = backend.previous_gone_timeout_s = 0.01
    return backend, made_fds, made_paths


def _closed(fd: int) -> bool:
    import os

    try:
        os.fstat(fd)
    except OSError:
        return True
    return False


@pytest.mark.parametrize("failure", ["interrupt after writing", "write error before writing", "permissions failed"])
def test_review5_2_a_password_file_is_owned_from_creation(failure: str, tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    """The reviewer's startup probes: a failure or Ctrl+C while the password file is being
    prepared. The descriptor is closed, the file is deleted by the failed start's own
    cleanup, and the adapter's clean report is true (the file really is gone)."""
    from types import SimpleNamespace

    from inferenceos.backends import resident_server as rs

    backend, fds, paths = _start_preparation(tmp_path, monkeypatch)
    real_write = Path.write_text
    expected: type[BaseException] = KeyboardInterrupt if failure.startswith("interrupt") else PermissionError

    def write_text(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self in paths:
            if failure == "interrupt after writing":
                real_write(self, data, *args, **kwargs)
                raise KeyboardInterrupt
            if failure == "write error before writing":
                raise PermissionError("denied")
        return real_write(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    if failure == "permissions failed":
        monkeypatch.setattr(rs, "restrict_api_key_file", lambda path: (_ for _ in ()).throw(PermissionError("acl")))
    engine = _real_engine(lambda: backend)
    engine._model = SimpleNamespace(path=str(tmp_path / "model.gguf"), name="m")
    with pytest.raises(expected):
        engine.start(DEFAULT)
    assert len(paths) == 1 and _closed(fds[0])
    assert not paths[0].exists()  # deleted by the failed start's own cleanup
    assert backend.state == "idle" and backend.api_key_file is None and backend.undeleted_key_files == []
    cleanup = engine.stop()
    assert cleanup.clean and engine.retry_held() is None


def test_review5_2_a_password_file_that_cannot_be_deleted_after_a_failed_start_stays_unresolved(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    backend, fds, paths = _start_preparation(tmp_path, monkeypatch)
    real_write, real_unlink = Path.write_text, Path.unlink

    def write_text(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self in paths:
            real_write(self, data, *args, **kwargs)
            raise KeyboardInterrupt
        return real_write(self, data, *args, **kwargs)

    def locked(self: Path, missing_ok: bool = False) -> None:
        if self in paths:
            raise PermissionError("held open")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "unlink", locked)
    engine = _real_engine(lambda: backend)
    engine._model = SimpleNamespace(path=str(tmp_path / "model.gguf"), name="m")
    with pytest.raises(KeyboardInterrupt):
        engine.start(DEFAULT)
    assert paths[0].exists() and backend.undeleted_key_files == [paths[0]] and _closed(fds[0])
    cleanup = engine.stop()
    assert not cleanup.clean and cleanup.credential_removed is False
    retry = engine.retry_held()
    assert retry["released"] is False and retry["password_files_left"] == [str(paths[0])]
    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert engine.retry_held()["released"] is True and not paths[0].exists()


def test_review5_2_a_current_key_file_still_held_counts_as_not_released(tmp_path: Path) -> None:
    class Backend:
        unconfirmed_pid = None
        undeleted_key_files: list[Path] = []
        api_key_file = tmp_path / "still-held.txt"

        def start(self, model: Any, params: Any) -> None:
            pass

        def is_active(self) -> bool:
            return False

    engine = _real_engine(Backend)
    engine.start(DEFAULT)
    cleanup = engine.stop()
    assert not cleanup.clean and "still-held.txt" in (cleanup.problem or "")


# --------------------------------------------------------------------------- #
# Review round 6                                                              #
# --------------------------------------------------------------------------- #
def test_review6_1_a_cleanup_during_password_file_preparation_never_leaves_an_untracked_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The reviewer's creation race: the start pauses just before writing its password
    file and a public cleanup() arrives. The cleanup now waits until the file is complete
    and on record, then deletes it; the start never writes it again, is cancelled before
    any process starts, and the adapter's clean report is true."""
    import threading
    from types import SimpleNamespace

    from inferenceos.backends import resident_server as rs

    backend, fds, paths = _start_preparation(tmp_path, monkeypatch)
    starts: list[int] = []
    cleanup_done = threading.Event()

    class Process:
        pid = 777777

        def start(self) -> int:
            starts.append(1)
            return self.pid

        def stop(self, **kwargs: Any) -> Any:
            return (0, "never started", True)

    def factory(*args: Any, **kwargs: Any) -> Any:
        assert cleanup_done.wait(5)  # deterministic: the cleanup finishes before the process exists
        return Process()

    backend._process_factory = factory
    entered, release = threading.Event(), threading.Event()
    real_write = Path.write_text

    def paused_write(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self in paths:
            entered.set()
            assert release.wait(5)
        return real_write(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", paused_write)
    monkeypatch.setattr(rs, "wait_process_exit", lambda pid, timeout_s=0: True)
    engine = _real_engine(lambda: backend)
    engine._model = SimpleNamespace(path=str(tmp_path / "model.gguf"), name="m")
    caught: list[BaseException] = []

    def run_start() -> None:
        try:
            engine.start(DEFAULT)
        except BaseException as exc:  # noqa: BLE001
            caught.append(exc)

    def run_cleanup() -> None:
        backend.cleanup()
        cleanup_done.set()

    first = threading.Thread(target=run_start)
    first.start()
    assert entered.wait(5)
    second = threading.Thread(target=run_cleanup)
    second.start()
    second.join(0.3)
    assert second.is_alive()  # waits for the password file's preparation, never interleaves with it
    release.set()
    first.join(5)
    second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert len(caught) == 1 and isinstance(caught[0], rs.ResidentStartCancelled)
    assert starts == []  # no process was started
    assert len(paths) == 1 and not paths[0].exists() and _closed(fds[0])
    assert backend.state == "idle" and backend.api_key_file is None and backend.undeleted_key_files == []
    cleanup = engine.stop()
    assert cleanup.clean and engine.retry_held() is None


def test_review6_1_a_start_already_cancelled_creates_no_password_file(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from inferenceos.backends import resident_server as rs

    backend, fds, paths = _start_preparation(tmp_path, monkeypatch)
    real_snapshot = rs.resource_snapshot

    def cancel_on_first_snapshot(pid: Any) -> Any:
        backend._cancel.set()  # a stop arrives before the password file is prepared
        return real_snapshot(pid)

    monkeypatch.setattr(rs, "resource_snapshot", cancel_on_first_snapshot)
    engine = _real_engine(lambda: backend)
    engine._model = SimpleNamespace(path=str(tmp_path / "model.gguf"), name="m")
    with pytest.raises(rs.ResidentStartCancelled):
        engine.start(DEFAULT)
    assert paths == [] and backend.state == "idle"
    assert engine.stop().clean


@pytest.mark.parametrize("where", ["terminate", "confirm exit", "delete password file"])
def test_review6_2_a_new_interruption_during_failed_start_cleanup_keeps_everything_on_record(
        where: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The narrowed guarantee: a new Ctrl+C during the failed start's cleanup propagates,
    with the original failure as its context; the backend is idle, what was not released
    stays on record, and a later cleanup releases it."""
    from types import SimpleNamespace

    from inferenceos.backends import resident_server as rs

    backend = rs.LlamaServerResidentBackend(server=Path("not-executed.exe"))
    backend.exit_timeout_s = backend.previous_gone_timeout_s = 0.01
    key = tmp_path / "inferenceos-key-unwind.txt"
    original = RuntimeError("the original start failure")

    def stop(**kwargs: Any) -> Any:
        if where == "terminate":
            raise KeyboardInterrupt
        return (0, "terminated", True)

    def failing_start(model: Any, params: Any) -> Any:
        key.write_text("secret\n", encoding="utf-8")
        backend._key_file = key
        backend._owned = SimpleNamespace(pid=888888, stop=stop)  # type: ignore[assignment]
        raise original

    backend._start = failing_start  # type: ignore[method-assign]

    def exit_check(pid: Any, timeout_s: float = 0) -> bool:
        if where == "confirm exit":
            raise KeyboardInterrupt
        return True

    real_unlink = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if where == "delete password file" and self == key:
            raise KeyboardInterrupt
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(rs, "wait_process_exit", exit_check)
    with patch.object(Path, "unlink", unlink):
        with pytest.raises(KeyboardInterrupt) as raised:
            backend.start(object(), object())  # type: ignore[arg-type]
    assert raised.value.__context__ is original
    assert backend.state == "idle" and backend._idle.is_set()
    if where in ("terminate", "confirm exit"):
        assert backend.unconfirmed_pid == 888888  # recorded before the stop was asked
    if where == "delete password file":
        assert backend.undeleted_key_files == [key] and key.exists()
    monkeypatch.setattr(rs, "wait_process_exit", lambda pid, timeout_s=0: True)
    backend.cleanup()
    assert backend.confirm_previous_gone(0.01) is True
    assert not key.exists() and backend.undeleted_key_files == [] and backend.unconfirmed_pid is None
