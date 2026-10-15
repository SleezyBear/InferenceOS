"""C4-3 hints change traversal, never eligibility, live evidence or acceptance."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from inferenceos.tune.history_candidates import HistoricalCandidates
from inferenceos.tune.history_cost import HistoricalCosts
from inferenceos.tune.planner import Planner
from tests.test_budgeted_tune import Clock, _world
from tests.test_history_cost import identity
from tests.test_search_planner import _structure
from scripts.evaluate_history_candidates import run_case, train


@pytest.fixture(scope="module")
def records(tmp_path_factory: pytest.TempPathFactory) -> Any:
    clock = Clock()
    with patch("time.perf_counter", clock):
        return train(tmp_path_factory.mktemp("candidate-training"), clock)


def inputs(records: Any) -> tuple[Any, ...]:
    run = records[0]
    events = [e for e in run.comparisons if e["category"] == "placement" and e["step"] == 1]
    ops = {o.operation_id: o for o in run.observations}
    ref = deepcopy(ops[events[0]["reference"]["measurement"]].request)
    candidates = {e["value"]: deepcopy(ops[e["candidate"]["measurement"]].request) for e in events}
    return "placement", tuple(e["value"] for e in events), ref, candidates, events[0]["goal"], 0.015


def rank(records: Any, args: Any) -> Any:
    snapshot = HistoricalCosts(records, identity(), origin="simulation")
    return HistoricalCandidates(snapshot).rank(*args)


def test_actual_persisted_comparisons_promote_only_existing_values(records: Any) -> None:
    args = inputs(records)
    ordered, event = rank(records, args)
    assert args[1] == (25, 0, 1) and ordered == (1, 25, 0)
    assert event["hints"]["1"]["run_count"] == 3
    assert all(c["reference_operation"] != c["candidate_operation"] for c in event["hints"]["1"]["comparisons"])
    assert set(ordered) == set(args[1])
    assert rank(records[:2], args)[0] == args[1]
    assert rank((records[0],) * 3, args)[0] == args[1]


@pytest.mark.parametrize("field,value", [
    ("hint_version", None), ("goal", None), ("goal", {}), ("path", "placement"),
    ("question", "another-workload"), ("gain", None), ("gain", float("nan")),
    ("gain", True), ("raw_noise", -1.0), ("raw_noise", float("inf")),
    ("kind", "final-side"), ("value", True),
])
def test_invalid_or_legacy_comparisons_cannot_hint(records: Any, field: str, value: Any) -> None:
    changed = tuple(r.model_copy(update={"comparisons": tuple({**e, field: value} for e in r.comparisons)}) for r in records)
    args = inputs(records)
    assert rank(changed, args)[0] == args[1]


@pytest.mark.parametrize("mutation", ["missing-op", "same-op", "wrong-ref", "wrong-candidate",
    "failed", "incomplete", "final-side", "retry", "invalid-answer", "missing-phase", "wrong-workload", "different-engine"])
def test_measurement_links_and_clean_completed_requests_are_required(records: Any, mutation: str) -> None:
    changed = []
    for original in records:
        r = original.model_copy(deep=True)
        for e in r.comparisons:
            if mutation == "missing-op":
                e["candidate"]["measurement"] = 999999
            elif mutation == "same-op":
                e["candidate"]["measurement"] = e["reference"]["measurement"]
            elif mutation in ("wrong-ref", "wrong-candidate"):
                side = "reference" if mutation == "wrong-ref" else "candidate"
                e[side]["configuration"]["threads"] = 999
        ops = []
        for op in r.observations:
            updates: dict[str, Any] = {}
            if mutation in ("failed", "incomplete"):
                updates["status"] = mutation
            elif mutation == "final-side":
                updates["kind"] = "final-side"
            elif mutation == "retry":
                updates["retries"] = 1
            elif mutation == "invalid-answer":
                updates["phases"] = {"raw": {"invalid_runs": 1}}
            elif mutation == "missing-phase":
                updates["phases"] = {"raw": None}
            elif mutation == "wrong-workload":
                op.request["workload"]["sha256"] = "f" * 64
            elif mutation == "different-engine" and op.request.get("runtime_identity"):
                op.request["runtime_identity"]["modules"]["engine.dll"] = "f" * 64
            ops.append(op.model_copy(update=updates))
        changed.append(r.model_copy(update={"observations": tuple(ops)}))
    args = inputs(records)
    assert rank(tuple(changed), args)[0] == args[1]


@pytest.mark.parametrize("mutation", ["stale", "failed", "cancelled", "unconfirmed", "unknown", "driver", "context"])
def test_failed_stale_unknown_or_changed_conditions_do_not_promote(records: Any, mutation: str) -> None:
    changed = []
    for r in records:
        changes: dict[str, Any] = {}
        if mutation == "stale":
            changes["started_at"] = datetime.now(timezone.utc) - timedelta(days=31)
        elif mutation in ("failed", "cancelled", "unconfirmed"):
            changes["disposition"] = {"outcome": mutation}
        elif mutation == "unknown":
            changes["origin"] = "unknown"
        else:
            i = deepcopy(r.identity)
            if mutation == "driver":
                i["machine"]["graphics_drivers"] = ["different"]
            else:
                i["context_length"] = 4096
            changes["identity"] = i
        changed.append(r.model_copy(update=changes))
    args = inputs(records)
    assert rank(tuple(changed), args)[0] == args[1]


def test_changed_base_and_candidate_scope_and_goal_do_not_reuse(records: Any) -> None:
    for field in ("gpu_offload_layers", "threads", "ubatch_size"):
        args = list(inputs(records))
        args[2]["configuration"][field] = 999
        for req in args[3].values():
            req["configuration"][field] = 999
        assert rank(records, args)[0] == args[1]
    args = list(inputs(records))
    args[4] = {**args[4], "first": {"different-goal": 1.0}}
    assert rank(records, args)[0] == args[1]


def test_new_negative_readings_override_old_success_without_blacklisting(records: Any) -> None:
    changed = []
    for r in records:
        events = deepcopy(r.comparisons)
        for e in events:
            e["gain"] = -0.5
        changed.append(r.model_copy(update={"comparisons": events}))
    args = inputs(records)
    assert rank(tuple(changed), args)[0] == args[1]  # values stay eligible
    # Many duplicate comparisons in one run still cannot make three independent runs.
    one = records[0].model_copy(update={"comparisons": records[0].comparisons * 50})
    assert rank((one,), args)[0] == args[1]


@pytest.mark.parametrize("bad", [(1,), (0, 0), (False, 1), (1, 0, 999), [1, 0]])
def test_planner_rejects_non_permutation_hints(bad: Any) -> None:
    p = Planner(_structure(), 33, 6, 512, lambda layers: 0.8, candidate_order=lambda step: bad)
    step = p.next_step()
    assert step is not None and step.values == (0, 1)
    assert len(p._planned) == 2 and p.readings == {"placement": [], "threads": [], "batch": []}
    assert any("ordering unavailable" in entry for entry in p.log)


def test_hint_exceptions_fall_back_but_keyboard_interrupt_propagates() -> None:
    error = KeyboardInterrupt("stop")
    def fail(step: Any) -> Any:
        raise error
    p = Planner(_structure(), 33, 6, 512, lambda layers: 0.8, candidate_order=fail)
    with pytest.raises(KeyboardInterrupt) as caught:
        p.next_step()
    assert caught.value is error and not p._planned
    def ordinary(step: Any) -> Any:
        raise ValueError("history unavailable")
    p = Planner(_structure(), 33, 6, 512, lambda layers: 0.8, candidate_order=ordinary)
    step = p.next_step()
    assert step is not None and step.values == (0, 1)


@pytest.mark.parametrize("test_name", [
    "test_review8_old_third_place_reading_is_never_chosen_after_a_placement_change",
    "test_threads_are_rechecked_after_batch_changes_their_setup",
])
def test_interaction_invariants_with_permuted_steps(test_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import tests.test_search_planner as existing
    original = Planner.__post_init__
    def init(p: Planner) -> None:
        original(p)
        p.candidate_order = lambda step: tuple(reversed(step.values))
    monkeypatch.setattr(Planner, "__post_init__", init)
    getattr(existing, test_name)()


def test_second_tune_tests_preferred_candidate_earlier_with_same_captured_base(records: Any, tmp_path: Path) -> None:
    clock = Clock()
    with patch("time.perf_counter", clock):
        _, control, _ = run_case(tmp_path / "control", clock, records, 1100, ordering=False)
        tuner, d, result = run_case(tmp_path / "hints", clock, records, 1100, ordering=True)
    assert control.steps[0]["values"] == [25, 0, 1]
    assert d.steps[0]["values"] == [1, 25, 0]
    assert d._candidate_hint_events
    first = [e for e in d.screenings if e["step"] == 1]
    assert [e["value"] for e in first] == [1, 25, 0]
    assert all(e["reference"]["setup"] == d.steps[0]["base"] for e in first)
    ops = {o.operation_id: o for o in tuner._learning.observations}
    assert all(e["candidate"]["measurement"] in ops for e in first)
    assert result.saved_setup is not None and result.final_configuration.gpu_offload_layers == 1
    assert sum(r.get("round") == "final check" for r in result.rounds) == 1


@pytest.mark.parametrize("mode", ["flat", "stuck", "cancel", "error"])
def test_past_success_cannot_save_after_bad_final_check(records: Any, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    from inferenceos.tune.pipeline import Tuner
    clock = Clock()
    original = Tuner._check_whole_setup
    error = RuntimeError("injected final failure")
    def final(t: Any, *args: Any, **kwargs: Any) -> Any:
        if mode == "error":
            raise error
        if mode == "cancel":
            t.harness._driver.request_cancel("stop")
        else:
            t.harness._inner._responder = _world(clock, flat=True,
                hang=(lambda *args: 10000) if mode == "stuck" else None)
        return original(t, *args, **kwargs)
    monkeypatch.setattr(Tuner, "_check_whole_setup", final)
    with patch("time.perf_counter", clock):
        _, d, result = run_case(tmp_path, clock, records, 2200, ordering=True)
    assert d._candidate_hint_events and result.saved_setup is None
    assert sum(r.get("round") == "final check" for r in result.rounds) <= 1


def test_reversed_world_remeasures_and_rejects_the_old_favourite(records: Any, tmp_path: Path) -> None:
    clock = Clock()
    with patch("time.perf_counter", clock):
        _, d, result = run_case(tmp_path, clock, records, 2200, ordering=True, reversed_world=True)
    assert d.steps[0]["values"][0] == 1
    assert next(e for e in d.screenings if e["category"] == "placement" and e["value"] == 1)["gain"] < 0
    assert result.final_configuration.gpu_offload_layers == 25


def test_fresh_process_can_read_and_rank_persisted_history(records: Any, tmp_path: Path) -> None:
    import json
    import subprocess
    import sys
    from inferenceos.tune.learning_store import TuneHistoryStore
    store = TuneHistoryStore(tmp_path)
    for record in records:
        store.append(record)
    before = store.path.read_bytes()
    code = """import json,sys
from pathlib import Path
from inferenceos.tune.learning_store import TuneHistoryStore
from tests.test_history_candidates import inputs,rank
r=TuneHistoryStore(Path(sys.argv[1])).recent()
print(json.dumps(rank(r,inputs(r))[0]))
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(tmp_path)],
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [1, 25, 0]
    assert store.path.read_bytes() == before


def test_unknown_runtime_or_missing_workload_keeps_original_order(records: Any, tmp_path: Path) -> None:
    from inferenceos.tune.planner import Step
    clock = Clock()
    with patch("time.perf_counter", clock):
        tuner, driver, _ = run_case(tmp_path, clock, records, 1100)
    assert driver._candidate_hint_events
    first = driver.steps[0]
    step = Step("placement", (25, 0, 1), "test", tuple(first["base"]))
    tuner.history_runtime_valid = lambda observed: False
    assert driver._candidate_order(step) == step.values
    tuner.history_runtime_valid = lambda observed: True
    driver._long = None
    step = Step("batch", (1024, 2048), "test", tuple(first["base"]))
    assert driver._candidate_order(step) == step.values


@pytest.mark.parametrize("name", [
    "test_review9_earlier_complete_setup_is_kept_as_a_candidate",
    "test_monitor_round2_unresolved_cleanup_mid_search_fails_and_saves_nothing",
])
def test_existing_complete_tune_invariants_with_live_hints(records: Any, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    import tests.test_budgeted_tune as existing
    from tests.test_history_cost import runtime
    original = existing._practical
    drivers = []
    def wired(*args: Any, **kwargs: Any) -> Any:
        tuner, store, driver, dogs = original(*args, **kwargs)
        tuner.history_identity, tuner.history_origin = identity(), "simulation"
        tuner.history_records = records
        tuner.history_runtime_reader = runtime
        tuner.history_runtime_valid = lambda observed: True
        driver.apply_candidate_hints = True
        drivers.append(driver)
        return tuner, store, driver, dogs
    monkeypatch.setattr(existing, "_practical", wired)
    clock = Clock()
    with patch("time.perf_counter", clock):
        if name.startswith("test_monitor"):
            getattr(existing, name)(tmp_path)
        else:
            getattr(existing, name)(tmp_path, clock)
    assert any(d._candidate_hint_events for d in drivers)


@pytest.mark.parametrize("target", [760, 800])
def test_drift_can_cost_a_confirmation_opportunity_but_never_bypass_checks(
        records: Any, tmp_path: Path, target: int) -> None:
    # These are deliberately retained negative evidence, not a benefit assertion.
    # Original timing tests assumed the default winner-first order; a stale hint can
    # use that time. Do not relax the target, thresholds or final check to hide it.
    clock = Clock()
    with patch("time.perf_counter", clock):
        _, control, before = run_case(tmp_path / "control", clock, records, target,
                                     ordering=False, slower_drift=True)
        _, driver, after = run_case(tmp_path / "hint", clock, records, target, ordering=True, slower_drift=True)
    assert before.saved_setup is not None
    assert after.saved_setup is None and after.outcome == "no change"
    assert driver._candidate_hint_events and driver.steps[0]["values"][0] == 1
    assert not any(r.get("round") == "final check" for r in after.rounds)
    search = next(r for r in after.rounds if r.get("round") == "practical search")
    assert search["stopped_for"].startswith("target")


@pytest.mark.parametrize("target,reversed_world,slower_drift", [
    *((target, reverse, False) for target in (700, 1100, 2200, 6000) for reverse in (False, True)),
    (760, False, True), (800, False, True),
])
def test_default_shadow_preserves_control_measurements_and_save_opportunity(
        records: Any, tmp_path: Path, target: int, reversed_world: bool, slower_drift: bool) -> None:
    clock = Clock()
    with patch("time.perf_counter", clock):
        tc, dc, control = run_case(tmp_path / "control", clock, records, target, ordering=False,
                                  reversed_world=reversed_world, slower_drift=slower_drift)
        ts, ds, shadow = run_case(tmp_path / "shadow", clock, records, target,
                                 reversed_world=reversed_world, slower_drift=slower_drift)
    assert not ds.apply_candidate_hints
    assert ds.steps == dc.steps and ds.screenings == dc.screenings
    assert shadow.outcome == control.outcome
    assert shadow.final_configuration == control.final_configuration
    assert (shadow.saved_setup is not None) == (control.saved_setup is not None)
    assert shadow.confirmation == control.confirmation
    assert next(r for r in shadow.rounds if r.get("round") == "time budget") == next(
        r for r in control.rounds if r.get("round") == "time budget")
    assert ts._learning.observations == tc._learning.observations
    event = ds._candidate_hint_events[0]
    assert event["mode"] == "shadow" and not event["applied"]
    assert event["before"] == event["after"] == [25, 0, 1]
    assert event["proposed_order"] == [1, 25, 0]
    if slower_drift:
        assert shadow.saved_setup is not None and shadow.final_configuration.gpu_offload_layers == 25


@pytest.mark.parametrize("outcome", ["not confirmed", "failed", "cancelled", "unknown"])
def test_new_clean_negatives_suppress_old_positives_regardless_of_run_outcome(
        records: Any, tmp_path: Path, outcome: str) -> None:
    from inferenceos.tune.learning_store import TuneHistoryStore
    now = datetime.now(timezone.utc)
    old = tuple(r.model_copy(update={"started_at": now - timedelta(days=2)}) for r in records)
    negative = []
    for index in range(6):
        r = records[index % 3].model_copy(deep=True)
        for e in r.comparisons:
            e["gain"] = -0.5
        negative.append(r.model_copy(update={"run_id": f"negative-{index}",
            "started_at": now - timedelta(minutes=index + 1), "disposition": {"outcome": outcome}}))
    store = TuneHistoryStore(tmp_path)
    for r in (*negative, *old):
        store.append(r)
    args = inputs(records)
    ordered, event = rank(store.recent(), args)
    assert ordered == args[1]
    assert event["hints"]["1"]["run_count"] == 9
    assert event["hints"]["1"]["median_margin"] < 0
    assert sum(c["may_promote"] for c in event["hints"]["1"]["comparisons"]) == 3


@pytest.mark.parametrize("failure", ["missing-gain", "missing-link", "failed", "incomplete", "retry", "invalid"])
def test_latest_unusable_attempt_does_not_revive_old_positive(records: Any, failure: str) -> None:
    changed = []
    for r in records:
        e = deepcopy(next(e for e in r.comparisons if e["category"] == "placement" and e["value"] == 1))
        e["kind"] = "weak repeat"
        op = next(o for o in r.observations if o.operation_id == e["candidate"]["measurement"])
        number = max(o.operation_id for o in r.observations) + 1
        updates: dict[str, Any] = {"operation_id": number}
        e["candidate"]["measurement"] = number
        if failure == "missing-gain":
            e["gain"] = None
        elif failure == "missing-link":
            e["candidate"]["measurement"] = None
        elif failure in ("failed", "incomplete"):
            updates["status"] = failure
        elif failure == "retry":
            updates["retries"] = 1
        else:
            updates["phases"] = {"raw": {"invalid_runs": 1}}
        changed.append(r.model_copy(update={"comparisons": (*r.comparisons, e),
            "observations": (*r.observations, op.model_copy(update=updates))}))
    args = inputs(records)
    ordered, event = rank(tuple(changed), args)
    assert ordered == args[1] and "1" not in event["hints"]
    assert set(event["suppressed_latest_attempts"]["1"]) == {r.run_id for r in records}
    # Failure is not a permanent blacklist. A later clean comparison can contribute.
    recovered = []
    for r, original in zip(changed, records):
        clean = deepcopy(next(e for e in original.comparisons if e["category"] == "placement" and e["value"] == 1))
        recovered.append(r.model_copy(update={"comparisons": (*r.comparisons, clean)}))
    assert rank(tuple(recovered), args)[0] == (1, 25, 0)


def test_failed_fresh_weak_repeat_from_complete_tunes_cannot_promote(tmp_path: Path) -> None:
    from dataclasses import replace
    import hashlib
    from inferenceos.contracts.tune_learning import TuneHistoryRun
    from tests.test_budgeted_tune import _practical
    from tests.test_history_cost import runtime
    from tests.experiment_helpers import SampleResult
    clock = Clock()
    runs = []
    with patch("time.perf_counter", clock):
        for index in range(3):
            state = {"repeat": False}
            normal = _world(clock, answer_s=5, long_answer_s=10,
                factor=lambda layers, threads, batch, long: 1.04 if layers == 1 else 1.0)
            def responder(task: Any, options: Any, call: Any) -> Any:
                sample = normal(task, options, call)
                return SampleResult(fail=True) if state["repeat"] and options.gpu_offload_layers == 1 else sample
            tuner, _, driver, _ = _practical(tmp_path / str(index), responder, 12000, clock=clock)
            tuner.rules = replace(tuner.rules, effort=tuner.rules.effort.model_copy(
                update={"minimum_improvement": 0.05}))
            tuner.history_identity, tuner.history_origin = identity(), "simulation"
            tuner.history_runtime_reader = runtime
            tuner.history_runtime_valid = lambda observed: True
            original = driver._repeat_weak
            def repeat(planner: Any) -> str:
                state["repeat"] = True
                try:
                    return original(planner)
                finally:
                    state["repeat"] = False
            with patch.object(driver, "_repeat_weak", repeat):
                result = driver.run()
            assert result.outcome == "no change" and result.saved_setup is None
            comparisons = [e for e in tuner._learning.comparisons if e["category"] == "placement" and e["value"] == 1]
            assert comparisons[0]["gain"] > 0
            assert comparisons[-1]["kind"] == "weak repeat" and comparisons[-1]["gain"] is None
            runs.append(TuneHistoryRun(run_id=result.run_id, started_at=datetime.now(timezone.utc), origin="simulation",
                identity=identity(), observations=tuple(tuner._learning.observations),
                comparisons=tuple(tuner._learning.comparisons), disposition={"outcome": result.outcome},
                source_report_sha256=hashlib.sha256(result.to_json().encode()).hexdigest()))
    args = inputs(runs)  # current minimum 1.5%, past minimum 5%
    ordered, event = rank(tuple(runs), args)
    assert ordered == args[1] and "1" not in event["hints"]
    assert len(event["suppressed_latest_attempts"]["1"]) == 3


def test_unconfirmed_positive_cannot_supply_third_promotion_run(records: Any) -> None:
    changed = (records[0], records[1], records[2].model_copy(update={"disposition": {"outcome": "not confirmed"}}))
    args = inputs(records)
    ordered, event = rank(changed, args)
    assert event["hints"]["1"]["median_margin"] > 0
    assert event["hints"]["1"]["promotion_run_count"] == 2
    assert ordered == args[1]


@pytest.mark.parametrize("scope", ["base", "candidate", "question", "goal"])
def test_later_failure_for_different_comparison_does_not_erase_reading(records: Any, scope: str) -> None:
    changed = []
    for r in records:
        failed = deepcopy(next(e for e in r.comparisons if e["category"] == "placement" and e["value"] == 1))
        failed.update(kind="weak repeat", gain=None)
        if scope in ("base", "candidate"):
            failed["reference" if scope == "base" else "candidate"]["configuration"]["threads"] = 999
        elif scope == "question":
            failed["question"] = "different-workload"
        else:
            failed["goal"] = {"different": "goal"}
        changed.append(r.model_copy(update={"comparisons": (*r.comparisons, failed)}))
    assert rank(tuple(changed), inputs(records))[0] == (1, 25, 0)
