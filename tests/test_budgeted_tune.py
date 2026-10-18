"""Experimental practical search (``tune/budgeted.py``): handoff of 6 Oct 2026 and the
review-8 corrections (soft target, best-so-far, sizing from the whole setup, weak
candidates, scoped evidence, cancellation everywhere, timing attribution)."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from inferenceos.contracts.tune import ChangeDecision
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.tune.budgeted import BudgetedTune
from inferenceos.tune.layer_top import LoadAttempt
from tests.experiment_helpers import SampleResult, ScriptedHarness
from tests.test_tune import CONTEXT, _settings, _tuner


class Clock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds


class ManualWatchdog:
    """A watchdog on the simulated clock: checked whenever simulated time advances."""

    def __init__(self, clock: Clock, on_expire: Callable[[str], None]) -> None:
        self.clock, self.on_expire = clock, on_expire
        self.armed: tuple[float, str] | None = None
        self.fired = False

    def arm(self, at: float, label: str) -> None:
        self.armed, self.fired = (at, label), False

    def tick(self) -> None:
        if self.armed is not None and self.clock.seconds >= self.armed[0]:
            label = self.armed[1]
            self.clock.seconds = self.armed[0]  # stopped at its deadline
            self.armed, self.fired = None, True
            self.on_expire(label)

    def disarm(self) -> bool:
        fired, self.armed, self.fired = self.fired, None, False
        return fired

    def close(self) -> None:
        pass


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[Clock]:
    c = Clock()
    monkeypatch.setattr(time, "perf_counter", c)
    yield c


class LoadingHarness(ScriptedHarness):
    """Every warm-up is a fresh load that costs simulated seconds."""

    def __init__(self, responder: Any, clock: Clock, load_s: float, dog: list[ManualWatchdog]) -> None:
        super().__init__(responder)
        self.clock, self.load_s, self.dog = clock, load_s, dog

    def run_once(self, task: Any, **kwargs: Any) -> Any:
        if kwargs.get("warmup"):
            self.clock.seconds += self.load_s
        run = super().run_once(task, **kwargs)
        for dog in self.dog:
            dog.tick()
        return run


def _world(clock: Clock | None, *, fails: set[int] | None = None, flat: bool = False,
           answer_s: float = 10.0, long_answer_s: float = 30.0, threads_gain: dict[int, float] | None = None,
           hang: Callable[[ProfileTask, ProfilerOptions], float] | None = None,
           fast: Callable[[int], bool] = lambda layers: layers >= 20,
           factor: Callable[[int, int | None, int | None, bool], float] | None = None) -> Any:
    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        assert task.context_length == CONTEXT
        long = task.workload.workload_id.startswith("long-")
        if clock is not None:
            clock.seconds += long_answer_s if long else answer_s
            if hang is not None:
                clock.seconds += hang(task, options)
        layers = options.gpu_offload_layers or 0
        if fails and layers in fails:
            return SampleResult(fail=True)
        speed, latency = 20.0, 2.0
        if not flat and fast(layers):
            speed, latency = 24.0, 1.7
        scale = 1.0 + (threads_gain or {}).get(options.threads or 0, 0.0)
        if factor is not None:
            speed, latency = 20.0, 2.0
            scale = factor(layers, options.threads, options.ubatch_size, long)
        return SampleResult(decode_tps=speed * scale, total_latency_s=latency / scale)
    return responder


def _practical(tmp_path: Path, responder: Any, target: float, *, clock: Clock | None = None,
               answers: list[bool] | None = None, load_s: float = 0.0, **settings: Any) -> tuple[Any, Any, BudgetedTune, list[ManualWatchdog]]:
    s = _settings(tmp_path, tune_experimental_budget_s=target, tune_heat_check_runs=0, **settings)
    tuner, _, _, store = _tuner(tmp_path, responder, answers=answers or [True] * 8, settings=s)
    dogs: list[ManualWatchdog] = []
    if clock is not None:
        tuner.harness = LoadingHarness(responder, clock, load_s, dogs)

    def factory(on_expire: Callable[[str], None]) -> ManualWatchdog:
        dog = ManualWatchdog(clock or Clock(), on_expire)
        dogs.append(dog)
        return dog

    driver = BudgetedTune(tuner, target, watchdog_factory=factory)
    return tuner, store, driver, dogs


def _round(result: Any, name: str) -> dict[str, Any]:
    return next(item for item in result.rounds if item.get("round") == name)


def _saved(tuner: Any, store: Any) -> Any:
    return store.find(tuner.model_path, context_length=CONTEXT, power=tuner.machine.power_source)


# --------------------------------------------------------------------------- #
def test_budget_zero_keeps_the_normal_tune(tmp_path: Path) -> None:
    s = _settings(tmp_path, tune_heat_check_runs=0)
    tuner, _, _, _ = _tuner(tmp_path, _world(None), answers=[True] * 8, settings=s)
    result = tuner.run()
    assert not any(item.get("round") in ("time budget", "practical search") for item in result.rounds)


def test_one_final_confirmation_with_a_fixed_look(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    result = driver.run()
    assert result.outcome == "saved"
    assert result.final_configuration.gpu_offload_layers == 25
    final = _round(result, "final check")
    blocks = final["blocks_per_question"]["short"]
    assert final["plan"] == {"short": "better"}
    assert sum(1 for item in result.rounds if item.get("round") == "final check") == 1
    fair = final["checks"][0]["fair"]
    assert fair["looks"] == [blocks] and fair["alpha_per_look"] == pytest.approx(0.05)
    assert _saved(tuner, store) is not None
    assert all(change.measured_gain is None for change in result.proposed_changes)


def test_review8_best_found_early_is_kept_when_exploration_stops(tmp_path: Path, clock: Clock) -> None:
    # Review 8: 800 s found +17.5% placement after 120 s, then returned "no change".
    tuner, store, driver, _ = _practical(tmp_path, _world(clock), 800.0, clock=clock)
    result = driver.run()
    search = _round(result, "practical search")
    assert search["stopped_for"].startswith("target")
    assert result.outcome == "saved"
    assert result.final_configuration.gpu_offload_layers == 25
    budget = _round(result, "time budget")
    assert budget["used_s"] <= 800.0 + budget["allowed_extension_s"]


def test_soft_target_lets_the_final_check_finish_and_explains_it(tmp_path: Path, clock: Clock) -> None:
    # Target 760 s: placement 25 is found, then exploration stops so the final check
    # (predicted from measured costs) fits. The final check's answers then run slower
    # than measured, so it finishes after the target, inside the extension: the pass is
    # kept and the overrun is explained.
    state = {"final": False}

    def slower(task: ProfileTask, options: ProfilerOptions) -> float:
        return 4.0 if state["final"] else 0.0

    tuner, store, driver, _ = _practical(tmp_path, _world(clock, hang=slower), 760.0, clock=clock)
    original = tuner._check_whole_setup

    def final(*args: Any, **kwargs: Any) -> Any:
        state["final"] = True
        return original(*args, **kwargs)

    tuner._check_whole_setup = final
    result = driver.run()
    budget = _round(result, "time budget")
    assert result.outcome == "saved"
    assert 0 < budget["extension_s"] <= budget["allowed_extension_s"]
    assert any("final check took" in reason for reason in budget["extension_reasons"])


def test_exploration_never_restarts_after_the_target(tmp_path: Path, clock: Clock) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(clock), 800.0, clock=clock)
    result = driver.run()
    search = _round(result, "practical search")
    stop_t = max(float(item["t"]) for item in search["timeline"] if "measured" in item)
    final = _round(result, "final check")
    assert stop_t <= final["started_at_s"]


def test_review8_small_tweak_does_not_make_a_large_gain_unaffordable(tmp_path: Path, clock: Clock) -> None:
    # The whole setup is sized from its own direct reading against the start, never from
    # its smallest component; and when the full candidate's check does not fit (here the
    # batch needs the long question), an affordable subset is confirmed instead.
    world = _world(clock, threads_gain={10: 0.04})
    tuner, store, driver, _ = _practical(tmp_path, world, 900.0, clock=clock)
    result = driver.run()
    search = _round(result, "practical search")
    eligible = [o for o in search["options"] if o["eligible"]]
    assert eligible, search["options"]
    assert result.outcome == "saved"
    assert result.final_configuration.gpu_offload_layers == 25


def test_affordable_alternative_is_chosen_when_the_full_candidate_does_not_fit(tmp_path: Path, clock: Clock) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(clock), 1000.0, clock=clock)
    driver._t0 = 0.0
    driver._hard = 900.0
    driver._save_reserve = 30.0
    options: list[dict[str, object]] = [
        {"setup": (25, 10, 1024), "eligible": True, "value": 0.30, "predicted_s": 2400.0},
        {"setup": (25, 10, 512), "eligible": True, "value": 0.20, "predicted_s": 700.0},
        {"setup": (25, 6, 512), "eligible": True, "value": 0.175, "predicted_s": 600.0},
    ]
    chosen = driver._choose(options)
    assert chosen is not None and chosen["setup"] == (25, 10, 512)
    assert "does not fit" in str(options[0]["why"])


def test_combined_check_is_sized_from_the_whole_setup(tmp_path: Path, clock: Clock) -> None:
    # CPU-only wins (+17.5%), and at CPU-only 10 threads add 4%: the combined setup is
    # measured as a whole, valued and sized from its own reading against the start (not
    # from the smaller thread step), and chosen.
    world = _world(clock, threads_gain={10: 0.04}, fast=lambda layers: layers == 0)
    tuner, store, driver, _ = _practical(tmp_path, world, 100000.0, clock=clock,
                                         optimization_mode="performance_first")
    result = driver.run()
    search = _round(result, "practical search")
    full = next(o for o in search["options"] if o["settings"] == {"gpu_offload_layers": 0, "threads": 10})
    alone = next(o for o in search["options"] if o["settings"] == {"gpu_offload_layers": 0})
    assert full["eligible"] and alone["eligible"]
    assert full["direct"]["short"] > alone["direct"]["short"]
    assert full["plan"]["short"][1] <= alone["plan"]["short"][1]
    assert result.outcome == "saved"
    assert (result.final_configuration.gpu_offload_layers, result.final_configuration.threads) == (0, 10)


def test_review8_consistent_weak_gain_is_confirmed_when_time_allows(tmp_path: Path, clock: Clock) -> None:
    # Review 8: a consistent ~3% gain with a huge allowance ended with no final check.
    # (Performance first: its smallest worthwhile gain, 1.5%, makes 3% worth keeping.)
    world = _world(clock, flat=True, threads_gain={10: 0.03})
    tuner, store, driver, _ = _practical(tmp_path, world, 100000.0, clock=clock,
                                         optimization_mode="performance_first")
    result = driver.run()
    assert any(item.get("round") == "final check" for item in result.rounds)
    assert result.outcome == "saved"
    assert result.final_configuration.threads == 10
    readings = _round(result, "practical search")["readings"]
    assert any(r["category"] == "threads" and r["value"] == 10 and r["repeats"] == 2 for r in readings)


def test_noise_level_weak_reading_does_not_get_a_long_final_check(tmp_path: Path, clock: Clock) -> None:
    # A 1% "gain" (pure noise level) is repeated, stays near the threshold, and is not
    # sent to a 16-block final check.
    world = _world(clock, flat=True, threads_gain={10: 0.012})
    tuner, store, driver, _ = _practical(tmp_path, world, 100000.0, clock=clock,
                                         optimization_mode="performance_first")
    result = driver.run()
    assert not any(item.get("round") == "final check" for item in result.rounds)
    assert result.outcome == "no change"


def test_weak_repeat_is_skipped_when_it_cannot_be_confirmed_in_time(tmp_path: Path, clock: Clock) -> None:
    world = _world(clock, flat=True, threads_gain={10: 0.03})
    tuner, store, driver, _ = _practical(tmp_path, world, 1200.0, clock=clock)
    result = driver.run()
    plan = _round(result, "practical search")["plan"]
    assert not any(line.startswith("repeat the weak") for line in plan) or any(
        item.get("round") == "final check" for item in result.rounds)


def test_missing_evidence_is_never_selected(tmp_path: Path) -> None:
    tuner, _, driver, _ = _practical(tmp_path, _world(None, fails={25}), 3600.0)
    result = driver.run()
    readings = _round(result, "practical search")["readings"]
    assert any(r["category"] == "placement" and r["value"] == 25 and r["gain"] is None for r in readings)
    assert result.final_configuration.gpu_offload_layers != 25


def test_no_improvement_saves_nothing(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None, flat=True), 3600.0)
    result = driver.run()
    assert result.outcome == "no change"
    assert result.saved_setup is None and _saved(tuner, store) is None


def test_review8_user_stop_during_approval_saves_nothing(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)

    class StopWhileAsking:
        def decide(self, change: Any, timeout_s: float = 0.0) -> ChangeDecision:
            driver.request_cancel("user pressed stop while being asked")
            return ChangeDecision(change_id=change.change_id, approved=True, answer="yes")

    tuner.approver = StopWhileAsking()
    result = driver.run()
    assert result.saved_setup is None and _saved(tuner, store) is None
    assert result.outcome.startswith("cancelled by the user")


def test_review8_user_stop_ends_load_only_probing(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    calls: list[int] = []

    def try_load(layers: int) -> LoadAttempt:
        calls.append(layers)
        if len(calls) == 2:
            driver.request_cancel("user pressed stop during load-only tries")
        return LoadAttempt(loaded=True, fits=layers <= 20)

    tuner.try_load = try_load
    result = driver.run()
    assert len(calls) == 2  # no load try was launched after the stop
    assert result.outcome.startswith("cancelled by the user")
    assert _saved(tuner, store) is None


def test_user_stop_during_discovery_publishes_nothing(tmp_path: Path) -> None:
    holder: dict[str, BudgetedTune] = {}
    base = _world(None)

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        if call == 12:
            holder["driver"].request_cancel("user stopped it")
        return base(task, options, call)  # type: ignore[no-any-return]

    tuner, store, driver, _ = _practical(tmp_path, responder, 3600.0)
    holder["driver"] = driver
    result = driver.run()
    assert driver.user_cancelled == "user stopped it"
    assert result.saved_setup is None and _saved(tuner, store) is None


def test_user_stop_before_a_safety_save_saves_nothing(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None, flat=True), 3600.0)
    original = tuner._starting_point

    def start(power: Any) -> Any:
        config, _ = original(power)
        return config.model_copy(update={"gpu_offload_layers": 25}), "saved setup (test)"

    class StopWhileAsking:
        def decide(self, change: Any, timeout_s: float = 0.0) -> ChangeDecision:
            driver.request_cancel("stop")
            return ChangeDecision(change_id=change.change_id, approved=True, answer="yes")

    tuner._starting_point = start
    tuner.try_load = lambda layers: LoadAttempt(loaded=True, fits=layers <= 18)
    tuner.approver = StopWhileAsking()
    result = driver.run()
    assert _saved(tuner, store) is None
    assert result.outcome.startswith("cancelled by the user")


def test_stuck_answer_is_stopped_and_its_evidence_missing(tmp_path: Path, clock: Clock) -> None:
    # The answer for CPU-only (0 layers) hangs for an hour: the watchdog stops it, the
    # reading is missing, exploration ends, and the best found so far is still confirmed.
    def hang(task: ProfileTask, options: ProfilerOptions) -> float:
        return 3600.0 if (options.gpu_offload_layers == 0) else 0.0

    tuner, store, driver, _ = _practical(tmp_path, _world(clock, hang=hang), 100000.0, clock=clock)
    result = driver.run()
    budget = _round(result, "time budget")
    assert budget["stuck"]
    assert "stuck" in _round(result, "practical search")["stopped_for"]
    assert clock.seconds < 3600.0  # stopped at the watchdog's deadline, not after the hang
    assert result.outcome == "saved" and result.final_configuration.gpu_offload_layers == 25


def test_stuck_final_check_is_not_accepted(tmp_path: Path, clock: Clock) -> None:
    state = {"final": False}

    def hang(task: ProfileTask, options: ProfilerOptions) -> float:
        return 5000.0 if state["final"] else 0.0

    tuner, store, driver, _ = _practical(tmp_path, _world(clock, hang=hang), 100000.0, clock=clock)
    original = tuner._check_whole_setup

    def final(*args: Any, **kwargs: Any) -> Any:
        state["final"] = True
        return original(*args, **kwargs)

    tuner._check_whole_setup = final
    result = driver.run()
    assert _round(result, "final check")["stuck"]
    assert result.saved_setup is None and _saved(tuner, store) is None


def test_partial_approval_saves_nothing_and_rechecks_nothing(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0, answers=[False])
    result = driver.run()
    assert result.outcome == "not saved: partial approval"
    assert result.saved_setup is None
    assert sum(1 for item in result.rounds if item.get("round") == "final check") == 1


def test_memory_safety_correction_follows_its_own_check(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None, flat=True), 3600.0)
    original = tuner._starting_point

    def start(power: Any) -> Any:
        config, _ = original(power)
        return config.model_copy(update={"gpu_offload_layers": 25}), "saved setup (test)"

    tuner._starting_point = start
    tuner.try_load = lambda layers: LoadAttempt(loaded=True, fits=layers <= 18)
    result = driver.run()
    assert any(item.get("round") == "memory-safety check" for item in result.rounds)
    assert "memory-safety correction" in result.outcome
    if result.saved_setup is not None:
        assert result.saved_setup.configuration.gpu_offload_layers == 18


def test_final_check_measurements_are_in_the_cost_ledger(tmp_path: Path, clock: Clock) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(clock), 100000.0, clock=clock, load_s=20.0)
    result = driver.run()
    kinds = {op.kind for op in driver.ledger.operations}
    assert "final-side" in kinds
    costs = _round(result, "time budget")["costs"]
    assert costs["seconds_by_kind"]["final-side"] > 0


def _interaction(layers: int, threads: int | None, batch: int | None, long: bool) -> float:
    """CPU-only helps (x1.2). At batch 512, 10 threads add x1.25; at batch 1024 they
    lose (x0.95). Batch 1024 reads the long question x1.4 faster."""
    batch = batch or 512
    out = 1.2 if layers == 0 else 1.0
    t = threads or 6
    if t == 10:
        out *= 1.25 if batch == 512 else 0.95
    if long and batch == 1024:
        out *= 1.4
    return out


def test_review9_earlier_complete_setup_is_kept_as_a_candidate(tmp_path: Path, clock: Clock) -> None:
    # CPU + 10 threads (batch 512) reads about +50% on both questions. Batch 1024 then
    # looks better on the long question at 10 threads, which changes the best thread
    # count (10 loses at batch 1024): the latest combination (CPU + batch 1024) reads
    # less on the goal. The earlier complete setup, already measured on both
    # questions, must stay a candidate and win.
    world = _world(clock, factor=_interaction)
    tuner, store, driver, _ = _practical(tmp_path, world, 100000.0, clock=clock)
    result = driver.run()
    search = _round(result, "practical search")
    earlier = next(o for o in search["options"] if o["settings"] == {"gpu_offload_layers": 0, "threads": 10})
    assert earlier["source"] == "earlier complete setup"
    assert earlier["direct"]["short"] is not None and earlier["direct"]["long"] is not None
    assert result.outcome == "saved"
    assert (result.final_configuration.gpu_offload_layers, result.final_configuration.threads,
            result.final_configuration.ubatch_size) == (0, 10, None)


def test_review9_much_slower_confirmation_stops_incomplete(tmp_path: Path, clock: Clock) -> None:
    # Answers take 30 s instead of 10 once the final check starts: finishing would go far
    # past target + extension, so the check stops at a measurement boundary, incomplete,
    # and nothing is saved (no early acceptance).
    state = {"final": False}

    def slower(task: ProfileTask, options: ProfilerOptions) -> float:
        return 20.0 if state["final"] else 0.0

    tuner, store, driver, _ = _practical(tmp_path, _world(clock, hang=slower), 760.0, clock=clock)
    original = tuner._check_whole_setup

    def final(*args: Any, **kwargs: Any) -> Any:
        state["final"] = True
        return original(*args, **kwargs)

    tuner._check_whole_setup = final
    result = driver.run()
    check = _round(result, "final check")
    budget = _round(result, "time budget")
    assert check["abandoned"]
    assert result.saved_setup is None and _saved(tuner, store) is None
    assert "stopped incomplete" in result.outcome
    # Stopped well before the 26 minutes it would have taken.
    assert budget["used_s"] < 760.0 + 2 * budget["allowed_extension_s"]


def test_review9_slightly_slower_confirmation_still_finishes(tmp_path: Path, clock: Clock) -> None:
    state = {"final": False}

    def slower(task: ProfileTask, options: ProfilerOptions) -> float:
        return 4.0 if state["final"] else 0.0

    tuner, store, driver, _ = _practical(tmp_path, _world(clock, hang=slower), 760.0, clock=clock)
    original = tuner._check_whole_setup

    def final(*args: Any, **kwargs: Any) -> Any:
        state["final"] = True
        return original(*args, **kwargs)

    tuner._check_whole_setup = final
    result = driver.run()
    assert result.outcome == "saved"
    assert _round(result, "final check")["abandoned"] is None


# --------------------------------------------------------------------------- #
# Review 10, item 3: publication is the commit (the atomic replace), not the start of
# saving. A stop before the commit's last check leaves the previous setup unchanged; a
# stop after it keeps the new setup and says so.
def _first_saved(tmp_path: Path) -> tuple[Any, Any, bytes]:
    """A previous saved setup in the file -- kept for the other power source, so the next
    tune does not start from it and still has something new to save."""
    from inferenceos.contracts.enums import PowerState

    first = tmp_path / "first"
    first.mkdir()
    tuner, store, driver, _ = _practical(first, _world(None), 3600.0)
    assert driver.run().outcome == "saved"
    saved = _saved(tuner, store)
    other = PowerState.BATTERY if saved.power_source is not PowerState.BATTERY else PowerState.AC
    # The second tune runs in ``tmp_path`` (its own data, no learned evidence).
    tuner, store, _, _ = _practical(tmp_path, _world(None), 3600.0)
    store.save(saved.model_copy(update={"power_source": other, "setup_id": "setup-previous"}))
    return tuner, store, store.path.read_bytes()


@pytest.mark.parametrize("point", ["preparing", "writing", "written", "committing"])
def test_review10_stop_before_the_commit_leaves_the_previous_setup(tmp_path: Path, point: str) -> None:
    _, _, before = _first_saved(tmp_path)
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    seen: list[str] = []

    def events(name: str) -> None:
        seen.append(name)
        if name == point:
            driver.request_cancel(f"stopped while saving ({name})")

    driver.save_events = events
    result = driver.run()
    assert result.saved_setup is None
    assert result.outcome.startswith("cancelled by the user")
    assert store.path.read_bytes() == before, "the previous saved setup changed"
    assert "committed" not in seen and not driver.cancel_after_publication
    assert not list(store.path.parent.glob("*.tmp")), "a temporary file was left behind"


def test_review10_stop_after_the_commit_keeps_the_new_setup_and_says_so(tmp_path: Path) -> None:
    _, _, before = _first_saved(tmp_path)
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)

    def events(name: str) -> None:
        if name == "committed":
            driver.request_cancel("stopped just after saving")

    driver.save_events = events
    result = driver.run()
    assert result.outcome == "saved" and result.saved_setup is not None
    assert store.path.read_bytes() != before
    assert _saved(tuner, store).setup_id == result.saved_setup.setup_id
    assert driver.cancel_after_publication
    assert _round(result, "time budget")["cancel_after_publication"] is True
    assert any("publication had completed" in note for note in result.notes)


def test_review10_stop_racing_the_commit_is_either_before_or_after(tmp_path: Path) -> None:
    """A stop from another thread while the commit holds its lock waits for it, and is
    then classified as after publication -- never a half state."""
    import threading

    _first_saved(tmp_path)
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    real_replace = __import__("os").replace
    stopper: list[threading.Thread] = []

    def slow_replace(src: Any, dst: Any) -> None:
        thread = threading.Thread(target=driver.request_cancel, args=("racing stop",))
        thread.start()
        stopper.append(thread)
        time.sleep(0.05)  # the stop is now waiting for the commit lock
        real_replace(src, dst)

    import inferenceos.tune.profiles as profiles

    original = profiles.os.replace
    profiles.os.replace = slow_replace  # type: ignore[assignment]
    try:
        result = driver.run()
    finally:
        profiles.os.replace = original
    stopper[0].join(5.0)
    assert result.saved_setup is not None and _saved(tuner, store).setup_id == result.saved_setup.setup_id
    assert driver.cancel_after_publication


@pytest.mark.parametrize("fail_at", ["write", "replace"])
def test_review10_failed_write_or_replace_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: str
) -> None:
    import inferenceos.tune.profiles as profiles

    tuner, store, before = _first_saved(tmp_path)
    previous = store.all()[0]

    def boom(*_: Any, **__: Any) -> None:
        raise OSError(f"disk refused the {fail_at}")

    monkeypatch.setattr(profiles.os, "fsync" if fail_at == "write" else "replace", boom)
    with pytest.raises(OSError):
        store.save(previous.model_copy(update={"setup_id": "setup-new"}))
    assert store.path.read_bytes() == before
    assert not list(store.path.parent.glob("*.tmp"))


# --------------------------------------------------------------------------- #
# Review 10, item 2: the final check is paced from completed comparison work, per
# question and per side, with the save-and-cleanup allowance in the projection.
def _pacer(tmp_path: Path, clock: Clock, *, blocks: dict[str, int], estimate: dict[str, float],
           target: float, hard: float, save: float = 30.0) -> BudgetedTune:
    tuner, _, driver, _ = _practical(tmp_path, _world(None), target)
    driver._t0, driver._target, driver._hard, driver._save_reserve = 0.0, target, hard, save
    driver._final = {"planned_blocks": dict(blocks), "cost": dict(estimate),
                     "ids": {f"{q}-id": q for q in blocks}, "q": {}, "measurements": [], "current": None}
    return driver


def _run_check(driver: BudgetedTune, clock: Clock, *, blocks: dict[str, int],
               cost: Callable[[str, str, int], float], retries: Callable[[str, str, int], int] = lambda *_: 0,
               discard: Callable[[str, int], bool] = lambda *_: False, extra_attempts: int = 2) -> str:
    """Emit the events ``compare_fairly`` sends, question by question; returns how the
    check ended ("finished" or the abandon reason)."""
    from inferenceos.tune.budgeted import Abandoned

    try:
        for q, n in blocks.items():
            usable = discarded = attempts = 0
            limit = n + extra_attempts
            while attempts < limit and usable < n:
                base = {"workload": f"{q}-id", "attempts": attempts, "usable": usable,
                        "discarded": discarded, "planned": n, "max_attempts": limit}
                driver.final_progress({"event": "boundary", **base})
                order = ("start", "candidate") if attempts % 2 == 0 else ("candidate", "start")
                for side in order:
                    tries = retries(q, side, attempts)
                    wall = cost(q, side, attempts) * (1 + tries)
                    clock.seconds += wall
                    driver.final_progress({"event": "measurement", **base, "side": side,
                                           "wall_s": wall, "retries": tries})
                attempts += 1
                if discard(q, attempts):
                    discarded += 1
                else:
                    usable += 1
                driver.final_progress({"event": "block", **base, "attempts": attempts, "usable": usable,
                                       "discarded": discarded, "counted": not discard(q, attempts)})
    except Abandoned as stop:
        return str(stop)
    return "finished"


def test_review10_slow_short_question_does_not_abandon_a_fast_long_one(tmp_path: Path, clock: Clock) -> None:
    """The reviewer's case: 8 short measurements planned at 60 s and 8 long at 180 s;
    the short ones really take 180 s. The true finish (3,080 s) is inside the 3,360 s
    allowance; the old single slowdown factor projected 5,960 s and stopped."""
    blocks = {"short": 4, "long": 4}
    clock.seconds = 170.0
    driver = _pacer(tmp_path, clock, blocks=blocks, estimate={"short": 60.0, "long": 180.0},
                    target=2800.0, hard=3360.0)
    ended = _run_check(driver, clock, blocks=blocks, cost=lambda q, side, i: 180.0)
    assert ended == "finished"
    assert clock.seconds + 30.0 == 3080.0
    projection = driver._final["last_projection"]  # type: ignore[index]
    assert projection["observed_s"]["short current"] == 180.0


def test_review10_slow_long_question_is_projected_from_its_own_pace(tmp_path: Path, clock: Clock) -> None:
    """The opposite: the short question runs at its estimate and the long one at 3x. Its
    slowness is only known once it has been measured, and then it stops the check."""
    blocks = {"short": 4, "long": 4}
    clock.seconds = 170.0
    driver = _pacer(tmp_path, clock, blocks=blocks, estimate={"short": 60.0, "long": 180.0},
                    target=1800.0, hard=2160.0)
    ended = _run_check(driver, clock, blocks=blocks,
                       cost=lambda q, side, i: 60.0 if q == "short" else 540.0)
    assert ended != "finished" and "long new" in ended
    # Stopped at the first boundary after one long block, not during the short question.
    assert driver._final["q"]["short"]["usable"] == 4  # type: ignore[index]
    assert driver._final["q"]["long"]["usable"] == 1  # type: ignore[index]


def test_review10_unequal_sides_are_estimated_separately(tmp_path: Path, clock: Clock) -> None:
    """The new setup loads twice as slowly as the current one: each side is projected
    from its own cost (an average of the two would be the same here, so check the
    record and the projection directly)."""
    blocks = {"short": 6}
    driver = _pacer(tmp_path, clock, blocks=blocks, estimate={"short": 60.0}, target=925.0, hard=1110.0)
    ended = _run_check(driver, clock, blocks=blocks,
                       cost=lambda q, side, i: 60.0 if side == "start" else 120.0)
    assert ended == "finished"  # 6 x (60 + 120) + 30 s save = 1,110 s, exactly the allowance
    projection = driver._final["last_projection"]  # type: ignore[index]
    assert projection["observed_s"] == {"short current": 60.0, "short new": 120.0}


def test_review10_unequal_sides_stop_when_the_new_side_makes_it_unaffordable(tmp_path: Path, clock: Clock) -> None:
    blocks = {"short": 8}
    driver = _pacer(tmp_path, clock, blocks=blocks, estimate={"short": 60.0}, target=960.0, hard=1152.0)
    ended = _run_check(driver, clock, blocks=blocks,
                       cost=lambda q, side, i: 60.0 if side == "start" else 180.0)
    # 8 x 240 = 1,920 s needed against 1,152 s: stopped after the first block.
    assert ended != "finished"
    assert driver._final["q"]["short"]["usable"] == 1  # type: ignore[index]


def test_review10_disturbed_repeats_cost_time_but_not_blocks(tmp_path: Path, clock: Clock) -> None:
    blocks = {"short": 4}
    driver = _pacer(tmp_path, clock, blocks=blocks, estimate={"short": 60.0}, target=900.0, hard=1080.0)
    ended = _run_check(driver, clock, blocks=blocks, cost=lambda q, side, i: 60.0,
                       retries=lambda q, side, i: 1 if (i, side) == (0, "start") else 0)
    assert ended == "finished"
    state = driver._final["q"]["short"]  # type: ignore[index]
    assert state["retries"] == 1 and state["usable"] == 4
    # The per-measurement cost excludes the repeat (60 s, not 120 s).
    assert driver._final["last_projection"]["observed_s"]["short current"] == 60.0  # type: ignore[index]
    assert clock.seconds == 4 * 120.0 + 60.0


def test_review10_discarded_blocks_do_not_reduce_what_is_still_required(tmp_path: Path, clock: Clock) -> None:
    blocks = {"short": 4}
    driver = _pacer(tmp_path, clock, blocks=blocks, estimate={"short": 60.0}, target=600.0, hard=720.0)
    seen: list[dict[str, Any]] = []
    original = driver._project_final

    def spy() -> dict[str, Any]:
        out = original()
        seen.append(out)
        return out

    driver._project_final = spy  # type: ignore[method-assign]
    _run_check(driver, clock, blocks=blocks, cost=lambda q, side, i: 60.0, discard=lambda q, a: a == 1)
    # After block 1 was discarded, still 4 blocks (480 s) to go.
    assert seen[1]["remaining_s"] == 480.0
    assert driver._final["q"]["short"]["discarded"] == 1  # type: ignore[index]


def test_review10_pacing_works_with_no_warm_up_runs(tmp_path: Path, clock: Clock) -> None:
    """The old pacing hooked warm-ups, so with none it never ran. Progress now comes
    from the comparison itself."""
    state = {"final": False}

    def slower(task: ProfileTask, options: ProfilerOptions) -> float:
        return 20.0 if state["final"] else 0.0

    tuner, store, driver, _ = _practical(tmp_path, _world(clock, hang=slower), 760.0, clock=clock)
    tuner.experiment_policy = tuner.experiment_policy.model_copy(update={"warmup_repetitions": 0})
    original = tuner._check_whole_setup

    def final(*args: Any, **kwargs: Any) -> Any:
        state["final"] = True
        return original(*args, **kwargs)

    tuner._check_whole_setup = final
    result = driver.run()
    check = _round(result, "final check")
    assert check["measurements_done"] > 0
    assert check["abandoned"] and result.saved_setup is None


def test_review10_a_completed_passing_check_is_kept_even_when_late(tmp_path: Path, clock: Clock) -> None:
    """No decision after the last block: a check that completes is never thrown away,
    however late it is."""
    blocks = {"short": 2}
    driver = _pacer(tmp_path, clock, blocks=blocks, estimate={"short": 60.0}, target=100.0, hard=400.0)
    ended = _run_check(driver, clock, blocks=blocks, cost=lambda q, side, i: 60.0 if i == 0 else 200.0)
    assert ended == "finished" and clock.seconds > 400.0


# --------------------------------------------------------------------------- #
# Review 10, item 6: eligibility and affordability come before the history is cut.
def test_review10_expensive_candidates_do_not_crowd_out_an_affordable_earlier_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inferenceos.tune.planner import CATEGORIES

    _, _, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    start = (20, 6, 512)
    driver._start_setup = start
    # Seven expensive setups read +20..+26%; the earlier winner reads +12% and is cheap.
    expensive = [(20 + i, 6, 512) for i in range(1, 8)]
    cheap = (20, 8, 512)
    gains = {s: 0.20 + 0.01 * i for i, s in enumerate(expensive)} | {cheap: 0.12}
    costs = {s: 5000.0 for s in expensive} | {cheap: 300.0}
    driver._meas = {(s, "short-short"): None for s in [start, *gains]}  # type: ignore[misc]
    monkeypatch.setattr(driver, "_vs_start", lambda setup, q: gains.get(setup) if q == "short" else None)
    monkeypatch.setattr(driver, "_final_cost", lambda setup, planning=False, assumed=None: costs.get(setup, 0.0))
    monkeypatch.setattr(driver, "_left", lambda: 900.0)

    history = driver._history()
    assert len(history) == 6 and history[0] == cheap

    class Planner:
        readings: dict[str, list[Any]] = {c: [] for c in CATEGORIES}

        def best_so_far(self) -> Any:
            return start

    options = driver._options(Planner())  # type: ignore[arg-type]
    chosen = driver._choose(options)
    assert chosen is not None and chosen["setup"] == cheap
    assert chosen["source"] == "earlier complete setup"
    status = [str(row["status"]) for row in driver._measured_status(chosen)]
    assert len(status) == 8  # every measured setup is accounted for
    assert status.count("cannot reasonably finish confirmation") == 7
    assert "chosen" in status


def test_review10_missing_evidence_is_told_apart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    start = (20, 6, 512)
    driver._start_setup = start
    batch = (20, 6, 1024)  # needs the long question too
    from inferenceos.profiling.workloads import get_workload

    import dataclasses

    driver._long = dataclasses.replace(get_workload("short-short"), workload_id="long-q")
    driver._meas = {(s, "short-short"): None for s in [start, batch]}  # type: ignore[misc]
    driver._meas[(start, "long-q")] = None  # type: ignore[assignment]
    monkeypatch.setattr(driver, "_vs_start", lambda setup, q: 0.10 if q == "short" and setup == batch else None)
    monkeypatch.setattr(driver, "_final_cost", lambda setup, planning=False, assumed=None: 100.0)
    monkeypatch.setattr(driver, "_left", lambda: 900.0)
    monkeypatch.setattr(driver, "_cost", lambda q: 50.0)
    rows = driver._measured_status(None)
    assert [r["status"] for r in rows] == ["missing evidence"]
    assert driver._to_complete() == [batch]


# --------------------------------------------------------------------------- #
# Reviewer question 3: a stop while the model file is being hashed (before the store
# emits "preparing") abandons the hash and publishes nothing.
def test_review10_stop_during_model_hashing_publishes_nothing(tmp_path: Path) -> None:
    import inferenceos.tune.profiles as profiles

    _, _, before = _first_saved(tmp_path)
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    # A model file of several 4 MiB chunks, so the hash checks for a stop in between.
    tuner.model_path.write_bytes(b"\x01" * (3 * 4 * 1024 * 1024 + 7))
    chunks: list[int] = []
    real_sha = profiles.file_sha256

    def counting_sha(path: Path, *, check: Any = None) -> str:
        def chunk_check() -> None:
            chunks.append(1)
            if len(chunks) == 2:
                driver.request_cancel("stopped while hashing")
            if check is not None:
                check()
        return real_sha(path, check=chunk_check)

    seen: list[str] = []
    driver.save_events = seen.append
    profiles.file_sha256 = counting_sha
    try:
        result = driver.run()
    finally:
        profiles.file_sha256 = real_sha
    assert seen == ["hashing"], seen  # never reached "preparing"
    assert len(chunks) == 2  # the hash stopped at the next chunk boundary
    assert result.saved_setup is None and result.outcome.startswith("cancelled by the user")
    assert store.path.read_bytes() == before


# --------------------------------------------------------------------------- #
# Reviewer question 4: two processes writing saved setups at once lose nothing.
_WRITER = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[4])
from inferenceos.config import Settings
from inferenceos.contracts.tune import SavedSetup
from inferenceos.tune.profiles import SetupStore
template = SavedSetup.model_validate(json.loads(Path(sys.argv[1]).read_text()))
store = SetupStore(Settings(), path=Path(sys.argv[2]))
for i in range(int(sys.argv[5])):
    model = template.model.model_copy(update={"path": f"{sys.argv[3]}-{i}.gguf"})
    store.save(template.model_copy(update={"model": model, "setup_id": f"{sys.argv[3]}-{i}"}))
'''


def test_review10_concurrent_writers_lose_no_update(tmp_path: Path) -> None:
    import subprocess
    import sys

    _, store, _ = _first_saved(tmp_path)
    template = tmp_path / "template.json"
    template.write_text(store.all()[0].model_dump_json(), encoding="utf-8")
    script = tmp_path / "writer.py"
    script.write_text(_WRITER, encoding="utf-8")
    src = str(Path(__file__).resolve().parents[1] / "src")
    procs = [subprocess.Popen([sys.executable, str(script), str(template), str(store.path), name, src, "25"])
             for name in ("writer-a", "writer-b", "writer-c")]
    for proc in procs:
        assert proc.wait(120) == 0
    ids = {setup.setup_id for setup in store.all()}
    expected = {f"writer-{w}-{i}" for w in "abc" for i in range(25)}
    assert expected <= ids, f"lost {len(expected - ids)} updates"


# --------------------------------------------------------------------------- #
# Reviewer question 5: pacing fed by the REAL compare_fairly progress callback.
def test_review10_pacing_from_real_compare_fairly(tmp_path: Path, clock: Clock) -> None:
    """compare_fairly (real code) with: AB/BA order, one disturbed repeat (block 1,
    candidate side), one discarded block (block 2: the current setup's runs fail), and
    a fixed look at 4 counted blocks reached on attempt 5. Its own progress events go
    to BudgetedTune.final_progress."""
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.optimization.keep import ModeRules
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.selftest import sample_model
    from inferenceos.tune.fair import compare_fairly
    from inferenceos.tune.measure import DisturbanceWatch
    from tests.experiment_helpers import build_test_policy

    settings = _settings(tmp_path, optimization_mode="performance_first", tune_fair_extra_attempts=2)
    rules = ModeRules.from_settings(settings)
    start = RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=20)
    candidate = RuntimeConfiguration(context_length=CONTEXT, gpu_offload_layers=25)
    measurements: list[str] = []

    class Harness(ScriptedHarness):
        def run_once(self, task: Any, **kwargs: Any) -> Any:
            if kwargs.get("warmup"):
                side = "start" if kwargs["options"].gpu_offload_layers == 20 else "candidate"
                measurements.append(side)
                clock.seconds += 10.0  # the fresh load
            clock.seconds += 3.0  # one answer
            return super().run_once(task, **kwargs)

    def responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        number = len(measurements)
        side = measurements[-1]
        if number == 2 and side == "candidate":
            return SampleResult(decode_tps=24.0, total_latency_s=1.7, cpu_percent=0.0)  # disturbed
        if number == 5 and side == "start":
            return SampleResult(fail=True)  # block 2's current setup fails: block discarded
        jitter = 1.0 + 0.01 * ((number * 7) % 3 - 1)
        if side == "candidate":
            return SampleResult(decode_tps=24.0 * jitter, total_latency_s=1.7 / jitter)
        return SampleResult(decode_tps=20.0 * jitter, total_latency_s=2.0 / jitter)

    _, _, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    driver._t0, driver._target, driver._hard, driver._save_reserve = 0.0, 3000.0, 3600.0, 30.0
    driver._final = {"planned_blocks": {"short": 4}, "cost": {"short": 25.0},
                     "ids": {"short-short": "short"}, "q": {}, "measurements": [], "current": None}
    events: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    original = driver._project_final

    def spy() -> dict[str, Any]:
        out = original()
        projections.append(out)
        return out

    driver._project_final = spy  # type: ignore[method-assign]

    def progress(event: dict[str, Any]) -> None:
        events.append(dict(event))
        driver.final_progress(event)

    policy = build_test_policy(warmup_repetitions=1, measured_repetitions=3, max_repetitions=3,
                               min_valid_samples=3, max_total_runs=4, noise_tolerance=0.0)
    watch = DisturbanceWatch(logical_cores=12, limit_percent=25.0, max_repeats=2)
    fair = compare_fairly(
        harness=Harness(responder), model=sample_model(), workload=get_workload("short-short"),
        start=start, candidate=candidate, policy=policy, settings=settings, rules=rules, watch=watch,
        card_id="gpu_layers_more", phase="final", mode="better", min_blocks=4, max_blocks=4, progress=progress,
    )
    sequence = [(e["event"], e.get("side"), e.get("counted"), e["attempts"], e["usable"], e["discarded"], e.get("retries"))
                for e in events]
    print("\nEVENTS (event, side, counted, attempts, usable, discarded, retries):")
    for row in sequence:
        print("  ", row)
    print("MEASUREMENT ORDER:", measurements)
    state = driver._final["q"]["short"]
    print("FINAL STATE:", {k: v for k, v in state.items() if k != "costs"},
          "costs:", {k: [round(x, 1) for x in v] for k, v in state["costs"].items()})
    print("PROJECTIONS (at each boundary):", [(p["at_s"], p["usable_blocks"], p["remaining_s"],
                                              p["projected_finish_s"]) for p in projections])
    # Order alternates per attempt: AB, BA, AB, BA, AB (the repeat is inside block 1's B).
    assert measurements == ["start", "candidate", "candidate", "candidate", "start", "start", "candidate",
                            "candidate", "start", "start", "candidate"]
    assert fair.decision == "better" and fair.attempts == 5 and len(fair.discarded_blocks) == 1
    assert [e["event"] for e in events].count("boundary") == 5  # none after the look decided
    assert state["attempts"] == 5 and state["usable"] == 4 and state["discarded"] == 1 and state["retries"] == 1
    # Before block 3 (one usable, one discarded): 3 blocks still required, at the
    # observed per-attempt costs (22 s each side: load 10 + warm-up + 3 answers).
    third = projections[2]
    assert third["usable_blocks"] == 1 and third["remaining_s"] == 3 * (22.0 + 22.0)


# --------------------------------------------------------------------------- #
# Reviewer follow-up, item 3: the store lock wait is cancellable, and only "lock held"
# errors are retried. The holder is a separate process using the same lock code
# (fcntl on POSIX, msvcrt on Windows).
_HOLDER = r'''
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from inferenceos.tune.profiles import _StoreLock
with _StoreLock(Path(sys.argv[1])):
    print("locked", flush=True)
    time.sleep(float(sys.argv[2]))
'''


def _hold_lock(tmp_path: Path, store: Any, seconds: float) -> Any:
    import subprocess
    import sys

    script = tmp_path / "holder.py"
    script.write_text(_HOLDER, encoding="utf-8")
    lock = store.path.with_name(store.path.name + ".lock")
    src = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.Popen([sys.executable, str(script), str(lock), str(seconds), src],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None and proc.stdout.readline().strip() == "locked"
    return proc


def test_review10_user_cancel_aborts_a_store_lock_wait(tmp_path: Path) -> None:
    import threading

    from inferenceos.tune.budgeted import Cancelled
    from inferenceos.tune.profiles import CommitGuard

    _, store, before = _first_saved(tmp_path)
    _, _, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    new = store.all()[0].model_copy(update={"setup_id": "setup-new", "context_length": 2048})
    holder = _hold_lock(tmp_path, store, 30.0)
    try:
        threading.Timer(0.3, driver.request_cancel, args=("stop while waiting for the lock",)).start()
        began = time.monotonic()
        with pytest.raises(Cancelled):
            store.save(new, guard=CommitGuard(check=driver.check_user, lock=driver._commit_lock))
        took = time.monotonic() - began
    finally:
        holder.kill()
        holder.wait(10)
    print(f"\ncancelled lock wait returned after {took:.2f} s")
    assert took < 2.0
    assert store.path.read_bytes() == before
    assert not list(store.path.parent.glob("*.tmp"))


def test_review10_second_writer_gets_the_lock_after_release(tmp_path: Path) -> None:
    _, store, _ = _first_saved(tmp_path)
    new = store.all()[0].model_copy(update={"setup_id": "setup-after-release", "context_length": 2048})
    holder = _hold_lock(tmp_path, store, 1.0)
    try:
        began = time.monotonic()
        store.save(new)  # no guard: waits for the holder, then writes
        took = time.monotonic() - began
    finally:
        holder.wait(10)
    print(f"\nsecond writer waited {took:.2f} s")
    assert 0.5 < took < 10.0
    assert "setup-after-release" in {s.setup_id for s in store.all()}


def test_review10_non_contention_lock_error_is_raised_at_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import errno

    import inferenceos.tune.profiles as profiles
    from inferenceos.tune.profiles import StoreBusy

    _, store, before = _first_saved(tmp_path)
    new = store.all()[0].model_copy(update={"setup_id": "setup-x", "context_length": 2048})

    def bad_handle(handle: Any) -> None:
        raise OSError(errno.EBADF, "Bad file descriptor")

    monkeypatch.setattr(profiles._StoreLock, "_try_lock", staticmethod(bad_handle))
    began = time.monotonic()
    with pytest.raises(OSError) as info:
        store.save(new)
    took = time.monotonic() - began
    assert not isinstance(info.value, StoreBusy) and info.value.errno == errno.EBADF
    assert took < 1.0
    assert store.path.read_bytes() == before


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="checks the real msvcrt.locking branch")
def test_review10_windows_lock_contention_errno_is_retried(tmp_path: Path) -> None:
    """On Windows, a second handle trying the held lock must fail with an errno that
    the store treats as contention (retried), not as a real error. Establishes the
    errno set on the actual platform (reviewer follow-up, item 4)."""
    from inferenceos.tune.profiles import _CONTENTION_ERRNOS, _lock_contended, _StoreLock

    lock = tmp_path / "x.lock"
    with _StoreLock(lock):
        with open(lock, "a+b") as other:
            with pytest.raises(OSError) as info:
                _StoreLock._try_lock(other)
    print(f"\nwindows contention error: errno={info.value.errno} ({info.value}); retried set={sorted(_CONTENTION_ERRNOS)}")
    assert _lock_contended(info.value)


# --------------------------------------------------------------------------- #
# Reviewer follow-up, item 5: every final candidate records what sized its check.
def test_review10_final_check_records_its_calibration_data(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    result = driver.run()
    assert result.outcome == "saved"
    final = _round(result, "final check")
    cal = final["calibration"]
    print("\ncalibration record:", cal)
    short = cal["short"]
    assert short["mode"] == "better" and short["workload"] == "short-short"
    assert isinstance(short["direct_whole_setup_reading"], float) and short["direct_whole_setup_reading"] > 0
    assert short["spread_used"] > 0 and short["spread_source"]
    assert short["chosen_blocks"] == final["blocks_per_question"]["short"]
    assert len(short["final_block_scores"]) == short["chosen_blocks"]
    assert short["final_decision"] == "better"
    assert short["constants"] == {"winner_shrink": 0.7, "spread_margin": 1.25, "z_power": 2.487,
                                  "status": "experimental / provisional"}
    assert short["status"] == "completed" and short["termination"] == "comparison completed"


# --------------------------------------------------------------------------- #
# Reviewer follow-up 3, item 4: an incomplete final comparison keeps its evidence in the
# calibration record (block scores, attempts, discarded blocks, why it ended), and stays
# ineligible for acceptance.


def _slower_final(tmp_path: Path, clock: Clock) -> tuple[Any, Any, BudgetedTune]:
    """Answers take 30 s instead of 10 once the final check starts: pacing abandons it
    after one counted block (review 9's case)."""
    state = {"final": False}

    def slower(task: ProfileTask, options: ProfilerOptions) -> float:
        return 20.0 if state["final"] else 0.0

    tuner, store, driver, _ = _practical(tmp_path, _world(clock, hang=slower), 760.0, clock=clock)
    original = tuner._check_whole_setup

    def final(*args: Any, **kwargs: Any) -> Any:
        state["final"] = True
        return original(*args, **kwargs)

    tuner._check_whole_setup = final
    return tuner, store, driver


def test_review11_abandoned_check_keeps_its_partial_evidence(tmp_path: Path, clock: Clock) -> None:
    tuner, store, driver = _slower_final(tmp_path, clock)
    result = driver.run()
    check = _round(result, "final check")
    short = check["calibration"]["short"]
    print("\nprogress:", check["progress"]["short"])
    print("calibration:", {k: short[k] for k in ("status", "final_block_scores", "final_decision", "final_attempts",
                                                 "final_discarded_blocks", "termination", "eligible_for_acceptance")})
    assert check["abandoned"] and result.saved_setup is None and _saved(tuner, store) is None  # unchanged
    assert check["progress"]["short"]["usable"] == 1 and check["progress"]["short"]["attempts"] == 1
    assert short["status"] == "incomplete, usable blocks"
    assert short["final_block_scores"] == [0.175]  # the one counted block (before: None)
    assert short["final_attempts"] == 1 and short["final_discarded_blocks"] == 0
    assert short["final_decision"] is None and short["eligible_for_acceptance"] is False
    assert short["termination"].startswith("abandoned by pacing: at ")
    assert short["chosen_blocks"] == 4 and short["direct_whole_setup_reading"] == 0.175  # sizing kept


def test_review11_user_stop_during_the_final_check_keeps_its_partial_evidence(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    original = driver.final_progress

    def progress(event: dict[str, Any]) -> None:
        original(event)
        if event.get("event") == "block" and event.get("counted") and event.get("usable") == 1:
            driver.request_cancel("stopped during the final check")

    driver.final_progress = progress  # type: ignore[method-assign]
    result = driver.run()
    assert result.outcome.startswith("cancelled by the user") and _saved(tuner, store) is None
    check = _round(result, "final check")
    short = check["calibration"]["short"]
    print("\ninterrupted:", check["interrupted"])
    print("calibration:", {k: short[k] for k in ("status", "final_block_scores", "final_attempts", "termination")})
    assert check["interrupted"] == "cancelled by the user: stopped during the final check"
    assert short["status"] == "incomplete, usable blocks" and len(short["final_block_scores"]) == 1
    assert short["final_attempts"] == 1 and short["eligible_for_acceptance"] is False
    assert short["termination"] == "cancelled by the user: stopped during the final check"


def test_review11_calibration_tells_the_four_states_apart() -> None:
    from types import SimpleNamespace

    sizing = {q: {"workload": f"{q}-q", "mode": "better", "chosen_blocks": 4} for q in ("short", "long", "third")}
    done = SimpleNamespace(block_scores=[0.1, 0.2, 0.15, 0.12], decision="better", attempts=4,
                           discarded_blocks=[], stopped_early="")
    progress = {"q": {
        "short": {"usable": 4, "attempts": 4, "discarded": 0, "block_scores": [0.1, 0.2, 0.15, 0.12], "discarded_why": []},
        "long": {"usable": 0, "attempts": 1, "discarded": 1, "block_scores": [], "discarded_why": ["not enough good runs"]},
    }, "measurements": [{"question": "long"}, {"question": "long"}]}
    cal = BudgetedTune._calibration_record(sizing, [("short-q", done)], progress,
                                           termination="cancelled by the user: stop")
    states = {q: cal[q]["status"] for q in cal}
    print("\nstates:", states)
    assert states == {"short": "completed", "long": "started, no usable block", "third": "not started"}
    assert cal["short"]["final_decision"] == "better" and cal["short"]["final_block_scores"] == [0.1, 0.2, 0.15, 0.12]
    assert cal["long"]["final_block_scores"] == [] and cal["long"]["final_discarded_blocks"] == 1
    assert cal["long"]["discarded_why"] == ["not enough good runs"] and cal["long"]["side_measurements"] == 2
    assert cal["long"]["termination"] == cal["third"]["termination"] == "cancelled by the user: stop"
    assert cal["long"]["eligible_for_acceptance"] is cal["third"]["eligible_for_acceptance"] is False


# --------------------------------------------------------------------------- #
# Reviewer follow-up 4, item 3: an unexpected error that ends the final check must still
# reach the exported report (through `tune`'s own report writing), with the partial
# evidence, the error's identity, cleanup, a failure status -- and nothing saved.


def test_review12_unexpected_final_check_error_reaches_the_exported_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import inferenceos.backends.llama_cpp_resident_adapter as lra
    import inferenceos.profiling.runner as runner
    import inferenceos.runtime.gpu_search as gpu_search
    import inferenceos.runtime.resident_workload as rw
    import inferenceos.tune.budgeted as budgeted
    import inferenceos.tune.commands as commands

    tuner, store, _, _ = _practical(tmp_path, _world(None), 3600.0)
    engine_calls: list[str] = []

    class Resident:
        def __init__(self, **_: Any) -> None:
            pass

        def stop(self) -> None:
            engine_calls.append("stop")

        def cancel(self) -> None:
            engine_calls.append("cancel")

        def target_identity(self) -> None:
            return None

    # `tune` itself (commands.run_tune) with the scripted tuner in place of the real one.
    monkeypatch.setattr(lra, "LlamaCppResidentAdapter", lambda settings: object())
    monkeypatch.setattr(rw, "ResidentActiveWorkload", Resident)
    monkeypatch.setattr(gpu_search, "ResidentMeasurementAdapter", lambda resident, explicit_defaults=True: object())
    monkeypatch.setattr(runner, "BaselineProfiler", lambda **_: object())
    monkeypatch.setattr(commands, "detect_hardware_profile", lambda settings: tuner.hardware)
    monkeypatch.setattr(commands, "check_machine", lambda settings, hardware: tuner.machine)
    monkeypatch.setattr(commands, "load_last_check", lambda settings: None)
    monkeypatch.setattr(commands, "save_check", lambda settings, machine: None)
    monkeypatch.setattr(commands, "load_model_profile", lambda path: tuner.model)
    monkeypatch.setattr(commands, "read_live_state", lambda: (0, None))
    monkeypatch.setattr(commands, "make_try_load", lambda **_: tuner.try_load)
    monkeypatch.setattr(commands, "make_piece_lines_reader", lambda **_: None)
    monkeypatch.setattr(commands, "Tuner", lambda **_: tuner)

    original = budgeted.BudgetedTune.final_progress

    def failing(self: BudgetedTune, event: dict[str, Any]) -> None:
        original(self, event)
        if event.get("event") == "boundary" and event.get("usable") == 1:
            raise RuntimeError("injected failure after one completed block")

    monkeypatch.setattr(budgeted.BudgetedTune, "final_progress", failing)
    out = tmp_path / "reports" / "tune_report.json"
    code = commands.run_tune(tuner.settings, model_path=tuner.model_path, context_length=CONTEXT,
                             approve_all=True, out=out)
    report = json.loads(out.read_text(encoding="utf-8"))  # the exported artifact
    final = next(r for r in report["rounds"] if r.get("round") == "final check")
    short = final["calibration"]["short"]
    print("\nexit code:", code, "| outcome:", report["outcome"])
    print("error:", {k: v for k, v in final["error"].items() if k != "traceback"})
    print("calibration:", {k: short[k] for k in ("status", "final_block_scores", "chosen_blocks",
                                                 "direct_whole_setup_reading", "termination",
                                                 "eligible_for_acceptance")})
    print("saved_setup:", report["saved_setup"], "| engine calls:", engine_calls)
    assert code == 1 and report["outcome"].startswith("failed: the final check stopped on an unexpected error")
    assert final["error"]["type"] == "RuntimeError"
    assert final["error"]["message"] == "injected failure after one completed block"
    assert "RuntimeError: injected failure" in final["error"]["traceback"]
    assert final["interrupted"] == "stopped by an error: RuntimeError: injected failure after one completed block"
    assert short["status"] == "incomplete, usable blocks" and short["final_block_scores"] == [0.175]
    assert short["chosen_blocks"] == 4 and short["direct_whole_setup_reading"] == 0.175  # sizing kept
    assert short["termination"] == final["interrupted"] and short["eligible_for_acceptance"] is False
    assert report["saved_setup"] is None and _saved(tuner, store) is None  # nothing approved or saved
    assert report["decisions"] == []  # nothing was put to approval
    assert "stop" in engine_calls  # cleanup ran (the driver's timed stop and tune's own)


def test_review12_user_stop_in_the_final_check_still_reads_as_cancelled(tmp_path: Path) -> None:
    """The error path does not swallow a user stop: it stays a cancellation."""
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    original = driver.final_progress

    def progress(event: dict[str, Any]) -> None:
        original(event)
        if event.get("event") == "block" and event.get("counted"):
            driver.request_cancel("stop")

    driver.final_progress = progress  # type: ignore[method-assign]
    result = driver.run()
    final = _round(result, "final check")
    assert result.outcome == "cancelled by the user: nothing new saved"
    assert final["interrupted"] == "cancelled by the user: stop" and final.get("error") is None
    assert _saved(tuner, store) is None


# ----------------------------------------------- reviewer, monitor round 2, item 4
class BlockingHarness(ScriptedHarness):
    """Behaves as the real profiler once a sampler's cleanup is unresolved: every later
    run_once raises MonitorCleanupUnresolved before starting anything."""

    def __init__(self, responder: Any, block_after: int | None) -> None:
        super().__init__(responder)
        self.block_after = block_after
        self.monitor_blocked: str | None = None
        self.refused = 0

    def run_once(self, task: Any, **kwargs: Any) -> Any:
        from inferenceos.telemetry.counters import MonitorCleanupUnresolved

        if self.monitor_blocked is not None:
            self.refused += 1
            raise MonitorCleanupUnresolved(f"no further measurement: {self.monitor_blocked}")
        run = super().run_once(task, **kwargs)
        if self.block_after is not None and len(self.calls) >= self.block_after:
            self.monitor_blocked = "a counter sampler's cleanup was not confirmed (scripted)"
        return run


def test_monitor_round2_unresolved_cleanup_mid_search_fails_and_saves_nothing(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    harness = BlockingHarness(_world(None), block_after=12)
    tuner.harness = harness
    result = driver.run()
    print("\noutcome:", result.outcome, "| answers:", len(harness.calls), "| refused:", harness.refused)
    assert result.outcome.startswith("failed: no further measurement: a counter sampler's cleanup")
    assert result.outcome.endswith("nothing saved")
    assert result.saved_setup is None and _saved(tuner, store) is None
    assert len(harness.calls) == 12 and harness.refused == 0  # stopped at close, before any extra attempt


def test_monitor_round2_unresolved_cleanup_before_saving_publishes_nothing(tmp_path: Path) -> None:
    tuner, store, driver, _ = _practical(tmp_path, _world(None), 3600.0)
    harness = BlockingHarness(_world(None), block_after=None)
    tuner.harness = harness

    class BlockWhileAsking:
        def decide(self, change: Any, timeout_s: float = 0.0) -> ChangeDecision:
            # The last measurement's close was unresolved: the approval comes after it.
            harness.monitor_blocked = "a counter sampler's cleanup was not confirmed (scripted)"
            return ChangeDecision(change_id=change.change_id, approved=True, answer="yes")

    tuner.approver = BlockWhileAsking()
    result = driver.run()
    print("\noutcome:", result.outcome)
    assert result.outcome.startswith("failed: not saved: a counter sampler's cleanup was not confirmed")
    assert result.saved_setup is None and _saved(tuner, store) is None


def test_monitor_round2_tune_report_for_a_blocked_normal_tune(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`tune` (commands.run_tune) on the normal search: the block propagates out of the
    tuner (no generic handler swallows it), and tune still writes a failure report and
    exits 1, with nothing saved."""
    import json

    import inferenceos.backends.llama_cpp_resident_adapter as lra
    import inferenceos.profiling.runner as runner
    import inferenceos.runtime.gpu_search as gpu_search
    import inferenceos.runtime.resident_workload as rw
    import inferenceos.tune.commands as commands

    s = _settings(tmp_path, tune_heat_check_runs=0)
    tuner, _, _, store = _tuner(tmp_path, _world(None), answers=[True] * 8, settings=s)
    harness = BlockingHarness(_world(None), block_after=6)
    tuner.harness = harness
    engine_calls: list[str] = []

    class Resident:
        def __init__(self, **_: Any) -> None:
            pass

        def stop(self) -> None:
            engine_calls.append("stop")

        def cancel(self) -> None:
            engine_calls.append("cancel")

        def target_identity(self) -> None:
            return None

    monkeypatch.setattr(lra, "LlamaCppResidentAdapter", lambda settings: object())
    monkeypatch.setattr(rw, "ResidentActiveWorkload", Resident)
    monkeypatch.setattr(gpu_search, "ResidentMeasurementAdapter", lambda resident, explicit_defaults=True: object())
    monkeypatch.setattr(runner, "BaselineProfiler", lambda **_: harness)
    monkeypatch.setattr(commands, "detect_hardware_profile", lambda settings: tuner.hardware)
    monkeypatch.setattr(commands, "check_machine", lambda settings, hardware: tuner.machine)
    monkeypatch.setattr(commands, "load_last_check", lambda settings: None)
    monkeypatch.setattr(commands, "save_check", lambda settings, machine: None)
    monkeypatch.setattr(commands, "load_model_profile", lambda path: tuner.model)
    monkeypatch.setattr(commands, "read_live_state", lambda: (0, None))
    monkeypatch.setattr(commands, "make_try_load", lambda **_: tuner.try_load)
    monkeypatch.setattr(commands, "make_piece_lines_reader", lambda **_: None)
    monkeypatch.setattr(commands, "Tuner", lambda **_: tuner)
    out = tmp_path / "reports" / "tune_report.json"
    code = commands.run_tune(tuner.settings, model_path=tuner.model_path, context_length=CONTEXT,
                             approve_all=True, out=out)
    report = json.loads(out.read_text(encoding="utf-8"))
    print("\nexit code:", code, "| report:", {k: report[k] for k in ("outcome", "saved_setup", "error")})
    assert code == 1 and report["outcome"].startswith("failed: no further measurement")
    assert report["error"]["type"] == "MonitorCleanupUnresolved" and report["saved_setup"] is None
    assert _saved(tuner, store) is None and harness.refused == 0 and "stop" in engine_calls
