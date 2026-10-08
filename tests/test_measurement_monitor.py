"""The experimental measurement-scoped monitor inside the real profiler.

Drives the real ``BaselineProfiler.run_once`` / ``finalize_runs`` and ``measure_steady``
with a scripted engine and a deterministic fake raw-counter sampler: readings every
20 ms on the real wall clock, a cumulative page count, and paging injected at chosen
offsets inside an answer. The platform layer is stubbed so these run off Windows.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

import inferenceos.profiling.runner as runner
from inferenceos.config import Settings
from inferenceos.contracts.enums import MeasurementMode, MetricQuality
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendCapabilities,
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
)
from inferenceos.native.base import GlobalMemory
from inferenceos.profiling.runner import BaselineProfiler, ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_hardware, sample_model
from inferenceos.telemetry.attribution import CPU_PERFORMANCE_SPEC, PAGING_SPEC, RawReading
from inferenceos.telemetry.counters import MeasurementMonitor

STEP = 0.02


class _Platform:
    def global_memory_status(self) -> GlobalMemory:
        return GlobalMemory(8 << 30, 4 << 30, 8 << 30, 4 << 30, 8 << 30, 4 << 30, 50)

    def power_status(self) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return lambda *a, **k: None


class World:
    """Paging events on the wall clock, and every fake sampler made."""

    def __init__(self) -> None:
        self.events: list[tuple[float, int]] = []
        self.samplers: list[FakeRawSampler] = []


WORLD = World()


class FakeRawSampler:
    """Reads both counters every STEP seconds from its start (deterministic on the
    wall clock); a reading exists only once its read time has passed."""

    def __init__(self, specs: Any, interval_s: float = 0.1) -> None:
        self.specs = ["|".join(s) for s in specs]
        self.t0: float | None = None
        self.stopped_at: float | None = None
        self.starts = self.stops = 0
        self.errors: list[str] = []
        self.discarded_before: float | None = None
        self.die_at: float | None = None  # stops producing readings after this time
        WORLD.samplers.append(self)

    @property
    def running(self) -> bool:
        return self.t0 is not None and self.stopped_at is None

    def start(self) -> bool:
        self.t0, self.starts = time.time(), self.starts + 1
        return True

    def readings(self) -> list[RawReading]:
        if self.t0 is None:
            return []
        until = min(x for x in (time.time() - 0.003, self.stopped_at, self.die_at) if x is not None)
        out: list[RawReading] = []
        k = 0
        while self.t0 + k * STEP <= until:
            t = self.t0 + k * STEP
            raw = sum(p for at, p in WORLD.events if at < t)
            for spec in self.specs:
                out.append(RawReading(spec=spec, read_before=t, read_after=t + 0.001,
                                      raw=raw if spec == PAGING_SPEC else 0, base=0, ts100=int(t * 1e7),
                                      counter_type=272696320 if spec == PAGING_SPEC else 542573824,
                                      calculated=None if spec == PAGING_SPEC or k == 0 else 95.0,
                                      arrival_wall=t + 0.002, arrival_perf=0.0))
            k += 1
        if self.discarded_before is not None:
            # The same retention as RawCounterSampler.discard_before: drop readings made
            # before the cut-off except the last one of each counter.
            cut = self.discarded_before
            last = {r.spec: r for r in out if r.read_after < cut}
            out = sorted([*last.values(), *[r for r in out if r.read_after >= cut]], key=lambda r: r.read_after)
        return out

    def has_reading(self, spec: str) -> bool:
        return self.t0 is not None and time.time() > self.t0

    def discard_before(self, wall: float) -> int:
        self.discarded_before = wall
        return 0

    def stop(self) -> dict[str, object]:
        self.stopped_at, self.stops = time.time(), self.stops + 1
        return {"exit_confirmed": True, "reader_joined": True, "pipe_closed": True, "resolved": True, "errors": []}


class Engine(BackendAdapter):
    def __init__(self) -> None:
        self.loaded = False
        self.loads = 0
        self.fresh = False
        self.answers = 0
        self.answer_s = 0.3
        self.pages: dict[int, list[tuple[float, int]]] = {}  # answer number -> (offset, pages)
        self.fail_on: int | None = None
        self.last_load: object = None

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="fake", version="1")

    def request_fresh_start(self) -> None:
        self.fresh = True

    def load(self, model: Any, params: Any) -> None:
        if not self.loaded or self.fresh:
            self.loads += 1
            self.last_load = object()
            self.loaded, self.fresh = True, False

    def unload(self) -> None:
        pass

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.answers += 1
        if self.fail_on == self.answers:
            raise RuntimeError("engine stopped")
        began = time.time()
        for offset, pages in self.pages.get(self.answers, []):
            WORLD.events.append((began + offset, pages))
        time.sleep(self.answer_s)
        return GenerationResult(text="ok", ttft_s=0.01, decode_tokens_per_sec=20.0, generated_tokens=4)

    def set_runtime_params(self, params: Any) -> None:
        pass

    def telemetry(self) -> BackendTelemetry:
        return BackendTelemetry()

    @property
    def is_loaded(self) -> bool:
        return self.loaded


@pytest.fixture(autouse=True)
def _platform(monkeypatch: pytest.MonkeyPatch) -> None:
    global WORLD
    WORLD = World()
    import inferenceos.telemetry.system as system

    monkeypatch.setattr(runner, "get_platform", lambda: _Platform())
    monkeypatch.setattr(system, "get_platform", lambda: _Platform())


def _profiler(engine: Engine, *, scoped: bool = True, finalize_wait_s: float = 1.0) -> BaselineProfiler:
    settings = Settings(tune_experimental_measurement_monitor=scoped)
    profiler = BaselineProfiler(settings=settings, adapter=engine, hardware=sample_hardware())

    def factory(paths: Any, interval_s: float = 0.1) -> MeasurementMonitor:
        return MeasurementMonitor(paths, interval_s, sampler_factory=FakeRawSampler,  # type: ignore[arg-type]
                                  finalize_wait_s=finalize_wait_s)

    profiler.monitor_factory = factory  # type: ignore[assignment]
    return profiler


def _run(profiler: BaselineProfiler, *, warmup: bool) -> Any:
    task = ProfileTask(model=sample_model(), workload=get_workload("short-short"), context_length=2048)
    return profiler.run_once(task, mode=MeasurementMode.WARM, sequence=1, repetition=0, warmup=warmup,
                             options=ProfilerOptions(telemetry_sampling=True))


def _faults(run: Any) -> Any:
    return run.observation.metrics.hard_page_faults_per_sec


def _measure(profiler: BaselineProfiler, engine: Engine, scored: int = 3) -> tuple[Any, list[Any]]:
    profiler.request_fresh_start()
    warm = _run(profiler, warmup=True)
    runs = [_run(profiler, warmup=False) for _ in range(scored)]
    return warm, profiler.finalize_runs(runs)


def test_default_stays_per_answer_and_finalize_changes_nothing() -> None:
    engine = Engine()
    profiler = BaselineProfiler(adapter=engine, hardware=sample_hardware())
    assert profiler.measurement_monitor is False  # opt-in only
    task = ProfileTask(model=sample_model(), workload=get_workload("short-short"), context_length=2048)
    run = profiler.run_once(task, mode=MeasurementMode.WARM, sequence=1, repetition=0, warmup=False,
                            options=ProfilerOptions(telemetry_sampling=False))
    assert profiler.finalize_runs([run])[0] is run  # a per-answer run comes back untouched
    assert profiler.finalize_runs([]) == [] and profiler._scoped == {}  # noqa: SLF001


def test_one_sampler_per_measurement_no_per_answer_wait_and_paging_attributed() -> None:
    engine = Engine()
    engine.pages = {3: [(0.15, 90)]}  # inside the second scored answer (answer 3 overall)
    profiler = _profiler(engine)
    warm, runs = _measure(profiler, engine)
    profiler.end_measurement()
    assert len(WORLD.samplers) == 1 and WORLD.samplers[0].starts == 1 and WORLD.samplers[0].stops == 1
    values = [_faults(r) for r in runs]
    print("\npaging per scored answer:", values)
    assert values[0] == 0.0 and values[2] == 0.0 and values[1] is not None and values[1] > 0
    assert _faults(warm) is None  # warm-ups are never finalized: unknown, not zero
    # No per-answer waiting: each answer only notes its interval.
    assert all(r.phase_timings_s["monitor_collect_s"] < 0.01 for r in runs)
    # One bounded wait for the reading that closes the last answer, reported separately.
    final = [e for e in profiler.monitor_events if e["event"] == "finalize"][0]
    assert final["measurement_paging"] == "measured" and 0 <= float(final["wait_s"]) < 0.2  # type: ignore[arg-type]
    assert runs[-1].phase_timings_s["monitor_finalize_s"] == final["wait_s"]
    assert [e["event"] for e in profiler.monitor_events] == ["open", "answer", "answer", "answer", "answer",
                                                             "finalize", "close"]
    assert runs[1].observation.metrics.cpu_performance_percent == 95.0


def test_one_ambiguous_answer_makes_the_measurement_unknown() -> None:
    engine = Engine()
    engine.pages = {2: [(0.005, 400)]}  # right at the first scored answer's start: an edge interval
    profiler = _profiler(engine)
    _, runs = _measure(profiler, engine)
    profiler.end_measurement()
    print("\npaging:", [_faults(r) for r in runs], "| notes:", profiler.last_finalize_notes)
    assert all(_faults(r) is None for r in runs)  # the median of the others would have hidden it
    assert all(r.metric_quality["hard_page_faults_per_sec"] is MetricQuality.UNAVAILABLE for r in runs)
    assert "ambiguous" in runs[0].notes and "paging unknown for this measurement" in profiler.last_finalize_notes[0]
    final = [e for e in profiler.monitor_events if e["event"] == "finalize"][0]
    first = final["answers"][0]["paging"]  # type: ignore[index]
    assert first["status"] == "unknown" and first["boundary_pages"] == 400  # evidence kept


def test_missing_closing_reading_is_unknown_after_a_bounded_wait() -> None:
    engine = Engine()
    profiler = _profiler(engine, finalize_wait_s=0.2)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)
    runs = [_run(profiler, warmup=False)]
    WORLD.samplers[0].die_at = time.time() - 0.05  # no reading after the answer will ever arrive
    began = time.perf_counter()
    final = profiler.finalize_runs(runs)
    took = time.perf_counter() - began
    profiler.end_measurement()
    assert _faults(final[0]) is None and "no reading after the answer" in (final[0].notes or "")
    assert 0.18 <= took < 0.6  # bounded


def test_clock_moved_during_the_answer_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = Engine()
    profiler = _profiler(engine)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)
    real_time = time.time
    calls = {"n": 0}

    def jumpy() -> float:
        calls["n"] += 1
        return real_time() + (5.0 if calls["n"] == 2 else 0.0)  # the answer's end reads +5 s

    from types import SimpleNamespace

    # Only the profiler's own clock reads jump (the sampler keeps the real clock).
    monkeypatch.setattr(runner, "time", SimpleNamespace(time=jumpy, perf_counter=time.perf_counter,
                                                        sleep=time.sleep))
    run = _run(profiler, warmup=False)
    monkeypatch.setattr(runner, "time", time)
    final = profiler.finalize_runs([run])
    profiler.end_measurement()
    assert _faults(final[0]) is None and "clock moved" in (final[0].notes or "")


def test_a_fresh_load_gets_a_fresh_scope_and_old_answers_stay_unknown() -> None:
    engine = Engine()
    profiler = _profiler(engine)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)
    old = _run(profiler, warmup=False)
    profiler.request_fresh_start()  # a disturbed repeat starts again from a fresh load
    _run(profiler, warmup=True)
    assert len(WORLD.samplers) == 2 and WORLD.samplers[0].stops == 1 and WORLD.samplers[1].running
    assert profiler.finalize_runs([old])[0].observation.metrics.hard_page_faults_per_sec is None
    profiler.end_measurement()
    assert not any(s.running for s in WORLD.samplers)


def test_an_exception_closes_the_sampler() -> None:
    engine = Engine()
    profiler = _profiler(engine)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)
    engine.fail_on = engine.answers + 1
    with pytest.raises(RuntimeError):
        _run(profiler, warmup=False)
    assert not WORLD.samplers[0].running and WORLD.samplers[0].stops == 1


def test_a_cancellation_closes_the_sampler() -> None:
    class Cancelled(BaseException):
        pass

    engine = Engine()
    profiler = _profiler(engine)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)

    def stop(_: GenerationRequest) -> GenerationResult:
        raise Cancelled()

    engine.generate = stop  # type: ignore[method-assign,assignment]
    with pytest.raises(Cancelled):
        _run(profiler, warmup=False)
    assert not WORLD.samplers[0].running


def test_measure_steady_finalizes_and_closes_and_the_guard_sees_it() -> None:
    from inferenceos.tune.fair import TurnReading, guard_change, measurement_value
    from inferenceos.tune.measure import DisturbanceWatch, measure_steady
    from tests.experiment_helpers import build_test_policy

    policy = build_test_policy(warmup_repetitions=1, measured_repetitions=3, max_repetitions=3,
                               min_valid_samples=3, max_total_runs=6, noise_tolerance=0.0)
    watch = DisturbanceWatch(logical_cores=12, limit_percent=1000.0, max_repeats=0)

    def measure(pages: dict[int, list[tuple[float, int]]]) -> Any:
        engine = Engine()
        engine.pages = pages
        return measure_steady(harness=_profiler(engine), model=sample_model(), workload=get_workload("short-short"),
                              config=RuntimeConfiguration(context_length=2048), policy=policy, phase="t", watch=watch)

    quiet = measure({})
    paged = measure({n: [(0.15, 120)] for n in (2, 3, 4)})  # every scored answer pages ~120 in < 1 s
    ambiguous = measure({3: [(0.005, 500)]})
    assert all(not s.running for s in WORLD.samplers)
    values = [measurement_value(m, "hard_page_faults_per_sec") for m in (quiet, paged, ambiguous)]
    print("\nmeasurement paging (quiet, paged, ambiguous):", values)
    assert values[0] == 0.0 and values[1] is not None and values[1] >= 100 and values[2] is None
    assert any("paging unknown for this measurement" in note for note in ambiguous.notes)

    def blocks(start: Any, cand: Any) -> list[TurnReading]:
        return [TurnReading(turn=n, order="s,c", start={"hard_page_faults_per_sec": start},
                            candidate={"hard_page_faults_per_sec": cand}, improvements={}) for n in range(1, 5)]

    assert (guard_change(blocks(values[0], values[1]), "hard_page_faults_per_sec") or 0) <= -1.0  # breach
    assert guard_change(blocks(values[0], values[2]), "hard_page_faults_per_sec") is None  # not measured
    # Scoring and repetitions are unchanged by the monitor.
    plain = measure_steady(harness=_profiler(Engine(), scoped=False), model=sample_model(),
                           workload=get_workload("short-short"), config=RuntimeConfiguration(context_length=2048),
                           policy=policy, phase="t", watch=watch)
    assert (plain.valid_runs, plain.warmup_runs) == (quiet.valid_runs, quiet.warmup_runs)


def test_retained_readings_are_bounded() -> None:
    engine = Engine()
    profiler = _profiler(engine)
    _, runs = _measure(profiler, engine)
    assert WORLD.samplers[0].discarded_before is not None
    assert WORLD.samplers[0].discarded_before >= max(
        r.identity.finished_at.timestamp() for r in runs) - 1.0
    profiler.end_measurement()


# ------------------------------------------------- reviewer, monitor round 2, item 2


def test_a_monitor_restart_never_erases_an_unknown_answer() -> None:
    """The reviewer's case: one answer, monitor restarted, another answer, both
    finalized together. The first answer's evidence is gone with its monitor, so the
    measurement's paging is unknown (before: [None, 0.0] and a median of 0.0)."""
    from inferenceos.profiling.statistics import summarize

    engine = Engine()
    profiler = _profiler(engine)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)
    first = _run(profiler, warmup=False)
    profiler.request_fresh_start()  # the monitor restarts
    _run(profiler, warmup=True)
    second = _run(profiler, warmup=False)
    final = profiler.finalize_runs([first, second])
    profiler.end_measurement()
    values = [_faults(r) for r in final]
    print("\nper-answer paging:", values, "| notes:", profiler.last_finalize_notes)
    assert values == [None, None]
    assert summarize("hard_page_faults_per_sec", values).count == 0  # nothing for a median to use
    assert "restarted before the answer was finalized" in profiler.last_finalize_notes[0]


def test_closure_before_finalization_is_unknown() -> None:
    engine = Engine()
    profiler = _profiler(engine)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)
    runs = [_run(profiler, warmup=False) for _ in range(2)]
    profiler.end_measurement()  # closed before anyone finalized
    final = profiler.finalize_runs(runs)
    assert [_faults(r) for r in final] == [None, None]
    assert "closed or restarted" in (final[0].notes or "")


def test_a_run_this_profiler_has_no_record_of_is_unknown() -> None:
    engine = Engine()
    profiler = _profiler(engine)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)
    run = _run(profiler, warmup=False)
    other = _profiler(Engine())  # e.g. a different profiler instance
    final = other.finalize_runs([run])
    assert _faults(final[0]) is None and "no record of its monitor" in (final[0].notes or "")
    profiler.end_measurement()


def test_finalizing_twice_changes_nothing() -> None:
    engine = Engine()
    profiler = _profiler(engine)
    _, once = _measure(profiler, engine, scored=2)
    twice = profiler.finalize_runs(once)
    profiler.end_measurement()
    assert [_faults(r) for r in twice] == [_faults(r) for r in once] == [0.0, 0.0]


def test_an_unexpected_reload_between_scored_answers_makes_the_measurement_unknown() -> None:
    from inferenceos.tune.fair import TurnReading, guard_change, measurement_value
    from inferenceos.tune.measure import DisturbanceWatch, measure_steady
    from tests.experiment_helpers import build_test_policy

    class Reloading(Engine):
        def generate(self, request: GenerationRequest) -> GenerationResult:
            result = super().generate(request)
            if self.answers == 2:
                self.fresh = True  # the engine reloads before the next scored answer
            return result

    policy = build_test_policy(warmup_repetitions=1, measured_repetitions=3, max_repetitions=3,
                               min_valid_samples=3, max_total_runs=6, noise_tolerance=0.0)
    watch = DisturbanceWatch(logical_cores=12, limit_percent=1000.0, max_repeats=0)
    engine = Reloading()
    m = measure_steady(harness=_profiler(engine), model=sample_model(), workload=get_workload("short-short"),
                       config=RuntimeConfiguration(context_length=2048), policy=policy, phase="t", watch=watch)
    per_run = [r.hard_page_faults_per_sec for r in m.run_metrics]
    value = measurement_value(m, "hard_page_faults_per_sec")
    print("\nreload mid-measurement: per-run paging", per_run, "| measurement value", value)
    assert engine.loads == 2 and len(WORLD.samplers) == 2  # the reload opened a second scope
    assert per_run == [None, None, None] and value is None
    blocks = [TurnReading(turn=n, order="s,c", start={"hard_page_faults_per_sec": 0.0},
                          candidate={"hard_page_faults_per_sec": value}, improvements={}) for n in range(1, 5)]
    assert guard_change(blocks, "hard_page_faults_per_sec") is None  # not measured: cannot pass
    assert any("restarted before the answer was finalized" in note for note in m.notes)


# ----------------------------------------------- reviewer, monitor round 2, item 4
class StuckSampler(FakeRawSampler):
    """A sampler whose cleanup cannot be confirmed (its process did not exit)."""

    def stop(self) -> dict[str, object]:
        super().stop()
        return {"exit_confirmed": False, "reader_joined": False, "pipe_closed": False, "resolved": False,
                "errors": ["kill: TimeoutExpired"]}


def _stuck_profiler(engine: Engine) -> BaselineProfiler:
    profiler = _profiler(engine)

    def factory(paths: Any, interval_s: float = 0.1) -> MeasurementMonitor:
        return MeasurementMonitor(paths, interval_s, sampler_factory=StuckSampler,  # type: ignore[arg-type]
                                  finalize_wait_s=1.0)

    profiler.monitor_factory = factory  # type: ignore[assignment]
    return profiler


def test_unresolved_cleanup_blocks_every_later_start() -> None:
    from inferenceos.telemetry.counters import MonitorCleanupUnresolved

    engine = Engine()
    profiler = _stuck_profiler(engine)
    _measure(profiler, engine, scored=1)
    profiler.end_measurement()  # the close reports resolved False
    loads, answers = engine.loads, engine.answers
    print("\nblocked:", profiler.monitor_blocked)
    assert profiler.monitor_blocked is not None and "kill: TimeoutExpired" in profiler.monitor_blocked
    assert [e["event"] for e in profiler.monitor_events][-2:] == ["close", "blocked"]
    for warmup in (True, False):
        caught_generic = False
        try:
            profiler.request_fresh_start()
            _run(profiler, warmup=warmup)
        except Exception:  # noqa: BLE001 - what generic retry handling would catch
            caught_generic = True
        except MonitorCleanupUnresolved as exc:
            print("refused:", exc)
        assert not caught_generic  # a BaseException: generic handlers cannot swallow and retry it
    assert len(WORLD.samplers) == 1  # no new sampler was started
    assert (engine.loads, engine.answers) == (loads, answers)  # no load, no answer
    assert profiler.monitor_blocked is not None  # still blocked: nothing lifts it


def test_unresolved_cleanup_stops_measure_steady() -> None:
    from inferenceos.telemetry.counters import MonitorCleanupUnresolved
    from inferenceos.tune.measure import DisturbanceWatch, measure_steady
    from tests.experiment_helpers import build_test_policy

    policy = build_test_policy(warmup_repetitions=1, measured_repetitions=3, max_repetitions=3,
                               min_valid_samples=3, max_total_runs=6, noise_tolerance=0.0)
    watch = DisturbanceWatch(logical_cores=12, limit_percent=1000.0, max_repeats=0)
    engine = Engine()
    profiler = _stuck_profiler(engine)
    first = measure_steady(harness=profiler, model=sample_model(), workload=get_workload("short-short"),
                           config=RuntimeConfiguration(context_length=2048), policy=policy, phase="t", watch=watch)
    print("\nfirst measurement runs:", first.valid_runs, "| blocked:", profiler.monitor_blocked)
    assert profiler.monitor_blocked is not None  # its close was unresolved
    answers = engine.answers
    with pytest.raises(MonitorCleanupUnresolved):
        measure_steady(harness=profiler, model=sample_model(), workload=get_workload("short-short"),
                       config=RuntimeConfiguration(context_length=2048), policy=policy, phase="t", watch=watch)
    assert engine.answers == answers and len(WORLD.samplers) == 1


def test_per_answer_sampler_unconfirmed_cleanup_also_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default per-answer sampler: a stop whose reader or process is still running
    after the bounded wait blocks the profiler the same way."""
    from inferenceos.telemetry.counters import MonitorCleanupUnresolved

    made: list[Any] = []

    class StuckPerAnswer:
        def __init__(self, paths: Any, interval_s: float = 0.25) -> None:
            self.cleanup_confirmed = True
            made.append(self)

        def start(self) -> None:
            pass

        def wait_for_first_sample(self, timeout_s: float = 5.0) -> bool:
            return True

        def stop(self) -> list[Any]:
            self.cleanup_confirmed = False
            return []

    monkeypatch.setattr(runner, "CounterSampler", StuckPerAnswer)
    engine = Engine()
    profiler = _profiler(engine, scoped=False)
    _run(profiler, warmup=False)
    assert profiler.monitor_blocked is not None and "per-answer" in profiler.monitor_blocked
    with pytest.raises(MonitorCleanupUnresolved):
        _run(profiler, warmup=False)
    assert len(made) == 1 and engine.answers == 1


def test_counter_sampler_stop_reports_an_unconfirmed_exit() -> None:
    import subprocess

    from inferenceos.telemetry.counters import CounterSampler

    class Proc:
        def __init__(self, exits: bool) -> None:
            self.exits = exits

        def terminate(self) -> None:
            pass

        def poll(self) -> int | None:
            return 0 if self.exits else None

        def wait(self, timeout: float | None = None) -> int:
            raise subprocess.TimeoutExpired("powershell", timeout or 0)

    for exits, expected in ((True, True), (False, False)):
        sampler = CounterSampler([r"\Memory\Pages Input/sec"])
        sampler._proc = Proc(exits)  # type: ignore[assignment]  # noqa: SLF001
        sampler.stop()
        assert sampler.cleanup_confirmed is expected


# ----------------------------------------------- reviewer, monitor round 3, item 3
def test_unresolved_cleanup_found_at_a_warm_up_transition_starts_nothing() -> None:
    """The block is discovered inside run_once: a warm-up (a new scope) closes the old
    scope, whose cleanup is unresolved. Nothing is loaded, no sampler starts, no answer."""
    from inferenceos.telemetry.counters import MonitorCleanupUnresolved

    engine = Engine()
    profiler = _stuck_profiler(engine)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)
    _run(profiler, warmup=False)
    assert profiler.monitor_blocked is None  # nothing discovered yet
    loads, answers = engine.loads, engine.answers
    engine.fresh = True  # the warm-up would load afresh
    with pytest.raises(MonitorCleanupUnresolved):
        _run(profiler, warmup=True)  # no request_fresh_start: the transition happens inside run_once
    print("\nwarm-up transition: samplers", len(WORLD.samplers), "| loads", engine.loads - loads,
          "| answers", engine.answers - answers, "| blocked:", profiler.monitor_blocked)
    assert profiler.monitor_blocked is not None
    assert len(WORLD.samplers) == 1 and engine.loads == loads and engine.answers == answers


def test_unresolved_cleanup_found_after_an_unexpected_reload_starts_nothing() -> None:
    """An unexpected reload is only seen after loading (the load token changed); closing
    the old scope then reports an unresolved cleanup: stop there -- no new sampler and
    no answer (the reviewer's reproduction returned an answer with two samplers)."""
    from inferenceos.telemetry.counters import MonitorCleanupUnresolved

    engine = Engine()
    profiler = _stuck_profiler(engine)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)
    _run(profiler, warmup=False)
    answers = engine.answers
    engine.fresh = True  # the engine reloads by itself before the next scored answer
    with pytest.raises(MonitorCleanupUnresolved):
        _run(profiler, warmup=False)
    print("\nunexpected reload: samplers", len(WORLD.samplers), "| answers", engine.answers - answers,
          "| blocked:", profiler.monitor_blocked)
    assert profiler.monitor_blocked is not None
    assert len(WORLD.samplers) == 1 and engine.answers == answers


def test_a_run_without_the_scoped_monitor_closes_an_open_scope_first() -> None:
    """A per-answer run after a scoped one: the open scope is closed and checked before
    loading, so an unresolved cleanup also stops the per-answer path."""
    from inferenceos.telemetry.counters import MonitorCleanupUnresolved

    engine = Engine()
    profiler = _stuck_profiler(engine)
    profiler.request_fresh_start()
    _run(profiler, warmup=True)
    profiler.measurement_monitor = False
    loads = engine.loads
    with pytest.raises(MonitorCleanupUnresolved):
        _run(profiler, warmup=False)
    assert engine.loads == loads and profiler.monitor_blocked is not None


# ----------------------------------------------- reviewer, monitor round 3, item 1
def test_bounded_paging_reaches_the_run_as_bounds_not_a_value() -> None:
    """Two 60-page bursts 1.0 s apart in a 1.6 s answer: the readings bound the 1 s peak
    to 60-120/s. The run carries both bounds; its single value stays empty."""
    engine = Engine()
    engine.answer_s = 1.6
    engine.pages = {2: [(0.2, 60), (1.2, 60)]}
    profiler = _profiler(engine)
    _, runs = _measure(profiler, engine, scored=1)
    profiler.end_measurement()
    m = runs[0].observation.metrics
    print("\nbounded run: value", m.hard_page_faults_per_sec, "| bounds", m.hard_page_faults_per_sec_lower,
          m.hard_page_faults_per_sec_upper, "| notes:", runs[0].notes.split("; ")[-1])
    assert m.hard_page_faults_per_sec is None
    assert (m.hard_page_faults_per_sec_lower, m.hard_page_faults_per_sec_upper) == (pytest.approx(60.0), pytest.approx(120.0))
    assert runs[0].metric_quality["hard_page_faults_per_sec"] is MetricQuality.UNAVAILABLE
    assert "paging bounded" in runs[0].notes
