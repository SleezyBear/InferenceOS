"""Search order and time budget, by simulation (no model, no laptop).

Handoff (6 Oct 2026): validate the planner and the experimental budgeted search cheaply,
against the actual orchestration, before any laptop time. This runs the **real** code
paths -- ``Tuner`` (Unbounded: layer search, paths, light checks, final check) and
``BudgetedTune`` (planner, verification reserve, one final confirmation) -- against
scripted model-like worlds on one simulated clock.

The worlds are shaped after the run histories (``claude/search-order-v2-2026-10-06.md``):
which settings helped, by roughly how much, how noisy each question was, and what one
load and one answer cost. They are **not** the models: the curves between recorded
readings are invented (interpolated), interactions are assumptions, and the costs are
per-measurement figures from the logs. So this shows how the orchestration behaves on
such shapes (order, stopping, error control, time split), never how fast a real tune is.

Reported per world and route:

* found: the true goal gain of the final setup (mean of short and long; known exactly
  here), and how often the ideal combination was reached;
* harmful: a final setup truly worse than 2% on either question;
* t_strong: simulated minutes until the route's *current candidate* first held at least
  90% of the ideal gain (selection, not acceptance);
* t_verified: simulated minutes until a setup with a gain was accepted (end of the
  final check), only for runs that accepted one;
* total: simulated minutes for the whole tune, and overrun past the budget.

    python scripts/simulate_search_order.py [--trials 10] [--budgets 30,45,60,90]
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inferenceos.contracts.experiment import RuntimeConfiguration  # noqa: E402
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask  # noqa: E402
from inferenceos.tune.approval import ScriptedChangeApprover  # noqa: E402
from inferenceos.tune.layer_top import LoadAttempt  # noqa: E402
from inferenceos.tune.model_facts import ModelFacts  # noqa: E402
from inferenceos.tune.pipeline import Tuner  # noqa: E402
from tests.experiment_helpers import SampleResult, ScriptedHarness  # noqa: E402
from tests.test_tune import _settings, _tuner  # noqa: E402

Curve = dict[int, tuple[float, float]]
#: A user stop counts as prompt when the tune has ended within this many seconds of it.
STOP_LATENCY_LIMIT_S = 30.0
_BASE = datetime(2026, 10, 6, tzinfo=timezone.utc)


def _interp(curve: Curve, x: int) -> tuple[float, float]:
    if x in curve:
        return curve[x]
    keys = sorted(curve)
    if x <= keys[0]:
        return curve[keys[0]]
    if x >= keys[-1]:
        return curve[keys[-1]]
    lo = max(k for k in keys if k < x)
    hi = min(k for k in keys if k > x)
    w = (x - lo) / (hi - lo)
    return tuple(curve[lo][i] * (1 - w) + curve[hi][i] * w for i in range(2))  # type: ignore[return-value]


@dataclass
class ModelWorld:
    name: str
    blocks: int
    top: int
    start: tuple[int, int | None, int | None]  # layers, threads, ubatch
    experts: bool
    share: Callable[[int], float]
    #: (short, long) log speed by layer count; interpolated between points.
    layers: Curve
    #: (short, long) log effect of a thread count; may depend on the layer count.
    threads: Callable[[int | None, int], tuple[float, float]] = lambda t, layers: (0.0, 0.0)
    #: (short, long) log effect of a prompt batch size (None = 512).
    batch: Callable[[int | None, int, int | None], tuple[float, float]] = lambda b, layers, t: (0.0, 0.0)
    load_s: float = 11.0
    answer_s: dict[str, float] = field(default_factory=lambda: {"short": 3.5, "long": 20.0})
    overhead_s: float = 1.5
    within: dict[str, float] = field(default_factory=lambda: {"short": 0.03, "long": 0.01})
    between: float = 0.012
    long_tokens: int = 1800
    #: Setups that cannot be measured (missing evidence): predicate on (layers, threads, batch).
    fails: Callable[[int, int | None, int | None], bool] = lambda layers, t, b: False
    #: Largest count that still runs (above ``top`` it runs but leaves too little free
    #: graphics memory, like Phi's inherited 32 layers); None = ``top``.
    runs_up_to: int | None = None
    #: The ideal setup (for "reached the ideal").
    ideal: tuple[int, int | None, int | None] | None = None
    #: The single setting carrying the largest part of the ideal gain (milestones).
    major: tuple[str, int] | None = None

    def effect(self, layers: int, threads: int | None, batch: int | None) -> tuple[float, float]:
        base = _interp(self.layers, layers)
        start = _interp(self.layers, self.start[0])
        th = self.threads(threads, layers)
        th0 = self.threads(self.start[1], self.start[0])
        ba = self.batch(batch, layers, threads)
        ba0 = self.batch(self.start[2], self.start[0], self.start[1])
        return tuple(base[i] - start[i] + th[i] - th0[i] + ba[i] - ba0[i] for i in range(2))  # type: ignore[return-value]

    def value(self, layers: int, threads: int | None, batch: int | None) -> float:
        s, lg = self.effect(layers, threads, batch)
        return 0.5 * ((math.exp(s) - 1) + (math.exp(lg) - 1))


def _threads_window(best: int, gain: float, at_layers: Callable[[int], bool] = lambda layers: True) -> Callable[[int | None, int], tuple[float, float]]:
    def f(t: int | None, layers: int) -> tuple[float, float]:
        t = t or 6
        if not at_layers(layers):
            return (-0.01 * abs(t - 6), -0.005 * abs(t - 6))
        if t < 6:
            return (-0.08 * (6 - t), -0.04 * (6 - t))
        d = abs(t - best)
        value = gain * max(0.0, 1 - 0.45 * d) - (0.04 if t >= 12 else 0.0)
        return (value, value * 0.4)
    return f


def _batch_table(table: dict[int, float], short_cost: dict[int, float] | None = None) -> Callable[[int | None, int, int | None], tuple[float, float]]:
    def f(b: int | None, layers: int, t: int | None) -> tuple[float, float]:
        b = b or 512
        return ((short_cost or {}).get(b, 0.0), table.get(b, 0.0))
    return f


def _earlier_setup_batch(b: int | None, layers: int, t: int | None) -> tuple[float, float]:
    """Review 9's interaction: 10 threads help at batch 512 (x1.25) but lose at batch 1024
    (x0.95); batch 1024 reads the long question x1.4 faster."""
    b, t = b or 512, t or 6
    short = long = 0.0
    if t == 10:
        short = long = math.log(1.25) if b == 512 else math.log(0.95)
    if b == 1024:
        long += math.log(1.4)
    return short, long


WORLDS: list[ModelWorld] = [
    ModelWorld(
        "Llama-like (fits fully; top wins; marginal batch)", blocks=28, top=29, start=(23, None, None),
        experts=False, share=lambda L: 0.08 if L >= 29 else 0.15 + (29 - L) * 0.03,
        layers={0: (-0.9, -0.8), 1: (-0.85, -0.75), 23: (0.0, 0.0), 29: (0.132, 0.19)},
        batch=_batch_table({64: -0.2, 128: -0.1, 256: -0.03, 1024: 0.017, 2048: 0.01}),
        load_s=11.0, answer_s={"short": 2.5, "long": 19.0}, overhead_s=1.4,
        within={"short": 0.03, "long": 0.01}, long_tokens=1850, ideal=(29, None, 1024), major=("gpu_offload_layers", 29),
    ),
    ModelWorld(
        "Granite-MoE-like (CPU wins; threads then batch)", blocks=32, top=33, start=(33, None, None),
        experts=True, share=lambda L: 1.0 if L == 0 else (0.9 if L == 1 else max(0.1, 1 - L / 33)),
        layers={0: (0.72, 0.30), 1: (0.70, 0.29), 8: (0.3, 0.15), 33: (0.0, 0.0)},
        threads=_threads_window(10, 0.04, at_layers=lambda L: L <= 1),
        batch=_batch_table({64: -0.25, 128: -0.12, 256: -0.04, 1024: 0.106, 2048: 0.09}),
        load_s=12.0, answer_s={"short": 2.0, "long": 21.0}, overhead_s=2.2,
        within={"short": 0.035, "long": 0.012}, long_tokens=1700, ideal=(0, 10, 1024), major=("gpu_offload_layers", 0),
    ),
    ModelWorld(
        "Phi-like (inherited start unsafe; batch 2048)", blocks=32, top=30, start=(32, None, None),
        experts=False, share=lambda L: 0.12 if L >= 30 else 0.2 + (30 - L) * 0.025,
        layers={0: (-0.8, -0.7), 1: (-0.78, -0.69), 30: (0.0, 0.0), 32: (0.0, 0.0)},
        batch=_batch_table({64: -0.3, 128: -0.15, 256: -0.05, 1024: 0.08, 2048: 0.137}),
        load_s=16.0, answer_s={"short": 4.5, "long": 19.0}, overhead_s=3.4,
        within={"short": 0.04, "long": 0.009}, long_tokens=1900, runs_up_to=32, ideal=(30, None, 2048), major=("ubatch_size", 2048),
    ),
    ModelWorld(
        "Granite-4H-like (partial; noisy short; threads + batch)", blocks=40, top=13, start=(13, None, None),
        experts=False, share=lambda L: 1.0 if L <= 1 else max(0.55, 1 - L / 41),
        layers={0: (0.04, 0.02), 1: (0.035, 0.02), 3: (0.034, 0.015), 4: (0.035, 0.015), 6: (0.03, 0.01), 13: (0.0, 0.0)},
        threads=_threads_window(9, 0.07),
        batch=_batch_table({64: -0.3, 128: -0.15, 256: -0.05, 1024: 0.09, 2048: 0.142}),
        load_s=16.0, answer_s={"short": 5.0, "long": 20.0}, overhead_s=3.4,
        within={"short": 0.06, "long": 0.015}, between=0.015, long_tokens=1800, ideal=(0, 9, 2048), major=("ubatch_size", 2048),
    ),
    ModelWorld(
        "Qwen-like (inherited; interior 10 layers +2.6%; short long-question)", blocks=32, top=14,
        start=(8, 12, 256), experts=False, share=lambda L: max(0.55, 1 - L / 33),
        layers={0: (-0.45, -0.3), 1: (-0.42, -0.3), 8: (0.0, 0.0), 10: (0.026, 0.01), 11: (-0.042, -0.02),
                12: (-0.001, 0.0), 14: (0.009, 0.0)},
        batch=_batch_table({64: -0.2, 128: -0.08, 256: 0.0, 512: -0.34}),
        load_s=20.0, answer_s={"short": 6.0, "long": 18.0}, overhead_s=0.2,
        within={"short": 0.025, "long": 0.019}, long_tokens=420, ideal=(10, 12, 256), major=("gpu_offload_layers", 10),
    ),
    ModelWorld(
        "Gemma-like (nothing helps)", blocks=26, top=27, start=(1, 9, 1024), experts=False,
        share=lambda L: 0.9 if L <= 1 else 0.3,
        layers={0: (-0.03, -0.02), 1: (0.0, 0.0), 27: (-0.1, -0.05)},
        threads=_threads_window(9, 0.0), load_s=8.0, answer_s={"short": 1.9, "long": 11.7}, overhead_s=1.2,
        long_tokens=1200, ideal=None,
    ),
    ModelWorld(
        "Interaction: threads help only at CPU-only; batch missing evidence at 2048",
        blocks=32, top=33, start=(33, None, None), experts=True,
        share=lambda L: 1.0 if L == 0 else max(0.1, 1 - L / 33),
        layers={0: (0.30, 0.10), 33: (0.0, 0.0)},
        threads=_threads_window(10, 0.06, at_layers=lambda L: L == 0),
        batch=_batch_table({1024: 0.08, 2048: 0.15}),
        fails=lambda L, t, b: b == 2048, load_s=12.0, answer_s={"short": 2.0, "long": 20.0},
        overhead_s=2.0, long_tokens=1800, ideal=(0, 10, 1024), major=("gpu_offload_layers", 0),
    ),
    ModelWorld(
        "Earlier complete setup: CPU+10 threads beats the later CPU+batch 1024",
        blocks=32, top=33, start=(33, None, None), experts=True,
        share=lambda L: 1.0 if L == 0 else max(0.1, 1 - L / 33),
        layers={0: (math.log(1.2), math.log(1.2)), 33: (0.0, 0.0)},
        batch=_earlier_setup_batch, load_s=12.0, answer_s={"short": 2.0, "long": 20.0},
        overhead_s=2.0, long_tokens=1800, ideal=(0, 10, None), major=("gpu_offload_layers", 0),
    ),
]




class Machine:
    """A laptop on one simulated clock: loads, answers, approval, saving and stopping all
    take simulated time, and the watchdog is driven by this clock (review 8)."""

    APPROVAL_S = 2.0
    SAVE_S = 1.0
    STOP_S = 3.0

    def __init__(self, rng: random.Random, world: ModelWorld) -> None:
        self.rng, self.world = rng, world
        self.seconds = 0.0
        self.loads = 0
        self.drift = 0.0
        self.armed: tuple[float, str] | None = None
        self.fired = False
        self.on_expire: Callable[[str], None] | None = None
        #: (time, setup, question) of every scored answer, for route-independent milestones.
        self.answers: list[tuple[float, tuple[int, int | None, int | None], str]] = []
        #: A user stop scheduled on this clock (review 9): fires wherever the clock
        #: passes it -- during a load, an answer, a load-only try, approval or saving.
        self.stop_at: float | None = None
        self.stop_cb: Callable[[], None] | None = None
        self.stop_fired_phase: str | None = None
        self.phase = "idle"
        self.engine_running = False
        self.save_times: dict[str, float] = {}

    def advance(self, dt: float, *, phase: str = "", interruptible: bool = True) -> None:
        """Move the clock. A scheduled user stop fires at its exact time; an interruptible
        operation (load, answer, load try) is cut there. A watched operation that passes
        its deadline stops there too (stuck)."""
        if phase:
            self.phase = phase
        if self.stop_at is not None and self.stop_fired_phase is None and self.seconds + dt >= self.stop_at:
            remaining = max(self.seconds + dt - self.stop_at, 0.0)
            self.seconds = max(self.seconds, self.stop_at)
            self.stop_fired_phase = self.phase
            if self.stop_cb is not None:
                self.stop_cb()
            if interruptible:
                raise RuntimeError("interrupted by the user's stop")
            dt = remaining
        if self.armed is not None and self.seconds + dt >= self.armed[0]:
            self.seconds = self.armed[0]
            label = self.armed[1]
            self.armed, self.fired = None, True
            if self.on_expire is not None:
                self.on_expire(label)
            raise RuntimeError(f"stopped: {label} appeared stuck")
        self.seconds += dt

    #: Simulated seconds of each saving step, keyed by the commit guard's events.
    #: "hashing" is the model file's content hash before the store prepares anything.
    #: Its 8 s is an assumption (a 2-6 GB file at about 0.5 GB/s), not a laptop
    #: measurement; the real hash checks for a stop between 4 MiB chunks, modelled
    #: here by a stop registering during the step and the guard refusing at once.
    SAVE_STEPS = {"hashing": ("saving: hashing the model file", 8.0),
                  "preparing": ("saving: preparing", 0.3), "writing": ("saving: writing", 0.4),
                  "committing": ("saving: about to commit", 0.1), "committed": ("after commit", 0.2)}

    def save_event(self, name: str) -> None:
        """A saving lifecycle event from ``profiles.CommitGuard``: record when it began
        and spend its simulated time (not interruptible: a stop only registers)."""
        phase, seconds = self.SAVE_STEPS.get(name, (f"saving: {name}", 0.0))
        self.save_times.setdefault(name, self.seconds)
        self.advance(seconds, phase=phase, interruptible=False)

    def new_load(self) -> None:
        self.loads += 1
        self.engine_running = True
        self.advance(self.world.load_s, phase="loading")
        self.drift = 0.3 * self.drift + math.sqrt(1 - 0.09) * self.rng.gauss(0.0, self.world.between)

    def stop_engine(self) -> None:
        if self.engine_running:
            self.engine_running = False
            self.seconds += self.STOP_S

    def respond(self, task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
        q = "long" if task.workload.workload_id.startswith("long-") else "short"
        layers = int(options.gpu_offload_layers or 0)
        hang = getattr(self.world, "hang", None)
        extra = hang(layers, options.threads, options.ubatch_size) if callable(hang) else 0.0
        self.advance(self.world.answer_s[q] + self.world.overhead_s + extra, phase="answering")
        limit = self.world.runs_up_to if self.world.runs_up_to is not None else self.world.top
        if self.world.fails(layers, options.threads, options.ubatch_size) or layers > limit:
            return SampleResult(fail=True)
        s, lg = self.world.effect(layers, options.threads, options.ubatch_size)
        e = s if q == "short" else lg
        within = self.world.within[q]
        speed = 20.0 * math.exp(self.drift + e + self.rng.gauss(0.0, within))
        latency = 2.0 * math.exp(-(self.drift + e + self.rng.gauss(0.0, within)))
        return SampleResult(decode_tps=speed, total_latency_s=latency)


class SimWatchdog:
    def __init__(self, machine: Machine, on_expire: Callable[[str], None]) -> None:
        self.machine = machine
        machine.on_expire = on_expire

    def arm(self, at: float, label: str) -> None:
        self.machine.armed, self.machine.fired = (at, label), False

    def disarm(self) -> bool:
        fired = self.machine.fired
        self.machine.armed, self.machine.fired = None, False
        return fired

    def close(self) -> None:
        self.machine.armed = None


class _Harness(ScriptedHarness):
    def __init__(self, machine: Machine) -> None:
        super().__init__(machine.respond)
        self.machine = machine

    def run_once(self, task: Any, **kwargs: Any) -> Any:
        options = kwargs["options"]
        if kwargs.get("warmup"):
            self.machine.new_load()
        began = self.machine.seconds
        run = super().run_once(task, **kwargs)
        end = self.machine.seconds
        q = "long" if task.workload.workload_id.startswith("long-") else "short"
        if not kwargs.get("warmup"):
            self.machine.answers.append(
                (end, (int(options.gpu_offload_layers or 0), options.threads, options.ubatch_size), q))
        identity = run.identity.model_copy(update={
            "started_at": _BASE + timedelta(seconds=began), "finished_at": _BASE + timedelta(seconds=end)})
        phases = {
            "load_s": self.machine.world.load_s if kwargs.get("warmup") else 0.0,
            "monitor_start_s": self.machine.world.overhead_s * 0.9,
            "request_s": self.machine.world.answer_s[q],
            "monitor_stop_s": self.machine.world.overhead_s * 0.1,
        }
        return run.model_copy(update={"identity": identity, "phase_timings_s": phases})


@contextmanager
def simulated_clock(machine: Machine) -> Iterator[None]:
    original = time.perf_counter
    time.perf_counter = lambda: machine.seconds
    try:
        yield
    finally:
        time.perf_counter = original


class _SimTuner(Tuner):
    """The real Tuner with the world's start, facts and share; records when the final
    check starts and ends (the same milestones for both routes)."""

    world: ModelWorld
    machine_sim: Machine
    finals: list[dict[str, Any]]

    def _starting_point(self, power: Any) -> tuple[RuntimeConfiguration, str]:
        layers, threads, batch = self.world.start
        return RuntimeConfiguration(context_length=self.context_length, gpu_offload_layers=layers,
                                    threads=threads, ubatch_size=batch), "simulated start"

    def _model_facts(self) -> ModelFacts:
        return ModelFacts(total_layers=self.world.blocks, experts=8 if self.world.experts else None)

    def _share(self, config: RuntimeConfiguration) -> float | None:
        return self.world.share(int(config.gpu_offload_layers or 0))

    def _check_whole_setup(self, start: Any, setup: Any, changes: Any, watch: Any, deadline: Any, *, label: str) -> Any:
        entry: dict[str, Any] = {"label": label, "start_t": self.machine_sim.seconds, "setup": setup}
        try:
            ok, confirmation, checks = super()._check_whole_setup(start, setup, changes, watch, deadline, label=label)
        except BaseException:
            entry["end_t"], entry["passed"] = self.machine_sim.seconds, False
            self.finals.append(entry)
            raise
        entry["end_t"], entry["passed"] = self.machine_sim.seconds, ok
        self.finals.append(entry)
        return ok, confirmation, checks

    def _save(self, **kwargs: Any) -> Any:
        if kwargs.get("guard") is not None:
            # Review 10: the practical route saves through the commit guard; its
            # lifecycle events (``Machine.save_event``) move the clock, so a stop lands
            # before or after the real commit, exactly as it would on the laptop.
            return super()._save(**kwargs)
        saved = super()._save(**kwargs)
        self.machine_sim.advance(Machine.SAVE_S, phase="saving", interruptible=False)
        return saved


class _TimedApprover(ScriptedChangeApprover):
    def __init__(self, machine: Machine) -> None:
        super().__init__([True] * 12)
        self.machine = machine

    def decide(self, change: Any, timeout_s: float = 0.0) -> Any:
        self.machine.advance(Machine.APPROVAL_S, phase="approval", interruptible=False)
        return super().decide(change, timeout_s=timeout_s)


@dataclass
class Outcome:
    value: float
    ideal_reached: bool
    harmful: bool
    accepted: bool
    #: Milestones (simulated seconds), the same definitions for both routes.
    t_first_major: float | None
    t_first_strong: float | None
    t_selected: float | None
    selected_value: float | None
    t_verified: float | None
    total: float
    extension: float
    #: User-stop scenarios: "unreached" (the tune ended first), "ok" (nothing published,
    #: ended promptly), "published" (saved anyway: a failure), "after publication" (the
    #: stop arrived while the approved setup was being written), "slow" (ended late).
    stop_status: str = ""
    final_end: float | None = None
    stop_phase: str | None = None
    stop_latency: float | None = None
    outcome: str = ""
    save_times: dict[str, float] = field(default_factory=dict)


def _setup_values(config: RuntimeConfiguration) -> tuple[int, int | None, int | None]:
    return int(config.gpu_offload_layers or 0), config.threads, config.ubatch_size


def one(world: ModelWorld, rng: random.Random, *, target_s: float | None,
        user_stop_at: float | None = None) -> Outcome:
    from inferenceos.tune.budgeted import BudgetedTune

    tmp = Path(tempfile.mkdtemp(prefix="simorder-"))
    settings = _settings(
        tmp, optimization_mode="performance_first",
        tune_paths="threads,reading_threads,flash_attention,prompt_batch",
        tune_diagnosis=False, tune_heat_check_runs=0, tune_path_check="light",
        tune_long_question_tokens=world.long_tokens, tune_between_load_spread=world.between,
        tune_offer_tradeoffs=False,
        **({"tune_experimental_budget_s": target_s} if target_s is not None else {}),
    )
    machine = Machine(rng, world)
    tuner, _, _, _ = _tuner(tmp, machine.respond, answers=[True] * 12, settings=settings, tuner_cls=_SimTuner)
    assert isinstance(tuner, _SimTuner)
    tuner.world, tuner.machine_sim, tuner.finals = world, machine, []
    tuner.model = tuner.model.model_copy(update={"n_layers": world.blocks})
    tuner.context_length = 8192
    tuner.harness = _Harness(machine)
    tuner.approver = _TimedApprover(machine)
    tuner.say = lambda line: None
    tuner.cancel_hook = machine.stop_engine

    def try_load(layers: int, *rest: Any) -> LoadAttempt:
        machine.loads += 1
        try:
            machine.advance(world.load_s, phase="load-only try")
        except RuntimeError as exc:
            return LoadAttempt(loaded=False, fits=False, error=str(exc))
        ok = layers <= world.top
        return LoadAttempt(loaded=ok, fits=ok)

    tuner.try_load = try_load
    driver: Any = None
    with simulated_clock(machine):
        if target_s is None:
            result = tuner.run()
        else:
            driver = BudgetedTune(tuner, target_s, watchdog_factory=lambda cb: SimWatchdog(machine, cb))
            driver.save_events = machine.save_event
            if user_stop_at is not None:
                machine.stop_at = user_stop_at
                machine.stop_cb = lambda: driver.request_cancel("simulated user stop")
            result = driver.run()
    final = _setup_values(result.final_configuration)
    value = world.value(*final)
    s, lg = world.effect(*final)
    harmful = min(math.exp(s) - 1, math.exp(lg) - 1) < -0.02
    accepted = result.saved_setup is not None and result.final_configuration != result.starting_configuration
    ideal_value = world.value(*world.ideal) if world.ideal else 0.0

    def has_major(setup: tuple[int, int | None, int | None]) -> bool:
        if world.major is None:
            return False
        slot = {"gpu_offload_layers": 0, "threads": 1, "ubatch_size": 2}[world.major[0]]
        return setup[slot] == world.major[1]

    t_major = next((t for t, setup, _ in machine.answers if has_major(setup)), None)
    t_strong = next((t for t, setup, _ in machine.answers
                     if ideal_value > 0 and world.value(*setup) >= 0.9 * ideal_value), None)
    finals = [f for f in tuner.finals if f["label"] == "final"]
    t_selected = finals[0]["start_t"] if finals else None
    selected_value = world.value(*_setup_values(finals[0]["setup"])) if finals else None
    t_verified = finals[0]["end_t"] if (finals and finals[0]["passed"] and accepted) else None
    stop_status, latency = "", None
    if user_stop_at is not None:
        if machine.stop_fired_phase is None:
            stop_status = "unreached"
        else:
            # Latency from the scheduled stop, not from when the tune noticed it.
            latency = machine.seconds - user_stop_at
            after_commit = bool(getattr(driver, "cancel_after_publication", False))
            if result.saved_setup is not None and after_commit:
                # The stop registered after the atomic replace: publication had completed.
                stop_status = "after publication"
            elif result.saved_setup is not None:
                stop_status = "published"  # saved although the stop came first: a failure
            elif latency > STOP_LATENCY_LIMIT_S:
                stop_status = "slow"
            else:
                stop_status = "ok"
    return Outcome(
        value=value, ideal_reached=world.ideal is not None and final == world.ideal, harmful=harmful,
        accepted=accepted, t_first_major=t_major, t_first_strong=t_strong, t_selected=t_selected,
        selected_value=selected_value, t_verified=t_verified, total=machine.seconds,
        extension=max(machine.seconds - target_s, 0.0) if target_s else 0.0, stop_status=stop_status,
        stop_phase=machine.stop_fired_phase, stop_latency=latency, outcome=result.outcome,
        final_end=finals[0]["end_t"] if finals else None, save_times=dict(machine.save_times),
    )


def _m(values: list[float]) -> str:
    return f"{statistics.mean(values) / 60:6.1f}" if values else "     -"


def run(trials: int, seed: int, targets: list[float | None], worlds: list[ModelWorld],
        say: Callable[[str], None] = print) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    say(f"{trials} simulated tunes per row; simulated minutes (mean over runs where the milestone happened)")
    say(f"{'route':12} {'gain':>6} {'vsU':>5} {'ideal':>5} {'harm':>4} {'verif':>5} "
        f"{'1st major':>9} {'1st strong':>10} {'selected':>8} {'verified':>8} {'total':>6} {'ext mean/max':>13}")
    for world in worlds:
        ideal = world.value(*world.ideal) if world.ideal else 0.0
        say(f"{world.name}  (ideal gain {ideal:+.1%}; major: {world.major})")
        unbounded_gain: float | None = None
        for target in targets:
            route = "unbounded" if target is None else f"target {target / 60:.0f}m"
            rng = random.Random(f"{seed}-{world.name}-{route}")
            results = [one(world, rng, target_s=target) for _ in range(trials)]
            gain = statistics.mean(r.value for r in results)
            if target is None:
                unbounded_gain = gain
            retained = (gain / unbounded_gain) if (unbounded_gain and unbounded_gain > 0) else float("nan")
            exts = [r.extension for r in results]
            row: dict[str, object] = {
                "gain": gain, "retained_vs_unbounded": retained,
                "ideal": sum(r.ideal_reached for r in results) / trials,
                "harmful": sum(r.harmful for r in results) / trials,
                "accepted": sum(r.accepted for r in results) / trials,
                "t_first_major_min": _mean_min([r.t_first_major for r in results]),
                "t_first_strong_min": _mean_min([r.t_first_strong for r in results]),
                "t_selected_min": _mean_min([r.t_selected for r in results]),
                "t_verified_min": _mean_min([r.t_verified for r in results]),
                "verified_share": sum(r.t_verified is not None for r in results) / trials,
                "total_min": statistics.mean(r.total for r in results) / 60,
                "extension_mean_s": statistics.mean(exts), "extension_max_s": max(exts),
            }
            out[f"{world.name} | {route}"] = row
            say(
                f"{route:12} {gain:+6.1%} {retained:5.0%} {row['ideal']:5.0%} {row['harmful']:4.0%} "
                f"{row['verified_share']:5.0%} {_m([r.t_first_major for r in results if r.t_first_major is not None]):>9} "
                f"{_m([r.t_first_strong for r in results if r.t_first_strong is not None]):>10} "
                f"{_m([r.t_selected for r in results if r.t_selected is not None]):>8} "
                f"{_m([r.t_verified for r in results if r.t_verified is not None]):>8} "
                f"{row['total_min']:6.1f} {statistics.mean(exts):6.0f}/{max(exts):5.0f}s"
            )
    return out


def _mean_min(values: list[float | None]) -> float | None:
    real = [v for v in values if v is not None]
    return statistics.mean(real) / 60 if real else None


def stops(trials: int, seed: int, say: Callable[[str], None] = print) -> dict[str, object]:
    """Explicit user stops and stuck operations on the simulated clock."""
    world = WORLDS[1]
    rng = random.Random(f"{seed}-stops")
    # Stops at random times over the whole tune (loading, answering, load tries, the
    # final check, approval, saving), on the simulated clock.
    times = [rng.uniform(0.0, 3200.0) for _ in range(max(trials, 8) * 2)]
    user = [one(world, rng, target_s=3600.0, user_stop_at=at) for at in times]
    counts: dict[str, int] = {}
    for r in user:
        counts[r.stop_status] = counts.get(r.stop_status, 0) + 1
    reached = [r for r in user if r.stop_status != "unreached"]
    phases: dict[str, int] = {}
    for r in reached:
        phases[str(r.stop_phase)] = phases.get(str(r.stop_phase), 0) + 1
    # Stops aimed at approval and at each saving step (review 10: the commit guard's own
    # lifecycle events): replay the same tune (same random stream) and schedule the stop
    # inside the step.
    aimed: list[Outcome] = []
    aimed_at: list[str] = []
    for k in range(max(trials // 4, 2)):
        base = one(world, random.Random(f"{seed}-aim-{k}"), target_s=3600.0)
        if base.final_end is None or not base.save_times:
            continue
        targets = [("approval", base.final_end + 1.0)] + [
            (name, base.save_times[name] + 0.05)
            for name in ("hashing", "preparing", "writing", "committing", "committed")
            if name in base.save_times]
        for name, at in targets:
            aimed.append(one(world, random.Random(f"{seed}-aim-{k}"), target_s=3600.0, user_stop_at=at))
            aimed_at.append(name)
    for name, r in zip(aimed_at, aimed):
        counts[f"aimed {name}:{r.stop_status}"] = counts.get(f"aimed {name}:{r.stop_status}", 0) + 1
        phases[str(r.stop_phase)] = phases.get(str(r.stop_phase), 0) + 1
    reached += [r for r in aimed if r.stop_status != "unreached"]
    lat = [r.stop_latency for r in reached if r.stop_latency is not None]
    say(f"user stops at random times: {counts}; phase when the stop arrived: {phases}; latency from the "
        f"scheduled stop: mean {statistics.mean(lat) if lat else 0:.1f} s, max {max(lat) if lat else 0:.1f} s "
        f"(limit {STOP_LATENCY_LIMIT_S:.0f} s)")
    hang_world = dataclasses_replace(world, name="hangs at 1 layer", hang=lambda L, t, b: 5000.0 if L == 1 else 0.0)
    hung = [one(hang_world, rng, target_s=3600.0) for _ in range(max(trials // 4, 1))]
    say(f"stuck answer (1 layer hangs 5000 s): mean total {statistics.mean(r.total for r in hung) / 60:.1f} min, "
        f"saved {sum(r.accepted for r in hung)}/{len(hung)}, max extension {max(r.extension for r in hung):.0f} s")
    return {"user_stops": counts, "phases": phases, "stuck_totals_min": [r.total / 60 for r in hung]}


def dataclasses_replace(world: ModelWorld, **changes: Any) -> ModelWorld:
    import copy

    new = copy.copy(world)
    for key, value in changes.items():
        setattr(new, key, value)
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--targets", default="unbounded,30,45,60,90")
    parser.add_argument("--worlds", default="")
    parser.add_argument("--stops", action="store_true")
    args = parser.parse_args(argv)
    targets: list[float | None] = [None if item == "unbounded" else float(item) * 60 for item in args.targets.split(",")]
    worlds = WORLDS
    if args.worlds:
        wanted = {int(i) for i in args.worlds.split(",")}
        worlds = [w for i, w in enumerate(WORLDS) if i in wanted]
    run(args.trials, args.seed, targets, worlds)
    if args.stops:
        stops(args.trials, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
