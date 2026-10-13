"""Experimental practical tune with a soft time target (internal; handoff of 6 Oct 2026,
revised after review 8).

Not a product mode and not a default: it runs only when the internal setting
``tune_experimental_budget_s`` (the *target*) is above zero. Unbounded is unchanged.

How time is used
----------------
* **Soft target, not a cutoff.** Exploration launches a measurement only when
  ``now + its measured cost + the final check's predicted cost + save reserve`` stays
  within the target. Once exploration stops it never restarts. A measurement pair that
  was started is finished (its candidate half may run past the target), and the final
  check starts whenever its predicted cost fits within ``target + extension`` (the
  extension is ``tune_experimental_extension_share`` of the target, at least
  ``..._extension_min_s``). A passing result is kept even if it finishes after the
  target; every overrun is explained from measured costs in the report.
* **Three different stops.**
  - *Target reached*: no new exploration; nothing running is interrupted.
  - *User cancellation* (``request_cancel``): the engine is stopped at once, every
    later step raises ``Cancelled``, and nothing new is published (checked again at
    the publication boundary, safety corrections included). Load-only tries,
    measurements, approval and saving all honour it.
  - *Operation appears stuck*: each answer and load try is watched against its expected
    time from this tune's measurements (``tune_experimental_stuck_factor`` times, at
    least ``..._stuck_grace_s`` longer). A stuck operation is stopped, its evidence is
    missing, exploration ends; a stuck final check is never accepted and never retried.
* **Best so far, at every moment.** The planner's best-so-far setup (the best clear
  placement reading even before placement is settled) is what the final check confirms
  when exploration ends early; collected evidence is never dropped by stopping.

What is confirmed
-----------------
Candidates for the final check are setups measured *as a whole*: the best-so-far setup
and its subsets (some settings back at the start), and weak setups that were measured
again. Each is valued by its own direct readings against the starting setup on the
questions the final check needs (never by adding component gains). The final check's
blocks are sized from those direct readings. The most valuable candidate whose check
fits in the time left is confirmed; a cheaper, affordable alternative is used rather
than dropping everything when a small extra setting makes the full candidate too costly.

Weak readings are eligible: once the clear gains are exhausted, the best weak readings
(at most ``tune_experimental_weak_repeats``) are measured again when the repeat and its
final check fit; a reading that holds up joins the candidate, and a weak setup can be
confirmed on its own when nothing clearer exists.

Statistical policy
------------------
Each question of the final check is one look at N blocks (AB/BA, fresh loads), N fixed
before the check from this tune's screening noise and the candidate's own direct
readings (about 80% power), at the 5% one-sided false-keep rate: "better" on found
questions (with the smallest worthwhile gain), "not worse" (2% margin) on guard
questions; all must pass. Screening readings only select. No retries, no automatic
per-setting checks, no fallback confirmation after a failed final check.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from inferenceos.cards.builtin.gpu_layers_more import CARD_ID as LAYERS_CARD_ID
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.experiment import ExperimentMeasurement, RuntimeConfiguration
from inferenceos.contracts.tune import ChangeDecision, ConfirmationResult, ProposedChange, TuneResult
from inferenceos.optimization.search.resources import build_resource_envelope
from inferenceos.profiling.workloads import BenchmarkWorkload
from inferenceos.telemetry.counters import MonitorCleanupUnresolved
from inferenceos.tune.approval import describe_change
from inferenceos.tune.layer_top import LoadAttempt, find_layer_top
from inferenceos.tune.measure import DisturbanceWatch, measure_steady, run_score_spread
from inferenceos.tune.noise_store import noise_key
from inferenceos.tune.paths import _batch_sizes
from inferenceos.tune.placement import engine_rule, max_count
from inferenceos.tune.planner import (
    BATCH,
    CATEGORIES,
    QUESTION,
    SETTING,
    SLOT,
    TABLES,
    Planner,
    Setup,
    Step,
    Structure,
)
from inferenceos.tune.profiles import CommitGuard, power_key, power_words
from inferenceos.tune.stats import t_quantile
from inferenceos.tune.verify_cost import CostLedger

if TYPE_CHECKING:
    from inferenceos.tune.pipeline import Tuner

#: C3 adapter: the card each practical change is recorded under comes from the path
#: declarations (identical to the old literal; checked in tests/test_path_declarations.py).
CARD_OF = dict(TABLES.card_of)
#: A batch-changing candidate whose direct short-question reading loses more than this
#: (and two noise widths) is not sent to a final check it would very likely fail.
GUARD_SCREEN_TOLERANCE = 0.03
#: EXPERIMENTAL / PROVISIONAL (reviewer follow-up, item 5): inflation of the screening
#: spread when sizing the final check's blocks. Chosen from simulation; the historical
#: calibration (jobs 090-107) does not observe the estimator Practical uses (a fresh
#: direct whole-setup reading), so it is neither confirmed nor changed. Every final
#: check records what it used (``_sizing_record``) so future runs can calibrate it.
SPREAD_MARGIN = 1.25
#: EXPERIMENTAL / PROVISIONAL: shrink of the direct reading for winner's curse when
#: sizing blocks (same status as ``SPREAD_MARGIN``).
WINNER_SHRINK = 0.7
Z_POWER = 1.645 + 0.842
#: EXPERIMENTAL / PROVISIONAL (confirmation planning, 9 Oct 2026). Before the one fresh
#: final check starts, its block count is re-planned for the chosen setup
#: (``_plan_confirmation``): enough blocks for ``PLAN_POWER`` under a *planning* spread
#: (the screening noise width x an uncertainty allowance), using the Student t critical
#: value of the actual test -- but only when that check still finishes by the soft
#: target; otherwise the selection-time count (``_blocks``) is kept, exactly as before.
#: The test itself, its alpha, its single look and the acceptance rule are unchanged.
#:
#: The allowance is how much larger the realised block-to-block spread of a fair
#: comparison may be than its screening noise width. From the 48 recorded comparisons of
#: jobs 090-107 (``q6_final_check_calibration.csv``; each realised spread corrected for
#: its own 4- or 16-block sampling error, assuming a log-normal ratio): median about
#: 0.8, 80th percentile about 1.3 over all comparisons, about 1.75 over the 10 fresh
#: whole-setup final checks. All of them used the 1.2% fallback load-to-load spread, so
#: 1.75 applies when this tune also uses the fallback. Not fitted to jobs 123/124.
NOISE_ALLOWANCE_FALLBACK = 1.75
#: When the load-to-load spread is learned for this model and question there is no
#: calibration yet; the existing selection margin is used.
NOISE_ALLOWANCE_LEARNED = SPREAD_MARGIN
#: Planned chance of a "better" decision when the planning spread and effect are right.
PLAN_POWER = 0.8
#: The final check never plans more blocks than this (the existing cap of ``_blocks``).
MAX_FINAL_BLOCKS = 16
#: Version of the confirmation plan record (``_plan_confirmation``). Reporting only.
#: Earlier records carry none: the first format (job 125) and the 9 Oct reporting-fix
#: format are told apart by their recorded rule text (``scripts/explain_practical_run.py``).
PLAN_FORMAT_VERSION = "confirmation-plan-3"
#: How far (in averaged noise widths) above the smallest worthwhile gain a repeated weak
#: reading must be to get a final check of its own.
WEAK_MARGIN = 0.5
#: Earlier complete setups kept as final-check candidates.
HISTORY_SIZE = 6


class Cancelled(BaseException):
    """The user stopped the tune. A BaseException, so no measurement loop swallows it as
    an ordinary failed run."""


class Abandoned(BaseException):
    """The final check stopped at a measurement boundary because finishing had become
    much more expensive than its allowance (never an accepted result)."""


class Watchdog(Protocol):
    def arm(self, at: float, label: str) -> None: ...
    def disarm(self) -> bool: ...
    def close(self) -> None: ...


class ThreadWatchdog:
    """Calls ``on_expire(label)`` if an armed operation passes its time (real clock)."""

    def __init__(self, on_expire: Callable[[str], None], poll_s: float = 0.5) -> None:
        self._on_expire = on_expire
        self._poll = poll_s
        self._lock = threading.Lock()
        self._armed: tuple[float, str] | None = None
        self._fired = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self._poll):
            with self._lock:
                armed = self._armed
                due = armed is not None and time.perf_counter() >= armed[0]
                if due:
                    self._armed = None
                    self._fired = True
            if due and armed is not None:
                self._on_expire(armed[1])

    def arm(self, at: float, label: str) -> None:
        with self._lock:
            self._armed = (at, label)
            self._fired = False

    def disarm(self) -> bool:
        with self._lock:
            fired = self._fired
            self._armed = None
            self._fired = False
            return fired

    def close(self) -> None:
        self._stop.set()


class _WatchedHarness:
    """The tune's measurement harness with every answer watched and the user's stop
    honoured between answers."""

    def __init__(self, inner: Any, driver: BudgetedTune) -> None:
        self._inner = inner
        self._driver = driver

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def run_once(self, task: Any, **kwargs: Any) -> Any:
        d = self._driver
        d.check_user()
        question = task.workload.workload_id
        expected = d.expected_run_s(question, warmup=bool(kwargs.get("warmup")))
        d.arm(expected, f"answer on {question}" + (" (warm-up with load)" if kwargs.get("warmup") else ""))
        try:
            run = self._inner.run_once(task, **kwargs)
        finally:
            fired = d.disarm()
        d.check_user()
        if fired:
            raise RuntimeError("stopped: the answer appeared stuck")
        return run


def fixed_start(text: str, context_length: int) -> tuple[RuntimeConfiguration, str]:
    """The practical search's fixed starting configuration (``tune_experimental_start_config``,
    internal evaluation only): a JSON object of ``RuntimeConfiguration`` fields; every field
    not given is the engine default. The context size is always the run's own and may not
    be set here. Raises ``ValueError`` on anything else (never silently ignored)."""
    import json

    try:
        fields = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"tune_experimental_start_config is not valid JSON: {exc}") from exc
    if not isinstance(fields, dict) or not fields:
        raise ValueError("tune_experimental_start_config must be a non-empty JSON object")
    if "context_length" in fields:
        raise ValueError("tune_experimental_start_config may not set the context size (it is the run's own)")
    config = RuntimeConfiguration(**fields, context_length=context_length)
    given = ", ".join(f"{k}={v}" for k, v in sorted(fields.items()))
    return config, f"fixed starting configuration for this evaluation ({given}; every other setting the engine default)"


@dataclass
class BudgetedTune:
    tuner: Tuner
    budget_s: float
    #: Makes the watchdog from the "stuck" callback; None = a real watchdog thread.
    watchdog_factory: Callable[[Callable[[str], None]], Watchdog] | None = None
    ledger: CostLedger = field(default_factory=CostLedger)
    user_cancelled: str | None = None
    #: The stop arrived after the new setup had been committed (it stays saved).
    cancel_after_publication: bool = False
    #: Saving lifecycle events (``preparing`` ... ``committed``), for tests and the
    #: simulator; see ``profiles.CommitGuard``.
    save_events: Callable[[str], None] | None = None
    stuck: list[str] = field(default_factory=list)
    timeline: list[dict[str, object]] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    #: Every operation in order (report only; practical-mode evaluation, 8 Oct 2026):
    #: load-only tries, screening and direct-reading measurements, long-question sizing,
    #: and each final-check measurement, with when it began, how long it took, what it
    #: measured and where its time went.
    operations: list[dict[str, object]] = field(default_factory=list)
    #: Every planner step as it was taken: when, what, and the planner's reason.
    steps: list[dict[str, object]] = field(default_factory=list)
    #: Every screening comparison, recorded when it happened and never changed afterwards
    #: (report only; reviewer, 8 Oct 2026): the complete reference and candidate setups
    #: and configurations, the question, which measurement (operation number) each side
    #: used, this comparison's own gain and noise, the planner's aggregate right after it,
    #: and the candidate's whole-setup gain against the original start as known then.
    screenings: list[dict[str, object]] = field(default_factory=list)
    _meas: dict[tuple[Setup, str], ExperimentMeasurement] = field(default_factory=dict)
    #: Each usable measurement object with the operation number that produced it (report only).
    _meas_ops: list[tuple[ExperimentMeasurement, int]] = field(default_factory=list)
    _t0: float = 0.0
    _target: float = 0.0
    _hard: float = 0.0
    _save_reserve: float = 0.0
    _watchdog: Watchdog | None = None
    _short: BenchmarkWorkload | None = None
    _long: BenchmarkWorkload | None = None
    _safety_pending: bool = False
    _load_try_s: float = 0.0
    _spread: dict[str, float] = field(default_factory=dict)
    #: Where each question's largest screening noise width came from (report only).
    _spread_info: dict[str, dict[str, object]] = field(default_factory=dict)
    #: The final check's confirmation plan, recorded before it starts (report only).
    _confirmation_plan: dict[str, Any] = field(default_factory=dict)
    _start_setup: Setup = (0, 0, 0)
    _safe_start: RuntimeConfiguration | None = None
    _planner: Planner | None = None
    _weak_tried: set[tuple[str, int, tuple[int, int]]] = field(default_factory=set)
    _rounds: list[dict[str, object]] = field(default_factory=list)
    _notes: list[str] = field(default_factory=list)
    _base: TuneResult | None = None
    _batch_left_out: bool = False
    _batch_retried: bool = False
    #: Progress of the final check while it runs (None otherwise): planned measurements
    #: per question, their predicted cost, and what has been done.
    _final: dict[str, Any] | None = None
    #: Taken by a cancellation to register and by the commit for its last check and the
    #: atomic replace: a stop is either before the commit or after it, never between.
    _commit_lock: threading.Lock = field(default_factory=threading.Lock)
    _published: bool = False

    # ------------------------------------------------------------------ #
    # Stops                                                                #
    # ------------------------------------------------------------------ #
    def request_cancel(self, reason: str = "stopped by the user") -> None:
        """Explicit user cancellation: stop the engine now; nothing new is published."""
        with self._commit_lock:
            if self.user_cancelled is not None:
                return
            self.user_cancelled = reason
            self.cancel_after_publication = self._published
        self._stop_engine(user=True)

    def _mark_published(self) -> None:
        self._published = True  # under the commit lock, right after the replace

    def check_user(self) -> None:
        if self.user_cancelled is not None:
            raise Cancelled(self.user_cancelled)

    def _on_stuck(self, label: str) -> None:
        self.stuck.append(f"{label} at {self._now():.0f} s")
        self._stop_engine()

    def _stop_engine(self, *, user: bool = False) -> None:
        # A user stop is sticky (nothing starts again); a stuck operation or the timed
        # cleanup only ends what runs, so the tune can go on measuring.
        hook = (self.tuner.user_stop_hook or self.tuner.cancel_hook) if user else self.tuner.cancel_hook
        if hook is not None:
            try:
                hook()
            except Exception:  # noqa: BLE001 - reported by the cleanup step
                pass

    def arm(self, expected_s: float, label: str) -> None:
        if self._watchdog is None:
            return
        factor = float(self.tuner.settings.tune_experimental_stuck_factor)
        grace = float(self.tuner.settings.tune_experimental_stuck_grace_s)
        self._watchdog.arm(time.perf_counter() + max(factor * expected_s, expected_s + grace), label)

    def disarm(self) -> bool:
        return self._watchdog.disarm() if self._watchdog is not None else False

    def expected_run_s(self, question: str, *, warmup: bool) -> float:
        """Expected seconds of one answer (plus the load for a warm-up), from this tune's
        measurements; generous defaults before anything was measured."""
        answer = self.ledger.answer_cost(question)
        if answer is None:
            cost = self.ledger.measurement_cost(question)
            runs = self.tuner._path_policy().measured_repetitions + 1
            answer = cost / (runs + 1) if cost else (
                float(self.tuner.settings.tune_long_question_seconds) if question.startswith("long") else 30.0
            )
        load = self.ledger.load_cost() or 120.0
        return answer + (load if warmup else 0.0)

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    def final_progress(self, event: dict[str, Any]) -> None:
        """Pace the final check from completed comparison work (review 10).

        Fed by ``compare_fairly``'s progress events. Measurement time, counted blocks,
        discarded blocks and disturbed repeats are recorded separately per question; a
        repeat or a discarded block costs time but does not reduce the blocks still
        required. At each block boundary the finish is projected as::

            now + sum over questions of remaining blocks x (cost of the current side
                  + cost of the new side) + the save-and-cleanup allowance

        where a side's cost is its own observed per-attempt cost (repeats excluded) once
        measured and the original estimate for work not yet observed. Decision (stuck
        answers are the watchdog's business, not this one's):

        * projected within target + extension: continue;
        * beyond that, but the remaining work is at most a quarter of the check and the
          overrun stays under one more extension: continue (experimental rule);
        * otherwise stop at this boundary (``Abandoned``). An incomplete check is never
          accepted, the planned look is unchanged, and nothing is decided after the last
          look -- a completed check is always kept."""
        f = self._final
        if f is None:
            return
        name = str(event.get("workload", ""))
        q = f["ids"].get(name, "short")
        state = f["q"].setdefault(q, {"usable": 0, "discarded": 0, "attempts": 0, "retries": 0,
                                      "costs": {"start": [], "candidate": []}, "finished": False,
                                      "block_scores": [], "discarded_why": []})
        kind = event.get("event")
        if f.get("current") not in (None, q):
            f["q"][f["current"]]["finished"] = True  # its comparison ended; the next began
        f["current"] = q
        state["planned"] = int(event.get("planned", state.get("planned", 0)) or 0)
        state["max_attempts"] = int(event.get("max_attempts", state.get("max_attempts", 0)) or 0)
        if kind == "measurement":
            retries = int(event.get("retries", 0) or 0)
            wall = float(event.get("wall_s", 0.0) or 0.0)
            state["retries"] += retries
            state["costs"][str(event.get("side", "start"))].append(wall / (1 + retries))
            f["measurements"].append({"question": q, "side": event.get("side"), "wall_s": round(wall, 2),
                                      "retries": retries, "at_s": round(self._now(), 1)})
            self.operations.append({
                "n": len(self.operations) + 1, "kind": "final", "label": f"final check {q} {event.get('side')}",
                "question": q, "side": event.get("side"), "block": event.get("block"),
                "began_s": round(self._now() - wall, 1), "wall_s": round(wall, 2), "retries": retries})
            return
        if kind == "block":
            state["attempts"] = int(event.get("attempts", 0) or 0)
            state["usable"] = int(event.get("usable", 0) or 0)
            state["discarded"] = int(event.get("discarded", 0) or 0)
            # The evidence itself, kept here so a comparison stopped before it returns
            # (abandoned, cancelled) still leaves its completed blocks on record
            # (reviewer follow-up 3, item 4).
            if event.get("counted") and event.get("score") is not None:
                state["block_scores"].append(float(event["score"]))
            elif not event.get("counted"):
                state["discarded_why"].append(str(event.get("why", "")))
            return
        if kind != "boundary":
            return
        projection = self._project_final()
        f["last_projection"] = projection
        remaining_s, total_s = projection["remaining_s"], projection["total_s"]
        if remaining_s <= 0:
            return
        projected = self._t0 + projection["projected_finish_s"]
        extension = self._hard - self._target
        modest = remaining_s <= 0.25 * total_s
        if projected <= self._hard or (modest and projected <= self._hard + extension):
            return
        slow = ", ".join(f"{k}: {v:.0f} s per measurement (estimated {f['cost'].get(k.split()[0], 0.0):.0f} s)"
                         for k, v in projection["observed_s"].items())
        f["abandoned"] = (
            f"at {self._now():.0f} s, after {projection['usable_blocks']} counted block(s), finishing was "
            f"projected at {projection['projected_finish_s']:.0f} s ({slow or 'no measurement observed'}), "
            f"beyond the target ({self._target - self._t0:.0f} s) plus its extension ({extension:.0f} s)")
        raise Abandoned(f["abandoned"])

    def _project_final(self) -> dict[str, Any]:
        f = self._final
        assert f is not None
        remaining = total = 0.0
        observed: dict[str, float] = {}
        usable_blocks = 0
        for q, blocks in f["planned_blocks"].items():
            state = f["q"].get(q)
            estimate = float(f["cost"].get(q, 0.0))
            side_cost = 0.0
            for side in ("start", "candidate"):
                seen = state["costs"][side] if state else []
                cost = sum(seen) / len(seen) if seen else estimate
                if seen:
                    observed[f"{q} {'current' if side == 'start' else 'new'}"] = cost
                side_cost += cost
            total += blocks * side_cost
            if state is None:
                remaining += blocks * side_cost
                continue
            usable_blocks += state["usable"]
            if state["finished"]:
                continue
            required = max(blocks - state["usable"], 0)
            attempts_left = max((state.get("max_attempts") or blocks) - state["attempts"], 0)
            remaining += min(required, attempts_left) * side_cost
        now = self._now()
        return {
            "at_s": round(now, 1), "usable_blocks": usable_blocks,
            "remaining_s": round(remaining, 1), "total_s": round(total, 1),
            "save_allowance_s": round(self._save_reserve, 1),
            "projected_finish_s": round(now + remaining + self._save_reserve, 1),
            "observed_s": {k: round(v, 1) for k, v in observed.items()},
            "retries": {q: st["retries"] for q, st in f["q"].items()},
            "discarded": {q: st["discarded"] for q, st in f["q"].items()},
        }

    # ------------------------------------------------------------------ #
    def run(self) -> TuneResult:
        t = self.tuner
        s = t.settings
        self._t0 = time.perf_counter()
        self._target = self._t0 + float(self.budget_s)
        extension = max(float(s.tune_experimental_extension_share) * float(self.budget_s),
                        float(s.tune_experimental_extension_min_s))
        self._hard = self._target + extension
        self._save_reserve = float(s.tune_experimental_save_reserve_s)
        self._watchdog = (self.watchdog_factory or ThreadWatchdog)(self._on_stuck)
        power = power_key(t.machine.power_source)
        fixed = str(getattr(s, "tune_experimental_start_config", "") or "").strip()
        if fixed:
            starting, source = fixed_start(fixed, t.context_length)
        else:
            starting, source = t._starting_point(power)
        self._notes = [
            f"experimental practical search: target {self.budget_s:.0f} s, extension up to "
            f"{extension:.0f} s (internal; not a product mode)"
        ]
        self._base = TuneResult(
            run_id=t.run_id or f"tune-practical-{int(time.time())}", model_path=str(t.model_path),
            context_length=t.context_length, effort_mode=t.rules.effort.mode.value,
            goal_mode=t.rules.goal.mode.value, time_limit_s=float(self.budget_s), machine=t.machine,
            machine_changes=list(t.machine_changes), starting_configuration=starting,
            starting_point_source=source, final_configuration=starting,
        )
        inner_harness, inner_try = t.harness, t.try_load
        t.harness = _WatchedHarness(inner_harness, self)
        if inner_try is not None:
            t.try_load = self._watched_try(inner_try)
        t.say(f"Experimental practical search: aiming for {self.budget_s / 60:.0f} minutes.")
        t.say(f"Starting setup ({source}): {starting}")
        try:
            return self._run(starting, power)
        except Cancelled as stop:
            self._notes.append(f"cancelled by the user ({stop}); nothing new was saved")
            return self._finish(outcome="cancelled by the user: nothing new saved")
        except MonitorCleanupUnresolved as blocked:
            # Experimental scoped monitor only: a sampler may still be running, so
            # nothing further is measured and nothing is published.
            self._notes.append(f"stopped: {blocked}; nothing new was saved")
            return self._finish(outcome=f"failed: {blocked}; nothing saved")
        finally:
            t.harness, t.try_load = inner_harness, inner_try
            if self._watchdog is not None:
                self._watchdog.close()

    def _watched_try(self, inner: Callable[..., LoadAttempt]) -> Callable[..., LoadAttempt]:
        def watched(*args: Any) -> LoadAttempt:
            self.check_user()
            self.arm(self.ledger.load_cost() or 120.0, "load-only try")
            began, at = time.perf_counter(), self._now()
            outcome: LoadAttempt | None = None
            try:
                outcome = inner(*args)
            finally:
                fired = self.disarm()
                self.operations.append({
                    "n": len(self.operations) + 1, "kind": "load-try", "label": f"load-only try {args[0] if args else ''}".strip(),
                    "began_s": round(at, 1), "wall_s": round(time.perf_counter() - began, 2),
                    "layers": args[0] if args else None,
                    "result": None if outcome is None else ("fits" if outcome.fits else "does not fit"),
                    "stuck": fired})
            self.check_user()
            if fired:
                return LoadAttempt(loaded=False, fits=False, error="stopped: the load appeared stuck")
            return outcome
        return watched

    # ------------------------------------------------------------------ #
    def _run(self, starting: RuntimeConfiguration, power: PowerState) -> TuneResult:  # noqa: C901
        t = self.tuner
        short = self._short = t.workload
        began = time.perf_counter()
        top, count_max, top_record = self._layer_top(starting)
        self._load_try_s = time.perf_counter() - began
        self.ledger.add("load-try", None, self._load_try_s, started_at=self._now())
        self._rounds.append(top_record)
        start_layers = int(starting.gpu_offload_layers or 0)
        safety: ProposedChange | None = None
        safe_start = starting
        if start_layers > top:
            safety = ProposedChange(
                change_id=f"{LAYERS_CARD_ID}:safety:{top}", card_id=LAYERS_CARD_ID,
                setting="gpu_offload_layers", description="Graphics card layers (memory safety)",
                old_value=start_layers, new_value=top, measured_gain=None,
                evidence=["the starting count does not leave enough graphics memory free", f"top={top}"],
            )
            safe_start = starting.model_copy(update={"gpu_offload_layers": top})
            self._safety_pending = True
            t.say(f"  memory safety: {describe_change(safety)} (no speed claim)")
        self._safe_start = safe_start

        facts = t._model_facts()
        machine = t._machine_facts_now()
        planner = self._planner = Planner(
            Structure(
                count_max=count_max, top=top, uses_experts=bool(facts.experts),
                mixed_kinds=bool(facts.other_layers),
                physical_cores=int(machine.physical_cores or t.hardware.cpu.physical_cores or 1),
                logical_cores=int(machine.logical_cores or t.hardware.cpu.logical_cores or 1),
            ),
            start_layers=int(safe_start.gpu_offload_layers or 0), start_threads=safe_start.threads,
            start_batch=safe_start.ubatch_size, wants_question_length=True,
            processor_share=lambda layers: t._share(safe_start.model_copy(update={"gpu_offload_layers": layers})),
            batch_fits=lambda layers, size: self._batch_fits(safe_start, layers, size),
            smallest_gain=float(t.rules.effort.minimum_improvement),
        )
        self._start_setup = planner.start_setup()

        # As in Unbounded, the starting setup itself is measured first: a memory-safety
        # correction's answers and paging are compared with it.
        start_reference = self._measure_config(starting, short, "start")
        if start_reference is None:
            return self._finish(outcome="failed: the starting setup could not be measured")
        if safety is None:
            self._meas[(self._start_setup, short.workload_id)] = start_reference
        elif self._measure(self._start_setup, short, "memory-safe start") is None:
            return self._finish(outcome="not saved: the memory-safe setup could not be measured", proposed=[safety])

        stopped_for = self._explore(planner)
        best = planner.best_so_far()
        self.timeline.append({"t": round(self._now(), 1), "candidate": self._settings_of(best), "event": "exploration ended"})
        discovery_s = self._now()

        # Direct readings: the best-so-far setup on every question its check needs (a
        # worthwhile measurement: it may use the extension), then on the goal's other
        # questions, and repeated weak setups on theirs, while that fits the target. Each
        # candidate is valued by its own readings, never by adding component gains.
        self._complete_readings(best, final_only=True, limit=self._hard)
        self._complete_readings(best, final_only=False, limit=self._target)
        for c in CATEGORIES:
            for r in planner.readings[c]:
                if r.count > 1 and r.gain is not None and r.gain > 0:
                    self._complete_readings(r.setup, final_only=False, limit=self._target)
        # Earlier complete setups that were promising (review 9): the best two get the
        # readings they lack, while that fits the target (evidence, not new exploration).
        for earlier in self._to_complete():
            self._complete_readings(earlier, final_only=False, limit=self._target)

        options = self._options(planner)
        chosen = self._choose(options)
        self._rounds.append({
            "round": "practical search",
            "plan": list(planner.log),
            # The planner's final aggregates (report only: when each piece of evidence
            # arrived is in the time budget's ``screenings``).
            "readings": [
                {"category": r.category, "value": r.value, "gain": r.gain, "noise": round(r.noise, 4),
                 "setup": list(r.setup), "repeats": r.count}
                for c in CATEGORIES for r in planner.readings[c]
            ],
            "start_setup": list(self._start_setup),
            "best_so_far": self._settings_of(best),
            "options": [
                {k: v for k, v in item.items() if k != "setup"} | {"settings": self._settings_of(item["setup"])}  # type: ignore[arg-type]
                for item in options
            ],
            "chosen": None if chosen is None else self._settings_of(chosen["setup"]),  # type: ignore[arg-type]
            "chosen_setup": None if chosen is None else list(chosen["setup"]),  # type: ignore[call-overload]
            "measured_setups": self._measured_status(chosen),
            "deferred": [{"category": r.category, "value": r.value, "gain": r.gain, "noise": round(r.noise, 4),
                          "setup": list(r.setup), "repeats": r.count}
                         for r in planner.deferred()],
            "stopped_for": stopped_for or "plan finished",
            "stuck": list(self.stuck),
            "discovery_s": round(discovery_s, 1),
            "timeline": list(self.timeline),
        })
        if stopped_for:
            self._notes.append(f"exploration stopped: {stopped_for}")

        safety_check: ConfirmationResult | None = None
        if safety is not None:
            self.check_user()
            t.say("Checking the memory-safe setup runs and answers correctly")
            safety_check, record = t._check_safety_correction(starting, safe_start, start_reference,
                                                              DisturbanceWatch(logical_cores=t.hardware.cpu.logical_cores,
                                                                               limit_percent=float(t.settings.tune_disturbance_cpu_percent),
                                                                               max_repeats=int(t.settings.tune_max_disturbed_repeats)),
                                                              None)
            self._safety_pending = False
            self._rounds.append(record)
            if not safety_check.confirmed:
                return self._finish(outcome="not saved: memory-safety correction unverified",
                                    proposed=[safety], confirmation=safety_check)
        if chosen is None:
            self._notes.append("no candidate whose final check fits: nothing new to confirm")
            return self._safety_only(starting, safe_start, safety, power, safety_check, outcome="no change")

        setup: Setup = chosen["setup"]  # type: ignore[assignment]
        final = self._config(setup)
        changes = self._changes(setup, planner)
        plan = self._final_plan(setup)
        selection_predicted = self._final_cost(setup)
        # Confirmation planning (9 Oct 2026): the block count of this one check is
        # chosen now, before it starts, and never changed once it has.
        plan, confirmation_plan = self._plan_confirmation(setup, plan)
        sizing = self._sizing_record(setup, plan)  # exactly what this check uses, recorded now
        for question, entry in sizing.items():
            entry["selection_blocks"] = confirmation_plan["questions"][question]["selection_blocks"]
        predicted = float(confirmation_plan["predicted_check_s"])
        self._confirmation_plan = confirmation_plan | {"selection_estimate_s": round(selection_predicted, 1)}
        looks = {self._workload(q).workload_id: {"min_blocks": n, "max_blocks": n}  # type: ignore[union-attr]
                 for q, (_, n) in plan.items()}
        start_at = self._now()
        if time.perf_counter() + predicted > self._target:
            over = time.perf_counter() + predicted - self._target
            self.extensions.append(
                f"final check started at {start_at:.0f} s with a predicted {predicted:.0f} s from measured "
                f"costs ({', '.join(f'{q}: {n} blocks x 2 at {self._cost(q) or 0:.0f} s' for q, (_, n) in plan.items())}); "
                f"projected to finish {over:.0f} s after the target")
        t.say("Confirming once (" + ", ".join(f"{q}: {m.replace('_', ' ')}, {n} blocks" for q, (m, n) in plan.items())
              + f"; about {predicted / 60:.0f} min)")
        t._final_looks = looks
        t._whole_setup_fairs = []
        stuck_before = len(self.stuck)
        began = time.perf_counter()
        self._final = {
            "planned_blocks": {q: n for q, (_, n) in plan.items()},
            "cost": {q: (self._cost(q) or 0.0) for q in ("short", "long")},
            "ids": {self._workload(q).workload_id: q for q in plan if self._workload(q) is not None},  # type: ignore[union-attr]
            "q": {}, "measurements": [], "current": None,
        }
        t._final_progress = self.final_progress
        abandoned: str | None = None
        interrupted: str | None = None
        failure: dict[str, str] | None = None
        confirmation = None
        checks: list[dict[str, object]] = []
        ok = False
        try:
            for change in changes:
                t._change_workloads[change.change_id] = self._workloads_of(change)
            ok, confirmation, checks = t._check_whole_setup(safe_start, final, changes, self._watch(), None, label="final")
        except Abandoned as stop:
            abandoned = str(stop)
            ok = False
        except Cancelled as stop:
            interrupted = f"cancelled by the user: {stop}"
            raise
        except Exception as exc:
            # Reviewer follow-up 4, item 3: an unexpected error ends the check but not
            # the report -- recorded below and returned as a failure (never approved or
            # saved), so `tune` still writes its report with the partial evidence.
            interrupted = f"stopped by an error: {type(exc).__name__}: {exc}"
            failure = {"type": type(exc).__name__, "message": str(exc),
                       "traceback": "".join(traceback.format_exception(exc))[-4000:]}
            ok = False
        except BaseException as exc:  # e.g. KeyboardInterrupt: recorded, then propagated
            interrupted = f"stopped by an error: {type(exc).__name__}: {exc}"
            raise
        finally:
            progress = self._final
            self._final = None
            t._final_looks = None
            t._final_progress = None
            fairs = list(t._whole_setup_fairs)
            for question_id, fair in t._whole_setup_fairs:
                for m in [*fair.start_runs, *fair.candidate_runs]:
                    self.ledger.add("final-side", question_id, m.elapsed_s or 0.0, started_at=self._now(), measurement=m)
            if interrupted is not None:
                # The check is being left by a user stop or an error: still record what
                # it measured (nothing of it is ever accepted).
                self._rounds.append({
                    "round": "final check", "checks": checks, "settings": self._settings_of(setup), "setup": list(setup),
                    "predicted_s": round(predicted, 1), "took_s": round(time.perf_counter() - began, 1),
                    "started_at_s": round(start_at, 1),
                    "blocks_per_question": {q: n for q, (_, n) in plan.items()},
                    "plan": {q: m for q, (m, _) in plan.items()}, "stuck": self.stuck[stuck_before:],
                    "abandoned": None, "interrupted": interrupted, "error": failure,
                    "measurements_done": len((progress or {}).get("measurements", [])),
                    "progress": {q: {k: v for k, v in st.items() if k != "costs"}
                                 for q, st in ((progress or {}).get("q") or {}).items()},
                    "calibration": self._calibration_record(sizing, fairs, progress, termination=interrupted),
                    "confirmation_plan": self._confirmation_plan,
                })
        if failure is not None:
            self._notes.append(f"the final check stopped on an unexpected error ({failure['type']}: "
                               f"{failure['message']}); nothing new saved, nothing approved")
            return self._finish(
                outcome=(f"failed: the final check stopped on an unexpected error "
                         f"({failure['type']}: {failure['message']}); nothing saved"),
                proposed=changes, confirmation=None)
        took = time.perf_counter() - began
        self.ledger.add("final", None, took, started_at=self._now())
        if time.perf_counter() > self._target:
            self.extensions.append(
                f"final check took {took:.0f} s against a predicted {predicted:.0f} s, finishing "
                f"{time.perf_counter() - self._target:.0f} s after the target")
        stuck_in_final = self.stuck[stuck_before:]
        self._rounds.append({
            "round": "final check", "checks": checks, "settings": self._settings_of(setup), "setup": list(setup),
            "predicted_s": round(predicted, 1), "took_s": round(took, 1), "started_at_s": round(start_at, 1),
            "blocks_per_question": {q: n for q, (_, n) in plan.items()},
            "plan": {q: m for q, (m, _) in plan.items()}, "stuck": stuck_in_final,
            "abandoned": abandoned, "interrupted": None, "error": None,
            "last_projection": (progress or {}).get("last_projection"),
            "measurements_done": len((progress or {}).get("measurements", [])),
            "progress": {q: {k: v for k, v in st.items() if k != "costs"}
                         for q, st in ((progress or {}).get("q") or {}).items()},
            # Calibration record (reviewer follow-up, item 5): per question, the direct
            # whole-setup reading and spread used to size N, N itself, and every block
            # score and the decision of the fair comparison that followed.
            "calibration": self._calibration_record(
                sizing, fairs, progress,
                termination="completed" if abandoned is None else f"abandoned by pacing: {abandoned}"),
            # What was planned before the check started (predictions) next to what it
            # then took (measured): the plan is never revised after the check began.
            "confirmation_plan": self._confirmation_plan | {"measured_check_s": round(took, 1)},
        })
        if abandoned is not None:
            self._notes.append(f"final check stopped incomplete: {abandoned}; nothing new saved (no retry)")
            return self._safety_only(starting, safe_start, safety, power, safety_check,
                                     outcome="not confirmed: check stopped incomplete (too slow to finish)",
                                     proposed=changes, confirmation=None)
        assert confirmation is not None
        if stuck_in_final and ok:
            ok = False
            self._notes.append("an answer of the final check appeared stuck and was stopped; not accepted")
        if not ok:
            self._notes.append("the candidate did not pass its final check; nothing new saved (no retry)")
            return self._safety_only(starting, safe_start, safety, power, safety_check,
                                     outcome="not confirmed", proposed=changes, confirmation=confirmation)

        everything = ([safety] if safety else []) + changes
        decisions: list[ChangeDecision] = []
        for change in everything:
            self.check_user()
            decisions.append(t.approver.decide(change, timeout_s=float(t.settings.tune_approval_timeout_s)))
            self.check_user()
        if not all(item.approved for item in decisions):
            self._notes.append("not every change was approved; the evidence is for the whole set, so nothing "
                               "new was saved (no automatic re-check of a subset)")
            return self._finish(outcome="not saved: partial approval", proposed=changes,
                                confirmation=confirmation, decisions=decisions)
        gain = confirmation.judgement.goal.gain if confirmation.judgement else None
        saved = self._publish(starting=starting, final=final, approved=everything, declined=[],
                              gain=gain, power=power, notes=["experimental practical search"])
        t.say(f"Saved for {power_words(power)}: {final}.")
        return self._finish(outcome="saved", proposed=changes, confirmation=confirmation,
                            decisions=decisions, saved=saved, final=final)

    # ------------------------------------------------------------------ #
    # Exploration                                                          #
    # ------------------------------------------------------------------ #
    def _explore(self, planner: Planner) -> str:
        """Run the planner until it is done or the next step does not fit the target.
        Returns why it stopped early ("" when the plan finished)."""
        from inferenceos.tune.coordinator import SearchTrace, coordinate
        from inferenceos.tune.practical_policy import PracticalPolicy

        policy = PracticalPolicy(self, planner)
        trace = self.tuner._search_trace = SearchTrace(policy.mode)
        try:
            coordinate(policy, trace)
        except BaseException:
            try:
                self._rounds.append(trace.failure_round())
            except BaseException:
                # Even allocating/appending diagnostics must not replace the cause.
                pass
            raise
        else:
            self._rounds.append(trace.round())
        return policy.stop_reason

    def _screen(self, planner: Planner, step: Step, value: int) -> str:
        workload = self._workload(step.question)
        assert workload is not None
        ref_setup, cand_setup = step.base, step.setup_for(value)
        missing = [x for x in (ref_setup, cand_setup) if (x, workload.workload_id) not in self._meas]
        need = (self._cost(step.question) or 0.0) * len(missing)
        prospective = list(planner.best_so_far())
        prospective[SLOT[step.category]] = value
        reserve = self._final_cost((prospective[0], prospective[1], prospective[2]), planning=True)
        if time.perf_counter() + need + reserve + self._save_reserve > self._target:
            return (f"target: measuring {step.category} {value} ({need:.0f} s) would push the final check "
                    f"(about {reserve:.0f} s) past the target")
        ref = self._measure(ref_setup, workload, f"{step.category}-reference")
        if (cand_setup, workload.workload_id) not in self._meas:
            cost = self._cost(step.question) or 0.0
            if time.perf_counter() + cost > self._hard:
                return "the extension would be exceeded"
            if time.perf_counter() + cost > self._target:
                self.extensions.append(
                    f"finished the started pair for {step.category} {value} past the target "
                    f"(measured cost {cost:.0f} s)")
        cand = self._measure(cand_setup, workload, f"{step.category}-{value}")
        gain, noise = self._gain(ref, cand, workload)
        planner.record(step.category, value, gain, noise, cand_setup)
        self._record_screening("screen", planner, step.category, value, workload, ref_setup, cand_setup,
                               ref, cand, gain, noise, step=len(self.steps))
        self.timeline.append({"t": round(self._now(), 1), "candidate": self._settings_of(planner.best_so_far()),
                              "measured": f"{step.category} {value}"})
        self.tuner.say(f"  {step.category} {value}: " + ("not measured" if gain is None else f"{gain:+.1%} (noise {noise:.1%})"))
        return ""

    def _size_long(self, planner: Planner, *, reopen: bool = False) -> bool:
        t = self.tuner
        prospective = list(planner.best_so_far())
        prospective[SLOT[BATCH]] = max(planner.batch_candidates() or [planner.current_batch * 2])
        long_cost = self._cost("long") or 0.0
        need = 2.0 * long_cost  # sizing read + the start's long reading
        # The batch's own screening (reference + 2 sizes) must fit as well.
        need += 3.0 * long_cost
        reserve = self._final_cost((prospective[0], prospective[1], prospective[2]), planning=True)
        if time.perf_counter() + need + reserve + self._save_reserve > self._target:
            if not reopen:
                t.say("  the prompt batch does not fit for now (its long-question check is too costly)")
                planner.log.append(
                    f"prompt batch left out for now: screening it ({need:.0f} s) and its check "
                    f"(about {reserve:.0f} s) do not fit the target")
                planner.set_question_length(None)
                self._batch_left_out = True
            return False
        assert self._safe_start is not None
        at = self._safe_start.model_copy(update={"gpu_offload_layers": planner.chosen_placement()})
        began, began_at = time.perf_counter(), self._now()
        long_q = t._long_question(at, self._watch(), None)
        self.ledger.add("size-long", "size-long", time.perf_counter() - began, started_at=self._now())
        self.operations.append({
            "n": len(self.operations) + 1, "kind": "size-long", "label": "size the long question",
            "began_s": round(began_at, 1), "wall_s": round(time.perf_counter() - began, 2),
            "settings": {"gpu_offload_layers": at.gpu_offload_layers, "threads": at.threads,
                         "ubatch_size": at.ubatch_size},
            "question": None if long_q is None else long_q.workload_id,
            "tokens": t._long_question_tokens})
        self._long = long_q
        if self._measure(self._start_setup, long_q, "start-long") is None:
            planner.log.append("the start could not be measured on the long question: prompt batch not explored")
            if not reopen:
                planner.set_question_length(None)
            self._long = None
            return False
        if reopen:
            return planner.reopen_batch(t._long_question_tokens)
        planner.set_question_length(t._long_question_tokens)
        return True

    def _repeat_weak(self, planner: Planner) -> str:
        """Once the plan is exhausted: measure the best weak reading again, if the repeat
        and the final check it could lead to fit the target."""
        if len(self._weak_tried) >= int(self.tuner.settings.tune_experimental_weak_repeats):
            return "none"
        candidates = [r for r in planner.weak_candidates()
                      if (r.category, r.value, r.context) not in self._weak_tried]
        if not candidates:
            return "none"
        reading = candidates[0]
        workload = self._workload(QUESTION[reading.category])
        if workload is None:
            return "none"
        self._weak_tried.add((reading.category, reading.value, reading.context))
        ref = list(reading.setup)
        ref[SLOT[reading.category]] = planner.reference(reading.category)
        ref_setup: Setup = (ref[0], ref[1], ref[2])
        need = 2.0 * (self._cost(QUESTION[reading.category]) or 0.0)
        reserve = self._final_cost(reading.setup, planning=True, assumed={QUESTION[reading.category]: reading.gain or 0.0})
        if time.perf_counter() + need + reserve + self._save_reserve > self._target:
            planner.log.append(f"weak {reading.category} {reading.value} ({reading.gain:+.1%}) not repeated: "
                               f"repeat {need:.0f} s + its check {reserve:.0f} s do not fit the target")
            return "none"
        planner.log.append(f"repeat the weak {reading.category} {reading.value} ({reading.gain:+.1%}) in setup {reading.setup}")
        r = self._measure(ref_setup, workload, "weak-repeat-reference", fresh=True)
        c = self._measure(reading.setup, workload, f"weak-repeat-{reading.category}-{reading.value}", fresh=True)
        gain, noise = self._gain(r, c, workload)
        planner.record(reading.category, reading.value, gain, noise, reading.setup)
        self._record_screening("weak repeat", planner, reading.category, reading.value, workload, ref_setup,
                               reading.setup, r, c, gain, noise, step=None)
        self.timeline.append({"t": round(self._now(), 1), "candidate": self._settings_of(planner.best_so_far()),
                              "measured": f"repeat {reading.category} {reading.value}"})
        return "repeated"

    # ------------------------------------------------------------------ #
    # Candidates for the final check                                       #
    # ------------------------------------------------------------------ #
    def _complete_readings(self, setup: Setup, *, final_only: bool, limit: float) -> None:
        if setup == self._start_setup:
            return
        questions = list(self._final_plan(setup, planning=True)) if final_only else ["short", "long"]
        for question in questions:
            workload = self._workload(question)
            if workload is None or (setup, workload.workload_id) in self._meas:
                continue
            if (self._start_setup, workload.workload_id) not in self._meas:
                continue
            cost = self._cost(question) or 0.0
            check = self._final_cost(setup, planning=True)
            if time.perf_counter() + cost + check + self._save_reserve > limit:
                continue
            if time.perf_counter() + cost > self._target:
                self.extensions.append(
                    f"direct reading of {self._settings_of(setup)} on the {question} question past the target "
                    f"(measured cost {cost:.0f} s)")
            self._measure(setup, workload, "direct reading")

    def _left(self) -> float:
        """Seconds a final check may still take: target + extension, minus the save
        allowance (and a pending safety check)."""
        left = self._hard - time.perf_counter() - self._save_reserve
        if self._safety_pending:
            left -= (self._cost("short") or 0.0) + self._load_try_s
        return left

    def _judge(self, setup: Setup, direct: dict[str, float | None],
               plan: dict[str, tuple[str, int]]) -> tuple[bool, str, list[str]]:
        """Whether ``setup`` may be final-checked on its own readings: (eligible, why,
        questions missing a reading)."""
        missing = [q for q in plan if direct.get(q) is None]
        if missing:
            return False, "missing evidence: not measured as a whole on " + ", ".join(missing), missing
        if any(mode == "better" and (direct[q] or 0.0) <= 0 for q, (mode, _) in plan.items()):
            return False, "its direct reading against the start is not a gain", missing
        if plan.get("short", ("", 0))[0] == "not_worse" and (direct["short"] or 0.0) < -max(
                GUARD_SCREEN_TOLERANCE, 2.0 * self._spread.get("short", 0.0)):
            return False, f"loses {direct['short']:+.1%} on the short question it must not lose on", missing
        return True, "", missing

    def _measured_setups(self) -> list[dict[str, Any]]:
        """Every exact setup measured in this tune (with its own context), with its
        readings against the start, whether it is promising, eligible and affordable,
        and the predicted cost of its check. Nothing measured is dropped from here."""
        smallest = float(self.tuner.rules.effort.minimum_improvement)
        left = self._left()
        rows: list[dict[str, Any]] = []
        for setup in sorted({key[0] for key in self._meas}):
            if setup == self._start_setup:
                continue
            direct = {q: self._vs_start(setup, q) for q in ("short", "long")}
            known = [d for d in direct.values() if d is not None]
            # Promising: on some question it reads at least the smallest worthwhile gain
            # plus half a noise width of one reading (the same bar as a weak repeat).
            promising = any(
                d is not None and d >= smallest + WEAK_MARGIN * self._spread.get(q, 0.0)
                for q, d in direct.items()
            )
            plan = self._final_plan(setup, planning=True)
            eligible, why, missing = self._judge(setup, direct, plan)
            predicted = self._final_cost(setup, planning=True)
            # Questions it has no reading on (its check may not need them all, but its
            # value counts an unread question as no gain); completing one costs one
            # measurement of this setup (the start's reading is reused).
            unread = [q for q in ("short", "long") if direct[q] is None and self._workload(q) is not None
                      and (self._start_setup, self._workload(q).workload_id) in self._meas]  # type: ignore[union-attr]
            completion = sum(self._cost(q) or 0.0 for q in unread)
            rows.append({
                "setup": setup, "direct": direct, "promising": bool(known and promising),
                "value": round(0.5 * sum(known), 4), "eligible": eligible, "why": why,
                "missing": missing, "unread": unread, "predicted_s": round(predicted, 1),
                "affordable": predicted <= left, "completion_affordable": predicted + completion <= left,
            })
        return rows

    def _history(self) -> list[Setup]:
        """Earlier complete setups worth keeping as candidates, at most ``HISTORY_SIZE``.

        Review 10: eligibility and affordability are applied *before* the list is cut,
        so expensive or ineligible setups cannot crowd out an affordable earlier winner:
        promising setups that are eligible and whose check fits come first (most valuable
        first), then the rest. Readings are kept, not re-explored."""
        rows = [r for r in self._measured_setups() if r["promising"]]
        rows.sort(key=lambda r: (bool(r["eligible"] and r["affordable"]), float(r["value"])), reverse=True)
        return [r["setup"] for r in rows[:HISTORY_SIZE]]

    def _to_complete(self) -> list[Setup]:
        """At most two promising setups that lack a reading and whose completion plus
        check would still fit, most valuable first (eligibility and affordability before
        the cut, review 10)."""
        rows = [r for r in self._measured_setups()
                if r["promising"] and r["unread"] and r["completion_affordable"]]
        rows.sort(key=lambda r: float(r["value"]), reverse=True)
        return [r["setup"] for r in rows[:2]]

    def _measured_status(self, chosen: dict[str, object] | None) -> list[dict[str, object]]:
        """What became of every measured setup, so none disappears silently."""
        left = self._left()
        out: list[dict[str, object]] = []
        for r in self._measured_setups():
            if chosen is not None and chosen["setup"] == r["setup"]:
                status = "chosen"
            elif r["missing"]:
                status = "missing evidence"
            elif not r["eligible"]:
                status = "not eligible: " + r["why"]
            elif float(r["predicted_s"]) > left:
                status = "cannot reasonably finish confirmation"
            elif not r["promising"]:
                status = "not selected (not promising)"
            else:
                status = "not selected"
            out.append({"settings": self._settings_of(r["setup"]), "setup": list(r["setup"]),
                        "direct": r["direct"],
                        "predicted_s": r["predicted_s"], "status": status})
        return out

    def _options(self, planner: Planner) -> list[dict[str, object]]:
        """Candidates for the final check, each a setup measured as a whole: the best-so-far
        setup and its subsets, earlier promising complete setups (``_history``), and
        repeated weak setups; each with its direct readings against the start and the
        predicted cost of its check."""
        best = planner.best_so_far()
        changed = [c for c in CATEGORIES if best[SLOT[c]] != self._start_setup[SLOT[c]]]
        setups: list[Setup] = []
        for size in range(len(changed), 0, -1):
            for subset in itertools.combinations(changed, size):
                out = list(self._start_setup)
                for c in subset:
                    out[SLOT[c]] = best[SLOT[c]]
                setups.append((out[0], out[1], out[2]))
        # Weak setups measured again stand on their own only when the averaged reading
        # is at least the smallest worthwhile gain plus half an (averaged) noise width
        # (``WEAK_MARGIN``): otherwise a long final check would most likely be spent on
        # noise. Simulation (review 8): one width lost most interior gains (Qwen-like);
        # zero sent pure noise to 16-block checks (Gemma-like).
        smallest = float(self.tuner.rules.effort.minimum_improvement)
        source: dict[Setup, str] = {setups[0]: "best so far"} if setups else {}
        for setup in setups[1:]:
            source.setdefault(setup, "subset of the best")
        for setup in self._history():
            if setup not in source:
                setups.append(setup)
                source[setup] = "earlier complete setup"
        core = set(setups)
        weak_only: dict[Setup, str] = {}
        for c in CATEGORIES:
            for r in planner.readings[c]:
                if r.count > 1 and r.gain is not None and r.gain > 0:
                    setups.append(r.setup)
                    if r.gain < smallest + WEAK_MARGIN * r.noise:
                        weak_only[r.setup] = (f"repeated reading {r.gain:+.1%} is below the smallest worthwhile "
                                              f"gain ({smallest:.1%}) plus {WEAK_MARGIN:g} x its noise ({r.noise:.1%})")
        out_list: list[dict[str, object]] = []
        for setup in dict.fromkeys(setups):
            if setup == self._start_setup:
                continue
            plan = self._final_plan(setup)
            direct = {q: self._vs_start(setup, q) for q in ("short", "long")}
            entry: dict[str, object] = {"setup": setup, "source": source.get(setup, "repeated weak setup"),
                                        "direct": direct, "plan": {q: list(v) for q, v in plan.items()}}
            eligible, why, _missing = self._judge(setup, direct, plan)
            if setup in weak_only and setup not in core:
                entry["eligible"] = False
                entry["why"] = weak_only[setup]
            elif not eligible:
                entry["eligible"] = False
                entry["why"] = why
            else:
                entry["eligible"] = True
            entry["value"] = round(0.5 * sum(v for v in direct.values() if v is not None), 4)
            entry["predicted_s"] = round(self._final_cost(setup), 1)
            out_list.append(entry)
        return sorted(out_list, key=lambda e: (bool(e["eligible"]), float(e["value"])), reverse=True)  # type: ignore[arg-type]

    def _choose(self, options: list[dict[str, object]]) -> dict[str, object] | None:
        """The most valuable eligible candidate whose check fits target + extension."""
        left = self._left()
        fitting = [o for o in options if o["eligible"] and float(o["predicted_s"]) <= left]  # type: ignore[arg-type]
        for o in options:
            if o["eligible"] and o not in fitting:
                o["why"] = (f"cannot reasonably finish confirmation: its check ({o['predicted_s']} s) does not "
                            f"fit the {max(left, 0):.0f} s left")
        if not fitting:
            return None
        fitting.sort(key=lambda o: (float(o["value"]), -float(o["predicted_s"])), reverse=True)  # type: ignore[arg-type]
        return fitting[0]

    def _final_plan(self, setup: Setup, *, planning: bool = False,
                    assumed: dict[str, float] | None = None) -> dict[str, tuple[str, int]]:
        """Questions, modes and blocks of the final check of ``setup`` against the start,
        sized from the setup's own direct readings against the start (not component
        sums). ``planning``: before the setup was measured on a question, plan the fewest
        blocks for it (``assumed`` gives a reading to plan with, e.g. a weak repeat)."""
        changed = {c for c in CATEGORIES if setup[SLOT[c]] != self._start_setup[SLOT[c]]}
        plan: dict[str, tuple[str, int]] = {}
        # Declared (C3): "better" on every question a change is judged on, "not worse" on
        # every question a change is only guarded on.
        for question, mode in TABLES.final_questions(changed).items():
            if mode == "better":
                plan[question] = ("better", self._blocks(question, self._effect(setup, question, planning, assumed)))
            else:
                plan[question] = ("not_worse", self._blocks(question, float(self.tuner.settings.tune_guard_margin)))
        return plan

    def _effect(self, setup: Setup, question: str, planning: bool, assumed: dict[str, float] | None) -> float:
        smallest = float(self.tuner.rules.effort.minimum_improvement)
        direct = self._vs_start(setup, question)
        if direct is None and assumed and question in assumed:
            direct = assumed[question]
        if direct is None:
            return 1.0 if planning else 0.0  # unread: fewest blocks while planning
        return max(WINNER_SHRINK * direct, smallest) if direct > 0 else 0.0

    def _blocks(self, question: str, effect: float) -> int:
        low = max(int(self.tuner.settings.tune_experimental_final_blocks), 2)
        spread = self._spread.get(question) or math.sqrt(2.0) * float(self.tuner.settings.tune_between_load_spread)
        if effect <= 0:
            return 16
        return int(min(max(math.ceil((Z_POWER * SPREAD_MARGIN * spread / effect) ** 2), low), 16))

    # ------------------------------------------------------------------ #
    # Confirmation planning (9 Oct 2026)                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def planned_power(blocks: int, effect: float, spread: float, alpha: float) -> float:
        """Approximate chance that a one-look fixed-count check of ``blocks`` blocks
        shows its bound beyond zero when the true effect is ``effect`` and the
        block-to-block spread is ``spread``: the Student t critical value of the actual
        test (one-sided ``alpha``, ``blocks - 1`` degrees of freedom) with the usual normal
        approximation of the non-central t. A planning number, never a decision."""
        if blocks < 2 or spread <= 0:
            return 0.0
        if effect <= 0:
            return 0.0
        critical = t_quantile(1.0 - alpha, blocks - 1)
        shift = effect * math.sqrt(blocks) / spread
        z = (shift - critical) / math.sqrt(1.0 + critical * critical / (2.0 * (blocks - 1)))
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _side_cost(self, config: RuntimeConfiguration, workload: BenchmarkWorkload) -> tuple[float | None, int]:
        """Median measured wall time of one fresh-load measurement of exactly ``config``
        on ``workload`` in this tune (load, warm-up, monitored answers, unload), and how
        many measurements it is from."""
        want = {"gpu_offload_layers": config.gpu_offload_layers, "threads": config.threads,
                "ubatch_size": config.ubatch_size}
        walls = sorted(float(op["wall_s"]) for op in self.operations  # type: ignore[arg-type]
                       if op.get("kind") == "measure" and op.get("question") == workload.workload_id
                       and op.get("error") is None and op.get("settings") == want)
        if not walls:
            return None, 0
        mid = len(walls) // 2
        return (walls[mid] if len(walls) % 2 else 0.5 * (walls[mid - 1] + walls[mid])), len(walls)

    def _plan_confirmation(self, setup: Setup, plan: dict[str, tuple[str, int]]
                           ) -> tuple[dict[str, tuple[str, int]], dict[str, Any]]:
        """Choose the final check's block count per question before it starts.

        ``plan`` is the selection-time plan (``_final_plan``; the candidate was chosen
        and judged affordable with it). For each question the count needed for
        ``PLAN_POWER`` is computed under a planning spread = screening noise width x
        ``NOISE_ALLOWANCE_*`` (by whether the load-to-load term was learned or the
        fallback), with the planned effect the selection used. The larger count is used
        only if the whole check -- both sides of every block, the re-attempt reserve,
        every required question, then saving and cleanup -- is predicted to finish by the
        soft target; if not, the largest count between the two that does; if not even
        the selection count does, the selection count is kept (the existing extension
        behaviour). Strong gains need no more than the minimum, so they stay cheap.

        Everything here is a prediction from this tune's measurements; the record says
        which numbers were measured and which are predicted, and why the count was
        chosen. Nothing about the test, its alpha, look or acceptance rule changes, and
        no block is ever added after the check has started."""
        t = self.tuner
        s = t.settings
        alpha = float(s.tune_fair_alpha)
        low = max(int(s.tune_experimental_final_blocks), 2)
        extra_attempts = max(int(s.tune_fair_extra_attempts), 0)
        fallback_spread = math.sqrt(2.0) * float(s.tune_between_load_spread)
        start_config = self._config(self._start_setup)
        final_config = self._config(setup)
        now = time.perf_counter()
        questions: dict[str, dict[str, Any]] = {}
        for question, (mode, nominal) in plan.items():
            workload = self._workload(question)
            assert workload is not None
            direct = self._vs_start(setup, question)
            if mode == "not_worse":
                effect = float(s.tune_guard_margin)
                effect_basis = "the guard's margin (a 'not worse' check must show a loss smaller than it)"
            else:
                effect = self._effect(setup, question, False, None)
                effect_basis = (f"{WINNER_SHRINK} x this setup's direct reading against the start, at least the "
                                f"smallest worthwhile gain ({float(t.rules.effort.minimum_improvement):.1%})")
            width = self._spread.get(question)
            info = dict(self._spread_info.get(question) or {})
            if width:
                learned = info.get("between_source") == "learned"
                spread_source = ("largest screening noise width on this question: sqrt(2 x (load-to-load^2 + "
                                 "run-to-run^2 / runs)), load-to-load " + ("learned for this model and question"
                                                                            if learned else "the fallback setting"))
            else:
                width, learned = fallback_spread, False
                spread_source = "fallback: sqrt(2) x tune_between_load_spread (no screening width on this question)"
            allowance = NOISE_ALLOWANCE_LEARNED if learned else NOISE_ALLOWANCE_FALLBACK
            planning_spread = width * allowance
            # The smallest count reaching the planned power, if any count up to the cap
            # does. When none does, the cap is the count aimed for (as before), but it
            # is never reported as reaching the target (reviewer, 9 Oct 2026).
            needed: int | None = next((n for n in range(low, MAX_FINAL_BLOCKS + 1)
                                       if self.planned_power(n, effect, planning_spread, alpha) >= PLAN_POWER), None)
            aim = needed if needed is not None else MAX_FINAL_BLOCKS
            start_cost, start_n = self._side_cost(start_config, workload)
            cand_cost, cand_n = self._side_cost(final_config, workload)
            question_cost = self._cost(question) or 0.0
            per_block = (start_cost if start_cost is not None else question_cost) + (
                cand_cost if cand_cost is not None else question_cost)
            questions[question] = {
                "workload": workload.workload_id, "mode": mode,
                "measured": {
                    "direct_reading_vs_start": None if direct is None else round(direct, 6),
                    "screening_noise_width": round(width, 6),
                    "noise_inputs": info or None,
                    "start_side_measurement_s": None if start_cost is None else round(start_cost, 2),
                    "start_side_from_measurements": start_n,
                    "candidate_side_measurement_s": None if cand_cost is None else round(cand_cost, 2),
                    "candidate_side_from_measurements": cand_n,
                    "question_median_measurement_s": round(question_cost, 2),
                },
                "predicted": {
                    "effect": round(effect, 6), "effect_basis": effect_basis,
                    "noise_source": spread_source,
                    "uncertainty_allowance": allowance,
                    "allowance_basis": ("learned load-to-load spread: the existing margin (no calibration of learned "
                                        "spreads yet)" if learned else
                                        "fallback load-to-load spread: about the 80th percentile of realised / "
                                        "screening spread in the fresh final checks of jobs 090-107"),
                    "planning_spread": round(planning_spread, 6),
                    "blocks_for_power": needed, "power_target": PLAN_POWER,
                    "power_target_attainable": needed is not None,
                    "block_cap": MAX_FINAL_BLOCKS,
                    "power_at_cap": round(self.planned_power(MAX_FINAL_BLOCKS, effect, planning_spread, alpha), 3),
                    "per_block_s": round(per_block, 1),
                    "side_costs_basis": ("measured medians of these exact setups on this question"
                                         if start_cost is not None and cand_cost is not None else
                                         "measured median of this question (a side was not measured on it)"),
                },
                "selection_blocks": nominal, "_per_block": per_block, "_effect": effect,
                "_planning_spread": planning_spread, "_needed": max(aim, nominal),
            }

        per_block_of = {name: float(q["_per_block"]) for name, q in questions.items()}

        def check_s(counts: dict[str, int]) -> float:
            # Both sides of every block plus the re-attempt reserve (one block per
            # question, as the selection estimate allows), every required question.
            total = sum(per_block_of[name] * (counts[name] + 1) for name in questions)
            if self._safety_pending:
                total += (self._cost("short") or 0.0) + self._load_try_s
            return total

        def finish(counts: dict[str, int]) -> float:
            return now + check_s(counts) + self._save_reserve

        selection = {name: int(q["selection_blocks"]) for name, q in questions.items()}
        counts = {name: int(q["_needed"]) for name, q in questions.items()}
        # Step down towards the selection counts, largest extra first, until it fits the
        # soft target (the extension stays for slower-than-predicted work).
        while finish(counts) > self._target and counts != selection:
            name = max((n for n in counts if counts[n] > selection[n]),
                       key=lambda n: (counts[n] - selection[n]) * per_block_of[n])
            counts[name] -= 1
        # Per question: did the chosen count reach the planned power, and if not, what
        # stopped it -- the soft target (time) or the block cap (no count up to the cap
        # reaches the target)?
        limits: dict[str, str] = {}
        for name, q in questions.items():
            chosen_power = self.planned_power(counts[name], q["_effect"], q["_planning_spread"], alpha)
            attainable = bool(q["predicted"]["power_target_attainable"])
            if chosen_power >= PLAN_POWER:
                limits[name] = "planned power reached"
            elif counts[name] < int(q["_needed"]):
                limits[name] = "limited by the soft target" + (
                    "" if attainable else f"; not attainable even at the {MAX_FINAL_BLOCKS}-block cap")
            else:
                limits[name] = f"planned power not attainable within the {MAX_FINAL_BLOCKS}-block cap"
            q["planned_power_limited_by"] = ("none" if limits[name] == "planned power reached" else
                                             "soft target" if limits[name].startswith("limited") else "block cap")
        detail = "; ".join(
            f"{name}: {limits[name]} ({self.planned_power(counts[name], q['_effect'], q['_planning_spread'], alpha):.2f} "
            f"planned power at {counts[name]} blocks)" for name, q in questions.items())
        if counts == selection:
            if all(v == "planned power reached" for v in limits.values()):
                reason = "selection count kept: it already gives the planned power under the planning spread"
            elif any(v.startswith("limited") for v in limits.values()):
                reason = ("selection count kept: more blocks were needed for the planned power under the planning "
                          "spread, but none fit the soft target")
            else:
                reason = (f"selection count kept at the {MAX_FINAL_BLOCKS}-block cap: the planned power is not "
                          "attainable within the cap")
        else:
            reason = ("more blocks than the selection count: the planned effect is small or uncertain against the "
                      "planning spread, and the larger check is predicted to finish by the soft target")
        reason += f" [{detail}]"
        if finish(selection) > self._target and counts == selection:
            reason += "; the check is predicted to need the extension (existing behaviour)"
        for name, q in questions.items():
            q["chosen_blocks"] = counts[name]
            q["predicted"]["power_at_chosen"] = round(self.planned_power(counts[name], q["_effect"],
                                                                         q["_planning_spread"], alpha), 3)
            q["predicted"]["power_at_selection_count"] = round(self.planned_power(
                selection[name], q["_effect"], q["_planning_spread"], alpha), 3)
            q["predicted"]["check_s"] = round(q["_per_block"] * (counts[name] + 1), 1)
            for key in ("_per_block", "_effect", "_planning_spread", "_needed"):
                q.pop(key)
        new_plan = {name: (mode, counts[name]) for name, (mode, _) in plan.items()}
        record = {
            "format_version": PLAN_FORMAT_VERSION,
            "status": "experimental / provisional",
            "rule": ("fixed block count chosen before the check: the count for the planned power under the "
                     "planning spread (the 16-block cap when no count up to it reaches that power) if the check "
                     "finishes by the soft target, else the largest count that does, never fewer than the "
                     "selection count; one look, the existing test and alpha"),
            "selection_limitation": ("candidate selection and its affordability still use the selection-time "
                                     "estimate (question-wide median cost, selection count); exact per-side costs "
                                     "and this plan apply only to the chosen setup's check"),
            "questions": questions,
            "alpha_one_sided": alpha, "re_attempt_reserve_blocks_per_question": 1,
            "max_attempts_per_question": {name: counts[name] + extra_attempts for name in counts},
            "predicted_check_s": round(check_s(counts), 1),
            "predicted_check_s_at_selection_counts": round(check_s(selection), 1),
            "save_and_cleanup_allowance_s": round(self._save_reserve, 1),
            "safety_check_included": bool(self._safety_pending),
            "planned_at_s": round(self._now(), 1),
            "soft_target_s": round(self._target - self._t0, 1),
            "hard_limit_s": round(self._hard - self._t0, 1),
            "available_to_target_s": round(self._target - now - self._save_reserve, 1),
            "predicted_finish_s": round(finish(counts) - self._t0, 1),
            "reason": reason,
            "measured_vs_predicted": ("'measured' holds readings and costs from this tune; 'predicted' holds the "
                                      "planning effect, spread, power and time, which the check itself does not use"),
        }
        return new_plan, record

    def _sizing_record(self, setup: Setup, plan: dict[str, tuple[str, int]]) -> dict[str, dict[str, Any]]:
        """What sized the final check, per question, at selection time."""
        out: dict[str, dict[str, Any]] = {}
        fallback = math.sqrt(2.0) * float(self.tuner.settings.tune_between_load_spread)
        smallest = float(self.tuner.rules.effort.minimum_improvement)
        for question, (mode, blocks) in plan.items():
            workload = self._workload(question)
            direct = self._vs_start(setup, question)
            spread = self._spread.get(question)
            effect = (float(self.tuner.settings.tune_guard_margin) if mode == "not_worse"
                      else self._effect(setup, question, False, None))
            out[question] = {
                "workload": workload.workload_id if workload is not None else None,
                "mode": mode,
                "direct_whole_setup_reading": None if direct is None else round(direct, 6),
                "spread_used": round(spread if spread else fallback, 6),
                "spread_source": "largest screening noise width on this question" if spread else
                                 "fallback sqrt(2) x tune_between_load_spread",
                "effect_used": round(effect, 6),
                "chosen_blocks": blocks,
                "smallest_gain": smallest,
                "constants": {"winner_shrink": WINNER_SHRINK, "spread_margin": SPREAD_MARGIN,
                              "z_power": round(Z_POWER, 4), "status": "experimental / provisional"},
            }
        return out

    @staticmethod
    def _calibration_record(sizing: dict[str, dict[str, Any]], fairs: list[tuple[str, Any]],
                            progress: dict[str, Any] | None = None, *,
                            termination: str = "completed") -> dict[str, dict[str, Any]]:
        """Per question: the sizing (``_sizing_record``) plus what the final comparison
        produced, with an explicit ``status``:

        * ``completed``: the comparison returned (its blocks and decision are recorded;
          ``stopped_early`` says why if it ran out of attempts or time on its own);
        * ``incomplete, usable blocks``: stopped (paced out, cancelled, error) after at
          least one counted block -- the counted block scores are kept;
        * ``started, no usable block``: stopped after it began but before any block
          counted;
        * ``not started``: never reached.

        ``termination`` is the explicit reason the final check ended. Evidence from an
        incomplete comparison is recorded for calibration only and is never eligible
        for acceptance (``eligible_for_acceptance``: False); keeping it avoids biasing
        later calibration towards checks that happened to finish."""
        out = {q: dict(rec) for q, rec in sizing.items()}
        by_id = {rec.get("workload"): q for q, rec in sizing.items()}
        states: dict[str, Any] = (progress or {}).get("q") or {}
        measured: dict[str, int] = {}
        for item in (progress or {}).get("measurements", []) or []:
            measured[item["question"]] = measured.get(item["question"], 0) + 1
        completed: set[str] = set()
        for question_id, fair in fairs:
            q = by_id.get(question_id, question_id)
            completed.add(q)
            entry = out.setdefault(q, {"workload": question_id})
            entry.update({
                "status": "completed",
                "final_block_scores": [round(float(x), 6) for x in fair.block_scores],
                "final_decision": fair.decision,
                "final_attempts": fair.attempts,
                "final_discarded_blocks": len(fair.discarded_blocks),
                "stopped_early": fair.stopped_early,
                "termination": "comparison completed",
            })
        for q, entry in out.items():
            if q in completed:
                continue
            state = states.get(q)
            if state is None:
                entry.update({
                    "status": "not started", "final_block_scores": None, "final_decision": None,
                    "final_attempts": 0, "final_discarded_blocks": 0, "side_measurements": 0,
                    "termination": "not reached" if termination == "completed" else termination,
                    "eligible_for_acceptance": False,
                })
                continue
            scores = [round(float(x), 6) for x in state.get("block_scores", [])]
            entry.update({
                "status": "incomplete, usable blocks" if scores else "started, no usable block",
                "final_block_scores": scores,
                "final_decision": None,
                "final_attempts": int(state.get("attempts", 0)),
                "final_discarded_blocks": int(state.get("discarded", 0)),
                "discarded_why": list(state.get("discarded_why", [])),
                "side_measurements": measured.get(q, 0),
                "termination": termination if termination != "completed" else "comparison did not return",
                "eligible_for_acceptance": False,
            })
        return out

    def _final_cost(self, setup: Setup, *, planning: bool = False, assumed: dict[str, float] | None = None) -> float:
        """Predicted seconds of ``setup``'s final check: two sides x (blocks + 1
        re-attempt) x one measurement, per question, from measured costs."""
        if setup == self._start_setup:
            return 0.0
        total = 0.0
        for question, (_, blocks) in self._final_plan(setup, planning=planning, assumed=assumed).items():
            total += 2 * (blocks + 1) * (self._cost(question) or 0.0)
        if self._safety_pending:
            total += (self._cost("short") or 0.0) + self._load_try_s
        return total

    # ------------------------------------------------------------------ #
    # Measurements                                                         #
    # ------------------------------------------------------------------ #
    def _workload(self, question: str) -> BenchmarkWorkload | None:
        return self._short if question == "short" else self._long

    def _config(self, setup: Setup) -> RuntimeConfiguration:
        assert self._safe_start is not None and self._planner is not None
        p = self._planner
        layers, threads, batch = setup
        return self._safe_start.model_copy(update={
            "gpu_offload_layers": layers,
            "threads": None if (threads == p.current_threads and self._safe_start.threads is None) else threads,
            "ubatch_size": None if (batch == p.current_batch and self._safe_start.ubatch_size is None) else batch,
        })

    def _settings_of(self, setup: Setup) -> dict[str, int]:
        return {SETTING[c]: setup[SLOT[c]] for c in CATEGORIES if setup[SLOT[c]] != self._start_setup[SLOT[c]]}

    def _watch(self) -> DisturbanceWatch:
        t = self.tuner
        return DisturbanceWatch(
            logical_cores=t.hardware.cpu.logical_cores,
            limit_percent=float(t.settings.tune_disturbance_cpu_percent),
            max_repeats=int(t.settings.tune_max_disturbed_repeats),
        )

    def _measure(self, setup: Setup, workload: BenchmarkWorkload, label: str, *,
                 fresh: bool = False) -> ExperimentMeasurement | None:
        key = (setup, workload.workload_id)
        if not fresh and key in self._meas:
            return self._meas[key]
        measurement = self._measure_config(self._config(setup), workload, label)
        if measurement is not None:
            self._meas[key] = measurement
        return measurement

    def _op_of(self, measurement: ExperimentMeasurement | None) -> int | None:
        """The operation number that produced this exact measurement object."""
        return next((n for m, n in self._meas_ops if m is measurement), None) if measurement is not None else None

    def _record_screening(self, kind: str, planner: Planner, category: str, value: int,
                          workload: BenchmarkWorkload, ref_setup: Setup, cand_setup: Setup,
                          ref: ExperimentMeasurement | None, cand: ExperimentMeasurement | None,
                          gain: float | None, noise: float, *, step: int | None) -> None:
        """One immutable screening event (report only; selection is unchanged). Each side
        names the operation that produced the measurement actually compared (None when
        that side could not be measured)."""
        def side(setup: Setup, used: ExperimentMeasurement | None) -> dict[str, object]:
            return {"setup": list(setup), "settings": self._settings_of(setup),
                    "configuration": self._config(setup).model_dump(mode="json"),
                    "measurement": self._op_of(used)}
        aggregate = next((r for r in reversed(planner.readings[category])
                          if r.value == int(value) and r.setup == tuple(cand_setup)), None)
        question = "short" if workload is self._short else "long"
        self.screenings.append({
            "n": len(self.screenings) + 1, "kind": kind, "at_s": round(self._now(), 1), "step": step,
            "category": category, "value": int(value), "question": workload.workload_id,
            "reference": side(ref_setup, ref), "candidate": side(cand_setup, cand),
            "gain": gain, "noise": round(noise, 4),
            "aggregate": None if aggregate is None else {
                "gain": aggregate.gain, "noise": round(aggregate.noise, 4), "count": aggregate.count},
            "vs_start": self._vs_start(cand_setup, question),
            "vs_start_question": workload.workload_id,
        })

    def _measure_config(self, config: RuntimeConfiguration, workload: BenchmarkWorkload,
                        label: str) -> ExperimentMeasurement | None:
        t = self.tuner
        policy = t._path_policy(workload)
        began, at = time.perf_counter(), self._now()
        try:
            measurement = measure_steady(
                harness=t.harness, model=t.model, workload=workload, config=config,
                policy=policy, phase=f"tune-practical-{label}", watch=self._watch(),
            )
        except Exception as exc:  # noqa: BLE001 - a failed measurement is missing evidence
            t.say(f"    could not measure ({type(exc).__name__}: {exc})")
            self.ledger.add("failed", workload.workload_id, time.perf_counter() - began, started_at=self._now())
            self._log_measurement(label, workload, config, at, time.perf_counter() - began, None,
                                  error=f"{type(exc).__name__}: {exc}")
            return None
        wall = time.perf_counter() - began
        op = self.ledger.add("measure", workload.workload_id, wall, started_at=self._now(), measurement=measurement)
        self._log_measurement(label, workload, config, at, wall, measurement, breakdown=op.breakdown)
        self._meas_ops.append((measurement, int(self.operations[-1]["n"])))  # type: ignore[call-overload]
        if workload.workload_id not in t._runs_by_question and measurement.valid_runs:
            t._set_runs_from(workload, measurement, wall)
        # Judged by the policy it was measured with (the first measurement of a question
        # sets the runs for later ones).
        if measurement.valid_runs < policy.min_valid_samples:
            return None
        return measurement

    def _log_measurement(self, label: str, workload: BenchmarkWorkload, config: RuntimeConfiguration,
                         at: float, wall: float, measurement: ExperimentMeasurement | None, *,
                         breakdown: dict[str, object] | None = None, error: str | None = None) -> None:
        """One measurement in the search-order record (report only)."""
        keep = ("load_s", "warmup_s", "scored_s", "monitor_overhead_s", "monitor_start_s", "monitor_stop_s",
                "request_s", "after_reads_s", "between_runs_s", "unload_s", "first_word_s_total",
                "runs_with_phases", "warmups_with_phases")
        self.operations.append({
            "n": len(self.operations) + 1, "kind": "measure", "label": label,
            "question": workload.workload_id, "began_s": round(at, 1), "wall_s": round(wall, 2),
            "settings": {"gpu_offload_layers": config.gpu_offload_layers, "threads": config.threads,
                         "ubatch_size": config.ubatch_size},
            "valid_runs": None if measurement is None else measurement.valid_runs,
            "phases": {k: v for k, v in (breakdown or {}).items() if k in keep},
            "error": error,
        })

    def _gain(self, ref: ExperimentMeasurement | None, main: ExperimentMeasurement | None,
              workload: BenchmarkWorkload) -> tuple[float | None, float]:
        from inferenceos.tune.pipeline import _screen_gain, noise_variant

        t = self.tuner
        runs = t._path_policy(workload).measured_repetitions
        between = float(t.settings.tune_between_load_spread)
        between_source = "fallback"
        if ref is not None:
            saved = t._noise_store().estimate(noise_key(
                t.model.name, ref.configuration.gpu_offload_layers, workload.workload_id,
                noise_variant(ref.configuration),
            ))
            if saved is not None:
                between, between_source = saved, "learned"
        within = run_score_spread(ref, t.rules.goal.first) if ref is not None else None
        noise = math.sqrt(2.0 * (between**2 + (within or 0.0) ** 2 / max(runs, 1)))
        key = "short" if workload is self._short else "long"
        if within is not None:
            if noise >= self._spread.get(key, 0.0):
                self._spread_info[key] = {
                    "between_load_spread": round(between, 6), "between_source": between_source,
                    "within_run_spread": round(float(within), 6), "runs": int(runs),
                    "reference_runs_measured": int(ref.valid_runs) if ref is not None else None,
                }
            self._spread[key] = max(self._spread.get(key, 0.0), noise)
        if ref is None or main is None:
            return None, noise
        return _screen_gain(ref, main, t.rules.goal), noise

    def _vs_start(self, setup: Setup, question: str) -> float | None:
        from inferenceos.tune.pipeline import _screen_gain

        workload = self._workload(question)
        if workload is None:
            return None
        start = self._meas.get((self._start_setup, workload.workload_id))
        mine = self._meas.get((setup, workload.workload_id))
        if start is None or mine is None:
            return None
        return _screen_gain(start, mine, self.tuner.rules.goal)

    def _cost(self, question: str) -> float | None:
        """Seconds for one fresh-load measurement on the question, from this tune; for the
        long question before it was measured: the short measurement's one-off part plus
        its answers at the long question's target length."""
        workload = self._workload(question)
        if workload is not None:
            known = self.ledger.measurement_cost(workload.workload_id)
            if known is not None:
                return known
        if question == "short":
            return None
        t = self.tuner
        short_cost = self.ledger.measurement_cost(t.workload.workload_id)
        if short_cost is None:
            return None
        per_answer = (t._long_question_record or {}).get("expected_seconds_per_answer")
        if not isinstance(per_answer, (int, float)):
            per_answer = float(t.settings.tune_long_question_seconds)
        warm = t.experiment_policy.warmup_repetitions
        short_answers = t._path_policy(t.workload).measured_repetitions + warm
        long_answers = t._path_policy(self._long).measured_repetitions + warm
        answer = self.ledger.answer_cost(t.workload.workload_id)
        one_off = short_cost - short_answers * answer if answer is not None else 0.5 * short_cost
        overhead = self.ledger.summary().get("monitor_overhead_per_run_s")
        overhead_s = float(overhead) if isinstance(overhead, (int, float)) else 0.0
        return max(one_off, 0.0) + long_answers * (float(per_answer) + overhead_s)

    # ------------------------------------------------------------------ #
    # Pieces                                                               #
    # ------------------------------------------------------------------ #
    def _layer_top(self, current: RuntimeConfiguration) -> tuple[int, int, dict[str, object]]:
        """The largest layer count that loads now (same rules as the layer round)."""
        t = self.tuner
        current_layers = int(current.gpu_offload_layers or 0)
        count_max = max_count(int(t.model.n_layers), engine_rule(t.machine.program_version))
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        used = t.graphics_memory_reader()
        envelope = build_resource_envelope(
            total_vram_bytes=t.machine.total_graphics_memory_bytes, used_vram_bytes=used,
            telemetry_available=used is not None, telemetry_observed_at=now if used is not None else None,
            telemetry_source="tune: graphics memory before measuring", now=now, model=t.model,
            context_length=t.context_length, current_layers=current_layers,
            total_layers=t.model.n_layers, policy=t.search_policy,
        )
        estimate: int | None = None
        if envelope.sufficient and envelope.feasible_max is not None:
            estimate = min(max(current_layers, int(envelope.feasible_max)), count_max)
        result = find_layer_top(
            n_layers=count_max, estimate=estimate, try_load=t._try_load_with(current),
            memory_readable=t.machine.total_graphics_memory_bytes is not None, say=t.say,
        )
        top = min(max(0, int(result.top)), count_max)
        reasons = list(result.reasons)
        if top == 0 and current_layers > 0 and result.tries:
            top = min(estimate if estimate is not None else current_layers, count_max)
            reasons.append(f"load-only tries failed; using the estimate of {top}")
        return top, count_max, {"round": "layer top", "top": top, "largest_count": count_max,
                                "estimate": estimate, "reasons": reasons}

    def _batch_fits(self, start: RuntimeConfiguration, layers: int, size: int) -> bool:
        context = self.tuner._path_context(start.model_copy(update={"gpu_offload_layers": layers}))
        allowed, _ = _batch_sizes(dataclasses.replace(context, question_tokens=None))
        return size in allowed

    def _changes(self, setup: Setup, planner: Planner) -> list[ProposedChange]:
        assert self._safe_start is not None
        changes: list[ProposedChange] = []
        config = self._config(setup)
        for category in CATEGORIES:
            if setup[SLOT[category]] == self._start_setup[SLOT[category]]:
                continue
            setting = SETTING[category]
            value = getattr(config, setting)
            direct = self._vs_start(setup, QUESTION[category])
            changes.append(ProposedChange(
                change_id=f"{CARD_OF[category]}:practical:{value}", card_id=CARD_OF[category],
                setting=setting, description=f"{category} (practical search)",
                old_value=getattr(self._safe_start, setting), new_value=value, measured_gain=None,
                evidence=[f"whole setup read {direct:+.1%} against the start on the {QUESTION[category]} question "
                          "(selection only, not evidence)" if direct is not None else "screened"],
            ))
        return changes

    def _workloads_of(self, change: ProposedChange) -> list[BenchmarkWorkload]:
        """Same coverage as Unbounded: layers and threads on the short question; prompt
        batch found on the long and guarded on the short."""
        assert self._short is not None
        category = TABLES.by_setting(change.setting)
        if category is not None:
            # Declared questions (judged, then guarded), when every one is available.
            workloads = [self._workload(q) for q in TABLES.questions_of(category)]
            if all(w is not None for w in workloads):
                return [w for w in workloads if w is not None]
        return [self._short]

    def _publish(self, *, starting: RuntimeConfiguration, final: RuntimeConfiguration,
                 approved: list[ProposedChange], declined: list[ProposedChange], gain: float | None,
                 power: PowerState, notes: list[str]) -> Any:
        """Publication happens at the commit -- the atomic replace of the saved-setups
        file -- not when saving begins (review 10). Hashing, serializing and the
        temporary file come first; a stop registered before the commit's last check
        leaves the previous setup unchanged; a stop after it keeps the new setup and is
        reported as arriving after publication. No rollback."""
        self.check_user()
        t = self.tuner
        assert self._base is not None
        began = time.perf_counter()
        guard = CommitGuard(check=self.check_user, lock=self._commit_lock,
                            committed=self._mark_published, event=self.save_events)
        saved = t._save(run_id=self._base.run_id, power=power, starting=starting, final=final,
                        approved=approved, declined=declined, confirmed_gain=gain, notes=notes,
                        guard=guard)
        self.ledger.add("save", None, time.perf_counter() - began, started_at=self._now())
        return saved

    def _safety_only(self, starting: RuntimeConfiguration, safe_start: RuntimeConfiguration,
                     safety: ProposedChange | None, power: PowerState,
                     safety_check: ConfirmationResult | None, *, outcome: str,
                     proposed: list[ProposedChange] | None = None,
                     confirmation: ConfirmationResult | None = None) -> TuneResult:
        """Nothing new confirmed: only a verified memory-safety correction may be saved
        (after its own yes); otherwise the last saved setup stays as it was."""
        t = self.tuner
        if safety is None or safety_check is None or not safety_check.confirmed:
            self._notes.append("nothing was saved; the last saved setup (if any) is unchanged")
            return self._finish(outcome=outcome, proposed=proposed or [], confirmation=confirmation)
        self.check_user()
        decision = t.approver.decide(safety, timeout_s=float(t.settings.tune_approval_timeout_s))
        self.check_user()
        if not decision.approved:
            return self._finish(outcome="not saved: memory-safety correction declined",
                                proposed=[safety, *(proposed or [])], decisions=[decision])
        saved = self._publish(starting=starting, final=safe_start, approved=[safety],
                              declined=list(proposed or []), gain=None, power=power,
                              notes=["memory-safety correction (no speed claim)"])
        return self._finish(outcome=f"{outcome} (memory-safety correction saved)",
                            proposed=[safety, *(proposed or [])], decisions=[decision], saved=saved,
                            final=safe_start, confirmation=safety_check)

    def _finish(self, *, outcome: str, proposed: list[ProposedChange] | None = None,
                confirmation: ConfirmationResult | None = None,
                decisions: list[ChangeDecision] | None = None, saved: Any = None,
                final: RuntimeConfiguration | None = None) -> TuneResult:
        t = self.tuner
        assert self._base is not None
        if self.cancel_after_publication:
            self._notes.append("a stop arrived after the new setup had been saved (publication had "
                               "completed), so it stays saved")
        began = time.perf_counter()
        self._stop_engine()  # cleanup, timed
        self.ledger.add("cleanup", None, time.perf_counter() - began, started_at=self._now())
        used = self._now()
        target = float(self.budget_s)
        self._rounds.append({
            "round": "time budget",
            "target_s": target,
            "allowed_extension_s": round(self._hard - self._target, 1),
            "used_s": round(used, 1),
            "extension_s": round(max(used - target, 0.0), 1),
            "extension_reasons": list(self.extensions),
            "user_cancelled": self.user_cancelled,
            "cancel_after_publication": self.cancel_after_publication,
            "stuck": list(self.stuck),
            "costs": self.ledger.summary(),
            # Report only (practical-mode evaluation): the search order in full.
            "operations": list(self.operations),
            "steps": list(self.steps),
            "screenings": list(self.screenings),
            "planner_log": list(self._planner.log) if self._planner is not None else [],
        })
        t.say(f"Time used: {used / 60:.1f} min against a {target / 60:.0f} min target ({outcome}).")
        return self._base.model_copy(update={
            "rounds": list(self._rounds), "notes": list(self._notes), "outcome": outcome,
            "proposed_changes": list(proposed or []), "confirmation": confirmation,
            "decisions": list(decisions or []), "saved_setup": saved,
            "final_configuration": final if final is not None else self._base.starting_configuration,
        })


__all__ = ["BudgetedTune", "Cancelled", "ThreadWatchdog", "Watchdog", "fixed_start"]
