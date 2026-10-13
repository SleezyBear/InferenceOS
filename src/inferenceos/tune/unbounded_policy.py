"""Unbounded's option-path search as a coordinator policy (C3-6, 11 Oct 2026).

This is the option-path loop that was inline in ``Tuner._run_search``, moved here
unchanged in behaviour. The policy owns every piece of that loop's state:

* the queue (``remaining``) and its order (``Diagnosis.order`` when diagnosis is on);
* ``tested`` (a path is never run twice on the exact same setup), ``finished``, the
  per-path ``revisits_left`` and ``layer_revisits_left``, the round ``number``;
* the carried configuration (``current``), the proposed ``changes``, the report's
  ``rounds`` and ``not_reached`` lists (shared with the caller, appended in place);
* the current ``diagnosis`` and the time limit (``path_deadline``).

The steps it runs are the tuner's own, unchanged: ``_facts_round``, ``_diagnose``,
``_check_switches``, ``_path_round`` (which calls ``OptionPath.check`` on the exact
current setup), ``_layer_round`` (the layer revisit) and ``merge_change``. After a kept
change on a path that shapes memory, the layer revisit runs at once, before the
affected finished paths are queued again and before the diagnosis is refreshed, as
before. The initial layer search, the memory-safety correction, the heat check, the
final check, approval and saving stay in ``Tuner._run_search``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from inferenceos.contracts.experiment import ExperimentMeasurement, RuntimeConfiguration
from inferenceos.contracts.tune import ProposedChange
from inferenceos.tune.approval import describe_change
from inferenceos.tune.catalog import CATALOG
from inferenceos.tune.coordinator import (
    ALREADY_TRIED,
    CHANGED,
    DIAGNOSIS,
    DONE,
    LAYER_REVISIT,
    NO_CHANGE,
    NO_TIME,
    NOT_APPLICABLE,
    PATH_ROUND,
    SWITCH_CHECK,
    UNBOUNDED,
    Action,
    Finished,
    Outcome,
    SearchTrace,
    snapshot,
)
from inferenceos.tune.diagnose import MEMORY_PATHS, NORMAL, PROBE, Diagnosis, affected_paths
from inferenceos.tune.measure import DisturbanceWatch
from inferenceos.tune.paths import OptionPath

if TYPE_CHECKING:
    from inferenceos.tune.pipeline import Tuner


class UnboundedPolicy:
    """The Unbounded option-path loop, driven by ``coordinate``."""

    mode = UNBOUNDED

    def __init__(
        self,
        tuner: Tuner,
        *,
        current: RuntimeConfiguration,
        changes: list[ProposedChange],
        rounds: list[dict[str, object]],
        not_reached: list[str],
        starting: RuntimeConfiguration,
        start_measurement: ExperimentMeasurement,
        watch: DisturbanceWatch,
        safety_ids: set[str],
        path_deadline: float | None,
    ) -> None:
        self.tuner = tuner
        self.current = current
        self.changes = changes
        self.rounds = rounds
        self.not_reached = not_reached
        self.starting = starting
        self.start_measurement = start_measurement
        self.watch = watch
        self.safety_ids = safety_ids
        self.path_deadline = path_deadline
        self.chosen_paths: list[OptionPath] = CATALOG.by_names(tuner._path_names())
        self.diagnosis: Diagnosis | None = None
        self.by_name = {item.name: item for item in self.chosen_paths}
        self.remaining: list[OptionPath] = list(self.chosen_paths)
        # Looking again (owner, 4 Oct 2026): after a kept change, the finished paths it
        # can affect run again, at most ``tune_path_revisits`` extra times each, within
        # the time left; a path is never run twice on the exact same setup.
        revisits = max(int(tuner.settings.tune_path_revisits), 0)
        self.revisits_left = {name: revisits for name in self.by_name}
        # The layer count is searched again after a kept change that moved card memory
        # (placement, conversation memory), within the same revisit limit.
        self.layer_revisits_left = revisits
        self.finished: set[str] = set()
        self.tested: set[tuple[str, str]] = set()
        self.number = 1
        #: Why each queued path is in the queue (report only).
        self.queued_because: dict[str, str] = {name: "in the chosen path order (tune_paths)" for name in self.by_name}

    # ------------------------------------------------------------------ #
    # Preparation                                                          #
    # ------------------------------------------------------------------ #
    def start(self, trace: SearchTrace) -> None:
        """The facts round and the first diagnosis, when there are paths to run."""
        tuner = self.tuner
        if self.chosen_paths:
            self.rounds.append(tuner._facts_round(self.current))
        # Diagnosis first (owner, 4 Oct 2026): it orders the paths and decides how
        # deeply to look at each; it skips nothing. Run again after each kept change.
        if self.chosen_paths and bool(tuner.settings.tune_diagnosis):
            action = trace.issue(kind=DIAGNOSIS, path="diagnosis", label="diagnosis (start)",
                                 reason="diagnosis first: it orders the paths and sets how deeply to look",
                                 decided_by="tune_diagnosis", carried=self.current)
            try:
                outcome = self._diagnose("start")
            except BaseException as exc:
                trace.failed("preparation", exc, action)
                raise
            trace.record(action, outcome)

    # ------------------------------------------------------------------ #
    # The loop                                                             #
    # ------------------------------------------------------------------ #
    def next_action(self, trace: SearchTrace) -> Action | Finished:
        if not self.remaining:
            return Finished("every chosen path and revisit was handled")
        decided_by = "queue order"
        if self.diagnosis is not None:
            order = self.diagnosis.order([item.name for item in self.remaining])
            self.remaining.sort(key=lambda item: order.index(item.name))
            decided_by = "Diagnosis.order"
        path = self.remaining.pop(0)
        self.number += 1
        again = path.name in self.finished
        label = path.label + (" (again, after a kept change)" if again else "")
        return trace.issue(
            kind=PATH_ROUND, path=path.name, label=f"Round {self.number}: {label}",
            reason=self.queued_because.get(path.name, ""), decided_by=decided_by, carried=self.current,
            handle=(path, again, label, self.number),
        )

    def execute(self, action: Action, trace: SearchTrace) -> Outcome:
        tuner = self.tuner
        path, again, label, number = action.handle
        key = (path.name, _setup_key(self.current))
        if key in self.tested:
            tuner.say(f"Round {number} ({label}) skipped: already tried on this exact setup.")
            return Outcome(ALREADY_TRIED, "already tried on this exact setup")
        if self.path_deadline is not None and time.perf_counter() >= self.path_deadline:
            self.not_reached.append(f"{label} (time limit reached)")
            tuner.say(f"Round {number} ({label}) not reached: time limit.")
            return Outcome(NO_TIME, "time limit reached")
        depth = self.diagnosis.priority(path.name) if self.diagnosis is not None else NORMAL
        if path.name == "engine_switches":
            trace.run_child(
                action, kind=SWITCH_CHECK, path="engine_switches", label="engine switch checks",
                reason="switches are checked to take effect and keep answers before any speed test",
                decided_by="Tuner._check_switches", carried=self.current,
                work=lambda: self._switch_check(depth),
            )
        tuner.say(f"Round {number}: {label}" + (" (quick probe)" if depth == PROBE else ""))
        path_round, path_change, new_config = tuner._path_round(
            path, self.current, self.watch, self.path_deadline, depth=depth,
        )
        self.tested.add(key)
        self.finished.add(path.name)
        if again:
            path_round["again"] = True
        if tuner._long_question_record is not None and not any(
            item.get("round") == "long test question" for item in self.rounds
        ):
            self.rounds.append(dict(tuner._long_question_record))
        self.rounds.append(path_round)
        if path_change is not None and new_config is not None:
            return Outcome(CHANGED, describe_change(path_change), carried_after=snapshot(new_config),
                           detail=(path_change, new_config))
        why = str(path_round.get("reason", ""))
        status = NOT_APPLICABLE if path_round.get("applies") is False else NO_CHANGE
        return Outcome(status, why, detail=path_round)

    def advance(self, action: Action, outcome: Outcome, trace: SearchTrace) -> None:
        """After every completed action: a kept change moves the carried setup and may
        bring back the layer count, finished paths and the diagnosis; otherwise the
        loop simply goes on (as before)."""
        tuner = self.tuner
        path: OptionPath = action.handle[0]
        if outcome.status in (ALREADY_TRIED, NO_TIME):
            return
        if outcome.status != CHANGED:
            path_round = outcome.detail
            tuner.say(f"  no change: {path_round.get('reason')}")
            return
        path_change, new_config = outcome.detail
        self.current = new_config
        self.changes = _merge(tuner, self.changes, path_change, self.starting, self.safety_ids)
        tuner.say(f"  found: {describe_change(path_change)}")
        touched = [path.name]
        layer_key = ("layers", _setup_key(self.current))
        if (
            path.name in MEMORY_PATHS
            and bool(tuner.settings.tune_layer_path)
            and self.layer_revisits_left > 0
            and layer_key not in self.tested
            and (self.path_deadline is None or time.perf_counter() < self.path_deadline)
        ):
            self.layer_revisits_left -= 1
            self.tested.add(layer_key)
            trace.run_child(
                action, kind=LAYER_REVISIT, path="layers", label="graphics card layers (again)",
                reason=f"a kept {path.name} change moved what fits on the card",
                decided_by="MEMORY_PATHS", carried=self.current,
                work=lambda: self._layer_revisit(touched),
            )
        queued = {item.name for item in self.remaining}
        for name in [item for changed in touched for item in affected_paths(changed)]:
            if name not in self.by_name or name in queued or name not in self.finished:
                continue
            if self.revisits_left[name] <= 0:
                continue
            self.revisits_left[name] -= 1
            self.remaining.append(self.by_name[name])
            queued.add(name)
            self.queued_because[name] = f"revisit after a kept change on {'/'.join(touched)} (affected_paths)"
            tuner.say(
                f"  will look again at {self.by_name[name].label}: the change can move "
                "what limits it"
            )
        if self.diagnosis is not None and self.remaining:
            # The bottleneck can move after a change: look again.
            trace.run_child(
                action, kind=DIAGNOSIS, path="diagnosis", label=f"diagnosis (after {path.label})",
                reason="the bottleneck can move after a kept change", decided_by="tune_diagnosis",
                carried=self.current, work=lambda: self._diagnose(f"after {path.label}", keep_old=True),
            )

    # ------------------------------------------------------------------ #
    # Nested steps (the tuner's own code)                                  #
    # ------------------------------------------------------------------ #
    def _switch_check(self, depth: str) -> Outcome:
        record = self.tuner._check_switches(self.current, depth, self.path_deadline)
        self.rounds.append(record)
        return Outcome(DONE, str(record.get("reason", "")) if isinstance(record, dict) else "")

    def _diagnose(self, label: str, *, keep_old: bool = False) -> Outcome:
        new_diagnosis, record = self.tuner._diagnose(self.current, self.watch, label, self.path_deadline)
        self.rounds.append(record)
        if new_diagnosis is not None or not keep_old:
            self.diagnosis = new_diagnosis
        why = str(record.get("skipped", "")) if isinstance(record, dict) else ""
        return Outcome(DONE, why)

    def _layer_revisit(self, touched: list[str]) -> Outcome:
        tuner = self.tuner
        tuner.say("Graphics card layers (again: the change moved what fits on the card)")
        layer_again, layer_change_again, _ = tuner._layer_round(
            self.current, self.start_measurement, self.watch,
            self.path_deadline if self.path_deadline is not None else float("inf"),
            revisit=True,
        )
        layer_again["again"] = True
        self.rounds.append(layer_again)
        if layer_change_again is not None:
            new_layers = layer_change_again.new_value
            assert isinstance(new_layers, int)
            self.current = self.current.model_copy(update={"gpu_offload_layers": new_layers})
            tuner._change_workloads[layer_change_again.change_id] = [tuner.workload]
            self.changes = _merge(tuner, self.changes, layer_change_again, self.starting, self.safety_ids)
            tuner.say(f"  found: {describe_change(layer_change_again)}")
            touched.append("layers")
            return Outcome(CHANGED, describe_change(layer_change_again), carried_after=snapshot(self.current))
        tuner.say(f"  no better layer count: {layer_again.get('reason')}")
        return Outcome(NO_CHANGE, str(layer_again.get("reason", "")))


def _setup_key(config: RuntimeConfiguration) -> str:
    from inferenceos.tune.pipeline import setup_key

    return setup_key(config)


def _merge(tuner: Any, changes: list[ProposedChange], change: ProposedChange,
           starting: RuntimeConfiguration, safety_ids: set[str]) -> list[ProposedChange]:
    from inferenceos.tune.pipeline import merge_change

    return merge_change(changes, change, tuner._change_workloads, starting, keep_apart=safety_ids)


__all__ = ["UnboundedPolicy"]
