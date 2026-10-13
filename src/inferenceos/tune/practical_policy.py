"""Practical's exploration dispatch, preserving BudgetedTune's existing decisions.

Planner owns candidate queues/choices. BudgetedTune owns evidence, costs, repetition
state and every measuring helper. This policy owns only traversal and the stop reason.
Post-exploration evidence completion and confirmation stay in BudgetedTune._run.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from inferenceos.tune.coordinator import (
    Action, CHANGED, DONE, Finished, NO_CHANGE, NO_TIME, NOT_APPLICABLE,
    Outcome, PRACTICAL, PRACTICAL_PATH_IDS, REOPEN_BATCH, SCREEN, SIZE_LONG,
    SearchTrace, WEAK_REPEAT, practical_step_action, snapshot,
)
from inferenceos.tune.planner import Planner, Step, TABLES

if TYPE_CHECKING:
    from inferenceos.tune.budgeted import BudgetedTune


class PracticalPolicy:
    mode = PRACTICAL

    def __init__(self, tune: BudgetedTune, planner: Planner) -> None:
        self.tune = tune
        self.planner = planner
        self.phase = "top"
        self.stop_reason = ""
        self.weak_outcome = ""

    def _control(self, trace: SearchTrace, kind: str, reason: str) -> Action:
        return trace.issue(kind=kind, path="prompt_batch" if kind != WEAK_REPEAT else "exploration",
                           label=kind, reason=reason, decided_by="PracticalPolicy (existing exploration order)",
                           carried=self.tune._config(self.planner.current_setup()))

    def next_action(self, trace: SearchTrace) -> Action | Finished:
        tune, planner = self.tune, self.planner
        if self.phase == "done":
            return Finished(self.stop_reason or "plan finished")
        if self.phase == "after weak":
            # No new check_user/next_step here: this is the tail of the SAME old
            # exploration iteration. An extra check could cut off a permitted pair.
            if tune._batch_left_out and not tune._batch_retried:
                tune._batch_retried = True
                return self._control(trace, REOPEN_BATCH, "one reconsideration after other readings")
            self.stop_reason = self.weak_outcome if self.weak_outcome not in ("", "none") else ""
            return Finished(self.stop_reason or "plan finished")
        tune.check_user()
        if tune.stuck:
            self.stop_reason = "an operation appeared stuck: " + tune.stuck[-1]
            return Finished(self.stop_reason)
        step = planner.next_step()
        if step is None and planner.waiting_for_length:
            return self._control(trace, SIZE_LONG, "planner is waiting for the long question's length")
        if step is None:
            return self._control(trace, WEAK_REPEAT, "no ordinary step; consider an existing weak reading")
        return practical_step_action(trace, step, planner.current_setup(), tune._config)

    def execute(self, action: Action, trace: SearchTrace) -> Outcome:
        tune, planner = self.tune, self.planner
        if action.kind == SCREEN:
            step: Step = action.handle
            missing = TABLES.missing_capability(step.category, {"long_question": tune._long is not None})
            if missing is not None:
                planner.log.append(f"{step.category} skipped: {missing}")
                return Outcome(NOT_APPLICABLE, missing)
            tune.steps.append({"n": len(tune.steps) + 1, "at_s": round(tune._now(), 1), "category": step.category,
                               "values": list(step.values), "reason": step.reason, "base": list(step.base),
                               "question": step.question})
            before = planner.current_setup()
            for value in step.values:
                # The step's base is captured once, even if an earlier value changes
                # the chosen configuration. Children describe actual attempted pairs.
                child = trace.issue(kind=SCREEN, path=PRACTICAL_PATH_IDS[step.category],
                                    label=f"{step.category} {value}", reason=step.reason,
                                    decided_by="Step.values", parent=action,
                                    carried=tune._config(planner.current_setup()), reference=tune._config(step.base),
                                    candidates=[tune._config(step.setup_for(value))])
                at = len(tune.screenings)
                try:
                    why = tune._screen(planner, step, value)
                except BaseException as exc:
                    trace.failed("screen candidate", exc, child)
                    raise
                trace.annotate(child, screenings=tune.screenings[at:], question=step.question)
                trace.record(child, Outcome(NO_TIME if why else DONE, why or "screened (not confirmed)"))
                if why:
                    return Outcome(NO_TIME, why)
            changed = before != planner.current_setup()
            return Outcome(CHANGED if changed else NO_CHANGE, "screening step completed (not confirmation)",
                           carried_after=snapshot(tune._config(planner.current_setup())) if changed else None)
        at = len(tune.screenings)
        if action.kind == WEAK_REPEAT:
            result = tune._repeat_weak(planner)
            # The helper's immutable comparison record is authoritative, including
            # the historical setup and fresh operation IDs. No second selection.
            trace.annotate(action, screenings=tune.screenings[at:])
            return Outcome(DONE, result, detail=result)
        if action.kind in (SIZE_LONG, REOPEN_BATCH):
            reopened = action.kind == REOPEN_BATCH
            result_long = tune._size_long(planner, reopen=reopened)
            trace.annotate(action, question=None if tune._long is None else tune._long.workload_id)
            return Outcome(DONE, "long question available" if result_long else "long question not opened",
                           detail=result_long)
        raise ValueError(f"unknown Practical action kind {action.kind!r}")

    def advance(self, action: Action, outcome: Outcome, trace: SearchTrace) -> None:
        if action.kind == WEAK_REPEAT:
            self.weak_outcome = outcome.detail
            self.phase = "top" if self.weak_outcome == "repeated" else "after weak"
        elif action.kind == REOPEN_BATCH:
            self.phase = "top" if outcome.detail else "done"
            if self.phase == "done":
                self.stop_reason = self.weak_outcome if self.weak_outcome not in ("", "none") else ""
        elif outcome.status == NO_TIME:
            self.stop_reason, self.phase = outcome.why, "done"
        # SCREEN (including skip/no-change) and SIZE_LONG go around the original
        # loop. Their helpers already updated planner/evidence state exactly once.
