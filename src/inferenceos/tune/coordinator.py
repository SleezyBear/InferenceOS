"""The shared search coordinator (C3-6, 11 Oct 2026).

One loop drives a tune's search through a mode-owned policy::

    action = policy.next_action(trace)        # or FINISHED
    outcome = policy.execute(action, trace)   # runs it through the mode's existing code
    trace.record(action, outcome)
    policy.advance(action, outcome, trace)    # after every completed outcome

The coordinator makes no ordering, eligibility, time, repeat or revisit decision of its
own, keeps no queue and no evidence: each policy owns its queue, limits, evidence and
choices, and supplies every action's configurations. What the coordinator adds is the
ordered trace of actions and outcomes (the ``search actions`` round in a report).

* **Configurations.** An action carries detached snapshots of the mode's *carried*
  configuration when it was issued, and, where the policy measures against one, its
  *reference* and *candidate* configurations, all supplied by the policy. A screening
  choice is not an acceptance: confirmation, approval and saving stay outside.
* **Outcomes.** ``changed`` (a change the mode keeps for now), ``no change``,
  ``not applicable``, ``already tried``, ``no time``, ``done`` (a nested step that
  proposes no change), and ``interrupted`` (an exception left the action: the
  exception is re-raised unchanged; nothing is retried).
* **Nesting.** Work that runs inside an action (an engine-switch check before a path,
  a layer revisit or a diagnosis after a kept change) is recorded as a child action.

Unbounded runs its option-path loop through ``UnboundedPolicy`` (``tune/pipeline.py``).
Practical is not wired yet (C3-7); ``practical_step_action`` shows how one of its
planner steps is represented: the reference is the step's captured ``base``, and every
candidate is built from that base, not from the planner's current choice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.tune.declarations import DECLARATIONS

UNBOUNDED, PRACTICAL = "unbounded", "practical"

#: Outcome statuses.
CHANGED, NO_CHANGE, NOT_APPLICABLE, ALREADY_TRIED, NO_TIME, DONE, INTERRUPTED = (
    "changed", "no change", "not applicable", "already tried", "no time", "done", "interrupted",
)
STATUSES = (CHANGED, NO_CHANGE, NOT_APPLICABLE, ALREADY_TRIED, NO_TIME, DONE, INTERRUPTED)

#: Action kinds.
PATH_ROUND, SWITCH_CHECK, LAYER_REVISIT, DIAGNOSIS, SCREEN = (
    "path round", "engine switch check", "layer revisit", "diagnosis", "screen",
)

#: The round name a report's coordinator trace is stored under.
TRACE_ROUND = "search actions"

#: Practical planner categories -> canonical path ids, from the declarations: the
#: planner's ``placement`` is the layer count (``layers``) and ``batch`` is
#: ``prompt_batch``. Canonical ``placement`` (per-piece placement) is a different path.
PRACTICAL_PATH_IDS: Mapping[str, str] = MappingProxyType(
    {d.practical.category: d.name for d in DECLARATIONS if d.practical is not None}
)

Snapshot = Mapping[str, Any]


def snapshot(config: RuntimeConfiguration | None) -> Snapshot | None:
    """A detached, read-only copy of a configuration for the record."""
    if config is None:
        return None
    return MappingProxyType(config.model_dump(mode="json", exclude_none=True))


@dataclass(frozen=True)
class Action:
    """One unit of search work, as the policy issued it."""

    id: int
    mode: str
    kind: str
    path: str
    label: str
    reason: str
    decided_by: str
    carried: Snapshot | None = None
    reference: Snapshot | None = None
    candidates: tuple[Snapshot, ...] = ()
    parent: int | None = None
    #: The policy's own handle for executing the action (an ``OptionPath``, a planner
    #: ``Step``); never read by the coordinator and never recorded.
    handle: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Outcome:
    status: str
    why: str = ""
    #: The mode's carried configuration after the action (when it changed).
    carried_after: Snapshot | None = None
    #: The policy's own result (a round record, a proposed change); never recorded.
    detail: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown outcome status {self.status!r}")


@dataclass(frozen=True)
class Finished:
    """The policy has no more work."""

    reason: str


class SearchPolicy(Protocol):
    mode: str

    def next_action(self, trace: SearchTrace) -> Action | Finished: ...

    def execute(self, action: Action, trace: SearchTrace) -> Outcome: ...

    def advance(self, action: Action, outcome: Outcome, trace: SearchTrace) -> None: ...


@dataclass
class SearchTrace:
    """The ordered record of actions and outcomes (report only; no decisions)."""

    mode: str
    entries: list[dict[str, Any]] = field(default_factory=list)
    finished: str | None = None
    _next_id: int = 1
    _open: dict[int, dict[str, Any]] = field(default_factory=dict)

    def issue(
        self,
        *,
        kind: str,
        path: str,
        label: str,
        reason: str,
        decided_by: str,
        carried: RuntimeConfiguration | None = None,
        reference: RuntimeConfiguration | None = None,
        candidates: Sequence[RuntimeConfiguration] = (),
        parent: Action | None = None,
        handle: Any = None,
    ) -> Action:
        """A new action, with snapshots of the configurations the policy supplies."""
        action = Action(
            id=self._next_id, mode=self.mode, kind=kind, path=path, label=label, reason=reason,
            decided_by=decided_by, carried=snapshot(carried), reference=snapshot(reference),
            candidates=tuple(s for s in (snapshot(c) for c in candidates) if s is not None),
            parent=parent.id if parent is not None else None, handle=handle,
        )
        self._next_id += 1
        entry: dict[str, Any] = {
            "id": action.id, "kind": kind, "path": path, "label": label, "reason": reason,
            "decided_by": decided_by, "parent": action.parent,
            "carried": dict(action.carried) if action.carried is not None else None,
        }
        if action.reference is not None:
            entry["reference"] = dict(action.reference)
        if action.candidates:
            entry["candidates"] = [dict(c) for c in action.candidates]
        self.entries.append(entry)
        self._open[action.id] = entry
        return action

    def record(self, action: Action, outcome: Outcome) -> None:
        entry = self._open.pop(action.id)
        entry["status"] = outcome.status
        entry["why"] = outcome.why
        if outcome.carried_after is not None:
            entry["carried_after"] = dict(outcome.carried_after)

    def interrupted(self, action: Action, error: BaseException) -> None:
        """An exception left ``action``: recorded once, never retried, never masked."""
        entry = self._open.pop(action.id, None)
        if entry is not None:
            entry["status"] = INTERRUPTED
            entry["why"] = f"{type(error).__name__}: {error}"

    def run_child(
        self, parent: Action, *, kind: str, path: str, label: str, reason: str, decided_by: str,
        carried: RuntimeConfiguration | None, work: Callable[[], Outcome],
    ) -> Outcome:
        """Record nested work (issued, run, recorded; interrupted if it raises)."""
        child = self.issue(kind=kind, path=path, label=label, reason=reason, decided_by=decided_by,
                           carried=carried, parent=parent)
        try:
            outcome = work()
        except BaseException as exc:
            self.interrupted(child, exc)
            raise
        self.record(child, outcome)
        return outcome

    def round(self) -> dict[str, object]:
        """The report's ``search actions`` round."""
        return {"round": TRACE_ROUND, "mode": self.mode, "finished": self.finished,
                "actions": [dict(entry) for entry in self.entries]}


def coordinate(policy: SearchPolicy, trace: SearchTrace) -> SearchTrace:
    """Drive ``policy`` until it is finished. Exceptions leave unchanged."""
    while True:
        action = policy.next_action(trace)
        if isinstance(action, Finished):
            trace.finished = action.reason
            return trace
        try:
            outcome = policy.execute(action, trace)
        except BaseException as exc:
            trace.interrupted(action, exc)
            raise
        trace.record(action, outcome)
        policy.advance(action, outcome, trace)


def practical_step_action(
    trace: SearchTrace,
    step: Any,
    chosen: tuple[int, int, int],
    config_of: Callable[[tuple[int, int, int]], RuntimeConfiguration],
    *,
    decided_by: str = "Planner.next_step",
) -> Action:
    """How a Practical planner ``Step`` is represented (C3-7 is not wired yet): the path is
    the canonical id of its category, the reference is the step's captured ``base`` and
    every candidate is ``step.setup_for(value)`` from that same base."""
    return trace.issue(
        kind=SCREEN, path=PRACTICAL_PATH_IDS[step.category], label=f"{step.category} {list(step.values)}",
        reason=step.reason, decided_by=decided_by, carried=config_of(chosen), reference=config_of(step.base),
        candidates=[config_of(step.setup_for(value)) for value in step.values], handle=step,
    )


__all__ = [
    "ALREADY_TRIED", "CHANGED", "DIAGNOSIS", "DONE", "INTERRUPTED", "LAYER_REVISIT", "NOT_APPLICABLE",
    "NO_CHANGE", "NO_TIME", "PATH_ROUND", "PRACTICAL", "PRACTICAL_PATH_IDS", "SCREEN", "STATUSES",
    "SWITCH_CHECK", "TRACE_ROUND", "UNBOUNDED", "Action", "Finished", "Outcome", "SearchPolicy",
    "SearchTrace", "coordinate", "practical_step_action", "snapshot",
]
