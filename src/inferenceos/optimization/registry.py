"""Explicit action registry.

Candidates may only be generated from registered :class:`CandidateAction`
definitions. The optimizer never invents shell commands, PowerShell commands,
executable invocations, registry edits, OS configuration changes, llama.cpp flags,
filesystem operations, or process-control operations. Unknown action ids are
rejected.
"""

from __future__ import annotations

import re

from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    OptimizationScope,
    ParameterType,
    PrivilegeRequirement,
)
from inferenceos.contracts.optimization import CandidateAction, ParameterSpec

_ACTION_ID_RE = re.compile(r"^[a-z][a-z0-9]*(\.[a-z][a-z0-9_]*)*$")


class ActionRegistryError(ValueError):
    """Raised when an action definition is invalid or duplicated."""


class UnknownActionError(KeyError):
    """Raised when an unregistered action id is resolved."""


def _validate_parameter(spec: ParameterSpec, *, action_id: str) -> None:
    if not spec.name:
        raise ActionRegistryError(f"{action_id}: parameter name must be non-empty")
    if spec.type is ParameterType.BOOLEAN:
        if spec.minimum is not None or spec.maximum is not None:
            raise ActionRegistryError(f"{action_id}.{spec.name}: boolean has no bounds")
    if spec.type in (ParameterType.INTEGER, ParameterType.FLOAT):
        if spec.minimum is not None and spec.maximum is not None and spec.minimum > spec.maximum:
            raise ActionRegistryError(f"{action_id}.{spec.name}: minimum > maximum")
    if spec.type is ParameterType.STRING and not spec.choices:
        # Free-form strings are only allowed when explicitly enumerated.
        raise ActionRegistryError(
            f"{action_id}.{spec.name}: string parameters require explicit choices"
        )


def validate_action(action: CandidateAction) -> None:
    if not _ACTION_ID_RE.match(action.action_id):
        raise ActionRegistryError(f"invalid action id: {action.action_id!r}")
    if not action.supported_bottlenecks:
        raise ActionRegistryError(f"{action.action_id}: no supported bottlenecks")
    names = [spec.name for spec in action.parameters]
    if len(names) != len(set(names)):
        raise ActionRegistryError(f"{action.action_id}: duplicate parameter names")
    for spec in action.parameters:
        _validate_parameter(spec, action_id=action.action_id)
    broad = action.scope in {OptimizationScope.EXTERNAL_PROCESS, OptimizationScope.SYSTEM_WIDE}
    elevated = action.privilege is PrivilegeRequirement.ELEVATION_REQUIRED
    if (broad or elevated) and not action.requires_human_approval:
        raise ActionRegistryError(
            f"{action.action_id}: external/system-wide/elevated actions must be "
            "approval-required"
        )

    # Phase 11R hard boundary: InferenceOS never autonomously controls unrelated
    # external processes or system-wide configuration. Any action whose scope is
    # external-process or system-wide must therefore be recommendation-only — it must
    # not be marked implemented (i.e. there is no executor for it).
    if action.implemented and action.scope in {
        OptimizationScope.EXTERNAL_PROCESS,
        OptimizationScope.SYSTEM_WIDE,
    }:
        raise ActionRegistryError(
            f"{action.action_id}: Phase 11R is recommendation-only; an "
            "external-process/system-wide action must not be implemented (no autonomous "
            "external-process control)"
        )
    if (
        action.scope is OptimizationScope.EXTERNAL_PROCESS
        and action.implemented
    ):
        raise ActionRegistryError(
            f"{action.action_id}: external-process actions are never executed by InferenceOS"
        )
    if action.implemented and action.privilege is PrivilegeRequirement.ELEVATION_REQUIRED:
        raise ActionRegistryError(
            f"{action.action_id}: Phase 11 never implements an elevation-required action"
        )
    if action.scope in {
        OptimizationScope.EXTERNAL_PROCESS,
        OptimizationScope.SYSTEM_WIDE,
    } and not action.requires_human_approval:
        raise ActionRegistryError(
            f"{action.action_id}: external/system-wide recommendations require a user decision"
        )


class ActionRegistry:
    """Deterministic, validated lookup of registered actions."""

    def __init__(self) -> None:
        self._actions: dict[str, CandidateAction] = {}

    def register(self, action: CandidateAction) -> CandidateAction:
        validate_action(action)
        if action.action_id in self._actions:
            raise ActionRegistryError(f"duplicate action id: {action.action_id}")
        self._actions[action.action_id] = action
        return action

    def __contains__(self, action_id: object) -> bool:
        return isinstance(action_id, str) and action_id in self._actions

    def __len__(self) -> int:
        return len(self._actions)

    def get(self, action_id: str) -> CandidateAction:
        try:
            return self._actions[action_id]
        except KeyError as exc:
            raise UnknownActionError(action_id) from exc

    def get_optional(self, action_id: str) -> CandidateAction | None:
        return self._actions.get(action_id)

    def all(self) -> list[CandidateAction]:
        return [self._actions[key] for key in sorted(self._actions)]

    def ids(self) -> list[str]:
        return sorted(self._actions)

    def for_bottleneck(self, kind: BottleneckKind) -> list[CandidateAction]:
        return [
            action
            for action in self.all()
            if kind in action.supported_bottlenecks
        ]
