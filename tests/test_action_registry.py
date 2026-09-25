"""Action registry: deterministic lookup, validation, unknown rejection."""

from __future__ import annotations

import pytest

from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    OptimizationScope,
    PrivilegeRequirement,
    RiskLevel,
)
from inferenceos.contracts.optimization import CandidateAction
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.registry import (
    ActionRegistry,
    ActionRegistryError,
    UnknownActionError,
)

EXPECTED_ACTIONS = {
    "runtime.right_size_context",
    "runtime.enable_mmap",
    "runtime.reduce_gpu_offload_layers",
    "runtime.increase_gpu_offload_layers",
    "runtime.reduce_threads",
    "external.suspend_process",
    "external.request_close",
    "external.reduce_priority",
    "system.increase_pagefile",
    "model.requantize",
}


def test_default_registry_ids() -> None:
    registry = build_default_registry()
    assert set(registry.ids()) == EXPECTED_ACTIONS
    assert len(registry) == len(EXPECTED_ACTIONS)


def test_lookup_is_deterministic() -> None:
    registry = build_default_registry()
    assert registry.ids() == sorted(EXPECTED_ACTIONS)
    assert registry.get("runtime.right_size_context").action_id == "runtime.right_size_context"


def test_for_bottleneck_maps_causally() -> None:
    registry = build_default_registry()
    paging = {action.action_id for action in registry.for_bottleneck(BottleneckKind.PAGING)}
    assert "runtime.right_size_context" in paging
    assert "runtime.enable_mmap" in paging
    assert "system.increase_pagefile" in paging
    # GPU tuning must not appear for a paging diagnosis.
    assert "runtime.reduce_gpu_offload_layers" not in paging


def test_unknown_action_rejected() -> None:
    registry = build_default_registry()
    with pytest.raises(UnknownActionError):
        registry.get("shell.run_command")
    assert "shell.run_command" not in registry


def test_duplicate_registration_rejected() -> None:
    registry = ActionRegistry()
    action = build_default_registry().get("runtime.right_size_context")
    registry.register(action)
    with pytest.raises(ActionRegistryError):
        registry.register(action)


def _action(**overrides: object) -> CandidateAction:
    base = build_default_registry().get("runtime.reduce_gpu_offload_layers").model_dump()
    base.update(overrides)
    return CandidateAction.model_validate(base)


def test_invalid_action_id_rejected() -> None:
    with pytest.raises(ActionRegistryError):
        ActionRegistry().register(_action(action_id="Bad ID"))


def test_action_without_supported_bottlenecks_rejected() -> None:
    with pytest.raises(ActionRegistryError):
        ActionRegistry().register(_action(supported_bottlenecks=[]))


def test_external_action_must_be_approval_required() -> None:
    with pytest.raises(ActionRegistryError):
        ActionRegistry().register(
            _action(
                scope=OptimizationScope.SYSTEM_WIDE,
                privilege=PrivilegeRequirement.ELEVATION_REQUIRED,
                requires_human_approval=False,
            )
        )


def test_elevated_action_must_be_approval_required() -> None:
    with pytest.raises(ActionRegistryError):
        ActionRegistry().register(
            _action(privilege=PrivilegeRequirement.ELEVATION_REQUIRED, requires_human_approval=False)
        )


def test_string_parameter_requires_choices() -> None:
    action = build_default_registry().get("external.suspend_process")
    spec = action.parameters[0]
    with pytest.raises(ActionRegistryError):
        ActionRegistry().register(
            _action(
                action_id="external.other",
                scope=OptimizationScope.EXTERNAL_PROCESS,
                requires_human_approval=True,
                risk_level=RiskLevel.MEDIUM,
                parameters=[spec.model_copy(update={"choices": []})],
            )
        )
