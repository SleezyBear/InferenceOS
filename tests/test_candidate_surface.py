"""R1: the candidate registry accurately describes genuinely executable surfaces."""

from __future__ import annotations

from inferenceos.contracts.enums import BottleneckKind
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.experiment.quality_policy import _SEMANTIC_ACTIONS
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.remediation import _gpu_underutilization_diagnosis


def test_increase_gpu_offload_action_is_registered_and_executable() -> None:
    registry = build_default_registry()
    action = registry.get("runtime.increase_gpu_offload_layers")
    assert BottleneckKind.GPU_UNDERUTILIZATION in action.supported_bottlenecks
    assert RuntimeConfigurationExecutor().supports(action.action_id)


def test_gpu_underutilization_generates_executable_increase_candidate() -> None:
    diagnosis = _gpu_underutilization_diagnosis(object())
    plan = generate_candidates(diagnosis, policy=OptimizationPolicy())
    candidate = next(
        (c for c in plan.candidates if c.action_id == "runtime.increase_gpu_offload_layers"),
        None,
    )
    assert candidate is not None
    assert candidate.executable is True
    target = candidate.parameters["gpu_offload_layers"]
    assert isinstance(target, int)
    assert target > 14 and target <= 28

    executor = RuntimeConfigurationExecutor()
    executor.set_configuration(RuntimeConfiguration(context_length=2048, gpu_offload_layers=14))
    action = build_default_registry().get(candidate.action_id)
    applied = executor.apply(action, candidate.parameters)
    assert applied.applied is True
    assert executor.current().gpu_offload_layers == target
    restored = executor.restore(RuntimeConfiguration(context_length=2048, gpu_offload_layers=14))
    assert restored.verified is True


def test_quality_policy_has_no_unregistered_action_references() -> None:
    registry = build_default_registry()
    unregistered = [action_id for action_id in _SEMANTIC_ACTIONS if action_id not in registry]
    assert unregistered == []
