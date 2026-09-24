"""Controlled execution abstraction: exact application, unsupported, restoration."""

from __future__ import annotations

from inferenceos.contracts.enums import RestorationStatus
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.registry import ActionRegistry
from inferenceos.experiment.executor import RuntimeConfigurationExecutor


def _registry() -> ActionRegistry:
    return build_default_registry()


def test_apply_sets_exact_parameter() -> None:
    executor = RuntimeConfigurationExecutor()
    executor.set_configuration(RuntimeConfiguration(context_length=2048))
    action = _registry().get("runtime.right_size_context")
    result = executor.apply(action, {"context_length": 1024})
    assert result.applied is True
    assert result.supported is True
    assert executor.current().context_length == 1024
    assert result.changes[0].prior_value == 2048
    assert result.changes[0].applied_value == 1024


def test_apply_mmap_exact_parameter() -> None:
    executor = RuntimeConfigurationExecutor()
    executor.set_configuration(RuntimeConfiguration(context_length=2048, use_mmap=False))
    action = _registry().get("runtime.enable_mmap")
    result = executor.apply(action, {"use_mmap": True})
    assert result.applied is True
    assert executor.current().use_mmap is True


def test_unsupported_action_not_applied() -> None:
    executor = RuntimeConfigurationExecutor()
    action = _registry().get("system.increase_pagefile")
    result = executor.apply(action, {"increase_bytes": 1024})
    assert result.applied is False
    assert result.supported is False
    assert "no safe runtime surface" in result.reason


def test_external_action_not_applied() -> None:
    executor = RuntimeConfigurationExecutor()
    action = _registry().get("external.suspend_process")
    result = executor.apply(action, {"process_name": "python.exe"})
    assert result.applied is False
    assert result.supported is False


def test_invalid_parameter_rejected() -> None:
    executor = RuntimeConfigurationExecutor()
    action = _registry().get("runtime.right_size_context")
    result = executor.apply(action, {"context_length": -5})
    assert result.applied is False
    assert "below minimum" in result.reason


def test_undeclared_parameter_rejected() -> None:
    executor = RuntimeConfigurationExecutor()
    action = _registry().get("runtime.right_size_context")
    result = executor.apply(action, {"cmd": "Remove-Item -Recurse"})
    assert result.applied is False
    assert "undeclared parameter" in result.reason


def test_restore_returns_to_exact_baseline() -> None:
    executor = RuntimeConfigurationExecutor()
    baseline = RuntimeConfiguration(context_length=2048, use_mmap=False)
    executor.set_configuration(baseline)
    action = _registry().get("runtime.right_size_context")
    executor.apply(action, {"context_length": 1024})
    restoration = executor.restore(baseline)
    assert restoration.status is RestorationStatus.RESTORED
    assert restoration.verified is True
    assert restoration.restored_configuration == baseline
    assert executor.current() == baseline


def test_restore_from_unmutated_state_is_verified() -> None:
    executor = RuntimeConfigurationExecutor()
    baseline = RuntimeConfiguration(context_length=2048)
    executor.set_configuration(baseline)
    restoration = executor.restore(baseline)
    assert restoration.verified is True
    assert restoration.changes == []
