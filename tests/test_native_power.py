"""B1: real Windows power/processor-performance probes (skipped off Windows)."""

from __future__ import annotations

import pytest

from inferenceos.contracts.enums import PowerState
from inferenceos.native import windows
from inferenceos.telemetry.system import observe_power_performance, sample_system_state

windows_only = pytest.mark.skipif(not windows.is_windows(), reason="Windows-only probe")


@windows_only
def test_power_status_observes_real_ac_or_battery() -> None:
    status = windows.power_status()
    assert status is not None
    assert status.ac_line in (True, False, None)
    assert status.battery_present in (True, False, None)
    if status.battery_percent is not None:
        assert 0 <= status.battery_percent <= 100


@windows_only
def test_processor_power_returns_coherent_entries() -> None:
    entries = windows.processor_power()
    # The API may be unavailable on some platforms/VMs; an empty list is honest.
    for entry in entries:
        assert entry.max_mhz > 0
        assert entry.current_mhz >= 0
        assert entry.limit_mhz >= 0


@windows_only
def test_observe_power_performance_is_coherent() -> None:
    observation = observe_power_performance()
    assert isinstance(observation.power_state, PowerState)
    if observation.ratio is not None:
        assert 0.0 <= observation.ratio <= 1.0
    if observation.current_mhz is not None:
        assert observation.current_mhz >= 0


@windows_only
def test_system_state_carries_power_and_performance() -> None:
    state = sample_system_state(include_gpu=False)
    assert state.power_state is not PowerState.UNKNOWN
    assert state.cpu_performance_ratio is None or 0.0 <= state.cpu_performance_ratio <= 1.0
