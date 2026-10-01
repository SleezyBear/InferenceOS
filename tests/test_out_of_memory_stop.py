"""A setup that does not fit in memory is not retried run after run."""

from __future__ import annotations

from typing import Any

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.experiment.measurement import is_out_of_memory, measure_configuration
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_model
from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy


class _NoRoomHarness(ScriptedHarness):
    def run_once(self, task: Any, **kwargs: Any) -> Any:
        self.calls.append((task.context_length, None, kwargs["options"].gpu_offload_layers))
        raise RuntimeError(
            "llama-server exited early with code 1: ggml_vulkan: "
            "vk::Device::allocateMemory: ErrorOutOfDeviceMemory"
        )


def test_out_of_memory_stops_the_measurement_at_the_first_failure() -> None:
    harness = _NoRoomHarness(lambda task, options, call: SampleResult())
    measurement = measure_configuration(
        profiler=harness,
        model=sample_model(),
        workload=get_workload("short-short"),
        config=RuntimeConfiguration(context_length=2048, gpu_offload_layers=24),
        policy=build_test_policy(),
        phase="t",
    )
    assert len(harness.calls) == 1
    assert measurement.valid_runs == 0
    assert any("does not fit" in note for note in measurement.notes)


def test_other_failures_are_still_retried() -> None:
    harness = ScriptedHarness(lambda task, options, call: SampleResult(fail=True))
    measure_configuration(
        profiler=harness,
        model=sample_model(),
        workload=get_workload("short-short"),
        config=RuntimeConfiguration(context_length=2048, gpu_offload_layers=24),
        policy=build_test_policy(),
        phase="t",
    )
    assert len(harness.calls) > 1


def test_out_of_memory_messages_are_recognised() -> None:
    assert is_out_of_memory(RuntimeError("unable to allocate Vulkan0 buffer"))
    assert is_out_of_memory(RuntimeError("ErrorOutOfDeviceMemory"))
    assert not is_out_of_memory(RuntimeError("connection reset"))
