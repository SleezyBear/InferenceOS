"""Live telemetry parsing/normalization and unavailable-signal handling."""

from __future__ import annotations

from inferenceos.native.windows import CounterSample
from inferenceos.telemetry.gpu import _parse_line
from inferenceos.telemetry.system import (
    aggregate_gpu_samples,
    compute_memory_pressure,
    hard_faults_from_samples,
    parse_gpu_adapter_instance,
    parse_gpu_engine_instance,
)

ENGINE_PATH = r"\\host\gpu engine(pid_4242_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d)\utilization percentage"
ADAPTER_PATH = r"\\host\gpu adapter memory(luid_0x00000000_0x00009383_phys_0)\dedicated usage"
FAULT_PATH = r"\\host\memory\pages input/sec"


def test_compute_memory_pressure_commit_ratio() -> None:
    assert compute_memory_pressure(16_000, 8_000, 12_000, 18_000) == 12_000 / 18_000


def test_compute_memory_pressure_physical_fallback() -> None:
    assert compute_memory_pressure(16_000, 4_000, None, None) == 0.75


def test_compute_memory_pressure_clamps() -> None:
    assert compute_memory_pressure(16_000, 8_000, 20_000, 18_000) == 1.0
    assert compute_memory_pressure(0, 0, None, None) == 0.0


def test_parse_gpu_engine_instance() -> None:
    parsed = parse_gpu_engine_instance(
        "pid_4242_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d"
    )
    assert parsed == ("00000000:00009383", 0, "3d")
    assert parse_gpu_engine_instance("garbage") is None


def test_parse_gpu_adapter_instance() -> None:
    assert (
        parse_gpu_adapter_instance("luid_0x00000000_0x00009383_phys_0")
        == "00000000:00009383"
    )
    assert parse_gpu_adapter_instance("garbage") is None


def test_aggregate_gpu_samples_sums_and_caps() -> None:
    samples = [
        CounterSample(ENGINE_PATH, "pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d", 60.0),
        CounterSample(ENGINE_PATH, "pid_2_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d", 70.0),
        CounterSample(ADAPTER_PATH, "luid_0x00000000_0x00009383_phys_0", 1_800_000_000.0),
    ]
    result = aggregate_gpu_samples(samples)
    assert len(result.engines) == 1
    assert result.engines[0].engine == "3d"
    assert result.engines[0].device_index == 0
    assert result.engines[0].utilization_percent == 100.0
    assert result.memory[0].memory_used_bytes == 1_800_000_000
    assert result.memory[0].memory_total_bytes is None


def test_aggregate_gpu_samples_assigns_device_indices_by_luid() -> None:
    samples = [
        CounterSample(ENGINE_PATH, "pid_1_luid_0x00000000_0x0000a937_phys_0_eng_0_engtype_3d", 5.0),
        CounterSample(ENGINE_PATH, "pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d", 10.0),
    ]
    result = aggregate_gpu_samples(samples)
    luid_order = [engine.device_index for engine in result.engines]
    assert luid_order == [0, 1]


def test_aggregate_gpu_samples_empty_is_empty() -> None:
    result = aggregate_gpu_samples([])
    assert result.engines == []
    assert result.memory == []


def test_hard_faults_from_samples() -> None:
    samples = [
        CounterSample(FAULT_PATH, None, 12.5),
        CounterSample(ENGINE_PATH, "x", 99.0),
    ]
    assert hard_faults_from_samples(samples) == 12.5
    assert hard_faults_from_samples([]) == 0.0


def test_sampler_heartbeat_is_not_parsed_as_a_sample() -> None:
    assert _parse_line("HB") is None


# --------------------------------------------------------------------------- #
# Step 2 bug: graphics card use was over-counted when several counter samples #
# were added together.                                                        #
# --------------------------------------------------------------------------- #
def test_aggregate_gpu_samples_repeated_sweeps_are_not_summed() -> None:
    # The runtime watcher passes every sample since its last check, so the same
    # counter can appear once per sweep. Three sweeps at 40% are 40%, not 100%.
    instance = "pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d"
    samples = [CounterSample(ENGINE_PATH, instance, 40.0) for _ in range(3)]
    result = aggregate_gpu_samples(samples)
    assert len(result.engines) == 1
    assert result.engines[0].utilization_percent == 40.0


def test_aggregate_gpu_samples_repeated_sweeps_average_per_process() -> None:
    a = "pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d"
    b = "pid_2_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d"
    samples = [
        CounterSample(ENGINE_PATH, a, 20.0),
        CounterSample(ENGINE_PATH, b, 10.0),
        CounterSample(ENGINE_PATH, a, 30.0),
        CounterSample(ENGINE_PATH, b, 10.0),
    ]
    result = aggregate_gpu_samples(samples)
    # Two processes sharing one engine still add up: avg(20,30) + avg(10,10) = 35.
    assert result.engines[0].utilization_percent == 35.0


def test_aggregate_gpu_samples_separate_engines_of_one_kind_are_not_summed() -> None:
    # Two separate engines of the same kind are reported as the busiest one
    # (as Task Manager does), not their sum.
    samples = [
        CounterSample(ENGINE_PATH, "pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_compute", 30.0),
        CounterSample(ENGINE_PATH, "pid_1_luid_0x00000000_0x00009383_phys_0_eng_1_engtype_compute", 50.0),
    ]
    result = aggregate_gpu_samples(samples)
    assert len(result.engines) == 1
    assert result.engines[0].engine == "compute"
    assert result.engines[0].utilization_percent == 50.0
