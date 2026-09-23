"""Pure parsing/aggregation for the persistent counter sampler."""

from __future__ import annotations

from inferenceos.native.windows import CounterSample
from inferenceos.telemetry.counters import (
    _parse_line,
    gpu_adapter_memory_peaks,
    gpu_engine_peaks,
    max_gpu_adapter_memory_bytes,
    max_gpu_engine_percent,
    max_value,
)

ENGINE = r"\\host\gpu engine(pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d)\utilization percentage"
COMPUTE = r"\\host\gpu engine(pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_compute)\utilization percentage"
ADAPTER = r"\\host\gpu adapter memory(luid_0x00000000_0x00009383_phys_0)\dedicated usage"
FAULT = r"\\host\memory\pages input/sec"


def test_parse_line() -> None:
    sample = _parse_line("path|instance|12.5")
    assert sample is not None
    assert sample.path == "path" and sample.instance == "instance" and sample.value == 12.5
    assert _parse_line("HB") is None
    assert _parse_line("a|b|notfloat") is None


def test_max_value() -> None:
    samples = [CounterSample(FAULT, None, 3.0), CounterSample(FAULT, None, 9.0)]
    assert max_value(samples, "pages input") == 9.0
    assert max_value(samples, "absent") is None


def test_gpu_engine_peaks_and_helpers() -> None:
    samples = [
        CounterSample(ENGINE, "pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d", 10.0),
        CounterSample(ENGINE, "pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_3d", 40.0),
        CounterSample(COMPUTE, "pid_1_luid_0x00000000_0x00009383_phys_0_eng_0_engtype_compute", 80.0),
    ]
    peaks = gpu_engine_peaks(samples)
    assert ("00000000:00009383", "3d", 40.0) in peaks
    assert ("00000000:00009383", "compute", 80.0) in peaks
    assert max_gpu_engine_percent(samples, "compute") == 80.0
    assert max_gpu_engine_percent(samples) == 80.0


def test_gpu_adapter_memory_peaks() -> None:
    samples = [
        CounterSample(ADAPTER, "luid_0x00000000_0x00009383_phys_0", 1_000_000.0),
        CounterSample(ADAPTER, "luid_0x00000000_0x00009383_phys_0", 2_000_000.0),
    ]
    peaks = gpu_adapter_memory_peaks(samples)
    assert peaks == [("00000000:00009383", 2_000_000)]
    assert max_gpu_adapter_memory_bytes(samples) == 2_000_000
