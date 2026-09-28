"""Phase 7 thread-integration regression tests.

GAP 1: the Phase 7 baseline reconstruction must preserve the requested/effective thread
distinction (requested runtime configuration, never substituting the observed effective
value).

GAP 2: the resident Phase 7 measurement path must expose the target
``process_cpu_percent`` using the existing psutil CPU-time-delta telemetry (no second
measurement mechanism).
"""

from __future__ import annotations

import os
import time

from inferenceos.backends.process_control import PeakRssAccumulator
from inferenceos.contracts.profiling import ProfilingConfig
from inferenceos.experiment.executor import apply_to_configuration
from inferenceos.experiment.service import baseline_configuration_from
from inferenceos.interfaces.backend_adapter import GenerationResult
from inferenceos.profiling.runner import _build_metrics
from inferenceos.selftest import sample_model


# --------------------------------------------------------------------------- #
# GAP 1 — truthful Phase 7 baseline reconstruction                               #
# --------------------------------------------------------------------------- #
def test_baseline_uses_requested_when_threads_field_is_unset() -> None:
    config = ProfilingConfig(
        threads=None, requested_threads=12, effective_threads=12,
        context_length=4096, gpu_layers_offloaded=0,
    )
    baseline = baseline_configuration_from(config, sample_model())
    assert baseline.threads == 12


def test_baseline_preserves_backend_selected_default() -> None:
    # requested unset + backend selected 6 -> the baseline stays "unset" (not explicit 6).
    config = ProfilingConfig(
        threads=None, requested_threads=None, effective_threads=6,
        context_length=4096, gpu_layers_offloaded=0,
    )
    baseline = baseline_configuration_from(config, sample_model())
    assert baseline.threads is None


def test_baseline_unknown_effective_fails_safe() -> None:
    requested_known = ProfilingConfig(
        threads=None, requested_threads=12, effective_threads=None,
        context_length=4096, gpu_layers_offloaded=0,
    )
    assert baseline_configuration_from(requested_known, sample_model()).threads == 12
    both_unknown = ProfilingConfig(
        threads=None, requested_threads=None, effective_threads=None,
        context_length=4096, gpu_layers_offloaded=0,
    )
    assert baseline_configuration_from(both_unknown, sample_model()).threads is None


def test_baseline_legacy_threads_field_still_wins() -> None:
    config = ProfilingConfig(
        threads=12, requested_threads=None, effective_threads=None,
        context_length=4096, gpu_layers_offloaded=0,
    )
    assert baseline_configuration_from(config, sample_model()).threads == 12


def test_baseline_non_thread_parameters_unchanged() -> None:
    config = ProfilingConfig(
        threads=None, requested_threads=12, effective_threads=99,
        context_length=4096, gpu_layers_offloaded=8, batch_size=512, ubatch_size=256,
        use_mmap=True, device_selector="vulkan0",
    )
    baseline = baseline_configuration_from(config, sample_model())
    assert baseline.threads == 12
    assert baseline.context_length == 4096
    assert baseline.gpu_offload_layers == 8
    assert baseline.batch_size == 512
    assert baseline.ubatch_size == 256
    assert baseline.use_mmap is True
    assert baseline.device == "vulkan0"


def test_baseline_reconstructs_restorable_thread_configuration() -> None:
    config = ProfilingConfig(
        threads=None, requested_threads=12, effective_threads=12,
        context_length=4096, gpu_layers_offloaded=0,
    )
    baseline = baseline_configuration_from(config, sample_model())
    candidate = apply_to_configuration("runtime.reduce_threads", {"threads": 6}, baseline)
    assert candidate.threads == 6
    restored = apply_to_configuration(
        "runtime.reduce_threads", {"threads": baseline.threads}, candidate
    )
    assert restored == baseline
    assert restored.threads == 12


# --------------------------------------------------------------------------- #
# GAP 2 — resident process_cpu_percent telemetry                                 #
# --------------------------------------------------------------------------- #
def test_peak_cpu_accumulator_observes_process_cpu() -> None:
    accumulator = PeakRssAccumulator()
    pid = os.getpid()
    accumulator.observe(pid)
    deadline = time.perf_counter() + 0.3
    value = 0
    while time.perf_counter() < deadline:
        value = (value * 1103515245 + 12345) & 0x7FFFFFFF
    accumulator.observe(pid)
    assert accumulator.peak_process_cpu_percent is not None
    assert accumulator.peak_process_cpu_percent > 0.0


def test_peak_cpu_requires_two_samples() -> None:
    accumulator = PeakRssAccumulator()
    accumulator.observe(os.getpid())
    assert accumulator.peak_process_cpu_percent is None


def test_peak_cpu_unavailable_process_is_safe() -> None:
    accumulator = PeakRssAccumulator()
    accumulator.observe(2_147_483_000)  # never a live pid
    assert accumulator.peak_process_cpu_percent is None
    assert accumulator.peak_rss_bytes is None


def test_phase7_metrics_map_process_cpu_from_generation_result() -> None:
    metrics = _build_metrics(
        result=GenerationResult(text="ok", peak_process_cpu_percent=123.4),
        samples=[],
        telemetry_sampling=False,
        system_cpu=10.0,
        memory_after=None,
        wall_s=1.0,
    )
    assert metrics.process_cpu_percent == 123.4


def test_phase7_metrics_unavailable_cpu_stays_none() -> None:
    metrics = _build_metrics(
        result=GenerationResult(text="ok"),
        samples=[],
        telemetry_sampling=False,
        system_cpu=10.0,
        memory_after=None,
        wall_s=1.0,
    )
    assert metrics.process_cpu_percent is None


def test_phase7_metrics_other_metrics_unchanged() -> None:
    metrics = _build_metrics(
        result=GenerationResult(
            text="ok",
            decode_tokens_per_sec=20.0,
            total_s=2.0,
            peak_rss_bytes=1000,
            final_rss_bytes=900,
            peak_process_cpu_percent=200.0,
        ),
        samples=[],
        telemetry_sampling=False,
        system_cpu=10.0,
        memory_after=None,
        wall_s=1.0,
    )
    assert metrics.decode_tokens_per_sec == 20.0
    assert metrics.total_latency_s == 2.0
    assert metrics.peak_rss_bytes == 1000
    assert metrics.final_rss_bytes == 900
    assert metrics.process_cpu_percent == 200.0

