"""Observation contracts: one measured (or observed) real inference run."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import BackendType, MeasurementMode, ObservationSource
from inferenceos.contracts.system_state import SystemState


class RunMetrics(ContractBase):
    """One real inference run's measured metrics.

    Units: durations in seconds, throughput in tokens/second, memory in bytes,
    utilization in percent, page faults per second.
    """

    model_load_time_s: float | None = None
    ttft_s: float | None = None
    prefill_latency_s: float | None = None
    decode_latency_s: float | None = None
    prompt_tokens_per_sec: float | None = None
    decode_tokens_per_sec: float | None = None
    total_latency_s: float | None = None
    process_wall_s: float | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    total_tokens: int | None = None
    peak_rss_bytes: int | None = None
    # Final RSS at the end of the measured interval, when distinguishable from the
    # peak; ``None`` when the process exits before a final sample can be taken.
    final_rss_bytes: int | None = None
    process_cpu_percent: float | None = None
    system_cpu_percent: float | None = None
    # CPU efficiency / contention telemetry over the measured generation interval. Lower
    # is better for all of these; ``None`` means unknown (never fabricated). ``*_per_token``
    # are normalized by the generated-token count and are ``None`` when that count is
    # missing or zero.
    cpu_seconds: float | None = None
    cpu_seconds_per_token: float | None = None
    context_switches_voluntary: int | None = None
    context_switches_involuntary: int | None = None
    context_switches_total: int | None = None
    context_switches_per_token: float | None = None
    gpu_utilization_percent: float | None = None
    gpu_memory_used_bytes: int | None = None
    gpu_memory_total_bytes: int | None = None
    hard_page_faults: int | None = None
    hard_page_faults_per_sec: float | None = None
    total_ram_bytes: int | None = None
    available_ram_bytes: int | None = None
    committed_bytes: int | None = None
    commit_limit_bytes: int | None = None
    # Average processor speed during the run as a share of its rated speed (Windows
    # "% Processor Performance": above 100 when boosting, falls when it throttles from
    # heat). ``None`` when not sampled.
    cpu_performance_percent: float | None = None


class Observation(ContractBase):
    observation_id: str
    captured_at: datetime
    source: ObservationSource = ObservationSource.MANUAL
    mode: MeasurementMode = MeasurementMode.COLD
    backend: BackendType | None = None
    device: str | None = None
    model_ref: str | None = None
    workload_ref: str | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    system_state: SystemState | None = None
    generated_text: str | None = None
    notes: str | None = None
