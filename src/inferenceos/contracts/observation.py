"""Observation contracts: one measured (or observed) real inference run."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import BackendType, MeasurementMode, ObservationSource
from inferenceos.contracts.system_state import SystemState


class RunMetrics(ContractBase):
    model_load_time_s: float | None = None
    ttft_s: float | None = None
    prefill_latency_s: float | None = None
    prompt_tokens_per_sec: float | None = None
    decode_tokens_per_sec: float | None = None
    total_latency_s: float | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    peak_rss_bytes: int | None = None
    gpu_utilization_percent: float | None = None
    hard_page_faults: int | None = None


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
