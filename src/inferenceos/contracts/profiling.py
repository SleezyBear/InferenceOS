"""Profiling / baseline contracts (Phase 3).

A :class:`ProfilingRun` is a durable, self-contained record of one real measured
inference run: enough hardware/model/workload/runtime identity to interpret the
metrics later, plus explicit metric provenance. :class:`BaselineReport` bundles a
set of runs and their aggregation.

These contracts describe *what happened*; they carry no diagnosis or optimization.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import BackendType, MeasurementMode, MetricQuality
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.observation import Observation
from inferenceos.contracts.workload import WorkloadProfile

PROFILING_SCHEMA_VERSION = "1.0"


class ProfilingConfig(ContractBase):
    """The exact execution configuration used for a profiling run."""

    backend_name: str | None = None
    backend_version: str | None = None
    backend_type: BackendType | None = None
    device: str | None = None
    device_selector: str | None = None
    gpu_layers_offloaded: int | None = None
    gpu_layers_total: int | None = None
    threads: int | None = None
    # The observed *requested* vs *effective* generation thread relationship, recorded
    # explicitly so neither is silently discarded. ``threads`` above is left as the
    # existing strategy input (unchanged); these two are the observed runtime state and
    # ``effective_threads`` is never inferred from CPU topology.
    requested_threads: int | None = None
    effective_threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    use_mmap: bool | None = None
    flash_attention: bool | None = None
    kv_cache_type: str | None = None
    log_verbosity: int | None = None
    ignore_eos: bool | None = None
    context_length: int | None = None
    n_predict: int | None = None
    seed: int | None = None
    temperature: float | None = None
    telemetry_sampling: bool = True
    extra: dict[str, str] = Field(default_factory=dict)


class RunIdentity(ContractBase):
    run_id: str
    profiler_version: str
    schema_version: str = PROFILING_SCHEMA_VERSION
    sequence: int = Field(default=0, ge=0)
    repetition: int = Field(default=0, ge=0)
    mode: MeasurementMode = MeasurementMode.COLD
    warmup: bool = False
    started_at: datetime
    finished_at: datetime | None = None


class RunContext(ContractBase):
    """Everything needed to interpret the run's metrics later."""

    hardware: HardwareProfile
    model: ModelProfile
    workload: WorkloadProfile
    config: ProfilingConfig


class ProfilingRun(ContractBase):
    schema_version: str = PROFILING_SCHEMA_VERSION
    identity: RunIdentity
    context: RunContext
    observation: Observation
    metric_quality: dict[str, MetricQuality] = Field(default_factory=dict)
    notes: str | None = None


class MetricStats(ContractBase):
    metric: str
    count: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    stdev: float | None = None
    min: float | None = None
    max: float | None = None
    cv: float | None = None


class RunGroupStats(ContractBase):
    """Aggregation of repeated runs sharing model + workload + mode + config."""

    model_ref: str
    workload_id: str
    mode: MeasurementMode
    repetitions: int = Field(ge=0)
    metrics: dict[str, MetricStats] = Field(default_factory=dict)


class BaselineReport(ContractBase):
    schema_version: str = PROFILING_SCHEMA_VERSION
    profiler_version: str
    suite_version: str
    generated_at: datetime
    hardware: HardwareProfile
    backend: ProfilingConfig
    workloads: list[WorkloadProfile] = Field(default_factory=list)
    runs: list[ProfilingRun] = Field(default_factory=list)
    groups: list[RunGroupStats] = Field(default_factory=list)
    cold_warm_semantics: str
    resident_warm_supported: bool = False
    limitations: list[str] = Field(default_factory=list)
    notes: str | None = None

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, data: str) -> "BaselineReport":
        return cls.model_validate_json(data)
