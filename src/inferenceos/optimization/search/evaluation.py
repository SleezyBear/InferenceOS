"""Empirical evaluation of individual candidate configurations.

A :class:`PointEvaluator` applies one candidate value, runs the real measurement
harness (warm-up + repeated generations), and classifies the result. Classification is
mechanical and conservative: an out-of-memory/allocation failure, a timeout, measured
VRAM above the pre-computed safe envelope, or too few valid samples makes the point
unsafe or invalid. No synthetic numbers are used: only metrics the harness measured.

The real evaluator reuses ``experiment.measurement.measure_configuration`` — the same
Phase 3 harness and repetition policy the Phase 7 engine uses — so comparisons are
methodologically identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from inferenceos.contracts.enums import CandidateEvaluationStatus
from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentPolicy,
    RuntimeConfiguration,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import MetricStats
from inferenceos.contracts.search import GpuOffloadResourceEnvelope
from inferenceos.experiment.measurement import (
    MeasurementHarness,
    measure_configuration,
)
from inferenceos.profiling.workloads import BenchmarkWorkload

_OOM_MARKERS = (
    "out of memory",
    "out-of-memory",
    "oom",
    "failed to allocate",
    "cannot allocate",
    "allocation failed",
    "insufficient memory",
)
_TIMEOUT_MARKERS = ("timeout", "timed out", "exceeded")


@dataclass(frozen=True)
class PointMeasurement:
    """One empirically measured candidate configuration."""

    layers: int
    status: CandidateEvaluationStatus
    valid: bool
    safe: bool
    samples: int = 0
    warmup_runs: int = 0
    metrics: dict[str, MetricStats] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    measured_vram_bytes: int | None = None
    measurement: ExperimentMeasurement | None = None

    @property
    def ok(self) -> bool:
        return self.valid and self.safe

    def metric_mean(self, name: str) -> float | None:
        stats = self.metrics.get(name)
        return stats.mean if stats is not None else None


class PointEvaluator(Protocol):
    """Structural interface: measure a single candidate value."""

    def measure(self, layers: int) -> PointMeasurement:
        ...

    def configuration_for(self, layers: int) -> RuntimeConfiguration:
        ...


def _mean(measurement: ExperimentMeasurement, name: str) -> float | None:
    stats = measurement.metrics.get(name)
    return stats.mean if stats is not None else None


def _matches(notes: list[str], markers: tuple[str, ...]) -> bool:
    joined = " ".join(notes).lower()
    return any(marker in joined for marker in markers)


class RuntimeConfigurationPointEvaluator:
    """Real evaluator over the existing measurement harness."""

    def __init__(
        self,
        *,
        profiler: MeasurementHarness,
        model: ModelProfile,
        workload: BenchmarkWorkload,
        experiment_policy: ExperimentPolicy,
        context_length: int,
        envelope: GpuOffloadResourceEnvelope,
        base_configuration: RuntimeConfiguration | None = None,
        seed: int = 1234,
        paging_per_sec_ceiling: float | None = None,
        deadline: float | None = None,
    ) -> None:
        self._profiler = profiler
        self._model = model
        self._workload = workload
        self._policy = experiment_policy
        self._context_length = context_length
        self._envelope = envelope
        self._base = base_configuration or RuntimeConfiguration()
        self._seed = seed
        self._paging_ceiling = paging_per_sec_ceiling
        self._deadline = deadline

    def configuration_for(self, layers: int) -> RuntimeConfiguration:
        return self._base.model_copy(
            update={
                "gpu_offload_layers": int(layers),
                "context_length": self._context_length,
            }
        )

    def measure(self, layers: int) -> PointMeasurement:
        config = self.configuration_for(layers)
        measurement = measure_configuration(
            profiler=self._profiler,
            model=self._model,
            workload=self._workload,
            config=config,
            policy=self._policy,
            phase=f"gpu-offload-search-{layers}",
            seed=self._seed,
            deadline=self._deadline,
        )
        reasons: list[str] = list(measurement.notes)
        measured_vram = _mean(measurement, "gpu_memory_used_bytes")
        vram_ceiling = _vram_ceiling(self._envelope)

        status = CandidateEvaluationStatus.OK
        safe = True
        valid = measurement.valid_runs >= self._policy.min_valid_samples

        if _matches(measurement.notes, _OOM_MARKERS):
            status = CandidateEvaluationStatus.OOM
            safe = False
            valid = False
            reasons.append("runtime reported an out-of-memory/allocation failure")
        elif _matches(measurement.notes, _TIMEOUT_MARKERS):
            status = CandidateEvaluationStatus.TIMEOUT
            safe = False
            valid = False
            reasons.append("candidate measurement exceeded its time budget")
        elif (
            measured_vram is not None
            and vram_ceiling is not None
            and measured_vram > vram_ceiling
        ):
            status = CandidateEvaluationStatus.VRAM_EXCEEDED
            safe = False
            reasons.append(
                f"measured VRAM {measured_vram:.0f} exceeds the safe ceiling "
                f"{vram_ceiling} (total - safety headroom)"
            )
        elif (
            self._paging_ceiling is not None
            and _mean(measurement, "hard_page_faults_per_sec") is not None
            and float(_mean(measurement, "hard_page_faults_per_sec") or 0.0)
            > self._paging_ceiling
        ):
            status = CandidateEvaluationStatus.PAGING
            safe = False
            reasons.append("candidate introduced paging beyond the configured ceiling")
        elif not valid:
            status = CandidateEvaluationStatus.INVALID
            reasons.append(
                f"insufficient valid measurements ({measurement.valid_runs} < "
                f"{self._policy.min_valid_samples})"
            )

        return PointMeasurement(
            layers=int(layers),
            status=status,
            valid=valid and status is CandidateEvaluationStatus.OK,
            safe=safe and status is CandidateEvaluationStatus.OK,
            samples=measurement.valid_runs,
            warmup_runs=measurement.warmup_runs,
            metrics=dict(measurement.metrics),
            reasons=reasons,
            measured_vram_bytes=int(measured_vram) if measured_vram is not None else None,
            measurement=measurement,
        )


def _vram_ceiling(envelope: GpuOffloadResourceEnvelope) -> int | None:
    if envelope.total_vram_bytes is None or envelope.safety_headroom_bytes is None:
        return None
    return max(envelope.total_vram_bytes - envelope.safety_headroom_bytes, 0)


class MappingPointEvaluator:
    """Deterministic evaluator over a fixed value->measurement map (tests/evidence)."""

    def __init__(
        self,
        measurements: dict[int, PointMeasurement],
        *,
        default: PointMeasurement | None = None,
        context_length: int | None = None,
        base_configuration: RuntimeConfiguration | None = None,
    ) -> None:
        self._measurements = dict(measurements)
        self._default = default
        self._context_length = context_length
        self._base = base_configuration or RuntimeConfiguration()
        self.calls: list[int] = []

    def configuration_for(self, layers: int) -> RuntimeConfiguration:
        return self._base.model_copy(
            update={"gpu_offload_layers": int(layers), "context_length": self._context_length}
        )

    def measure(self, layers: int) -> PointMeasurement:
        self.calls.append(int(layers))
        if int(layers) in self._measurements:
            return self._measurements[int(layers)]
        if self._default is not None:
            return self._default
        raise KeyError(f"no scripted measurement for layers={layers}")
