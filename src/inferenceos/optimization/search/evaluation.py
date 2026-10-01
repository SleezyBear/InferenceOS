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

from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
    "outofdevicememory",
)
_TIMEOUT_MARKERS = ("timeout", "timed out", "exceeded")

# Statuses that must never be retried: they represent a safety/environment boundary
# rather than a transient measurement contamination.
_TERMINAL_UNSAFE = frozenset(
    {
        CandidateEvaluationStatus.OOM,
        CandidateEvaluationStatus.VRAM_EXCEEDED,
        CandidateEvaluationStatus.TIMEOUT,
    }
)

# State-probe tokens that indicate the target is not in a healthy, stable state. A
# recovery is detected by the before/after probe delta, not by a static token.
_UNHEALTHY_MARKERS = ("health:lost", "health:unrecoverable", "unready")

ContaminationProbe = Callable[[], list[str]]
AppliedConfigurationProbe = Callable[[], RuntimeConfiguration | None]
TelemetryAgeProbe = Callable[[], float | None]


@dataclass(frozen=True)
class MeasurementAttemptRecord:
    """Evidence of one measurement attempt (valid or invalid)."""

    attempt: int
    status: CandidateEvaluationStatus
    valid: bool
    safe: bool
    samples: int
    warmup_runs: int
    decode_tokens_per_sec: float | None
    total_latency_s: float | None
    reasons: list[str] = field(default_factory=list)


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
    # Measurement-robustness provenance (populated by the robust evaluator / attempt).
    attempts: int = 1
    valid_attempts: int = 1
    invalid_attempts: int = 0
    attempt_records: tuple[MeasurementAttemptRecord, ...] = ()
    stabilized: bool = True
    contamination_reasons: list[str] = field(default_factory=list)

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
    """Real evaluator over the existing measurement harness.

    One call to :meth:`measure` is one *attempt*. Contamination signals that make an
    attempt non-authoritative (failed runs, insufficient warmup, excessive primary-metric
    variance, a resident recovery/health change, a configuration mismatch, or paging) are
    recorded as ``INVALID`` rather than being accepted as a valid (possibly degraded)
    performance result. Retrying invalid attempts is the job of
    :class:`RobustPointEvaluator`; this class never retries and never selects a best run.
    """

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
        max_primary_cv: float | None = None,
        contamination_probe: ContaminationProbe | None = None,
        applied_configuration_probe: AppliedConfigurationProbe | None = None,
        telemetry_age_probe: TelemetryAgeProbe | None = None,
        telemetry_max_age_s: float | None = None,
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
        self._max_primary_cv = max_primary_cv
        self._contamination_probe = contamination_probe
        self._applied_configuration_probe = applied_configuration_probe
        self._telemetry_age_probe = telemetry_age_probe
        self._telemetry_max_age_s = telemetry_max_age_s

    def configuration_for(self, layers: int) -> RuntimeConfiguration:
        return self._base.model_copy(
            update={
                "gpu_offload_layers": int(layers),
                "context_length": self._context_length,
            }
        )

    def _contamination_reasons(
        self, measurement: ExperimentMeasurement, *, before: list[str], after: list[str]
    ) -> list[str]:
        reasons: list[str] = []
        if measurement.invalid_runs > 0:
            reasons.append(f"generation_failures={measurement.invalid_runs}")
        if _matches(measurement.notes, ("warm-up run failed", "warm-up failed")):
            reasons.append("warmup_failed")
        if measurement.warmup_runs < self._policy.warmup_repetitions:
            reasons.append(
                f"warmup_runs={measurement.warmup_runs} < required "
                f"{self._policy.warmup_repetitions}"
            )
        if self._max_primary_cv is not None:
            cv = measurement.metrics.get("decode_tokens_per_sec")
            if cv is not None and cv.cv is not None and cv.cv > self._max_primary_cv:
                reasons.append(
                    f"primary_cv={cv.cv:.3f} > max_measure_cv={self._max_primary_cv:.3f}"
                )
        if after != before:
            reasons.append("contamination_probe_changed:" + ",".join(after))
        for token in after:
            lowered = token.lower()
            if any(marker in lowered for marker in _UNHEALTHY_MARKERS):
                reasons.append(f"unhealthy_state:{token}")
        if self._applied_configuration_probe is not None:
            applied = self._applied_configuration_probe()
            expected = self.configuration_for(measurement.configuration.gpu_offload_layers or 0)
            if applied is not None and (
                applied.gpu_offload_layers != expected.gpu_offload_layers
                or applied.context_length != expected.context_length
            ):
                reasons.append(
                    "applied_configuration_mismatch:"
                    f"expected={expected.gpu_offload_layers}/{expected.context_length} "
                    f"applied={applied.gpu_offload_layers}/{applied.context_length}"
                )
        if self._telemetry_age_probe is not None and self._telemetry_max_age_s is not None:
            try:
                age = self._telemetry_age_probe()
            except Exception:  # noqa: BLE001
                age = None
            if age is not None and age > self._telemetry_max_age_s:
                reasons.append(
                    f"telemetry_age_s={age:.1f} > max={self._telemetry_max_age_s:.1f}"
                )
        return reasons

    def measure(self, layers: int) -> PointMeasurement:
        config = self.configuration_for(layers)
        before = _probe(self._contamination_probe)
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
        after = _probe(self._contamination_probe)
        contamination = self._contamination_reasons(
            measurement, before=before, after=after
        )

        reasons: list[str] = list(measurement.notes) + contamination
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
        elif contamination:
            status = CandidateEvaluationStatus.INVALID
            reasons.append("measurement contaminated: " + "; ".join(contamination))
        elif not valid:
            status = CandidateEvaluationStatus.INVALID
            reasons.append(
                f"insufficient valid measurements ({measurement.valid_runs} < "
                f"{self._policy.min_valid_samples})"
            )

        stabilized = (
            status is CandidateEvaluationStatus.OK
            and measurement.warmup_runs >= self._policy.warmup_repetitions
        )
        attempt_status = status
        point = PointMeasurement(
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
            attempts=1,
            valid_attempts=1 if (valid and status is CandidateEvaluationStatus.OK) else 0,
            invalid_attempts=0 if (valid and status is CandidateEvaluationStatus.OK) else 1,
            attempt_records=(
                MeasurementAttemptRecord(
                    attempt=1,
                    status=attempt_status,
                    valid=valid and status is CandidateEvaluationStatus.OK,
                    safe=safe and status is CandidateEvaluationStatus.OK,
                    samples=measurement.valid_runs,
                    warmup_runs=measurement.warmup_runs,
                    decode_tokens_per_sec=_mean(measurement, "decode_tokens_per_sec"),
                    total_latency_s=_mean(measurement, "total_latency_s"),
                    reasons=list(reasons[-4:]),
                ),
            ),
            stabilized=stabilized,
            contamination_reasons=list(contamination),
        )
        return point


def _probe(probe: ContaminationProbe | None) -> list[str]:
    if probe is None:
        return []
    try:
        return [str(item) for item in probe()]
    except Exception as exc:  # noqa: BLE001 - a failed probe is itself a contamination signal
        return [f"probe_error:{type(exc).__name__}"]


class RobustPointEvaluator:
    """Bounded, invalidity-only retries around a :class:`PointEvaluator`.

    Retries are issued **only** when an attempt is invalid/contaminated (or induced a
    retryable paging event). A valid attempt is authoritative even if its performance is
    poor: the loop stops at the first valid attempt, so retries can never be used to
    cherry-pick the fastest result. Every attempt (valid or invalid) is preserved in
    ``attempt_records``; if all attempts are invalid the candidate is ``INVALID`` with
    explicit reasons.
    """

    def __init__(self, inner: PointEvaluator, *, max_attempts: int = 3) -> None:
        self._inner = inner
        self._max_attempts = max(1, int(max_attempts))

    def configuration_for(self, layers: int) -> RuntimeConfiguration:
        return self._inner.configuration_for(layers)

    def measure(self, layers: int) -> PointMeasurement:
        attempts: list[PointMeasurement] = []
        records: list[MeasurementAttemptRecord] = []
        valid_attempt: PointMeasurement | None = None
        for index in range(1, self._max_attempts + 1):
            point = self._inner.measure(layers)
            attempts.append(point)
            records.append(
                MeasurementAttemptRecord(
                    attempt=index,
                    status=point.status,
                    valid=point.valid,
                    safe=point.safe,
                    samples=point.samples,
                    warmup_runs=point.warmup_runs,
                    decode_tokens_per_sec=point.metric_mean("decode_tokens_per_sec"),
                    total_latency_s=point.metric_mean("total_latency_s"),
                    reasons=list(point.reasons[-4:]),
                )
            )
            if point.valid and point.safe and point.status is CandidateEvaluationStatus.OK:
                valid_attempt = point
                break
            if point.status in _TERMINAL_UNSAFE:
                break
            # Otherwise retry: the attempt was invalid/contaminated (or paging).

        base = valid_attempt if valid_attempt is not None else attempts[-1]
        valid_count = sum(
            1 for item in attempts if item.valid and item.safe
            and item.status is CandidateEvaluationStatus.OK
        )
        invalid_count = len(attempts) - valid_count
        return replace(
            base,
            attempts=len(attempts),
            valid_attempts=valid_count,
            invalid_attempts=invalid_count,
            attempt_records=tuple(records),
            stabilized=base.stabilized and valid_count >= 1,
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
