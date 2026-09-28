"""Measurement-robustness tests for the GPU-offload search.

Covers: contamination detection (variance, failed runs, warm-up, applied-config
mismatch, health/probe change, telemetry age, paging, OOM), bounded invalidity-only
retries, anti-cherry-picking, attempt evidence, and contract propagation. Existing
search/decision/resource semantics are asserted unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.contracts.enums import CandidateEvaluationStatus, MeasurementMode
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.profiling import ProfilingRun
from inferenceos.contracts.search import GpuOffloadResourceEnvelope
from inferenceos.optimization.search.evaluation import (
    PointMeasurement,
    RobustPointEvaluator,
    RuntimeConfigurationPointEvaluator,
)
from inferenceos.optimization.search.policy import SearchPolicy
from inferenceos.optimization.search.runner import run_gpu_offload_search
from inferenceos.optimization.search.sampling import sample_layers
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.statistics import summarize
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_model
from tests.diagnosis_fixtures import make_run
from tests.experiment_helpers import build_test_policy

_OK = CandidateEvaluationStatus.OK
_INVALID = CandidateEvaluationStatus.INVALID
_OOM = CandidateEvaluationStatus.OOM
_PAGING = CandidateEvaluationStatus.PAGING


def _pm(
    layers: int,
    *,
    status: CandidateEvaluationStatus = _OK,
    valid: bool = True,
    safe: bool = True,
    samples: int = 2,
    decode: float = 20.0,
    latency: float = 2.0,
    reasons: list[str] | None = None,
) -> PointMeasurement:
    return PointMeasurement(
        layers=layers,
        status=status,
        valid=valid,
        safe=safe,
        samples=samples,
        warmup_runs=1,
        metrics={
            "decode_tokens_per_sec": summarize("decode_tokens_per_sec", [decode, decode]),
            "total_latency_s": summarize("total_latency_s", [latency, latency]),
        },
        reasons=list(reasons or []),
    )


def _envelope(*, feasible_max: int = 18) -> GpuOffloadResourceEnvelope:
    return GpuOffloadResourceEnvelope(
        sufficient=True,
        total_vram_bytes=4 * 1024**3,
        used_vram_bytes=2 * 1024**3,
        safety_headroom_bytes=600 * 1024**2,
        current_layers=8,
        total_layers=28,
        feasible_min=8,
        feasible_max=feasible_max,
        context_length=2048,
    )


class _Harness:
    """Scripted measurement harness returning deterministic runs."""

    def __init__(
        self,
        measured: list[float],
        *,
        warmup_value: float = 20.0,
        fail_measured: int = 0,
        fail_warmup: bool = False,
        paging: float = 0.0,
        vram: int = 1024**3,
        oom: bool = False,
    ) -> None:
        self.measured = list(measured)
        self.warmup_value = warmup_value
        self.fail_measured = fail_measured
        self.fail_warmup = fail_warmup
        self.paging = paging
        self.vram = vram
        self.oom = oom
        self._index = 0

    def run_once(
        self,
        task: ProfileTask,
        *,
        mode: MeasurementMode,
        sequence: int,
        repetition: int,
        warmup: bool,
        options: ProfilerOptions,
    ) -> ProfilingRun:
        if self.oom:
            raise RuntimeError("failed to allocate Vulkan buffer: out of memory")
        if warmup:
            if self.fail_warmup:
                raise RuntimeError("warm-up failed")
            value = self.warmup_value
        else:
            if self.fail_measured > 0:
                self.fail_measured -= 1
                raise RuntimeError("measured run failed")
            value = self.measured[self._index % len(self.measured)]
            self._index += 1
        return make_run(
            model_name="robust",
            workload_id="short-short",
            metrics={
                "decode_tokens_per_sec": value,
                "total_latency_s": 2.0,
                "gpu_memory_used_bytes": self.vram,
                "hard_page_faults_per_sec": self.paging,
            },
        )


def _evaluator(harness: _Harness, **kwargs: object) -> RuntimeConfigurationPointEvaluator:
    return RuntimeConfigurationPointEvaluator(
        profiler=harness,
        model=sample_model(),
        workload=get_workload("short-short"),
        experiment_policy=build_test_policy(),
        context_length=2048,
        envelope=_envelope(),
        base_configuration=RuntimeConfiguration(context_length=2048, gpu_offload_layers=8),
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Contamination detection (inner evaluator)                                    #
# --------------------------------------------------------------------------- #
def test_clean_measurement_is_valid_and_stabilized() -> None:
    point = _evaluator(_Harness([20.0, 20.0]), max_primary_cv=0.10).measure(10)
    assert point.status is _OK
    assert point.ok is True
    assert point.stabilized is True
    assert point.contamination_reasons == []
    assert point.attempts == 1 and point.valid_attempts == 1


def test_high_variance_measurement_is_invalid() -> None:
    point = _evaluator(_Harness([10.0, 30.0]), max_primary_cv=0.10).measure(16)
    assert point.status is _INVALID
    assert point.ok is False
    assert any("primary_cv" in reason for reason in point.contamination_reasons)


def test_generation_failure_invalidates_attempt() -> None:
    point = _evaluator(_Harness([20.0], fail_measured=2), max_primary_cv=0.10).measure(16)
    assert point.status is _INVALID
    assert any("generation_failures" in reason for reason in point.contamination_reasons)


def test_warmup_failure_invalidates_attempt() -> None:
    point = _evaluator(_Harness([20.0]), max_primary_cv=0.10).measure(10)
    assert point.status is _OK
    failed = _evaluator(_Harness([20.0], fail_warmup=True), max_primary_cv=0.10).measure(10)
    assert failed.status is _INVALID
    assert any("warmup" in reason for reason in failed.contamination_reasons)


def test_applied_configuration_mismatch_invalidates_attempt() -> None:
    point = _evaluator(
        _Harness([20.0, 20.0]),
        max_primary_cv=0.10,
        applied_configuration_probe=lambda: RuntimeConfiguration(
            context_length=2048, gpu_offload_layers=99
        ),
    ).measure(16)
    assert point.status is _INVALID
    assert any("applied_configuration_mismatch" in r for r in point.contamination_reasons)


def test_contamination_probe_change_invalidates_attempt() -> None:
    calls = {"n": 0}

    def probe() -> list[str]:
        calls["n"] += 1
        return ["health:healthy", f"recovery:{0 if calls['n'] == 1 else 1}"]

    point = _evaluator(_Harness([20.0, 20.0]), max_primary_cv=0.10, contamination_probe=probe).measure(16)
    assert point.status is _INVALID
    assert any("contamination_probe_changed" in r for r in point.contamination_reasons)


def test_telemetry_staleness_invalidates_attempt() -> None:
    point = _evaluator(
        _Harness([20.0, 20.0]),
        max_primary_cv=0.10,
        telemetry_age_probe=lambda: 100.0,
        telemetry_max_age_s=30.0,
    ).measure(16)
    assert point.status is _INVALID
    assert any("telemetry_age_s" in r for r in point.contamination_reasons)


def test_paging_ceiling_is_unsafe_paging() -> None:
    point = _evaluator(_Harness([20.0, 20.0], paging=100.0), paging_per_sec_ceiling=10.0).measure(16)
    assert point.status is _PAGING
    assert point.safe is False


def test_oom_is_terminal_unsafe() -> None:
    point = _evaluator(_Harness([20.0], oom=True)).measure(16)
    assert point.status is _OOM
    assert point.safe is False


# --------------------------------------------------------------------------- #
# Bounded, invalidity-only retries                                             #
# --------------------------------------------------------------------------- #
class _ScriptedInner:
    def __init__(self, scripted: dict[int, list[PointMeasurement]]) -> None:
        self._scripted = {k: list(v) for k, v in scripted.items()}
        self.calls: list[int] = []

    def configuration_for(self, layers: int) -> RuntimeConfiguration:
        return RuntimeConfiguration(context_length=2048, gpu_offload_layers=layers)

    def measure(self, layers: int) -> PointMeasurement:
        self.calls.append(layers)
        seq = self._scripted[layers]
        return seq.pop(0) if len(seq) > 1 else seq[0]


def test_clean_first_attempt_is_not_retried() -> None:
    inner = _ScriptedInner({16: [_pm(16, decode=22.0)]})
    result = RobustPointEvaluator(inner, max_attempts=3).measure(16)
    assert inner.calls == [16]
    assert result.attempts == 1 and result.valid_attempts == 1
    assert result.valid is True


def test_invalid_first_attempt_is_retried_then_valid() -> None:
    inner = _ScriptedInner({16: [_pm(16, status=_INVALID, valid=False, reasons=["primary_cv"]), _pm(16, decode=23.0)]})
    result = RobustPointEvaluator(inner, max_attempts=3).measure(16)
    assert inner.calls == [16, 16]
    assert result.attempts == 2
    assert result.valid_attempts == 1 and result.invalid_attempts == 1
    assert result.valid is True
    assert len(result.attempt_records) == 2
    assert result.attempt_records[0].valid is False
    assert result.attempt_records[1].valid is True


def test_all_attempts_invalid_yields_invalid() -> None:
    inner = _ScriptedInner(
        {16: [_pm(16, status=_INVALID, valid=False), _pm(16, status=_INVALID, valid=False), _pm(16, status=_INVALID, valid=False)]}
    )
    result = RobustPointEvaluator(inner, max_attempts=3).measure(16)
    assert inner.calls == [16, 16, 16]
    assert result.valid is False and result.safe is True
    assert result.attempts == 3 and result.valid_attempts == 0 and result.invalid_attempts == 3


def test_valid_poor_result_is_not_retried() -> None:
    inner = _ScriptedInner({16: [_pm(16, decode=5.0), _pm(16, decode=99.0)]})
    result = RobustPointEvaluator(inner, max_attempts=3).measure(16)
    assert inner.calls == [16]  # a valid-but-poor result is authoritative, never retried
    assert result.metric_mean("decode_tokens_per_sec") == 5.0


def test_retry_cannot_select_the_fastest_attempt() -> None:
    # First valid attempt is slow; a later (better) attempt must never be reached.
    inner = _ScriptedInner({16: [_pm(16, decode=8.0), _pm(16, decode=40.0)]})
    result = RobustPointEvaluator(inner, max_attempts=3).measure(16)
    assert inner.calls == [16]
    assert result.metric_mean("decode_tokens_per_sec") == 8.0


def test_terminal_unsafe_attempt_is_not_retried() -> None:
    inner = _ScriptedInner({16: [_pm(16, status=_OOM, valid=False, safe=False)]})
    result = RobustPointEvaluator(inner, max_attempts=3).measure(16)
    assert inner.calls == [16]
    assert result.status is _OOM and result.safe is False


def test_paging_attempt_is_retryable() -> None:
    inner = _ScriptedInner({16: [_pm(16, status=_PAGING, valid=False, safe=False), _pm(16, decode=23.0)]})
    result = RobustPointEvaluator(inner, max_attempts=3).measure(16)
    assert inner.calls == [16, 16]
    assert result.valid is True and result.valid_attempts == 1


def test_retry_budget_is_deterministic_and_bounded() -> None:
    inner = _ScriptedInner(
        {16: [_pm(16, status=_INVALID, valid=False) for _ in range(5)]}
    )
    result = RobustPointEvaluator(inner, max_attempts=2).measure(16)
    assert inner.calls == [16, 16]
    assert result.attempts == 2


def test_invalid_attempts_are_preserved_in_evidence() -> None:
    inner = _ScriptedInner(
        {16: [_pm(16, status=_INVALID, valid=False, reasons=["primary_cv=0.5"]), _pm(16, decode=22.0)]}
    )
    result = RobustPointEvaluator(inner, max_attempts=3).measure(16)
    assert len(result.attempt_records) == result.attempts == 2
    assert "primary_cv=0.5" in result.attempt_records[0].reasons


# --------------------------------------------------------------------------- #
# Runner contract: attempt evidence is propagated                              #
# --------------------------------------------------------------------------- #
def test_runner_propagates_attempt_evidence() -> None:
    baseline = _pm(8, decode=20.0)
    candidate = _pm(12, decode=24.0)
    inner = _ScriptedInner(
        {
            8: [baseline],
            12: [_pm(12, status=_INVALID, valid=False, reasons=["primary_cv=0.4"]), candidate],
        }
    )
    evaluator = RobustPointEvaluator(inner, max_attempts=3)
    result = run_gpu_offload_search(
        envelope=_envelope(feasible_max=12),
        evaluator=evaluator,
        search_policy=SearchPolicy(sample_step_layers=4, max_samples=3),
        experiment_policy=build_test_policy(),
    )
    point = next(p for p in result.points if p.layers == 12)
    assert point.attempts == 2
    assert point.invalid_attempts == 1 and point.valid_attempts == 1
    assert len(point.attempt_records) == 2
    assert point.attempt_records[0].valid is False
    assert result.selection.selected_layers == 12


def test_sampling_is_unchanged_by_robustness() -> None:
    # Robustness is a measurement concern only; the sampling plan is untouched.
    assert sample_layers(8, 28, SearchPolicy()) == [8, 12, 16, 20, 24, 28]
