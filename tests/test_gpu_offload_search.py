"""Deterministic tests for the resource-aware GPU-offload search.

Covers resource constraints, feasible-range construction, deterministic sampling,
candidate evaluation (warm-up/repeat/validity/OOM/VRAM), multi-objective selection,
and the context/KV dependence of the feasible range. All measurements are scripted
deterministically; the real harness path is exercised separately.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from inferenceos.contracts.enums import (
    BackendType,
    CandidateEvaluationStatus,
    MeasurementMode,
    SearchSelectionKind,
)
from inferenceos.contracts.experiment import ExperimentPolicy, RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import MetricStats, ProfilingRun
from inferenceos.contracts.search import GpuOffloadResourceEnvelope, GpuOffloadSearchResult
from inferenceos.optimization.search.evaluation import (
    MappingPointEvaluator,
    PointMeasurement,
    RuntimeConfigurationPointEvaluator,
)
from inferenceos.optimization.search.policy import SearchPolicy
from inferenceos.optimization.search.resources import build_resource_envelope
from inferenceos.optimization.search.runner import run_gpu_offload_search
from inferenceos.optimization.search.sampling import planned_probe, sample_layers
from inferenceos.profiling.statistics import summarize
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_model
from tests.diagnosis_fixtures import make_run

_GIB = 1024**3


def _policy(**overrides: object) -> SearchPolicy:
    base: dict[str, object] = {
        "safety_headroom_fraction": 0.15,
        "runtime_reserve_fraction": 0.05,
        "telemetry_max_age_s": 30.0,
        "sample_step_layers": 2,
        "max_samples": 6,
        "probe_step_layers": 2,
        "diminishing_returns_margin": 0.05,
        "kv_element_bytes": 2,
    }
    base.update(overrides)
    return SearchPolicy(**base)  # type: ignore[arg-type]


def _model(**overrides: object) -> ModelProfile:
    defaults: dict[str, object] = {
        "n_layers": 32,
        "weight_bytes": 3 * _GIB,
        "kv_cache_bytes_per_token": 128 * 1024,
    }
    defaults.update(overrides)
    return sample_model().model_copy(update=defaults)


def _envelope(
    *,
    current: int = 8,
    total: int = 32,
    total_vram: int = 8 * _GIB,
    used_vram: int | None = None,
    context_length: int = 4096,
    model: ModelProfile | None = None,
    telemetry_available: bool = True,
    observed_at: datetime | None = None,
    policy: SearchPolicy | None = None,
) -> GpuOffloadResourceEnvelope:
    policy = policy or _policy()
    model = model or _model()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    observed = observed_at if observed_at is not None else now
    if used_vram is None:
        kv = model.estimate_kv_cache_bytes(context_length, kv_dtype_bytes=policy.kv_element_bytes)
        per_layer = -(-model.weight_bytes // total)
        used_vram = kv + current * per_layer
    return build_resource_envelope(
        total_vram_bytes=total_vram,
        used_vram_bytes=used_vram,
        telemetry_available=telemetry_available,
        telemetry_observed_at=observed,
        telemetry_source="test_gpu_counters",
        now=now,
        model=model,
        context_length=context_length,
        current_layers=current,
        total_layers=total,
        policy=policy,
    )


def _stats(name: str, value: float) -> MetricStats:
    return summarize(name, [value, value, value])


def point(
    layers: int,
    *,
    tps: float = 20.0,
    latency: float = 2.0,
    ttft: float = 0.2,
    vram: int | None = None,
    rss: int | None = None,
    status: CandidateEvaluationStatus = CandidateEvaluationStatus.OK,
    safe: bool = True,
    valid: bool = True,
    samples: int = 3,
    reasons: list[str] | None = None,
) -> PointMeasurement:
    metrics: dict[str, MetricStats] = {
        "decode_tokens_per_sec": _stats("decode_tokens_per_sec", tps),
        "total_latency_s": _stats("total_latency_s", latency),
        "ttft_s": _stats("ttft_s", ttft),
    }
    if vram is not None:
        metrics["gpu_memory_used_bytes"] = _stats("gpu_memory_used_bytes", float(vram))
    if rss is not None:
        metrics["peak_rss_bytes"] = _stats("peak_rss_bytes", float(rss))
    return PointMeasurement(
        layers=layers,
        status=status,
        valid=valid,
        safe=safe,
        samples=samples,
        warmup_runs=1,
        metrics=metrics,
        reasons=list(reasons or []),
        measured_vram_bytes=vram,
    )


def _search_envelope(current: int = 8, feasible_max: int = 18) -> GpuOffloadResourceEnvelope:
    return GpuOffloadResourceEnvelope(
        sufficient=True,
        total_vram_bytes=8 * _GIB,
        used_vram_bytes=3 * _GIB,
        free_vram_bytes=5 * _GIB,
        other_consumers_bytes=1 * _GIB,
        safety_headroom_bytes=int(0.15 * 8 * _GIB),
        runtime_reserve_bytes=int(0.05 * 8 * _GIB),
        kv_cache_bytes=256 * 1024**2,
        per_layer_weight_bytes=96 * 1024**2,
        weight_bytes_estimate=3 * _GIB,
        context_length=4096,
        current_layers=current,
        total_layers=32,
        feasible_min=current,
        feasible_max=feasible_max,
    )


# --------------------------------------------------------------------------- #
# Resource constraints                                                          #
# --------------------------------------------------------------------------- #
def test_insufficient_vram_keeps_range_at_current() -> None:
    envelope = _envelope(total_vram=512 * 1024**2, model=_model(weight_bytes=4 * _GIB))
    assert envelope.sufficient is True
    assert envelope.feasible_max == envelope.feasible_min == 8


def test_sufficient_vram_expands_range() -> None:
    envelope = _envelope(total_vram=8 * _GIB)
    assert envelope.feasible_max is not None and envelope.feasible_max > 8


def test_other_gpu_consumers_reduce_range() -> None:
    baseline = _envelope()
    consumed = _envelope(used_vram=(baseline.used_vram_bytes or 0) + 3 * _GIB)
    assert baseline.feasible_max is not None
    assert consumed.feasible_max is not None
    assert consumed.feasible_max < baseline.feasible_max


def test_safety_headroom_reduces_range() -> None:
    model = _model(weight_bytes=6 * _GIB)
    small = _envelope(model=model, policy=_policy(safety_headroom_fraction=0.05))
    large = _envelope(model=model, policy=_policy(safety_headroom_fraction=0.50))
    assert small.feasible_max is not None and large.feasible_max is not None
    assert large.feasible_max < small.feasible_max
    assert large.safety_headroom_bytes is not None
    assert small.safety_headroom_bytes is not None
    assert large.safety_headroom_bytes > small.safety_headroom_bytes


def test_missing_vram_telemetry_is_insufficient() -> None:
    envelope = _envelope(telemetry_available=False)
    assert envelope.sufficient is False
    assert envelope.feasible_max == envelope.feasible_min == 8
    assert any("telemetry unavailable" in reason for reason in envelope.reasons)


def test_stale_vram_telemetry_is_insufficient() -> None:
    stale = datetime(2026, 1, 1, tzinfo=timezone.utc) - timedelta(minutes=5)
    envelope = _envelope(observed_at=stale)
    assert envelope.telemetry_fresh is False
    assert envelope.sufficient is False
    assert any("stale" in reason for reason in envelope.reasons)


def test_context_changes_feasible_range() -> None:
    short = _envelope(context_length=4096)
    long = _envelope(context_length=32768)
    assert short.kv_cache_bytes is not None and long.kv_cache_bytes is not None
    assert long.kv_cache_bytes > short.kv_cache_bytes
    assert short.feasible_max is not None and long.feasible_max is not None
    assert long.feasible_max < short.feasible_max


def test_kv_cache_memory_affects_feasible_range() -> None:
    small_kv = _envelope(
        model=_model(weight_bytes=3 * _GIB, kv_cache_bytes_per_token=16 * 1024),
        total_vram=3 * _GIB,
    )
    large_kv = _envelope(
        model=_model(weight_bytes=3 * _GIB, kv_cache_bytes_per_token=256 * 1024),
        total_vram=3 * _GIB,
    )
    assert small_kv.feasible_max is not None and large_kv.feasible_max is not None
    assert large_kv.feasible_max < small_kv.feasible_max


# --------------------------------------------------------------------------- #
# Range construction                                                           #
# --------------------------------------------------------------------------- #
def test_range_preserves_current_as_minimum() -> None:
    envelope = _envelope(current=8)
    assert envelope.feasible_min == 8
    assert envelope.feasible_max is not None and envelope.feasible_max >= 8


def test_range_full_model_fits() -> None:
    envelope = _envelope(total=16, total_vram=64 * _GIB)
    # Everything fits: 16 blocks plus the output step (count 17 = L+1, review 2).
    assert envelope.feasible_max == 17


def test_range_full_model_does_not_fit() -> None:
    envelope = _envelope(total=32, total_vram=8 * _GIB, model=_model(weight_bytes=12 * _GIB))
    assert envelope.feasible_max is not None and envelope.feasible_max < 32


def test_current_at_safe_maximum() -> None:
    envelope = _envelope(
        current=20, total=32, total_vram=8 * _GIB, model=_model(weight_bytes=12 * _GIB)
    )
    assert envelope.feasible_max == 20


# --------------------------------------------------------------------------- #
# Candidate sampling                                                           #
# --------------------------------------------------------------------------- #
def test_sampling_deterministic_bounded_and_ordered() -> None:
    points = sample_layers(8, 18, _policy())
    assert points == sorted(set(points))
    assert points[0] == 8
    assert points[-1] == 18
    assert len(points) <= 6


def test_sampling_includes_upper_boundary_and_current() -> None:
    points = sample_layers(8, 19, _policy(sample_step_layers=2, max_samples=6))
    assert 8 in points and 19 in points


def test_sampling_no_duplicates_and_current_only_when_no_room() -> None:
    assert sample_layers(8, 8, _policy()) == [8]
    assert sample_layers(8, 7, _policy()) == [8]


def test_planned_probe_is_bounded_not_maximum() -> None:
    assert planned_probe(8, 18, _policy(probe_step_layers=2)) == 10
    assert planned_probe(8, 10, _policy(probe_step_layers=8)) == 10
    assert planned_probe(8, 8, _policy()) == 8


# --------------------------------------------------------------------------- #
# Selection                                                                     #
# --------------------------------------------------------------------------- #
def _run(points: dict[int, PointMeasurement], *, feasible_max: int = 12) -> GpuOffloadSearchResult:
    evaluator = MappingPointEvaluator(points, context_length=4096)
    return run_gpu_offload_search(
        envelope=_search_envelope(feasible_max=feasible_max),
        evaluator=evaluator,
        search_policy=_policy(),
        experiment_policy=ExperimentPolicy(),
    )


def test_higher_offload_improves_performance_selects_one_value() -> None:
    result = _run(
        {
            8: point(8, tps=20.0),
            10: point(10, tps=24.0),
            12: point(12, tps=28.0),
        }
    )
    assert result.selection.kind is SearchSelectionKind.SELECT
    assert result.selection.selected_layers == 12


def test_more_offload_without_improvement_is_no_change() -> None:
    result = _run(
        {
            8: point(8, tps=20.0),
            10: point(10, tps=20.2),
            12: point(12, tps=20.1),
        }
    )
    assert result.selection.kind is SearchSelectionKind.NO_CHANGE
    assert result.selection.selected_layers is None


def test_throughput_improves_but_latency_regresses_is_no_change() -> None:
    result = _run(
        {
            8: point(8, tps=20.0, latency=2.0),
            10: point(10, tps=28.0, latency=4.0),
        },
        feasible_max=10,
    )
    assert result.selection.kind is SearchSelectionKind.NO_CHANGE


def test_highest_feasible_layer_count_is_not_automatically_selected() -> None:
    result = _run(
        {
            8: point(8, tps=20.0),
            10: point(10, tps=26.0),
            12: point(12, tps=26.5),
            14: point(14, tps=26.6, latency=5.0),  # materially worse latency
        },
        feasible_max=14,
    )
    assert result.selection.kind is SearchSelectionKind.SELECT
    assert result.selection.selected_layers == 10
    assert result.selection.selected_layers != 14


def test_current_configuration_wins_is_no_change() -> None:
    result = _run(
        {
            8: point(8, tps=22.0),
            10: point(10, tps=21.0),
            12: point(12, tps=20.0),
        }
    )
    assert result.selection.kind is SearchSelectionKind.NO_CHANGE


def test_selected_candidate_satisfies_resource_constraints() -> None:
    result = _run(
        {
            8: point(8, tps=20.0),
            10: point(10, tps=25.0),
            12: point(12, tps=26.0),
        },
        feasible_max=12,
    )
    assert result.selection.kind is SearchSelectionKind.SELECT
    assert result.selection.selected_layers is not None
    assert result.selection.selected_layers <= (result.resource.feasible_max or 0)


def test_oom_candidate_is_rejected_and_stops_search() -> None:
    evaluator = MappingPointEvaluator(
        {
            8: point(8, tps=20.0),
            10: point(10, tps=20.1),
            12: point(
                12,
                status=CandidateEvaluationStatus.OOM,
                safe=False,
                valid=False,
                reasons=["cuMemAlloc failed: out of memory"],
            ),
        },
        context_length=4096,
    )
    result = run_gpu_offload_search(
        envelope=_search_envelope(),
        evaluator=evaluator,
        search_policy=_policy(),
        experiment_policy=ExperimentPolicy(),
    )
    assert result.selection.kind is SearchSelectionKind.NO_CHANGE
    assert 14 not in evaluator.calls
    assert any("oom" in line for line in result.selection.comparisons) or any(
        "oom" in reason for reason in result.limitations
    )


def test_vram_exceeded_candidate_is_rejected_and_stops_search() -> None:
    evaluator = MappingPointEvaluator(
        {
            8: point(8, tps=20.0, vram=1 * _GIB),
            10: point(10, tps=20.1, vram=1 * _GIB),
            12: point(
                12,
                status=CandidateEvaluationStatus.VRAM_EXCEEDED,
                safe=False,
                valid=False,
                reasons=["measured VRAM exceeds the safe ceiling"],
            ),
        },
        context_length=4096,
    )
    result = run_gpu_offload_search(
        envelope=_search_envelope(),
        evaluator=evaluator,
        search_policy=_policy(),
        experiment_policy=ExperimentPolicy(),
    )
    assert result.selection.kind is SearchSelectionKind.NO_CHANGE
    assert 14 not in evaluator.calls


def test_high_variance_result_becomes_insufficient_evidence() -> None:
    evaluator = MappingPointEvaluator(
        {
            8: point(8, tps=20.0),
            10: point(
                10,
                status=CandidateEvaluationStatus.INVALID,
                valid=False,
                safe=True,
                reasons=["insufficient valid measurements (1 < 2)"],
            ),
        },
        context_length=4096,
    )
    result = run_gpu_offload_search(
        envelope=_search_envelope(feasible_max=10),
        evaluator=evaluator,
        search_policy=_policy(),
        experiment_policy=ExperimentPolicy(),
    )
    assert result.selection.kind is SearchSelectionKind.NO_CHANGE
    assert 12 not in evaluator.calls


# --------------------------------------------------------------------------- #
# Evaluation: warm-up, repeats, OOM classification via the real harness path   #
# --------------------------------------------------------------------------- #
class _FakeHarness:
    def __init__(
        self, *, fail_measured: bool = False, oom: bool = False, vram: int = 1 * _GIB
    ) -> None:
        self.calls: list[bool] = []
        self._fail_measured = fail_measured
        self._oom = oom
        self._vram = vram

    def run_once(
        self,
        task: object,
        *,
        mode: MeasurementMode,
        sequence: int,
        repetition: int,
        warmup: bool,
        options: object,
    ) -> ProfilingRun:
        self.calls.append(warmup)
        if self._oom:
            raise RuntimeError("failed to allocate Vulkan buffer: out of memory")
        if self._fail_measured and not warmup:
            raise RuntimeError("invalid run")
        layers = getattr(options, "gpu_offload_layers", 0)
        return make_run(
            model_name="qwen-search",
            workload_id="short-short",
            condition="search",
            gpu_offload_layers=int(layers),
            gpu_offload_total=32,
            backend=BackendType.VULKAN,
            metrics={
                "decode_tokens_per_sec": 20.0,
                "total_latency_s": 2.0,
                "gpu_memory_used_bytes": float(self._vram),
            },
        )


def _evaluator(
    harness: _FakeHarness, envelope: GpuOffloadResourceEnvelope | None = None
) -> RuntimeConfigurationPointEvaluator:
    return RuntimeConfigurationPointEvaluator(
        profiler=harness,
        model=_model(),
        workload=get_workload("short-short"),
        experiment_policy=ExperimentPolicy(),
        context_length=4096,
        envelope=envelope or _search_envelope(),
        base_configuration=RuntimeConfiguration(),
    )


def test_measurement_excludes_warmup_and_requires_repeats() -> None:
    harness = _FakeHarness()
    measurement = _evaluator(harness).measure(10)
    assert measurement.ok is True
    assert measurement.warmup_runs >= 1
    assert measurement.samples >= 2
    assert harness.calls[0] is True  # warm-up first, then measured runs
    assert any(flag is False for flag in harness.calls)


def test_single_valid_measurement_is_insufficient_evidence() -> None:
    harness = _FakeHarness(fail_measured=True)
    measurement = _evaluator(harness).measure(10)
    assert measurement.valid is False
    assert measurement.status is CandidateEvaluationStatus.INVALID
    assert measurement.samples == 0


def test_oom_measurement_is_classified_unsafe() -> None:
    harness = _FakeHarness(oom=True)
    measurement = _evaluator(harness).measure(10)
    assert measurement.safe is False
    assert measurement.status is CandidateEvaluationStatus.OOM


def test_actual_vram_exceeding_envelope_is_classified_unsafe() -> None:
    harness = _FakeHarness(vram=2 * _GIB)
    envelope = GpuOffloadResourceEnvelope(
        sufficient=True,
        total_vram_bytes=2 * _GIB,
        used_vram_bytes=512 * 1024**2,
        safety_headroom_bytes=512 * 1024**2,
        current_layers=8,
        total_layers=32,
        feasible_min=8,
        feasible_max=10,
        context_length=4096,
    )
    measurement = _evaluator(harness, envelope).measure(10)
    assert measurement.safe is False
    assert measurement.status is CandidateEvaluationStatus.VRAM_EXCEEDED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
