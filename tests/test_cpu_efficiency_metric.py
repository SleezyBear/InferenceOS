"""Phase 12D: CPU efficiency/contention telemetry and the corrected reduce_threads
causal objective.

Proves that the technical decision for a CPU-execution diagnosis is driven by throughput
and CPU-seconds/token (with context-switch reduction as supporting evidence) and that raw
``process_cpu_percent`` can no longer, by itself, reject an otherwise improved thread
configuration. Diagnosis and economics behavior are unchanged.
"""

from __future__ import annotations

import os
import time

from inferenceos.backends.process_control import PeakRssAccumulator
from inferenceos.contracts.enums import (
    BackendType,
    ExperimentDecisionKind,
    MeasurementMode,
    QualityStatus,
    RestorationStatus,
)
from inferenceos.contracts.experiment import (
    ApplicationResult,
    ExperimentComparison,
    ExperimentMeasurement,
    QualityValidationResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.observation import RunMetrics
from inferenceos.contracts.profiling import MetricStats
from inferenceos.diagnosis.groups import GroupObservation
from inferenceos.diagnosis.rules import CPU_SATURATION_HIGH, _cpu_saturation
from inferenceos.experiment.comparison import compare
from inferenceos.experiment.decision import decide
from inferenceos.interfaces.backend_adapter import GenerationResult
from inferenceos.profiling.runner import _build_metrics
from inferenceos.profiling.statistics import summarize
from inferenceos.selftest import sample_model, sample_workload
from tests.experiment_helpers import build_test_policy


def _measurement(values: dict[str, list[float]]) -> ExperimentMeasurement:
    return ExperimentMeasurement(
        measurement_id="m",
        phase="x",
        configuration=RuntimeConfiguration(),
        repetitions=2,
        valid_runs=2,
        run_ids=["a", "b"],
        metrics={name: summarize(name, vals) for name, vals in values.items()},
    )


def _quality() -> QualityValidationResult:
    return QualityValidationResult(
        status=QualityStatus.OK,
        validated=True,
        execution_ok=True,
        structural_ok=True,
    )


def _application() -> ApplicationResult:
    return ApplicationResult(
        applied=True, action_id="runtime.reduce_threads", supported=True, reason="ok"
    )


def _metrics(
    *,
    cpu_seconds: float | None = None,
    tokens: int | None = 64,
    context_switches_total: int | None = None,
) -> RunMetrics:
    return _build_metrics(
        result=GenerationResult(
            text="ok",
            generated_tokens=tokens,
            cpu_seconds=cpu_seconds,
            context_switches_total=context_switches_total,
        ),
        samples=[],
        telemetry_sampling=False,
        system_cpu=10.0,
        memory_after=None,
        wall_s=1.0,
    )


# --------------------------------------------------------------------------- #
# 1/2. CPU-seconds and CPU-seconds/token                                        #
# --------------------------------------------------------------------------- #
def test_peak_cpu_accumulator_reports_cpu_seconds() -> None:
    accumulator = PeakRssAccumulator()
    pid = os.getpid()
    accumulator.observe(pid)
    deadline = time.perf_counter() + 0.25
    value = 0
    while time.perf_counter() < deadline:
        value = (value * 1103515245 + 12345) & 0x7FFFFFFF
    accumulator.observe(pid)
    assert accumulator.cpu_seconds is not None
    assert accumulator.cpu_seconds > 0.0


def test_cpu_seconds_unknown_with_single_sample() -> None:
    accumulator = PeakRssAccumulator()
    accumulator.observe(os.getpid())
    assert accumulator.cpu_seconds is None


def test_cpu_seconds_per_token_calculation() -> None:
    metrics = _metrics(cpu_seconds=0.5, tokens=100)
    assert metrics.cpu_seconds_per_token == 0.005


def test_cpu_seconds_per_token_unknown_for_missing_or_zero_tokens() -> None:
    assert _metrics(cpu_seconds=0.5, tokens=0).cpu_seconds_per_token is None
    assert _metrics(cpu_seconds=0.5, tokens=None).cpu_seconds_per_token is None
    assert _metrics(cpu_seconds=None, tokens=100).cpu_seconds_per_token is None


# --------------------------------------------------------------------------- #
# 3/4. Context-switch telemetry                                                 #
# --------------------------------------------------------------------------- #
def test_peak_cpu_accumulator_reports_context_switches() -> None:
    accumulator = PeakRssAccumulator()
    pid = os.getpid()
    accumulator.observe(pid)
    deadline = time.perf_counter() + 0.2
    value = 0
    while time.perf_counter() < deadline:
        value = (value * 1103515245 + 12345) & 0x7FFFFFFF
        time.sleep(0)  # encourage voluntary switches
    accumulator.observe(pid)
    assert accumulator.context_switches_voluntary is not None
    assert accumulator.context_switches_involuntary is not None
    assert accumulator.context_switches_total == (
        accumulator.context_switches_voluntary + accumulator.context_switches_involuntary
    )


def test_missing_context_switch_telemetry_is_unknown() -> None:
    accumulator = PeakRssAccumulator()
    accumulator.observe(2_147_483_000)  # never a live pid
    assert accumulator.context_switches_total is None
    assert accumulator.cpu_seconds is None


def test_context_switches_per_token_calculation() -> None:
    metrics = _metrics(context_switches_total=200, tokens=100)
    assert metrics.context_switches_per_token == 2.0


def test_context_switches_per_token_unknown_without_tokens() -> None:
    assert _metrics(context_switches_total=200, tokens=0).context_switches_per_token is None
    assert _metrics(context_switches_total=None, tokens=100).context_switches_per_token is None


# --------------------------------------------------------------------------- #
# 5-8. Corrected reduce_threads causal objective                                #
# --------------------------------------------------------------------------- #
def _cpu_comparison(
    baseline: dict[str, list[float]], candidate: dict[str, list[float]]
) -> ExperimentComparison:
    return compare(
        baseline=_measurement(baseline),
        candidate=_measurement(candidate),
        bottleneck_metric="decode_tokens_per_sec",
        bottleneck_direction="increase",
        quality=_quality(),
        policy=build_test_policy(),
        diagnosed_bottleneck="cpu",
    )


def test_lower_cpu_seconds_per_token_is_an_improvement() -> None:
    comparison = _cpu_comparison(
        {"decode_tokens_per_sec": [10.0, 10.0], "cpu_seconds_per_token": [0.5, 0.5]},
        {"decode_tokens_per_sec": [10.0, 10.0], "cpu_seconds_per_token": [0.35, 0.35]},
    )
    assert comparison.deltas["cpu_seconds_per_token"].role == "bottleneck_primary"
    assert comparison.deltas["cpu_seconds_per_token"].improved is True
    assert comparison.bottleneck_improved is True


def test_higher_raw_cpu_percent_does_not_reject_improved_threads() -> None:
    comparison = _cpu_comparison(
        {
            "decode_tokens_per_sec": [10.0, 10.0],
            "cpu_seconds_per_token": [0.5, 0.5],
            "process_cpu_percent": [700.0, 700.0],
        },
        {
            "decode_tokens_per_sec": [15.0, 15.0],
            "cpu_seconds_per_token": [0.35, 0.35],
            "process_cpu_percent": [900.0, 900.0],
        },
    )
    # Raw CPU utilisation rose, but it is only a secondary resource signal.
    assert comparison.deltas["process_cpu_percent"].role == "bottleneck_secondary"
    assert comparison.deltas["process_cpu_percent"].regressed is True
    assert comparison.bottleneck_improved is True
    decision = decide(
        comparison=comparison,
        quality=_quality(),
        application=_application(),
        restoration_status=RestorationStatus.NOT_REQUIRED,
        measurement_valid=True,
        precondition_ok=True,
        precondition_kind=None,
        precondition_reason="ok",
        policy=build_test_policy(),
        errors=[],
    )
    assert decision.kind is ExperimentDecisionKind.IMPROVED
    assert decision.keep is True


def test_throughput_improvement_is_valid_evidence() -> None:
    comparison = _cpu_comparison(
        {"decode_tokens_per_sec": [10.0, 10.0], "process_cpu_percent": [700.0, 700.0]},
        {"decode_tokens_per_sec": [15.0, 15.0], "process_cpu_percent": [900.0, 900.0]},
    )
    assert comparison.deltas["decode_tokens_per_sec"].role == "bottleneck_primary"
    assert comparison.bottleneck_improved is True


def test_context_switch_reduction_is_supporting_evidence() -> None:
    comparison = _cpu_comparison(
        {
            "decode_tokens_per_sec": [10.0, 10.0],
            "cpu_seconds_per_token": [0.5, 0.5],
            "context_switches_per_token": [10_000.0, 10_000.0],
        },
        {
            "decode_tokens_per_sec": [14.0, 14.0],
            "cpu_seconds_per_token": [0.4, 0.4],
            "context_switches_per_token": [1_000.0, 1_000.0],
        },
    )
    assert comparison.deltas["context_switches_per_token"].role == "bottleneck_secondary"
    assert comparison.deltas["context_switches_per_token"].improved is True


# --------------------------------------------------------------------------- #
# 9. Diagnosis unchanged                                                        #
# --------------------------------------------------------------------------- #
def test_cpu_diagnosis_still_uses_process_cpu_saturation() -> None:
    group = GroupObservation(
        model_ref="m",
        workload_id="w",
        mode=MeasurementMode.WARM,
        condition="c",
        runs=(),
        n=1,
        stats={
            "process_cpu_percent": MetricStats(
                metric="process_cpu_percent", count=1, mean=500.0
            )
        },
        memory_pressure_mean=None,
        model=sample_model(),
        workload=sample_workload(),
        backend_type=BackendType.CPU,
        gpu_layers_offloaded=0,
        gpu_layers_total=28,
    )
    assert _cpu_saturation(group, 6) == 500.0 / 600.0
    assert CPU_SATURATION_HIGH == 0.70


# --------------------------------------------------------------------------- #
# 10/11. Economics unchanged + GPU comparison unchanged                         #
# --------------------------------------------------------------------------- #
def test_gpu_underutilization_comparison_unchanged() -> None:
    comparison = compare(
        baseline=_measurement({"decode_tokens_per_sec": [20.0, 20.0]}),
        candidate=_measurement({"decode_tokens_per_sec": [30.0, 30.0]}),
        bottleneck_metric="decode_tokens_per_sec",
        bottleneck_direction="increase",
        quality=_quality(),
        policy=build_test_policy(),
        diagnosed_bottleneck="gpu_underutilization",
    )
    assert comparison.deltas["decode_tokens_per_sec"].role == "bottleneck_primary"
    # The CPU-efficiency metric is not made primary for a GPU diagnosis.
    assert comparison.deltas["cpu_seconds_per_token"].role == "bottleneck_secondary"
    assert comparison.bottleneck_improved is True

