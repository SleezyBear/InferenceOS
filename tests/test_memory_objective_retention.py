"""Memory-objective retention for memory-surfaced actions.

Memory efficiency is a legitimate optimization objective: a ``MEMORY_MANAGEMENT``
candidate (e.g. enabling mmap) may be retained on a **measured memory-objective
reduction** even when throughput does not improve — while a ``CONTEXT_MANAGEMENT``
candidate still requires a performance outcome, and a memory saving is never reported
as a performance claim.
"""

from __future__ import annotations

from typing import Any

from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    BackendType,
    BottleneckKind,
    PerformanceState,
)
from inferenceos.contracts.experiment import (
    ExperimentComparison,
    ExperimentResult,
    MeasurementDelta,
)
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.preference import build_optimization_preference
from inferenceos.runtime.economics import (
    evaluate_retention_economics,
    measured_memory_improvement,
)
from inferenceos.runtime.policy import build_runtime_policy
from inferenceos.selftest import sample_experiment_result
from tests.diagnosis_fixtures import make_report, make_run
from tests.experiment_helpers import SampleResult
from tests.loop_helpers import make_loop, report_for_config
from tests.optimization_helpers import make_candidate, synthetic_context


def _candidate(action_id: str) -> Any:
    return make_candidate(synthetic_context(BottleneckKind.PAGING), action_id=action_id, cost=3.0)


def _result(rss_improvement: float | None, perf_improvement: float | None) -> ExperimentResult:
    result = sample_experiment_result()
    deltas: dict[str, MeasurementDelta] = {}
    if rss_improvement is not None:
        deltas["peak_rss_bytes"] = MeasurementDelta(
            metric="peak_rss_bytes", role="bottleneck_primary", direction="lower_is_better",
            baseline_mean=1_300_000_000.0,
            candidate_mean=1_300_000_000.0 * (1.0 - rss_improvement),
            absolute_delta=-1_300_000_000.0 * rss_improvement,
            relative_delta=-rss_improvement,
            improved=rss_improvement > 0.0, significant=rss_improvement > 0.0,
        )
    if perf_improvement is not None:
        deltas["decode_tokens_per_sec"] = MeasurementDelta(
            metric="decode_tokens_per_sec", role="objective", direction="higher_is_better",
            baseline_mean=20.0, candidate_mean=20.0 * (1.0 + perf_improvement),
            relative_delta=perf_improvement, improved=perf_improvement > 0.0, significant=True,
        )
    comparison = ExperimentComparison(
        deltas=deltas, bottleneck_metric="peak_rss_bytes", bottleneck_improved=True
    )
    return result.model_copy(update={"comparison": comparison})


def _retention(action_id: str, rss: float | None, perf: float | None) -> Any:
    return evaluate_retention_economics(
        result=_result(rss, perf),
        candidate=_candidate(action_id),
        policy=build_runtime_policy(Settings()),
        optimization_preference=build_optimization_preference(Settings()),
        performance_state=PerformanceState.NORMAL,
    )


# --------------------------------------------------------------------------- #
# Economics: memory objective for MEMORY_MANAGEMENT only                         #
# --------------------------------------------------------------------------- #
def test_memory_improvement_is_measured() -> None:
    assert measured_memory_improvement(_result(0.30, None)) == 0.30
    assert measured_memory_improvement(_result(0.0, None)) is None


def test_mmap_candidate_retained_on_measured_memory_reduction() -> None:
    econ = _retention("runtime.enable_mmap", 0.30, None)
    assert econ.beneficial is True
    assert econ.measured_improvement == 0.30
    assert any("objective=memory" in line for line in econ.rationale)
    assert any("measured_memory_improvement=30.00%" in line for line in econ.rationale)


def test_mmap_memory_below_mode_minimum_is_not_retained() -> None:
    # BALANCED requires 5%; a measured 1% memory reduction is not worth retaining.
    assert _retention("runtime.enable_mmap", 0.01, None).beneficial is False


def test_non_memory_card_still_requires_performance_outcome() -> None:
    # Only memory-management cards may be kept for a memory-only gain; any other card
    # (here: threads) still needs a measured performance outcome.
    econ = _retention("runtime.reduce_threads", 0.30, None)
    assert econ.beneficial is False
    assert econ.measured_improvement is None


def test_memory_candidate_retained_on_performance_outcome() -> None:
    econ = _retention("runtime.enable_mmap", None, 0.30)
    assert econ.beneficial is True
    assert any("objective=performance" in line for line in econ.rationale)


def test_memory_candidate_without_any_outcome_is_not_retained() -> None:
    assert _retention("runtime.enable_mmap", 0.0, None).beneficial is False


# --------------------------------------------------------------------------- #
# End-to-end loop: only the mmap candidate is reachable, retained on memory     #
# --------------------------------------------------------------------------- #
def _mmap_report() -> Any:
    runs = []
    for condition, faults, pressure, latency in (
        ("control", 0.5, 0.6, 1.4),
        ("control", 0.6, 0.6, 1.4),
        ("pressured", 17000.0, 0.85, 2.6),
        ("pressured", 18000.0, 0.86, 2.7),
    ):
        run = make_run(
            condition=condition,
            backend=BackendType.CPU,
            gpu_offload_layers=0,
            gpu_offload_total=28,
            hard_required_context_tokens=None,
            memory_pressure=pressure,
            metrics={
                "decode_tokens_per_sec": 24.0 if faults < 1.0 else 9.0,
                "total_latency_s": latency,
                "hard_page_faults_per_sec": faults,
                "peak_rss_bytes": 1_300_000_000,
                "available_ram_bytes": float(8 * 1024**3),
                "total_ram_bytes": float(16 * 1024**3),
            },
        )
        config = run.context.config.model_copy(update={"context_length": 2048, "use_mmap": False})
        runs.append(
            run.model_copy(update={"context": run.context.model_copy(update={"config": config})})
        )
    return make_report(runs)


def _responder(*, rss_when_mmap: int, decode: float = 20.0) -> Any:
    def respond(task: Any, options: Any, call: int) -> SampleResult:
        if options.use_mmap:
            return SampleResult(rss_bytes=rss_when_mmap, hard_faults_per_sec=1.0, decode_tps=decode)
        return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=decode)
    return respond


def test_loop_only_mmap_is_eligible() -> None:
    report = _mmap_report()
    diag = BottleneckDetector().detect(report, source_report_path="<test>")
    plan = generate_candidates(diag, report=report)
    eligible = {c.action_id for c in plan.candidates if c.executable}
    assert eligible == {"runtime.enable_mmap"}


def test_loop_retains_mmap_for_measured_memory_reduction() -> None:
    report = _mmap_report()
    stack = make_loop(
        responder=_responder(rss_when_mmap=900_000_000),
        report=report,
        observer=lambda config: report_for_config(report, config),
    )
    result = stack.loop.run(report, report_path="<test-mmap-memory>")
    assert result.changed is True
    retained = [i for i in result.run.iterations if i.retained]
    assert retained
    assert retained[0].selected_action_id == "runtime.enable_mmap"
    # The retention is a memory-objective result, not a performance claim.
    assert any("memory" in note for note in retained[0].notes)


def test_loop_does_not_retain_mmap_without_any_measured_outcome() -> None:
    report = _mmap_report()
    stack = make_loop(
        responder=_responder(rss_when_mmap=1_300_000_000),
        report=report,
    )
    result = stack.loop.run(report, report_path="<test-mmap-noop>")
    assert result.changed is False
    experimented = [i for i in result.run.iterations if i.experiment_id is not None]
    assert experimented
    assert all(not i.retained for i in experimented)
