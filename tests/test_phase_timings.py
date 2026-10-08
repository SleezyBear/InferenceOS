"""Per-run phase timings and the cost ledger (handoff, 6 Oct 2026: verification cost)."""

from __future__ import annotations

import sys
import time
from typing import Any

import pytest

from inferenceos.contracts.enums import MeasurementMode
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendCapabilities,
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
)
from inferenceos.profiling.runner import BaselineProfiler, ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from inferenceos.selftest import sample_hardware, sample_model
from inferenceos.tune.verify_cost import CostLedger, measurement_breakdown


class _SlowAdapter(BackendAdapter):
    def __init__(self) -> None:
        self.loaded = False

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="fake", version="1")

    def load(self, model: Any, params: Any) -> None:
        time.sleep(0.02)
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        time.sleep(0.03)
        return GenerationResult(text="ok", ttft_s=0.01, decode_tokens_per_sec=20.0, generated_tokens=4)

    def set_runtime_params(self, params: Any) -> None:
        pass

    def telemetry(self) -> BackendTelemetry:
        return BackendTelemetry()

    @property
    def is_loaded(self) -> bool:
        return self.loaded


@pytest.mark.skipif(sys.platform != "win32", reason="the profiler's platform layer is Windows-only")
def test_run_once_records_where_its_time_went() -> None:
    profiler = BaselineProfiler(adapter=_SlowAdapter(), hardware=sample_hardware())
    task = ProfileTask(model=sample_model(), workload=get_workload("short-short"), context_length=2048)
    run = profiler.run_once(
        task, mode=MeasurementMode.WARM, sequence=1, repetition=0, warmup=False,
        options=ProfilerOptions(telemetry_sampling=False),
    )
    phases = run.phase_timings_s
    for name in ("load_s", "monitor_start_s", "request_s", "monitor_stop_s", "after_reads_s", "unload_s"):
        assert name in phases, name
    assert phases["load_s"] >= 0.015
    assert phases["request_s"] >= 0.025
    assert phases["monitor_start_s"] < 0.01  # no monitor when sampling is off
    assert phases["first_word_s"] == pytest.approx(0.01)


def test_ledger_costs_come_from_this_tune_only() -> None:
    ledger = CostLedger()
    assert ledger.measurement_cost("short-short") is None
    ledger.add("measure", "short-short", 30.0)
    ledger.add("measure", "short-short", 40.0)
    ledger.add("measure", "short-short", 35.0)
    ledger.add("final", None, 500.0)
    assert ledger.measurement_cost("short-short") == 35.0
    summary = ledger.summary()
    assert summary["seconds_by_kind"] == {"measure": 105.0, "final": 500.0}


def test_review8_warmup_load_is_attributed_not_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review 8: a 120 s measurement whose warm-up carried a 100 s load reported zero load
    and 100 s "between runs". The warm-up's phases are now kept (never scored)."""
    from inferenceos.experiment.measurement import measure_configuration
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy

    clock = {"t": 0.0}
    monkeypatch.setattr(time, "perf_counter", lambda: clock["t"])

    class Harness(ScriptedHarness):
        def run_once(self, task: Any, **kwargs: Any) -> Any:
            warm = bool(kwargs.get("warmup"))
            phases = {"load_s": 100.0 if warm else 0.0, "monitor_start_s": 1.0, "request_s": 4.0,
                      "monitor_stop_s": 0.0}
            clock["t"] += sum(phases.values())
            run = super().run_once(task, **kwargs)
            return run.model_copy(update={"phase_timings_s": phases})

    harness = Harness(lambda task, options, call: SampleResult(decode_tps=20.0 + call * 0.01))
    policy = build_test_policy(measured_repetitions=3, max_repetitions=3, min_valid_samples=3,
                               max_total_runs=6, noise_tolerance=0.0)
    measurement = measure_configuration(
        profiler=harness, model=sample_model(), workload=get_workload("short-short"),
        config=RuntimeConfiguration(context_length=2048), policy=policy, phase="t",
    )
    assert measurement.valid_runs == 3 and measurement.warmup_runs == 1
    assert len(measurement.runs) == 3  # the warm-up is not scored
    assert measurement.warmup_phase_timings and measurement.warmup_phase_timings[0]["load_s"] == 100.0
    assert measurement.elapsed_s == pytest.approx(120.0)
    out = measurement_breakdown(measurement)
    assert out["load_s"] == 100.0
    assert out["warmup_s"] == pytest.approx(105.0)
    assert out["scored_s"] == pytest.approx(15.0)
    assert out["between_runs_s"] == pytest.approx(0.0)
    ledger = CostLedger()
    ledger.add("measure", "short-short", measurement.elapsed_s or 0.0, measurement=measurement)
    assert ledger.load_cost() == 100.0
    assert ledger.answer_cost("short-short") == pytest.approx(5.0)
