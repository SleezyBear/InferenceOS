"""Runtime integration tests: resource-aware GPU-offload search through the monitor.

These drive the real ``RuntimeMonitor`` opportunity -> search -> approval ->
apply/validate/rollback flow with the scripted telemetry/harness doubles. The search
is genuinely executed (resource envelope, feasible range, sampling, measurement,
comparison, selection); only the physical inference is scripted.

A separate test exercises the resident measurement shim against a fake resident
server to prove candidate evaluation keeps the target owned/alive (no orphan) and that
the known-good configuration is restorable.
"""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    ApprovalOutcome,
    ExperimentDecisionKind,
    MeasurementMode,
    SearchSelectionKind,
)
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.lifecycle import TargetHealth
from inferenceos.contracts.profiling import ProfilingRun
from inferenceos.contracts.runtime import RuntimeBudget, RuntimeObservation, RuntimePolicy
from inferenceos.contracts.system_state import GPUMemoryState, SystemState
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.runtime.approval import ScriptedApprovalProvider
from inferenceos.runtime.checkpoint import (
    capture_runtime_checkpoint,
    restore_runtime_checkpoint,
)
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.runtime.telemetry import RuntimeTelemetrySampler
from inferenceos.runtime.workload import ActiveWorkload, config_key
from inferenceos.selftest import sample_hardware, sample_model
from tests.experiment_helpers import (
    Responder,
    SampleResult,
    ScriptedHarness,
    build_test_policy,
)
from tests.resident_helpers import FakeResidentBackend, synthetic_model
from tests.runtime_helpers import ScriptedWorkload, make_runtime_policy

_GIB = 1024**3


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


class GpuUnderutilizationSampler(RuntimeTelemetrySampler):
    """Baseline is healthy; afterwards the GPU is idle while the CPU is saturated."""

    def __init__(
        self, *, gpu_telemetry: bool = True, observed_at: datetime | None = None
    ) -> None:
        self.count = 0
        self._gpu_telemetry = gpu_telemetry
        self._observed_at = observed_at

    def _metrics(self, condition: str) -> dict[str, float]:
        if condition == "baseline":
            return {
                "process_cpu_percent": 100.0,
                "gpu_utilization_percent": 62.0,
                "gpu_memory_used_bytes": float(512 * 1024**2),
                "decode_tokens_per_sec": 24.0,
                "total_latency_s": 1.4,
                "ttft_s": 0.12,
                "peak_rss_bytes": 1_300_000_000.0,
                "memory_pressure": 0.5,
                "hard_page_faults_per_sec": 0.5,
                "available_ram_bytes": float(8 * _GIB),
                "total_ram_bytes": float(16 * _GIB),
                "system_cpu_percent": 25.0,
            }
        return {
            "process_cpu_percent": 520.0,
            "gpu_utilization_percent": 5.0,
            "gpu_memory_used_bytes": float(512 * 1024**2),
            "decode_tokens_per_sec": 6.0,
            "total_latency_s": 5.0,
            "ttft_s": 0.6,
            "peak_rss_bytes": 1_300_000_000.0,
            "memory_pressure": 0.5,
            "hard_page_faults_per_sec": 0.5,
            "available_ram_bytes": float(8 * _GIB),
            "total_ram_bytes": float(16 * _GIB),
            "system_cpu_percent": 55.0,
        }

    def sample(
        self, workload: ActiveWorkload, *, sequence: int, condition: str
    ) -> RuntimeObservation:
        self.count += 1
        configuration = workload.configuration()
        snapshot = workload.snapshot()
        descriptor = workload.descriptor()
        now = datetime.now(timezone.utc)
        metrics = dict(self._metrics(condition))
        if not self._gpu_telemetry:
            metrics.pop("gpu_memory_used_bytes", None)
        state = SystemState(
            captured_at=now,
            total_ram_bytes=16 * _GIB,
            available_ram_bytes=8 * _GIB,
            memory_pressure=0.5,
            cpu_percent_total=40.0,
            hard_page_faults_per_sec=0.5,
            gpu_memory=(
                [GPUMemoryState(device_index=0, memory_used_bytes=512 * 1024**2)]
                if self._gpu_telemetry
                else []
            ),
            gpu_telemetry_available=self._gpu_telemetry,
            gpu_observed_at=(
                (self._observed_at or now) if self._gpu_telemetry else None
            ),
            gpu_source="windows_gpu_engine_counters" if self._gpu_telemetry else "unavailable",
        )
        return RuntimeObservation(
            observation_id=f"rtobs-gpu-{self.count}",
            sequence=sequence,
            captured_at=now,
            workload_state=snapshot.state,
            generating=snapshot.generating,
            context_length=configuration.context_length,
            config_key=config_key(configuration),
            workload_ref=descriptor.workload_id,
            model_ref=descriptor.model_ref,
            metrics=metrics,
            system_state=state,
            condition=condition,
        )


def _budget(duration: float = 12.0) -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=duration,
        telemetry_interval_s=0.5,
        observation_window_size=40,
        baseline_samples=4,
        max_deep_analyses=0,
        max_interventions=2,
        max_cumulative_intervention_s=600.0,
        max_optimizer_overhead_s=120.0,
        approval_timeout_s=10.0,
        max_alerts_per_condition=2,
    )


def _improving_responder(task: object, options: object, call: int) -> SampleResult:
    layers = getattr(options, "gpu_offload_layers", 0) or 0
    if layers <= 8:
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)
    if layers <= 16:
        return SampleResult(decode_tps=30.0, total_latency_s=2.0)
    # Beyond 16 layers throughput plateaus and latency regresses: must not be chosen.
    return SampleResult(decode_tps=30.5, total_latency_s=3.0)


def _flat_responder(task: object, options: object, call: int) -> SampleResult:
    return SampleResult(decode_tps=20.0, total_latency_s=2.0)


def _failing_above_11_responder(task: object, options: object, call: int) -> SampleResult:
    layers = getattr(options, "gpu_offload_layers", 0) or 0
    if layers >= 11:
        return SampleResult(fail=True)
    return _flat_responder(task, options, call)


def _regressing_responder(task: object, options: object, call: int) -> SampleResult:
    return SampleResult(decode_tps=15.0, total_latency_s=4.0)


class AdvancingHarness(ScriptedHarness):
    """A scripted harness that advances the fake clock so search time is measurable."""

    def __init__(self, responder: Responder, clock: FakeClock) -> None:
        super().__init__(responder)
        self._clock = clock

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
        run = super().run_once(
            task,
            mode=mode,
            sequence=sequence,
            repetition=repetition,
            warmup=warmup,
            options=options,
        )
        self._clock.t += 0.05
        return run


def _monitor(
    workload: ScriptedWorkload,
    *,
    harness: ScriptedHarness,
    approvals: ScriptedApprovalProvider,
    clock: FakeClock,
    sampler: RuntimeTelemetrySampler | None = None,
    policy: RuntimePolicy | None = None,
) -> RuntimeMonitor:
    return RuntimeMonitor(
        workload=workload,
        sampler=sampler or GpuUnderutilizationSampler(),
        harness=harness,
        policy=policy or make_runtime_policy(budget=_budget()),
        approval_provider=approvals,
        hardware=sample_hardware(),
        experiment_policy=build_test_policy(),
        clock=clock,
        sleep=clock.sleep,
    )


def _gpu_workload() -> ScriptedWorkload:
    return ScriptedWorkload(
        model=sample_model(),
        configuration=RuntimeConfiguration(context_length=2048, gpu_offload_layers=8),
    )


# --------------------------------------------------------------------------- #
# Search integration                                                           #
# --------------------------------------------------------------------------- #
def test_gpu_diagnosis_runs_search_and_selects_non_maximum_value() -> None:
    clock = FakeClock()
    workload = _gpu_workload()
    harness = AdvancingHarness(_improving_responder, clock)
    approvals = ScriptedApprovalProvider(
        [ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED]
    )
    monitor = _monitor(workload, harness=harness, approvals=approvals, clock=clock)
    result = monitor.run()
    run = result.run

    assert run.gpu_searches, "the GPU-offload search should have run"
    search = run.gpu_searches[0]
    assert search.resource.sufficient is True
    assert search.resource.current_layers == 8
    assert search.resource.feasible_min == 8
    assert search.resource.feasible_max is not None and search.resource.feasible_max > 8
    # The current configuration is always the baseline measurement.
    assert search.points[0].layers == 8
    assert search.points[0].decision == "baseline"
    assert search.selection.kind is SearchSelectionKind.SELECT
    # Empirically selected (plateau/regression beyond 16 layers), never the maximum.
    assert search.selection.selected_layers == 11
    assert search.selection.selected_layers != search.resource.feasible_max
    # Higher candidates were measured and rejected for objective regression.
    rejected = {point.layers for point in search.points if point.decision == "regression"}
    assert {17, 20, 24} <= rejected

    # The selected value flowed through approval -> experiment -> retain.
    assert run.approvals and run.approvals[0].outcome is ApprovalOutcome.APPROVED
    assert run.interventions, "the selected candidate should have been applied"
    intervention = run.interventions[0]
    assert intervention.parameters.get("gpu_offload_layers") == 11
    assert intervention.retained is True
    assert intervention.decision is ExperimentDecisionKind.IMPROVED
    assert workload.configuration().gpu_offload_layers == 11
    # Context is never changed by the search/intervention.
    assert workload.configuration().context_length == 2048
    # Search time is accounted separately (the harness advances the monitor clock).
    assert run.overhead.gpu_search_s > 0.0


def test_context_is_preserved_across_every_searched_candidate() -> None:
    clock = FakeClock()
    workload = _gpu_workload()
    harness = ScriptedHarness(_improving_responder)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED])
    monitor = _monitor(workload, harness=harness, approvals=approvals, clock=clock)
    run = monitor.run().run
    assert run.gpu_searches
    # Every harness call made during the search used the same configured context.
    search_contexts = {call[0] for call in harness.calls}
    assert search_contexts == {2048}


def test_no_change_search_does_not_approve_apply_or_intervene() -> None:
    clock = FakeClock()
    workload = _gpu_workload()
    harness = ScriptedHarness(_flat_responder)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED])
    monitor = _monitor(workload, harness=harness, approvals=approvals, clock=clock)
    run = monitor.run().run

    assert run.gpu_searches, "the GPU-offload search should have run"
    assert run.gpu_searches[0].selection.kind is SearchSelectionKind.NO_CHANGE
    assert run.interventions == []
    assert run.approvals == []  # NO_CHANGE must not request approval
    assert run.cooldowns, "NO_CHANGE should engage the existing cooldown"
    # Nothing was applied: the workload stays at the baseline offload.
    assert workload.configuration().gpu_offload_layers == 8


def test_selected_candidate_cannot_bypass_approval() -> None:
    clock = FakeClock()
    workload = _gpu_workload()
    harness = ScriptedHarness(_improving_responder)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.REJECTED])
    monitor = _monitor(workload, harness=harness, approvals=approvals, clock=clock)
    run = monitor.run().run

    assert run.gpu_searches and run.gpu_searches[0].selection.kind is SearchSelectionKind.SELECT
    assert run.approvals and run.approvals[0].outcome is ApprovalOutcome.REJECTED
    assert run.interventions == []
    assert workload.configuration().gpu_offload_layers == 8


def test_selected_candidate_cannot_bypass_rollback() -> None:
    clock = FakeClock()
    workload = _gpu_workload()
    # The search selects on the improving harness; the later Phase 7 validation uses a
    # harness that measures no improvement, so the selection must NOT be retained.
    harness = ScriptedHarness(_improving_responder)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED])
    monitor = _monitor(workload, harness=harness, approvals=approvals, clock=clock)
    monitor._harness = ScriptedHarness(_regressing_responder)
    run = monitor.run().run

    assert run.gpu_searches and run.gpu_searches[0].selection.kind is SearchSelectionKind.SELECT
    assert run.interventions
    intervention = run.interventions[0]
    assert intervention.retained is False
    # Known-good baseline restored (no partial application survived).
    assert workload.configuration().gpu_offload_layers == 8


def test_unsafe_candidate_is_rejected_and_search_stops() -> None:
    clock = FakeClock()
    workload = _gpu_workload()
    harness = ScriptedHarness(_failing_above_11_responder)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED])
    monitor = _monitor(workload, harness=harness, approvals=approvals, clock=clock)
    run = monitor.run().run

    assert run.gpu_searches
    search = run.gpu_searches[0]
    measured_layers = {point.layers for point in search.points}
    # 11 was measured (and invalid); higher candidates were not attempted.
    assert 11 in measured_layers
    assert 14 not in measured_layers
    assert search.selection.kind is SearchSelectionKind.NO_CHANGE
    assert run.interventions == []
    assert workload.configuration().gpu_offload_layers == 8


def test_missing_gpu_telemetry_is_conservative() -> None:
    clock = FakeClock()
    workload = _gpu_workload()
    harness = ScriptedHarness(_improving_responder)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED])
    monitor = _monitor(
        workload,
        harness=harness,
        approvals=approvals,
        clock=clock,
        sampler=GpuUnderutilizationSampler(gpu_telemetry=False),
    )
    run = monitor.run().run
    # With no VRAM telemetry the resource envelope is insufficient, so no candidate is
    # eligible and the search is not started (conservative fail-closed).
    assert run.gpu_searches == []
    assert run.interventions == []
    assert workload.configuration().gpu_offload_layers == 8


def test_stale_gpu_telemetry_is_conservative() -> None:
    from datetime import timedelta

    stale = datetime.now(timezone.utc) - timedelta(minutes=10)
    clock = FakeClock()
    workload = _gpu_workload()
    harness = ScriptedHarness(_improving_responder)
    approvals = ScriptedApprovalProvider([ApprovalOutcome.APPROVED])
    monitor = _monitor(
        workload,
        harness=harness,
        approvals=approvals,
        clock=clock,
        sampler=GpuUnderutilizationSampler(observed_at=stale),
    )
    run = monitor.run().run
    assert run.gpu_searches == []
    assert run.interventions == []
    assert workload.configuration().gpu_offload_layers == 8


# --------------------------------------------------------------------------- #
# Resident measurement shim: ownership / no orphan / restoration               #
# --------------------------------------------------------------------------- #
def test_resident_measurement_shim_keeps_target_alive_and_restorable(tmp_path: object) -> None:
    from pathlib import Path

    from inferenceos.optimization.search.evaluation import RuntimeConfigurationPointEvaluator
    from inferenceos.optimization.search.resources import build_resource_envelope
    from inferenceos.optimization.search.policy import SearchPolicy
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.runtime.gpu_search import build_search_harness

    settings = Settings()
    backend = FakeResidentBackend()
    adapter = LlamaCppResidentAdapter(settings=settings, backend=backend)
    # The fake backend uses synthetic PIDs; accept its captured identity so the health
    # probe exercises the ownership/reload path rather than the real OS PID check.
    adapter.verify_process_identity = lambda identity: True  # type: ignore[method-assign]
    model_path = Path(str(tmp_path)) / "resident-search.gguf"
    model_path.write_bytes(b"gguf")
    model = synthetic_model(name="resident-search").model_copy(
        update={"path": str(model_path)}
    )
    baseline = RuntimeConfiguration(context_length=2048, gpu_offload_layers=2)
    workload = ResidentActiveWorkload(
        settings=settings,
        model=model,
        adapter=adapter,
        workload_id="short-short",
        configuration=baseline,
    )
    workload.start()
    assert adapter.is_loaded is True
    pid_before = adapter.pid
    checkpoint = capture_runtime_checkpoint(
        workload,
        reason="test resident search",
        context_identity=config_key(baseline),
    )
    assert checkpoint.verified is True

    assert workload.pause() is True
    harness = build_search_harness(
        workload, settings=settings, hardware=sample_hardware(), injected=None
    )
    envelope = build_resource_envelope(
        total_vram_bytes=4 * _GIB,
        used_vram_bytes=512 * 1024**2,
        telemetry_available=True,
        telemetry_observed_at=datetime.now(timezone.utc),
        telemetry_source="test",
        now=datetime.now(timezone.utc),
        model=model,
        context_length=2048,
        current_layers=2,
        total_layers=model.n_layers,
        policy=SearchPolicy(),
    )
    evaluator = RuntimeConfigurationPointEvaluator(
        profiler=harness,
        model=model,
        workload=get_workload("short-short"),
        experiment_policy=build_test_policy(),
        context_length=2048,
        envelope=envelope,
        base_configuration=baseline,
    )
    measurement = evaluator.measure(3)
    assert measurement.valid is True
    # The resident target stayed loaded and owned across the candidate measurement.
    assert adapter.is_loaded is True
    assert workload.target_health() == TargetHealth.HEALTHY
    assert workload.recovery_records() == []

    # Restore the known-good configuration; the target is still healthy and alive.
    assert restore_runtime_checkpoint(workload, checkpoint) is True
    assert workload.resume() is True
    assert adapter.is_loaded is True
    assert workload.target_health() == TargetHealth.HEALTHY
    assert adapter.pid is not None
    # A controlled reload changed the process but never left an orphan/extra server.
    assert backend.is_running() is True
    workload.stop()
    assert backend.is_running() is False
