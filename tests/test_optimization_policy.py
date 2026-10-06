"""User optimization preference, adaptive economics, search budget and resume tests.

These tests cover the user-controlled optimization policy (BALANCED /
PERFORMANCE_FIRST / MINIMIZE_OPTIMIZATION_COST / CUSTOM), the performance-state driven
adaptive urgency, the bounded search budget, and the explicit search-duration-aware
resume-verification lifecycle that replaces the old fixed 45 s check.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    ApprovalOutcome,
    BottleneckKind,
    DegradedPerformancePolicy,
    InterventionClass,
    OptimizationMode,
    PerformanceState,
    ResumeVerificationOutcome,
    SearchSelectionKind,
)
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.optimization import BenefitEstimate, OptimizationCandidate
from inferenceos.contracts.profiling import ProfilingRun
from inferenceos.contracts.runtime import (
    RuntimeBudget,
    RuntimeObservation,
    RuntimeOpportunity,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.preference import (
    build_optimization_preference,
    candidate_measured_improvement,
    classify_performance_state,
)
from inferenceos.runtime.approval import ScriptedApprovalProvider
from inferenceos.runtime.economics import evaluate_intervention_economics
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.resume import compute_resume_timeout
from inferenceos.runtime.telemetry import RuntimeTelemetrySampler
from inferenceos.runtime.verification import ControlledGpuUnderutilizationDetector
from inferenceos.runtime.workload import ActiveWorkload
from inferenceos.selftest import sample_hardware, sample_model
from tests.experiment_helpers import SampleResult, ScriptedHarness, build_test_policy
from tests.optimization_helpers import make_candidate, synthetic_context
from tests.runtime_helpers import ScriptedWorkload as BaseScriptedWorkload
from tests.runtime_helpers import make_runtime_policy

_GIB = 1024**3


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #
def _perf_candidate(improvement: float) -> OptimizationCandidate:
    ctx = synthetic_context(BottleneckKind.GPU_UNDERUTILIZATION)
    candidate = make_candidate(ctx, action_id="runtime.increase_gpu_offload_layers")
    benefit = BenefitEstimate(
        metric="decode_tokens_per_sec",
        direction="increase",
        unit="tokens/sec",
        known=True,
        magnitude_low=improvement,
        magnitude_high=improvement,
        confidence=0.8,
    )
    return candidate.model_copy(update={"benefit": benefit})


def _opportunity() -> RuntimeOpportunity:
    from inferenceos.contracts.enums import AnomalyClass, EvidenceStrength

    return RuntimeOpportunity(
        opportunity_id="opp",
        anomaly=AnomalyClass.PERSISTENT,
        condition_key="k",
        kind="gpu_underutilization",
        persistence_samples=3,
        deviation_score=0.9,
        confidence=0.85,
        strength=EvidenceStrength.MEDIUM,
    )


def _beneficial(
    candidate: OptimizationCandidate, preference: Any, state: PerformanceState
) -> Any:
    action = build_default_registry().get(candidate.action_id)
    return evaluate_intervention_economics(
        _opportunity(),
        candidate=candidate,
        action=action,
        policy=make_runtime_policy(),
        optimization_preference=preference,
        performance_state=state,
    )


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


class HealthyGpuSampler:
    """Workload performance stays healthy; GPU utilization indicates underutilization.

    Unlike the production double that degrades the workload, this keeps decode/latency
    constant so the performance state stays NORMAL, isolating the economics decision
    from the adaptive-urgency behaviour.
    """

    def __init__(self, *, gpu_telemetry: bool = True) -> None:
        self.count = 0
        self._gpu_telemetry = gpu_telemetry

    def _metrics(self, condition: str) -> dict[str, float]:
        gpu = 62.0 if condition == "baseline" else 5.0
        cpu = 100.0 if condition == "baseline" else 520.0
        # Workload performance (decode/latency/ttft) is held constant so the measured
        # performance state stays NORMAL; a host-side memory deviation triggers the
        # runtime opportunity that the controlled detector attributes to GPU underuse.
        pressure = 0.5 if condition == "baseline" else 0.85
        faults = 0.5 if condition == "baseline" else 4000.0
        return {
            "process_cpu_percent": cpu,
            "gpu_utilization_percent": gpu,
            "gpu_memory_used_bytes": float(512 * 1024**2),
            "decode_tokens_per_sec": 24.0,
            "total_latency_s": 1.4,
            "ttft_s": 0.12,
            "peak_rss_bytes": 1_300_000_000.0,
            "memory_pressure": pressure,
            "hard_page_faults_per_sec": faults,
            "available_ram_bytes": float(8 * _GIB),
            "total_ram_bytes": float(16 * _GIB),
            "system_cpu_percent": 40.0,
        }

    def sample(
        self, workload: ActiveWorkload, *, sequence: int, condition: str
    ) -> RuntimeObservation:
        from inferenceos.contracts.system_state import GPUMemoryState, SystemState
        from inferenceos.runtime.workload import config_key

        self.count += 1
        configuration = workload.configuration()
        snapshot = workload.snapshot()
        descriptor = workload.descriptor()
        now = datetime.now(timezone.utc)
        metrics = dict(self._metrics(condition))
        state = SystemState(
            captured_at=now,
            total_ram_bytes=16 * _GIB,
            available_ram_bytes=8 * _GIB,
            memory_pressure=0.5,
            cpu_percent_total=40.0,
            hard_page_faults_per_sec=0.5,
            gpu_memory=[GPUMemoryState(device_index=0, memory_used_bytes=512 * 1024**2)],
            gpu_telemetry_available=True,
            gpu_observed_at=now,
            gpu_source="test",
        )
        return RuntimeObservation(
            observation_id=f"rtobs-healthy-{self.count}",
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


class AdvancingHarness(ScriptedHarness):
    """Advances the monitor clock on every run so a long search is measurable."""

    def __init__(
        self, responder: Any, clock: _FakeClock, *, seconds_per_run: float = 5.0
    ) -> None:
        super().__init__(responder)
        self._clock = clock
        self._seconds_per_run = seconds_per_run

    def run_once(
        self,
        task: Any,
        *,
        mode: Any,
        sequence: int,
        repetition: int,
        warmup: bool,
        options: Any,
    ) -> ProfilingRun:
        run = super().run_once(
            task,
            mode=mode,
            sequence=sequence,
            repetition=repetition,
            warmup=warmup,
            options=options,
        )
        self._clock.t += self._seconds_per_run
        return run


def _gpu_workload() -> BaseScriptedWorkload:
    return BaseScriptedWorkload(
        model=sample_model(),
        configuration=RuntimeConfiguration(context_length=2048, gpu_offload_layers=8),
    )


def _gpu_budget() -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=400.0,
        telemetry_interval_s=0.5,
        observation_window_size=40,
        baseline_samples=4,
        max_deep_analyses=0,
        max_interventions=2,
        max_cumulative_intervention_s=600.0,
        max_optimizer_overhead_s=300.0,
        approval_timeout_s=10.0,
        max_alerts_per_condition=2,
    )


def _small_improvement_responder(task: Any, options: Any, call: int) -> SampleResult:
    layers = getattr(options, "gpu_offload_layers", 0) or 0
    if layers <= 8:
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)
    # +4% writing speed: +2% on the balanced score (fixed weights, 3 Oct 2026), above
    # Performance first's 1.5% smallest gain and below Balanced's.
    if layers == 11:
        return SampleResult(decode_tps=20.8, total_latency_s=2.0)
    return SampleResult(decode_tps=20.8, total_latency_s=2.4)


def _flat_responder(task: Any, options: Any, call: int) -> SampleResult:
    return SampleResult(decode_tps=20.0, total_latency_s=2.0)


def _improving_responder(task: Any, options: Any, call: int) -> SampleResult:
    layers = getattr(options, "gpu_offload_layers", 0) or 0
    if layers <= 8:
        return SampleResult(decode_tps=20.0, total_latency_s=2.0)
    if layers <= 16:
        return SampleResult(decode_tps=30.0, total_latency_s=2.0)
    return SampleResult(decode_tps=30.5, total_latency_s=3.0)


def _monitor(
    workload: ActiveWorkload,
    *,
    harness: Any,
    clock: Any,
    settings: Settings,
    approvals: ScriptedApprovalProvider,
    sampler: RuntimeTelemetrySampler | None = None,
) -> RuntimeMonitor:
    actual_sampler = sampler or cast(RuntimeTelemetrySampler, HealthyGpuSampler())
    return RuntimeMonitor(
        workload=workload,
        sampler=actual_sampler,
        harness=harness,
        policy=make_runtime_policy(budget=_gpu_budget()),
        approval_provider=approvals,
        detector=ControlledGpuUnderutilizationDetector(),  # type: ignore[arg-type]
        hardware=sample_hardware(),
        experiment_policy=build_test_policy(min_effect=0.01, noise_tolerance=0.01),
        settings=settings,
        clock=clock,
        sleep=clock.sleep,
    )


# --------------------------------------------------------------------------- #
# 1-8  User preference                                                          #
# --------------------------------------------------------------------------- #
def test_default_mode_is_balanced() -> None:
    assert Settings().optimization_mode == "balanced"
    assert build_optimization_preference(Settings()).mode is OptimizationMode.BALANCED


def test_preference_persists_via_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCEOS_OPTIMIZATION_MODE", "performance_first")
    assert build_optimization_preference(Settings()).mode is OptimizationMode.PERFORMANCE_FIRST


def test_performance_first_accepts_small_supported_improvement() -> None:
    preference = build_optimization_preference(
        Settings(optimization_mode="performance_first")
    )
    economics = _beneficial(_perf_candidate(0.03), preference, PerformanceState.NORMAL)
    assert economics.beneficial is True
    assert economics.improvement_justified is True
    assert economics.required_improvement == pytest.approx(0.015)


def test_balanced_rejects_trivial_improvement() -> None:
    preference = build_optimization_preference(Settings(optimization_mode="balanced"))
    economics = _beneficial(_perf_candidate(0.03), preference, PerformanceState.NORMAL)
    assert economics.beneficial is False
    assert economics.improvement_justified is False
    assert economics.required_improvement == pytest.approx(0.05)


def test_minimize_optimization_cost_stays_conservative() -> None:
    preference = build_optimization_preference(
        Settings(optimization_mode="minimize_optimization_cost")
    )
    small = _beneficial(_perf_candidate(0.08), preference, PerformanceState.NORMAL)
    assert small.beneficial is False
    large = _beneficial(_perf_candidate(0.20), preference, PerformanceState.NORMAL)
    assert large.beneficial is True


def test_custom_values_are_respected() -> None:
    settings = Settings(
        optimization_mode="custom",
        optimization_minimum_improvement_percent=2.0,
        optimization_maximum_search_time_s=100.0,
    )
    preference = build_optimization_preference(settings)
    assert preference.mode is OptimizationMode.CUSTOM
    assert preference.minimum_improvement == pytest.approx(0.02)
    assert preference.maximum_search_time_s == pytest.approx(100.0)


def test_invalid_custom_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        build_optimization_preference(
            Settings(optimization_mode="custom", optimization_maximum_search_time_s=100.0)
        )
    with pytest.raises(ValueError):
        build_optimization_preference(
            Settings(
                optimization_mode="custom",
                optimization_minimum_improvement_percent=0.0,
                optimization_maximum_search_time_s=100.0,
            )
        )
    with pytest.raises(ValueError):
        build_optimization_preference(Settings(optimization_mode="not_a_mode"))
    with pytest.raises(ValueError):
        build_optimization_preference(
            Settings(
                optimization_mode="custom",
                optimization_minimum_improvement_percent=5.0,
                optimization_maximum_search_time_s=-1.0,
            )
        )


def test_preference_does_not_weaken_approval_or_safety_classification() -> None:
    registry = build_default_registry()
    policy = make_runtime_policy()
    for mode in ("balanced", "performance_first", "minimize_optimization_cost"):
        preference = build_optimization_preference(Settings(optimization_mode=mode))
        external = _beneficial(_perf_candidate(0.5), preference, PerformanceState.NORMAL)
        # The candidate is a direct action; the external action classification is stable.
        _ = registry
        assert external.requires_approval is True
    opportunity = _opportunity()
    external_econ = evaluate_intervention_economics(
        opportunity,
        candidate=None,
        action=registry.get("system.increase_pagefile"),
        policy=policy,
        optimization_preference=build_optimization_preference(
            Settings(optimization_mode="performance_first")
        ),
        performance_state=PerformanceState.NORMAL,
    )
    assert external_econ.intervention_class is InterventionClass.EXTERNAL_SYSTEM_WIDE
    assert external_econ.requires_approval is True


# --------------------------------------------------------------------------- #
# 9-13  Performance urgency                                                     #
# --------------------------------------------------------------------------- #
def test_normal_state_uses_normal_economics() -> None:
    from inferenceos.contracts.preference import OptimizationPreference

    preference = OptimizationPreference(minimum_improvement=0.05)
    assert preference.minimum_improvement_for(PerformanceState.NORMAL) == pytest.approx(0.05)


def test_degraded_state_lowers_requirement() -> None:
    preference = build_optimization_preference(Settings(optimization_mode="balanced"))
    assert preference.minimum_improvement_for(PerformanceState.DEGRADED) < (
        preference.minimum_improvement_for(PerformanceState.NORMAL)
    )


def test_severely_degraded_allows_small_safe_improvement() -> None:
    preference = build_optimization_preference(
        Settings(optimization_mode="performance_first")
    )
    economics = _beneficial(
        _perf_candidate(0.005), preference, PerformanceState.SEVERELY_DEGRADED
    )
    assert economics.beneficial is True


def test_unknown_state_does_not_guess() -> None:
    preference = build_optimization_preference(Settings(optimization_mode="balanced"))
    assessment = classify_performance_state(
        baseline_metrics={},
        recent_metrics={"decode_tokens_per_sec": 6.0},
        preference=preference,
    )
    assert assessment.state is PerformanceState.UNKNOWN
    assert assessment.measured is False
    # UNKNOWN falls back to the normal requirement, not a degraded one.
    assert preference.minimum_improvement_for(assessment.state) == pytest.approx(0.05)


def test_performance_state_is_classified_from_measured_metrics() -> None:
    preference = build_optimization_preference(Settings(optimization_mode="balanced"))
    normal = classify_performance_state(
        baseline_metrics={"decode_tokens_per_sec": 24.0},
        recent_metrics={"decode_tokens_per_sec": 23.0},
        preference=preference,
    )
    assert normal.state is PerformanceState.NORMAL
    degraded = classify_performance_state(
        baseline_metrics={"decode_tokens_per_sec": 24.0},
        recent_metrics={"decode_tokens_per_sec": 16.0},
        preference=preference,
    )
    assert degraded.state is PerformanceState.DEGRADED
    severe = classify_performance_state(
        baseline_metrics={"total_latency_s": 2.0},
        recent_metrics={"total_latency_s": 4.0},
        preference=preference,
    )
    assert severe.state is PerformanceState.SEVERELY_DEGRADED


def test_candidate_measured_improvement_ignores_byte_benefits() -> None:
    ctx = synthetic_context(BottleneckKind.MODEL_FOOTPRINT)
    byte_candidate = make_candidate(ctx, action_id="runtime.enable_mmap")
    assert candidate_measured_improvement(byte_candidate) is None
    assert candidate_measured_improvement(_perf_candidate(0.02)) == pytest.approx(0.02)


# --------------------------------------------------------------------------- #
# 14-17  Search budget                                                          #
# --------------------------------------------------------------------------- #
def test_each_mode_produces_a_bounded_search_budget() -> None:
    settings = Settings(phase10_gpu_search_time_limit_s=240.0)
    balanced = build_optimization_preference(settings).search_budget_for(
        PerformanceState.NORMAL
    )
    performance = build_optimization_preference(
        Settings(phase10_gpu_search_time_limit_s=240.0, optimization_mode="performance_first")
    ).search_budget_for(PerformanceState.NORMAL)
    minimize = build_optimization_preference(
        Settings(
            phase10_gpu_search_time_limit_s=240.0,
            optimization_mode="minimize_optimization_cost",
        )
    ).search_budget_for(PerformanceState.NORMAL)
    assert balanced == pytest.approx(240.0)
    assert performance > balanced
    assert minimize < balanced
    hard_cap = settings.optimization_max_search_time_hard_cap_s
    for budget in (balanced, performance, minimize):
        assert 0 < budget <= hard_cap


def test_custom_search_budget_is_respected() -> None:
    preference = build_optimization_preference(
        Settings(
            optimization_mode="custom",
            optimization_minimum_improvement_percent=3.0,
            optimization_maximum_search_time_s=100.0,
        )
    )
    assert preference.search_budget_for(PerformanceState.NORMAL) == pytest.approx(100.0)


def test_there_is_no_unlimited_search() -> None:
    preference = build_optimization_preference(
        Settings(
            optimization_mode="custom",
            optimization_minimum_improvement_percent=3.0,
            optimization_maximum_search_time_s=999999.0,
        )
    )
    assert preference.maximum_search_time_s == pytest.approx(
        Settings().optimization_max_search_time_hard_cap_s
    )
    # Even a severely degraded state cannot exceed the hard cap.
    assert preference.search_budget_for(PerformanceState.SEVERELY_DEGRADED) <= (
        preference.maximum_search_time_hard_cap_s
    )


# --------------------------------------------------------------------------- #
# 18-26  Resume verification                                                    #
# --------------------------------------------------------------------------- #
def test_short_search_resume_timeout_is_base_allowance() -> None:
    timeout = compute_resume_timeout(
        search_duration_s=0.0,
        configured_search_budget_s=240.0,
        base_allowance_s=30.0,
        search_factor=0.25,
        max_search_allowance_s=120.0,
        hard_max_s=300.0,
    )
    assert timeout == pytest.approx(30.0)


def test_long_search_resume_timeout_exceeds_forty_five_seconds() -> None:
    timeout = compute_resume_timeout(
        search_duration_s=300.0,
        configured_search_budget_s=240.0,
        base_allowance_s=30.0,
        search_factor=0.25,
        max_search_allowance_s=120.0,
        hard_max_s=300.0,
    )
    assert timeout > 45.0
    assert timeout == pytest.approx(30.0 + 75.0)


def test_resume_timeout_is_bounded_by_hard_maximum() -> None:
    timeout = compute_resume_timeout(
        search_duration_s=1_000_000.0,
        configured_search_budget_s=240.0,
        base_allowance_s=30.0,
        search_factor=0.25,
        max_search_allowance_s=120.0,
        hard_max_s=300.0,
    )
    assert timeout == pytest.approx(150.0)
    assert timeout <= 300.0


class _DelayedResumeWorkload(BaseScriptedWorkload):
    def __init__(self, *, fail_first: int = 0, **kwargs: Any) -> None:
        kwargs.setdefault(
            "configuration", RuntimeConfiguration(context_length=2048, gpu_offload_layers=8)
        )
        super().__init__(**kwargs)
        self.verify_calls = 0
        self._fail_first = fail_first

    def verify_resume(self, *, previous_requests: int, timeout_s: float) -> bool:
        self.verify_calls += 1
        return self.verify_calls > self._fail_first


class _DisappearingWorkload(BaseScriptedWorkload):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault(
            "configuration", RuntimeConfiguration(context_length=2048, gpu_offload_layers=8)
        )
        super().__init__(**kwargs)

    def verify_resume(self, *, previous_requests: int, timeout_s: float) -> bool:
        self.terminate_externally()
        return False


def test_temporary_readiness_delay_retries_then_verifies() -> None:
    clock = _FakeClock()
    workload = _DelayedResumeWorkload(fail_first=1)
    monitor = _monitor(
        workload,
        harness=ScriptedHarness(_improving_responder),
        clock=clock,
        settings=Settings(),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED]),
    )
    result = monitor.run()
    assert result.termination.reason.value != "safety_terminal"
    records = result.run.resume_verifications
    assert records and records[0].outcome is ResumeVerificationOutcome.VERIFIED
    assert records[0].attempts >= 2


def test_long_search_resumes_without_safety_terminal() -> None:
    clock = _FakeClock()
    workload = _gpu_workload()
    harness = AdvancingHarness(_improving_responder, clock, seconds_per_run=20.0)
    monitor = _monitor(
        workload,
        harness=harness,
        clock=clock,
        settings=Settings(),
        approvals=ScriptedApprovalProvider(
            [ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED]
        ),
    )
    result = monitor.run()
    assert result.termination.reason.value != "safety_terminal"
    records = result.run.resume_verifications
    assert records, "a resume verification record should have been produced"
    verification = records[0]
    assert verification.outcome is ResumeVerificationOutcome.VERIFIED
    assert verification.search_duration_s > 45.0
    assert verification.timeout_s > 45.0
    assert workload.start_calls >= 1


def test_process_disappearance_remains_terminal() -> None:
    clock = _FakeClock()
    workload = _DisappearingWorkload()
    monitor = _monitor(
        workload,
        harness=ScriptedHarness(_improving_responder),
        clock=clock,
        settings=Settings(),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED]),
    )
    result = monitor.run()
    assert result.termination.reason.value == "safety_terminal"
    assert result.termination.safety is True


class _ChangingIdentityWorkload(BaseScriptedWorkload):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault(
            "configuration", RuntimeConfiguration(context_length=2048, gpu_offload_layers=8)
        )
        super().__init__(**kwargs)
        self._create_time = 1.0

    def target_identity(self) -> Any:
        return SimpleNamespace(pid=424242, create_time=self._create_time)

    def verify_resume(self, *, previous_requests: int, timeout_s: float) -> bool:
        self._create_time = 2.0
        return False


def test_identity_change_remains_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _FakeClock()
    workload = _ChangingIdentityWorkload()
    monkeypatch.setattr(
        "inferenceos.runtime.monitor.process_identity",
        lambda pid: SimpleNamespace(pid=pid, create_time=1.0),
    )
    monitor = _monitor(
        workload,
        harness=ScriptedHarness(_improving_responder),
        clock=clock,
        settings=Settings(),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED]),
    )
    result = monitor.run()
    assert result.termination.reason.value == "safety_terminal"
    assert result.run.resume_verifications
    assert result.run.resume_verifications[0].outcome is ResumeVerificationOutcome.TERMINAL


def test_no_orphan_thread_after_monitor_run() -> None:
    clock = _FakeClock()
    before = threading.active_count()
    monitor = _monitor(
        _gpu_workload(),
        harness=ScriptedHarness(_improving_responder),
        clock=clock,
        settings=Settings(),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED]),
    )
    monitor.run()
    assert threading.active_count() <= before + 1


# --------------------------------------------------------------------------- #
# 27-31  Integration: preference + economics + apply/retention                  #
# --------------------------------------------------------------------------- #
def test_balanced_rejects_small_supported_improvement() -> None:
    clock = _FakeClock()
    workload = _gpu_workload()
    monitor = _monitor(
        workload,
        harness=ScriptedHarness(_small_improvement_responder),
        clock=clock,
        settings=Settings(optimization_mode="balanced"),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED]),
    )
    result = monitor.run()
    run = result.run
    assert run.gpu_searches
    assert run.gpu_searches[0].selection.kind is SearchSelectionKind.SELECT
    assert run.interventions == []
    assert workload.configuration().gpu_offload_layers == 8
    assert run.optimization_preference is not None
    assert run.optimization_preference.mode is OptimizationMode.BALANCED
    assert run.performance_state is not None
    assert run.performance_state.state is PerformanceState.NORMAL


def test_performance_first_accepts_small_supported_improvement_end_to_end() -> None:
    clock = _FakeClock()
    workload = _gpu_workload()
    monitor = _monitor(
        workload,
        harness=ScriptedHarness(_small_improvement_responder),
        clock=clock,
        settings=Settings(optimization_mode="performance_first"),
        approvals=ScriptedApprovalProvider(
            [ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED]
        ),
    )
    result = monitor.run()
    run = result.run
    assert run.gpu_searches
    assert run.gpu_searches[0].selection.kind is SearchSelectionKind.SELECT
    assert run.interventions, "PERFORMANCE_FIRST should accept the small improvement"
    assert run.interventions[0].retained is True
    assert workload.configuration().gpu_offload_layers == 11
    assert run.gpu_searches[0].optimization_mode == "performance_first"


def test_balanced_full_intervention_is_retained_and_preference_recorded() -> None:
    clock = _FakeClock()
    workload = _gpu_workload()
    monitor = _monitor(
        workload,
        harness=ScriptedHarness(_improving_responder),
        clock=clock,
        settings=Settings(optimization_mode="balanced"),
        approvals=ScriptedApprovalProvider(
            [ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED]
        ),
    )
    result = monitor.run()
    run = result.run
    assert run.optimization_preference is not None
    assert run.gpu_searches and run.gpu_searches[0].candidates_evaluated > 0
    assert run.gpu_searches[0].search_budget_s is not None
    assert run.gpu_searches[0].optimization_mode == "balanced"
    assert run.performance_state is not None
    assert run.interventions
    assert run.interventions[0].retained is True
    assert workload.configuration().gpu_offload_layers == 11


def test_no_change_remains_no_change_with_preference() -> None:
    clock = _FakeClock()
    workload = _gpu_workload()
    monitor = _monitor(
        workload,
        harness=ScriptedHarness(_flat_responder),
        clock=clock,
        settings=Settings(optimization_mode="performance_first"),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED]),
    )
    result = monitor.run()
    run = result.run
    assert run.gpu_searches
    assert run.gpu_searches[0].selection.kind is SearchSelectionKind.NO_CHANGE
    assert run.interventions == []
    assert workload.configuration().gpu_offload_layers == 8
