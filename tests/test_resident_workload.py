"""Phase 10R resident workload, session provenance, approval, controlled detector."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from inferenceos.config import Settings
from inferenceos.contracts.enums import ApprovalOutcome
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.lifecycle import ProcessIdentity, ResidentGenerationMeasurement
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.runtime import RuntimeBudget
from inferenceos.interfaces.backend_adapter import (
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.runtime.approval import CallbackApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.runtime.verification import (
    ControlledFootprintDetector,
    ScriptedRuntimeSampler,
    ScriptedWorkload,
    healthy_metrics,
)
from inferenceos.selftest import sample_hardware, sample_model
from tests.diagnosis_fixtures import make_report, make_run
from tests.runtime_helpers import make_runtime_policy


class FakeResidentAdapter:
    """Deterministic stand-in for the resident llama-server adapter."""

    def __init__(self, *, generate_delay_s: float = 0.05) -> None:
        self._loaded = False
        self._params = RuntimeParams()
        self._pid = 5000
        self._identity = ProcessIdentity(pid=self._pid, create_time=1.0)
        self._last_generation: ResidentGenerationMeasurement | None = None
        self._delay = generate_delay_s
        self.loads = 0
        self.reloads = 0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def identity(self) -> ProcessIdentity | None:
        return self._identity

    @property
    def last_generation(self) -> ResidentGenerationMeasurement | None:
        return self._last_generation

    @property
    def params(self) -> RuntimeParams:
        return self._params

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        self.loads += 1
        self._loaded = True
        self._params = params
        self._pid += 1
        self._identity = ProcessIdentity(pid=self._pid, create_time=float(self._pid))

    def unload(self) -> None:
        self._loaded = False

    def set_runtime_params(self, params: RuntimeParams) -> None:
        self.reloads += 1
        updates = {name: getattr(params, name) for name in params.model_fields_set}
        self._params = self._params.model_copy(update=updates)
        self._pid += 1
        self._identity = ProcessIdentity(pid=self._pid, create_time=float(self._pid))

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._loaded:
            raise RuntimeError("not loaded")
        time.sleep(self._delay)
        now = datetime.now(timezone.utc)
        self._last_generation = ResidentGenerationMeasurement(
            submitted_at=now,
            first_token_at=now,
            completed_at=now,
            ttft_s=self._delay / 2.0,
            total_s=self._delay,
            generated_tokens=request.n_predict,
            prompt_tokens=8,
            decode_tokens_per_sec=20.0,
            streamed=True,
            text="ok",
        )
        return GenerationResult(
            text="ok",
            generated_tokens=request.n_predict,
            prompt_tokens=8,
            decode_tokens_per_sec=20.0,
            ttft_s=self._delay / 2.0,
            total_s=self._delay,
            peak_rss_bytes=1_000_000_000,
        )


def _resident(*, delay: float = 0.05) -> tuple[ResidentActiveWorkload, FakeResidentAdapter]:
    adapter = FakeResidentAdapter(generate_delay_s=delay)
    workload = ResidentActiveWorkload(
        settings=Settings(),
        model=sample_model(),
        adapter=adapter,
        workload_id="short-short",
        configuration=RuntimeConfiguration(context_length=2048, gpu_offload_layers=0),
        request_gap_s=0.01,
    )
    return workload, adapter


def test_resident_workload_target_is_distinct_and_serves_requests() -> None:
    workload, adapter = _resident()
    workload.start()
    try:
        deadline = time.perf_counter() + 5.0
        while workload.completed_requests() < 2 and time.perf_counter() < deadline:
            time.sleep(0.05)
        assert workload.completed_requests() >= 2
        assert workload.target_pid() == adapter.pid
        assert workload.target_pid() != os.getpid()
        identity = workload.target_identity()
        assert identity is not None and identity.pid == adapter.pid
        records = workload.request_records()
        assert records
        for record in records:
            assert record.completed_at is not None
            assert record.submitted_at <= record.completed_at
            assert record.success is True
    finally:
        workload.stop()


def test_resident_workload_pause_drains_and_resume_restarts() -> None:
    workload, _adapter = _resident(delay=0.2)
    workload.start()
    try:
        deadline = time.perf_counter() + 5.0
        while workload.completed_requests() < 1 and time.perf_counter() < deadline:
            time.sleep(0.05)
        assert workload.pause() is True
        assert workload.snapshot().generating is False
        paused_at = workload.completed_requests()
        time.sleep(0.4)
        assert workload.completed_requests() == paused_at
        assert workload.resume() is True
        deadline = time.perf_counter() + 5.0
        while workload.completed_requests() <= paused_at and time.perf_counter() < deadline:
            time.sleep(0.05)
        assert workload.completed_requests() > paused_at
    finally:
        workload.stop()


def test_resident_workload_set_configuration_reloads_target() -> None:
    workload, adapter = _resident()
    workload.start()
    try:
        reloads_before = adapter.reloads
        workload.set_configuration(
            RuntimeConfiguration(context_length=512, gpu_offload_layers=0)
        )
        assert adapter.reloads == reloads_before + 1
        assert adapter.params.context_length == 512
    finally:
        workload.stop()


def test_callback_approval_provider_is_a_real_decision_boundary() -> None:
    from inferenceos.contracts.runtime import RuntimeRecommendation

    seen: list[str] = []

    def decide(recommendation: RuntimeRecommendation) -> ApprovalOutcome:
        seen.append(recommendation.proposal_id)
        return ApprovalOutcome.REJECTED

    provider = CallbackApprovalProvider(decide)
    recommendation = RuntimeRecommendation(
        proposal_id="prop-1",
        opportunity_id="opp-1",
        condition="test",
        why_it_matters="x",
        analysis_summary="y",
    )
    record = provider.request(recommendation, timeout_s=1.0)
    assert record.outcome is ApprovalOutcome.REJECTED
    assert record.proposal_id == "prop-1"
    assert seen == ["prop-1"]


def test_controlled_detector_grounds_on_real_rss() -> None:
    metrics = {
        "peak_rss_bytes": 1_500_000_000,
        "available_ram_bytes": 8 * 1024**3,
        "total_ram_bytes": 16 * 1024**3,
        "decode_tokens_per_sec": 20.0,
        "hard_page_faults_per_sec": 0.5,
    }
    report = make_report([make_run(metrics=dict(metrics), condition="a")])
    forced = ControlledFootprintDetector(force=True).detect(report)
    assert forced.meaningful_opportunity is True
    assert forced.primary is not None
    assert forced.primary.evidence[0].observed_value == 1_500_000_000
    real = ControlledFootprintDetector(force=False).detect(report)
    assert real.meaningful_opportunity is False or real.insufficient_evidence


class _PidWorkload(ScriptedWorkload):
    """Scripted workload that advertises a distinct target PID."""

    def target_pid(self) -> int | None:
        return os.getpid() + 12345


def _short_budget() -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=3.0,
        telemetry_interval_s=0.5,
        observation_window_size=20,
        baseline_samples=2,
        max_deep_analyses=1,
        max_interventions=1,
        max_cumulative_intervention_s=60.0,
        max_optimizer_overhead_s=60.0,
        approval_timeout_s=5.0,
        max_alerts_per_condition=1,
    )


def test_monitor_session_provenance_records_distinct_target_and_run_id() -> None:
    sampler = ScriptedRuntimeSampler(healthy_metrics)
    monitor = RuntimeMonitor(
        workload=_PidWorkload(),
        sampler=sampler,
        policy=make_runtime_policy(budget=_short_budget()),
        approval_provider=CallbackApprovalProvider(lambda _rec: ApprovalOutcome.REJECTED),
        hardware=sample_hardware(),
        run_id="rtsession-test-run",
        session_id="rtsession-test",
    )
    result = monitor.run()
    session = result.run.session
    assert session is not None
    assert session.run_id == "rtsession-test-run"
    assert session.session_id == "rtsession-test"
    assert session.monitor_pid == os.getpid()
    assert session.target_pid == os.getpid() + 12345
    assert session.distinct_processes is True
    assert result.run.run_id == "rtsession-test-run"
