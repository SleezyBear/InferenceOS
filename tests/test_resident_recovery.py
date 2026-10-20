"""B2 resident-server crash detection and bounded recovery tests.

These use a deterministic, identity-aware fake adapter so the recovery state machine
is exercised without starting a real server. The identity-mismatch test deliberately
feeds a controlled identity fixture so the PID-reuse safety logic is proven without
depending on a real Windows PID reuse event.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone

import pytest

from inferenceos.runtime import resident_workload as resident_module
from inferenceos.backends.process_control import identity_matches, process_identity
from inferenceos.config import Settings
from inferenceos.contracts.enums import ApprovalOutcome
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.lifecycle import (
    ProcessIdentity,
    ResidentGenerationMeasurement,
    ResidentLoadMeasurement,
    ResidentRecoveryState,
    ResidentTargetRecovery,
    ResourceSnapshot,
    TargetHealth,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.runtime import RuntimeBudget
from inferenceos.interfaces.backend_adapter import (
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.runtime.approval import CallbackApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.runtime.verification import (
    ScriptedRuntimeSampler,
    ScriptedWorkload,
    healthy_metrics,
)
from inferenceos.selftest import sample_hardware, sample_model
from tests.runtime_helpers import make_runtime_policy


def _snapshot(pid: int | None) -> ResourceSnapshot:
    return ResourceSnapshot(pid=pid, exists=pid is not None, rss_bytes=1024 if pid else None)


class RecoverableFakeAdapter:
    """Identity-aware fake resident adapter for bounded-recovery tests."""

    def __init__(self) -> None:
        self._loaded = False
        self._pid = 7000
        self._create_time = 1.0
        self._params = RuntimeParams()
        self._last_load: ResidentLoadMeasurement | None = None
        self._last_generation: ResidentGenerationMeasurement | None = None
        self.load_calls = 0
        self.cleanup_calls = 0
        self.fail_load_times = 0
        self.ready = True
        self.param_overrides: dict[str, object] = {}
        self.process_pids: set[int] = set()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def identity(self) -> ProcessIdentity | None:
        if not self._loaded:
            return None
        return ProcessIdentity(pid=self._pid, create_time=self._create_time)

    @property
    def last_generation(self) -> ResidentGenerationMeasurement | None:
        return self._last_generation

    @property
    def params(self) -> RuntimeParams:
        return self._params

    @property
    def last_load(self) -> ResidentLoadMeasurement | None:
        return self._last_load

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        self.load_calls += 1
        if self.fail_load_times > 0:
            self.fail_load_times -= 1
            raise RuntimeError("injected reload failure")
        self._loaded = True
        self._pid += 1
        self.process_pids.add(self._pid)
        self._create_time = float(self._pid)
        self._params = params.model_copy(update=self.param_overrides)
        self._last_load = ResidentLoadMeasurement(
            model_ref=model.name,
            model_path=model.path,
            pid=self._pid,
            identity=self.identity,
            host="127.0.0.1",
            port=19000 + self._pid,
            ready=self.ready,
            startup_s=0.1,
            ready_s=0.1 if self.ready else None,
            before=_snapshot(None),
            after=_snapshot(self._pid),
            resident_verified=True,
        )

    def unload(self) -> None:
        self._loaded = False

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        self._loaded = False

    def telemetry(self) -> BackendTelemetry:
        # Synthetic engine observations, separate from the public requested params.
        return BackendTelemetry(gpu_layers_offloaded=self._params.gpu_offload_layers,
                                observed={"context_length": self._params.context_length})

    def set_runtime_params(self, params: RuntimeParams) -> None:
        self._params = self._params.model_copy(
            update={name: getattr(params, name) for name in params.model_fields_set}
        )

    def verify_process_identity(self, identity: ProcessIdentity) -> bool:
        return bool(
            self._loaded
            and identity.pid == self._pid
            and abs(identity.create_time - self._create_time) <= 0.01
        )

    def kill(self) -> None:
        """Simulate the real server process exiting."""
        self._loaded = False

    def process_exists(self, pid: int) -> bool:
        """Synthetic process table, independent of host PID occupancy."""
        return pid in self.process_pids and self._loaded and self._pid == pid

    def simulate_pid_reuse(self) -> None:
        """Keep an occupied PID but give it a different creation time."""
        self._create_time += 1000.0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._loaded:
            raise RuntimeError("not loaded")
        now = datetime.now(timezone.utc)
        self._last_generation = ResidentGenerationMeasurement(
            submitted_at=now,
            first_token_at=now,
            completed_at=now,
            ttft_s=0.01,
            total_s=0.02,
            generated_tokens=request.n_predict,
            prompt_tokens=4,
            decode_tokens_per_sec=50.0,
            streamed=True,
            text="ok",
        )
        return GenerationResult(
            text="ok",
            generated_tokens=request.n_predict,
            prompt_tokens=4,
            decode_tokens_per_sec=50.0,
            ttft_s=0.01,
            total_s=0.02,
        )


def _workload(adapter: RecoverableFakeAdapter) -> ResidentActiveWorkload:
    return ResidentActiveWorkload(
        settings=Settings(),
        model=sample_model(),
        adapter=adapter,
        workload_id="short-short",
        configuration=RuntimeConfiguration(context_length=2048, gpu_offload_layers=0),
        request_gap_s=0.01,
    )


def _provision(workload: ResidentActiveWorkload) -> None:
    """Provision the target without starting the serve thread (deterministic)."""
    workload._ensure_loaded()  # noqa: SLF001 - deliberate unit-level setup
    workload._capture_target()  # noqa: SLF001


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> RecoverableFakeAdapter:
    fake = RecoverableFakeAdapter()
    host_exists = resident_module.process_exists

    def exists(pid: int) -> bool:
        if pid in fake.process_pids:
            return fake.process_exists(pid)
        return host_exists(pid)

    # Patch only this fake's registered PIDs, never every process or production's
    # conservative exit check. Unknown PIDs still consult the real host evidence.
    monkeypatch.setattr(resident_module, "process_exists", exists)
    return fake


def test_identity_match_and_controlled_mismatch() -> None:
    live = process_identity(os.getpid())
    assert live is not None
    assert identity_matches(live) is True

    stale = live.model_copy(update={"create_time": live.create_time - 1000.0})
    assert identity_matches(stale) is False


def test_target_exit_detected_and_lost(adapter: RecoverableFakeAdapter) -> None:
    workload = _workload(adapter)
    _provision(workload)
    captured = workload.target_identity()
    assert captured is not None
    assert workload.target_health() == TargetHealth.HEALTHY

    adapter.kill()
    assert workload.target_health() == TargetHealth.LOST
    assert workload.target_alive() is False
    # The captured identity survives the loss so the loss is detectable.
    assert workload.target_identity() == captured


def test_identity_mismatch_is_treated_as_lost_not_a_new_target(adapter: RecoverableFakeAdapter) -> None:
    workload = _workload(adapter)
    _provision(workload)
    adapter.simulate_pid_reuse()
    # The adapter still reports "loaded" but the captured identity no longer matches
    # the process now at that PID; it must not be accepted as the original target.
    assert adapter.is_loaded is True
    assert workload.target_health() == TargetHealth.LOST


def test_successful_bounded_recovery_verifies_everything(adapter: RecoverableFakeAdapter) -> None:
    workload = _workload(adapter)
    _provision(workload)
    old = workload.target_identity()
    assert old is not None

    adapter.kill()
    recovery = workload.recover_target(reason="process_exited")

    assert recovery.attempted is True
    assert recovery.attempt_count == 1
    assert recovery.recovered is True
    assert recovery.unrecoverable is False
    assert recovery.new_identity is not None
    assert (recovery.new_identity.pid, recovery.new_identity.create_time) != (
        old.pid,
        old.create_time,
    )
    assert recovery.identity_verified is True
    assert recovery.readiness_verified is True
    assert recovery.configuration_verified is True
    assert recovery.model_verified is True
    assert recovery.backend_verified is True
    assert recovery.orphan_free is True
    assert adapter.cleanup_calls >= 1
    states = [event.state for event in recovery.events]
    assert ResidentRecoveryState.TARGET_LOST in states
    assert ResidentRecoveryState.TARGET_RECOVERED in states
    assert workload.target_health() == TargetHealth.HEALTHY
    # Real inference works again after recovery.
    result = adapter.generate(GenerationRequest(prompt="hi", n_predict=2))
    assert result.text == "ok"


@pytest.mark.parametrize("host_unknown", [False, True])
def test_fake_recovery_is_independent_of_host_pid_occupancy(
    adapter: RecoverableFakeAdapter, monkeypatch: pytest.MonkeyPatch, host_unknown: bool,
) -> None:
    from inferenceos.backends import process_control

    assert process_control.psutil is not None

    def occupied(_pid: int) -> bool:
        if host_unknown:
            raise OSError("host existence query unavailable")
        return True

    monkeypatch.setattr(process_control.psutil, "pid_exists", occupied)
    workload = _workload(adapter)
    _provision(workload)
    old = adapter.pid
    assert old is not None and resident_module.process_exists(old)
    adapter.kill()
    assert not resident_module.process_exists(old)
    recovery = workload.recover_target(reason="process_exited")
    assert recovery.recovered and recovery.orphan_free
    assert adapter.pid is not None and resident_module.process_exists(adapter.pid)
    # Only registered fake PIDs are isolated; real/unknown identities still fail
    # closed, even when the OS query itself fails.
    assert resident_module.process_exists(123456789)
    assert not workload._no_orphan(123456789)
    adapter.cleanup()
    assert not resident_module.process_exists(adapter.pid)


def test_recovery_is_bounded_and_never_retried_after_failure(adapter: RecoverableFakeAdapter) -> None:
    workload = _workload(adapter)
    _provision(workload)
    adapter.kill()
    adapter.fail_load_times = 5
    load_calls_before = adapter.load_calls

    first = workload.recover_target(reason="process_exited")
    assert first.attempted is True
    assert first.attempt_count == 1
    assert first.recovered is False
    assert first.unrecoverable is True
    assert adapter.load_calls == load_calls_before + 1
    assert ResidentRecoveryState.TARGET_UNRECOVERABLE in [
        event.state for event in first.events
    ]

    # A second request for recovery must not start a second attempt.
    second = workload.recover_target(reason="process_exited")
    assert second.attempted is False
    assert second.recovered is False
    assert second.unrecoverable is True
    assert adapter.load_calls == load_calls_before + 1
    assert workload.unrecoverable is True
    assert workload.target_health() == TargetHealth.UNRECOVERABLE


def test_readiness_failure_makes_target_unrecoverable(adapter: RecoverableFakeAdapter) -> None:
    workload = _workload(adapter)
    _provision(workload)
    adapter.kill()
    adapter.ready = False

    recovery = workload.recover_target(reason="process_exited")
    assert recovery.recovered is False
    assert recovery.unrecoverable is True
    assert recovery.readiness_verified is False
    assert workload.target_health() == TargetHealth.UNRECOVERABLE


def test_configuration_mismatch_makes_target_unrecoverable(adapter: RecoverableFakeAdapter) -> None:
    workload = _workload(adapter)
    _provision(workload)
    adapter.kill()
    adapter.param_overrides = {"context_length": 123}

    recovery = workload.recover_target(reason="process_exited")
    assert recovery.recovered is False
    assert recovery.unrecoverable is True
    assert recovery.configuration_verified is False


def test_cleanup_runs_on_failed_recovery(adapter: RecoverableFakeAdapter) -> None:
    workload = _workload(adapter)
    _provision(workload)
    adapter.kill()
    adapter.fail_load_times = 5
    cleanup_before = adapter.cleanup_calls

    workload.recover_target(reason="process_exited")
    assert adapter.cleanup_calls >= cleanup_before + 1


def test_no_implicit_reload_after_unrecoverable(adapter: RecoverableFakeAdapter) -> None:
    workload = _workload(adapter)
    _provision(workload)
    adapter.kill()
    adapter.fail_load_times = 5
    workload.recover_target(reason="process_exited")
    load_calls = adapter.load_calls

    # The initial-provisioning path must not silently reload an unrecoverable target.
    try:
        workload._ensure_loaded()  # noqa: SLF001
        raised = False
    except Exception:
        raised = True
    assert raised is True
    assert adapter.load_calls == load_calls


class _RecoveringScriptedWorkload(ScriptedWorkload):
    """Scripted workload that reports a target whose identity changes once."""

    def __init__(self) -> None:
        super().__init__()
        live = process_identity(os.getpid())
        assert live is not None
        self._live_identity = live
        self._lost = False
        self._recovered = False
        self.recovery_requests = 0

    def target_identity(self) -> ProcessIdentity | None:
        if self._lost and not self._recovered:
            # Same PID, different creation time: PID reuse / identity change.
            return self._live_identity.model_copy(
                update={"create_time": self._live_identity.create_time - 1000.0}
            )
        return self._live_identity

    def target_pid(self) -> int | None:
        return self._live_identity.pid

    def lose_target(self) -> None:
        self._lost = True
        self._recovered = False

    def recover_target(self, *, reason: str) -> ResidentTargetRecovery:
        self.recovery_requests += 1
        self._recovered = True
        now = datetime.now(timezone.utc)
        return ResidentTargetRecovery(
            recovery_id="rrec-scripted",
            reason=reason,
            started_at=now,
            finished_at=now,
            attempted=True,
            attempt_count=1,
            recovered=True,
            identity_verified=True,
            readiness_verified=True,
            configuration_verified=True,
            model_verified=True,
            backend_verified=True,
            orphan_free=True,
        )


def _budget() -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=4.0,
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


def test_monitor_detects_loss_recovers_once_and_continues() -> None:
    workload = _RecoveringScriptedWorkload()
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=ScriptedRuntimeSampler(healthy_metrics),
        policy=make_runtime_policy(budget=_budget()),
        approval_provider=CallbackApprovalProvider(lambda _rec: ApprovalOutcome.REJECTED),
        hardware=sample_hardware(),
        run_id="b2-monitor-recovery",
        session_id="b2-monitor-recovery",
    )

    def _lose_after_a_moment() -> None:
        time.sleep(0.3)
        workload.lose_target()

    killer = threading.Thread(target=_lose_after_a_moment, daemon=True)
    killer.start()
    result = monitor.run()
    killer.join(timeout=2.0)

    assert workload.recovery_requests == 1
    assert result.run.target_recoveries
    recovery = result.run.target_recoveries[0]
    assert recovery.recovered is True
    assert recovery.attempt_count == 1
    # The monitor recorded the explicit lifecycle states and kept monitoring.
    states = [event.state.value for event in result.run.states]
    assert "target_lost" in states
    assert "target_recovered" in states
    assert result.final_state.value != "target_unrecoverable"


def test_monitor_terminates_safety_when_recovery_fails() -> None:
    class _FailingRecoveryWorkload(_RecoveringScriptedWorkload):
        def recover_target(self, *, reason: str) -> ResidentTargetRecovery:
            self.recovery_requests += 1
            now = datetime.now(timezone.utc)
            return ResidentTargetRecovery(
                recovery_id="rrec-scripted-fail",
                reason=reason,
                started_at=now,
                finished_at=now,
                attempted=True,
                attempt_count=1,
                recovered=False,
                unrecoverable=True,
                failure_reason="injected recovery failure",
            )

    workload = _FailingRecoveryWorkload()
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=ScriptedRuntimeSampler(healthy_metrics),
        policy=make_runtime_policy(budget=_budget()),
        approval_provider=CallbackApprovalProvider(lambda _rec: ApprovalOutcome.REJECTED),
        hardware=sample_hardware(),
        run_id="b2-monitor-unrecoverable",
        session_id="b2-monitor-unrecoverable",
    )

    def _lose_after_a_moment() -> None:
        time.sleep(0.3)
        workload.lose_target()

    killer = threading.Thread(target=_lose_after_a_moment, daemon=True)
    killer.start()
    result = monitor.run()
    killer.join(timeout=2.0)

    assert result.run.target_recoveries
    assert result.run.target_recoveries[0].unrecoverable is True
    # The unrecoverable lifecycle state is recorded explicitly, and the run fails
    # closed (safety termination) rather than continuing against an unverified target.
    assert "target_unrecoverable" in [
        event.state.value for event in result.run.states
    ]
    assert result.termination.safety is True
    # Recovery is attempted exactly once for the failure event.
    assert workload.recovery_requests == 1
