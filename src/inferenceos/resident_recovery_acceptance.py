"""B2 resident-server crash detection, bounded recovery, and unrecoverable handling.

Real evidence: a real resident ``llama-server`` is started, real inference is served,
the actual server process is terminated, the monitor detects the loss, performs exactly
one controlled recovery through the existing resident-server architecture, and real
inference resumes against the verified replacement.

Controlled evidence: a controlled identity fixture proves PID/identity mismatch is
treated as target loss (not accepted as the original target), and an injected reload
failure proves the recovery bound reaches an explicit unrecoverable state without
retrying.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.backends.process_control import (
    identity_matches,
    process_exists,
    process_identity,
)
from inferenceos.config import Settings
from inferenceos.contracts.enums import ApprovalOutcome
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.lifecycle import ResidentRecoveryState, TargetHealth
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.runtime import RuntimeBudget, RuntimePolicy
from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
from inferenceos.models.loader import load_model_profile
from inferenceos.runtime.approval import CallbackApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.resident_workload import (
    ResidentActiveWorkload,
    ResidentTargetUnrecoverable,
)
from inferenceos.runtime.telemetry import SystemTelemetrySampler
from inferenceos.runtime.verification import ControlledFootprintDetector
from inferenceos.selftest import sample_hardware


def _server_path(settings: object) -> Path:
    base = Path(getattr(settings, "llama_cpp_dir"))
    candidate = base / "llama-server.exe"
    return candidate if candidate.is_file() else base / "llama-server"


def _largest_model(settings: object) -> ModelProfile:
    models_dir = Path(getattr(settings, "models_dir"))
    profiles = [load_model_profile(path) for path in sorted(models_dir.glob("*.gguf"))]
    if not profiles:
        raise FileNotFoundError(f"no .gguf models found in {models_dir}")
    return max(profiles, key=lambda profile: profile.file_size_bytes)


def _kill_process_tree(pid: int) -> bool:
    """Abruptly terminate the real target process (simulated crash)."""
    import psutil

    try:
        process = psutil.Process(pid)
    except Exception:  # noqa: BLE001
        return False
    try:
        process.kill()
    except Exception:  # noqa: BLE001
        return False
    for child in process.children(recursive=True):
        try:
            child.kill()
        except Exception:  # noqa: BLE001
            pass
    return True


def _as_int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _recovery_policy(duration_s: float) -> RuntimePolicy:
    return RuntimePolicy(
        budget=RuntimeBudget(
            max_duration_s=duration_s,
            telemetry_interval_s=0.5,
            observation_window_size=80,
            baseline_samples=3,
            max_deep_analyses=0,
            max_interventions=1,
            max_cumulative_intervention_s=60.0,
            max_optimizer_overhead_s=120.0,
            approval_timeout_s=5.0,
            max_alerts_per_condition=1,
        ),
        diagnosis_cooldown_s=0.0,
    )


def _make_adapter(settings: object, server: Path) -> LlamaCppResidentAdapter:
    return LlamaCppResidentAdapter(settings=settings, server=server)  # type: ignore[arg-type]


def _make_workload(
    settings: object, model: ModelProfile, adapter: object, *, request_gap_s: float = 0.0
) -> ResidentActiveWorkload:
    return ResidentActiveWorkload(
        settings=settings,  # type: ignore[arg-type]
        model=model,
        adapter=adapter,  # type: ignore[arg-type]
        workload_id="short-short",
        configuration=RuntimeConfiguration(context_length=1024, gpu_offload_layers=0),
        request_gap_s=request_gap_s,
    )


class _FailingReloadAdapter:
    """Controlled fixture: delegate to a real adapter but fail the recovery reload."""

    def __init__(self, inner: object, *, fail_after: int = 1) -> None:
        self._inner = inner
        self._fail_after = fail_after
        self.load_calls = 0

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        self.load_calls += 1
        if self.load_calls > self._fail_after:
            raise RuntimeError("injected recovery reload failure")
        inner_load: Any = getattr(self._inner, "load")
        inner_load(model, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _run_real_recovery_scenario(
    settings: object, model: ModelProfile, server: Path
) -> dict[str, object]:
    """Start a real server, kill it mid-run, and let the monitor recover it once."""
    adapter = _make_adapter(settings, server)
    workload = _make_workload(settings, model, adapter)
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=SystemTelemetrySampler(settings=settings),  # type: ignore[arg-type]
        policy=_recovery_policy(45.0),
        approval_provider=CallbackApprovalProvider(lambda _rec: ApprovalOutcome.REJECTED),
        detector=ControlledFootprintDetector(force=False),  # type: ignore[arg-type]
        hardware=sample_hardware(),
        run_id="b2-real-recovery",
        session_id="b2-real-recovery",
        settings=settings,  # type: ignore[arg-type]
    )
    captured: dict[str, object] = {}
    kill_done = threading.Event()

    def _kill_when_serving() -> None:
        deadline = time.perf_counter() + 60.0
        while time.perf_counter() < deadline and not kill_done.is_set():
            pid = workload.target_pid()
            if pid is not None and workload.completed_requests() >= 1:
                identity = workload.target_identity()
                captured["old_pid"] = pid
                captured["old_identity"] = identity
                captured["requests_before_loss"] = workload.completed_requests()
                captured["killed"] = _kill_process_tree(pid)
                kill_done.set()
                return
            time.sleep(0.1)

    killer = threading.Thread(target=_kill_when_serving, daemon=True)
    try:
        killer.start()
        result = monitor.run()
    finally:
        killer.join(timeout=5.0)

    recoveries = result.run.target_recoveries
    attempted = [record for record in recoveries if record.attempted]
    recovery = attempted[0] if attempted else (recoveries[-1] if recoveries else None)
    old_pid = captured.get("old_pid")
    new_pid = recovery.new_pid if recovery is not None else workload.target_pid()
    requests_after = workload.completed_requests()
    recovered_identity = recovery.new_identity if recovery is not None else None
    # Verify inference and the live identity fixture while the replacement is alive.
    post_recovery_inference = False
    identity_fixture_ok = False
    try:
        if recovery is not None and recovery.recovered:
            result_gen = adapter.generate(
                GenerationRequest(prompt="Recovery check.", n_predict=8)
            )
            post_recovery_inference = bool(result_gen.text)
            if recovered_identity is not None:
                stale_identity = recovered_identity.model_copy(
                    update={"create_time": recovered_identity.create_time - 1000.0}
                )
                identity_fixture_ok = bool(
                    identity_matches(recovered_identity)
                    and not identity_matches(stale_identity)
                )
    finally:
        workload.stop()

    old_gone = not (isinstance(old_pid, int) and process_exists(old_pid))
    new_gone = not (isinstance(new_pid, int) and process_exists(new_pid))
    states = (
        [event.state for event in recovery.events] if recovery is not None else []
    )
    captured_old_identity = captured.get("old_identity")
    old_identity_dump = (
        captured_old_identity.model_dump(mode="json")
        if hasattr(captured_old_identity, "model_dump")
        else None
    )
    return {
        "old_pid": old_pid,
        "old_identity": old_identity_dump,
        "kill_succeeded": bool(captured.get("killed")),
        "requests_before_loss": captured.get("requests_before_loss"),
        "recovery": recovery.model_dump(mode="json") if recovery is not None else None,
        "recovery_records": len(recoveries),
        "attempted_recovery_count": len(attempted),
        "recovery_states": states,
        "has_target_lost_event": ResidentRecoveryState.TARGET_LOST in states,
        "has_target_recovered_event": ResidentRecoveryState.TARGET_RECOVERED in states,
        "new_pid": new_pid,
        "new_identity": (
            recovered_identity.model_dump(mode="json")
            if recovered_identity is not None
            else None
        ),
        "identity_changed": bool(
            recovery is not None
            and recovery.old_identity is not None
            and recovery.new_identity is not None
            and (
                recovery.old_identity.pid,
                recovery.old_identity.create_time,
            )
            != (recovery.new_identity.pid, recovery.new_identity.create_time)
        ),
        "identity_fixture_ok": identity_fixture_ok,
        "post_recovery_inference": post_recovery_inference,
        "requests_after": requests_after,
        "requests_after_loss": requests_after - _as_int(captured.get("requests_before_loss")),
        "monitor_termination": result.termination.reason.value,
        "final_state": result.final_state.value,
        "old_process_gone": old_gone,
        "replacement_process_gone": new_gone,
        "orphan_free": old_gone and new_gone,
    }


def _run_unrecoverable_scenario(
    settings: object, model: ModelProfile, server: Path
) -> dict[str, object]:
    """A real target is killed and the recovery reload is made to fail exactly once."""
    inner = _make_adapter(settings, server)
    adapter = _FailingReloadAdapter(inner, fail_after=1)
    workload = _make_workload(settings, model, adapter)
    workload.start()
    try:
        deadline = time.perf_counter() + 30.0
        while workload.completed_requests() < 1 and time.perf_counter() < deadline:
            time.sleep(0.1)
        old_pid = workload.target_pid()
        old_identity = workload.target_identity()
        workload.pause()
        killed = _kill_process_tree(old_pid) if old_pid is not None else False
        health_before = workload.target_health()
        first = workload.recover_target(reason="process_exited")
        load_calls_after_first = adapter.load_calls
        second = workload.recover_target(reason="process_exited")
        load_calls_after_second = adapter.load_calls
        # A request attempted against an unrecoverable target must fail deterministically.
        request_failed = False
        workload.resume()
        try:
            workload._prepare_target()  # noqa: SLF001 - exercising the request gate
        except ResidentTargetUnrecoverable:
            request_failed = True
    finally:
        workload.stop()

    old_gone = not (isinstance(old_pid, int) and process_exists(old_pid))
    new_pid = first.new_pid
    orphan_free = old_gone and not (
        isinstance(new_pid, int) and process_exists(new_pid)
    )
    return {
        "old_pid": old_pid,
        "old_identity": (
            old_identity.model_dump(mode="json") if old_identity is not None else None
        ),
        "kill_succeeded": killed,
        "health_before_recovery": health_before,
        "first_recovery": first.model_dump(mode="json"),
        "second_recovery": second.model_dump(mode="json"),
        "load_calls_after_first": load_calls_after_first,
        "load_calls_after_second": load_calls_after_second,
        "second_attempted_extra_load": load_calls_after_second > load_calls_after_first,
        "unrecoverable": workload.unrecoverable,
        "snapshot_unrecoverable": workload.snapshot().unrecoverable,
        "request_failed_deterministically": request_failed,
        "old_process_gone": old_gone,
        "orphan_free": orphan_free,
    }


def run_resident_recovery_acceptance(
    settings: Settings, *, model_profile: ModelProfile | None = None
) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    scenarios: dict[str, object] = {}
    checks: list[tuple[str, bool]] = []

    model = model_profile or _largest_model(settings)
    server = _server_path(settings)

    # ------------------------------------------------------------------ #
    # Recovery case (real server, real crash, monitor-driven recovery)     #
    # ------------------------------------------------------------------ #
    real: dict[str, object] = {}
    real_ok = False
    try:
        real = _run_real_recovery_scenario(settings, model, server)
        recovery = real.get("recovery")
        assert isinstance(recovery, dict)
        real_ok = bool(
            real["kill_succeeded"]
            and real["has_target_lost_event"]
            and real["has_target_recovered_event"]
            and real["attempted_recovery_count"] == 1
            and recovery["attempted"] is True
            and recovery["attempt_count"] == 1
            and recovery["recovered"] is True
            and recovery["identity_verified"] is True
            and recovery["readiness_verified"] is True
            and recovery["configuration_verified"] is True
            and recovery["model_verified"] is True
            and recovery["backend_verified"] is True
            and real["identity_changed"] is True
            and real["post_recovery_inference"] is True
            and _as_int(real["requests_after_loss"]) >= 1
            and real["orphan_free"] is True
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:  # noqa: BLE001
        real = {"error": f"{type(exc).__name__}: {exc}"}
    scenarios["real_recovery"] = real
    checks.append(
        (
            "B2a: a real resident server crash is detected and recovered exactly once",
            real_ok,
        )
    )
    checks.append(
        (
            "B2b: the replacement identity/readiness/configuration/model/backend are "
            "verified before the target is accepted",
            real_ok,
        )
    )

    # ------------------------------------------------------------------ #
    # Controlled identity fixture (PID reuse safety)                       #
    # ------------------------------------------------------------------ #
    fixture_ok = False
    fixture: dict[str, object] = {}
    try:
        live = process_identity(os.getpid())
        assert live is not None
        stale = live.model_copy(update={"create_time": live.create_time - 1000.0})
        fixture = {
            "live_identity": live.model_dump(mode="json"),
            "stale_identity": stale.model_dump(mode="json"),
            "live_matches": identity_matches(live),
            "stale_matches": identity_matches(stale),
        }
        fixture_ok = identity_matches(live) and not identity_matches(stale)
    except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001
        fixture = {"error": f"{type(exc).__name__}: {exc}"}
    scenarios["identity_fixture"] = fixture
    checks.append(
        (
            "B2c: a controlled identity mismatch is treated as target loss, not as the "
            "original target",
            bool(real.get("identity_fixture_ok")) and fixture_ok,
        )
    )

    # ------------------------------------------------------------------ #
    # Unrecoverable case (controlled reload failure)                       #
    # ------------------------------------------------------------------ #
    unrecoverable_ok = False
    unrecoverable: dict[str, object] = {}
    try:
        unrecoverable = _run_unrecoverable_scenario(settings, model, server)
        first = unrecoverable["first_recovery"]
        second = unrecoverable["second_recovery"]
        assert isinstance(first, dict) and isinstance(second, dict)
        unrecoverable_ok = bool(
            unrecoverable["kill_succeeded"]
            and unrecoverable["health_before_recovery"] == TargetHealth.LOST
            and first["attempted"] is True
            and first["attempt_count"] == 1
            and first["recovered"] is False
            and first["unrecoverable"] is True
            and second["attempted"] is False
            and unrecoverable["second_attempted_extra_load"] is False
            and unrecoverable["unrecoverable"] is True
            and unrecoverable["snapshot_unrecoverable"] is True
            and unrecoverable["orphan_free"] is True
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:  # noqa: BLE001
        unrecoverable = {"error": f"{type(exc).__name__}: {exc}"}
    scenarios["unrecoverable"] = unrecoverable
    checks.append(
        (
            "B2d: a failed recovery reaches an explicit unrecoverable state and never "
            "starts a second attempt",
            unrecoverable_ok,
        )
    )

    return scenarios, checks
