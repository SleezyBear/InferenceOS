"""Phase 10R resident active workload.

Unlike the older process-per-generation workload, this keeps a **real resident
``llama-server``** running (Phase 5R infrastructure) and drives it with a real client
thread issuing sequential inference requests. The monitor therefore observes a target
process that is genuinely distinct from the monitor process, and each request exposes
real submission / first-token / completion timestamps.

The workload never kills an in-flight request: ``pause`` stops accepting new work and
drains the current request to completion before returning.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from inferenceos.backends.process_control import process_exists, process_identity
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.config import Settings
from inferenceos.contracts.enums import WorkloadState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.lifecycle import (
    ProcessIdentity,
    ResidentGenerationMeasurement,
    ResidentRecoveryEvent,
    ResidentRecoveryState,
    ResidentTargetRecovery,
    TargetHealth,
    TargetLossReason,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.runtime import (
    RuntimeRequestRecord,
    WorkloadDescriptor,
    WorkloadSnapshot,
)
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.interfaces.backend_adapter import (
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.logging_config import get_logger
from inferenceos.profiling.workloads import BenchmarkWorkload, get_workload
from inferenceos.runtime.workload import ActiveWorkload, runtime_params_from_config

logger = get_logger("runtime.resident_workload")

# B2: the recovery policy is deliberately bounded to a single controlled attempt per
# failure event. A replacement that successfully serves a real request earns the right
# to recover again from a later, independent failure; a replacement that dies before
# serving anything makes the target unrecoverable instead of looping.
MAX_RECOVERY_ATTEMPTS = 1


class ResidentTargetUnrecoverable(RuntimeError):
    """The resident target was lost and the single permitted recovery did not succeed."""


@dataclass
class _RecoveryVerification:
    identity_verified: bool
    readiness_verified: bool
    configuration_verified: bool
    model_verified: bool
    backend_verified: bool
    new_identity: ProcessIdentity | None
    new_pid: int | None

    @property
    def all_verified(self) -> bool:
        return all(
            (
                self.identity_verified,
                self.readiness_verified,
                self.configuration_verified,
                self.model_verified,
                self.backend_verified,
            )
        )


class ResidentInferenceAdapter(Protocol):
    """The subset of the resident adapter the workload depends on (injectable)."""

    @property
    def is_loaded(self) -> bool: ...

    @property
    def pid(self) -> int | None: ...

    @property
    def identity(self) -> ProcessIdentity | None: ...

    @property
    def last_generation(self) -> ResidentGenerationMeasurement | None: ...

    def load(self, model: ModelProfile, params: RuntimeParams) -> None: ...

    def unload(self) -> None: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def set_runtime_params(self, params: RuntimeParams) -> None: ...



class ResidentActiveWorkload(ActiveWorkload):
    """A genuinely resident, actively-serving inference workload."""

    def __init__(
        self,
        *,
        settings: Settings,
        model: ModelProfile,
        adapter: ResidentInferenceAdapter | None = None,
        workload: BenchmarkWorkload | None = None,
        workload_id: str = "short-short",
        configuration: RuntimeConfiguration | None = None,
        request_gap_s: float = 0.05,
    ) -> None:
        self._settings = settings
        self._model = model
        self._workload = workload or get_workload(workload_id)
        self._adapter: ResidentInferenceAdapter = adapter or LlamaCppResidentAdapter(
            settings=settings
        )
        self._configuration = configuration or RuntimeConfiguration(
            context_length=2048, gpu_offload_layers=0
        )
        self._request_gap_s = max(request_gap_s, 0.0)

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._accepting = False
        self._terminated = False
        self._last_error: str | None = None
        self._last_metrics: dict[str, float] = {}
        self._last_text: str | None = None
        self._requests: list[RuntimeRequestRecord] = []
        self._completed = 0
        self._in_flight = False
        self._in_flight_id: str | None = None
        self._in_flight_started_at: datetime | None = None
        self._target_pid: int | None = None
        self._target_identity: ProcessIdentity | None = None
        # B2 bounded-recovery state. ``_recovery_lock`` is the single authoritative
        # transition: monitor and serve thread can both request recovery but only one
        # attempt is ever made for a failure event.
        self._recovery_lock = threading.Lock()
        self._recoveries: list[ResidentTargetRecovery] = []
        self._unrecoverable = False
        self._recovery_permitted = True
        self._recovery_enabled = bool(
            getattr(settings, "resident_recovery_enabled", True)
        )
        self.pause_calls = 0
        self.resume_calls = 0

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #
    @property
    def adapter(self) -> ResidentInferenceAdapter:
        return self._adapter

    def target_identity(self) -> ProcessIdentity | None:
        """The captured PID-reuse-safe identity of the resident target.

        This is the *captured* identity, not a fresh read of the adapter: after the
        process exits the captured identity must remain available so the loss (or a
        PID reused by an unrelated process) is detectable rather than silently
        disappearing.
        """
        return self._target_identity

    def target_pid(self) -> int | None:
        return self._target_pid

    def target_alive(self) -> bool:
        identity = self._target_identity
        if identity is None:
            return False
        current = process_identity(identity.pid)
        return current is not None and abs(current.create_time - identity.create_time) < 0.01

    @property
    def unrecoverable(self) -> bool:
        """True once the single permitted recovery has failed for this operation."""
        return self._unrecoverable

    def recovery_records(self) -> list[ResidentTargetRecovery]:
        """All bounded recovery lifecycles observed for this workload."""
        return list(self._recoveries)

    def target_health(self) -> str:
        """Classify the resident target as ``HEALTHY``, ``LOST`` or ``UNRECOVERABLE``.

        Identity is checked with the adapter's live verifier where the adapter can
        provide one; a PID alone is never trusted.
        """
        if self._unrecoverable:
            return TargetHealth.UNRECOVERABLE
        adapter = self._adapter
        try:
            loaded = adapter.is_loaded
        except Exception:  # noqa: BLE001 - health probing must never raise
            loaded = False
        if not loaded:
            return TargetHealth.LOST
        identity = self._target_identity
        if identity is None:
            return TargetHealth.HEALTHY
        verdict = self._adapter_identity_live(identity)
        if verdict is False:
            return TargetHealth.LOST
        return TargetHealth.HEALTHY

    def _adapter_identity_live(self, identity: ProcessIdentity) -> bool | None:
        """Ask the adapter to verify a captured identity; None when unsupported."""
        verifier = getattr(self._adapter, "verify_process_identity", None)
        if not callable(verifier):
            return None
        try:
            result = verifier(identity)
        except Exception:  # noqa: BLE001 - verification failure is a loss, not a crash
            return False
        return result if isinstance(result, bool) else None

    def request_records(self) -> list[RuntimeRequestRecord]:
        with self._lock:
            return list(self._requests)

    def last_request(self) -> RuntimeRequestRecord | None:
        with self._lock:
            return self._requests[-1] if self._requests else None

    def in_flight_request_id(self) -> str | None:
        with self._lock:
            return self._in_flight_id

    def completed_requests(self) -> int:
        with self._lock:
            return self._completed

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._reset_recovery_state()
        self._ensure_loaded()
        self._capture_target()
        self._running = True
        self._accepting = True
        self._thread = threading.Thread(
            target=self._serve, name="inferenceos-resident-workload", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._accepting = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=320.0)
        self._thread = None
        try:
            if self._adapter.is_loaded:
                self._adapter.unload()
        except Exception as exc:  # noqa: BLE001 - cleanup must not crash
            logger.warning("resident workload unload failed: %s", exc)
        self._target_pid = None
        self._target_identity = None

    def _reset_recovery_state(self) -> None:
        self._unrecoverable = False
        self._recovery_permitted = True
        self._recoveries = []

    def _capture_target(self) -> None:
        self._target_pid = self._adapter.pid
        self._target_identity = self._adapter.identity

    def _ensure_loaded(self) -> None:
        """Provision the target once (initial start). Never an implicit retry."""
        if self._adapter.is_loaded:
            self._capture_target()
            return
        if self._unrecoverable:
            raise ResidentTargetUnrecoverable(
                "resident target is unrecoverable after a failed bounded recovery"
            )
        params = runtime_params_from_config(
            self._configuration,
            n_predict=self._workload.n_predict,
            seed=self._workload.seed,
            ignore_eos=self._workload.ignore_eos,
        )
        self._adapter.load(self._model, params)
        self._capture_target()

    def _serve(self) -> None:
        while self._running and not self._unrecoverable:
            with self._lock:
                accepting = self._accepting
            if not accepting:
                time.sleep(0.02)
                continue
            self._serve_one()
            time.sleep(self._request_gap_s)

    def _prepare_target(self) -> None:
        """Ensure a healthy target before a request, recovering at most once."""
        health = self.target_health()
        if health == TargetHealth.HEALTHY:
            return
        if health == TargetHealth.UNRECOVERABLE:
            raise ResidentTargetUnrecoverable(
                "resident target is unrecoverable; refusing further requests"
            )
        outcome = self.recover_target(reason=self._loss_reason())
        if not outcome.recovered:
            raise ResidentTargetUnrecoverable(
                outcome.failure_reason or "resident target recovery did not succeed"
            )

    def _loss_reason(self) -> str:
        try:
            loaded = self._adapter.is_loaded
        except Exception:  # noqa: BLE001
            loaded = False
        return TargetLossReason.PROCESS_EXITED if not loaded else TargetLossReason.IDENTITY_CHANGED

    def _serve_one(self) -> None:
        request_id = f"rreq-{uuid.uuid4().hex[:10]}"
        submitted = datetime.now(timezone.utc)
        with self._lock:
            self._in_flight = True
            self._in_flight_id = request_id
            self._in_flight_started_at = submitted
        try:
            self._prepare_target()
            result = self._adapter.generate(
                GenerationRequest(
                    prompt=self._workload.prompt,
                    n_predict=self._workload.n_predict,
                    seed=self._workload.seed,
                    temperature=self._workload.temperature,
                )
            )
            generation = self._adapter.last_generation
            completed = datetime.now(timezone.utc)
            record = RuntimeRequestRecord(
                request_id=request_id,
                submitted_at=submitted,
                first_token_at=generation.first_token_at if generation is not None else None,
                completed_at=completed,
                latency_s=result.total_s,
                generated_tokens=result.generated_tokens,
                decode_tokens_per_sec=result.decode_tokens_per_sec,
                success=bool(result.text),
            )
            metrics: dict[str, float] = {}
            for name, value in (
                ("decode_tokens_per_sec", result.decode_tokens_per_sec),
                ("total_latency_s", result.total_s),
                ("ttft_s", result.ttft_s),
                ("peak_rss_bytes", result.peak_rss_bytes),
                ("final_rss_bytes", result.final_rss_bytes),
            ):
                if value is not None:
                    metrics[name] = float(value)
            with self._lock:
                self._requests.append(record)
                self._completed += 1
                self._last_metrics = metrics
                self._last_text = result.text
                self._last_error = None
                # A replacement that actually served a request is a new healthy
                # reference and earns the right to recover from a later failure.
                self._recovery_permitted = True
        except Exception as exc:  # noqa: BLE001 - a failed request must not crash
            with self._lock:
                self._requests.append(
                    RuntimeRequestRecord(
                        request_id=request_id,
                        submitted_at=submitted,
                        completed_at=datetime.now(timezone.utc),
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                self._last_error = f"{type(exc).__name__}: {exc}"
            # A generation that failed because the target died flows into the same
            # bounded recovery mechanism; it never creates a separate reload path.
            self._maybe_recover_after_failure()
        finally:
            with self._lock:
                self._in_flight = False
                self._in_flight_id = None
                self._in_flight_started_at = None

    # ------------------------------------------------------------------ #
    # Bounded recovery (B2)                                                #
    # ------------------------------------------------------------------ #
    def _maybe_recover_after_failure(self) -> None:
        try:
            if self.target_health() == TargetHealth.LOST:
                self.recover_target(reason=TargetLossReason.GENERATION_FAILED)
        except Exception as exc:  # noqa: BLE001 - failure handling must never crash
            logger.warning("resident target recovery after failure errored: %s", exc)

    def recover_target(self, *, reason: str) -> ResidentTargetRecovery:
        """Perform at most one controlled recovery under the authoritative lock.

        The monitor and the serve thread may both request recovery for the same
        failure; the lock and the ``_recovery_permitted`` bound guarantee exactly one
        controlled attempt. A lost target that cannot be verified is marked
        unrecoverable rather than retried.
        """
        with self._recovery_lock:
            record = ResidentTargetRecovery(
                recovery_id=f"rrec-{uuid.uuid4().hex[:10]}",
                reason=reason,
                old_pid=self._target_pid,
                old_identity=self._target_identity,
                max_attempts=MAX_RECOVERY_ATTEMPTS,
            )
            if self._unrecoverable:
                return self._complete(
                    record,
                    ResidentRecoveryState.TARGET_UNRECOVERABLE,
                    "target already unrecoverable; no further attempts permitted",
                    attempted=False,
                    unrecoverable=True,
                )
            if self.target_health() == TargetHealth.HEALTHY:
                return self._complete(
                    record,
                    ResidentRecoveryState.TARGET_RECOVERED,
                    "target is already healthy; no recovery needed",
                    attempted=False,
                    recovered=True,
                    store=False,
                )
            if not self._recovery_enabled:
                return self._complete(
                    record,
                    ResidentRecoveryState.TARGET_UNRECOVERABLE,
                    "bounded resident recovery is disabled",
                    attempted=False,
                    unrecoverable=True,
                )
            if not self._recovery_permitted:
                return self._complete(
                    record,
                    ResidentRecoveryState.TARGET_UNRECOVERABLE,
                    "the single permitted recovery for this failure sequence was already "
                    "consumed; refusing to start an unbounded restart loop",
                    attempted=False,
                    unrecoverable=True,
                )
            # Consume the single permitted attempt for this failure sequence.
            self._recovery_permitted = False
            record = record.model_copy(update={"attempted": True, "attempt_count": 1})
            record = self._perform_recovery(record, reason=reason)
            self._recoveries.append(record)
            if record.recovered:
                logger.info(
                    "resident target recovered (reason=%s): pid %s -> %s",
                    reason,
                    record.old_pid,
                    record.new_pid,
                )
            else:
                self._unrecoverable = True
                logger.error(
                    "resident target unrecoverable (reason=%s): %s",
                    reason,
                    record.failure_reason,
                )
            return record

    def _perform_recovery(
        self, record: ResidentTargetRecovery, *, reason: str
    ) -> ResidentTargetRecovery:
        events = [
            ResidentRecoveryEvent(
                sequence=0,
                state=ResidentRecoveryState.TARGET_LOST,
                detail="resident target lost; starting the single permitted controlled recovery",
                old_pid=record.old_pid,
                old_identity=record.old_identity,
                data={
                    "reason": reason,
                    "attempt_count": record.attempt_count,
                    "max_attempts": record.max_attempts,
                },
            )
        ]
        cleanup_complete = self._cleanup_failed_target()
        load_error: str | None = None
        try:
            params = runtime_params_from_config(
                self._configuration,
                n_predict=self._workload.n_predict,
                seed=self._workload.seed,
                ignore_eos=self._workload.ignore_eos,
            )
            self._adapter.load(self._model, params)
        except Exception as exc:  # noqa: BLE001 - a failed reload is unrecoverable
            load_error = f"{type(exc).__name__}: {exc}"

        if load_error is not None:
            cleanup_complete = self._cleanup_failed_target() and cleanup_complete
            return record.model_copy(
                update={
                    "finished_at": datetime.now(timezone.utc),
                    "recovered": False,
                    "unrecoverable": True,
                    "failure_reason": f"controlled reload failed: {load_error}",
                    "cleanup_complete": cleanup_complete,
                    "orphan_free": self._no_orphan(record.old_pid),
                    "events": events
                    + [
                        ResidentRecoveryEvent(
                            sequence=1,
                            state=ResidentRecoveryState.TARGET_UNRECOVERABLE,
                            detail="controlled reload failed; target cannot be recovered",
                            old_pid=record.old_pid,
                            old_identity=record.old_identity,
                            data={"load_error": load_error},
                        )
                    ],
                }
            )

        verification = self._verify_recovered_target(record.old_identity)
        new_identity = verification.new_identity
        if verification.all_verified:
            self._capture_target()
            return record.model_copy(
                update={
                    "finished_at": datetime.now(timezone.utc),
                    "new_pid": verification.new_pid,
                    "new_identity": new_identity,
                    "identity_verified": True,
                    "readiness_verified": True,
                    "configuration_verified": True,
                    "model_verified": True,
                    "backend_verified": True,
                    "recovered": True,
                    "unrecoverable": False,
                    "cleanup_complete": cleanup_complete,
                    "orphan_free": self._no_orphan(record.old_pid),
                    "events": events
                    + [
                        ResidentRecoveryEvent(
                            sequence=1,
                            state=ResidentRecoveryState.TARGET_RECOVERED,
                            detail=(
                                "replacement target verified: identity, readiness, "
                                "configuration, model and backend"
                            ),
                            old_pid=record.old_pid,
                            old_identity=record.old_identity,
                            new_pid=verification.new_pid,
                            new_identity=new_identity,
                            data=self._verification_data(verification),
                        )
                    ],
                }
            )

        # Verification failed: clean the replacement up and fail closed.
        cleanup_complete = self._cleanup_failed_target() and cleanup_complete
        return record.model_copy(
            update={
                "finished_at": datetime.now(timezone.utc),
                "new_pid": verification.new_pid,
                "new_identity": new_identity,
                "identity_verified": verification.identity_verified,
                "readiness_verified": verification.readiness_verified,
                "configuration_verified": verification.configuration_verified,
                "model_verified": verification.model_verified,
                "backend_verified": verification.backend_verified,
                "recovered": False,
                "unrecoverable": True,
                "failure_reason": "recovered target failed verification",
                "cleanup_complete": cleanup_complete,
                "orphan_free": self._no_orphan(record.old_pid),
                "events": events
                + [
                    ResidentRecoveryEvent(
                        sequence=1,
                        state=ResidentRecoveryState.TARGET_UNRECOVERABLE,
                        detail="recovered target failed verification; target unrecoverable",
                        old_pid=record.old_pid,
                        old_identity=record.old_identity,
                        new_pid=verification.new_pid,
                        new_identity=new_identity,
                        data=self._verification_data(verification),
                    )
                ],
            }
        )

    @staticmethod
    def _verification_data(verification: _RecoveryVerification) -> dict[str, object]:
        return {
            "identity_verified": verification.identity_verified,
            "readiness_verified": verification.readiness_verified,
            "configuration_verified": verification.configuration_verified,
            "model_verified": verification.model_verified,
            "backend_verified": verification.backend_verified,
        }

    def _verify_recovered_target(
        self, old_identity: ProcessIdentity | None
    ) -> _RecoveryVerification:
        new_identity = self._adapter.identity
        loaded = bool(self._adapter.is_loaded)
        verdict = (
            self._adapter_identity_live(new_identity) if new_identity is not None else None
        )
        identity_verified = bool(new_identity is not None and loaded and verdict is not False)
        if new_identity is not None and old_identity is not None and identity_verified:
            same = new_identity.pid == old_identity.pid and abs(
                new_identity.create_time - old_identity.create_time
            ) <= 0.01
            identity_verified = not same
        last_load = getattr(self._adapter, "last_load", None)
        readiness_verified = bool(
            loaded and (last_load is None or bool(getattr(last_load, "ready", False)))
        )
        model_verified = bool(
            loaded
            and (last_load is None or getattr(last_load, "model_ref", None) == self._model.name)
        )
        configuration_verified = self._configuration_matches()
        return _RecoveryVerification(
            identity_verified=identity_verified,
            readiness_verified=readiness_verified,
            configuration_verified=configuration_verified,
            model_verified=model_verified,
            backend_verified=loaded,
            new_identity=new_identity,
            new_pid=new_identity.pid if new_identity is not None else None,
        )

    def _configuration_matches(self) -> bool:
        params = getattr(self._adapter, "params", None)
        if params is None:
            return True
        expected = runtime_params_from_config(
            self._configuration,
            n_predict=self._workload.n_predict,
            seed=self._workload.seed,
            ignore_eos=self._workload.ignore_eos,
        )
        for name in (
            "context_length",
            "gpu_offload_layers",
            "threads",
            "batch_size",
            "ubatch_size",
            "use_mmap",
            "device",
            "kv_cache_type",
        ):
            if getattr(params, name, None) != getattr(expected, name, None):
                return False
        return True

    def _cleanup_failed_target(self) -> bool:
        cleanup = getattr(self._adapter, "cleanup", None)
        try:
            if callable(cleanup):
                cleanup()
            elif self._adapter.is_loaded:
                self._adapter.unload()
        except Exception as exc:  # noqa: BLE001 - cleanup must not crash recovery
            logger.warning("resident target cleanup failed: %s", exc)
            return False
        return True

    def _no_orphan(self, old_pid: int | None) -> bool:
        if old_pid is None or old_pid == self._target_pid:
            return True
        try:
            return not process_exists(old_pid)
        except Exception:  # noqa: BLE001
            return True

    def _complete(
        self,
        record: ResidentTargetRecovery,
        state: str,
        detail: str,
        *,
        attempted: bool,
        recovered: bool = False,
        unrecoverable: bool = False,
        store: bool = True,
    ) -> ResidentTargetRecovery:
        finished = record.model_copy(
            update={
                "finished_at": datetime.now(timezone.utc),
                "attempted": attempted,
                "recovered": recovered,
                "unrecoverable": unrecoverable,
                "failure_reason": None if recovered else detail,
                "orphan_free": self._no_orphan(record.old_pid),
                "events": [
                    ResidentRecoveryEvent(
                        sequence=0,
                        state=state,
                        detail=detail,
                        old_pid=record.old_pid,
                        old_identity=record.old_identity,
                    )
                ],
            }
        )
        if store:
            self._recoveries.append(finished)
        if unrecoverable:
            self._unrecoverable = True
        return finished

    def pause(self) -> bool:
        """Stop accepting new work and drain the in-flight request to completion."""
        self.pause_calls += 1
        with self._lock:
            self._accepting = False
        deadline = time.perf_counter() + 300.0
        while time.perf_counter() < deadline:
            with self._lock:
                if not self._in_flight:
                    return True
            time.sleep(0.05)
        return False

    def resume(self) -> bool:
        self.resume_calls += 1
        with self._lock:
            if not self._running or self._terminated:
                return False
            self._accepting = True
        return True

    # ------------------------------------------------------------------ #
    # ActiveWorkload                                                       #
    # ------------------------------------------------------------------ #
    def descriptor(self) -> WorkloadDescriptor:
        with self._lock:
            configuration = self._configuration
        return WorkloadDescriptor(
            workload_id=self._workload.workload_id,
            workload_kind=self._workload.kind,
            model_ref=self._model.name,
            model_path=self._model.path,
            backend_name="llama.cpp (resident)",
            configuration=configuration,
            context_length=configuration.context_length,
            prompts_ref=self._workload.to_workload_profile().prompt_set_ref,
            owned=True,
            controllable=True,
            pause_supported=True,
            checkpoint_supported=True,
            resume_supported=True,
            limitations=[
                "resident llama-server target; pause drains the in-flight request",
                "checkpoint covers runtime configuration + identity, not process memory",
            ],
        )

    def snapshot(self) -> WorkloadSnapshot:
        with self._lock:
            if self._terminated:
                state = WorkloadState.TERMINATED_EXTERNALLY
            elif not self._running:
                state = WorkloadState.STOPPED
            elif self._in_flight and not self._accepting:
                state = WorkloadState.DRAINING
            elif not self._accepting:
                state = WorkloadState.PAUSED
            else:
                state = WorkloadState.RUNNING
            return WorkloadSnapshot(
                state=state,
                running=self._running and not self._terminated,
                generating=self._in_flight,
                terminated_externally=self._terminated,
                safe_to_pause=not self._in_flight,
                can_resume=self._running and not self._terminated,
                requests_served=self._completed,
                in_flight_request_id=self._in_flight_id,
                last_error=self._last_error,
                unrecoverable=self._unrecoverable,
            )

    def configuration(self) -> RuntimeConfiguration:
        with self._lock:
            return self._configuration

    def set_configuration(self, configuration: RuntimeConfiguration) -> None:
        with self._lock:
            self._configuration = configuration
        # Apply to the live resident target via the controlled reload semantics so the
        # intervention genuinely changes the running server (not just an in-memory field).
        if self._adapter.is_loaded:
            self._adapter.set_runtime_params(
                runtime_params_from_config(
                    configuration,
                    n_predict=self._workload.n_predict,
                    seed=self._workload.seed,
                    ignore_eos=self._workload.ignore_eos,
                )
            )
            self._target_pid = self._adapter.pid
            self._target_identity = self._adapter.identity

    def model_profile(self) -> ModelProfile:
        return self._model

    def workload_profile(self) -> WorkloadProfile:
        return self._workload.to_workload_profile()

    def last_metrics(self) -> dict[str, float]:
        with self._lock:
            return dict(self._last_metrics)

    def last_text(self) -> str | None:
        with self._lock:
            return self._last_text

    def monitor_pid(self) -> int:
        return os.getpid()
