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

import inspect
import os
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol

from inferenceos.cards.safety import check_context_unchanged
from inferenceos.backends.process_control import process_exists, process_identity
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.backends.observed import ObservedConfigurationBaseline, observed_configuration
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
from inferenceos.contracts.workload import ContextRequirement, WorkloadProfile
from inferenceos.interfaces.backend_adapter import (
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.logging_config import get_logger
from inferenceos.profiling.workloads import BenchmarkWorkload, get_workload
from inferenceos.runtime.workload import (
    ActiveWorkload,
    ContextRequirementViolation,
    context_requirement_violation,
    runtime_params_from_config,
)

logger = get_logger("runtime.resident_workload")

# B2: the recovery policy is deliberately bounded to a single controlled attempt per
# failure event. A replacement that successfully serves a real request earns the right
# to recover again from a later, independent failure; a replacement that dies before
# serving anything makes the target unrecoverable instead of looping.
MAX_RECOVERY_ATTEMPTS = 1


class ResidentTargetUnrecoverable(RuntimeError):
    """The resident target was lost and the single permitted recovery did not succeed."""


class ResidentTargetCancelled(RuntimeError):
    """The user stopped the resident target; nothing starts until allow_restart()."""


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
        context_requirement: ContextRequirement | None = None,
        request_gap_s: float = 0.05,
    ) -> None:
        self._settings = settings
        self._model = model
        self._workload = workload or get_workload(workload_id)
        if (
            context_requirement is not None
            and self._workload.context_requirement is None
        ):
            # An explicitly-declared hard requirement (e.g. a controlled verification
            # scenario or a caller that owns the workload contract).
            self._workload = replace(
                self._workload, context_requirement=context_requirement
            )
        self._adapter: ResidentInferenceAdapter = adapter or LlamaCppResidentAdapter(
            settings=settings
        )
        self._configuration = configuration or RuntimeConfiguration(
            context_length=2048, gpu_offload_layers=0
        )
        self._request_gap_s = max(request_gap_s, 0.0)

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        #: Set by :meth:`cancel` (a user stop): no start until :meth:`allow_restart`.
        self._cancelled = False
        #: Bumped by every cancel(); start() compares it before and after its load.
        self._cancel_count = 0
        #: Serialises stop / cancel / allow_restart (cancel sets its flag and halts the
        #: engine before taking it).
        self._stop_lock = threading.RLock()
        #: Longest wait for the serving thread when stopping (seconds).
        self.join_timeout_s = 320.0
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
        self._configuration_baseline: ObservedConfigurationBaseline | None = None
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
        """Load the target (if needed) and start the serving thread.

        Publication protocol (reviewer follow-up): the checks and the publication of
        ``_running`` / ``_accepting`` / ``_thread`` happen under ``_stop_lock``; the long
        model load does not (a cancel must still be able to interrupt it). A cancel
        that lands at any point after ``start()`` began -- during the load, or between
        the load and the publication -- wins: nothing is published, the engine stays
        halted, and ``ResidentTargetCancelled`` is raised. ``_cancel_count`` (bumped by
        every cancel) detects a cancel even if an ``allow_restart()`` followed it."""
        with self._stop_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._cancelled:
                # A user stop is sticky: only an explicit allow_restart() lifts it.
                raise ResidentTargetCancelled(
                    "the resident target was stopped by the user; call allow_restart() first"
                )
            cancels_at_start = self._cancel_count
            self._resume_engine()
            self._reset_recovery_state()
        try:
            self._ensure_loaded()  # long; not under _stop_lock
            self._capture_target()
        except BaseException as exc:
            if self._cancel_count != cancels_at_start:
                raise ResidentTargetCancelled(
                    "the resident target was stopped by the user while it was starting"
                ) from exc
            raise
        with self._stop_lock:
            if self._cancel_count != cancels_at_start or self._cancelled:
                # The cancel won between the load and here: publish nothing; make sure
                # the engine is down (idempotent with what cancel() already did).
                self._halt_and_unload()
                raise ResidentTargetCancelled(
                    "the resident target was stopped by the user before serving began"
                )
            self._running = True
            self._accepting = True
            thread = threading.Thread(
                target=self._serve, name="inferenceos-resident-workload", daemon=True
            )
            self._thread = thread
            thread.start()

    def stop(self) -> None:
        """Stop in any state (review 10): a load still in progress is cancelled and a
        loaded target unloaded. Safe from another thread and more than once. Afterwards
        a later load may start the target again -- once the old process is confirmed
        gone -- unless :meth:`cancel` was used (then only after :meth:`allow_restart`)."""
        with self._stop_lock:
            self._halt_and_unload()
            if self._cancelled:
                return
            try:
                self._resume_engine()
            except Exception as exc:  # noqa: BLE001 - stays stopped until it is gone
                logger.warning("resident target not restartable yet: %s", exc)

    def cancel(self) -> None:
        """A user stop: like :meth:`stop`, but sticky -- nothing starts again (no reload
        half, no recovery, no start()) until :meth:`allow_restart`.

        The flag is set and the engine halted *before* waiting for any other stop in
        progress, so a user stop is never delayed by, nor undone by, a concurrent
        non-sticky stop: that stop's resume re-checks the flag under the engine's lock
        (``_resume_engine``)."""
        self._cancelled = True
        self._cancel_count += 1
        halt = getattr(self._adapter, "halt", None)
        if callable(halt):
            try:
                if "by_user" in inspect.signature(halt).parameters:
                    halt(by_user=True)  # a reload it cancels reads "stopped by the user"
                else:
                    halt()
            except Exception as exc:  # noqa: BLE001 - cleanup must not crash
                logger.warning("resident workload stop failed: %s", exc)
        with self._stop_lock:
            self._halt_and_unload()

    def _halt_and_unload(self) -> None:
        self._running = False
        self._accepting = False
        # First end the engine (this also ends an answer in progress), then wait for
        # the serving thread -- not the other way round.
        halt = getattr(self._adapter, "halt", None)
        if callable(halt):
            try:
                halt()
            except Exception as exc:  # noqa: BLE001 - cleanup must not crash
                logger.warning("resident workload stop failed: %s", exc)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self.join_timeout_s)
        if thread is None or not thread.is_alive():
            self._thread = None
        try:
            if self._adapter.is_loaded or bool(getattr(self._adapter, "is_active", False)):
                self._adapter.unload()
        except Exception as exc:  # noqa: BLE001 - cleanup must not crash
            logger.warning("resident workload unload failed: %s", exc)
        self._target_pid = None
        self._target_identity = None

    def _resume_engine(self) -> None:
        """Lift the engine's halt unless a user stop is (or becomes) in force. Raises if
        the serving thread is still alive or the old process is not confirmed gone."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            raise RuntimeError("the previous resident worker has not returned; not restarting")
        resume = getattr(self._adapter, "resume", None)
        if not callable(resume):
            return
        if "unless" in inspect.signature(resume).parameters:
            resume(unless=lambda: self._cancelled)
        elif not self._cancelled:  # an adapter without the guarded resume
            resume()

    def allow_restart(self) -> None:
        """Explicitly lift a user stop (or a stop) -- only once the serving thread has
        returned and the previous engine process is confirmed gone."""
        with self._stop_lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                raise RuntimeError("the previous resident worker has not returned; not restarting")
            self._cancelled = False
            self._resume_engine()

    def _reset_recovery_state(self) -> None:
        self._unrecoverable = False
        self._recovery_permitted = True
        self._recoveries = []

    def _capture_target(self) -> None:
        self._target_pid = self._adapter.pid
        self._target_identity = self._adapter.identity
        try:
            self._configuration_baseline = ObservedConfigurationBaseline.capture(
                self._configuration, getattr(self._adapter, "telemetry")())
        except Exception:
            self._configuration_baseline = None

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
            # Preparation can yield after _serve read accepting. Recheck admission
            # atomically with marking in-flight so pause either drains this request
            # or excludes it. Synchronous helper calls keep their existing contract.
            if (threading.current_thread() is self._thread
                    and (not self._running or not self._accepting)):
                return
            self._in_flight = True
            self._in_flight_id = request_id
            self._in_flight_started_at = submitted
        try:
            self._prepare_target()
            # Request-level safety: never send a request the configured context cannot
            # hold under the workload's *known* hard requirement. Unknown requirements
            # never block serving.
            violation = context_requirement_violation(
                self._configuration, self._workload.context_requirement
            )
            if violation is not None:
                raise ContextRequirementViolation(violation)
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
            if self._cancelled:
                # A user stop is sticky: recovery never restarts the engine after it.
                return self._complete(
                    record,
                    ResidentRecoveryState.TARGET_UNRECOVERABLE,
                    "stopped by the user; no recovery until allow_restart()",
                    attempted=False,
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
        load_error: str | None = None if cleanup_complete else "previous target resources were not released"
        try:
            params = runtime_params_from_config(
                self._configuration,
                n_predict=self._workload.n_predict,
                seed=self._workload.seed,
                ignore_eos=self._workload.ignore_eos,
            )
            if cleanup_complete:
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
        identity_verified = bool(new_identity is not None and loaded and verdict is True)
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
        from inferenceos.interfaces.backend_adapter import BackendTelemetry

        reader = getattr(self._adapter, "telemetry", None)
        try:
            telemetry = reader() if callable(reader) else None
            if not isinstance(telemetry, BackendTelemetry):
                return False
            shown, contradictions, unknown = observed_configuration(self._configuration, telemetry)
        except Exception:
            return False
        # Recovery promises verified configuration, unlike a user-requested start
        # which may explicitly report unknown settings. A request is not evidence.
        baseline = self._configuration_baseline
        return (not contradictions and not unknown and baseline is not None
                and not baseline.problems_after(shown))

    def _cleanup_failed_target(self) -> bool:
        cleanup = getattr(self._adapter, "cleanup", None)
        try:
            if callable(cleanup):
                cleanup()
            else:
                self._adapter.unload()
        except Exception as exc:  # noqa: BLE001 - cleanup must not crash recovery
            logger.warning("resident target cleanup failed: %s", exc)
            return False
        evidence = getattr(self._adapter, "release_evidence", None)
        if callable(evidence):
            try:
                return bool(evidence().clean)
            except Exception:
                return False
        return not self._adapter.is_loaded

    def _no_orphan(self, old_pid: int | None) -> bool:
        if old_pid is None:
            return True
        try:
            if (old_pid == self._target_pid and self._adapter.is_loaded
                    and self._adapter.pid == old_pid and self.target_health() == TargetHealth.HEALTHY):
                # A healthy target still owned here is not an orphan. A stale
                # captured PID after a failed recovery is not that proof.
                return True
            return not process_exists(old_pid)
        except Exception:  # noqa: BLE001
            return False

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
        # Automatic context-safety check at the live model: a running model's context
        # size is never changed by a configuration update (tests and undo alike).
        if self._adapter.is_loaded:
            with self._lock:
                current = self._configuration
            check_context_unchanged(current, configuration, card_id="resident-workload")
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
            # A deliberate configuration change establishes a new recovery baseline;
            # recovery itself must verify the old one before publishing a new target.
            self._capture_target()

    def restart(self) -> None:
        """A fresh load of the running model with the same setup (see the adapter)."""
        restart = getattr(self._adapter, "restart", None)
        if not self._adapter.is_loaded or not callable(restart):
            return
        restart()
        self._target_pid = self._adapter.pid
        self._target_identity = self._adapter.identity

    def model_profile(self) -> ModelProfile:
        return self._model

    def workload_profile(self) -> WorkloadProfile:
        return self._workload.to_workload_profile()

    def last_metrics(self) -> dict[str, float]:
        with self._lock:
            return dict(self._last_metrics)

    def effective_threads(self) -> int | None:
        """Observed effective generation thread count reported by the resident backend.

        Returns ``None`` when the backend has no observed value (never inferred from CPU
        topology). The value is refreshed on every controlled reload because the backend
        rebuilds its telemetry when it starts.
        """
        try:
            telemetry = getattr(self._adapter, "telemetry", None)
            if not callable(telemetry):
                return None
            value = telemetry().effective_threads
            return int(value) if isinstance(value, int) and value > 0 else None
        except Exception:  # noqa: BLE001 - unavailable observation is unknown, not fatal
            return None

    def effective_mmap(self) -> bool | None:
        """Observed *effective* weight-load mode reported by the resident backend.

        Returns ``None`` when the backend has no observed value (never inferred from the
        configured ``use_mmap``). Refreshed on every controlled reload.
        """
        try:
            telemetry = getattr(self._adapter, "telemetry", None)
            if not callable(telemetry):
                return None
            value = telemetry().mmap_effective
            return value if isinstance(value, bool) else None
        except Exception:  # noqa: BLE001 - unavailable observation is unknown, not fatal
            return None

    def last_text(self) -> str | None:
        with self._lock:
            return self._last_text

    def monitor_pid(self) -> int:
        return os.getpid()
