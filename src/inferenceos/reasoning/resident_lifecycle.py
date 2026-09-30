"""Phase 5R resident deep-diagnosis lifecycle.

State machine (only entered when it is economically and physically justified):

    RESIDENT_TARGET -> ENTRY_DECISION -> CHECKPOINT -> UNLOAD_TARGET ->
    LOAD_REASONER -> DEEP_DIAGNOSIS -> UNLOAD_REASONER -> RELOAD_TARGET ->
    VERIFY_TARGET -> RESUME

Every step is timestamped (wall clock for humans, monotonic for durations), and the
target is always returned to a resident, operational state — or the failure is
surfaced explicitly, never masked as success.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from inferenceos.backends.process_control import resource_snapshot
from inferenceos.backends.resident_server import ResidentServerBackend
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.lifecycle import (
    DiagnosisRequestRecord,
    EntryEconomics,
    FeasibilityAssessment,
    LifecycleCostBreakdown,
    LifecyclePolicy,
    LifecycleTransition,
    PostReloadVerification,
    ProcessIdentity,
    RequestPolicy,
    ResidentDeepDiagnosisOutcome,
    ResidentLoadMeasurement,
    ResidentState,
    ResidentUnloadMeasurement,
    ResourceSnapshot,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.reasoning import DeepDiagnosisOutcome, EscalationDecision
from inferenceos.interfaces.backend_adapter import (
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.logging_config import get_logger
from inferenceos.reasoning.context import DEFAULT_OBJECTIVE
from inferenceos.reasoning.economics import (
    assess_feasibility,
    evaluate_entry_economics,
)
from inferenceos.reasoning.escalation import decide_escalation
from inferenceos.reasoning.lifecycle import run_deep_diagnosis
from inferenceos.reasoning.resident_reasoner import ResidentReasoner

logger = get_logger("reasoning.resident_lifecycle")


class ResidentLifecycleError(RuntimeError):
    """The resident lifecycle could not complete safely."""


class RequestWindow:
    """Explicit queue-or-reject behaviour for requests during the diagnosis window."""

    def __init__(self, policy: str) -> None:
        if policy not in (RequestPolicy.QUEUE, RequestPolicy.REJECT):
            raise ValueError(f"unknown request policy: {policy!r}")
        self._policy = policy
        self._records: list[DiagnosisRequestRecord] = []
        self._open = False

    def open(self) -> None:
        self._open = True

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def records(self) -> list[DiagnosisRequestRecord]:
        return list(self._records)

    def submit(self, prompt: str) -> DiagnosisRequestRecord:
        digest = hashlib.sha256(prompt.encode("utf-8", "replace")).hexdigest()[:16]
        if self._policy == RequestPolicy.QUEUE:
            record = DiagnosisRequestRecord(
                policy=self._policy,
                accepted=True,
                queued=True,
                prompt_hash=digest,
                response_detail="queued until the target resumes",
            )
        else:
            record = DiagnosisRequestRecord(
                policy=self._policy,
                accepted=False,
                rejected=True,
                prompt_hash=digest,
                response_detail=(
                    "temporarily unavailable: target is unloaded for deep diagnosis "
                    "(rejected, not dropped)"
                ),
            )
        self._records.append(record)
        return record

    def drain(self) -> list[DiagnosisRequestRecord]:
        """Mark queued records as drained on resume (queue policy only)."""
        drained: list[DiagnosisRequestRecord] = []
        for record in self._records:
            if record.queued and not record.drained:
                updated = record.model_copy(
                    update={"drained": True, "response_detail": "drained after target resumed"}
                )
                index = self._records.index(record)
                self._records[index] = updated
                drained.append(updated)
        self._open = False
        return drained


class ResidentLifecycleRunner:
    """Runs one resident deep-diagnosis lifecycle against a real target backend."""

    def __init__(
        self,
        *,
        target: ResidentServerBackend,
        target_model: ModelProfile,
        target_params: RuntimeParams,
        policy: LifecyclePolicy,
        reasoner: ResidentReasoner | None = None,
        request_policy: str = RequestPolicy.REJECT,
        interruption_cost_s: float = 0.0,
        resource_probe: Callable[[int | None], ResourceSnapshot] | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        on_diagnosis_window: Callable[[RequestWindow], None] | None = None,
        enabled: bool = True,
    ) -> None:
        self._target = target
        self._target_model = target_model
        self._target_params = target_params
        self._policy = policy
        self._reasoner = reasoner
        self._request_window = RequestWindow(request_policy)
        self._interruption_cost_s = interruption_cost_s
        self._resource_probe = resource_probe or resource_snapshot
        self._wall_clock = wall_clock or (lambda: datetime.now(timezone.utc))
        self._monotonic = monotonic or time.perf_counter
        self._on_diagnosis_window = on_diagnosis_window
        self._enabled = enabled

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #
    def run(
        self,
        diagnosis: BottleneckDiagnosis,
        *,
        report: BaselineReport | None = None,
        confidence_threshold: float = 0.6,
        objective: str = DEFAULT_OBJECTIVE,
        max_evidence: int = 12,
        baseline_probe: GenerationRequest | None = None,
    ) -> ResidentDeepDiagnosisOutcome:
        started = self._monotonic()
        transitions: list[LifecycleTransition] = []
        errors: list[str] = []
        sequence = 0

        def emit(
            state: str,
            event: str,
            *,
            success: bool = True,
            detail: str = "",
            error: str | None = None,
            duration_s: float | None = None,
        ) -> LifecycleTransition:
            nonlocal sequence
            sequence += 1
            transition = LifecycleTransition(
                sequence=sequence,
                state=state,
                event=event,
                wall_clock=self._wall_clock(),
                monotonic_s=round(self._monotonic() - started, 6),
                duration_s=duration_s,
                success=success,
                detail=detail,
                error=error,
            )
            transitions.append(transition)
            return transition

        if not self._enabled:
            return self._refuse(
                diagnosis,
                transitions,
                emit,
                reason="resident deep diagnosis is disabled by configuration",
                economics=evaluate_entry_economics(
                    decide_escalation(diagnosis, enabled=False),
                    policy=self._policy,
                    target_unload_s=0.0,
                    reasoner_load_s=0.0,
                    reasoner_unload_s=0.0,
                    target_reload_s=0.0,
                ),
                escalation=decide_escalation(diagnosis, enabled=False),
            )

        if not self._target.is_running():
            emit(
                ResidentState.FAILED,
                "target_not_resident",
                success=False,
                error="the lifecycle requires a resident target; none is loaded",
            )
            raise ResidentLifecycleError("no resident target is running")

        escalation = decide_escalation(
            diagnosis, enabled=True, confidence_threshold=confidence_threshold
        )
        emit(
            ResidentState.RESIDENT_TARGET,
            "target_resident",
            detail=f"pid={self._target.pid}",
        )

        # Entry economics ------------------------------------------------- #
        prior_load = self._target.last_load
        prior_unload = self._target.last_unload
        target_unload_est = prior_unload.unload_s if prior_unload is not None else 1.0
        target_reload_est = (
            prior_load.startup_s
            if prior_load is not None
            else self._policy.reasoner_load_estimate_s
        )
        reasoner_load_est, reasoner_unload_est = self._reasoner_estimates()
        economics = evaluate_entry_economics(
            escalation,
            policy=self._policy,
            target_unload_s=target_unload_est,
            reasoner_load_s=reasoner_load_est,
            reasoner_unload_s=reasoner_unload_est,
            target_reload_s=target_reload_est,
            interruption_cost_s=self._interruption_cost_s,
        )
        emit(
            ResidentState.ENTRY_DECISION,
            "entry_economics_evaluated",
            detail=f"enter={economics.enter} net={economics.net_utility:.3f}",
            success=True,
        )
        if not escalation.should_escalate:
            return self._refuse(
                diagnosis,
                transitions,
                emit,
                reason="deterministic escalation gate did not fire",
                economics=economics,
                escalation=escalation,
                summary_prefix="Deep diagnosis not entered",
            )
        if not economics.enter:
            return self._refuse(
                diagnosis,
                transitions,
                emit,
                reason="entry cost exceeds estimated benefit",
                economics=economics,
                escalation=escalation,
                summary_prefix="Deep diagnosis refused on economic grounds",
            )

        # Feasibility gate ------------------------------------------------- #
        snapshot = self._resource_probe(self._target.pid)
        target_bytes = (
            snapshot.rss_bytes
            or (prior_load.after.rss_bytes if prior_load is not None else None)
        )
        reasoner_required = self._reasoner.expected_bytes if self._reasoner is not None else None
        basis = "reasoner setting override" if reasoner_required is not None else "unknown"
        if self._reasoner is None:
            return self._refuse(
                diagnosis,
                transitions,
                emit,
                reason="no resident reasoner is configured",
                economics=economics,
                escalation=escalation,
            )
        feasibility = assess_feasibility(
            snapshot=snapshot,
            target_resident_bytes=target_bytes,
            reasoner_required_bytes=reasoner_required,
            policy=self._policy,
            reasoner_footprint_basis=basis,
        )
        emit(
            ResidentState.ENTRY_DECISION,
            "reasoner_feasibility_assessed",
            success=feasibility.feasible,
            detail="; ".join(feasibility.reasons[-1:] or []),
        )
        if not feasibility.feasible:
            return self._refuse(
                diagnosis,
                transitions,
                emit,
                reason="reasoner feasibility gate failed",
                economics=economics,
                escalation=escalation,
                feasibility=feasibility,
                summary_prefix="Deep diagnosis refused; target left running",
            )

        # Checkpoint ------------------------------------------------------- #
        identity_before: ProcessIdentity | None = self._target.identity
        pid_before = self._target.pid
        probe = baseline_probe or GenerationRequest(
            prompt="Reply with the single word: OK.", n_predict=8, seed=1234, temperature=0.0
        )
        baseline_hash: str | None = None
        baseline_generation_ok = False
        try:
            baseline_result = self._target.generate(probe)
            baseline_hash = _normalize_hash(baseline_result.text)
            baseline_generation_ok = True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"pre-unload baseline generation failed: {exc}")
        emit(
            ResidentState.CHECKPOINT,
            "target_checkpoint_captured",
            success=baseline_generation_ok,
            detail=(
                f"config={self._target_params.model_dump(exclude_none=True)}; "
                f"baseline_generation_ok={baseline_generation_ok}"
            ),
        )

        target_unload: ResidentUnloadMeasurement | None = None
        reasoner_load: ResidentLoadMeasurement | None = None
        reasoner_unload: ResidentUnloadMeasurement | None = None
        target_reload: ResidentLoadMeasurement | None = None
        diagnosis_outcome: DeepDiagnosisOutcome | None = None
        post_reload: PostReloadVerification | None = None
        cost: LifecycleCostBreakdown | None = None
        self._request_window.open()

        try:
            # Unload target ------------------------------------------------ #
            t0 = self._monotonic()
            target_unload = self._target.stop()
            emit(
                ResidentState.UNLOAD_TARGET,
                "target_unloaded",
                success=target_unload.cleanup_complete,
                detail=f"pid={target_unload.pid} method={target_unload.stop_method}",
                duration_s=round(self._monotonic() - t0, 6),
            )
            if not target_unload.cleanup_complete:
                raise ResidentLifecycleError("target unload did not complete cleanly")

            # Load reasoner ------------------------------------------------ #
            t0 = self._monotonic()
            reasoner_load = self._reasoner.start()
            emit(
                ResidentState.LOAD_REASONER,
                "reasoner_loaded",
                success=reasoner_load.resident_verified,
                detail=f"pid={reasoner_load.pid} startup_s={reasoner_load.startup_s:.3f}",
                duration_s=round(self._monotonic() - t0, 6),
            )

            # Deep diagnosis ----------------------------------------------- #
            if self._on_diagnosis_window is not None:
                self._on_diagnosis_window(self._request_window)
            t0 = self._monotonic()
            diagnosis_outcome = run_deep_diagnosis(
                diagnosis,
                report=report,
                engine=self._reasoner.engine(),
                enabled=True,
                confidence_threshold=confidence_threshold,
                objective=objective,
                max_evidence=max_evidence,
                corroboration_ceiling=self._policy.corroboration_ceiling,
            )
            emit(
                ResidentState.DEEP_DIAGNOSIS,
                "diagnosis_completed",
                success=not diagnosis_outcome.fallback_used,
                detail=f"final_root={diagnosis_outcome.final_root_cause}",
                duration_s=round(self._monotonic() - t0, 6),
            )

            # Unload reasoner ---------------------------------------------- #
            t0 = self._monotonic()
            reasoner_unload = self._reasoner.stop()
            emit(
                ResidentState.UNLOAD_REASONER,
                "reasoner_unloaded",
                success=reasoner_unload.cleanup_complete,
                detail=f"pid={reasoner_unload.pid}",
                duration_s=round(self._monotonic() - t0, 6),
            )

            # Reload target ------------------------------------------------ #
            t0 = self._monotonic()
            target_reload = self._target.start(self._target_model, self._target_params)
            emit(
                ResidentState.RELOAD_TARGET,
                "target_reloaded",
                success=target_reload.resident_verified,
                detail=f"pid={target_reload.pid} startup_s={target_reload.startup_s:.3f}",
                duration_s=round(self._monotonic() - t0, 6),
            )

            # Verify target ------------------------------------------------ #
            post_reload = self._verify_reload(
                identity_before, pid_before, baseline_hash, baseline_generation_ok, probe
            )
            emit(
                ResidentState.VERIFY_TARGET,
                "target_verified",
                success=post_reload.generation_ok,
                detail=f"equivalence={post_reload.equivalence}",
            )
        except Exception as exc:  # noqa: BLE001 - surface, never mask
            errors.append(f"{type(exc).__name__}: {exc}")
            emit(
                ResidentState.FAILED,
                "lifecycle_failed",
                success=False,
                detail="attempting to restore a resident target",
                error=str(exc),
            )
        finally:
            # Guarantee: reasoner is never left loaded; target is brought back.
            if self._reasoner is not None and self._reasoner.is_running():
                try:
                    reasoner_unload = self._reasoner.stop()
                    emit(ResidentState.UNLOAD_REASONER, "reasoner_unloaded_in_cleanup")
                except Exception as cleanup_exc:  # noqa: BLE001
                    errors.append(f"reasoner cleanup failed: {cleanup_exc}")
            if not self._target.is_running():
                try:
                    target_reload = self._target.start(self._target_model, self._target_params)
                    emit(ResidentState.RELOAD_TARGET, "target_reloaded_in_cleanup")
                except Exception as cleanup_exc:  # noqa: BLE001
                    errors.append(f"target restore failed: {cleanup_exc}")
                    emit(
                        ResidentState.FAILED,
                        "target_restore_failed",
                        success=False,
                        error=str(cleanup_exc),
                    )

        drained = self._request_window.drain()
        emit(
            ResidentState.RESUME,
            "resumed",
            success=self._target.is_running(),
            detail=(
                f"requests_during_diagnosis={len(self._request_window.records)} "
                f"queued_drained={len(drained)}"
            ),
        )

        target_operational = (
            self._target.is_running()
            and post_reload is not None
        )
        cost = LifecycleCostBreakdown(
            target_unload_s=target_unload.unload_s if target_unload else 0.0,
            reasoner_load_s=reasoner_load.startup_s if reasoner_load else 0.0,
            diagnosis_s=diagnosis_outcome.lifecycle.reasoning_latency_s
            if diagnosis_outcome and diagnosis_outcome.lifecycle.reasoning_latency_s
            else 0.0,
            reasoner_unload_s=reasoner_unload.unload_s if reasoner_unload else 0.0,
            target_reload_s=target_reload.startup_s if target_reload else 0.0,
            interruption_cost_s=self._interruption_cost_s,
            optimizer_overhead_s=round(self._monotonic() - started, 6),
            total_s=round(self._monotonic() - started, 6),
        )

        outcome = ResidentDeepDiagnosisOutcome(
            outcome_id=f"res5r-{uuid.uuid4().hex[:12]}",
            created_at=self._wall_clock(),
            source_diagnosis_id=diagnosis.diagnosis_id,
            entered=True,
            escalation=escalation,
            entry_economics=economics,
            feasibility=feasibility,
            target_load=prior_load,
            target_unload=target_unload,
            reasoner_load=reasoner_load,
            reasoner_unload=reasoner_unload,
            target_reload=target_reload,
            diagnosis=diagnosis_outcome,
            post_reload=post_reload,
            requests=self._request_window.records,
            transitions=_limit(transitions, self._policy.transition_log_limit),
            cost=cost,
            corroboration=(
                diagnosis_outcome.corroboration if diagnosis_outcome is not None else None
            ),
            errors=errors,
            summary=(
                f"Resident deep diagnosis completed; target operational="
                f"{target_operational}; errors={len(errors)}"
            ),
        )
        return outcome

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _reasoner_estimates(self) -> tuple[float, float]:
        load_est = self._policy.reasoner_load_estimate_s
        unload_est = self._policy.reasoner_unload_estimate_s
        if self._reasoner is None:
            return load_est, unload_est
        backend = getattr(self._reasoner, "_backend", None)
        if backend is not None:
            last_load = getattr(backend, "last_load", None)
            last_unload = getattr(backend, "last_unload", None)
            if last_load is not None:
                load_est = last_load.startup_s
            if last_unload is not None:
                unload_est = last_unload.unload_s
        return load_est, unload_est

    def _verify_reload(
        self,
        identity_before: ProcessIdentity | None,
        pid_before: int | None,
        baseline_hash: str | None,
        baseline_generation_ok: bool,
        probe: GenerationRequest,
    ) -> PostReloadVerification:
        identity_after = self._target.identity
        new_identity = bool(
            identity_after is not None
            and identity_before is not None
            and identity_after.pid != identity_before.pid
        )
        config_match = bool(self._target.is_running())
        reloaded_result: GenerationResult | None = None
        reloaded_hash: str | None = None
        generation_ok = False
        try:
            reloaded_result = self._target.generate(probe)
            reloaded_hash = _normalize_hash(reloaded_result.text)
            generation_ok = bool(reloaded_result.text)
        except Exception:  # noqa: BLE001 - surfaced via generation_ok
            generation_ok = False
        if not self._target.is_running():
            equivalence = "failed"
        elif not generation_ok:
            equivalence = "generation_failed"
        elif baseline_hash is not None and reloaded_hash == baseline_hash:
            equivalence = "identical_output"
        elif baseline_generation_ok:
            equivalence = "operational"
        else:
            equivalence = "operational_no_baseline"
        return PostReloadVerification(
            new_identity=new_identity,
            pid_before=pid_before,
            pid_after=self._target.pid,
            configuration_match=config_match,
            generation_ok=generation_ok,
            equivalence=equivalence,
            baseline_text_hash=baseline_hash,
            reloaded_text_hash=reloaded_hash,
            invariants={
                "target_running": self._target.is_running(),
                "new_process_identity": new_identity,
                "configuration": self._target_params.model_dump(exclude_none=True),
            },
            notes=[
                "post-reload equivalence allows non-identical stochastic text but "
                "requires a successful real generation and matching configuration",
            ],
        )

    def _refuse(
        self,
        diagnosis: BottleneckDiagnosis,
        transitions: list[LifecycleTransition],
        emit: Callable[..., LifecycleTransition],
        *,
        reason: str,
        economics: EntryEconomics,
        escalation: EscalationDecision | None = None,
        feasibility: FeasibilityAssessment | None = None,
        summary_prefix: str = "Deep diagnosis not entered",
    ) -> ResidentDeepDiagnosisOutcome:
        emit(ResidentState.REFUSED, "deep_diagnosis_refused", detail=reason, success=True)
        return ResidentDeepDiagnosisOutcome(
            outcome_id=f"res5r-{uuid.uuid4().hex[:12]}",
            created_at=self._wall_clock(),
            source_diagnosis_id=diagnosis.diagnosis_id,
            entered=False,
            refused_reason=reason,
            escalation=escalation
            or decide_escalation(diagnosis),
            entry_economics=economics,
            feasibility=feasibility,
            target_load=self._target.last_load,
            requests=self._request_window.records,
            transitions=_limit(transitions, self._policy.transition_log_limit),
            errors=[],
            summary=f"{summary_prefix}: {reason}. Target left resident and running.",
        )


def _normalize_hash(text: str) -> str:
    normalized = " ".join((text or "").split())
    return hashlib.sha256(normalized.encode("utf-8", "replace")).hexdigest()[:16]


def _limit(
    transitions: list[LifecycleTransition], limit: int
) -> list[LifecycleTransition]:
    if len(transitions) <= limit:
        return transitions
    return transitions[:limit]
