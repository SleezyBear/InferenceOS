"""Phase 10 runtime adaptive monitor: conservative active-runtime adaptation.

The monitor observes an active workload, temporally qualifies deviations, reuses the
Phase 4 diagnosis engine (with runtime temporal qualification), evaluates explicit
intervention economics, *recommends* deep analysis rather than forcing it, requires
explicit user approval for disruptive active-runtime changes, checkpoints before
pausing, reuses Phase 5/6/7/8 machinery for the controlled intervention, always
returns the workload safely to a verified state, and applies cooldown/anti-thrashing.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

from inferenceos.backends.process_control import process_identity
from inferenceos.config import Settings
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    AnomalyClass,
    ApprovalOutcome,
    BackendType,
    ExperimentDecisionKind,
    InterventionClass,
    KnowledgeWriteStatus,
    MeasurementMode,
    MetricQuality,
    ObservationSource,
    Reversibility,
    RestorationStatus,
    RuntimeState,
    RuntimeTerminationReason,
)
from inferenceos.contracts.experiment import (
    ExperimentPolicy,
    ExperimentResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.external import ExternalAdvisor, ExternalRecommendationRun
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.accounting import NetBenefitAssessment
from inferenceos.contracts.knowledge import ApplicabilityContext
from inferenceos.contracts.lifecycle import ResidentTargetRecovery, TargetLossReason
from inferenceos.contracts.model import ModelProfile
from inferenceos.optimizer_accounting import (
    AccountingPolicy,
    assess_net_benefit,
    build_overhead_attribution,
    distortion_from_stability,
)
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.optimization import (
    CandidateAction,
    CandidatePlan,
    OptimizationCandidate,
)
from inferenceos.contracts.optimization_loop import KnowledgeSelectionEffect
from inferenceos.contracts.reasoning import DeepDiagnosisOutcome
from inferenceos.contracts.profiling import (
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunIdentity,
)
from inferenceos.contracts.runtime import (
    ApprovalRecord,
    InterventionEconomics,
    ObservationWindow,
    ResourceImpact,
    RuntimeBaseline,
    RuntimeBudget,
    RuntimeCheckpoint,
    RuntimeCooldown,
    RuntimeIntervention,
    RuntimeKnowledgeQuery,
    RuntimeMonitorResult,
    RuntimeMonitorRun,
    RuntimeNotification,
    RuntimeObservation,
    RuntimeOpportunity,
    RuntimeOverhead,
    RuntimePolicy,
    RuntimeRecommendation,
    RuntimeSessionProvenance,
    RuntimeStateEvent,
    RuntimeTermination,
)
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.experiment.measurement import MeasurementHarness
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.experiment.service import ExperimentInputError, run_candidate_experiment
from inferenceos.hardware.profiler import detect_hardware_profile
from inferenceos.interfaces.reasoning_engine import ReasoningEngine
from inferenceos.knowledge.fingerprint import build_current_context
from inferenceos.knowledge.service import KnowledgeService
from inferenceos.knowledge.store import KnowledgeError
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy, build_policy
from inferenceos.optimization.registry import ActionRegistry
from inferenceos.reasoning.escalation import decide_escalation
from inferenceos.reasoning.lifecycle import run_deep_diagnosis
from inferenceos.reasoning.providers.base import ReasoningProvider
from inferenceos.runtime.approval import ApprovalProvider, AutoRejectApprovalProvider
from inferenceos.runtime.checkpoint import (
    capture_runtime_checkpoint,
    restore_runtime_checkpoint,
)
from inferenceos.runtime.economics import classify_action, evaluate_intervention_economics
from inferenceos.runtime.knowledge import knowledge_adjustment, query_knowledge, to_record
from inferenceos.runtime.notification import CollectingNotificationSink, NotificationSink
from inferenceos.runtime.telemetry import RuntimeTelemetrySampler, SystemTelemetrySampler
from inferenceos.runtime.window import ObservationBuffer
from inferenceos.runtime.workload import ActiveWorkload, RuntimeWorkloadExecutor, config_key

RUNTIME_PROFILER_VERSION = "phase10.runtime"

_TERMINAL = {
    RuntimeTerminationReason.DURATION_REACHED,
    RuntimeTerminationReason.MONITORING_BUDGET_EXHAUSTED,
    RuntimeTerminationReason.INTERVENTION_LIMIT_REACHED,
    RuntimeTerminationReason.EXTERNAL_TERMINATION,
    RuntimeTerminationReason.OPERATOR_REQUIRED,
    RuntimeTerminationReason.SAFETY_TERMINAL,
    RuntimeTerminationReason.ERROR,
}


def _metric_means(baseline: RuntimeBaseline | None) -> dict[str, float]:
    if baseline is None:
        return {}
    return {
        name: stats.mean
        for name, stats in baseline.metrics.items()
        if stats.mean is not None
    }


class RuntimeMonitor:
    """Conservative active-runtime monitor over a controlled workload."""

    def __init__(
        self,
        *,
        workload: ActiveWorkload,
        sampler: RuntimeTelemetrySampler | None = None,
        harness: MeasurementHarness | None = None,
        policy: RuntimePolicy | None = None,
        approval_provider: ApprovalProvider | None = None,
        notification_sink: NotificationSink | None = None,
        knowledge_service: KnowledgeService | None = None,
        registry: ActionRegistry | None = None,
        optimization_policy: OptimizationPolicy | None = None,
        experiment_policy: ExperimentPolicy | None = None,
        reasoning_engine: ReasoningEngine | None = None,
        proposal_provider: ReasoningProvider | None = None,
        enable_reasoning_proposals: bool = False,
        detector: BottleneckDetector | None = None,
        hardware: HardwareProfile | None = None,
        settings: Settings | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        diagnosis_confidence_threshold: float = 0.6,
        external_advisor: ExternalAdvisor | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        session_notes: list[str] | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._workload = workload
        self._sampler = sampler or SystemTelemetrySampler(settings=self._settings)
        self._harness = harness
        self._policy = policy or RuntimePolicy()
        self._approval = approval_provider or AutoRejectApprovalProvider()
        self._sink = notification_sink or CollectingNotificationSink()
        self._knowledge = knowledge_service
        self._registry = registry or build_default_registry()
        self._optimization_policy = optimization_policy or build_policy(self._settings)
        self._experiment_policy = experiment_policy or build_experiment_policy(self._settings)
        self._reasoning_engine = reasoning_engine
        self._proposal_provider = proposal_provider
        self._enable_reasoning_proposals = enable_reasoning_proposals
        self._detector = detector or BottleneckDetector()
        self._hardware = hardware
        self._clock = clock or time.perf_counter
        self._sleep = sleep or time.sleep
        self._confidence_threshold = diagnosis_confidence_threshold
        self._external_advisor = external_advisor
        # Phase 10R: runtime session identity + monitor process identity.
        self._run_id_override = run_id
        self._session_id = session_id or f"rtsession-{uuid.uuid4().hex[:10]}"
        # B-approval: caller-supplied provenance notes (e.g. unattended auto-approval
        # acceptance) persisted verbatim into the session provenance.
        self._session_notes = list(session_notes or [])
        self._monitor_pid = os.getpid()
        self._monitor_identity = process_identity(self._monitor_pid)
        self._target_identity_first: object | None = None
        # B2: explicit bounded resident-target recovery lifecycles.
        self._recoveries: list[ResidentTargetRecovery] = []

        # Mutable run state (declared here so the type checker sees one definition).
        self._state: RuntimeState = RuntimeState.INITIALIZING
        self._events: list[RuntimeStateEvent] = []
        self._buffer: ObservationBuffer = ObservationBuffer(
            capacity=self._policy.budget.observation_window_size,
            baseline_count=self._policy.budget.baseline_samples,
        )
        self._baseline: RuntimeBaseline | None = None
        self._baselines: list[RuntimeBaseline] = []
        self._baseline_observations: list[RuntimeObservation] = []
        self._last_window: ObservationWindow | None = None
        self._observations_count = 0
        self._diagnosis_events = 0
        self._deep_analysis_events = 0
        self._opportunities: list[RuntimeOpportunity] = []
        self._recommendations: list[RuntimeRecommendation] = []
        self._notifications: list[RuntimeNotification] = []
        self._approvals: list[ApprovalRecord] = []
        self._checkpoints: list[RuntimeCheckpoint] = []
        self._interventions: list[RuntimeIntervention] = []
        self._cooldowns: list[RuntimeCooldown] = []
        self._cooldown_until: dict[str, float] = {}
        self._alert_counts: dict[str, int] = {}
        self._suppressed: set[str] = set()
        self._knowledge_queries: list[RuntimeKnowledgeQuery] = []
        self._external_runs: list[ExternalRecommendationRun] = []
        self._external_consulted: set[str] = set()
        self._last_diagnosis_at: float = -1e12
        self._overhead: dict[str, float] = {}
        self._errors: list[str] = []
        self._budget: RuntimeBudget = self._policy.budget
        self._pending_rebase = False
        self._observations_since_rebase = 0
        self._started_wall = self._clock()
        self._fatal: RuntimeTermination | None = None

        self._reset()

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #
    def run(self, *, duration_s: float | None = None) -> RuntimeMonitorResult:
        self._reset()
        self._budget = self._policy.budget
        if duration_s is not None:
            self._budget = self._budget.model_copy(update={"max_duration_s": duration_s})
        now = datetime.now(timezone.utc)
        self._started_wall = self._clock()
        self._set_state(RuntimeState.INITIALIZING, "attaching to active workload")
        try:
            self._workload.start()
            self._target_identity_first = self._workload_target_identity()
            self._establish_baseline()
            self._set_state(RuntimeState.MONITORING, "monitoring active workload")
            termination = self._monitor_loop()
        except Exception as exc:  # noqa: BLE001 - never leave the workload uncontrolled
            self._errors.append(f"{type(exc).__name__}: {exc}")
            self._set_state(RuntimeState.ERROR_SAFE, f"unhandled error: {exc}")
            termination = RuntimeTermination(
                reason=RuntimeTerminationReason.ERROR,
                detail=str(exc),
                clean=False,
                at_observation=self._observations_count,
            )
        finally:
            if self._state not in (RuntimeState.TERMINAL_SAFE, RuntimeState.ERROR_SAFE):
                self._set_state(RuntimeState.TERMINAL_SAFE, "monitoring ended")
        return self._finish(termination, now)

    # ------------------------------------------------------------------ #
    # Baseline / loop                                                      #
    # ------------------------------------------------------------------ #
    def _workload_target_identity(self) -> object | None:
        getter = getattr(self._workload, "target_identity", None)
        if callable(getter):
            try:
                value: object = getter()
                return value
            except Exception:  # noqa: BLE001 - identity probing must not crash monitoring
                return None
        return None

    def _workload_target_pid(self) -> int | None:
        getter = getattr(self._workload, "target_pid", None)
        if callable(getter):
            try:
                value: int | None = getter()
                return value
            except Exception:  # noqa: BLE001
                return None
        return None

    def _target_identity_ok(self) -> bool | None:
        """Verify the resident target identity against the live OS process.

        Returns ``None`` when no independent target identity is available (e.g. the
        process-per-generation workload), ``True`` when it matches, and ``False`` when
        the process exited or a PID was reused — the caller must fail closed.
        """
        identity = self._workload_target_identity()
        if identity is None:
            return None
        pid = getattr(identity, "pid", None)
        create_time = getattr(identity, "create_time", None)
        if pid is None or create_time is None:
            return None
        live = process_identity(int(pid))
        if live is None:
            return False
        return abs(live.create_time - float(create_time)) <= 0.01

    def _attempt_target_recovery(self, *, reason: str) -> bool:
        """Delegate one bounded recovery to the workload and record the lifecycle.

        The workload owns the single recovery transition; the monitor only observes
        the outcome. Recovery is never attempted twice for the same loss because the
        workload refuses a second attempt and reports unrecoverable.
        """
        recoverer = getattr(self._workload, "recover_target", None)
        if not callable(recoverer):
            return False
        self._set_state(
            RuntimeState.TARGET_LOST,
            "resident target lost; requesting one bounded recovery",
            data={"reason": reason},
        )
        try:
            recovery = recoverer(reason=reason)
        except Exception as exc:  # noqa: BLE001 - recovery failure must be recorded
            self._errors.append(f"resident target recovery errored: {type(exc).__name__}: {exc}")
            self._set_state(
                RuntimeState.TARGET_UNRECOVERABLE,
                "resident target recovery raised; entering unrecoverable state",
            )
            return False
        if not isinstance(recovery, ResidentTargetRecovery):
            self._errors.append("resident target recovery returned no evidence record")
            self._set_state(
                RuntimeState.TARGET_UNRECOVERABLE,
                "resident target recovery returned no evidence; failing closed",
            )
            return False
        self._recoveries.append(recovery)
        if recovery.recovered:
            self._set_state(
                RuntimeState.TARGET_RECOVERED,
                "resident target recovered and verified",
                data={
                    "old_pid": recovery.old_pid,
                    "new_pid": recovery.new_pid,
                    "identity_verified": recovery.identity_verified,
                    "readiness_verified": recovery.readiness_verified,
                    "configuration_verified": recovery.configuration_verified,
                },
            )
            # A verified replacement is a new reference; re-anchor the baseline.
            self._pending_rebase = True
            self._observations_since_rebase = 0
            return True
        self._set_state(
            RuntimeState.TARGET_UNRECOVERABLE,
            "resident target recovery failed; entering unrecoverable state",
            data={
                "old_pid": recovery.old_pid,
                "new_pid": recovery.new_pid,
                "failure_reason": recovery.failure_reason,
            },
        )
        return False

    def _collect_recoveries(self) -> list[ResidentTargetRecovery]:
        getter = getattr(self._workload, "recovery_records", None)
        if callable(getter):
            try:
                records = getter()
            except Exception:  # noqa: BLE001 - evidence collection must not crash
                records = None
            if isinstance(records, list):
                return records
        return list(self._recoveries)

    def _establish_baseline(self) -> None:
        required = max(self._policy.budget.baseline_samples, 2)
        for _ in range(required):
            tick = self._clock()
            self._collect_observation(condition="baseline")
            # Pace the baseline window so real generations complete and observed
            # throughput is represented (and so baseline drift is time-anchored).
            self._sleep(max(0.0, self._interval() - (self._clock() - tick)))
        descriptor = self._workload.descriptor()
        baseline = self._buffer.establish_baseline(
            workload_id=descriptor.workload_id,
            model_ref=descriptor.model_ref,
            backend_name=descriptor.backend_name,
            configuration=self._workload.configuration(),
            context_length=self._workload.configuration().context_length,
            source="initial",
        )
        if baseline is None:
            raise RuntimeError("could not establish a runtime baseline")
        self._baseline = baseline
        self._baselines.append(baseline)
        self._baseline_observations = [
            o for o in self._buffer.observations if o.condition == "baseline"
        ]

    def _monitor_loop(self) -> RuntimeTermination:
        while True:
            if self._fatal is not None:
                return self._fatal
            elapsed = self._clock() - self._started_wall
            self._budget = self._budget.model_copy(
                update={"optimizer_overhead_used_s": self._optimizer_overhead()}
            )
            exhaustion = self._budget.exhaustion(elapsed_s=elapsed)
            if exhaustion is not None:
                return self._terminate(exhaustion, f"budget reached: {exhaustion.value}")

            snapshot = self._workload.snapshot()
            if snapshot.terminated_externally:
                return self._terminate(
                    RuntimeTerminationReason.EXTERNAL_TERMINATION,
                    "the workload was terminated externally",
                )
            if snapshot.unrecoverable:
                return self._terminate(
                    RuntimeTerminationReason.SAFETY_TERMINAL,
                    "resident target is unrecoverable after the single permitted "
                    "recovery attempt; failing closed",
                )
            identity_ok = self._target_identity_ok()
            if identity_ok is False:
                if not self._attempt_target_recovery(reason=TargetLossReason.IDENTITY_CHANGED):
                    return self._terminate(
                        RuntimeTerminationReason.SAFETY_TERMINAL,
                        "resident target was lost and bounded recovery did not produce a "
                        "verified replacement; failing closed",
                    )

            if self._state is RuntimeState.COOLDOWN and not self._cooling_any():
                self._set_state(RuntimeState.MONITORING, "cooldown expired")

            tick = self._clock()
            self._collect_observation(condition="observed")
            self._maybe_rebase()

            window = self._buffer.qualify(
                threshold=self._policy.deviation_threshold,
                hysteresis=self._policy.deviation_threshold * self._policy.hysteresis_ratio,
                persistence=self._policy.persistence_samples,
            )
            self._last_window = window
            self._handle_window(window)

            if self._state is RuntimeState.AWAITING_USER_APPROVAL:
                # Should never persist past an approval call; defensive.
                self._set_state(RuntimeState.MONITORING, "stale approval state reset")

            self._sleep(max(0.0, self._interval() - (self._clock() - tick)))

    def _handle_window(self, window: ObservationWindow) -> None:
        # After a retained change we first settle a fresh baseline before considering
        # further changes (anti-thrashing: never stack interventions on stale baselines).
        if self._pending_rebase:
            return
        key = window.condition_key
        if window.anomaly in (AnomalyClass.PERSISTENT, AnomalyClass.REPEATED) and key is not None:
            if key in self._suppressed or self._cooling(key):
                return
            self._process_opportunity(window)
        # TRANSIENT / RECOVERED / NONE: keep monitoring without acting.

    def _process_opportunity(self, window: ObservationWindow) -> None:
        started = self._clock()
        opportunity = self._build_opportunity(window)
        self._set_state(
            RuntimeState.OPPORTUNITY_SUSPECTED,
            f"suspected opportunity: {opportunity.summary}",
        )
        now = self._clock()
        if now - self._last_diagnosis_at < self._policy.diagnosis_cooldown_s:
            self._opportunities.append(opportunity)
            self._set_state(RuntimeState.MONITORING, "diagnosis cooldown active")
            return
        self._last_diagnosis_at = now

        self._set_state(RuntimeState.DIAGNOSING, "running deterministic diagnosis")
        report = self._build_report()
        diagnosis = self._detector.detect(report, source_report_path="<runtime-monitor>")
        self._budget = self._budget.model_copy(
            update={"diagnoses_used": self._budget.diagnoses_used + 1}
        )
        self._diagnosis_events += 1
        self._record_overhead("diagnosis_s", self._clock() - started)

        primary = diagnosis.primary
        opportunity = opportunity.model_copy(
            update={
                "diagnosis_id": diagnosis.diagnosis_id,
                "diagnosis_kind": primary.kind if primary is not None else None,
                "confidence": diagnosis.confidence,
                "strength": diagnosis.evidence_strength,
                "meaningful": diagnosis.meaningful_opportunity,
            }
        )

        # Phase 11 extension: consult external attribution/recommendation for this
        # persistent condition (never duplicated monitoring; read-only observation).
        self._consult_external_advisor(opportunity, diagnosis)

        escalation = decide_escalation(
            diagnosis, enabled=True, confidence_threshold=self._confidence_threshold
        )
        reasoning_required = bool(
            escalation.should_escalate
            and self._reasoning_engine is not None
            and self._reasoning_engine.is_available()
            and self._budget.deep_analyses_used < self._budget.max_deep_analyses
        )

        if not diagnosis.meaningful_opportunity and not reasoning_required:
            self._opportunities.append(opportunity)
            self._set_state(
                RuntimeState.MONITORING,
                f"no meaningful opportunity: {diagnosis.no_opportunity_reason or diagnosis.summary}",
            )
            return

        candidate, action, knowledge_records = self._select_candidate(
            diagnosis, report, outcome=None
        )
        self._knowledge_queries.extend(knowledge_records)
        opportunity = opportunity.model_copy(
            update={
                "candidate_ids": [candidate.candidate_id] if candidate is not None else [],
            }
        )
        self._opportunities.append(opportunity)

        economics = evaluate_intervention_economics(
            opportunity,
            candidate=candidate,
            action=action,
            policy=self._policy,
            reasoning_required=reasoning_required,
            measured_overhead_s=self._optimizer_overhead(),
        )
        if not economics.beneficial:
            self._set_state(RuntimeState.COOLDOWN, "intervention is not worth the disruption")
            self._start_cooldown(
                opportunity.condition_key,
                "Intervention economics did not justify interrupting the workload.",
                trigger="economics",
            )
            return

        recommendation = self._build_recommendation(
            opportunity, candidate, action, economics, reasoning_required
        )
        self._recommendations.append(recommendation)

        count = self._alert_counts.get(opportunity.condition_key, 0)
        if count >= self._policy.max_alerts_per_condition:
            self._suppressed.add(opportunity.condition_key)
            self._set_state(RuntimeState.COOLDOWN, "alert limit reached for this condition")
            self._start_cooldown(
                opportunity.condition_key,
                "Maximum alerts for this unchanged condition already delivered.",
                trigger="alert_limit",
            )
            return
        self._alert_counts[opportunity.condition_key] = count + 1

        self._set_state(
            RuntimeState.DEEP_ANALYSIS_RECOMMENDED,
            f"deep analysis recommended for {opportunity.kind}",
        )
        self._notify(recommendation)

        if economics.requires_approval:
            approval = self._request_approval(recommendation)
            if approval.outcome is not ApprovalOutcome.APPROVED:
                self._set_state(RuntimeState.COOLDOWN, f"approval {approval.outcome.value}")
                self._start_cooldown(
                    opportunity.condition_key,
                    f"Approval outcome was {approval.outcome.value}; no intervention.",
                    trigger="approval",
                )
                return
        else:
            approval = ApprovalRecord(
                approval_id=f"appr-{uuid.uuid4().hex[:10]}",
                proposal_id=recommendation.proposal_id,
                outcome=ApprovalOutcome.NOT_REQUIRED,
                context_identity=recommendation.context_identity,
                detail="non-disruptive action permitted by policy",
            )
            self._approvals.append(approval)

        self._run_intervention(
            opportunity, recommendation, approval, diagnosis, report, reasoning_required
        )

    # ------------------------------------------------------------------ #
    # Phase 11 external integration                                        #
    # ------------------------------------------------------------------ #
    def _consult_external_advisor(
        self, opportunity: RuntimeOpportunity, diagnosis: BottleneckDiagnosis
    ) -> None:
        advisor = self._external_advisor
        if advisor is None:
            return
        key = opportunity.condition_key
        if key in self._external_consulted:
            return
        self._external_consulted.add(key)
        started = self._clock()
        try:
            external_run = advisor.advise(
                target_pid=self._workload.target_pid(),
                target_identity=None,
                target_degraded=bool(
                    diagnosis.meaningful_opportunity or diagnosis.insufficient_evidence
                ),
                target_metrics=self._latest_observed_metrics(),
                context_identity=key,
                diagnosed_bottleneck=(
                    diagnosis.primary.kind.value
                    if diagnosis.primary is not None
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001 - external advice must never crash monitoring
            self._errors.append(f"external advisor failed: {type(exc).__name__}: {exc}")
            return
        self._record_overhead("diagnosis_s", self._clock() - started)
        self._external_runs.append(external_run)
        self._events.append(
            RuntimeStateEvent(
                state=self._state,
                timestamp=datetime.now(timezone.utc),
                observation=self._observations_count,
                detail=(
                    "external attribution consulted: "
                    f"{len(external_run.recommendations)} recommendation(s), "
                    f"{len(external_run.action_results)} action(s)"
                ),
            )
        )

    def _latest_observed_metrics(self) -> dict[str, float]:
        observations = self._buffer.observations
        if not observations:
            return {}
        return dict(observations[-1].metrics)

    # ------------------------------------------------------------------ #
    # Approval / notification                                              #
    # ------------------------------------------------------------------ #
    def _request_approval(self, recommendation: RuntimeRecommendation) -> ApprovalRecord:
        self._set_state(RuntimeState.AWAITING_USER_APPROVAL, "waiting for user approval")
        started = self._clock()
        approval = self._approval.request(
            recommendation, timeout_s=self._budget.approval_timeout_s
        )
        self._record_overhead("notification_s", self._clock() - started)
        if approval.outcome is ApprovalOutcome.APPROVED and not self._approval_valid(
            approval, recommendation
        ):
            approval = approval.model_copy(
                update={
                    "outcome": ApprovalOutcome.STALE,
                    "detail": "approval does not match the current proposal/context",
                }
            )
        self._approvals.append(approval)
        return approval

    def _approval_valid(
        self, approval: ApprovalRecord, recommendation: RuntimeRecommendation
    ) -> bool:
        return (
            approval.proposal_id == recommendation.proposal_id
            and approval.context_identity == recommendation.context_identity
        )

    def _notify(self, recommendation: RuntimeRecommendation) -> None:
        notification = RuntimeNotification(
            notification_id=f"notif-{uuid.uuid4().hex[:10]}",
            proposal_id=recommendation.proposal_id,
            title="InferenceOS runtime optimization recommendation",
            summary=recommendation.why_it_matters,
            recommendation=recommendation,
        )
        self._notifications.append(notification)
        self._sink.notify(notification)

    # ------------------------------------------------------------------ #
    # Intervention                                                         #
    # ------------------------------------------------------------------ #
    def _run_intervention(
        self,
        opportunity: RuntimeOpportunity,
        recommendation: RuntimeRecommendation,
        approval: ApprovalRecord,
        diagnosis: BottleneckDiagnosis,
        report: BaselineReport,
        reasoning_required: bool,
    ) -> None:
        started = self._clock()
        intervention = RuntimeIntervention(
            intervention_id=f"rtint-{uuid.uuid4().hex[:10]}",
            proposal_id=recommendation.proposal_id,
            approval_outcome=approval.outcome,
            opportunity_detected_at=opportunity.detected_at,
            target_pid=self._workload_target_pid(),
            target_identity_ok=self._target_identity_ok(),
        )
        descriptor = self._workload.descriptor()

        # --- Checkpoint (never interrupt without a verified checkpoint) ---
        self._set_state(RuntimeState.CHECKPOINTING, "capturing protected workload state")
        ck_started = self._clock()
        checkpoint = capture_runtime_checkpoint(
            self._workload,
            reason=f"runtime {opportunity.kind} intervention",
            context_identity=recommendation.context_identity,
            previous_baseline_id=self._baseline.baseline_id if self._baseline else None,
        )
        self._record_overhead("checkpoint_s", self._clock() - ck_started)
        self._checkpoints.append(checkpoint)
        intervention = intervention.model_copy(update={"checkpoint_id": checkpoint.checkpoint_id})
        if not checkpoint.verified:
            self._errors.append("runtime checkpoint could not be verified")
            intervention = intervention.model_copy(
                update={"errors": ["checkpoint not verified"], "finished_at": datetime.now(timezone.utc)}
            )
            self._interventions.append(intervention)
            self._set_state(RuntimeState.COOLDOWN, "checkpoint failed; workload not interrupted")
            self._start_cooldown(
                opportunity.condition_key,
                "Checkpoint could not be established; workload left untouched.",
                trigger="checkpoint_failed",
            )
            return

        # --- Pause (drain in-flight; never kill) ---
        self._set_state(RuntimeState.PAUSING, "draining the in-flight generation")
        pre_pause = self._workload.snapshot()
        in_flight_at_start = pre_pause.generating
        in_flight_request_id = pre_pause.in_flight_request_id
        pause_started = self._clock()
        drained = self._workload.pause()
        self._record_overhead("pause_s", self._clock() - pause_started)
        post_pause = self._workload.snapshot()
        in_flight_completed = not post_pause.generating
        intervention = intervention.model_copy(
            update={
                "in_flight_at_start": in_flight_at_start,
                "in_flight_request_id": in_flight_request_id,
                # The in-flight request was drained to completion before any mutation.
                "in_flight_completed_at": (
                    datetime.now(timezone.utc)
                    if (in_flight_at_start and in_flight_completed)
                    else None
                ),
                "deferred_for_in_flight": bool(in_flight_at_start),
            }
        )
        if not drained:
            resumed_after_failure = self._workload.resume()
            self._errors.append("workload pause/drain did not complete")
            intervention = intervention.model_copy(
                update={
                    "finished_at": datetime.now(timezone.utc),
                    "errors": ["pause/drain failed"],
                    "resume_verified": resumed_after_failure,
                }
            )
            self._interventions.append(intervention)
            self._set_state(RuntimeState.ERROR_SAFE, "pause/drain failed; workload resumed")
            self._fatal = RuntimeTermination(
                reason=RuntimeTerminationReason.ERROR,
                detail="pause/drain failed; the workload was not interrupted and the run stopped",
                clean=False,
                safety=True,
                at_observation=self._observations_count,
            )
            return

        # --- Deep analysis (bounded) + controlled experiment ---
        self._set_state(RuntimeState.INTERVENING, "running controlled intervention")
        outcome = None
        if reasoning_required and self._reasoning_engine is not None:
            r_started = self._clock()
            outcome = run_deep_diagnosis(
                diagnosis,
                report=report,
                engine=self._reasoning_engine,
                enabled=True,
                confidence_threshold=self._confidence_threshold,
            )
            self._record_overhead("reasoning_s", self._clock() - r_started)
            self._budget = self._budget.model_copy(
                update={"deep_analyses_used": self._budget.deep_analyses_used + 1}
            )
            self._deep_analysis_events += 1

        plan = generate_candidates(
            diagnosis,
            report=report,
            outcome=outcome,
            registry=self._registry,
            policy=self._optimization_policy,
            provider=self._proposal_provider,
            enable_reasoning=self._enable_reasoning_proposals and self._proposal_provider is not None,
        )
        candidate, action, knowledge_records = self._select_candidate(
            diagnosis, report, outcome=outcome, plan=plan
        )
        self._knowledge_queries.extend(knowledge_records)
        if candidate is None or action is None:
            self._workload.resume()
            resume_verified = self._workload.verify_resume(
                previous_requests=pre_pause.requests_served, timeout_s=30.0
            )
            intervention = intervention.model_copy(
                update={
                    "finished_at": datetime.now(timezone.utc),
                    "notes": ["no eligible candidate after deep analysis"],
                    "resume_verified": resume_verified,
                    "duration_s": self._clock() - started,
                }
            )
            self._interventions.append(intervention)
            self._set_state(RuntimeState.COOLDOWN, "no eligible candidate")
            self._start_cooldown(
                opportunity.condition_key, "No eligible candidate to apply.", trigger="no_candidate"
            )
            return

        intervention = intervention.model_copy(
            update={
                "candidate_id": candidate.candidate_id,
                "action_id": action.action_id,
                "parameters": dict(candidate.parameters),
            }
        )
        executor = RuntimeWorkloadExecutor(self._workload, backend_name=descriptor.backend_name)
        result: ExperimentResult | None = None
        # The intervention mutation only begins here, after the in-flight request has
        # been drained to completion and a verified checkpoint exists.
        applied_at = datetime.now(timezone.utc)
        try:
            exp_started = self._clock()
            result = run_candidate_experiment(
                candidate,
                plan=plan,
                diagnosis=diagnosis,
                report=report,
                outcome=outcome,
                workload_id=descriptor.workload_id,
                registry=self._registry,
                optimization_policy=self._optimization_policy,
                experiment_policy=self._experiment_policy,
                executor=executor,
                profiler=self._harness_or_default(),
            )
            self._record_overhead("experiment_s", self._clock() - exp_started)
        except ExperimentInputError as exc:
            self._record_overhead("experiment_s", self._clock() - started)
            self._errors.append(f"runtime experiment input error: {exc}")

        self._set_state(RuntimeState.VALIDATING, "validating the intervention")
        validation_started = self._clock()
        retained = bool(
            result is not None
            and result.status is ExperimentDecisionKind.IMPROVED
            and result.decision_detail is not None
            and result.decision_detail.keep
        )
        quality = result.quality if result is not None else None
        restoration = (
            result.restoration.status if result is not None and result.restoration is not None
            else RestorationStatus.NOT_REQUIRED
        )
        self._record_overhead("validation_s", self._clock() - validation_started)

        # --- Resume (verified) ---
        self._set_state(RuntimeState.RESUMING, "resuming the workload")
        resume_started = self._clock()
        resumed = self._workload.resume()
        resumed_at = datetime.now(timezone.utc)
        resume_verified = resumed and self._workload.verify_resume(
            previous_requests=pre_pause.requests_served, timeout_s=45.0
        )
        self._record_overhead("resume_s", self._clock() - resume_started)

        if not resume_verified:
            self._errors.append("workload resume could not be verified")
            intervention = intervention.model_copy(
                update={
                    "finished_at": datetime.now(timezone.utc),
                    "resume_verified": False,
                    "errors": ["resume not verified"],
                    "duration_s": self._clock() - started,
                }
            )
            self._interventions.append(intervention)
            self._set_state(
                RuntimeState.TERMINAL_SAFE,
                "workload resume could not be verified; surfacing the exact condition",
            )
            self._fatal = RuntimeTermination(
                reason=RuntimeTerminationReason.SAFETY_TERMINAL,
                detail="workload resume could not be verified",
                clean=False,
                safety=True,
                at_observation=self._observations_count,
            )
            return

        # --- Finalize keep/revert + knowledge persistence ---
        knowledge_status = None
        if result is not None:
            if retained and result.request is not None:
                self._workload.set_configuration(result.request.experiment_configuration)
                self._pending_rebase = True
                self._observations_since_rebase = 0
            else:
                # The Phase 7 engine restored the baseline; enforce the checkpoint too.
                restore_runtime_checkpoint(self._workload, checkpoint)
                self._pending_rebase = False
            knowledge_status = self._persist(result)

        duration = self._clock() - started
        final_notes = [
            "in-flight generation drained to completion"
            if in_flight_completed
            else "no in-flight generation at pause",
            f"retained={retained}",
        ]
        if result is not None:
            if result.baseline_measurement is not None:
                final_notes.extend(
                    f"baseline: {note}" for note in result.baseline_measurement.notes
                )
            if result.candidate_measurement is not None:
                final_notes.extend(
                    f"candidate: {note}" for note in result.candidate_measurement.notes
                )
            final_notes.extend(f"experiment_error: {err}" for err in result.errors)
        intervention = intervention.model_copy(
            update={
                "finished_at": datetime.now(timezone.utc),
                "decision": result.status if result is not None else None,
                "experiment_id": result.experiment_id if result is not None else None,
                "retained": retained,
                "quality_status": quality.status if quality is not None else None,
                "restoration_status": restoration,
                "validation_ok": bool(
                    result is not None
                    and result.status is ExperimentDecisionKind.IMPROVED
                    and quality is not None
                    and quality.validated
                ),
                "resume_verified": resume_verified,
                "duration_s": duration,
                "knowledge_write_status": knowledge_status,
                "intervention_applied_at": applied_at,
                "resumed_at": resumed_at,
                "target_identity_ok": self._target_identity_ok(),
                "notes": final_notes,
            }
        )
        self._interventions.append(intervention)
        self._budget = self._budget.model_copy(
            update={
                "interventions_used": self._budget.interventions_used + 1,
                "cumulative_intervention_s": self._budget.cumulative_intervention_s + duration,
            }
        )
        self._set_state(RuntimeState.COOLDOWN, "intervention complete; cooling down")
        self._start_cooldown(
            opportunity.condition_key,
            "Intervention completed; anti-thrashing cooldown applied.",
            trigger="intervention",
        )

    # ------------------------------------------------------------------ #
    # Candidate selection / report building                                #
    # ------------------------------------------------------------------ #
    def _select_candidate(
        self,
        diagnosis: BottleneckDiagnosis,
        report: BaselineReport,
        *,
        outcome: DeepDiagnosisOutcome | None,
        plan: CandidatePlan | None = None,
    ) -> tuple[OptimizationCandidate | None, CandidateAction | None, list[RuntimeKnowledgeQuery]]:
        plan = plan or generate_candidates(
            diagnosis,
            report=report,
            outcome=outcome,
            registry=self._registry,
            policy=self._optimization_policy,
            provider=self._proposal_provider,
            enable_reasoning=self._enable_reasoning_proposals and self._proposal_provider is not None,
        )
        eligible = [
            candidate
            for candidate in plan.candidates
            if candidate.eligibility.value == "eligible" and candidate.executable
        ]
        context = self._applicability_context(diagnosis, report)
        scored: list[tuple[float, str, OptimizationCandidate, CandidateAction]] = []
        queries: list[RuntimeKnowledgeQuery] = []
        for candidate in eligible:
            action = self._registry.get_optional(candidate.action_id)
            if action is None:
                continue
            if classify_action(action, self._policy) is InterventionClass.EXTERNAL_SYSTEM_WIDE:
                # External/system-wide actions remain out of Phase 10 scope.
                continue
            adjustment = 0.0
            if self._knowledge is not None:
                try:
                    evidence = query_knowledge(
                        candidate,
                        service=self._knowledge,
                        current_context=context,
                        policy=self._policy,
                    )
                except KnowledgeError as exc:  # pragma: no cover - store guarded
                    self._errors.append(f"runtime knowledge query failed: {exc}")
                    evidence = None
                if evidence is not None:
                    queries.append(to_record(evidence))
                    if evidence.effect is KnowledgeSelectionEffect.REJECTED:
                        continue
                    adjustment = knowledge_adjustment(evidence, self._policy)
            base = candidate.ranking.composite_score if candidate.ranking is not None else 0.0
            scored.append((base + adjustment, candidate.candidate_id, candidate, action))
        scored.sort(key=lambda item: (-item[0], item[1]))
        if not scored:
            return None, None, queries
        _, _, candidate, action = scored[0]
        return candidate, action, queries

    def _applicability_context(
        self, diagnosis: BottleneckDiagnosis, report: BaselineReport
    ) -> ApplicabilityContext:
        primary = diagnosis.primary
        conditions: dict[str, float] = {}
        observation = self._buffer.observations[-1] if self._buffer.total else None
        if observation is not None:
            for name in ("available_ram_bytes", "hard_page_faults_per_sec", "peak_rss_bytes"):
                if name in observation.metrics:
                    conditions[name] = observation.metrics[name]
        return build_current_context(
            hardware=self._hardware_profile(),
            model=self._workload.model_profile(),
            workload=self._workload.workload_profile(),
            backend=report.backend,
            runtime=self._workload.configuration(),
            diagnosis=primary.kind.value if primary is not None else None,
            conditions=conditions,
        )

    def _build_report(self) -> BaselineReport:
        descriptor = self._workload.descriptor()
        model = self._workload.model_profile()
        workload = self._workload.workload_profile()
        hardware = self._hardware_profile()
        recent = self._buffer.observations[-max(self._policy.persistence_samples, 2) :]
        runs = [
            self._to_run(obs, condition="control", model=model, workload=workload, hardware=hardware)
            for obs in self._baseline_observations
        ] + [
            self._to_run(obs, condition="observed", model=model, workload=workload, hardware=hardware)
            for obs in recent
        ]
        backend = runs[0].context.config if runs else ProfilingConfig()
        return BaselineReport(
            profiler_version=RUNTIME_PROFILER_VERSION,
            suite_version="phase10",
            generated_at=datetime.now(timezone.utc),
            hardware=hardware,
            backend=backend,
            workloads=[workload],
            runs=runs,
            groups=[],
            cold_warm_semantics=(
                "runtime monitoring: baseline window vs recent window under the active workload"
            ),
            resident_warm_supported=False,
            limitations=[
                "runtime observations are a bounded window, not the full Phase 3 harness",
                "resource consumption alone is not treated as a bottleneck",
            ],
            notes=f"runtime window for {descriptor.workload_id}",
        )

    def _to_run(
        self,
        observation: RuntimeObservation,
        *,
        condition: str,
        model: ModelProfile,
        workload: WorkloadProfile,
        hardware: HardwareProfile,
    ) -> ProfilingRun:
        configuration = self._workload.descriptor().configuration
        offloaded = configuration.gpu_offload_layers or 0
        backend_type = BackendType.VULKAN if offloaded > 0 else BackendType.CPU
        config = ProfilingConfig(
            backend_name=self._workload.descriptor().backend_name,
            backend_type=backend_type,
            gpu_layers_offloaded=configuration.gpu_offload_layers,
            gpu_layers_total=configuration.gpu_offload_layers,
            context_length=observation.context_length,
            use_mmap=configuration.use_mmap,
            extra={"condition": condition},
        )
        metrics = RunMetrics(
            decode_tokens_per_sec=observation.metrics.get("decode_tokens_per_sec"),
            total_latency_s=observation.metrics.get("total_latency_s"),
            peak_rss_bytes=(
                int(observation.metrics["peak_rss_bytes"])
                if "peak_rss_bytes" in observation.metrics
                else None
            ),
            prompt_tokens_per_sec=observation.metrics.get("prompt_tokens_per_sec"),
            system_cpu_percent=observation.metrics.get("system_cpu_percent"),
            gpu_utilization_percent=observation.metrics.get("gpu_utilization_percent"),
            hard_page_faults_per_sec=observation.metrics.get("hard_page_faults_per_sec"),
            available_ram_bytes=(
                int(observation.metrics["available_ram_bytes"])
                if "available_ram_bytes" in observation.metrics
                else None
            ),
            total_ram_bytes=(
                int(observation.metrics["total_ram_bytes"])
                if "total_ram_bytes" in observation.metrics
                else None
            ),
        )
        return ProfilingRun(
            identity=RunIdentity(
                run_id=observation.observation_id,
                profiler_version=RUNTIME_PROFILER_VERSION,
                mode=MeasurementMode.WARM,
                started_at=observation.captured_at,
                finished_at=observation.captured_at,
            ),
            context=RunContext(
                hardware=hardware, model=model, workload=workload, config=config
            ),
            observation=Observation(
                observation_id=observation.observation_id,
                captured_at=observation.captured_at,
                source=ObservationSource.CONTINUOUS_MONITOR,
                mode=MeasurementMode.WARM,
                backend=backend_type,
                model_ref=model.name,
                workload_ref=workload.workload_id,
                metrics=metrics,
                system_state=observation.system_state,
            ),
            metric_quality={
                "decode_tokens_per_sec": MetricQuality.MEASURED,
                "hard_page_faults_per_sec": MetricQuality.MEASURED,
            },
        )

    # ------------------------------------------------------------------ #
    # Opportunity / recommendation                                         #
    # ------------------------------------------------------------------ #
    def _build_opportunity(self, window: ObservationWindow) -> RuntimeOpportunity:
        descriptors = [
            f"{d.metric}: {d.baseline_mean:.4g} -> {d.recent_mean:.4g} "
            f"({d.relative_delta:+.1%})"
            for d in window.deviations
            if d.exceeds_threshold and d.baseline_mean is not None and d.recent_mean is not None
        ]
        kind = "workload_change" if window.context_changed else self._dominant_metric(window)
        deviation_score = max(
            (d.deviation_score for d in window.deviations), default=0.0
        )
        summary = (
            f"{kind} sustained for {window.persistence_samples} sample(s)"
            if not window.context_changed
            else "active workload configuration changed"
        )
        return RuntimeOpportunity(
            opportunity_id=f"rtopp-{uuid.uuid4().hex[:10]}",
            anomaly=window.anomaly,
            condition_key=window.condition_key or f"{kind}:unknown",
            kind=kind,
            persistence_samples=window.persistence_samples,
            deviation_score=deviation_score,
            context_changed=window.context_changed,
            summary=summary,
            evidence=descriptors + list(window.rationale),
        )

    def _dominant_metric(self, window: ObservationWindow) -> str:
        best = max(
            window.deviations,
            key=lambda d: d.deviation_score if d.exceeds_threshold else 0.0,
            default=None,
        )
        return best.metric if best is not None else "telemetry"

    def _build_recommendation(
        self,
        opportunity: RuntimeOpportunity,
        candidate: OptimizationCandidate | None,
        action: CandidateAction | None,
        economics: InterventionEconomics,
        reasoning_required: bool,
    ) -> RuntimeRecommendation:
        candidate_action = action
        if candidate is not None and candidate_action is not None:
            action_id: str | None = candidate_action.action_id
            params = dict(candidate.parameters)
            expected_benefit = (
                f"{candidate.benefit.metric} ({candidate.benefit.direction}); "
                f"confidence {candidate.benefit.confidence:.2f}"
            )
            reversible = candidate_action.reversibility is Reversibility.REVERSIBLE
        else:
            action_id = None
            params = {}
            expected_benefit = "unknown until measured"
            reversible = True
        costs = economics.costs
        expected_cost = sum(costs.values())
        disruption = costs.get("pause", 0.0) + costs.get("experiment", 0.0) + costs.get("reload", 0.0)
        context_identity = self._context_identity()
        now = datetime.now(timezone.utc)
        return RuntimeRecommendation(
            proposal_id=f"rtprop-{uuid.uuid4().hex[:10]}",
            created_at=now,
            opportunity_id=opportunity.opportunity_id,
            condition=opportunity.summary
            + (f" | evidence: {'; '.join(opportunity.evidence[:3])}" if opportunity.evidence else ""),
            why_it_matters=(
                "A sustained condition may materially affect the active workload; "
                "a controlled evaluation could improve it without harming quality."
            ),
            analysis_summary=(
                "Run bounded deep analysis, then a controlled experiment (apply candidate, "
                "measure real before/after, validate quality, keep or restore)."
                if reasoning_required
                else "Run a controlled experiment (apply candidate, measure real before/after, "
                "validate quality, keep or restore)."
            ),
            proposed_action_id=action_id,
            proposed_parameters=params,
            expected_benefit=expected_benefit,
            expected_cost_s=expected_cost,
            expected_disruption_s=disruption,
            pause_required=True,
            checkpoint_planned=True,
            reversible=reversible,
            expected_learning=(
                "Whether the suspected condition is causal and whether the candidate improves "
                "the workload within quality tolerance."
            ),
            context_identity=context_identity,
            expires_at=now + timedelta(seconds=self._budget.approval_timeout_s),
            requires_approval=economics.requires_approval,
            intervention_class=economics.intervention_class,
        )

    def _context_identity(self) -> str:
        descriptor = self._workload.descriptor()
        material = "|".join(
            [
                descriptor.workload_id,
                descriptor.model_ref,
                descriptor.backend_name,
                config_key(self._workload.configuration()),
                self._baseline.baseline_id if self._baseline is not None else "no-baseline",
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------ #
    # Cooldown / persistence                                              #
    # ------------------------------------------------------------------ #
    def _start_cooldown(self, key: str, reason: str, *, trigger: str) -> None:
        expires = self._clock() + self._policy.intervention_cooldown_s
        self._cooldown_until[key] = expires
        self._cooldowns.append(
            RuntimeCooldown(
                cooldown_id=f"rtcd-{uuid.uuid4().hex[:10]}",
                key=key,
                condition_key=key,
                reason=reason,
                trigger=trigger,
                expires_at=datetime.now(timezone.utc)
                + timedelta(seconds=self._policy.intervention_cooldown_s),
            )
        )

    def _cooling(self, key: str) -> bool:
        self._prune_cooldowns()
        return key in self._cooldown_until

    def _cooling_any(self) -> bool:
        self._prune_cooldowns()
        return bool(self._cooldown_until)

    def _prune_cooldowns(self) -> None:
        now = self._clock()
        for key in [k for k, until in self._cooldown_until.items() if until <= now]:
            del self._cooldown_until[key]

    def _persist(self, result: ExperimentResult) -> KnowledgeWriteStatus | None:
        if self._knowledge is None:
            return None
        try:
            persist = self._knowledge.persist_result(result)
            return persist.optimization.status
        except KnowledgeError as exc:  # pragma: no cover - store guarded
            self._errors.append(f"runtime knowledge persistence failed: {exc}")
            return None

    def _maybe_rebase(self) -> None:
        if not self._pending_rebase:
            return
        self._observations_since_rebase += 1
        if self._observations_since_rebase < max(self._policy.budget.baseline_samples, 2):
            return
        descriptor = self._workload.descriptor()
        baseline = self._buffer.rebase(
            workload_id=descriptor.workload_id,
            model_ref=descriptor.model_ref,
            backend_name=descriptor.backend_name,
            configuration=self._workload.configuration(),
            context_length=self._workload.configuration().context_length,
            source="post_intervention",
        )
        if baseline is not None:
            self._baseline = baseline
            self._baselines.append(baseline)
            self._baseline_observations = self._buffer.observations[
                -max(self._policy.budget.baseline_samples, 2) :
            ]
        self._pending_rebase = False

    # ------------------------------------------------------------------ #
    # Overhead / termination / run assembly                               #
    # ------------------------------------------------------------------ #
    def _harness_or_default(self) -> MeasurementHarness:
        if self._harness is not None:
            return self._harness
        from inferenceos.profiling.runner import BaselineProfiler

        adapter = getattr(self._workload, "adapter", None)
        # Reuse the workload's adapter only when it is idle. A resident adapter that
        # currently owns a running server cannot be re-loaded by the harness, so we use
        # a fresh measurement adapter in that case.
        if adapter is not None and not getattr(adapter, "is_loaded", False):
            self._harness = BaselineProfiler(settings=self._settings, adapter=adapter)
        else:
            self._harness = BaselineProfiler(settings=self._settings)
        return self._harness

    def _hardware_profile(self) -> HardwareProfile:
        if self._hardware is None:
            self._hardware = detect_hardware_profile(self._settings)
        return self._hardware

    def _collect_observation(self, *, condition: str) -> RuntimeObservation:
        started = self._clock()
        observation = self._sampler.sample(
            self._workload, sequence=self._observations_count, condition=condition
        )
        self._record_overhead("detection_s", self._clock() - started)
        self._buffer.append(observation)
        self._observations_count += 1
        self._budget = self._budget.model_copy(
            update={"observations_used": self._budget.observations_used + 1}
        )
        return observation

    def _record_overhead(self, key: str, seconds: float) -> None:
        self._overhead[key] = self._overhead.get(key, 0.0) + max(0.0, seconds)

    def _optimizer_overhead(self) -> float:
        return sum(
            self._overhead.get(key, 0.0)
            for key in ("detection_s", "diagnosis_s", "reasoning_s", "checkpoint_s", "validation_s", "notification_s")
        )

    def _interval(self) -> float:
        return self._budget.telemetry_interval_s

    def _set_state(self, state: RuntimeState, detail: str, data: dict[str, object] | None = None) -> None:
        self._state = state
        self._events.append(
            RuntimeStateEvent(
                state=state,
                timestamp=datetime.now(timezone.utc),
                observation=self._observations_count,
                detail=detail,
                data=data or {},
            )
        )

    def _terminate(self, reason: RuntimeTerminationReason, detail: str) -> RuntimeTermination:
        safety = reason is RuntimeTerminationReason.SAFETY_TERMINAL
        clean = reason in _TERMINAL and reason not in (
            RuntimeTerminationReason.SAFETY_TERMINAL,
            RuntimeTerminationReason.ERROR,
        )
        if reason not in (RuntimeTerminationReason.SAFETY_TERMINAL, RuntimeTerminationReason.ERROR):
            self._set_state(RuntimeState.TERMINAL_SAFE, detail)
        return RuntimeTermination(
            reason=reason,
            detail=detail,
            clean=clean,
            safety=safety,
            at_observation=self._observations_count,
        )

    def _finish(self, termination: RuntimeTermination, started_at: datetime) -> RuntimeMonitorResult:
        descriptor = self._workload.descriptor()
        overhead = self._build_overhead()
        resource = self._resource_impact()
        run_id = self._run_id_override or f"rtrun-{uuid.uuid4().hex[:12]}"
        session = self._build_session_provenance(run_id, descriptor, started_at)
        run = RuntimeMonitorRun(
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            workload=descriptor,
            policy=self._policy,
            budget=self._budget,
            baseline=self._baseline,
            baselines=list(self._baselines),
            states=list(self._events),
            window=self._last_window,
            opportunities=list(self._opportunities),
            recommendations=list(self._recommendations),
            notifications=list(self._notifications),
            approvals=list(self._approvals),
            checkpoints=list(self._checkpoints),
            interventions=list(self._interventions),
            cooldowns=list(self._cooldowns),
            knowledge_queries=list(self._knowledge_queries),
            external_runs=list(self._external_runs),
            target_recoveries=self._collect_recoveries(),
            net_benefit=self._build_net_benefit(),
            observations_collected=self._observations_count,
            diagnosis_events=self._diagnosis_events,
            deep_analysis_events=self._deep_analysis_events,
            final_state=self._state,
            termination=termination,
            overhead=overhead,
            resource_impact=resource,
            session=session,
            errors=list(self._errors),
            summary=self._summary(termination),
        )
        return RuntimeMonitorResult(
            run=run,
            termination=termination,
            final_state=self._state,
            observations_collected=self._observations_count,
            summary=run.summary,
        )

    def _build_session_provenance(
        self, run_id: str, descriptor: object, started_at: datetime
    ) -> RuntimeSessionProvenance:
        target_pid: int | None = None
        target_create_time: float | None = None
        identity = self._workload_target_identity()
        if identity is not None:
            target_pid = getattr(identity, "pid", None)
            target_create_time = getattr(identity, "create_time", None)
        if target_pid is None:
            getter = getattr(self._workload, "target_pid", None)
            if callable(getter):
                try:
                    target_pid = getter()
                except Exception:  # noqa: BLE001
                    target_pid = None
        distinct = target_pid is not None and target_pid != self._monitor_pid
        notes = list(self._session_notes) + [
            "monitor process identity recorded independently of the target",
        ]
        if target_pid is None:
            notes.append("no independent resident target identity was available")
        elif distinct:
            notes.append("monitor PID and resident target PID are distinct")
        else:
            notes.append("monitor and target share a PID (self-monitoring arrangement)")
        return RuntimeSessionProvenance(
            run_id=run_id,
            session_id=self._session_id,
            started_at=started_at,
            monitor_pid=self._monitor_pid,
            monitor_create_time=(
                self._monitor_identity.create_time if self._monitor_identity else None
            ),
            target_pid=target_pid,
            target_create_time=target_create_time,
            target_alive=self._target_identity_ok(),
            distinct_processes=distinct,
            model_ref=getattr(descriptor, "model_ref", None),
            model_path=getattr(descriptor, "model_path", None),
            backend_name=getattr(descriptor, "backend_name", None),
            hardware_hash=hashlib.sha256(
                str(self._hardware_profile().model_dump(mode="json")).encode("utf-8")
            ).hexdigest()[:32],
            notes=notes,
        )

    def _build_net_benefit(self) -> NetBenefitAssessment:
        """H4: consolidate existing overhead/benefit/distortion evidence.

        Baseline metrics are the initial baseline; measured metrics are the most
        recent (post-intervention) baseline. Overhead is the existing runtime
        ``optimizer_overhead_s`` (exclusive-phase sum) with the target-workload window
        ``monitoring_wall_s - optimizer_overhead_s`` as the reference. No new hot-path
        timing is introduced.
        """
        baseline = self._baselines[0] if self._baselines else self._baseline
        measured = self._baselines[-1] if self._baselines else self._baseline
        baseline_metrics = _metric_means(baseline)
        measured_metrics = _metric_means(measured)

        overhead = self._build_overhead()
        target_workload_s = max(
            0.0, overhead.monitoring_wall_s - overhead.optimizer_overhead_s
        )
        attribution = build_overhead_attribution(
            inclusive_overhead_s=overhead.optimizer_overhead_s,
            components={
                "detection_s": overhead.detection_s,
                "diagnosis_s": overhead.diagnosis_s,
                "reasoning_s": overhead.reasoning_s,
                "checkpoint_s": overhead.checkpoint_s,
                "validation_s": overhead.validation_s,
                "notification_s": overhead.notification_s,
            },
            baseline_target_workload_s=target_workload_s,
            decision_count=len(self._interventions),
            notes=[
                "pause/experiment/resume are target-interruption costs, excluded from "
                "optimizer overhead",
            ],
        )
        impact = self._resource_impact()
        distortion = distortion_from_stability(
            cv=impact.decode_tps_cv,
            within_tolerance=impact.throughput_within_tolerance,
            tolerance=max(self._policy.deviation_threshold, 0.25),
        )
        return assess_net_benefit(
            baseline_metrics=baseline_metrics,
            measured_metrics=measured_metrics,
            overhead=attribution,
            distortion=distortion,
            policy=AccountingPolicy.from_settings(self._settings),
        )

    def _build_overhead(self) -> RuntimeOverhead:
        total = sum(self._overhead.values())
        return RuntimeOverhead(
            detection_s=self._overhead.get("detection_s", 0.0),
            diagnosis_s=self._overhead.get("diagnosis_s", 0.0),
            reasoning_s=self._overhead.get("reasoning_s", 0.0),
            checkpoint_s=self._overhead.get("checkpoint_s", 0.0),
            pause_s=self._overhead.get("pause_s", 0.0),
            experiment_s=self._overhead.get("experiment_s", 0.0),
            validation_s=self._overhead.get("validation_s", 0.0),
            resume_s=self._overhead.get("resume_s", 0.0),
            notification_s=self._overhead.get("notification_s", 0.0),
            monitoring_wall_s=self._clock() - self._started_wall,
            optimizer_overhead_s=self._optimizer_overhead(),
            total_s=total,
            sample_count=self._observations_count,
            notes=[
                "monitoring overhead is measured separately from workload metrics",
                "experiment/pause/resume are intervention costs, not monitoring overhead",
            ],
        )

    def _resource_impact(self) -> ResourceImpact:
        baseline_observations = [
            o for o in self._buffer.observations if o.condition == "baseline"
        ]
        observations = baseline_observations or self._buffer.observations
        values = [o.metrics["decode_tokens_per_sec"] for o in observations if "decode_tokens_per_sec" in o.metrics]
        if not values:
            values = [
                o.metrics["decode_tokens_per_sec"]
                for o in self._buffer.observations
                if "decode_tokens_per_sec" in o.metrics
            ]
        mean = sum(values) / len(values) if values else None
        cv = None
        if mean and len(values) >= 2:
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            cv = (variance ** 0.5) / mean if mean else None
        within = cv is None or cv <= max(self._policy.deviation_threshold, 0.25)
        return ResourceImpact(
            sampling_overhead_s=self._overhead.get("detection_s", 0.0),
            decode_tps_mean=mean,
            decode_tps_cv=cv,
            throughput_within_tolerance=within,
            notes=["decode throughput stability measured across the observation window"],
        )

    def _summary(self, termination: RuntimeTermination) -> str:
        interventions = len(self._interventions)
        recoveries = self._collect_recoveries()
        recovered = sum(1 for record in recoveries if record.recovered)
        unrecoverable = sum(1 for record in recoveries if record.unrecoverable)
        return (
            f"Phase 10 runtime monitor observed {self._observations_count} sample(s), "
            f"{self._diagnosis_events} diagnosis event(s), {self._deep_analysis_events} deep "
            f"analysis event(s), {interventions} intervention(s); resident recoveries "
            f"attempted/recovered/unrecoverable="
            f"{sum(r.attempt_count for r in recoveries)}/{recovered}/{unrecoverable}; "
            f"terminated ({termination.reason.value}): {termination.detail}"
        )

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #
    def _reset(self) -> None:
        self._state = RuntimeState.INITIALIZING
        self._fatal = None
        self._events = []
        self._buffer = ObservationBuffer(
            capacity=self._policy.budget.observation_window_size,
            baseline_count=self._policy.budget.baseline_samples,
        )
        self._baseline = None
        self._baselines = []
        self._baseline_observations = []
        self._last_window = None
        self._observations_count = 0
        self._diagnosis_events = 0
        self._deep_analysis_events = 0
        self._opportunities = []
        self._recommendations = []
        self._notifications = []
        self._approvals = []
        self._checkpoints = []
        self._interventions = []
        self._cooldowns = []
        self._cooldown_until = {}
        self._alert_counts = {}
        self._suppressed = set()
        self._knowledge_queries = []
        self._external_runs = []
        self._external_consulted = set()
        self._recoveries = []
        self._last_diagnosis_at = -1e12
        self._overhead = {}
        self._errors = []
        self._budget = self._policy.budget
        self._pending_rebase = False
        self._observations_since_rebase = 0
        self._started_wall = self._clock()
