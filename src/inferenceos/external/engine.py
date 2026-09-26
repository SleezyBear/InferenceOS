"""Phase 11R external recommendation engine (recommendation-only).

Orchestrates the staged flow:

    OBSERVE -> ATTRIBUTE -> ASSESS IMPACT -> GENERATE RECOMMENDATION
    -> PRESENT -> USER DECISION -> RECORD

There is deliberately **no execution stage**. InferenceOS observes external processes
and produces evidence-backed, provenance-bound recommendations; the user decides.
Accepting a recommendation records user intent only and never mutates the external
process.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

from inferenceos.contracts.enums import (
    ApprovalOutcome,
    ExternalRecommendationClass,
    ExternalRecommendationStatus,
    ExternalResource,
    ExternalRunTerminationReason,
    InterventionClass,
    KnowledgeWriteStatus,
)
from inferenceos.contracts.external import (
    AttributionAssessment,
    ExternalActionSafety,
    ExternalApproval,
    ExternalBudget,
    ExternalCooldown,
    ExternalKnowledgeRecord,
    ExternalObservationReport,
    ExternalOverhead,
    ExternalPolicy,
    ExternalProcessIdentity,
    ExternalRecommendation,
    ExternalRecommendationRun,
    ExternalTermination,
    ExternalUserDecision,
)
from inferenceos.contracts.runtime import RuntimeRecommendation
from inferenceos.external.actions import evaluate_external_action_safety
from inferenceos.external.attribution import assess_external_attribution
from inferenceos.external.knowledge import ExternalKnowledgeError, ExternalKnowledgeStore
from inferenceos.external.observer import ProcessObserver
from inferenceos.external.recommendation import revalidate_recommendation
from inferenceos.external.recommender import build_recommendations
from inferenceos.optimization.registry import ActionRegistry
from inferenceos.runtime.approval import ApprovalProvider

AfterMetricsProvider = Callable[[], dict[str, float]]


class ExternalRecommendationEngine:
    """Deterministic, conservative external recommendation orchestrator."""

    def __init__(
        self,
        *,
        observer: ProcessObserver,
        registry: ActionRegistry,
        policy: ExternalPolicy,
        approval_provider: ApprovalProvider,
        knowledge_store: ExternalKnowledgeStore | None = None,
        own_pid: int | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._observer = observer
        self._registry = registry
        self._policy = policy
        self._approval = approval_provider
        self._knowledge = knowledge_store
        # H6: the optimizer's own process must never be a selectable external
        # candidate. Default to this process's PID when the caller does not supply one
        # so production constructors cannot silently omit self-exclusion.
        self._own_pid = own_pid if own_pid is not None else os.getpid()
        self._clock = clock or time.perf_counter
        self._sleep = sleep or time.sleep

        self._target_pid: int | None = None
        self._target_identity: ExternalProcessIdentity | None = None
        self._run_id = ""
        self._session_id = ""
        self._diagnosis_id: str | None = None
        self._diagnosed_bottleneck: str | None = None
        self._diagnosed_resource: ExternalResource | None = None
        self._reports: list[ExternalObservationReport] = []
        self._assessments: list[AttributionAssessment] = []
        self._recommendations: list[ExternalRecommendation] = []
        self._user_decisions: list[ExternalUserDecision] = []
        self._safety_checks: list[ExternalActionSafety] = []
        self._approvals: list[ExternalApproval] = []
        self._knowledge_queries: list[ExternalKnowledgeRecord] = []
        self._knowledge_writes: list[ExternalKnowledgeRecord] = []
        self._cooldowns: list[ExternalCooldown] = []
        self._cooldown_until: dict[str, float] = {}
        self._alert_counts: dict[str, int] = {}
        self._suppressed: set[str] = set()
        self._overhead: dict[str, float] = {}
        self._errors: list[str] = []
        self._budget: ExternalBudget = self._policy.budget

        self._reset()

    # ------------------------------------------------------------------ #
    # Observation / evaluation (reusable)                                  #
    # ------------------------------------------------------------------ #
    def observe_once(self) -> ExternalObservationReport:
        started = self._clock()
        report = self._observer.observe(
            target_pid=self._target_pid,
            own_pid=self._own_pid,
            max_processes=self._policy.budget.max_processes,
            scope=self._policy.scope,
        )
        self._record_overhead("observation_s", self._clock() - started)
        return report

    def evaluate_report(
        self,
        report: ExternalObservationReport,
        *,
        target_degraded: bool,
        target_metrics: dict[str, float],
        context_identity: str,
        diagnosed_resource: ExternalResource | None = None,
        diagnosed_bottleneck: str | None = None,
        now: datetime | None = None,
    ) -> tuple[list[ExternalKnowledgeRecord], list[ExternalRecommendation]]:
        started = self._clock()
        knowledge_records: list[ExternalKnowledgeRecord] = []
        for observation in report.processes:
            identity = observation.identity
            if identity.is_target or identity.pid == self._own_pid:
                continue
            records = self._query_knowledge(identity.identity_key)
            knowledge_records.extend(records)
        assessments = assess_external_attribution(
            report,
            self._policy,
            target_degraded=target_degraded,
            comparisons=[],
            knowledge=knowledge_records,
        )
        self._assessments.extend(assessments)
        recommendations = build_recommendations(
            assessments,
            registry=self._registry,
            policy=self._policy,
            context_identity=context_identity,
            diagnosed_resource=(
                diagnosed_resource if diagnosed_resource is not None else self._diagnosed_resource
            ),
            diagnosed_bottleneck=(
                diagnosed_bottleneck
                if diagnosed_bottleneck is not None
                else self._diagnosed_bottleneck
            ),
            target_identity=self._target_identity,
            target_degraded=target_degraded,
            run_id=self._run_id,
            session_id=self._session_id,
            diagnosis_id=self._diagnosis_id,
            observation_report_id=report.report_id,
            observed_at=report.captured_at,
            now=now,
        )
        self._record_overhead("attribution_s", self._clock() - started)
        return knowledge_records, recommendations

    def revalidate(
        self,
        recommendation: ExternalRecommendation,
        *,
        current_run_id: str | None = None,
        current_session_id: str | None = None,
        current_context_identity: str | None = None,
        revalidate_process: bool = True,
        now: datetime | None = None,
    ) -> ExternalRecommendation:
        """Revalidate a previously generated recommendation against the current context."""

        return revalidate_recommendation(
            recommendation,
            current_run_id=current_run_id if current_run_id is not None else self._run_id,
            current_session_id=(
                current_session_id if current_session_id is not None else self._session_id
            ),
            current_context_identity=current_context_identity,
            revalidate_process=revalidate_process,
            now=now,
        )

    # ------------------------------------------------------------------ #
    # Main bounded run                                                     #
    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        target_pid: int | None = None,
        target_identity: ExternalProcessIdentity | None = None,
        target_degraded: bool = False,
        target_metrics: dict[str, float] | None = None,
        context_identity: str = "",
        diagnosed_bottleneck: str | None = None,
        diagnosed_resource: ExternalResource | None = None,
        diagnosis_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        duration_s: float | None = None,
        max_cycles: int | None = None,
        after_metrics_provider: AfterMetricsProvider | None = None,
    ) -> ExternalRecommendationRun:
        _ = after_metrics_provider
        self._reset()
        self._target_pid = target_pid
        self._target_identity = target_identity
        self._run_id = run_id or f"extrun-{uuid.uuid4().hex[:12]}"
        self._session_id = session_id or self._run_id
        self._diagnosis_id = diagnosis_id
        self._diagnosed_bottleneck = diagnosed_bottleneck
        self._diagnosed_resource = diagnosed_resource
        self._budget = self._policy.budget
        if duration_s is not None:
            self._budget = self._budget.model_copy(update={"max_duration_s": duration_s})
        started_at = datetime.now(timezone.utc)
        started = self._clock()
        target_metrics = dict(target_metrics or {})
        termination: ExternalTermination | None = None
        cycles = 0
        any_recommendation = False

        while termination is None:
            elapsed = self._clock() - started
            reason = self._budget.exhaustion(elapsed_s=elapsed)
            if reason is not None:
                termination = ExternalTermination(
                    reason=reason, detail=f"budget reached: {reason.value}"
                )
                break

            report = self.observe_once()
            self._reports.append(report)
            self._budget = self._budget.model_copy(
                update={"observations_used": self._budget.observations_used + 1}
            )
            knowledge_records, recommendations = self.evaluate_report(
                report,
                target_degraded=target_degraded,
                target_metrics=target_metrics,
                context_identity=context_identity,
            )
            self._knowledge_queries.extend(knowledge_records)
            if recommendations:
                any_recommendation = True

            decision_termination = self._handle_recommendations(
                recommendations,
                target_degraded=target_degraded,
                target_metrics=target_metrics,
                context_identity=context_identity,
            )
            if decision_termination is not None:
                termination = decision_termination
                break

            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                if self._diagnosed_resource is None:
                    termination = ExternalTermination(
                        reason=ExternalRunTerminationReason.NO_MEANINGFUL_BOTTLENECK,
                        detail=(
                            "no meaningful diagnosed bottleneck; external processes are "
                            "not ranked or recommended on raw usage"
                        ),
                    )
                else:
                    termination = ExternalTermination(
                        reason=ExternalRunTerminationReason.NO_RECOMMENDATION,
                        detail="bounded observation window completed with no recommendation",
                    )
                break
            self._sleep(self._policy.observation_interval_s)

        finished_at = datetime.now(timezone.utc)
        if termination is None:  # pragma: no cover - defensive
            termination = ExternalTermination(
                reason=ExternalRunTerminationReason.NO_RECOMMENDATION,
                detail="run ended without an explicit termination",
            )
        return ExternalRecommendationRun(
            run_id=self._run_id,
            created_at=started_at,
            started_at=started_at,
            finished_at=finished_at,
            scope=self._policy.scope,
            policy=self._policy,
            budget=self._budget,
            target_pid=self._target_pid,
            target_identity=self._target_identity,
            session_id=self._session_id,
            diagnosis_id=self._diagnosis_id,
            diagnosed_bottleneck=self._diagnosed_bottleneck,
            observation_reports=self._reports,
            assessments=self._assessments,
            recommendations=self._recommendations,
            user_decisions=self._user_decisions,
            safety_checks=self._safety_checks,
            approvals=self._approvals,
            knowledge_queries=self._knowledge_queries,
            knowledge_writes=self._knowledge_writes,
            cooldowns=self._cooldowns,
            termination=termination,
            overhead=self._build_overhead(),
            errors=self._errors,
            summary=self._summary(termination, any_recommendation),
        )

    # ------------------------------------------------------------------ #
    # Recommendation handling: terminate at the user decision               #
    # ------------------------------------------------------------------ #
    def _handle_recommendations(
        self,
        recommendations: list[ExternalRecommendation],
        *,
        target_degraded: bool,
        target_metrics: dict[str, float],
        context_identity: str,
    ) -> ExternalTermination | None:
        _ = (target_degraded, target_metrics)
        for recommendation in recommendations:
            identity = recommendation.identity
            identity_key = identity.identity_key if identity is not None else "none"
            started = self._clock()
            safety = evaluate_external_action_safety(
                recommendation.proposed_action_id,
                registry=self._registry,
                policy=self._policy,
                identity=identity,
            )
            safety = safety.model_copy(
                update={"recommendation_id": recommendation.recommendation_id}
            )
            self._safety_checks.append(safety)
            self._record_overhead("recommendation_s", self._clock() - started)

            key = identity_key
            presented = recommendation.model_copy(
                update={"status": ExternalRecommendationStatus.PRESENTED}
            )
            self._recommendations.append(presented)
            self._budget = self._budget.model_copy(
                update={"recommendations_used": self._budget.recommendations_used + 1}
            )

            if key in self._suppressed or self._cooling(key):
                continue

            if (
                recommendation.classification is ExternalRecommendationClass.INFORMATIONAL
                or recommendation.proposed_action_id is None
            ):
                self._persist_decision(
                    presented,
                    identity_key,
                    outcome="informational",
                    status=ExternalRecommendationStatus.PRESENTED,
                    context_identity=context_identity,
                )
                self._register_alert(key)
                continue

            # A user decision is requested. The user decides — InferenceOS never
            # executes the proposed action, whatever the decision.
            approval = self._request_approval(presented)
            self._approvals.append(approval)
            status = _status_for_outcome(approval.outcome)
            decision = ExternalUserDecision(
                decision_id=f"extdec-{uuid.uuid4().hex[:10]}",
                recommendation_id=presented.recommendation_id,
                run_id=self._run_id,
                decision=status,
                decided_by=approval.decided_by,
                context_identity=context_identity,
                process_identity_key=identity_key,
                autonomous_execution=False,
                detail=(
                    f"user decision {status.value}; InferenceOS performed no external-process "
                    "mutation"
                ),
            )
            self._user_decisions.append(decision)
            self._replace_recommendation(
                presented.recommendation_id,
                status=status,
                stale_reason=(
                    approval.detail
                    if status is ExternalRecommendationStatus.STALE
                    else None
                ),
            )
            decision_rec = presented.model_copy(
                update={
                    "status": status,
                    "stale_reason": (
                        approval.detail
                        if status is ExternalRecommendationStatus.STALE
                        else None
                    ),
                }
            )
            self._persist_decision(
                decision_rec,
                identity_key,
                outcome=f"user_decision_{status.value}",
                status=status,
                context_identity=context_identity,
            )

            if status is ExternalRecommendationStatus.ACCEPTED:
                self._start_cooldown(key, "recommendation accepted (no execution)", trigger="decision")
                return ExternalTermination(
                    reason=ExternalRunTerminationReason.RECOMMENDATION_ACCEPTED,
                    detail=(
                        "user accepted the recommendation; recorded as intent only, no "
                        "external-process mutation performed"
                    ),
                )
            self._start_cooldown(key, f"user decision {status.value}", trigger="decision")
            return ExternalTermination(
                reason=ExternalRunTerminationReason.APPROVAL_DECLINED,
                detail=(
                    f"user decision {status.value}; no external action taken and none is "
                    "authorized"
                ),
                safety=status is ExternalRecommendationStatus.STALE,
            )
        return None

    def _replace_recommendation(
        self,
        recommendation_id: str,
        *,
        status: ExternalRecommendationStatus,
        stale_reason: str | None,
    ) -> None:
        for index, existing in enumerate(self._recommendations):
            if existing.recommendation_id == recommendation_id:
                self._recommendations[index] = existing.model_copy(
                    update={"status": status, "stale_reason": stale_reason}
                )
                return

    # ------------------------------------------------------------------ #
    # Approval                                                             #
    # ------------------------------------------------------------------ #
    def _request_approval(self, recommendation: ExternalRecommendation) -> ExternalApproval:
        started = self._clock()
        runtime_recommendation = RuntimeRecommendation(
            proposal_id=recommendation.recommendation_id,
            opportunity_id=recommendation.attribution_id or recommendation.recommendation_id,
            condition=recommendation.neutral_description,
            why_it_matters=recommendation.why_it_matters,
            analysis_summary=(
                "Recommend a human action for the identified external process. Phase 11R "
                "does not execute it; accepting records user intent only."
            ),
            proposed_action_id=recommendation.proposed_action_id,
            proposed_parameters=recommendation.proposed_parameters,
            expected_benefit=recommendation.expected_benefit,
            expected_cost_s=recommendation.expected_cost_s,
            expected_disruption_s=recommendation.expected_disruption_s,
            pause_required=False,
            checkpoint_planned=False,
            reversible=False,
            expected_learning="whether the external process matters to the target workload",
            context_identity=recommendation.context_identity,
            expires_at=recommendation.expires_at,
            requires_approval=True,
            intervention_class=InterventionClass.EXTERNAL_SYSTEM_WIDE,
        )
        record = self._approval.request(
            runtime_recommendation, timeout_s=self._policy.approval_timeout_s
        )
        self._record_overhead("approval_s", self._clock() - started)
        outcome = record.outcome
        detail = record.detail
        if outcome is ApprovalOutcome.APPROVED and (
            record.proposal_id != recommendation.recommendation_id
            or record.context_identity != recommendation.context_identity
        ):
            outcome = ApprovalOutcome.STALE
            detail = "approval does not match the current proposal/context"
        return ExternalApproval(
            approval_id=record.approval_id,
            recommendation_id=recommendation.recommendation_id,
            action_id=recommendation.proposed_action_id,
            outcome=outcome,
            process_identity_key=(
                recommendation.identity.identity_key
                if recommendation.identity is not None
                else None
            ),
            context_identity=recommendation.context_identity,
            requested_at=record.requested_at,
            decided_at=record.decided_at,
            expires_at=recommendation.expires_at,
            decided_by=record.decided_by,
            detail=detail,
        )

    # ------------------------------------------------------------------ #
    # Knowledge / cooldown / overhead                                      #
    # ------------------------------------------------------------------ #
    def _query_knowledge(self, identity_key: str) -> list[ExternalKnowledgeRecord]:
        if self._knowledge is None:
            return []
        started = self._clock()
        try:
            records = self._knowledge.find_for_identity(identity_key)
        except ExternalKnowledgeError as exc:
            self._errors.append(f"external knowledge query failed: {exc}")
            records = []
        self._record_overhead("knowledge_query_s", self._clock() - started)
        return records

    def _persist_decision(
        self,
        recommendation: ExternalRecommendation,
        identity_key: str,
        *,
        outcome: str,
        status: ExternalRecommendationStatus,
        context_identity: str,
    ) -> None:
        if self._knowledge is None:
            return
        record = ExternalKnowledgeRecord(
            record_id=f"extrec-{uuid.uuid4().hex[:12]}",
            recommendation_id=recommendation.recommendation_id,
            identity_key=identity_key,
            executable_identity=(
                recommendation.identity.executable_identity
                if recommendation.identity is not None
                else None
            ),
            resource=recommendation.resource,
            classification=recommendation.classification,
            action_id=recommendation.proposed_action_id,
            outcome=outcome,
            approved=status is ExternalRecommendationStatus.ACCEPTED,
            executed=False,
            verified=False,
            restored=False,
            context_identity=context_identity,
            evidence_summary=list(recommendation.evidence_summary),
            detail=recommendation.neutral_description,
        )
        status_write = self._knowledge.record(record)
        if status_write in (KnowledgeWriteStatus.PERSISTED, KnowledgeWriteStatus.DUPLICATE):
            self._knowledge_writes.append(record)

    def _register_alert(self, key: str) -> None:
        count = self._alert_counts.get(key, 0) + 1
        self._alert_counts[key] = count
        if count >= self._policy.max_alerts_per_identity:
            self._suppressed.add(key)

    def _start_cooldown(self, key: str, reason: str, *, trigger: str) -> None:
        self._cooldown_until[key] = self._clock() + self._policy.cooldown_s
        self._cooldowns.append(
            ExternalCooldown(
                cooldown_id=f"extcd-{uuid.uuid4().hex[:10]}",
                key=key,
                identity_key=key,
                reason=reason,
                trigger=trigger,
                expires_at=datetime.now(timezone.utc)
                + timedelta(seconds=self._policy.cooldown_s),
            )
        )

    def _cooling(self, key: str) -> bool:
        now = self._clock()
        for candidate in [k for k, until in self._cooldown_until.items() if until <= now]:
            del self._cooldown_until[candidate]
        return key in self._cooldown_until

    def _record_overhead(self, key: str, seconds: float) -> None:
        self._overhead[key] = self._overhead.get(key, 0.0) + max(0.0, seconds)

    def _build_overhead(self) -> ExternalOverhead:
        total = sum(self._overhead.values())
        return ExternalOverhead(
            observation_s=self._overhead.get("observation_s", 0.0),
            attribution_s=self._overhead.get("attribution_s", 0.0),
            knowledge_query_s=self._overhead.get("knowledge_query_s", 0.0),
            recommendation_s=self._overhead.get("recommendation_s", 0.0),
            approval_s=self._overhead.get("approval_s", 0.0),
            total_s=total,
            notes=[
                "external observation is read-only and bounded",
                "Phase 11R has no action or verification stage by design",
            ],
        )

    def _summary(self, termination: ExternalTermination, any_recommendation: bool) -> str:
        return (
            f"Phase 11R external recommendations: {len(self._reports)} observation(s), "
            f"{len(self._assessments)} attribution assessment(s), "
            f"{len(self._recommendations)} recommendation(s), "
            f"{len(self._user_decisions)} user decision(s), 0 external mutations; "
            f"terminated ({termination.reason.value}): {termination.detail}"
        )

    def _reset(self) -> None:
        self._target_pid = None
        self._target_identity = None
        self._run_id = ""
        self._session_id = ""
        self._diagnosis_id = None
        self._diagnosed_bottleneck = None
        self._diagnosed_resource = None
        self._reports = []
        self._assessments = []
        self._recommendations = []
        self._user_decisions = []
        self._safety_checks = []
        self._approvals = []
        self._knowledge_queries = []
        self._knowledge_writes = []
        self._cooldowns = []
        self._cooldown_until = {}
        self._alert_counts = {}
        self._suppressed = set()
        self._overhead = {}
        self._errors = []
        self._budget = self._policy.budget


def _status_for_outcome(outcome: ApprovalOutcome) -> ExternalRecommendationStatus:
    if outcome is ApprovalOutcome.APPROVED:
        return ExternalRecommendationStatus.ACCEPTED
    if outcome is ApprovalOutcome.REJECTED:
        return ExternalRecommendationStatus.REJECTED
    if outcome is ApprovalOutcome.STALE:
        return ExternalRecommendationStatus.STALE
    if outcome is ApprovalOutcome.TIMEOUT:
        return ExternalRecommendationStatus.EXPIRED
    return ExternalRecommendationStatus.PRESENTED
