"""Phase 11 external recommendation engine.

Orchestrates the staged flow:

    OBSERVE -> ATTRIBUTE -> ASSESS IMPACT -> GENERATE RECOMMENDATION -> SAFETY CHECK
    -> PRESENT/APPROVE -> (OPTIONAL SUPPORTED ACTION) -> VERIFY -> RECORD

The default is observe -> recommend -> ask user. No external action is ever executed
without explicit contextual approval and immediate process-identity revalidation.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

from inferenceos.contracts.enums import (
    ApprovalOutcome,
    ExternalActionSafetyClass,
    ExternalRecommendationClass,
    ExternalRunTerminationReason,
    KnowledgeWriteStatus,
)
from inferenceos.contracts.external import (
    AttributionAssessment,
    ExternalActionCheckpoint,
    ExternalActionResult,
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
    ExternalVerification,
)
from inferenceos.contracts.runtime import RuntimeRecommendation
from inferenceos.external.actions import (
    ExternalActionExecutor,
    evaluate_external_action_safety,
)
from inferenceos.external.attribution import assess_external_attribution
from inferenceos.external.checkpoint import capture_external_checkpoint
from inferenceos.external.knowledge import ExternalKnowledgeError, ExternalKnowledgeStore
from inferenceos.external.observer import ProcessObserver
from inferenceos.external.recommender import build_recommendations
from inferenceos.external.verification import verify_external_action
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
        executor: ExternalActionExecutor | None = None,
        own_pid: int | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._observer = observer
        self._registry = registry
        self._policy = policy
        self._approval = approval_provider
        self._knowledge = knowledge_store
        self._executor = executor or ExternalActionExecutor()
        self._own_pid = own_pid
        self._clock = clock or time.perf_counter
        self._sleep = sleep or time.sleep

        self._target_pid: int | None = None
        self._target_identity: ExternalProcessIdentity | None = None
        self._reports: list[ExternalObservationReport] = []
        self._assessments: list[AttributionAssessment] = []
        self._recommendations: list[ExternalRecommendation] = []
        self._safety_checks: list[ExternalActionSafety] = []
        self._approvals: list[ExternalApproval] = []
        self._checkpoints: list[ExternalActionCheckpoint] = []
        self._action_results: list[ExternalActionResult] = []
        self._verifications: list[ExternalVerification] = []
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
        )
        self._record_overhead("attribution_s", self._clock() - started)
        return knowledge_records, recommendations

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
        duration_s: float | None = None,
        max_cycles: int | None = None,
        after_metrics_provider: AfterMetricsProvider | None = None,
    ) -> ExternalRecommendationRun:
        self._reset()
        self._target_pid = target_pid
        self._target_identity = target_identity
        self._budget = self._policy.budget
        if duration_s is not None:
            self._budget = self._budget.model_copy(update={"max_duration_s": duration_s})
        started_at = datetime.now(timezone.utc)
        started = self._clock()
        target_metrics = dict(target_metrics or {})
        termination: ExternalTermination | None = None
        cycles = 0

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

            action_termination = self._handle_recommendations(
                recommendations,
                target_degraded=target_degraded,
                target_metrics=target_metrics,
                context_identity=context_identity,
                after_metrics_provider=after_metrics_provider,
            )
            if action_termination is not None:
                termination = action_termination
                break

            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                termination = ExternalTermination(
                    reason=ExternalRunTerminationReason.NO_RECOMMENDATION,
                    detail="bounded observation window completed with no action",
                )
                break
            self._sleep(self._policy.observation_interval_s)

        finished_at = datetime.now(timezone.utc)
        run = ExternalRecommendationRun(
            run_id=f"extrun-{uuid.uuid4().hex[:12]}",
            started_at=started_at,
            finished_at=finished_at,
            scope=self._policy.scope,
            policy=self._policy,
            budget=self._budget,
            target_pid=self._target_pid,
            target_identity=self._target_identity,
            observation_reports=self._reports,
            assessments=self._assessments,
            recommendations=self._recommendations,
            safety_checks=self._safety_checks,
            approvals=self._approvals,
            checkpoints=self._checkpoints,
            action_results=self._action_results,
            verifications=self._verifications,
            knowledge_queries=self._knowledge_queries,
            knowledge_writes=self._knowledge_writes,
            cooldowns=self._cooldowns,
            termination=termination,
            overhead=self._build_overhead(),
            errors=self._errors,
            summary=self._summary(termination),
        )
        return run

    # ------------------------------------------------------------------ #
    # Recommendation handling                                             #
    # ------------------------------------------------------------------ #
    def _handle_recommendations(
        self,
        recommendations: list[ExternalRecommendation],
        *,
        target_degraded: bool,
        target_metrics: dict[str, float],
        context_identity: str,
        after_metrics_provider: AfterMetricsProvider | None,
    ) -> ExternalTermination | None:
        for recommendation in recommendations:
            self._recommendations.append(recommendation)
            self._budget = self._budget.model_copy(
                update={"recommendations_used": self._budget.recommendations_used + 1}
            )
            identity = recommendation.identity
            identity_key = identity.identity_key if identity is not None else "none"
            started = self._clock()
            safety = evaluate_external_action_safety(
                recommendation.proposed_action_id,
                registry=self._registry,
                policy=self._policy,
                identity=identity,
            )
            safety = safety.model_copy(update={"recommendation_id": recommendation.recommendation_id})
            self._safety_checks.append(safety)
            self._record_overhead("recommendation_s", self._clock() - started)

            key = identity_key
            if key in self._suppressed or self._cooling(key):
                continue

            if not recommendation.supported or not safety.allowed:
                self._persist_decision(
                    recommendation,
                    identity_key,
                    outcome=(
                        "informational"
                        if recommendation.classification is ExternalRecommendationClass.INFORMATIONAL
                        else "unsupported_or_blocked"
                    ),
                    approved=False,
                    executed=False,
                    verified=False,
                    restored=False,
                    context_identity=context_identity,
                )
                self._register_alert(key)
                continue

            # Supported, approval-gated action.
            approval = self._request_approval(recommendation)
            self._approvals.append(approval)
            if approval.outcome is not ApprovalOutcome.APPROVED:
                self._persist_decision(
                    recommendation,
                    identity_key,
                    outcome=f"approval_{approval.outcome.value}",
                    approved=False,
                    executed=False,
                    verified=False,
                    restored=False,
                    context_identity=context_identity,
                )
                self._start_cooldown(key, f"approval {approval.outcome.value}", trigger="approval")
                return ExternalTermination(
                    reason=ExternalRunTerminationReason.APPROVAL_DECLINED,
                    detail=f"approval outcome {approval.outcome.value}; no external action taken",
                )

            return self._execute_supported(
                recommendation,
                safety=safety,
                approval=approval,
                identity_key=identity_key,
                target_metrics=target_metrics,
                context_identity=context_identity,
                after_metrics_provider=after_metrics_provider,
            )
        return None

    def _execute_supported(
        self,
        recommendation: ExternalRecommendation,
        *,
        safety: ExternalActionSafety,
        approval: ExternalApproval,
        identity_key: str,
        target_metrics: dict[str, float],
        context_identity: str,
        after_metrics_provider: AfterMetricsProvider | None,
    ) -> ExternalTermination:
        identity = recommendation.identity
        action_id = recommendation.proposed_action_id
        assert identity is not None and action_id is not None  # guaranteed by supported

        started = self._clock()
        checkpoint = capture_external_checkpoint(
            identity,
            action_id,
            target_workload_state=target_metrics,
            approval_id=approval.approval_id,
        )
        self._checkpoints.append(checkpoint)
        self._record_overhead("checkpoint_s", self._clock() - started)
        if not checkpoint.verified or not checkpoint.restore_supported:
            self._persist_decision(
                recommendation,
                identity_key,
                outcome="checkpoint_failed",
                approved=True,
                executed=False,
                verified=False,
                restored=False,
                context_identity=context_identity,
            )
            self._start_cooldown(identity_key, "checkpoint could not be established", trigger="checkpoint")
            return ExternalTermination(
                reason=ExternalRunTerminationReason.SAFETY_TERMINAL,
                detail="external action checkpoint could not be verified; action not executed",
                clean=False,
                safety=True,
            )

        action_started = self._clock()
        result = self._executor.execute(
            action_id=action_id,
            identity=identity,
            parameters=recommendation.proposed_parameters,
            checkpoint=checkpoint,
        )
        result = result.model_copy(
            update={
                "recommendation_id": recommendation.recommendation_id,
                "approval_id": approval.approval_id,
            }
        )
        self._action_results.append(result)
        self._record_overhead("action_s", self._clock() - action_started)
        self._budget = self._budget.model_copy(
            update={
                "actions_used": self._budget.actions_used + 1,
                "cumulative_action_s": (
                    self._budget.cumulative_action_s + result.duration_s
                ),
            }
        )

        after_metrics = after_metrics_provider() if after_metrics_provider is not None else None
        verification = verify_external_action(
            action_id=action_id,
            identity=identity,
            action_result=result,
            checkpoint=checkpoint,
            executor=self._executor,
            restore_after=self._policy.restore_after_action,
            target_healthy=True,
            before_metrics=target_metrics or None,
            after_metrics=after_metrics,
        )
        self._verifications.append(verification)
        self._persist_decision(
            recommendation,
            identity_key,
            outcome="verified" if verification.verified else "verification_failed",
            approved=True,
            executed=result.executed,
            verified=verification.verified,
            restored=verification.restoration_verified,
            context_identity=context_identity,
        )
        self._start_cooldown(identity_key, "external action performed", trigger="action")
        if not verification.verified:
            return ExternalTermination(
                reason=ExternalRunTerminationReason.VERIFICATION_FAILURE,
                detail=verification.failure_reason or "external action verification failed",
                clean=False,
                safety=True,
            )
        return ExternalTermination(
            reason=ExternalRunTerminationReason.ACTION_COMPLETED,
            detail="approved external action executed and verified",
        )

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
                "Apply the registered external action to the identified process, verify "
                "the change, measure the target effect, and restore the prior state."
            ),
            proposed_action_id=recommendation.proposed_action_id,
            proposed_parameters=recommendation.proposed_parameters,
            expected_benefit=recommendation.expected_benefit,
            expected_cost_s=recommendation.expected_cost_s,
            expected_disruption_s=recommendation.expected_disruption_s,
            pause_required=False,
            checkpoint_planned=True,
            reversible=recommendation.reversible,
            expected_learning="whether the external process materially affects the target",
            context_identity=recommendation.context_identity,
            expires_at=recommendation.expires_at,
            requires_approval=True,
            intervention_class=_intervention_class(recommendation),
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
        approved: bool,
        executed: bool,
        verified: bool,
        restored: bool,
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
            approved=approved,
            executed=executed,
            verified=verified,
            restored=restored,
            context_identity=context_identity,
            evidence_summary=list(recommendation.evidence_summary),
            detail=recommendation.neutral_description,
        )
        status = self._knowledge.record(record)
        if status in (KnowledgeWriteStatus.PERSISTED, KnowledgeWriteStatus.DUPLICATE):
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
            checkpoint_s=self._overhead.get("checkpoint_s", 0.0),
            action_s=self._overhead.get("action_s", 0.0),
            verification_s=self._overhead.get("verification_s", 0.0),
            total_s=total,
            notes=["external observation is read-only; action time is measured separately"],
        )

    def _summary(self, termination: ExternalTermination) -> str:
        return (
            f"Phase 11 external run: {len(self._reports)} observation(s), "
            f"{len(self._assessments)} attribution assessment(s), "
            f"{len(self._recommendations)} recommendation(s), "
            f"{len(self._action_results)} action(s); terminated "
            f"({termination.reason.value}): {termination.detail}"
        )

    def _reset(self) -> None:
        self._target_pid = None
        self._target_identity = None
        self._reports = []
        self._assessments = []
        self._recommendations = []
        self._safety_checks = []
        self._approvals = []
        self._checkpoints = []
        self._action_results = []
        self._verifications = []
        self._knowledge_queries = []
        self._knowledge_writes = []
        self._cooldowns = []
        self._cooldown_until = {}
        self._alert_counts = {}
        self._suppressed = set()
        self._overhead = {}
        self._errors = []
        self._budget = self._policy.budget


def _intervention_class(recommendation: ExternalRecommendation):  # type: ignore[no-untyped-def]
    from inferenceos.contracts.enums import InterventionClass

    if recommendation.safety_class is ExternalActionSafetyClass.SAFE_REVERSIBLE:
        return InterventionClass.DISRUPTIVE_ADAPTATION
    return InterventionClass.EXTERNAL_SYSTEM_WIDE
