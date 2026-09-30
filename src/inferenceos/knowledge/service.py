"""High-level knowledge service: Phase 7 ``ExperimentResult`` → durable knowledge.

The service is the single translation point from the authoritative Phase 7 experiment
contract into persistent records. It never re-implements experiment logic, never
executes anything, and treats stored knowledge as conditional evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inferenceos.contracts.enums import (
    Decision,
    ExperimentDecisionKind,
    KnowledgeEventType,
    KnowledgeWriteStatus,
    OptimizationOutcome,
    RestorationStatus,
)
from inferenceos.contracts.experiment import ExperimentComparison, ExperimentResult
from inferenceos.contracts.knowledge import (
    ApplicabilityContext,
    ApplicabilityResult,
    KnowledgeHealth,
    KnowledgeQuery,
    KnowledgeWriteResult,
    OptimizationRecord,
    RollbackRecord,
    StorageStats,
)
from inferenceos.knowledge.applicability import evaluate_applicability
from inferenceos.knowledge.fingerprint import (
    build_context_from_request,
    conditions_from_metrics,
)
from inferenceos.knowledge.policy import KnowledgePolicy
from inferenceos.knowledge.rollback import build_rollback_record
from inferenceos.knowledge.store import (
    KnowledgeUnavailableError,
    SQLiteKnowledgeStore,
)
from inferenceos.optimization.registry import ActionRegistry

_CLASSIFICATION_RANK = {
    "directly_applicable": 0,
    "conditionally_applicable": 1,
    "requires_revalidation": 2,
    "insufficient_context": 3,
    "not_applicable": 4,
}


def outcome_from_result(result: ExperimentResult) -> OptimizationOutcome:
    """Map a Phase 7 terminal status to an explicit (non-Boolean) outcome."""
    kind = result.status
    if kind is ExperimentDecisionKind.IMPROVED:
        return OptimizationOutcome.IMPROVED
    if kind is ExperimentDecisionKind.NO_MEANINGFUL_IMPROVEMENT:
        return OptimizationOutcome.NO_MEANINGFUL_IMPROVEMENT
    if kind is ExperimentDecisionKind.REGRESSION:
        return OptimizationOutcome.REGRESSION
    if kind is ExperimentDecisionKind.QUALITY_REGRESSION:
        return OptimizationOutcome.QUALITY_REGRESSION
    if kind in (
        ExperimentDecisionKind.PRECONDITION_FAILURE,
        ExperimentDecisionKind.APPLICATION_FAILURE,
        ExperimentDecisionKind.UNSAFE,
        ExperimentDecisionKind.REJECTED,
    ):
        return OptimizationOutcome.BLOCKED
    if kind is ExperimentDecisionKind.RESTORATION_FAILURE:
        return OptimizationOutcome.FAILED
    if kind in (ExperimentDecisionKind.INVALID, ExperimentDecisionKind.MEASUREMENT_FAILURE):
        return OptimizationOutcome.INVALID
    return OptimizationOutcome.FAILED


@dataclass
class PersistResult:
    optimization: KnowledgeWriteResult
    rollback: KnowledgeWriteResult | None = None
    record: OptimizationRecord | None = None
    rollback_record: RollbackRecord | None = None

    @property
    def persisted(self) -> bool:
        return self.optimization.status in (
            KnowledgeWriteStatus.PERSISTED,
            KnowledgeWriteStatus.DUPLICATE,
        )


@dataclass
class KnowledgeService:
    """Durable knowledge + applicability over one :class:`SQLiteKnowledgeStore`."""

    store: SQLiteKnowledgeStore
    registry: ActionRegistry
    policy: KnowledgePolicy = field(default_factory=KnowledgePolicy)

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #
    def build_record(
        self,
        result: ExperimentResult,
        *,
        retained: bool | None = None,
        not_retained_reason: str | None = None,
    ) -> OptimizationRecord:
        """Build the durable record for one experiment.

        ``retained`` is the caller's final keep decision when it can differ from the
        experiment's own technical verdict (for example a measured improvement that a
        later cost check threw out). ``retained=False`` on a technically improved
        result is recorded as ``REVERTED`` with ``keep=False``, so the store never
        claims a change was kept when it was undone.
        """
        request = result.request
        if request is None:
            raise ValueError("experiment result has no request/context to persist")
        baseline_metrics = (
            result.baseline_snapshot.metrics if result.baseline_snapshot is not None else {}
        )
        context = build_context_from_request(request, baseline_metrics=baseline_metrics)
        action = self.registry.get_optional(request.action_id)
        decision = result.decision_detail
        comparison = result.comparison
        quality = result.quality
        overhead = result.overhead.total_s if result.overhead is not None else None
        rollback_id = (
            f"rb-{request.experiment_id}"
            if result.application is not None and result.application.applied
            else None
        )
        outcome = outcome_from_result(result)
        keep = bool(decision.keep) if decision is not None else result.decision is Decision.KEEP
        evidence = list(decision.evidence) if decision is not None else []
        if retained is False and keep:
            keep = False
            if outcome is OptimizationOutcome.IMPROVED:
                outcome = OptimizationOutcome.REVERTED
            evidence.append(
                "not kept: " + (not_retained_reason or "rejected after measurement")
            )
        return OptimizationRecord(
            record_id=f"opt-{request.experiment_id}",
            experiment_id=request.experiment_id,
            created_at=result.created_at,
            candidate_id=request.candidate_id,
            action_id=request.action_id,
            action_registered=action is not None,
            parameters=dict(request.parameters),
            diagnosis_id=request.source_diagnosis_id,
            diagnosed_bottleneck=request.diagnosed_bottleneck,
            outcome=outcome,
            decision=result.status,
            keep=keep,
            regression=bool(decision.regression) if decision is not None else False,
            quality_regression=(
                bool(decision.quality_regression) if decision is not None else False
            ),
            bottleneck_improved=(
                bool(decision.bottleneck_improved) if decision is not None else False
            ),
            restoration_status=(
                result.restoration.status
                if result.restoration is not None
                else RestorationStatus.NOT_REQUIRED
            ),
            environment_restored=result.environment_restored,
            quality_status=quality.status if quality is not None else None,
            quality_validated=bool(quality.validated) if quality is not None else False,
            quality_similarity=quality.similarity if quality is not None else None,
            context=context,
            baseline_metrics=dict(baseline_metrics),
            candidate_metrics=(
                dict(result.candidate_measurement.metrics)
                if result.candidate_measurement is not None
                else {}
            ),
            measured_benefit=_measured_benefit(comparison),
            measured_cost=_measured_cost(result),
            experiment_overhead_s=overhead,
            evidence=evidence,
            uncertainty=comparison.uncertainty if comparison is not None else "",
            rollback_id=rollback_id,
            provenance="phase7-experiment",
            summary=result.summary or (result.notes or ""),
        )

    def persist_result(
        self,
        result: ExperimentResult,
        *,
        retained: bool | None = None,
        not_retained_reason: str | None = None,
    ) -> PersistResult:
        try:
            record = self.build_record(
                result, retained=retained, not_retained_reason=not_retained_reason
            )
            optimization = self.store.record_optimization(record)
        except KnowledgeUnavailableError as exc:
            return PersistResult(
                optimization=KnowledgeWriteResult(
                    status=KnowledgeWriteStatus.UNAVAILABLE,
                    experiment_id=result.experiment_id,
                    detail=f"knowledge store unavailable: {exc}",
                )
            )
        if optimization.status in (
            KnowledgeWriteStatus.FAILED,
            KnowledgeWriteStatus.CONFLICT,
            KnowledgeWriteStatus.UNAVAILABLE,
        ):
            return PersistResult(optimization=optimization, record=record)

        if optimization.status is KnowledgeWriteStatus.PERSISTED:
            self.store.append_event(
                record.record_id,
                KnowledgeEventType.RECORDED,
                f"outcome={record.outcome.value}",
                {"action_id": record.action_id, "outcome": record.outcome.value},
            )

        rollback_write: KnowledgeWriteResult | None = None
        rollback_record: RollbackRecord | None = None
        try:
            rollback_record = build_rollback_record(result, registry=self.registry)
        except ValueError:
            rollback_record = None
        if rollback_record is not None:
            rollback_write = self.store.record_rollback(rollback_record)
            if (
                rollback_write.status is KnowledgeWriteStatus.PERSISTED
                and optimization.status is KnowledgeWriteStatus.PERSISTED
            ):
                self.store.link_rollback(record.record_id, rollback_record.rollback_id)
        return PersistResult(
            optimization=optimization,
            rollback=rollback_write,
            record=record,
            rollback_record=rollback_record,
        )

    # ------------------------------------------------------------------ #
    # Retrieval / applicability                                           #
    # ------------------------------------------------------------------ #
    def history(self, action_id: str) -> list[OptimizationRecord]:
        return self.store.history_for_action(action_id)

    def query_applicable(
        self,
        action_id: str,
        current: ApplicabilityContext,
        *,
        outcome: OptimizationOutcome | None = None,
        limit: int | None = None,
    ) -> list[ApplicabilityResult]:
        query = KnowledgeQuery(
            action_id=action_id,
            outcome=outcome,
            limit=limit or self.policy.lookup_limit,
        )
        records = self.store.find_records(query, newest_first=False)
        results = [
            evaluate_applicability(record, current, self.policy) for record in records
        ]
        results.sort(
            key=lambda result: (
                _CLASSIFICATION_RANK.get(result.classification.value, 9),
                result.experiment_id,
            )
        )
        return results

    def all_records(self, *, limit: int | None = None) -> list[OptimizationRecord]:
        return self.store.all_records(limit=limit or self.policy.lookup_limit)

    def get_rollback(self, rollback_id: str) -> RollbackRecord | None:
        return self.store.get_rollback(rollback_id)

    def rollback_for_experiment(self, experiment_id: str) -> RollbackRecord | None:
        return self.store.rollback_for_experiment(experiment_id)

    def mark_revalidation_required(self, record_id: str, reason: str) -> None:
        self.store.mark_revalidation_required(record_id, reason)

    def mark_reverted(self, record_id: str, reason: str) -> None:
        self.store.mark_reverted(record_id, reason)

    def health(self) -> KnowledgeHealth:
        return self.store.health()

    def stats(self) -> StorageStats:
        return self.store.stats()

    def current_context_from_record(
        self, record: OptimizationRecord, current: ApplicabilityContext | None
    ) -> ApplicabilityContext:
        return current or record.context


def _measured_benefit(comparison: ExperimentComparison | None) -> dict[str, float]:
    if comparison is None:
        return {}
    benefit: dict[str, float] = {}
    for metric, delta in comparison.deltas.items():
        if delta.role in ("bottleneck_primary", "bottleneck_secondary", "objective"):
            if delta.relative_delta is not None:
                benefit[metric] = delta.relative_delta
    return benefit


def _measured_cost(result: ExperimentResult) -> dict[str, float]:
    cost: dict[str, float] = {}
    if result.overhead is not None:
        cost["experiment_overhead_s"] = result.overhead.total_s
        cost["measurement_s"] = result.overhead.measurement_s
    return cost


def build_service(
    db_path: str | None, registry: ActionRegistry, policy: KnowledgePolicy
) -> KnowledgeService:
    path = db_path or policy.db_path or "data/knowledge.db"
    store = SQLiteKnowledgeStore(path)
    return KnowledgeService(store=store, registry=registry, policy=policy)


