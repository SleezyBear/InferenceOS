"""Phase 9 pre-deployment autonomous optimization-loop orchestrator.

The loop is an *orchestrator*, not a second experiment implementation. It reuses the
real Phase 4 detector, the Phase 5 reasoning lifecycle, the Phase 6 candidate
generator, the Phase 7 experiment engine/executor, and the Phase 8 knowledge store.

Lifecycle per iteration:

    OBSERVE -> DIAGNOSE -> (DEEP-DIAGNOSE) -> GENERATE -> KNOWLEDGE-QUERY ->
    SELECT -> CHECKPOINT -> EXPERIMENT -> VALIDATE/DECIDE -> KEEP/REVERT ->
    PERSIST -> (RE-OBSERVE) -> CONTINUE/TERMINATE

A clean "no meaningful optimization opportunity" outcome is a success; the loop is
fully capable of making zero changes. Every mutation is checkpointed and restored via
the Phase 7 executor. Restoration failure is a safety-terminal condition.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    ApprovalRequirement,
    CandidateEligibility,
    CandidatePlanOutcome,
    Decision,
    ExperimentDecisionKind,
    KnowledgeAvailability,
    KnowledgeHealthStatus,
    KnowledgeSelectionEffect,
    LoopDecisionKind,
    LoopIterationOutcome,
    LoopState,
    LoopTerminationReason,
    RestorationStatus,
)
from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentPolicy,
    ExperimentResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.knowledge import (
    ApplicabilityContext,
    KnowledgeHealth,
    StorageStats,
)
from inferenceos.contracts.optimization import CandidatePlan, OptimizationCandidate
from inferenceos.contracts.optimization_loop import (
    KnowledgeEvidence,
    LoopContext,
    LoopDecision,
    LoopIteration,
    LoopOverhead,
    LoopPolicy,
    LoopResult,
    LoopStageEvent,
    LoopTermination,
    OptimizationRun,
    SafetyCheckpoint,
)
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.reasoning import DeepDiagnosisOutcome
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.experiment.executor import ActionExecutor
from inferenceos.experiment.measurement import MeasurementHarness
from inferenceos.experiment.service import (
    ExperimentInputError,
    baseline_configuration_from,
    run_candidate_experiment,
)
from inferenceos.interfaces.reasoning_engine import ReasoningEngine
from inferenceos.knowledge.fingerprint import build_current_context
from inferenceos.knowledge.service import KnowledgeService
from inferenceos.knowledge.store import KnowledgeError
from inferenceos.loop.checkpoint import (
    capture_checkpoint,
    restore_checkpoint,
    verify_checkpoint,
)
from inferenceos.loop.knowledge import (
    gather_knowledge_evidence,
    selection_adjustment,
    unavailable_evidence,
)
from inferenceos.optimization.context import OptimizationContext, build_optimization_context
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.registry import ActionRegistry
from inferenceos.reasoning.escalation import decide_escalation
from inferenceos.reasoning.lifecycle import run_deep_diagnosis
from inferenceos.reasoning.providers.base import ReasoningProvider

Observer = Callable[[RuntimeConfiguration], BaselineReport | None]

_TERMINAL_FAILURES = {
    ExperimentDecisionKind.APPLICATION_FAILURE,
    ExperimentDecisionKind.MEASUREMENT_FAILURE,
    ExperimentDecisionKind.INVALID,
}


def _canonical_params(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))


class OptimizationLoop:
    """Deterministic orchestrator for one unattended pre-deployment optimization run."""

    def __init__(
        self,
        *,
        executor: ActionExecutor,
        profiler: MeasurementHarness,
        registry: ActionRegistry,
        optimization_policy: OptimizationPolicy,
        experiment_policy: ExperimentPolicy,
        knowledge_service: KnowledgeService,
        loop_policy: LoopPolicy,
        workload_id: str = "short-short",
        seed: int = 1234,
        detector: BottleneckDetector | None = None,
        reasoning_engine: ReasoningEngine | None = None,
        proposal_provider: ReasoningProvider | None = None,
        enable_reasoning_proposals: bool = False,
        observer: Observer | None = None,
        diagnosis_confidence_threshold: float = 0.6,
    ) -> None:
        self._executor = executor
        self._profiler = profiler
        self._registry = registry
        self._optimization_policy = optimization_policy
        self._experiment_policy = experiment_policy
        self._knowledge = knowledge_service
        self._policy = loop_policy
        self._workload_id = workload_id
        self._seed = seed
        self._detector = detector or BottleneckDetector()
        self._reasoning_engine = reasoning_engine
        self._proposal_provider = proposal_provider
        self._enable_reasoning_proposals = enable_reasoning_proposals
        self._observer = observer
        self._confidence_threshold = diagnosis_confidence_threshold

        # Per-run state (reset in ``_reset``).
        self._report: BaselineReport | None = None
        self._report_path: str | None = None
        self._initial_configuration = RuntimeConfiguration()
        self._current_configuration = RuntimeConfiguration()
        self._budget = loop_policy.budget
        self._events: list[LoopStageEvent] = []
        self._iterations: list[LoopIteration] = []
        self._attempted: set[tuple[str, str]] = set()
        self._retained_ids: list[str] = []
        self._errors: list[str] = []
        self._overhead: dict[str, float] = {}
        self._wall_started = 0.0
        self._knowledge_health = KnowledgeHealthStatus.OK
        self._knowledge_available = True
        self._storage: StorageStats | None = None
        self._reasoning_available = False
        self._reobserve_pending = False

    # ------------------------------------------------------------------ #
    # Public entry point                                                   #
    # ------------------------------------------------------------------ #
    def run(
        self,
        report: BaselineReport,
        *,
        diagnosis: BottleneckDiagnosis | None = None,
        outcome: DeepDiagnosisOutcome | None = None,
        report_path: str | None = None,
    ) -> LoopResult:
        self._reset()
        self._report = report
        self._report_path = report_path
        self._wall_started = time.perf_counter()
        self._reasoning_available = bool(
            self._reasoning_engine is not None and self._reasoning_engine.is_available()
        )

        if not report.runs:
            return self._finish_terminal(
                self._termination(
                    LoopTerminationReason.OPERATOR_REQUIRED,
                    "no observation runs are available; cannot establish a baseline",
                )
            )

        first_model = report.runs[0].context.model
        self._initial_configuration = baseline_configuration_from(
            report.runs[0].context.config, first_model
        )
        self._current_configuration = self._initial_configuration
        self._executor.set_configuration(self._initial_configuration)
        self._emit(
            LoopState.INITIALIZED,
            "baseline runtime configuration captured and protected",
            data={"configuration": self._initial_configuration.as_dict()},
        )
        self._check_knowledge_health()
        if self._policy.require_knowledge and not self._knowledge_available:
            return self._finish_terminal(
                self._termination(
                    LoopTerminationReason.KNOWLEDGE_UNAVAILABLE,
                    "durable knowledge is required but the store is unavailable",
                )
            )

        pending_diagnosis = diagnosis
        pending_outcome = outcome
        termination: LoopTermination | None = None
        while termination is None:
            termination = self._iteration(pending_diagnosis, pending_outcome)
            pending_diagnosis = None
            pending_outcome = None
        return self._finish_terminal(termination)

    # ------------------------------------------------------------------ #
    # Iteration                                                            #
    # ------------------------------------------------------------------ #
    def _iteration(
        self,
        diagnosis: BottleneckDiagnosis | None,
        precomputed_outcome: DeepDiagnosisOutcome | None,
    ) -> LoopTermination | None:
        wall = time.perf_counter() - self._wall_started
        optimizer_overhead = self._optimizer_overhead()
        reason = self._budget.exhaustion(
            wall_clock_s=wall, optimizer_overhead_s=optimizer_overhead
        )
        if reason is not None:
            return self._termination(reason, f"budget reached: {reason.value}")

        if self._reobserve_pending:
            reobserve = self._reobserve()
            if reobserve is not None:
                return reobserve

        started = time.perf_counter()
        index = self._budget.iterations_used + 1
        self._budget = self._budget.model_copy(update={"iterations_used": index})
        iteration = LoopIteration(iteration=index)
        self._emit(LoopState.OBSERVING, "observing current state", iteration=index)

        observation_started = time.perf_counter()
        report = self._report
        assert report is not None
        self._record_overhead("observation_s", time.perf_counter() - observation_started)

        # --- Diagnose (+ bounded deep diagnosis when warranted) ---
        diagnosis_started = time.perf_counter()
        base_diagnosis = diagnosis or self._detector.detect(
            report, source_report_path=self._report_path or "<phase9>"
        )
        self._record_overhead("diagnosis_s", time.perf_counter() - diagnosis_started)
        self._emit(
            LoopState.DIAGNOSING,
            f"diagnosis {base_diagnosis.diagnosis_id}",
            iteration=index,
            data={
                "meaningful_opportunity": base_diagnosis.meaningful_opportunity,
                "insufficient_evidence": base_diagnosis.insufficient_evidence,
                "summary": base_diagnosis.summary,
            },
        )

        deep = self._deep_diagnose(base_diagnosis, report, index, precomputed_outcome)
        if isinstance(deep, LoopTermination):
            iteration = self._close(iteration, started, outcome=LoopIterationOutcome.SAFETY_STOP)
            self._iterations.append(iteration)
            return deep
        outcome: DeepDiagnosisOutcome | None = deep
        if outcome is not None:
            iteration = iteration.model_copy(
                update={
                    "deep_diagnosis_used": True,
                    "deep_diagnosis_outcome_id": outcome.outcome_id,
                    "deep_diagnosis_reason": outcome.escalation.reason_code,
                }
            )

        ctx = build_optimization_context(
            base_diagnosis,
            report=report,
            outcome=outcome,
            policy=self._optimization_policy,
        )
        model = ctx.model
        if model is None:
            return self._finish_iteration_error(
                iteration,
                started,
                "no model profile available from the observation context",
                LoopTerminationReason.OPERATOR_REQUIRED,
            )
        baseline_config = baseline_configuration_from(ctx.config, model)

        # --- Generate candidates ---
        generation_started = time.perf_counter()
        plan = generate_candidates(
            base_diagnosis,
            report=report,
            outcome=outcome,
            registry=self._registry,
            policy=self._optimization_policy,
            provider=self._proposal_provider,
            enable_reasoning=(
                self._enable_reasoning_proposals and self._proposal_provider is not None
            ),
        )
        self._record_overhead("candidate_generation_s", time.perf_counter() - generation_started)
        self._emit(
            LoopState.GENERATING,
            f"candidate plan {plan.plan_id}: {plan.outcome.value}",
            iteration=index,
            data={"terminal_reason": plan.terminal_reason},
        )
        iteration = iteration.model_copy(
            update={
                "diagnosis_id": base_diagnosis.diagnosis_id,
                "diagnosis_kind": (
                    ctx.effective_kind.value if ctx.effective_kind is not None else None
                ),
                "diagnosis_confidence": ctx.effective_confidence,
                "diagnosis_strength": ctx.effective_strength,
                "plan_id": plan.plan_id,
                "plan_outcome": plan.outcome,
                "plan_terminal_reason": plan.terminal_reason,
                "plan": plan,
                "candidates_considered": len(plan.candidates),
                "eligible_candidate_ids": list(plan.eligible_ids),
            }
        )

        terminal = self._plan_terminal(plan)
        if terminal is not None:
            self._emit(LoopState.TERMINATED, terminal.detail, iteration=index)
            iteration = self._close(iteration, started, outcome=LoopIterationOutcome.NO_CANDIDATE)
            self._iterations.append(iteration)
            return terminal

        # --- Knowledge query + selection ---
        current_context = self._applicability_context(
            base_diagnosis, ctx, report, baseline_config
        )
        query_started = time.perf_counter()
        evidence_map = self._gather_knowledge(plan, current_context)
        self._record_overhead("knowledge_query_s", time.perf_counter() - query_started)
        if self._policy.require_knowledge and any(
            evidence.availability is KnowledgeAvailability.UNAVAILABLE
            for evidence in evidence_map.values()
        ):
            iteration = self._close(
                iteration, started, outcome=LoopIterationOutcome.NO_CANDIDATE
            )
            iteration = iteration.model_copy(
                update={"knowledge": list(evidence_map.values())}
            )
            self._iterations.append(iteration)
            return self._termination(
                LoopTerminationReason.KNOWLEDGE_UNAVAILABLE,
                "durable knowledge is required but the knowledge query failed",
            )
        self._emit(
            LoopState.KNOWLEDGE_QUERY,
            f"consulted {sum(e.records_considered for e in evidence_map.values())} prior record(s)",
            iteration=index,
        )

        selection_started = time.perf_counter()
        decision, selected = self._select(plan, evidence_map)
        self._record_overhead("selection_s", time.perf_counter() - selection_started)
        self._emit(
            LoopState.SELECTING,
            f"decision {decision.kind.value}: {decision.reason}",
            iteration=index,
            data={"ranked": decision.ranked_candidate_ids},
        )
        iteration = iteration.model_copy(
            update={
                "decision": decision,
                "knowledge": list(evidence_map.values()),
                "rejected_candidates": self._knowledge_rejections(evidence_map),
            }
        )
        if selected is None or decision.kind is not LoopDecisionKind.PROCEED:
            reason_ = decision.termination_reason or LoopTerminationReason.NO_ELIGIBLE_CANDIDATES
            iteration = self._close(
                iteration, started, outcome=LoopIterationOutcome.NO_CANDIDATE
            )
            self._iterations.append(iteration)
            return self._termination(reason_, decision.reason)

        # --- Checkpoint ---
        checkpoint_started = time.perf_counter()
        checkpoint = capture_checkpoint(
            configuration=baseline_config,
            model_ref=model.name,
            model_path=model.path,
            hardware_hostname=report.hardware.hostname,
            workload_id=self._workload_id,
            diagnosis_id=base_diagnosis.diagnosis_id,
            context_identity=current_context.fingerprint.identity_hash,
            baseline_metrics=self._baseline_metric_means(report),
            notes=["protected optimization-controlled target runtime configuration"],
        )
        self._record_overhead("checkpoint_s", time.perf_counter() - checkpoint_started)
        self._emit(
            LoopState.CHECKPOINTING,
            f"checkpoint {checkpoint.checkpoint_id}",
            iteration=index,
        )
        iteration = iteration.model_copy(
            update={
                "checkpoint": checkpoint,
                "selected_candidate_id": selected.candidate_id,
                "selected_action_id": selected.action_id,
                "selected_parameters": dict(selected.parameters),
            }
        )

        # --- Experiment (Phase 7 owns restoration) ---
        experiment_started = time.perf_counter()
        self._emit(
            LoopState.EXPERIMENTING,
            f"experimenting candidate {selected.candidate_id} ({selected.action_id})",
            iteration=index,
        )
        try:
            result = run_candidate_experiment(
                selected,
                plan=plan,
                diagnosis=base_diagnosis,
                report=report,
                outcome=outcome,
                workload_id=self._workload_id,
                registry=self._registry,
                optimization_policy=self._optimization_policy,
                experiment_policy=self._experiment_policy,
                executor=self._executor,
                profiler=self._profiler,
                seed=self._seed,
            )
        except ExperimentInputError as exc:
            self._record_overhead("experiment_s", time.perf_counter() - experiment_started)
            self._attempted.add((selected.action_id, _canonical_params(selected.parameters)))
            return self._finish_iteration_error(
                iteration,
                started,
                f"experiment input error: {exc}",
                LoopTerminationReason.EXECUTION_FAILURE,
            )
        self._record_overhead("experiment_s", time.perf_counter() - experiment_started)
        self._budget = self._budget.model_copy(
            update={"experiments_used": self._budget.experiments_used + 1}
        )

        self._emit(
            LoopState.DECIDING,
            f"experiment {result.experiment_id}: "
            f"{result.status.value if result.status else 'unknown'}",
            iteration=index,
        )
        iteration = self._apply_result(iteration, result, checkpoint)
        self._iterations.append(iteration)

        # --- Persist knowledge ---
        self._emit(
            LoopState.PERSISTING,
            f"persisting outcome of {result.experiment_id}",
            iteration=index,
        )
        self._persist(result)

        # --- Terminal conditions from the experiment ---
        if result.restoration is not None and result.restoration.status is RestorationStatus.FAILED:
            return self._termination(
                LoopTerminationReason.RESTORATION_FAILURE,
                "baseline restoration failed; the protected state is not trustworthy",
                safety=True,
            )
        if result.status is ExperimentDecisionKind.QUALITY_REGRESSION:
            return self._termination(
                LoopTerminationReason.QUALITY_SAFETY_STOP,
                "quality regression detected; the candidate was not retained",
            )
        if result.executed and result.status in _TERMINAL_FAILURES:
            return self._termination(
                LoopTerminationReason.EXECUTION_FAILURE,
                f"experiment failed: {result.status.value}",
            )

        # --- Keep/revert side effects ---
        self._attempted.add((selected.action_id, _canonical_params(selected.parameters)))
        if iteration.retained:
            self._current_configuration = selected_configuration(result, baseline_config)
            self._executor.set_configuration(self._current_configuration)
            self._retained_ids.append(selected.candidate_id)
            self._reobserve_pending = True
        else:
            self._emit(
                LoopState.RESTORING,
                "verifying the protected baseline was restored",
                iteration=index,
            )
            if not verify_checkpoint(self._executor, checkpoint):
                return self._termination(
                    LoopTerminationReason.RESTORATION_FAILURE,
                    "executor does not hold the protected baseline after revert",
                    safety=True,
                )
            self._current_configuration = baseline_config
        return None

    # ------------------------------------------------------------------ #
    # Deep diagnosis                                                       #
    # ------------------------------------------------------------------ #
    def _deep_diagnose(
        self,
        diagnosis: BottleneckDiagnosis,
        report: BaselineReport,
        index: int,
        precomputed: DeepDiagnosisOutcome | None = None,
    ) -> DeepDiagnosisOutcome | LoopTermination | None:
        if precomputed is not None:
            return precomputed
        escalation = decide_escalation(
            diagnosis,
            enabled=self._policy.allow_deep_diagnosis,
            confidence_threshold=self._confidence_threshold,
        )
        if not escalation.should_escalate:
            return None
        if self._budget.deep_diagnoses_used >= self._budget.max_deep_diagnoses:
            if self._policy.require_reasoning_when_escalated:
                return self._termination(
                    LoopTerminationReason.DEEP_DIAGNOSIS_BUDGET_EXHAUSTED,
                    "reasoning is required but the deep-diagnosis budget is exhausted",
                )
            return None
        if self._reasoning_engine is None or not self._reasoning_available:
            if self._policy.require_reasoning_when_escalated:
                return self._termination(
                    LoopTerminationReason.REASONING_UNAVAILABLE,
                    "reasoning is required but no reasoning engine is available",
                )
            return None

        started = time.perf_counter()
        self._emit(
            LoopState.DEEP_DIAGNOSING,
            f"escalating ({escalation.reason_code}); deterministic only would be insufficient",
            iteration=index,
            data={"rationale": escalation.rationale},
        )
        outcome = run_deep_diagnosis(
            diagnosis,
            report=report,
            engine=self._reasoning_engine,
            enabled=True,
            confidence_threshold=self._confidence_threshold,
        )
        self._record_overhead("deep_reasoning_s", time.perf_counter() - started)
        self._budget = self._budget.model_copy(
            update={"deep_diagnoses_used": self._budget.deep_diagnoses_used + 1}
        )
        return outcome

    # ------------------------------------------------------------------ #
    # Knowledge / selection                                                #
    # ------------------------------------------------------------------ #
    def _gather_knowledge(
        self, plan: CandidatePlan, current_context: ApplicabilityContext
    ) -> dict[str, KnowledgeEvidence]:
        evidence: dict[str, KnowledgeEvidence] = {}
        for candidate in plan.candidates:
            if candidate.eligibility is not CandidateEligibility.ELIGIBLE:
                continue
            evidence[candidate.candidate_id] = gather_knowledge_evidence(
                candidate,
                service=self._knowledge,
                current_context=current_context,
                policy=self._policy,
            )
        return evidence

    def _select(
        self, plan: CandidatePlan, evidence_map: dict[str, KnowledgeEvidence]
    ) -> tuple[LoopDecision, OptimizationCandidate | None]:
        eligible = [
            candidate
            for candidate in plan.candidates
            if candidate.eligibility is CandidateEligibility.ELIGIBLE and candidate.executable
        ]
        scored: list[tuple[float, str, OptimizationCandidate]] = []
        for candidate in eligible:
            base = candidate.ranking.composite_score if candidate.ranking is not None else 0.0
            evidence = evidence_map.get(candidate.candidate_id)
            adjustment = selection_adjustment(evidence, self._policy) if evidence else 0.0
            scored.append((base + adjustment, candidate.candidate_id, candidate))
        # Deterministic ordering: higher score, then candidate id.
        scored.sort(key=lambda item: (-item[0], item[1]))
        considered = scored[: self._policy.max_candidates_per_iteration]

        ranked_ids = [candidate.candidate_id for _, _, candidate in considered]

        skipped: list[str] = []
        for score, candidate_id, candidate in considered:
            key = (candidate.action_id, _canonical_params(candidate.parameters))
            if key in self._attempted:
                skipped.append(f"{candidate_id}: already attempted this run")
                continue
            evidence = evidence_map.get(candidate_id)
            if evidence is not None and evidence.effect is KnowledgeSelectionEffect.REJECTED:
                skipped.append(
                    f"{candidate_id}: rejected by directly-applicable historical evidence"
                )
                continue
            if (
                self._policy.autonomous_only
                and candidate.approval is ApprovalRequirement.REQUIRED
            ):
                skipped.append(f"{candidate_id}: approval required; not autonomous")
                continue
            knowledge = [evidence] if evidence is not None else []
            return (
                LoopDecision(
                    kind=LoopDecisionKind.PROCEED,
                    reason=(
                        f"selected {candidate.action_id} ({candidate_id}) at knowledge-adjusted "
                        f"score {score:.4f}"
                    ),
                    selected_candidate_id=candidate_id,
                    action_id=candidate.action_id,
                    params=dict(candidate.parameters),
                    ranked_candidate_ids=ranked_ids,
                    knowledge=knowledge,
                ),
                candidate,
            )

        reason = (
            "eligible candidates were all rejected or already attempted: "
            + "; ".join(skipped)
            if skipped
            else "no candidate passed the eligibility/safety gates"
        )
        termination_reason = (
            LoopTerminationReason.ALL_CANDIDATES_REJECTED
            if skipped
            else LoopTerminationReason.NO_ELIGIBLE_CANDIDATES
        )
        return (
            LoopDecision(
                kind=LoopDecisionKind.TERMINATE,
                reason=reason,
                ranked_candidate_ids=ranked_ids,
                termination_reason=termination_reason,
                knowledge=list(evidence_map.values()),
            ),
            None,
        )

    def _knowledge_rejections(
        self, evidence_map: dict[str, KnowledgeEvidence]
    ) -> dict[str, str]:
        return {
            candidate_id: (
                "directly-applicable negative historical evidence"
                if evidence.effect is KnowledgeSelectionEffect.REJECTED
                else evidence.detail
            )
            for candidate_id, evidence in evidence_map.items()
            if evidence.effect is KnowledgeSelectionEffect.REJECTED
        }

    # ------------------------------------------------------------------ #
    # Result handling                                                      #
    # ------------------------------------------------------------------ #
    def _apply_result(
        self, iteration: LoopIteration, result: ExperimentResult, checkpoint: SafetyCheckpoint
    ) -> LoopIteration:
        decision = result.decision_detail
        retained = (
            decision is not None
            and decision.keep
            and result.status is ExperimentDecisionKind.IMPROVED
        )
        restoration_status = (
            result.restoration.status if result.restoration is not None else RestorationStatus.NOT_REQUIRED
        )
        verified = bool(result.restoration is not None and result.restoration.verified)
        if retained:
            outcome = LoopIterationOutcome.CANDIDATE_KEPT
        elif result.restoration is not None and result.restoration.status is RestorationStatus.FAILED:
            outcome = LoopIterationOutcome.RESTORATION_FAILURE
        elif not result.executed:
            outcome = LoopIterationOutcome.CANDIDATE_REJECTED
        else:
            outcome = LoopIterationOutcome.CANDIDATE_REVERTED

        return iteration.model_copy(
            update={
                "experiment_id": result.experiment_id,
                "experiment_status": result.status,
                "experiment_decision": (
                    Decision.KEEP if retained else (result.decision if result.decision else Decision.REVERT)
                ),
                "retained": retained,
                "restoration_status": restoration_status,
                "restoration_verified": verified,
                "baseline_metrics": self._metric_means(result.baseline_measurement),
                "candidate_metrics": self._metric_means(result.candidate_measurement),
                "outcome": outcome,
                "errors": list(result.errors),
                "finished_at": datetime.now(timezone.utc),
                "overhead_s": result.overhead.total_s if result.overhead is not None else 0.0,
                "notes": [
                    f"experiment={result.experiment_id}",
                    f"status={result.status.value if result.status else 'unknown'}",
                    f"retained={retained}",
                    f"restoration={restoration_status.value}",
                ],
            }
        )

    def _persist(self, result: ExperimentResult) -> None:
        started = time.perf_counter()
        try:
            persist = self._knowledge.persist_result(result)
            status = persist.optimization.status
        except KnowledgeError as exc:
            self._errors.append(f"knowledge persistence failed: {exc}")
            self._record_overhead("persistence_s", time.perf_counter() - started)
            return
        self._record_overhead("persistence_s", time.perf_counter() - started)
        if self._iterations:
            iteration = self._iterations[-1]
            self._iterations[-1] = iteration.model_copy(update={"persist_status": status})
        if self._policy.require_knowledge and status.value in ("failed", "unavailable"):
            self._errors.append(f"required knowledge persistence did not succeed: {status.value}")

    def _plan_terminal(self, plan: CandidatePlan) -> LoopTermination | None:
        if plan.outcome is CandidatePlanOutcome.NO_OPPORTUNITY:
            return self._termination(
                LoopTerminationReason.NO_MEANINGFUL_OPPORTUNITY, plan.terminal_reason
            )
        if plan.outcome is CandidatePlanOutcome.INSUFFICIENT_EVIDENCE:
            return self._termination(
                LoopTerminationReason.INSUFFICIENT_EVIDENCE, plan.terminal_reason
            )
        if plan.outcome is CandidatePlanOutcome.NO_VIABLE_CANDIDATES:
            return self._termination(
                LoopTerminationReason.NO_ELIGIBLE_CANDIDATES, plan.terminal_reason
            )
        return None

    def _reobserve(self) -> LoopTermination | None:
        self._reobserve_pending = False
        if self._observer is None:
            return self._termination(
                LoopTerminationReason.COMPLETED,
                "retained optimization; no observer is available to re-measure the new baseline",
            )
        started = time.perf_counter()
        self._emit(LoopState.RE_OBSERVING, "re-observing under the retained configuration")
        new_report = self._observer(self._current_configuration)
        self._record_overhead("observation_s", time.perf_counter() - started)
        if new_report is None or not new_report.runs:
            return self._termination(
                LoopTerminationReason.COMPLETED,
                "retained optimization; re-observation produced no usable runs",
            )
        self._report = new_report
        return None

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    def _applicability_context(
        self,
        diagnosis: BottleneckDiagnosis,
        ctx: OptimizationContext,
        report: BaselineReport,
        configuration: RuntimeConfiguration,
    ) -> ApplicabilityContext:
        return build_current_context(
            hardware=report.hardware,
            model=ctx.model,
            workload=ctx.workload,
            backend=report.backend,
            runtime=configuration,
            diagnosis=(
                ctx.effective_kind.value if ctx.effective_kind is not None else None
            ),
            conditions=self._conditions,
        )

    def _baseline_metric_means(self, report: BaselineReport) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if report.runs:
            run_metrics = report.runs[0].observation.metrics
            for name in ("peak_rss_bytes", "hard_page_faults_per_sec", "decode_tokens_per_sec"):
                value: object = getattr(run_metrics, name, None)
                if isinstance(value, (int, float)):
                    metrics[name] = float(value)
        return metrics

    def _metric_means(self, measurement: ExperimentMeasurement | None) -> dict[str, float]:
        if measurement is None:
            return {}
        means: dict[str, float] = {}
        for name in ("peak_rss_bytes", "hard_page_faults_per_sec", "decode_tokens_per_sec"):
            stats = measurement.metrics.get(name)
            if stats is not None and stats.mean is not None:
                means[name] = float(stats.mean)
        return means

    def _check_knowledge_health(self) -> None:
        try:
            health = self._knowledge.health()
            self._knowledge_health = health.status
            self._knowledge_available = health.status is not KnowledgeHealthStatus.UNAVAILABLE
            self._storage = self._knowledge.stats()
        except KnowledgeError as exc:
            self._knowledge_health = KnowledgeHealthStatus.UNAVAILABLE
            self._knowledge_available = False
            self._storage = None
            self._errors.append(f"knowledge store health check failed: {exc}")

    def _record_overhead(self, key: str, seconds: float) -> None:
        self._overhead[key] = self._overhead.get(key, 0.0) + max(0.0, seconds)

    def _optimizer_overhead(self) -> float:
        return sum(
            value for key, value in self._overhead.items() if key != "experiment_s"
        )

    def _emit(
        self,
        stage: LoopState,
        detail: str,
        *,
        iteration: int = 0,
        data: dict[str, object] | None = None,
    ) -> LoopStageEvent:
        event = LoopStageEvent(
            stage=stage,
            timestamp=datetime.now(timezone.utc),
            iteration=iteration,
            detail=detail,
            data=data or {},
        )
        self._events.append(event)
        return event

    def _close(
        self, iteration: LoopIteration, started: float, *, outcome: LoopIterationOutcome
    ) -> LoopIteration:
        return iteration.model_copy(
            update={
                "finished_at": datetime.now(timezone.utc),
                "outcome": outcome,
                "overhead_s": time.perf_counter() - started,
            }
        )

    def _finish_iteration_error(
        self,
        iteration: LoopIteration,
        started: float,
        detail: str,
        reason: LoopTerminationReason,
    ) -> LoopTermination:
        self._errors.append(detail)
        self._iterations.append(
            self._close(iteration, started, outcome=LoopIterationOutcome.EXECUTION_ERROR)
        )
        return self._termination(reason, detail)

    def _termination(
        self, reason: LoopTerminationReason, detail: str, *, safety: bool = False
    ) -> LoopTermination:
        clean = reason in {
            LoopTerminationReason.NO_MEANINGFUL_OPPORTUNITY,
            LoopTerminationReason.INSUFFICIENT_EVIDENCE,
            LoopTerminationReason.NO_ELIGIBLE_CANDIDATES,
            LoopTerminationReason.ALL_CANDIDATES_REJECTED,
            LoopTerminationReason.OPTIMIZATION_BUDGET_EXHAUSTED,
            LoopTerminationReason.EXPERIMENT_BUDGET_EXHAUSTED,
            LoopTerminationReason.TIME_BUDGET_EXHAUSTED,
            LoopTerminationReason.OVERHEAD_BUDGET_EXHAUSTED,
            LoopTerminationReason.DEEP_DIAGNOSIS_BUDGET_EXHAUSTED,
            LoopTerminationReason.COMPLETED,
        }
        return LoopTermination(
            reason=reason,
            detail=detail,
            clean=clean and not safety,
            safety=safety,
            at_iteration=self._budget.iterations_used,
        )

    def _finish_terminal(self, termination: LoopTermination) -> LoopResult:
        wall = time.perf_counter() - self._wall_started
        self._budget = self._budget.model_copy(
            update={
                "wall_clock_used_s": wall,
                "optimizer_overhead_used_s": self._optimizer_overhead(),
            }
        )
        self._emit(
            LoopState.TERMINATED,
            f"terminated: {termination.reason.value}",
            data={"detail": termination.detail, "clean": termination.clean},
        )
        overhead = LoopOverhead(
            observation_s=self._overhead.get("observation_s", 0.0),
            diagnosis_s=self._overhead.get("diagnosis_s", 0.0),
            deep_reasoning_s=self._overhead.get("deep_reasoning_s", 0.0),
            candidate_generation_s=self._overhead.get("candidate_generation_s", 0.0),
            knowledge_query_s=self._overhead.get("knowledge_query_s", 0.0),
            selection_s=self._overhead.get("selection_s", 0.0),
            checkpoint_s=self._overhead.get("checkpoint_s", 0.0),
            experiment_s=self._overhead.get("experiment_s", 0.0),
            persistence_s=self._overhead.get("persistence_s", 0.0),
            orchestration_s=self._overhead.get("orchestration_s", 0.0),
            optimizer_overhead_s=self._optimizer_overhead(),
            total_s=sum(self._overhead.values()),
            notes=[
                "experiment measurement time is tracked separately from optimizer overhead",
            ],
        )
        changed = bool(self._retained_ids)
        run = OptimizationRun(
            run_id=f"run-{uuid.uuid4().hex[:12]}",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            context=self._loop_context(),
            initial_configuration=self._initial_configuration,
            final_configuration=self._current_configuration,
            policy=self._policy,
            budget=self._budget,
            iterations=list(self._iterations),
            state_events=list(self._events),
            termination=termination,
            retained_candidate_ids=list(self._retained_ids),
            changed=changed,
            knowledge_health=self._knowledge_health,
            knowledge_available=self._knowledge_available,
            reasoning_available=self._reasoning_available,
            knowledge=self._knowledge_health_obj(),
            storage=self._storage,
            overhead=overhead,
            errors=list(self._errors),
            summary=self._summary(termination, changed),
        )
        return LoopResult(
            run=run,
            termination=termination,
            initial_configuration=self._initial_configuration,
            final_configuration=self._current_configuration,
            changed=changed,
            iterations_performed=len(self._iterations),
            report_path=self._report_path,
            summary=run.summary,
        )

    def _knowledge_health_obj(self) -> KnowledgeHealth | None:
        try:
            return self._knowledge.health()
        except KnowledgeError:
            return None

    def _loop_context(self) -> LoopContext:
        report = self._report
        model = report.runs[0].context.model if report is not None and report.runs else None
        return LoopContext(
            report_source=self._report_path,
            hardware_hostname=report.hardware.hostname if report is not None else None,
            model_ref=model.name if model is not None else None,
            model_architecture=model.architecture if model is not None else None,
            workload_id=self._workload_id,
            backend_name=report.backend.backend_name if report is not None else None,
        )

    def _summary(self, termination: LoopTermination, changed: bool) -> str:
        state = "optimized" if changed else "retained current configuration"
        return (
            f"Phase 9 loop {state} over {self._budget.iterations_used} iteration(s) and "
            f"{self._budget.experiments_used} experiment(s); terminated "
            f"({termination.reason.value}): {termination.detail}"
        )

    def _reset(self) -> None:
        self._events = []
        self._iterations = []
        self._attempted = set()
        self._retained_ids = []
        self._errors = []
        self._overhead = {}
        self._budget = self._policy.budget
        self._knowledge_health = KnowledgeHealthStatus.OK
        self._knowledge_available = True
        self._storage = None
        self._reobserve_pending = False
        self._current_configuration = RuntimeConfiguration()

    @property
    def _conditions(self) -> dict[str, float]:
        report = self._report
        if report is None or not report.runs:
            return {}
        metrics = report.runs[0].observation.metrics
        conditions: dict[str, float] = {}
        for name in ("available_ram_bytes", "hard_page_faults_per_sec", "peak_rss_bytes"):
            value: object = getattr(metrics, name, None)
            if isinstance(value, (int, float)):
                conditions[name] = float(value)
        return conditions


def selected_configuration(
    result: ExperimentResult, fallback: RuntimeConfiguration
) -> RuntimeConfiguration:
    if result.request is not None:
        return result.request.experiment_configuration
    return fallback


def run_optimization_loop(
    report: BaselineReport,
    *,
    executor: ActionExecutor,
    profiler: MeasurementHarness,
    registry: ActionRegistry,
    optimization_policy: OptimizationPolicy,
    experiment_policy: ExperimentPolicy,
    knowledge_service: KnowledgeService,
    loop_policy: LoopPolicy,
    workload_id: str = "short-short",
    seed: int = 1234,
    reasoning_engine: ReasoningEngine | None = None,
    proposal_provider: ReasoningProvider | None = None,
    enable_reasoning_proposals: bool = False,
    observer: Observer | None = None,
    diagnosis: BottleneckDiagnosis | None = None,
    outcome: DeepDiagnosisOutcome | None = None,
    report_path: str | None = None,
) -> LoopResult:
    """Convenience wrapper constructing and running one loop."""
    loop = OptimizationLoop(
        executor=executor,
        profiler=profiler,
        registry=registry,
        optimization_policy=optimization_policy,
        experiment_policy=experiment_policy,
        knowledge_service=knowledge_service,
        loop_policy=loop_policy,
        workload_id=workload_id,
        seed=seed,
        reasoning_engine=reasoning_engine,
        proposal_provider=proposal_provider,
        enable_reasoning_proposals=enable_reasoning_proposals,
        observer=observer,
    )
    return loop.run(report, diagnosis=diagnosis, outcome=outcome, report_path=report_path)
