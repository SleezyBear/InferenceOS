"""Phase 7 experiment engine: apply -> measure -> validate -> decide -> restore.

The engine owns the experiment lifecycle. It captures a reproducible baseline, applies
exactly one registered candidate through the controlled executor, measures real
inference before and after, validates quality, compares, decides, and — always, via a
``finally`` block — restores and verifies the protected baseline configuration.

It never persists a deployment, never learns from history, and never chains another
experiment (Phase 8+).
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.contracts.enums import (
    Decision,
    ExperimentDecisionKind,
    ExperimentStage,
    RestorationStatus,
)
from inferenceos.contracts.experiment import (
    BaselineSnapshot,
    ExperimentComparison,
    ExperimentDecision,
    ExperimentEvent,
    ExperimentMeasurement,
    ExperimentOverhead,
    ExperimentRequest,
    ExperimentResult,
    QualityScore,
    QualityValidationResult,
    RestorationResult,
    RestorationVerification,
    RuntimeConfiguration,
)
from inferenceos.contracts.optimization import CandidateAction
from inferenceos.experiment.comparison import compare
from inferenceos.experiment.decision import decide
from inferenceos.experiment.executor import ActionExecutor
from inferenceos.experiment.measurement import MeasurementHarness, measure_configuration
from inferenceos.experiment.quality import validate_quality
from inferenceos.experiment.quality_policy import optimization_magnitude
from inferenceos.experiment.provenance import build_provenance
from inferenceos.experiment.restoration import RestorationVerifier
from inferenceos.profiling.workloads import BenchmarkWorkload


def run_experiment(
    request: ExperimentRequest,
    *,
    executor: ActionExecutor,
    profiler: MeasurementHarness,
    workload: BenchmarkWorkload,
    action: CandidateAction | None,
    revalidation_ok: bool = True,
    revalidation_kind: ExperimentDecisionKind | None = None,
    revalidation_reason: str = "ok",
    restoration_verifier: RestorationVerifier | None = None,
) -> ExperimentResult:
    started = time.perf_counter()
    deadline = started + min(request.time_limit_s, request.policy.time_limit_s)
    events: list[ExperimentEvent] = [
        _event(
            ExperimentStage.REQUESTED,
            f"candidate={request.candidate_id} action={request.action_id}",
            parameters=request.parameters,
        )
    ]
    errors: list[str] = []
    timings = {
        "baseline_setup_s": 0.0,
        "application_s": 0.0,
        "measurement_s": 0.0,
        "validation_s": 0.0,
        "restoration_s": 0.0,
    }

    baseline_snapshot: BaselineSnapshot | None = None
    baseline_measurement: ExperimentMeasurement | None = None
    candidate_measurement: ExperimentMeasurement | None = None
    application = None
    quality: QualityValidationResult | None = None
    comparison: ExperimentComparison | None = None
    provisional: ExperimentDecision | None = None
    restoration: RestorationResult | None = None
    executed = False
    measurement_valid = False

    try:
        if not revalidation_ok:
            events.append(
                _event(ExperimentStage.PRECONDITIONS_REVALIDATED, f"failed: {revalidation_reason}")
            )
            errors.append(revalidation_reason)
        else:
            events.append(_event(ExperimentStage.PRECONDITIONS_REVALIDATED, "ok"))
            if action is None:
                errors.append("no resolved action supplied for execution")
                events.append(_event(ExperimentStage.FAILED, "no resolved action"))
            else:
                # --- Baseline ---
                set_started = time.perf_counter()
                executor.set_configuration(request.baseline_configuration)
                baseline_measurement = measure_configuration(
                    profiler=profiler,
                    model=request.model,
                    workload=workload,
                    config=request.baseline_configuration,
                    policy=request.policy,
                    phase="baseline",
                    seed=request.seed,
                    deadline=deadline,
                    max_runs=request.policy.max_total_runs,
                )
                timings["baseline_setup_s"] += time.perf_counter() - set_started
                baseline_snapshot = _capture_snapshot(
                    request, request.baseline_configuration, baseline_measurement
                )
                if restoration_verifier is not None:
                    try:
                        restoration_verifier.capture(request.baseline_configuration)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(
                            f"restoration baseline capture failed: {type(exc).__name__}: {exc}"
                        )
                events.append(
                    _event(
                        ExperimentStage.BASELINE_CAPTURED,
                        f"valid_runs={baseline_measurement.valid_runs} "
                        f"invalid_runs={baseline_measurement.invalid_runs}",
                    )
                )

                # --- Apply ---
                apply_started = time.perf_counter()
                application = executor.apply(action, request.parameters)
                timings["application_s"] += time.perf_counter() - apply_started
                executed = application.applied
                events.append(
                    _event(
                        ExperimentStage.CANDIDATE_APPLIED,
                        application.reason,
                        applied=application.applied,
                        changes=[change.model_dump() for change in application.changes],
                    )
                )
                if not application.applied:
                    errors.append(application.reason)

                baseline_ok = (
                    baseline_measurement.valid_runs >= request.policy.min_valid_samples
                )
                remaining_budget = (
                    request.policy.max_total_runs - baseline_measurement.total_runs
                )
                required_budget = (
                    request.policy.warmup_repetitions + request.policy.min_valid_samples
                )
                if not baseline_ok:
                    errors.append("insufficient valid baseline samples")
                elif not application.applied:
                    pass
                elif remaining_budget < required_budget:
                    errors.append(
                        "experiment run budget exhausted before candidate measurement"
                    )
                else:
                    # --- Measure candidate ---
                    events.append(_event(ExperimentStage.MEASUREMENT_STARTED, "candidate"))
                    measure_started = time.perf_counter()
                    candidate_measurement = measure_configuration(
                        profiler=profiler,
                        model=request.model,
                        workload=workload,
                        config=executor.current(),
                        policy=request.policy,
                        phase="candidate",
                        seed=request.seed,
                        deadline=deadline,
                        max_runs=remaining_budget,
                    )
                    timings["measurement_s"] += time.perf_counter() - measure_started
                    events.append(
                        _event(
                            ExperimentStage.MEASUREMENT_COMPLETED,
                            f"valid_runs={candidate_measurement.valid_runs} "
                            f"invalid_runs={candidate_measurement.invalid_runs}",
                        )
                    )

                    validation_started = time.perf_counter()
                    quality = validate_quality(
                        baseline=baseline_measurement,
                        candidate=candidate_measurement,
                        workload=workload,
                        policy=request.policy,
                        action_id=request.action_id,
                        diagnosed_bottleneck=request.diagnosed_bottleneck,
                        diagnosis_confidence=request.diagnosis_confidence,
                        evidence_strength=request.diagnosis_evidence_strength,
                        optimization_magnitude=optimization_magnitude(
                            request.baseline_configuration, request.experiment_configuration
                        ),
                        model=request.model,
                    )
                    comparison = compare(
                        baseline=baseline_measurement,
                        candidate=candidate_measurement,
                        bottleneck_metric=request.bottleneck_metric,
                        bottleneck_direction=request.bottleneck_direction,
                        quality=quality,
                        policy=request.policy,
                        diagnosed_bottleneck=request.diagnosed_bottleneck,
                    )
                    timings["validation_s"] += time.perf_counter() - validation_started
                    events.append(
                        _event(ExperimentStage.QUALITY_VALIDATED, quality.status.value)
                    )
                    events.append(
                        _event(
                            ExperimentStage.COMPARISON_COMPLETED,
                            f"bottleneck_improved={comparison.bottleneck_improved} "
                            f"objective_regressed={comparison.objective_regressed}",
                        )
                    )
                    measurement_valid = (
                        baseline_measurement.valid_runs >= request.policy.min_valid_samples
                        and candidate_measurement.valid_runs >= request.policy.min_valid_samples
                    )
                    provisional = decide(
                        comparison=comparison,
                        quality=quality,
                        application=application,
                        restoration_status=RestorationStatus.PENDING,
                        measurement_valid=measurement_valid,
                        precondition_ok=True,
                        precondition_kind=None,
                        precondition_reason="ok",
                        policy=request.policy,
                        errors=errors,
                    )
                    events.append(
                        _event(ExperimentStage.DECISION_MADE, provisional.kind.value)
                    )
    except Exception as exc:  # noqa: BLE001 - never leave a mutation uncontrolled
        errors.append(f"{type(exc).__name__}: {exc}")
        events.append(_event(ExperimentStage.FAILED, f"{type(exc).__name__}: {exc}"))
    finally:
        # --- Mandatory restoration ---
        events.append(_event(ExperimentStage.RESTORATION_STARTED, "restoring protected baseline"))
        restore_started = time.perf_counter()
        if request.policy.restore_baseline:
            try:
                restoration = executor.restore(request.baseline_configuration)
            except Exception as exc:  # noqa: BLE001
                restoration = RestorationResult(
                    attempted=True,
                    status=RestorationStatus.FAILED,
                    expected_configuration=request.baseline_configuration,
                    restored_configuration=executor.current(),
                    verified=False,
                    reason=f"restoration raised: {type(exc).__name__}: {exc}",
                    restored_at=datetime.now(timezone.utc),
                )
        else:  # pragma: no cover - policy default restores
            restoration = RestorationResult(
                attempted=False,
                status=RestorationStatus.NOT_REQUIRED,
                expected_configuration=request.baseline_configuration,
                restored_configuration=executor.current(),
                verified=executor.current() == request.baseline_configuration,
                reason="restoration disabled by policy",
                restored_at=datetime.now(timezone.utc),
            )
        timings["restoration_s"] += time.perf_counter() - restore_started
        if (
            restoration is not None
            and restoration_verifier is not None
            and request.policy.restore_baseline
        ):
            try:
                verification = restoration_verifier.verify(request.baseline_configuration)
            except Exception as exc:  # noqa: BLE001
                verification = RestorationVerification(
                    attempted=True,
                    running=False,
                    configuration_match=False,
                    probe_match=False,
                    verified=False,
                    reason=f"restoration verification raised: {type(exc).__name__}: {exc}",
                    measured_at=datetime.now(timezone.utc),
                )
            restoration = restoration.model_copy(update={"verification": verification})
            if (
                restoration.status is RestorationStatus.RESTORED
                and not verification.verified
            ):
                restoration = restoration.model_copy(
                    update={
                        "status": RestorationStatus.FAILED,
                        "verified": False,
                        "reason": (
                            f"{restoration.reason}; independent behavioural "
                            "verification failed"
                        ),
                    }
                )
        events.append(
            _event(
                ExperimentStage.RESTORATION_COMPLETED,
                restoration.status.value,
                verified=restoration.verified,
            )
        )

    # --- Final decision (restoration status included) ---
    if restoration is not None and restoration.status is RestorationStatus.FAILED:
        final_decision = decide(
            comparison=comparison,
            quality=quality,
            application=application,
            restoration_status=RestorationStatus.FAILED,
            measurement_valid=measurement_valid,
            precondition_ok=revalidation_ok,
            precondition_kind=revalidation_kind,
            precondition_reason=revalidation_reason,
            policy=request.policy,
            errors=errors,
        )
    elif provisional is not None:
        final_decision = provisional.model_copy(
            update={"restoration_status": restoration.status if restoration else RestorationStatus.PENDING}
        )
    else:
        final_decision = decide(
            comparison=None,
            quality=None,
            application=application,
            restoration_status=restoration.status if restoration else RestorationStatus.PENDING,
            measurement_valid=measurement_valid,
            precondition_ok=revalidation_ok,
            precondition_kind=revalidation_kind,
            precondition_reason=revalidation_reason,
            policy=request.policy,
            errors=errors,
        )

    total_s = time.perf_counter() - started
    overhead = ExperimentOverhead(
        baseline_setup_s=timings["baseline_setup_s"],
        application_s=timings["application_s"],
        measurement_s=timings["measurement_s"],
        validation_s=timings["validation_s"],
        restoration_s=timings["restoration_s"],
        total_s=total_s,
        notes=[
            "workload metrics come from the backend's own counters; this overhead is "
            "experiment orchestration only",
            "the optimizer itself is measured separately from the target model",
        ],
    )
    environment_restored = bool(restoration is not None and restoration.verified)
    summary = _summary(final_decision, comparison, quality, environment_restored)

    run_id = request.run_id or request.experiment_id
    provenance = build_provenance(
        request,
        run_id=run_id,
        source_report_path=request.source_report_path,
        source_report_hash=request.source_report_hash,
    )

    return ExperimentResult(
        experiment_id=request.experiment_id,
        run_id=run_id,
        provenance=provenance,
        candidate_id=request.candidate_id,
        action_id=request.action_id,
        parameters=request.parameters,
        created_at=datetime.now(timezone.utc),
        status=final_decision.kind,
        request=request,
        baseline_snapshot=baseline_snapshot,
        baseline_measurement=baseline_measurement,
        candidate_measurement=candidate_measurement,
        application=application,
        quality=quality,
        comparison=comparison,
        decision_detail=final_decision,
        restoration=restoration,
        events=events,
        overhead=overhead,
        errors=errors,
        executed=executed,
        environment_restored=environment_restored,
        baseline_observation_id=_first_run_id(baseline_measurement),
        result_observation_id=_first_run_id(candidate_measurement),
        performance_delta=_performance_delta(comparison),
        quality_before=_quality_before(quality),
        quality_after=_quality_after(quality),
        tolerance_band=request.policy.quality_similarity_min,
        decision=_legacy_decision(final_decision.kind),
        notes=summary,
        reproducibility=request.reproducibility,
        summary=summary,
    )


def _event(stage: ExperimentStage, detail: str = "", **data: object) -> ExperimentEvent:
    return ExperimentEvent(
        stage=stage, timestamp=datetime.now(timezone.utc), detail=detail, data=data
    )


def _capture_snapshot(
    request: ExperimentRequest,
    configuration: RuntimeConfiguration,
    measurement: ExperimentMeasurement,
) -> BaselineSnapshot:
    size, content_hash = _file_identity(request.model.path)
    reference = next((text for text in measurement.generated_texts if text), None)
    return BaselineSnapshot(
        snapshot_id=f"snap-{uuid.uuid4().hex[:10]}",
        configuration=configuration,
        model_ref=request.model.name,
        model_path=request.model.path,
        model_file_bytes=size,
        model_content_hash=content_hash,
        backend_name=request.backend.backend_name,
        backend_version=request.backend.backend_version,
        backend_type=request.backend.backend_type,
        device=request.backend.device,
        hardware_hostname=request.hardware.hostname,
        workload_id=request.workload.workload_id,
        workload_prompt_hash=request.workload.prompt_set_ref,
        seed=request.seed,
        baseline_run_ids=measurement.run_ids,
        metrics=measurement.metrics,
        quality_reference_text=reference,
        quality_reference_hash=(
            hashlib.sha256(" ".join(reference.split()).encode("utf-8")).hexdigest()[:16]
            if reference
            else None
        ),
    )


def _file_identity(path: str) -> tuple[int | None, str | None]:
    model_path = Path(path)
    if not model_path.is_file():
        return None, None
    try:
        size = model_path.stat().st_size
        digest = hashlib.sha256()
        digest.update(str(size).encode("ascii"))
        with model_path.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
            if size > 2 * 1024 * 1024:
                handle.seek(-1024 * 1024, os.SEEK_END)
                digest.update(handle.read(1024 * 1024))
        return size, digest.hexdigest()[:32]
    except OSError:  # pragma: no cover - defensive
        return None, None


def _first_run_id(measurement: ExperimentMeasurement | None) -> str | None:
    if measurement is None or not measurement.run_ids:
        return None
    return measurement.run_ids[0]


def _performance_delta(comparison: ExperimentComparison | None) -> dict[str, float]:
    if comparison is None:
        return {}
    delta: dict[str, float] = {}
    for metric, item in comparison.deltas.items():
        if item.relative_delta is not None:
            delta[metric] = item.relative_delta
        elif item.absolute_delta is not None:
            delta[metric] = item.absolute_delta
    return delta


def _quality_before(quality: QualityValidationResult | None) -> list[QualityScore]:
    if quality is None:
        return []
    return [
        QualityScore(
            metric="reference_available",
            value=1.0 if quality.reference_available else 0.0,
        )
    ]


def _quality_after(quality: QualityValidationResult | None) -> list[QualityScore]:
    if quality is None:
        return []
    scores = [
        QualityScore(metric="output_valid", value=1.0 if quality.validated else 0.0)
    ]
    if quality.similarity is not None:
        scores.append(QualityScore(metric="reference_similarity", value=quality.similarity))
    return scores


def _legacy_decision(kind: ExperimentDecisionKind) -> Decision:
    if kind is ExperimentDecisionKind.IMPROVED:
        return Decision.KEEP
    if kind in {
        ExperimentDecisionKind.REGRESSION,
        ExperimentDecisionKind.QUALITY_REGRESSION,
        ExperimentDecisionKind.RESTORATION_FAILURE,
    }:
        return Decision.REVERT
    if kind is ExperimentDecisionKind.NO_MEANINGFUL_IMPROVEMENT:
        return Decision.NO_OP
    return Decision.PENDING


def _summary(
    decision: ExperimentDecision,
    comparison: ExperimentComparison | None,
    quality: QualityValidationResult | None,
    restored: bool,
) -> str:
    return (
        f"Experiment {decision.kind.value}: {decision.reason} "
        f"(quality={quality.status.value if quality else 'not validated'}, "
        f"restored={restored}, confidence={decision.confidence:.2f})"
    )
