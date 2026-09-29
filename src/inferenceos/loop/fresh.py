"""Phase 9R fresh-run orchestrator.

A fresh run independently performs characterization -> model introspection -> fresh
baseline -> (detector or controlled diagnosis) -> the existing Phase 9 optimization
loop, and binds every run-critical artifact to a unique run identity with provenance.
It never consumes a pre-existing Phase 9 report or diagnosis to establish its state.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.config import Settings
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    ExperimentDecisionKind,
    LoopTerminationReason,
    MeasurementMode,
)
from inferenceos.contracts.experiment import ExperimentPolicy, RuntimeConfiguration
from inferenceos.contracts.fresh import (
    ArtifactKind,
    DeploymentState,
    DeploymentStatus,
    FreshRunPolicy,
    FreshRunProvenance,
    FreshRunResult,
    RunArtifactRef,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.optimization_loop import LoopPolicy, LoopResult
from inferenceos.contracts.profiling import BaselineReport, ProfilingRun
from inferenceos.contracts.reasoning import DeepDiagnosisOutcome
from inferenceos.experiment.executor import ActionExecutor
from inferenceos.hardware.profiler import detect_hardware_profile
from inferenceos.knowledge.service import KnowledgeService
from inferenceos.logging_config import get_logger
from inferenceos.loop.engine import OptimizationLoop
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.registry import ActionRegistry
from inferenceos.profiling.runner import BaselineProfiler, ProfilerOptions, ProfileTask

logger = get_logger("loop.fresh")

DeepDiagnosisRunner = Callable[
    [BottleneckDiagnosis, BaselineReport], DeepDiagnosisOutcome | None
]
DiagnosisFactory = Callable[
    [BaselineReport, HardwareProfile, ModelProfile], BottleneckDiagnosis
]
Observer = Callable[[RuntimeConfiguration], BaselineReport | None]


class BaselineProfilerLike(Protocol):
    """The subset of the Phase 3 profiler a fresh run relies on (injectable)."""

    def make_tasks(
        self,
        models: Sequence[ModelProfile],
        workload_ids: Sequence[str] | None,
        context_override: int | None = None,
    ) -> Sequence[ProfileTask]: ...

    def profile(
        self,
        tasks: Sequence[ProfileTask],
        *,
        modes: Sequence[MeasurementMode],
        repetitions: int = 3,
        options: ProfilerOptions | None = None,
    ) -> BaselineReport: ...

    def run_once(
        self,
        task: ProfileTask,
        *,
        mode: MeasurementMode,
        sequence: int,
        repetition: int,
        warmup: bool,
        options: ProfilerOptions,
    ) -> ProfilingRun: ...


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:32]


def artifact_matches_run(artifact: RunArtifactRef, run_id: str) -> bool:
    return artifact.run_id == run_id


@dataclass
class FreshRunExecution:
    """Fresh-run result plus the underlying loop result (both run-bound)."""

    result: FreshRunResult
    loop: LoopResult


@dataclass
class FreshRunOrchestrator:
    """Drives the complete pipeline from a fresh target/model state."""

    settings: Settings
    target: LlamaCppResidentAdapter
    executor: ActionExecutor
    registry: ActionRegistry
    knowledge_service: KnowledgeService
    loop_policy: LoopPolicy
    optimization_policy: OptimizationPolicy
    experiment_policy: ExperimentPolicy
    model_path: Path
    policy: FreshRunPolicy = field(default_factory=FreshRunPolicy)
    run_id: str | None = None
    diagnosis_factory: DiagnosisFactory | None = None
    deep_diagnosis_runner: DeepDiagnosisRunner | None = None
    observer: Observer | None = None
    hardware: HardwareProfile | None = None
    model: ModelProfile | None = None
    profiler_factory: Callable[[], BaselineProfilerLike] | None = None
    profiler_options: ProfilerOptions | None = None

    def run(self) -> FreshRunExecution:
        run_id = self.run_id or (
            f"phase9r-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:6]}"
        )
        artifacts: list[RunArtifactRef] = []
        checks: list[str] = []

        # --- CHARACTERIZE ------------------------------------------------- #
        t0 = time.perf_counter()
        hardware = self.hardware or detect_hardware_profile(self.settings)
        characterization_s = time.perf_counter() - t0
        hardware_hash = _hash(hardware.model_dump(mode="json"))
        artifacts.append(
            RunArtifactRef(
                artifact_id=f"{run_id}-hardware",
                kind=ArtifactKind.CHARACTERIZATION,
                run_id=run_id,
                content_hash=hardware_hash,
                notes=["fresh hardware/system characterization from this run"],
            )
        )
        checks.append(
            f"characterization produced by this run (host={hardware.hostname!r}, "
            f"hash={hardware_hash})"
        )

        # --- INTROSPECT --------------------------------------------------- #
        t0 = time.perf_counter()
        model = self.model or load_model_profile(self.model_path)
        introspection_s = time.perf_counter() - t0
        model_hash = _hash(model.model_dump(mode="json"))
        artifacts.append(
            RunArtifactRef(
                artifact_id=f"{run_id}-model",
                kind=ArtifactKind.MODEL_INTROSPECTION,
                run_id=run_id,
                content_hash=model_hash,
                notes=["fresh GGUF introspection from this run"],
            )
        )
        checks.append(f"model introspected by this run (hash={model_hash})")

        # --- FRESH BASELINE ----------------------------------------------- #
        t0 = time.perf_counter()
        profiler: BaselineProfilerLike = (
            self.profiler_factory()
            if self.profiler_factory is not None
            else BaselineProfiler(settings=self.settings, adapter=self.target)
        )
        tasks = profiler.make_tasks(
            [model], self.policy.workload_ids, self.policy.context_override
        )
        if self.policy.context_requirement is not None:
            # Declare the workload's hard context requirement for this run (the only
            # authoritative safety bound a context optimizer may rely on).
            tasks = [
                replace(
                    task,
                    workload=replace(
                        task.workload,
                        context_requirement=self.policy.context_requirement,
                    ),
                )
                for task in tasks
            ]
        report = profiler.profile(
            tasks,
            modes=list(self.policy.modes),
            repetitions=self.policy.baseline_repetitions,
            options=self.profiler_options,
        )
        baseline_s = time.perf_counter() - t0
        baseline_path = self.settings.reports_dir / f"{run_id}_baseline.json"
        baseline_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        baseline_hash = _file_hash(baseline_path)
        baseline_artifact_id = f"{run_id}-baseline"
        artifacts.append(
            RunArtifactRef(
                artifact_id=baseline_artifact_id,
                kind=ArtifactKind.BASELINE,
                run_id=run_id,
                content_hash=baseline_hash,
                path=str(baseline_path),
                notes=["fresh baseline measured during this run and persisted run-scoped"],
            )
        )
        checks.append(
            f"baseline measured by this run ({len(report.runs)} run(s)) at "
            f"{baseline_path.name}"
        )
        if self.policy.require_fresh_baseline and not report.runs:
            raise RuntimeError("fresh baseline produced no runs")

        # Stale-artifact resistance: any other run-tagged baseline is explicitly
        # ignored and recorded (never consumed as current evidence).
        for candidate in sorted(self.settings.reports_dir.glob("*_baseline.json")):
            if candidate.name.startswith(run_id):
                continue
            checks.append(
                f"ignored stale baseline artifact {candidate.name!r} (not this run)"
            )

        # --- DIAGNOSIS ---------------------------------------------------- #
        controlled = False
        diagnosis: BottleneckDiagnosis | None = None
        if self.diagnosis_factory is not None:
            diagnosis = self.diagnosis_factory(report, hardware, model)
            controlled = True
            artifacts.append(
                RunArtifactRef(
                    artifact_id=f"{run_id}-diagnosis",
                    kind=ArtifactKind.DIAGNOSIS,
                    run_id=run_id,
                    content_hash=_hash(diagnosis.model_dump(mode="json")),
                    notes=["controlled diagnosis grounded in this run's fresh baseline"],
                )
            )
            checks.append(
                f"controlled diagnosis {diagnosis.diagnosis_id} grounded in fresh "
                "baseline measurements"
            )

        # --- OPTIMIZATION LOOP -------------------------------------------- #
        loop = OptimizationLoop(
            executor=self.executor,
            profiler=profiler,
            registry=self.registry,
            optimization_policy=self.optimization_policy,
            experiment_policy=self.experiment_policy,
            knowledge_service=self.knowledge_service,
            loop_policy=self.loop_policy,
            observer=self.observer,
            deep_diagnosis_runner=self.deep_diagnosis_runner,
            run_id=run_id,
        )
        loop_result = loop.run(
            report, diagnosis=diagnosis, report_path=str(baseline_path)
        )

        # --- PROVENANCE + DEPLOYMENT -------------------------------------- #
        provenance, extra_artifacts, provenance_checks = _build_provenance(
            run_id, loop_result, report, baseline_path, diagnosis
        )
        artifacts.extend(extra_artifacts)
        provenance = provenance.model_copy(update={"artifacts": artifacts})
        checks.extend(provenance_checks)
        deployment = _build_deployment(run_id, loop_result, model, provenance)
        summary = (
            f"Fresh run {run_id}: {loop_result.termination.reason.value} "
            f"(deployment={deployment.status.value}, changed={loop_result.changed})"
        )
        result = FreshRunResult(
            run_id=run_id,
            policy=self.policy,
            hardware_summary={
                "hostname": hardware.hostname,
                "physical_cores": hardware.cpu.physical_cores,
                "logical_cores": hardware.cpu.logical_cores,
                "total_ram_bytes": hardware.memory.total_bytes,
                "backends": [b.value for b in hardware.backends_available],
            },
            model_summary={
                "name": model.name,
                "architecture": model.architecture,
                "quantization": model.quantization.value,
                "path": model.path,
            },
            baseline_report_path=str(baseline_path),
            baseline_artifact_id=baseline_artifact_id,
            diagnosis_id=provenance.diagnosis_id,
            diagnosis_controlled=controlled,
            provenance=provenance,
            deployment=deployment,
            characterization_s=characterization_s,
            introspection_s=introspection_s,
            baseline_s=baseline_s,
            stale_artifact_checks=checks,
            summary=summary,
        )
        return FreshRunExecution(result=result, loop=loop_result)


def _build_provenance(
    run_id: str,
    loop_result: LoopResult,
    report: BaselineReport,
    baseline_path: Path,
    diagnosis: BottleneckDiagnosis | None,
) -> tuple[FreshRunProvenance, list[RunArtifactRef], list[str]]:
    artifacts: list[RunArtifactRef] = []
    checks: list[str] = []
    experiment_ids: list[str] = []
    knowledge_ids: list[str] = []
    reasoning_ids: list[str] = []
    plan_id: str | None = None
    diagnosis_id = diagnosis.diagnosis_id if diagnosis is not None else None

    for iteration in loop_result.run.iterations:
        if iteration.diagnosis_id:
            diagnosis_id = diagnosis_id or iteration.diagnosis_id
        if iteration.deep_diagnosis_outcome_id:
            reasoning_ids.append(iteration.deep_diagnosis_outcome_id)
            artifacts.append(
                RunArtifactRef(
                    artifact_id=f"{run_id}-reasoning-{iteration.deep_diagnosis_outcome_id}",
                    kind=ArtifactKind.REASONING,
                    run_id=run_id,
                    notes=["deep reasoning bound to this run"],
                )
            )
        if iteration.plan_id:
            plan_id = plan_id or iteration.plan_id
            artifacts.append(
                RunArtifactRef(
                    artifact_id=f"{run_id}-plan-{iteration.plan_id}",
                    kind=ArtifactKind.CANDIDATE_PLAN,
                    run_id=run_id,
                    notes=["candidate plan generated during this run"],
                )
            )
        if iteration.experiment_id:
            experiment_ids.append(iteration.experiment_id)
            knowledge_ids.append(f"opt-{iteration.experiment_id}")
            artifacts.append(
                RunArtifactRef(
                    artifact_id=f"{run_id}-experiment-{iteration.experiment_id}",
                    kind=ArtifactKind.EXPERIMENT,
                    run_id=run_id,
                    notes=[f"experiment status={iteration.experiment_status}"],
                )
            )
    report_hash = _file_hash(baseline_path)
    provenance = FreshRunProvenance(
        run_id=run_id,
        hardware_hash=_hash(report.hardware.model_dump(mode="json")),
        model_hash=_hash(report.runs[0].context.model.model_dump(mode="json"))
        if report.runs
        else None,
        backend_key=report.backend.backend_name,
        baseline_artifact_id=f"{run_id}-baseline",
        diagnosis_id=diagnosis_id,
        candidate_plan_id=plan_id,
        experiment_ids=experiment_ids,
        knowledge_record_ids=knowledge_ids,
        reasoning_outcome_ids=reasoning_ids,
        artifacts=[],
        notes=[f"baseline_hash={report_hash}"],
    )
    bound = all(a.run_id == run_id for a in artifacts)
    checks.append(
        f"all run-critical artifacts bound to run {run_id}: {bound}"
    )
    return provenance, artifacts, checks


def _build_deployment(
    run_id: str,
    loop_result: LoopResult,
    model: ModelProfile,
    provenance: FreshRunProvenance,
) -> DeploymentState:
    termination = loop_result.termination.reason
    if termination in (
        LoopTerminationReason.RESTORATION_FAILURE,
        LoopTerminationReason.QUALITY_SAFETY_STOP,
    ):
        status = DeploymentStatus.SAFETY_STOP
    elif loop_result.changed:
        status = DeploymentStatus.RETAINED
    elif any(
        iteration.outcome.value
        in {"candidate_reverted", "candidate_rejected"}
        or iteration.experiment_status is ExperimentDecisionKind.QUALITY_REGRESSION
        for iteration in loop_result.run.iterations
    ):
        status = DeploymentStatus.CANDIDATE_REJECTED
    elif termination in (
        LoopTerminationReason.NO_MEANINGFUL_OPPORTUNITY,
        LoopTerminationReason.INSUFFICIENT_EVIDENCE,
        LoopTerminationReason.NO_ELIGIBLE_CANDIDATES,
        LoopTerminationReason.ALL_CANDIDATES_REJECTED,
    ):
        status = DeploymentStatus.NO_CANDIDATE
    else:
        status = DeploymentStatus.NO_CHANGE

    source_experiment = next(
        (
            iteration.experiment_id
            for iteration in reversed(loop_result.run.iterations)
            if iteration.retained and iteration.experiment_id
        ),
        None,
    )
    return DeploymentState(
        run_id=run_id,
        status=status,
        model_ref=model.name,
        model_path=model.path,
        backend_name=provenance.backend_key,
        baseline_configuration=loop_result.initial_configuration,
        final_configuration=loop_result.final_configuration,
        retained=loop_result.changed,
        retained_candidate_ids=list(loop_result.run.retained_candidate_ids),
        source_experiment_id=source_experiment,
        provenance_hash=_hash(provenance.model_dump(mode="json")),
        notes=[
            "recommended/retained state is the final configuration; no server launcher is "
            "started by Phase 9R",
            f"termination={termination.value}",
        ],
    )
