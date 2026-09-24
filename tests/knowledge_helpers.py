"""Shared fixtures for Phase 8 persistent-knowledge tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.enums import ExperimentDecisionKind
from inferenceos.contracts.experiment import ExperimentResult, RuntimeConfiguration
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.knowledge import ApplicabilityContext, OptimizationRecord
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import ProfilingConfig
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.knowledge import (
    KnowledgePolicy,
    KnowledgeService,
    SQLiteKnowledgeStore,
    build_current_context,
    build_knowledge_policy,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.selftest import (
    sample_experiment_result,
    sample_hardware,
    sample_model,
    sample_runtime_configuration,
    sample_workload,
)


def temp_db_path(name: str = "knowledge.db") -> Path:
    directory = Path(tempfile.mkdtemp(prefix="io_knowledge_"))
    return directory / name


def make_service(
    db_path: Path | None = None, *, policy: KnowledgePolicy | None = None
) -> KnowledgeService:
    path = db_path or temp_db_path()
    return KnowledgeService(
        store=SQLiteKnowledgeStore(path),
        registry=build_default_registry(),
        policy=policy or build_knowledge_policy(Settings()),
    )


def base_result() -> ExperimentResult:
    return sample_experiment_result()


def record_for(result: ExperimentResult, service: KnowledgeService) -> OptimizationRecord:
    return service.build_record(result)


def with_decision(result: ExperimentResult, kind: ExperimentDecisionKind) -> ExperimentResult:
    decision = result.decision_detail.model_copy(update={"kind": kind}) if result.decision_detail else None
    return result.model_copy(update={"status": kind, "decision_detail": decision})


def current_context(
    record: OptimizationRecord,
    *,
    hardware: HardwareProfile | None = None,
    model: ModelProfile | None = None,
    include_model: bool = True,
    workload: WorkloadProfile | None = None,
    backend: ProfilingConfig | None = None,
    runtime: RuntimeConfiguration | None = None,
    diagnosis: str | None = None,
    conditions: dict[str, float] | None = None,
) -> ApplicabilityContext:
    chosen_model = record.context.model if model is None and include_model else model
    return build_current_context(
        hardware=hardware if hardware is not None else record.context.hardware,
        model=chosen_model,
        workload=workload if workload is not None else record.context.workload,
        backend=backend if backend is not None else record.context.backend,
        runtime=runtime if runtime is not None else record.context.runtime_configuration,
        diagnosis=diagnosis if diagnosis is not None else record.diagnosed_bottleneck,
        conditions=conditions if conditions is not None else record.context.fingerprint.conditions,
    )


def altered_hardware() -> HardwareProfile:
    hardware = sample_hardware()
    cpu = hardware.cpu.model_copy(
        update={
            "physical_cores": hardware.cpu.physical_cores + 4,
            "logical_cores": hardware.cpu.logical_cores + 8,
        }
    )
    return hardware.model_copy(update={"cpu": cpu})


def altered_model() -> ModelProfile:
    model = sample_model()
    return model.model_copy(update={"n_layers": model.n_layers + 8})


def other_workload() -> WorkloadProfile:
    workload = sample_workload()
    return workload.model_copy(update={"workload_id": "long-short"})


__all__ = [
    "altered_hardware",
    "altered_model",
    "base_result",
    "current_context",
    "make_service",
    "other_workload",
    "record_for",
    "sample_runtime_configuration",
    "temp_db_path",
    "with_decision",
]
