"""High-level Phase 7 service: Phase 6 candidate -> revalidated -> real experiment.

Ties the Phase 6 contracts/registry/policy to the Phase 7 engine so the CLI (and the
real acceptance) is thin and reproducible.
"""

from __future__ import annotations

import uuid

from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.experiment import (
    ExperimentPolicy,
    ExperimentRequest,
    ExperimentResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.optimization import CandidatePlan, OptimizationCandidate
from inferenceos.contracts.profiling import BaselineReport, ProfilingConfig
from inferenceos.contracts.reasoning import DeepDiagnosisOutcome
from inferenceos.experiment.eligibility import DEFAULT_TTL_S, revalidate_candidate
from inferenceos.experiment.engine import run_experiment
from inferenceos.experiment.executor import ActionExecutor, apply_to_configuration
from inferenceos.experiment.measurement import MeasurementHarness
from inferenceos.experiment.restoration import RestorationVerifier
from inferenceos.optimization.context import build_optimization_context
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.registry import ActionRegistry
from inferenceos.profiling.workloads import (
    BenchmarkWorkload,
    choose_context_length,
    get_workload,
)


class ExperimentInputError(ValueError):
    """Raised when a candidate cannot be turned into a runnable experiment."""


def baseline_configuration_from(
    config: ProfilingConfig, model: ModelProfile
) -> RuntimeConfiguration:
    context_length = config.context_length or choose_context_length(model)
    gpu_layers = (
        config.gpu_layers_offloaded if config.gpu_layers_offloaded is not None else 99
    )
    return RuntimeConfiguration(
        threads=config.threads,
        batch_size=config.batch_size,
        ubatch_size=config.ubatch_size,
        gpu_offload_layers=gpu_layers,
        use_mmap=config.use_mmap,
        context_length=context_length,
        flash_attention=config.flash_attention,
        kv_cache_type=config.kv_cache_type,
        # Use the backend-defined device *selector* (Phase 3 leaves it unset); the
        # resolved telemetry device name is not a valid backend selector.
        device=config.device_selector,
    )


def run_candidate_experiment(
    candidate: OptimizationCandidate,
    *,
    plan: CandidatePlan | None,
    diagnosis: BottleneckDiagnosis,
    report: BaselineReport | None,
    outcome: DeepDiagnosisOutcome | None,
    workload_id: str,
    registry: ActionRegistry,
    optimization_policy: OptimizationPolicy,
    experiment_policy: ExperimentPolicy,
    executor: ActionExecutor,
    profiler: MeasurementHarness,
    seed: int = 1234,
    ttl_s: float = DEFAULT_TTL_S,
    workload: BenchmarkWorkload | None = None,
    run_id: str | None = None,
    restoration_verifier: RestorationVerifier | None = None,
) -> ExperimentResult:
    workload = workload or get_workload(workload_id)
    context = build_optimization_context(
        diagnosis, report=report, outcome=outcome, policy=optimization_policy
    )
    if context.model is None:
        raise ExperimentInputError(
            "no model profile available from the diagnosis context; run a baseline first"
        )
    model = context.model

    revalidation = revalidate_candidate(
        candidate,
        plan=plan,
        context=context,
        registry=registry,
        policy=optimization_policy,
        executor=executor,
        ttl_s=ttl_s,
    )

    baseline_config = baseline_configuration_from(context.config, model)
    if revalidation.action is not None and executor.supports(revalidation.action.action_id):
        experiment_config = apply_to_configuration(
            revalidation.action.action_id, candidate.parameters, baseline_config
        )
    else:
        experiment_config = baseline_config

    workload_profile = context.workload or workload.to_workload_profile()
    request = ExperimentRequest(
        experiment_id=f"exp-{uuid.uuid4().hex[:12]}",
        run_id=run_id,
        candidate_id=candidate.candidate_id,
        action_id=candidate.action_id,
        parameters=dict(candidate.parameters),
        source_diagnosis_id=diagnosis.diagnosis_id,
        source_plan_id=plan.plan_id if plan is not None else None,
        source_bottleneck_id=candidate.source_bottleneck_id,
        diagnosed_bottleneck=candidate.diagnosis_kind.value,
        diagnosis_confidence=candidate.diagnosis_confidence,
        diagnosis_evidence_strength=candidate.diagnosis_strength,
        bottleneck_metric=candidate.benefit.metric,
        bottleneck_direction=candidate.benefit.direction,
        model=model,
        hardware=context.hardware,
        workload=workload_profile,
        backend=context.config,
        baseline_configuration=baseline_config,
        experiment_configuration=experiment_config,
        policy=experiment_policy,
        seed=seed,
        time_limit_s=experiment_policy.time_limit_s,
        source_report_path=diagnosis.context.source_report_path,
        reproducibility={
            "candidate_id": candidate.candidate_id,
            "action_id": candidate.action_id,
            "parameters": dict(candidate.parameters),
            "diagnosis_id": diagnosis.diagnosis_id,
            "plan_id": plan.plan_id if plan is not None else None,
            "model": {
                "path": model.path,
                "architecture": model.architecture,
                "quantization": model.quantization.value,
                "trained_context_length": model.trained_context_length,
            },
            "workload_id": workload_id,
            "seed": seed,
            "experiment_policy": experiment_policy.as_dict(),
            "bottleneck_metric": candidate.benefit.metric,
            "bottleneck_direction": candidate.benefit.direction,
        },
    )

    return run_experiment(
        request,
        executor=executor,
        profiler=profiler,
        workload=workload,
        action=revalidation.action,
        revalidation_ok=revalidation.ok,
        revalidation_kind=revalidation.kind,
        revalidation_reason=revalidation.reason,
        restoration_verifier=restoration_verifier,
    )
