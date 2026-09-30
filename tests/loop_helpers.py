"""Shared fixtures for Phase 9 autonomous-loop tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from inferenceos.contracts.enums import RestorationStatus
from inferenceos.contracts.experiment import (
    ExperimentPolicy,
    RestorationResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.optimization_loop import LoopPolicy, OptimizationBudget
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.interfaces.reasoning_engine import ReasoningEngine
from inferenceos.knowledge.service import KnowledgeService
from inferenceos.loop.engine import OptimizationLoop
from inferenceos.optimization.keep import ModeRules
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.reasoning.providers.base import ReasoningProvider
from tests.experiment_helpers import Responder, SampleResult, ScriptedHarness, build_test_policy
from tests.knowledge_helpers import temp_db_path
from tests.knowledge_helpers import make_service as make_knowledge_service
from tests.reasoning_helpers import paging_report


def patched_paging_report(
    *, use_mmap: bool = False, context_length: int = 2048
) -> BaselineReport:
    """A real paging report whose runs declare a known mmap/context configuration."""
    report = paging_report()
    runs = []
    for run in report.runs:
        config = run.context.config.model_copy(
            update={"use_mmap": use_mmap, "context_length": context_length}
        )
        runs.append(run.model_copy(update={"context": run.context.model_copy(update={"config": config})}))
    return BaselineReport(
        profiler_version=report.profiler_version,
        suite_version=report.suite_version,
        generated_at=report.generated_at,
        hardware=report.hardware,
        backend=runs[0].context.config,
        workloads=list(report.workloads),
        runs=runs,
        groups=[],
        cold_warm_semantics=report.cold_warm_semantics,
        resident_warm_supported=False,
        limitations=list(report.limitations),
        notes="phase9 test report",
    )


def report_for_config(base: BaselineReport, config: RuntimeConfiguration) -> BaselineReport:
    """Re-observe a report under a retained configuration (test observer)."""
    runs = []
    for run in base.runs:
        existing = run.context.config
        updated = existing.model_copy(
            update={
                "use_mmap": config.use_mmap,
                "context_length": config.context_length,
                "gpu_layers_offloaded": (
                    config.gpu_offload_layers
                    if config.gpu_offload_layers is not None
                    else existing.gpu_layers_offloaded
                ),
                "threads": config.threads if config.threads is not None else existing.threads,
            }
        )
        runs.append(run.model_copy(update={"context": run.context.model_copy(update={"config": updated})}))
    return BaselineReport(
        profiler_version=base.profiler_version,
        suite_version=base.suite_version,
        generated_at=base.generated_at,
        hardware=base.hardware,
        backend=runs[0].context.config,
        workloads=list(base.workloads),
        runs=runs,
        groups=[],
        cold_warm_semantics=base.cold_warm_semantics,
        resident_warm_supported=False,
        limitations=list(base.limitations),
        notes="phase9 re-observation",
    )


@dataclass
class LoopStack:
    loop: OptimizationLoop
    executor: RuntimeConfigurationExecutor
    harness: ScriptedHarness
    service: KnowledgeService
    report: BaselineReport


def default_loop_policy(**overrides: object) -> LoopPolicy:
    base: dict[str, object] = {
        "budget": OptimizationBudget(
            max_iterations=4,
            max_experiments=4,
            max_wall_clock_s=300.0,
            max_optimizer_overhead_s=120.0,
            max_deep_diagnoses=1,
        ),
        "skip_directly_negative": True,
        "reobserve_after_retention": True,
    }
    base.update(overrides)
    return LoopPolicy.model_validate(base)


def make_test_experiment_policy(**overrides: object) -> ExperimentPolicy:
    return build_test_policy(measured_repetitions=2, min_valid_samples=2, max_repetitions=3, **overrides)


class FaultyExecutor(RuntimeConfigurationExecutor):
    """Executor whose restore always fails (controlled restoration-failure seam)."""

    def restore(self, expected: RuntimeConfiguration) -> RestorationResult:
        return RestorationResult(
            attempted=True,
            status=RestorationStatus.FAILED,
            expected_configuration=expected,
            restored_configuration=self.current(),
            verified=False,
            reason="injected restoration failure",
            restored_at=datetime.now(timezone.utc),
        )


def make_loop(
    *,
    responder: Responder,
    report: BaselineReport | None = None,
    executor: RuntimeConfigurationExecutor | None = None,
    db_path: Path | None = None,
    loop_policy: LoopPolicy | None = None,
    knowledge_service: KnowledgeService | None = None,
    optimization_policy: OptimizationPolicy | None = None,
    experiment_policy: ExperimentPolicy | None = None,
    observer: Callable[[RuntimeConfiguration], BaselineReport | None] | None = None,
    reasoning_engine: ReasoningEngine | None = None,
    proposal_provider: ReasoningProvider | None = None,
    enable_reasoning_proposals: bool = False,
    deep_diagnosis_runner: Callable[..., object] | None = None,
    run_id: str | None = None,
    mode_rules: ModeRules | None = None,
) -> LoopStack:
    report = report or patched_paging_report()
    executor = executor or RuntimeConfigurationExecutor()
    harness = ScriptedHarness(responder)
    service = knowledge_service or make_knowledge_service(db_path or temp_db_path())
    loop = OptimizationLoop(
        executor=executor,
        profiler=harness,
        registry=build_default_registry(),
        optimization_policy=optimization_policy or OptimizationPolicy(),
        experiment_policy=experiment_policy or make_test_experiment_policy(),
        knowledge_service=service,
        loop_policy=loop_policy or default_loop_policy(),
        workload_id="short-short",
        observer=observer,
        reasoning_engine=reasoning_engine,
        proposal_provider=proposal_provider,
        enable_reasoning_proposals=enable_reasoning_proposals,
        deep_diagnosis_runner=deep_diagnosis_runner,  # type: ignore[arg-type]
        run_id=run_id,
        mode_rules=mode_rules,
    )
    return LoopStack(loop=loop, executor=executor, harness=harness, service=service, report=report)


def no_improvement_responder(task: ProfileTask, options: ProfilerOptions, call: int) -> SampleResult:
    return SampleResult(
        rss_bytes=1_300_000_000,
        hard_faults_per_sec=5.0,
        decode_tps=20.0,
        text="A key value cache stores attention state.",
    )


def context_improvement_responder(
    task: ProfileTask, options: ProfilerOptions, call: int
) -> SampleResult:
    # Keyed on the executable memory action (mmap), not context: context right-sizing
    # never generates a candidate and is never applied by the loop.
    if options.use_mmap:
        return SampleResult(
            rss_bytes=900_000_000,
            hard_faults_per_sec=1.0,
            decode_tps=28.0,
            text="A key value cache stores attention state.",
        )
    return SampleResult(
        rss_bytes=1_300_000_000,
        hard_faults_per_sec=5.0,
        decode_tps=20.0,
        text="A key value cache stores attention state.",
    )


def quality_regression_responder(
    task: ProfileTask, options: ProfilerOptions, call: int
) -> SampleResult:
    if options.use_mmap:
        return SampleResult(
            rss_bytes=900_000_000,
            hard_faults_per_sec=1.0,
            decode_tps=30.0,
            text="!!!! 1234 5678",
        )
    return SampleResult(
        rss_bytes=1_300_000_000,
        hard_faults_per_sec=5.0,
        decode_tps=20.0,
        text="A key value cache stores attention state.",
    )


__all__ = [
    "FaultyExecutor",
    "LoopStack",
    "context_improvement_responder",
    "default_loop_policy",
    "make_loop",
    "no_improvement_responder",
    "patched_paging_report",
    "quality_regression_responder",
    "report_for_config",
    "make_test_experiment_policy",
]
