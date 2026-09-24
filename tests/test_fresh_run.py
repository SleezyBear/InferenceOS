"""Fresh-run orchestration: run identity, provenance, stale-artifact resistance."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import LoopTerminationReason, MeasurementMode
from inferenceos.contracts.experiment import ExperimentPolicy
from inferenceos.contracts.fresh import (
    ArtifactKind,
    DeploymentStatus,
    FreshRunPolicy,
    RunArtifactRef,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.optimization_loop import LoopPolicy
from inferenceos.contracts.profiling import BaselineReport, ProfilingRun
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.loop.fresh import FreshRunOrchestrator, artifact_matches_run
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.selftest import sample_hardware, sample_model
from tests.diagnosis_fixtures import make_report, make_run
from tests.knowledge_helpers import make_service, temp_db_path


class FakeFreshProfiler:
    """Deterministic baseline profiler substitute (no real inference)."""

    def __init__(self, report: BaselineReport) -> None:
        self._report = report

    def make_tasks(
        self,
        models: Sequence[ModelProfile],
        workload_ids: Sequence[str] | None,
        context_override: int | None = None,
    ) -> list[ProfileTask]:
        return []

    def profile(
        self,
        tasks: Sequence[ProfileTask],
        *,
        modes: Sequence[MeasurementMode],
        repetitions: int = 3,
        options: ProfilerOptions | None = None,
    ) -> BaselineReport:
        return self._report

    def run_once(
        self,
        task: ProfileTask,
        *,
        mode: MeasurementMode,
        sequence: int,
        repetition: int,
        warmup: bool,
        options: ProfilerOptions,
    ) -> ProfilingRun:
        raise AssertionError("no experiment is expected in the no-op scenario")


def _healthy_report() -> BaselineReport:
    metrics = {
        "decode_tokens_per_sec": 24.0,
        "prompt_tokens_per_sec": 90.0,
        "total_latency_s": 1.4,
        "prefill_latency_s": 0.45,
        "decode_latency_s": 0.9,
        "model_load_time_s": 2.0,
        "process_cpu_percent": 120.0,
        "system_cpu_percent": 25.0,
        "gpu_utilization_percent": 70.0,
        "peak_rss_bytes": 1_300_000_000,
        "available_ram_bytes": 8 * 1024**3,
        "total_ram_bytes": 16 * 1024**3,
        "hard_page_faults_per_sec": 0.5,
        "prompt_tokens": 40,
        "generated_tokens": 24,
    }
    return make_report(
        [
            make_run(metrics=dict(metrics), condition="a"),
            make_run(metrics=dict(metrics), condition="b"),
        ]
    )


def _orchestrator(tmp_path: Path, run_id: str) -> tuple[FreshRunOrchestrator, Settings]:
    settings = Settings(
        reports_dir=tmp_path / "reports",
        data_dir=tmp_path / "data",
        logs_dir=tmp_path / "logs",
    )
    settings.ensure_directories()
    orchestrator = FreshRunOrchestrator(
        settings=settings,
        target=object(),  # type: ignore[arg-type]
        executor=RuntimeConfigurationExecutor(),
        registry=build_default_registry(),
        knowledge_service=make_service(temp_db_path()),
        loop_policy=LoopPolicy(),
        optimization_policy=OptimizationPolicy(),
        experiment_policy=ExperimentPolicy(),
        model_path=Path("synthetic.gguf"),
        policy=FreshRunPolicy(workload_ids=["short-short"], baseline_repetitions=2),
        run_id=run_id,
        hardware=sample_hardware(),
        model=sample_model(),
        profiler_factory=lambda: FakeFreshProfiler(_healthy_report()),
    )
    return orchestrator, settings


def test_fresh_noop_binds_run_and_deploys_no_change(tmp_path: Path) -> None:
    orchestrator, settings = _orchestrator(tmp_path, "phase9r-testrun")
    execution = orchestrator.run()
    result = execution.result
    loop = execution.loop
    assert loop.termination.reason in {
        LoopTerminationReason.NO_MEANINGFUL_OPPORTUNITY,
        LoopTerminationReason.INSUFFICIENT_EVIDENCE,
    }
    assert loop.run.budget.experiments_used == 0
    assert loop.changed is False
    assert loop.run.run_id == "phase9r-testrun"
    assert result.provenance.run_id == "phase9r-testrun"
    assert result.provenance.all_bound_to_run() is True
    assert Path(result.baseline_report_path or "").name.startswith("phase9r-testrun")
    assert result.deployment.status in {
        DeploymentStatus.NO_CANDIDATE,
        DeploymentStatus.NO_CHANGE,
    }
    assert result.characterization_s >= 0.0
    assert result.baseline_s > 0.0


def test_fresh_run_ignores_stale_baseline_artifact(tmp_path: Path) -> None:
    orchestrator, settings = _orchestrator(tmp_path, "phase9r-freshrun")
    decoy = settings.reports_dir / "phase9r_decoy_baseline.json"
    decoy.write_text('{"stale": true}', encoding="utf-8")
    before = decoy.read_bytes()
    execution = orchestrator.run()
    assert decoy.read_bytes() == before
    checks = execution.result.stale_artifact_checks
    assert any("phase9r_decoy_baseline.json" in note for note in checks)
    assert Path(execution.result.baseline_report_path or "").name.startswith("phase9r-freshrun")


def test_artifact_matches_run_helper() -> None:
    ref = RunArtifactRef(artifact_id="a", kind=ArtifactKind.BASELINE, run_id="run-x")
    assert artifact_matches_run(ref, "run-x") is True
    assert artifact_matches_run(ref, "run-y") is False


def test_loop_run_identity_is_propagated() -> None:
    from tests.loop_helpers import make_loop, no_improvement_responder

    stack = make_loop(
        responder=no_improvement_responder,
        report=_healthy_report(),
        run_id="phase9r-runid-test",
    )
    result = stack.loop.run(stack.report)
    assert result.run.run_id == "phase9r-runid-test"


def test_injected_deep_diagnosis_runner_is_used() -> None:
    from tests.loop_helpers import default_loop_policy, make_loop, no_improvement_responder
    from tests.reasoning_helpers import insufficient_report

    calls: list[str] = []

    def runner(diagnosis: BottleneckDiagnosis, report: BaselineReport) -> None:
        calls.append(diagnosis.diagnosis_id)
        return None

    stack = make_loop(
        responder=no_improvement_responder,
        report=insufficient_report(),
        loop_policy=default_loop_policy(allow_deep_diagnosis=True),
        deep_diagnosis_runner=runner,
    )
    stack.loop.run(stack.report)
    assert calls, "the injected resident deep-diagnosis runner must be invoked on escalation"


def test_prior_knowledge_does_not_substitute_fresh_evidence(tmp_path: Path) -> None:
    from tests.knowledge_helpers import base_result

    orchestrator, _settings = _orchestrator(tmp_path, "phase9r-know"
                                            )
    # Seed prior knowledge with an unrelated historical result.
    seeded = orchestrator.knowledge_service.persist_result(base_result())
    assert seeded.record is not None
    execution = orchestrator.run()
    result = execution.result
    # The run still produced its own fresh baseline and diagnosis.
    assert Path(result.baseline_report_path or "").name.startswith("phase9r-know")
    assert result.provenance.baseline_artifact_id == "phase9r-know-baseline"
    assert result.diagnosis_id is not None
    assert result.provenance.all_bound_to_run() is True
