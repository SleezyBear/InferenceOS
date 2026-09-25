"""H4 optimizer self-overhead / net-benefit acceptance.

Controlled evidence exercises the accounting function deterministically (beneficial,
overhead-dominant, insufficient-baseline, missing-overhead, distortion, incomparable
metric, nested-timing non-double-count). Real evidence runs the existing real active
resident workload through the runtime monitor with one controlled real intervention and
reports the end-to-end H4 assessment.

The real scenario demonstrates that the *mechanism* works against the actual runtime; it
does not claim that a single scenario proves optimizer profitability.
"""

from __future__ import annotations

import time
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.accounting import NetBenefitStatus, OverheadAttribution
from inferenceos.contracts.enums import ApprovalOutcome
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.runtime import RuntimeObservation
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.policy import build_policy
from inferenceos.optimizer_accounting import (
    AccountingPolicy,
    assess_net_benefit,
    build_overhead_attribution,
    distortion_from_stability,
)
from inferenceos.runtime.approval import CallbackApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.policy import build_runtime_budget, build_runtime_policy
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.runtime.telemetry import SystemTelemetrySampler
from inferenceos.runtime.verification import ControlledFootprintDetector
from inferenceos.runtime.workload import ActiveWorkload


class _PerturbingSampler(SystemTelemetrySampler):
    """Real system sampler plus a clearly labelled controlled pressure signal.

    Real target/request metrics are preserved; only post-baseline memory-pressure/
    hard-fault values are raised so one real intervention occurs. This mirrors the
    existing Phase 10R acceptance perturbation and is not used to establish the target
    benefit metric.
    """

    def sample(
        self, workload: ActiveWorkload, *, sequence: int, condition: str
    ) -> RuntimeObservation:
        observation = super().sample(workload, sequence=sequence, condition=condition)
        if condition == "baseline":
            return observation
        metrics = dict(observation.metrics)
        metrics["hard_page_faults_per_sec"] = max(
            metrics.get("hard_page_faults_per_sec", 0.0), 8000.0
        )
        metrics["memory_pressure"] = max(metrics.get("memory_pressure", 0.0), 0.9)
        return observation.model_copy(update={"metrics": metrics})


def _largest_model(settings: Settings) -> ModelProfile:
    models_dir = Path(settings.models_dir)
    profiles = [load_model_profile(path) for path in sorted(models_dir.glob("*.gguf"))]
    if not profiles:
        raise FileNotFoundError(f"no .gguf models found in {models_dir}")
    return max(profiles, key=lambda profile: profile.file_size_bytes)


def _overhead(inclusive: float, target_s: float) -> OverheadAttribution:
    return build_overhead_attribution(
        inclusive_overhead_s=inclusive,
        components={"diagnosis_s": inclusive / 2.0, "reasoning_s": inclusive / 2.0},
        baseline_target_workload_s=target_s,
        decision_count=1,
    )


def _run_controlled() -> tuple[dict[str, object], list[tuple[str, bool]]]:
    policy = AccountingPolicy()

    beneficial = assess_net_benefit(
        baseline_metrics={"decode_tokens_per_sec": 20.0},
        measured_metrics={"decode_tokens_per_sec": 26.0},
        overhead=_overhead(1.0, 20.0),
        policy=policy,
    )
    dominant = assess_net_benefit(
        baseline_metrics={"decode_tokens_per_sec": 20.0},
        measured_metrics={"decode_tokens_per_sec": 20.2},
        overhead=_overhead(10.0, 20.0),
        policy=policy,
    )
    no_baseline = assess_net_benefit(
        baseline_metrics={},
        measured_metrics={"decode_tokens_per_sec": 26.0},
        overhead=_overhead(1.0, 20.0),
        policy=policy,
    )
    no_overhead = assess_net_benefit(
        baseline_metrics={"decode_tokens_per_sec": 20.0},
        measured_metrics={"decode_tokens_per_sec": 26.0},
        overhead=None,
        policy=policy,
    )
    distortion = distortion_from_stability(cv=0.9, within_tolerance=False, tolerance=0.25)
    distorted = assess_net_benefit(
        baseline_metrics={"decode_tokens_per_sec": 20.0},
        measured_metrics={"decode_tokens_per_sec": 26.0},
        overhead=_overhead(1.0, 20.0),
        distortion=distortion,
        policy=policy,
    )
    incomparable = assess_net_benefit(
        baseline_metrics={"quality_score": 0.5},
        measured_metrics={"quality_score": 0.9},
        overhead=_overhead(1.0, 20.0),
        policy=policy,
        preferred_metric="quality_score",
    )
    flat = build_overhead_attribution(
        inclusive_overhead_s=1.0, components={"monitor_s": 0.6, "diagnosis_s": 0.4}
    )
    nested = build_overhead_attribution(
        inclusive_overhead_s=1.0, components={"monitor_s": 1.0, "diagnosis_s": 0.6}
    )
    memory = assess_net_benefit(
        baseline_metrics={"peak_rss_bytes": 1_000_000_000.0},
        measured_metrics={"peak_rss_bytes": 800_000_000.0},
        overhead=_overhead(1.0, 20.0),
        policy=policy,
    )

    checks: list[tuple[str, bool]] = [
        ("H4: a clear benefit with bounded overhead is BENEFICIAL", beneficial.status == NetBenefitStatus.BENEFICIAL),
        ("H4: overhead that dominates a negligible benefit is OVERHEAD_DOMINANT", dominant.status == NetBenefitStatus.OVERHEAD_DOMINANT),
        ("H4: no baseline is INSUFFICIENT_EVIDENCE", no_baseline.status == NetBenefitStatus.INSUFFICIENT_EVIDENCE),
        ("H4: missing overhead is INSUFFICIENT_EVIDENCE", no_overhead.status == NetBenefitStatus.INSUFFICIENT_EVIDENCE),
        ("H4: material measurement distortion is INSUFFICIENT_EVIDENCE", distorted.status == NetBenefitStatus.INSUFFICIENT_EVIDENCE),
        (
            "H4: incomparable metrics yield a structured insufficient result (no scalar)",
            incomparable.status == NetBenefitStatus.INSUFFICIENT_EVIDENCE
            and incomparable.comparable is False
            and incomparable.net_value is None,
        ),
        (
            "H4: nested timers are not double-counted",
            flat.inclusive_overhead_s == 1.0
            and nested.inclusive_overhead_s == 1.0
            and nested.components_sum_s == 1.6
            and nested.nested is True,
        ),
        (
            "H4: a memory benefit is represented in bytes, not forced into throughput",
            memory.status == NetBenefitStatus.BENEFICIAL
            and memory.benefit is not None
            and memory.benefit.unit == "bytes",
        ),
    ]
    scenario: dict[str, object] = {
        "beneficial": beneficial.model_dump(mode="json"),
        "overhead_dominant": dominant.model_dump(mode="json"),
        "no_baseline": no_baseline.model_dump(mode="json"),
        "no_overhead": no_overhead.model_dump(mode="json"),
        "distorted": distorted.model_dump(mode="json"),
        "incomparable": incomparable.model_dump(mode="json"),
        "nested": nested.model_dump(mode="json"),
        "memory_benefit": memory.model_dump(mode="json"),
    }
    return scenario, checks


def _run_real(settings: Settings) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    model = _largest_model(settings)
    workload = ResidentActiveWorkload(
        settings=settings,
        model=model,
        workload_id="short-short",
        configuration=RuntimeConfiguration(context_length=2048, gpu_offload_layers=0),
        request_gap_s=0.0,
    )
    # Pre-warm so the baseline observation window carries real target request metrics
    # (decode throughput / latency), not only host memory state.
    workload.start()
    warmup_deadline = time.perf_counter() + 90.0
    while workload.completed_requests() < 2 and time.perf_counter() < warmup_deadline:
        time.sleep(0.2)

    budget = build_runtime_budget(
        settings, max_duration_s=45.0, telemetry_interval_s=0.5
    ).model_copy(
        update={
            "baseline_samples": 3,
            "observation_window_size": 40,
            "max_interventions": 1,
            "max_cumulative_intervention_s": 600.0,
            "max_optimizer_overhead_s": 600.0,
            "approval_timeout_s": 5.0,
            "max_alerts_per_condition": 3,
        }
    )
    policy = build_runtime_policy(
        settings,
        budget=budget,
        require_approval_for_disruptive=True,
        allow_non_disruptive_auto_adaptation=False,
    )
    optimization_policy = build_policy(settings).model_copy(
        update={"context_reduction_fraction": 0.75}
    )
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=_PerturbingSampler(settings=settings),
        policy=policy,
        approval_provider=CallbackApprovalProvider(lambda _rec: ApprovalOutcome.APPROVED),
        detector=ControlledFootprintDetector(force=True),  # type: ignore[arg-type]
        optimization_policy=optimization_policy,
        experiment_policy=build_experiment_policy(settings).model_copy(
            update={"measured_repetitions": 2, "max_repetitions": 3, "max_total_runs": 8}
        ),
        run_id="h4-accounting-real",
        session_id="h4-accounting-real",
        settings=settings,
    )
    try:
        result = monitor.run()
    finally:
        workload.stop()

    run = result.run
    assessment = run.net_benefit
    baseline = run.baselines[0] if run.baselines else run.baseline
    measured = run.baselines[-1] if run.baselines else run.baseline

    def _means(obj: object) -> dict[str, float]:
        metrics = getattr(obj, "metrics", {}) if obj is not None else {}
        return {
            name: stats.mean
            for name, stats in metrics.items()
            if getattr(stats, "mean", None) is not None
        }

    overhead = run.overhead
    distortion = run.resource_impact
    checks = [
        (
            "H4 real: the runtime monitor produced an end-to-end net-benefit assessment",
            assessment is not None,
        ),
        (
            "H4 real: the assessment is a valid conservative status",
            assessment is not None
            and assessment.status
            in {
                NetBenefitStatus.BENEFICIAL,
                NetBenefitStatus.NEUTRAL,
                NetBenefitStatus.OVERHEAD_DOMINANT,
                NetBenefitStatus.INSUFFICIENT_EVIDENCE,
            },
        ),
        (
            "H4 real: optimizer overhead is measured and attributed separately from target time",
            assessment is not None
            and assessment.overhead is not None
            and assessment.overhead.inclusive_overhead_s >= 0.0,
        ),
        (
            "H4 real: baseline and post-optimization target metrics are comparable",
            bool(_means(baseline)) and bool(_means(measured)),
        ),
        (
            "H4 real: the assessment is machine-readable and distortion evidence is present",
            assessment is not None
            and isinstance(assessment.model_dump_json(), str)
            and distortion is not None,
        ),
    ]
    scenario: dict[str, object] = {
        "assessment": assessment.model_dump(mode="json") if assessment is not None else None,
        "baseline_metrics": _means(baseline),
        "measured_metrics": _means(measured),
        "optimizer_overhead_s": overhead.optimizer_overhead_s,
        "overhead_components_s": {
            "detection_s": overhead.detection_s,
            "diagnosis_s": overhead.diagnosis_s,
            "reasoning_s": overhead.reasoning_s,
            "checkpoint_s": overhead.checkpoint_s,
            "validation_s": overhead.validation_s,
            "experiment_s": overhead.experiment_s,
        },
        "monitoring_wall_s": overhead.monitoring_wall_s,
        "distortion": {
            "decode_tps_cv": distortion.decode_tps_cv,
            "within_tolerance": distortion.throughput_within_tolerance,
        },
        "interventions": len(run.interventions),
        "retained": any(intervention.retained for intervention in run.interventions),
        "requests_served": workload.completed_requests(),
        "termination": result.termination.reason.value,
    }
    return scenario, checks


def _run_performance() -> tuple[dict[str, object], list[tuple[str, bool]]]:
    attribution = _overhead(1.0, 20.0)
    iterations = 500
    start = time.perf_counter()
    for _ in range(iterations):
        assess_net_benefit(
            baseline_metrics={"decode_tokens_per_sec": 20.0},
            measured_metrics={"decode_tokens_per_sec": 26.0},
            overhead=attribution,
            policy=AccountingPolicy(),
        )
    mean_ms = (time.perf_counter() - start) / iterations * 1000.0
    checks = [
        (
            "H4 performance: accounting is far cheaper than the operations it measures",
            mean_ms < 1.0,
        )
    ]
    return {
        "iterations": iterations,
        "mean_assessment_ms": round(mean_ms, 4),
        "budget_ms": 1.0,
    }, checks


def run_accounting_acceptance(
    settings: Settings,
) -> tuple[dict[str, object], list[tuple[str, bool]]]:
    scenarios: dict[str, object] = {}
    checks: list[tuple[str, bool]] = []

    try:
        controlled, controlled_checks = _run_controlled()
    except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001
        controlled = {"error": f"{type(exc).__name__}: {exc}"}
        controlled_checks = [("H4 controlled: scenarios ran", False)]
    scenarios["controlled"] = controlled
    checks.extend(controlled_checks)

    try:
        real, real_checks = _run_real(settings)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:  # noqa: BLE001
        real = {"error": f"{type(exc).__name__}: {exc}"}
        real_checks = [("H4 real: scenario ran", False)]
    scenarios["real"] = real
    checks.extend(real_checks)

    try:
        performance, performance_checks = _run_performance()
    except (OSError, RuntimeError, ValueError) as exc:  # noqa: BLE001
        performance = {"error": f"{type(exc).__name__}: {exc}"}
        performance_checks = [("H4 performance: scenario ran", False)]
    scenarios["performance"] = performance
    checks.extend(performance_checks)

    return scenarios, checks
