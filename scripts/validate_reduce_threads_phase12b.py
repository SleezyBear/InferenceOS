"""Phase 12B — bounded end-to-end validation of `runtime.reduce_threads`.

Uses the EXISTING candidate preparation, robust Phase 7 measurement/quality lifecycle,
economics, and resident control path. It does NOT search thread counts and does NOT
modify production code; if the existing lifecycle cannot represent the observed
effective-thread baseline, it reports the exact integration gap.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\validate_reduce_threads_phase12b.py \\
        --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ab_effectiveness as ab  # noqa: E402
import validate_cpu_thread_surface as v  # noqa: E402

from inferenceos.config import Settings  # noqa: E402
from inferenceos.contracts.enums import (  # noqa: E402
    AnomalyClass,
    EvidenceStrength,
    PerformanceState,
)
from inferenceos.contracts.experiment import RuntimeConfiguration  # noqa: E402
from inferenceos.contracts.optimization import CandidatePlan  # noqa: E402
from inferenceos.contracts.profiling import BaselineReport, ProfilingConfig  # noqa: E402
from inferenceos.contracts.runtime import RuntimeOpportunity  # noqa: E402
from inferenceos.diagnosis.detector import BottleneckDetector  # noqa: E402
from inferenceos.experiment.policy import build_experiment_policy  # noqa: E402
from inferenceos.experiment.service import (  # noqa: E402
    baseline_configuration_from,
    run_candidate_experiment,
)
from inferenceos.external.controlled import launch_cpu_spinner  # noqa: E402
from inferenceos.hardware.profiler import detect_hardware_profile  # noqa: E402
from inferenceos.models.loader import load_model_profile  # noqa: E402
from inferenceos.optimization.actions import build_default_registry  # noqa: E402
from inferenceos.optimization.generator import generate_candidates  # noqa: E402
from inferenceos.optimization.policy import build_policy  # noqa: E402
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter  # noqa: E402
from inferenceos.profiling.runner import BaselineProfiler  # noqa: E402
from inferenceos.runtime.economics import (  # noqa: E402
    evaluate_intervention_economics,
    evaluate_retention_economics,
)
from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter  # noqa: E402
from inferenceos.optimization.preference import (  # noqa: E402
    build_optimization_preference,
    classify_performance_state,
)
from inferenceos.runtime.policy import build_runtime_policy  # noqa: E402
from inferenceos.runtime.resident_workload import ResidentActiveWorkload  # noqa: E402
from inferenceos.runtime.workload import RuntimeWorkloadExecutor  # noqa: E402


def _patch_effective(report: BaselineReport, effective: int) -> BaselineReport:
    """Represent the report as the runtime path does: ``threads`` unset, the requested
    value carried in ``requested_threads``, and the observed value in
    ``effective_threads``. No manual baseline override is supplied."""
    runs = []
    for run in report.runs:
        config = run.context.config.model_copy(
            update={
                "threads": None,
                "requested_threads": run.context.config.threads,
                "effective_threads": effective,
            }
        )
        runs.append(run.model_copy(update={"context": run.context.model_copy(update={"config": config})}))
    return report.model_copy(update={"runs": runs})


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--spinners", type=int, default=4)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_directories()
    model = load_model_profile(Path(args.model))
    hardware = detect_hardware_profile(settings)
    physical = hardware.cpu.physical_cores
    logical = hardware.cpu.logical_cores
    server_path = settings.llama_cpp_dir / "llama-server.exe"
    total_layers = model.n_layers
    requests = ab.build_fixed_requests(
        rounds=args.rounds, n_predict=args.n_predict, seed=args.seed, temperature=0.0, ignore_eos=False
    )
    workload_profile = requests[0].to_benchmark_workload().to_workload_profile()
    registry = build_default_registry()
    optimization_policy = build_policy(settings)
    experiment_policy = build_experiment_policy(settings)
    runtime_policy = build_runtime_policy(settings)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, Any] = {
        "schema": "reduce_threads_phase12b.v1",
        "timestamp": timestamp,
        "hardware": {"physical_cores": physical, "logical_cores": logical, "cpu": hardware.cpu.name},
        "requested_threads": logical,
    }
    print(f"physical={physical} logical={logical} layers={total_layers}")

    # ------------------------------------------------------------------ #
    # A/B. Real CPU diagnosis (offload not configured) + candidate prep    #
    # ------------------------------------------------------------------ #
    print("--- real CPU diagnosis (CPU-only, observed effective threads) ---")
    control, control_meta, _ = v.measure_condition(
        requests=requests, model=model, workload=workload_profile, hardware=hardware,
        server_path=server_path, settings=settings, scenario="p12b", condition="control",
        gpu_layers=0, threads=logical, total_layers=total_layers, contention_procs=0, spinner_seconds=60.0,
    )
    contended, contended_meta, _ = v.measure_condition(
        requests=requests, model=model, workload=workload_profile, hardware=hardware,
        server_path=server_path, settings=settings, scenario="p12b", condition="contended",
        gpu_layers=0, threads=logical, total_layers=total_layers,
        contention_procs=args.spinners, spinner_seconds=300.0,
    )
    effective_observed = (contended_meta["server_log_n_threads"] or [None])[-1]
    report = BaselineReport(
        profiler_version="phase12b",
        suite_version="phase12b",
        generated_at=datetime.now(timezone.utc),
        hardware=hardware,
        backend=control[0].context.config if control else ProfilingConfig(),
        workloads=[workload_profile],
        runs=control + contended,
        groups=[],
        cold_warm_semantics="controlled CPU contention (CPU-only)",
        resident_warm_supported=True,
        limitations=["investigation-validation"],
        notes="phase12b reduce_threads",
    )
    if effective_observed is not None:
        report = _patch_effective(report, int(effective_observed))
    diagnosis = BottleneckDetector().detect(report, source_report_path="<phase12b>")
    payload["baseline"] = {"control": control_meta, "contended": contended_meta}
    payload["observed_effective_threads"] = effective_observed
    payload["diagnosis"] = v._diagnosis_summary(diagnosis)
    print(
        f"  control decode={control_meta['decode_mean']:.2f} contended decode={contended_meta['decode_mean']:.2f} "
        f"diagnosis={payload['diagnosis']['primary']} effective={effective_observed}"
    )

    plan = generate_candidates(diagnosis, report=report, registry=registry, policy=optimization_policy)
    candidates = [c for c in plan.candidates if c.action_id == "runtime.reduce_threads"]
    candidate = candidates[0] if candidates else None
    payload["candidate"] = (
        {
            "eligibility": candidate.eligibility.value,
            "executable": candidate.executable,
            "parameters": dict(candidate.parameters),
            "preconditions": [
                {"check_id": p.check_id, "satisfied": p.satisfied, "detail": p.detail}
                for p in candidate.preconditions
            ],
            "rejection_reason": candidate.rejection_reason,
        }
        if candidate is not None
        else None
    )
    print("  candidate:", payload["candidate"])

    # ------------------------------------------------------------------ #
    # Integration-gap check: runtime-faithful report (threads dropped)     #
    # ------------------------------------------------------------------ #
    runtime_config = (control[0].context.config if control else ProfilingConfig()).model_copy(
        update={"threads": None, "requested_threads": int(logical), "effective_threads": effective_observed}
    )
    baseline_from_runtime = baseline_configuration_from(runtime_config, model)
    payload["integration_gap_check"] = {
        "runtime_profiling_config": {
            "threads": runtime_config.threads,
            "requested_threads": runtime_config.requested_threads,
            "effective_threads": runtime_config.effective_threads,
        },
        "baseline_configuration_from_threads": baseline_from_runtime.threads,
        "gap": baseline_from_runtime.threads is None,
    }
    print("  integration gap check:", payload["integration_gap_check"])

    # ------------------------------------------------------------------ #
    # Pre-experiment economics: permission to *attempt* the intervention   #
    # ------------------------------------------------------------------ #
    preference = build_optimization_preference(settings)
    drop = None
    if control_meta["decode_mean"] and contended_meta["decode_mean"]:
        drop = max(0.0, min(1.0, 1.0 - contended_meta["decode_mean"] / control_meta["decode_mean"]))
    opportunity = RuntimeOpportunity(
        opportunity_id="opp-phase12c",
        anomaly=AnomalyClass.PERSISTENT,
        condition_key="telemetry:cpu:phase12c",
        kind="cpu",
        persistence_samples=runtime_policy.persistence_samples,
        deviation_score=drop if drop is not None else 0.5,
        confidence=diagnosis.confidence,
        strength=diagnosis.evidence_strength,
        meaningful=diagnosis.meaningful_opportunity,
    )
    performance_state = classify_performance_state(
        baseline_metrics={"decode_tokens_per_sec": control_meta["decode_mean"]},
        recent_metrics={"decode_tokens_per_sec": contended_meta["decode_mean"]},
        preference=preference,
    )
    pre_economics = None
    if candidate is not None:
        pre_economics = evaluate_intervention_economics(
            opportunity,
            candidate=candidate,
            action=registry.get("runtime.reduce_threads"),
            policy=runtime_policy,
            optimization_preference=preference,
            performance_state=performance_state.state,
        )
        payload["pre_economics"] = {
            "permitted": pre_economics.beneficial,
            "measured_improvement": pre_economics.measured_improvement,
            "improvement_justified": pre_economics.improvement_justified,
            "cost_terms": pre_economics.costs,
            "total_cost_s": sum(pre_economics.costs.values()),
            "cost_score": pre_economics.cost_score,
            "net_value": pre_economics.net_value,
            "requires_approval": pre_economics.requires_approval,
            "intervention_class": pre_economics.intervention_class.value,
            "rationale": pre_economics.rationale,
        }
        print(
            "  pre-economics permitted=", pre_economics.beneficial,
            "cost_s=", round(sum(pre_economics.costs.values()), 1),
            "budget_s=", preference.maximum_experiment_cost_s,
        )


    # ------------------------------------------------------------------ #
    # Phase 7 lifecycle (existing engine) with the corrected baseline     #
    # ------------------------------------------------------------------ #
    print("--- Phase 7 lifecycle (existing engine) ---")
    experiment_summary: dict[str, Any] = {}
    if candidate is None or candidate.parameters.get("threads") is None:
        experiment_summary = {"skipped": "no actionable thread candidate"}
    elif pre_economics is None or not pre_economics.beneficial:
        experiment_summary = {"skipped": "pre-experiment economics did not permit the attempt"}
    else:
        workload = ResidentActiveWorkload(
            settings=settings,
            model=model,
            adapter=LlamaCppResidentAdapter(settings=settings, server=server_path),
            workload_id="short-short",
            configuration=RuntimeConfiguration(context_length=4096, gpu_offload_layers=0, threads=logical),
        )
        spinners = [launch_cpu_spinner(300.0) for _ in range(args.spinners)]
        for ctx in spinners:
            ctx.__enter__()
        try:
            workload.start()
            effective_before = workload.effective_threads()
            # Production intervention semantics: drain/pause the serve loop before the
            # controlled experiment so measurements are not raced by concurrent service.
            workload.pause()
            profiler = BaselineProfiler(
                settings=settings,
                adapter=ResidentMeasurementAdapter(workload),
                hardware=hardware,
            )
            try:
                result = run_candidate_experiment(
                    candidate,
                    plan=plan,
                    diagnosis=diagnosis,
                    report=report,
                    outcome=None,
                    workload_id=workload.workload_profile().workload_id,
                    registry=registry,
                    optimization_policy=optimization_policy,
                    experiment_policy=experiment_policy,
                    executor=RuntimeWorkloadExecutor(workload, backend_name=workload.descriptor().backend_name),
                    profiler=profiler,
                    seed=args.seed,
                )
            finally:
                workload.resume()
            effective_after_engine = workload.effective_threads()

            # Post-measurement economics: technical improvement is necessary but not
            # sufficient; the measured Phase 7 benefit must also be worth retaining.
            technical_retained = bool(
                result.status.value == "improved"
                and result.decision_detail is not None
                and result.decision_detail.keep
            )
            retention_economics = evaluate_retention_economics(
                result=result,
                candidate=candidate,
                policy=runtime_policy,
                optimization_preference=preference,
                performance_state=performance_state.state,
            )
            retained = bool(technical_retained and retention_economics.beneficial)
            effective_retained = None
            if retained:
                workload.pause()
                executor = RuntimeWorkloadExecutor(
                    workload, backend_name=workload.descriptor().backend_name
                )
                executor.set_configuration(result.request.experiment_configuration)
                workload.resume()
                effective_retained = workload.effective_threads()
                # Validation harness: prove retention, then return the machine to baseline.
                workload.pause()
                executor.restore(result.request.baseline_configuration)
                workload.resume()
            effective_after = workload.effective_threads()
            payload["post_economics"] = {
                "technical_retained": technical_retained,
                "retained": retained,
                "measured_improvement": retention_economics.measured_improvement,
                "required_improvement": retention_economics.required_improvement,
                "beneficial": retention_economics.beneficial,
                "retained_change_cost_s": sum(retention_economics.costs.values()),
                "rationale": retention_economics.rationale,
            }
            print(
                "  post-economics measured=", retention_economics.measured_improvement,
                "beneficial=", retention_economics.beneficial, "retained=", retained,
            )
            experiment_summary = {
                "baseline_configuration_threads": result.request.baseline_configuration.threads,
                "experiment_configuration_threads": result.request.experiment_configuration.threads,
                "effective_threads_before": effective_before,
                "effective_threads_after_engine_restore": effective_after_engine,
                "effective_threads_when_retained": effective_retained,
                "effective_threads_after": effective_after,
                "technical_retained": technical_retained,
                "retained": retained,
                "status": result.status.value,
                "decision": {
                    "kind": result.decision_detail.kind.value,
                    "keep": result.decision_detail.keep,
                    "reason": result.decision_detail.reason,
                    "confidence": result.decision_detail.confidence,
                },
                "baseline_metrics": {
                    "decode_tokens_per_sec": result.baseline_measurement.metrics["decode_tokens_per_sec"].mean
                    if result.baseline_measurement
                    else None,
                    "process_cpu_percent": result.baseline_measurement.metrics["process_cpu_percent"].mean
                    if result.baseline_measurement
                    else None,
                    "valid_runs": result.baseline_measurement.valid_runs if result.baseline_measurement else None,
                    "invalid_runs": result.baseline_measurement.invalid_runs if result.baseline_measurement else None,
                },
                "candidate_metrics": {
                    "decode_tokens_per_sec": result.candidate_measurement.metrics["decode_tokens_per_sec"].mean
                    if result.candidate_measurement
                    else None,
                    "process_cpu_percent": result.candidate_measurement.metrics["process_cpu_percent"].mean
                    if result.candidate_measurement
                    else None,
                    "valid_runs": result.candidate_measurement.valid_runs if result.candidate_measurement else None,
                    "invalid_runs": result.candidate_measurement.invalid_runs if result.candidate_measurement else None,
                },
                "comparison": (
                    {
                        "bottleneck_improved": result.comparison.bottleneck_improved,
                        "objective_improved": result.comparison.objective_improved,
                        "objective_regressed": result.comparison.objective_regressed,
                        "resource_regressed": result.comparison.resource_regressed,
                        "deltas": {
                            name: delta.relative_delta
                            for name, delta in result.comparison.deltas.items()
                            if delta.relative_delta is not None
                        },
                    }
                    if result.comparison
                    else None
                ),
                "quality": (
                    {
                        "status": result.quality.status.value,
                        "validated": result.quality.validated,
                        "unacceptable_output": result.quality.unacceptable_output,
                    }
                    if result.quality
                    else None
                ),
                "restoration": (
                    {
                        "status": result.restoration.status.value,
                        "verified": result.restoration.verified,
                        "restored_threads": result.restoration.restored_configuration.threads,
                        "exact": result.restoration.restored_configuration
                        == result.request.baseline_configuration,
                    }
                    if result.restoration
                    else None
                ),
                "environment_restored": result.environment_restored,
                "applied": result.executed,
            }
            print("  phase7:", json.dumps(experiment_summary["decision"], default=str))
        finally:
            for ctx in reversed(spinners):
                try:
                    ctx.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
            try:
                workload.stop()
            except Exception:  # noqa: BLE001
                pass
    payload["phase7_experiment"] = experiment_summary

    # ------------------------------------------------------------------ #
    # Recovery / orphan check                                              #
    # ------------------------------------------------------------------ #
    import psutil

    time.sleep(2.0)
    payload["recovery"] = {
        "llama_server_processes": [
            p.pid for p in psutil.process_iter(["name"]) if (p.info.get("name") or "").lower() == "llama-server.exe"
        ],
        "free_ram_bytes": int(psutil.virtual_memory().available),
    }
    print("recovery:", payload["recovery"])

    out = ROOT / "artifacts" / "ab_effectiveness" / f"reduce_threads_phase12c_{timestamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

