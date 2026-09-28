"""Repeatability validation for `runtime.reduce_threads` (12 vs 6 threads).

Runs N bounded repetitions of the existing Phase 7 lifecycle (baseline 12 threads vs
candidate 6 threads) under the same model/context/ngl/workload/seed/contention, and reports
per-repetition and aggregate results plus direction-consistency. No production behavior is
changed. The machine is restored to the exact baseline at the end.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\validate_reduce_threads_repeatability.py ^
        --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf --reps 3
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
from inferenceos.contracts.enums import AnomalyClass  # noqa: E402
from inferenceos.contracts.experiment import RuntimeConfiguration  # noqa: E402
from inferenceos.contracts.profiling import BaselineReport, ProfilingConfig  # noqa: E402
from inferenceos.contracts.runtime import RuntimeOpportunity  # noqa: E402
from inferenceos.diagnosis.detector import BottleneckDetector  # noqa: E402
from inferenceos.experiment.policy import build_experiment_policy  # noqa: E402
from inferenceos.experiment.service import run_candidate_experiment  # noqa: E402
from inferenceos.external.controlled import launch_cpu_spinner  # noqa: E402
from inferenceos.hardware.profiler import detect_hardware_profile  # noqa: E402
from inferenceos.models.loader import load_model_profile  # noqa: E402
from inferenceos.optimization.actions import build_default_registry  # noqa: E402
from inferenceos.optimization.generator import generate_candidates  # noqa: E402
from inferenceos.optimization.policy import build_policy  # noqa: E402
from inferenceos.optimization.preference import build_optimization_preference  # noqa: E402
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter  # noqa: E402
from inferenceos.profiling.runner import BaselineProfiler  # noqa: E402
from inferenceos.runtime.policy import build_runtime_policy  # noqa: E402
from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter  # noqa: E402
from inferenceos.runtime.resident_workload import ResidentActiveWorkload  # noqa: E402
from inferenceos.runtime.workload import RuntimeWorkloadExecutor  # noqa: E402

TRACKED = (
    "decode_tokens_per_sec",
    "cpu_seconds_per_token",
    "process_cpu_percent",
    "context_switches_voluntary",
    "context_switches_involuntary",
    "context_switches_total",
    "context_switches_per_token",
    "peak_rss_bytes",
    "hard_page_faults_per_sec",
    "total_latency_s",
)

# Expected direction of the 12 -> 6 change (True = higher is better).
HIGHER_IS_BETTER = {
    "decode_tokens_per_sec": True,
    "cpu_seconds_per_token": False,
    "process_cpu_percent": False,
    "context_switches_voluntary": False,
    "context_switches_involuntary": False,
    "context_switches_total": False,
    "context_switches_per_token": False,
    "peak_rss_bytes": False,
    "hard_page_faults_per_sec": False,
    "total_latency_s": False,
}


def _patch_effective(report: BaselineReport, effective: int) -> BaselineReport:
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


def _stats(measurement: Any, name: str) -> float | None:
    if measurement is None:
        return None
    stats = measurement.metrics.get(name)
    return stats.mean if stats is not None else None


def _rep_metrics(measurement: Any) -> dict[str, Any]:
    values = {name: _stats(measurement, name) for name in TRACKED}
    values["valid_runs"] = measurement.valid_runs if measurement else 0
    values["invalid_runs"] = measurement.invalid_runs if measurement else 0
    return values


def _aggregate(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--spinners", type=int, default=4)
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
        rounds=args.rounds, n_predict=args.n_predict, seed=args.seed,
        temperature=0.0, ignore_eos=False,
    )
    workload_profile = requests[0].to_benchmark_workload().to_workload_profile()
    registry = build_default_registry()
    optimization_policy = build_policy(settings)
    experiment_policy = build_experiment_policy(settings)
    runtime_policy = build_runtime_policy(settings)
    preference = build_optimization_preference(settings)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, Any] = {
        "schema": "reduce_threads_repeatability.v1",
        "timestamp": timestamp,
        "hardware": {"physical_cores": physical, "logical_cores": logical, "cpu": hardware.cpu.name},
        "reps_requested": args.reps,
        "spinners": args.spinners,
        "repetitions": [],
    }
    print(f"physical={physical} logical={logical} layers={total_layers} reps={args.reps}")

    # --- One diagnosis + candidate preparation (normal production path) ---
    control, control_meta, _ = v.measure_condition(
        requests=requests, model=model, workload=workload_profile, hardware=hardware,
        server_path=server_path, settings=settings, scenario="rep", condition="control",
        gpu_layers=0, threads=logical, total_layers=total_layers, contention_procs=0, spinner_seconds=60.0,
    )
    contended, contended_meta, _ = v.measure_condition(
        requests=requests, model=model, workload=workload_profile, hardware=hardware,
        server_path=server_path, settings=settings, scenario="rep", condition="contended",
        gpu_layers=0, threads=logical, total_layers=total_layers,
        contention_procs=args.spinners, spinner_seconds=60.0,
    )
    effective_observed = (contended_meta["server_log_n_threads"] or [None])[-1]
    report = BaselineReport(
        profiler_version="phase12e", suite_version="phase12e",
        generated_at=datetime.now(timezone.utc), hardware=hardware,
        backend=control[0].context.config if control else ProfilingConfig(),
        workloads=[workload_profile], runs=control + contended, groups=[],
        cold_warm_semantics="repeatability diagnosis", resident_warm_supported=True,
        limitations=["validation"], notes="reduce_threads repeatability",
    )
    if effective_observed is not None:
        report = _patch_effective(report, int(effective_observed))
    diagnosis = BottleneckDetector().detect(report, source_report_path="<phase12e>")
    plan = generate_candidates(diagnosis, report=report, registry=registry, policy=optimization_policy)
    candidate = next((c for c in plan.candidates if c.action_id == "runtime.reduce_threads"), None)
    payload["diagnosis"] = v._diagnosis_summary(diagnosis)
    payload["candidate"] = (
        {"eligibility": candidate.eligibility.value, "parameters": dict(candidate.parameters)}
        if candidate is not None else None
    )
    print("diagnosis:", payload["diagnosis"]["primary"], "candidate:", payload["candidate"])
    if candidate is None or candidate.parameters.get("threads") is None:
        print("no actionable candidate; aborting")
        return 2

    spinners = [launch_cpu_spinner(float(120 * args.reps + 120)) for _ in range(args.spinners)]
    for ctx in spinners:
        ctx.__enter__()
    try:
        for rep in range(1, args.reps + 1):
            print(f"--- repetition {rep}/{args.reps} ---")
            workload = ResidentActiveWorkload(
                settings=settings, model=model,
                adapter=LlamaCppResidentAdapter(settings=settings, server=server_path),
                workload_id="short-short",
                configuration=RuntimeConfiguration(context_length=4096, gpu_offload_layers=0, threads=logical),
            )
            workload.start()
            effective_before = workload.effective_threads()
            workload.pause()
            profiler = BaselineProfiler(
                settings=settings, adapter=ResidentMeasurementAdapter(workload), hardware=hardware
            )
            try:
                result = run_candidate_experiment(
                    candidate, plan=plan, diagnosis=diagnosis, report=report, outcome=None,
                    workload_id=workload.workload_profile().workload_id, registry=registry,
                    optimization_policy=optimization_policy, experiment_policy=experiment_policy,
                    executor=RuntimeWorkloadExecutor(workload, backend_name=workload.descriptor().backend_name),
                    profiler=profiler, seed=args.seed,
                )
            finally:
                workload.resume()
            effective_after = workload.effective_threads()
            workload.stop()

            base_m = _rep_metrics(result.baseline_measurement)
            cand_m = _rep_metrics(result.candidate_measurement)
            changes: dict[str, float | None] = {}
            for name in TRACKED:
                b, c = base_m.get(name), cand_m.get(name)
                changes[name] = ((c - b) / b) if (b not in (None, 0) and c is not None) else None
            rep_data = {
                "repetition": rep,
                "effective_threads_before": effective_before,
                "effective_threads_after": effective_after,
                "baseline": base_m,
                "candidate": cand_m,
                "relative_change": changes,
                "status": result.status.value,
                "technical_keep": bool(result.decision_detail and result.decision_detail.keep),
                "bottleneck_improved": (
                    result.comparison.bottleneck_improved if result.comparison else None
                ),
                "quality_status": result.quality.status.value if result.quality else None,
                "quality_validated": result.quality.validated if result.quality else None,
                "restoration_exact": (
                    result.restoration.restored_configuration == result.request.baseline_configuration
                    if result.restoration else None
                ),
                "environment_restored": result.environment_restored,
            }
            payload["repetitions"].append(rep_data)
            print(
                "  decode {:.2f}->{:.2f} ({:+.1%})  cpu_s/tok {:.4f}->{:.4f} ({:+.1%})  "
                "ctx/tok {:.0f}->{:.0f} ({:+.1%})  quality={} restored={}".format(
                    base_m["decode_tokens_per_sec"], cand_m["decode_tokens_per_sec"],
                    changes["decode_tokens_per_sec"] or 0.0,
                    base_m["cpu_seconds_per_token"], cand_m["cpu_seconds_per_token"],
                    changes["cpu_seconds_per_token"] or 0.0,
                    base_m["context_switches_per_token"] or 0.0,
                    cand_m["context_switches_per_token"] or 0.0,
                    changes["context_switches_per_token"] or 0.0,
                    result.quality.status.value if result.quality else None,
                    rep_data["restoration_exact"],
                )
            )
    finally:
        for ctx in reversed(spinners):
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass

    # --- Analysis ---
    reps = payload["repetitions"]
    aggregate: dict[str, Any] = {}
    for name in TRACKED:
        baseline_values = [r["baseline"][name] for r in reps if r["baseline"][name] is not None]
        candidate_values = [r["candidate"][name] for r in reps if r["candidate"][name] is not None]
        change_values = [r["relative_change"][name] for r in reps if r["relative_change"][name] is not None]
        expected_higher = HIGHER_IS_BETTER[name]
        consistent = [
            (v > 0) == expected_higher for v in change_values
        ] if change_values else []
        aggregate[name] = {
            "baseline": _aggregate([float(x) for x in baseline_values]),
            "candidate": _aggregate([float(x) for x in candidate_values]),
            "relative_change": _aggregate([float(x) for x in change_values]),
            "expected_direction": "increase" if expected_higher else "decrease",
            "reps_in_expected_direction": sum(1 for c in consistent if c),
            "reps_valid_for_direction": len(consistent),
            "consistent": bool(consistent) and all(consistent),
        }
    payload["aggregate"] = aggregate
    payload["quality"] = {
        "statuses": [r["quality_status"] for r in reps],
        "all_validated": all(r["quality_validated"] for r in reps),
    }
    payload["resource_validity"] = {
        "all_restored": all(r["environment_restored"] and r["restoration_exact"] for r in reps),
        "invalid_runs": sum(r["baseline"]["invalid_runs"] + r["candidate"]["invalid_runs"] for r in reps),
    }
    print("aggregate:")
    for name in ("decode_tokens_per_sec", "cpu_seconds_per_token", "context_switches_per_token", "process_cpu_percent"):
        agg = aggregate[name]
        print(
            f"  {name}: mean rel {agg['relative_change']['mean']:+.1%} "
            f"median {agg['relative_change']['median']:+.1%} "
            f"[{agg['relative_change']['min']:+.1%},{agg['relative_change']['max']:+.1%}] "
            f"consistent={agg['consistent']} ({agg['reps_in_expected_direction']}/{agg['reps_valid_for_direction']})"
        )

    out = ROOT / "artifacts" / "ab_effectiveness" / f"reduce_threads_repeatability_{timestamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
