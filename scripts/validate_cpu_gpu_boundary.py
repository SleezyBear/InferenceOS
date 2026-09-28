"""Phase 12A — CPU/GPU diagnosis boundary controlled experiment (investigation only).

Determines whether the current diagnosis can discriminate:
    CASE A: GPU configured + GPU actually limiting + CPU not independently limiting
    CASE B: GPU configured + GPU not limiting + CPU demonstrably limiting

It does NOT modify the diagnosis rules or any production code. It runs bounded,
reversible resident llama-server measurements at three fixed GPU-offload levels
(CPU-only, partial, full), each with a control and a bounded CPU-contention condition,
runs the production detector, and records what it diagnoses plus whether the diagnosis
yields any actionable candidate. It also verifies the new requested/effective thread
telemetry during a resident experiment.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\validate_cpu_gpu_boundary.py \\
        --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import json
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
from inferenceos.contracts.experiment import RuntimeConfiguration  # noqa: E402
from inferenceos.contracts.profiling import BaselineReport, ProfilingConfig  # noqa: E402
from inferenceos.diagnosis.detector import BottleneckDetector  # noqa: E402
from inferenceos.hardware.profiler import detect_hardware_profile  # noqa: E402
from inferenceos.models.loader import load_model_profile  # noqa: E402
from inferenceos.optimization.actions import build_default_registry  # noqa: E402
from inferenceos.optimization.generator import generate_candidates  # noqa: E402
from inferenceos.optimization.policy import build_policy  # noqa: E402


def _candidate_summary(diagnosis: Any, report: BaselineReport, policy: Any, registry: Any) -> dict[str, Any]:
    plan = generate_candidates(diagnosis, report=report, registry=registry, policy=policy)
    eligible = [
        {
            "action_id": c.action_id,
            "parameters": dict(c.parameters),
            "eligibility": c.eligibility.value,
            "executable": c.executable,
        }
        for c in plan.candidates
        if c.eligibility.value == "eligible" and c.executable
    ]
    reasons = [
        {"action_id": c.action_id, "eligibility": c.eligibility.value}
        for c in plan.candidates
        if c.eligibility.value != "eligible" or not c.executable
    ]
    return {
        "total_candidates": len(plan.candidates),
        "actionable_candidates": eligible,
        "non_actionable_candidates": reasons,
    }


def _would_be_cpu_rule(diagnosis: Any, report: BaselineReport, diagnosis_summary: dict[str, Any]) -> dict[str, Any]:
    """Recompute the CPU rule's own criteria from the same group means, WITHOUT changing
    the rule, to show whether it would have fired had it not been suppressed."""
    from inferenceos.diagnosis.groups import extract_groups
    from inferenceos.diagnosis.rules import (
        _best_higher,
        _cpu_saturation,
        RuleContext,
        CPU_SATURATION_HIGH,
        GPU_UTIL_LOW,
    )

    groups = extract_groups(report)
    physical = max(report.hardware.cpu.physical_cores, 1)
    logical = max(report.hardware.cpu.logical_cores, 1)
    ctx = RuleContext(groups=groups, physical_cores=physical, logical_cores=logical)
    rows: list[dict[str, Any]] = []
    for group in groups:
        saturation = _cpu_saturation(group, physical)
        gpu = group.mean("gpu_utilization_percent")
        best = _best_higher(ctx, group, "decode_tokens_per_sec")
        ratio = best[1] if best is not None else None
        rows.append(
            {
                "condition": group.condition,
                "cpu_saturation": saturation,
                "cpu_saturation_ge_threshold": bool(
                    saturation is not None and saturation >= CPU_SATURATION_HIGH
                ),
                "gpu_utilization_percent": gpu,
                "gpu_below_low_threshold": bool(gpu is not None and gpu < GPU_UTIL_LOW),
                "offload_configured": group.offload_configured,
                "comparable_throughput_ratio": ratio,
                "comparable_ratio_ge_1_5": bool(ratio is not None and ratio >= 1.5),
                "cpu_rule_would_fire_absent_suppression": bool(
                    saturation is not None
                    and saturation >= CPU_SATURATION_HIGH
                    and (gpu is not None and gpu < GPU_UTIL_LOW)
                    and (ratio is not None and ratio >= 1.5)
                ),
            }
        )
    return {"physical_cores": physical, "logical_cores": logical, "groups": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=2)
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
        rounds=args.rounds, n_predict=args.n_predict, seed=args.seed,
        temperature=0.0, ignore_eos=False,
    )
    workload_profile = requests[0].to_benchmark_workload().to_workload_profile()
    policy = build_policy(settings)
    registry = build_default_registry()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, Any] = {
        "schema": "cpu_gpu_boundary_experiment.v1",
        "timestamp": timestamp,
        "hardware": {"physical_cores": physical, "logical_cores": logical, "cpu": hardware.cpu.name},
        "model": {"name": model.name, "n_layers": model.n_layers},
        "request_count": len(requests),
        "conditions": {},
    }
    print(f"physical={physical} logical={logical} layers={total_layers} requests={len(requests)}")

    # Scenarios: fixed GPU configuration per scenario; CPU pressure varied.
    scenarios = (
        ("cpu_only", 0, "GPU offload NOT configured (CPU-only)"),
        ("partial_offload", 8, "GPU offload configured (partial, 8/28)"),
        ("full_offload", total_layers, f"GPU offload configured (full, {total_layers}/{total_layers})"),
    )

    for name, offload, description in scenarios:
        print(f"--- {name}: {description} ---")
        control, control_meta, _ = v.measure_condition(
            requests=requests, model=model, workload=workload_profile, hardware=hardware,
            server_path=server_path, settings=settings, scenario=name, condition="control",
            gpu_layers=offload, threads=logical, total_layers=total_layers,
            contention_procs=0, spinner_seconds=60.0,
        )
        contended, contended_meta, _ = v.measure_condition(
            requests=requests, model=model, workload=workload_profile, hardware=hardware,
            server_path=server_path, settings=settings, scenario=name, condition="contended",
            gpu_layers=offload, threads=logical, total_layers=total_layers,
            contention_procs=max(physical, 4), spinner_seconds=150.0,
        )
        report = BaselineReport(
            profiler_version="phase12a.boundary",
            suite_version="phase12a",
            generated_at=datetime.now(timezone.utc),
            hardware=hardware,
            backend=control[0].context.config if control else ProfilingConfig(),
            workloads=[workload_profile],
            runs=control + contended,
            groups=[],
            cold_warm_semantics="controlled CPU-pressure perturbation at fixed GPU configuration",
            resident_warm_supported=True,
            limitations=["investigation-only controlled condition"],
            notes=f"phase12a boundary {name}",
        )
        diagnosis = BottleneckDetector().detect(report, source_report_path=f"<phase12a-{name}>")
        summary = v._diagnosis_summary(diagnosis)
        payload["conditions"][name] = {
            "description": description,
            "gpu_layers": offload,
            "control": control_meta,
            "contended": contended_meta,
            "diagnosis": summary,
            "candidate_actionability": _candidate_summary(diagnosis, report, policy, registry),
            "would_be_cpu_rule": _would_be_cpu_rule(diagnosis, report, summary),
        }
        print(
            f"  primary={summary['primary']} meaningful={summary['meaningful_opportunity']} "
            f"conf={summary['primary_confidence']} actionable="
            f"{[c['action_id'] for c in payload['conditions'][name]['candidate_actionability']['actionable_candidates']]}"
        )

    # ------------------------------------------------------------------ #
    # Effective-thread telemetry verification (resident path)             #
    # ------------------------------------------------------------------ #
    print("--- effective-thread telemetry (resident path) ---")
    resident_checks: list[dict[str, Any]] = []
    for requested in (logical, None):
        workload = v.ResidentActiveWorkload(  # type: ignore[attr-defined]
            settings=settings,
            model=model,
            adapter=v.LlamaCppResidentAdapter(settings=settings, server=server_path),  # type: ignore[attr-defined]
            workload_id="short-short",
            configuration=RuntimeConfiguration(context_length=4096, gpu_offload_layers=8, threads=requested),
        )
        try:
            workload.start()
            resident_checks.append(
                {
                    "requested_threads": workload.configuration().threads,
                    "effective_threads": workload.effective_threads(),
                }
            )
            print(
                f"  requested={workload.configuration().threads} "
                f"effective={workload.effective_threads()}"
            )
        finally:
            try:
                workload.stop()
            except Exception:  # noqa: BLE001
                pass
    payload["effective_thread_telemetry"] = resident_checks

    # ------------------------------------------------------------------ #
    # Recovery                                                            #
    # ------------------------------------------------------------------ #
    import psutil

    time.sleep(2.0)
    payload["recovery"] = {
        "llama_server_processes": [
            p.pid for p in psutil.process_iter(["name"])
            if (p.info.get("name") or "").lower() == "llama-server.exe"
        ],
        "free_ram_bytes": int(psutil.virtual_memory().available),
    }
    print("recovery:", payload["recovery"])

    out = ROOT / "artifacts" / "ab_effectiveness" / f"cpu_gpu_boundary_{timestamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
