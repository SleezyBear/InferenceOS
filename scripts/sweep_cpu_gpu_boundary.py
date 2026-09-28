"""Phase 12A — bounded CPU-pressure sweep at fixed partial GPU offload (investigation).

Tests whether a *milder* CPU pressure (than the boundary experiment) keeps the target
process CPU saturation at/above the CPU rule's threshold while still depressing
throughput comparatively, which would be the condition under which the
``offload_configured -> suppress CPU`` rule could produce a demonstrated false-negative.

Does not modify any production code.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ab_effectiveness as ab  # noqa: E402
import validate_cpu_gpu_boundary as b  # noqa: E402
import validate_cpu_thread_surface as v  # noqa: E402

from inferenceos.config import Settings  # noqa: E402
from inferenceos.contracts.profiling import BaselineReport, ProfilingConfig  # noqa: E402
from inferenceos.diagnosis.detector import BottleneckDetector  # noqa: E402
from inferenceos.hardware.profiler import detect_hardware_profile  # noqa: E402
from inferenceos.models.loader import load_model_profile  # noqa: E402
from inferenceos.optimization.actions import build_default_registry  # noqa: E402
from inferenceos.optimization.policy import build_policy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-layers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--spinners", type=int, nargs="+", default=[2, 3, 4])
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

    print(f"physical={physical} logical={logical} layers={total_layers} gpu_layers={args.gpu_layers}")
    print("--- control ---")
    control, control_meta, _ = v.measure_condition(
        requests=requests, model=model, workload=workload_profile, hardware=hardware,
        server_path=server_path, settings=settings, scenario="sweep", condition="control",
        gpu_layers=args.gpu_layers, threads=logical, total_layers=total_layers,
        contention_procs=0, spinner_seconds=60.0,
    )
    payload = {
        "schema": "cpu_gpu_boundary_sweep.v1",
        "timestamp": timestamp,
        "gpu_layers": args.gpu_layers,
        "physical_cores": physical,
        "logical_cores": logical,
        "control": control_meta,
        "levels": [],
    }
    print(
        f"  control decode={control_meta['decode_mean']:.2f} "
        f"procCPU={control_meta['telemetry']['process_cpu_mean']:.1f} "
        f"GPU={control_meta['telemetry']['gpu_util_mean']:.1f}"
    )
    for spinners in args.spinners:
        print(f"--- contended spinners={spinners} ---")
        contended, contended_meta, _ = v.measure_condition(
            requests=requests, model=model, workload=workload_profile, hardware=hardware,
            server_path=server_path, settings=settings, scenario="sweep",
            condition=f"contended_{spinners}", gpu_layers=args.gpu_layers, threads=logical,
            total_layers=total_layers, contention_procs=spinners, spinner_seconds=150.0,
        )
        report = BaselineReport(
            profiler_version="phase12a.boundary-sweep",
            suite_version="phase12a",
            generated_at=datetime.now(timezone.utc),
            hardware=hardware,
            backend=control[0].context.config if control else ProfilingConfig(),
            workloads=[workload_profile],
            runs=control + contended,
            groups=[],
            cold_warm_semantics="bounded CPU-pressure sweep at fixed partial offload",
            resident_warm_supported=True,
            limitations=["investigation-only"],
            notes=f"phase12a sweep spinners={spinners}",
        )
        diagnosis = BottleneckDetector().detect(report, source_report_path=f"<sweep-{spinners}>")
        summary = v._diagnosis_summary(diagnosis)
        would = b._would_be_cpu_rule(diagnosis, report, summary)
        contended_row = next(r for r in would["groups"] if r["condition"] == f"contended_{spinners}")
        entry = {
            "spinners": spinners,
            "contended": contended_meta,
            "diagnosis": summary,
            "cpu_rule": contended_row,
        }
        payload["levels"].append(entry)
        print(
            f"  decode={contended_meta['decode_mean']:.2f} "
            f"procCPU={contended_meta['telemetry']['process_cpu_mean']:.1f} "
            f"sat={contended_row['cpu_saturation']:.3f} sat>=.70={contended_row['cpu_saturation_ge_threshold']} "
            f"GPU={contended_row['gpu_utilization_percent']:.1f} "
            f"ratio={contended_row['comparable_throughput_ratio']:.2f} "
            f"cpu_would_fire={contended_row['cpu_rule_would_fire_absent_suppression']} "
            f"primary={summary['primary']}"
        )
    out = ROOT / "artifacts" / "ab_effectiveness" / f"cpu_gpu_boundary_sweep_{timestamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
