"""Final managed-vs-unmanaged GPU-offload effectiveness A/B.

Unmanaged arm A: a resident llama-server started directly at the baseline GPU-layer
configuration, serving the fixed stationary workload. No monitor, no intervention.

Managed arm B: the same resident workload through the *production* InferenceOS runtime
(the real GPU-offload search, comparison/decision, economics, approval, application and
Phase 7 retention), with the measurement-robustness evaluator enabled.

B is separated into B-pre and B-post at the retained intervention boundary so the
within-run effect (B-pre -> B-post) can be measured independently of the aggregate
all-A-vs-post-B comparison.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\gpu_offload_final_effectiveness.py \\
        --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf --context 4096 --gpu-layers 8 \\
        --rounds 6 --n-predict 64 --seed 1234
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_ab_effectiveness as ab  # noqa: E402

from inferenceos.config import Settings  # noqa: E402
from inferenceos.contracts.experiment import RuntimeConfiguration  # noqa: E402
from inferenceos.experiment.policy import build_experiment_policy  # noqa: E402
from inferenceos.models.loader import load_model_profile  # noqa: E402
from inferenceos.runtime.approval import AutoApproveApprovalProvider  # noqa: E402
from inferenceos.runtime.monitor import RuntimeMonitor  # noqa: E402
from inferenceos.runtime.notification import CollectingNotificationSink  # noqa: E402
from inferenceos.runtime.policy import build_runtime_budget, build_runtime_policy  # noqa: E402
from inferenceos.runtime.resident_workload import ResidentActiveWorkload  # noqa: E402
from inferenceos.runtime.telemetry import SystemTelemetrySampler  # noqa: E402
from inferenceos.runtime.verification import (  # noqa: E402
    ControlledGpuUnderutilizationDetector,
)

MODEL_SHA256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
METRICS = (
    ("decode_tokens_per_sec", True),
    ("ttft_s", False),
    ("total_latency_s", False),
    ("decode_s", False),
    ("peak_rss_bytes", False),
)


def _values(outcomes: list[ab.RequestOutcome], field: str) -> list[float]:
    out: list[float] = []
    for item in outcomes:
        value = getattr(item, field, None)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _summary(outcomes: list[ab.RequestOutcome]) -> dict[str, object]:
    return {
        "requests": len(outcomes),
        "successes": sum(1 for item in outcomes if item.success),
        "failures": sum(1 for item in outcomes if not item.success),
        "decode_mean": (
            statistics.fmean(_values(outcomes, "decode_tokens_per_sec"))
            if _values(outcomes, "decode_tokens_per_sec")
            else None
        ),
        "ttft_mean": (
            statistics.fmean(_values(outcomes, "ttft_s")) if _values(outcomes, "ttft_s") else None
        ),
        "latency_mean": (
            statistics.fmean(_values(outcomes, "total_latency_s"))
            if _values(outcomes, "total_latency_s")
            else None
        ),
    }


def _compare(
    a: list[float], b: list[float], higher_is_better: bool, min_effect: float
) -> dict[str, object]:
    return ab.metric_verdict(
        baseline_values=a, measured_values=b, higher_is_better=higher_is_better, min_effect=min_effect
    )


def _matched(
    a_by_index: dict[int, ab.RequestOutcome],
    b_by_index: dict[int, ab.RequestOutcome],
    indices: list[int],
    min_effect: float,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for field, higher in METRICS:
        pairs_a: list[float] = []
        pairs_b: list[float] = []
        for index in indices:
            a = a_by_index.get(index)
            b = b_by_index.get(index)
            if a is None or b is None or not a.success or not b.success:
                continue
            av = getattr(a, field, None)
            bv = getattr(b, field, None)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                pairs_a.append(float(av))
                pairs_b.append(float(bv))
        result[field] = _compare(pairs_a, pairs_b, higher, min_effect)
    return result


def run_unmanaged(
    *, requests, model, server_path, settings, context, gpu_layers, log_path, stabilization_s
) -> tuple[list[ab.RequestOutcome], dict[str, object], object]:
    launched = ab.start_unmanaged_server(
        server=server_path,
        model_path=Path(model.path),
        host=settings.resident_host,
        context_length=context,
        gpu_layers=gpu_layers,
        threads=None,
        ignore_eos=False,
        log_path=log_path,
        readiness_timeout_s=settings.resident_readiness_timeout_s,
    )
    pid_holder = {"pid": launched.pid}
    collector = ab.TelemetryCollector(
        jsonl_path=log_path.with_suffix(".telemetry.jsonl"),
        interval_s=0.5,
        target_pid_provider=lambda: pid_holder["pid"],
    )
    collector.start()
    time.sleep(max(stabilization_s, 0.0))
    outcomes: list[ab.RequestOutcome] = []
    try:
        for request in requests:
            outcomes.append(
                ab.run_unmanaged_request(
                    test="A",
                    request=request,
                    host=launched.host,
                    port=launched.port,
                    server_pid=launched.pid,
                    timeout_s=settings.resident_request_timeout_s,
                )
            )
    finally:
        telemetry = collector.stop()
        stop_report = ab.stop_unmanaged_server(launched)
        pid_holder["pid"] = None
    identity = {
        "server_pid": launched.pid,
        "server_identity": launched.identity.model_dump() if launched.identity else None,
        "command": launched.command,
        "startup_s": launched.startup_s,
        "ready_s": launched.ready_s,
        "stop": stop_report,
    }
    return outcomes, {"telemetry": asdict(telemetry), "identity": identity}, telemetry


def run_managed(
    *,
    requests,
    model,
    server_path,
    settings,
    context,
    gpu_layers,
    duration_s,
    stabilization_s,
    label,
) -> tuple[list[ab.RequestOutcome], object | None, object | None, dict[str, object]]:
    from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter

    outcomes: list[ab.RequestOutcome] = []
    workload = ab.FixedSequenceResidentWorkload(
        fixed_requests=requests,
        outcomes=outcomes,
        test_label=label,
        settings=settings,
        model=model,
        adapter=LlamaCppResidentAdapter(settings=settings, server=server_path),
        configuration=RuntimeConfiguration(context_length=context, gpu_offload_layers=gpu_layers),
        request_gap_s=0.05,
    )
    collector = ab.TelemetryCollector(
        jsonl_path=Path(settings.reports_dir) / f"{label}_during.jsonl",
        interval_s=0.5,
        target_pid_provider=workload.target_pid,
    )
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=SystemTelemetrySampler(settings=settings, include_gpu=True),
        policy=build_runtime_policy(
            settings, budget=build_runtime_budget(settings, max_duration_s=duration_s)
        ),
        approval_provider=AutoApproveApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        detector=ControlledGpuUnderutilizationDetector(force=True),
        experiment_policy=build_experiment_policy(settings),
        settings=settings,
        session_notes=[
            "Final managed-vs-unmanaged GPU-offload effectiveness experiment.",
            "The managed arm ran the production optimizer; no policy or threshold was "
            "modified.",
        ],
    )
    collector.start()
    monitor_run = None
    monitor_error = None
    try:
        # Inter-arm environmental settle. The monitor owns the workload lifecycle (it
        # starts the resident target and establishes a workload-aware baseline).
        time.sleep(max(stabilization_s, 0.0))
        result = monitor.run(duration_s=duration_s)
        monitor_run = result.run
    except Exception as exc:  # noqa: BLE001
        monitor_error = f"{type(exc).__name__}: {exc}"
    finally:
        telemetry = collector.stop()
        try:
            workload.stop()
        except Exception:  # noqa: BLE001
            pass
    return outcomes, monitor_run, telemetry, {"monitor_error": monitor_error}


def _intervention_boundary(monitor_run) -> str | None:
    if monitor_run is None:
        return None
    boundary: str | None = None
    for intervention in monitor_run.interventions:
        if not intervention.retained and not intervention.parameters:
            continue
        candidate = intervention.resumed_at or intervention.finished_at or intervention.started_at
        if candidate is not None:
            stamp = candidate.isoformat()
            if boundary is None or stamp > boundary:
                boundary = stamp
    return boundary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stabilization-seconds", type=float, default=60.0)
    parser.add_argument("--monitor-duration-seconds", type=float, default=360.0)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument(
        "--arm-order", choices=["ab", "ba"], default="ab",
        help="arm order to counterbalance temporal/thermal bias",
    )
    parser.add_argument(
        "--diagnostic-economics", action="store_true",
        help="also run a clearly separated diagnostic managed pass with benefit_scale_s=300",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = Settings()
    settings.ensure_directories()
    model = load_model_profile(Path(args.model))
    server_path = settings.llama_cpp_dir / "llama-server.exe"
    requests = ab.build_fixed_requests(
        rounds=args.rounds, n_predict=args.n_predict, seed=args.seed,
        temperature=0.0, ignore_eos=False,
    )
    min_effect = float(getattr(settings, "phase7_min_effect", 0.05))
    started = datetime.now(timezone.utc)
    payload: dict[str, object] = {
        "schema": "gpu_offload_final_effectiveness.v1",
        "started_at": started.isoformat(),
        "environment": {
            "model": model.name,
            "model_path": model.path,
            "model_sha256_expected": MODEL_SHA256,
            "context_length": args.context,
            "starting_gpu_layers": args.gpu_layers,
            "rounds": args.rounds,
            "n_predict": args.n_predict,
            "seed": args.seed,
            "request_count": len(requests),
            "server_path": str(server_path),
            "stabilization_seconds": args.stabilization_seconds,
            "monitor_duration_seconds": args.monitor_duration_seconds,
            "phase10_benefit_scale_s": settings.phase10_benefit_scale_s,
            "phase7_min_effect": min_effect,
            "phase7_noise_tolerance": settings.phase7_noise_tolerance,
            "phase6_gpu_offload_max_measure_attempts": settings.phase6_gpu_offload_max_measure_attempts,
        },
        "arm_order": args.arm_order,
    }

    def do_a():
        print("=== A: unmanaged resident llama-server (fixed layers) ===")
        outcomes, extra, telemetry = run_unmanaged(
            requests=requests, model=model, server_path=server_path, settings=settings,
            context=args.context, gpu_layers=args.gpu_layers,
            log_path=Path(settings.logs_dir) / "final_ab_a_server.log",
            stabilization_s=args.stabilization_seconds,
        )
        print("  A summary", _summary(outcomes))
        return outcomes, extra, telemetry

    def do_b(settings_obj, label):
        print(f"=== {label}: managed (production optimizer) ===")
        outcomes, monitor, telemetry, extra = run_managed(
            requests=requests, model=model, server_path=server_path, settings=settings_obj,
            context=args.context, gpu_layers=args.gpu_layers,
            duration_s=args.monitor_duration_seconds,
            stabilization_s=args.stabilization_seconds, label=label,
        )
        print("  B summary", _summary(outcomes))
        return outcomes, monitor, telemetry, extra

    if args.arm_order == "ab":
        a_outcomes, a_extra, a_telemetry = do_a()
        b_outcomes, monitor_run, b_telemetry, b_extra = do_b(settings, "B_managed")
    else:
        b_outcomes, monitor_run, b_telemetry, b_extra = do_b(settings, "B_managed")
        a_outcomes, a_extra, a_telemetry = do_a()
    boundary = _intervention_boundary(monitor_run)
    for item in b_outcomes:
        item.post_intervention = bool(boundary is not None and item.submitted_at >= boundary)
    pre = [item for item in b_outcomes if not item.post_intervention]
    post = [item for item in b_outcomes if item.post_intervention]
    print("  B boundary", boundary, "pre", len(pre), "post", len(post), _summary(b_outcomes))

    a_by_index = {item.index: item for item in a_outcomes}
    post_indices = [item.index for item in post]
    pre_indices = [item.index for item in pre]

    comparisons = {
        "aggregate_A_vs_B_pre": {
            field: _compare(_values(a_outcomes, field), _values(pre, field), higher, min_effect)
            for field, higher in METRICS
        },
        "aggregate_A_vs_B_post": {
            field: _compare(_values(a_outcomes, field), _values(post, field), higher, min_effect)
            for field, higher in METRICS
        },
        "aggregate_B_pre_vs_B_post": {
            field: _compare(_values(pre, field), _values(post, field), higher, min_effect)
            for field, higher in METRICS
        },
        "matched_A_vs_B_post": _matched(a_by_index, {i.index: i for i in b_outcomes}, post_indices, min_effect),
        "matched_A_vs_B_pre": _matched(a_by_index, {i.index: i for i in b_outcomes}, pre_indices, min_effect),
    }

    optimization = None
    if monitor_run is not None:
        optimization =         {
            "termination": monitor_run.termination.reason.value,
            "termination_detail": monitor_run.termination.detail,
            "errors": list(monitor_run.errors),
            "diagnosis_events": monitor_run.diagnosis_events,
            "searches": [s.model_dump(mode="json") for s in monitor_run.gpu_searches],
            "recommendations": len(monitor_run.recommendations),
            "approvals": [a.outcome.value for a in monitor_run.approvals],
            "interventions": [
                {
                    "retained": i.retained,
                    "parameters": i.parameters,
                    "decision": i.decision.value if i.decision else None,
                    "quality_status": i.quality_status.value if i.quality_status else None,
                    "resume_verified": i.resume_verified,
                }
                for i in monitor_run.interventions
            ],
            "cooldowns": [c.trigger for c in monitor_run.cooldowns],
            "overhead": monitor_run.overhead.model_dump(mode="json"),
            "net_benefit": (
                json.loads(monitor_run.net_benefit.model_dump_json())
                if monitor_run.net_benefit is not None
                else None
            ),
            "final_configuration": monitor_run.workload.configuration.model_dump(),
            "boundary": boundary,
        }

    payload["results"] = {
        "A_unmanaged": {
            "summary": _summary(a_outcomes),
            "telemetry": a_extra["telemetry"],
            "identity": a_extra["identity"],
            "outcomes": [o.to_dict() for o in a_outcomes],
        },
        "B_managed": {
            "summary": _summary(b_outcomes),
            "pre_summary": _summary(pre),
            "post_summary": _summary(post),
            "telemetry": asdict(b_telemetry),
            "outcomes": [o.to_dict() for o in b_outcomes],
            "optimization": optimization,
        },
        "comparisons": comparisons,
    }

    diagnostic = None
    if args.diagnostic_economics:
        print("=== DIAGNOSTIC ECONOMICS PASS: benefit_scale_s=300 ===")
        diag_settings = settings.model_copy(update={"phase10_benefit_scale_s": 300.0})
        d_outcomes, d_monitor, d_telemetry, _ = run_managed(
            requests=requests, model=model, server_path=server_path, settings=diag_settings,
            context=args.context, gpu_layers=args.gpu_layers,
            duration_s=args.monitor_duration_seconds, stabilization_s=args.stabilization_seconds,
            label="B_diagnostic",
        )
        d_boundary = _intervention_boundary(d_monitor)
        for item in d_outcomes:
            item.post_intervention = bool(d_boundary is not None and item.submitted_at >= d_boundary)
        d_post = [item for item in d_outcomes if item.post_intervention]
        d_pre = [item for item in d_outcomes if not item.post_intervention]
        diagnostic = {
            "label": "DIAGNOSTIC ECONOMICS PASS (benefit_scale_s=300)",
            "summary": _summary(d_outcomes),
            "pre_summary": _summary(d_pre),
            "post_summary": _summary(d_post),
            "boundary": d_boundary,
            "interventions": (
                [
                    {
                        "retained": i.retained,
                        "parameters": i.parameters,
                        "decision": i.decision.value if i.decision else None,
                        "quality_status": i.quality_status.value if i.quality_status else None,
                    }
                    for i in d_monitor.interventions
                ]
                if d_monitor is not None
                else []
            ),
            "searches": (
                [s.selection.model_dump(mode="json") for s in d_monitor.gpu_searches]
                if d_monitor is not None
                else []
            ),
            "B_pre_vs_B_post": {
                field: _compare(_values(d_pre, field), _values(d_post, field), higher, min_effect)
                for field, higher in METRICS
            },
            "matched_A_vs_B_post": _matched(
                a_by_index, {i.index: i for i in d_outcomes},
                [i.index for i in d_post], min_effect,
            ),
            "outcomes": [o.to_dict() for o in d_outcomes],
        }
        payload["diagnostic_economics"] = diagnostic

    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    prefix = args.out_prefix or str(
        ROOT / "artifacts" / "ab_effectiveness"
        / f"gpu_offload_final_effectiveness_{started.strftime('%Y%m%dT%H%M%SZ')}"
    )
    out = Path(prefix + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"json: {out}")

    # Concise console verdict.
    print("\n--- summary ---")
    for name, comp in comparisons.items():
        dec = comp["decode_tokens_per_sec"]
        base_mean = dec["baseline"]["mean"]
        measured_mean = dec["measured"]["mean"]
        if base_mean and measured_mean:
            print(
                f"{name}: decode {base_mean:.2f} -> {measured_mean:.2f} "
                f"({dec['relative_change']:+.1%}) improved={dec['improved']}"
            )
        else:
            print(f"{name}: insufficient samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
