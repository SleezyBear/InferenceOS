"""Final multi-profile effectiveness experiment.

Runs the unmanaged control and the three production optimization profiles
(BALANCED, PERFORMANCE_FIRST, MINIMIZE_OPTIMIZATION_COST) against the same stationary
workload, the same model/build/backend/context, and the same starting GPU-layer
configuration. The only intentional difference is the optimization policy.

Design:
* Every arm serves the identical fixed workload (4 prompts x 6 rounds = 24 requests,
  64 generated tokens, seed 1234, temperature 0).
* UNMANAGED: a directly-started resident llama-server at ``-ngl 8``; no optimizer.
* Managed arms: the production ``RuntimeMonitor`` runs the real GPU-offload search,
  technical comparison/decision, adaptive economics, approval, application and Phase 7
  retention against the live resident target. The retained configuration (if any) is
  then measured with a second identical workload pass (``post``).
* Arms are executed in a counterbalanced order per repetition.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\gpu_optimization_profiles_experiment.py \\
        --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf --context 4096 --gpu-layers 8 \\
        --rounds 6 --n-predict 64 --seed 1234 --repetitions 1
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
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_ab_effectiveness as ab  # noqa: E402

from inferenceos.backends.llama_cpp_resident_adapter import (  # noqa: E402
    LlamaCppResidentAdapter,
)
from inferenceos.config import Settings  # noqa: E402
from inferenceos.contracts.experiment import RuntimeConfiguration  # noqa: E402
from inferenceos.experiment.policy import build_experiment_policy  # noqa: E402
from inferenceos.models.loader import load_model_profile  # noqa: E402
from inferenceos.runtime.approval import AutoApproveApprovalProvider  # noqa: E402
from inferenceos.runtime.monitor import RuntimeMonitor  # noqa: E402
from inferenceos.runtime.notification import CollectingNotificationSink  # noqa: E402
from inferenceos.runtime.policy import (  # noqa: E402
    build_runtime_budget,
    build_runtime_policy,
)
from inferenceos.runtime.telemetry import SystemTelemetrySampler  # noqa: E402
from inferenceos.runtime.verification import (  # noqa: E402
    ControlledGpuUnderutilizationDetector,
)

METRICS: tuple[tuple[str, bool], ...] = (
    ("decode_tokens_per_sec", True),
    ("ttft_s", False),
    ("total_latency_s", False),
    ("decode_s", False),
    ("peak_rss_bytes", False),
    ("final_rss_bytes", False),
)

PROFILES: dict[str, str | None] = {
    "UNMANAGED": None,
    "BALANCED": "balanced",
    "PERFORMANCE_FIRST": "performance_first",
    "MINIMIZE_OPTIMIZATION_COST": "minimize_optimization_cost",
}
# Counterbalanced orders (each profile appears in each position once per two reps).
COUNTERBALANCED_ORDERS: tuple[tuple[str, ...], ...] = (
    ("UNMANAGED", "BALANCED", "PERFORMANCE_FIRST", "MINIMIZE_OPTIMIZATION_COST"),
    ("MINIMIZE_OPTIMIZATION_COST", "PERFORMANCE_FIRST", "BALANCED", "UNMANAGED"),
)


def _values(outcomes: Sequence[ab.RequestOutcome], field: str) -> list[float]:
    out: list[float] = []
    for item in outcomes:
        value = getattr(item, field, None)
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _summary(outcomes: Sequence[ab.RequestOutcome]) -> dict[str, Any]:
    return {
        "requests": len(outcomes),
        "successes": sum(1 for item in outcomes if item.success),
        "failures": sum(1 for item in outcomes if not item.success),
        "decode_tokens_per_sec": ab.summarize_values(_values(outcomes, "decode_tokens_per_sec")),
        "ttft_s": ab.summarize_values(_values(outcomes, "ttft_s")),
        "total_latency_s": ab.summarize_values(_values(outcomes, "total_latency_s")),
        "decode_s": ab.summarize_values(_values(outcomes, "decode_s")),
        "peak_rss_bytes": ab.summarize_values(_values(outcomes, "peak_rss_bytes")),
        "final_rss_bytes": ab.summarize_values(_values(outcomes, "final_rss_bytes")),
    }


def _compare(
    a: Sequence[float], b: Sequence[float], higher: bool, min_effect: float
) -> dict[str, Any]:
    return ab.metric_verdict(
        baseline_values=a, measured_values=b, higher_is_better=higher, min_effect=min_effect
    )


def _matched(
    a_by_index: dict[int, ab.RequestOutcome],
    b_outcomes: Sequence[ab.RequestOutcome],
    min_effect: float,
) -> dict[str, Any]:
    b_by_index: dict[int, ab.RequestOutcome] = {}
    for item in b_outcomes:
        b_by_index.setdefault(item.index, item)
    shared = sorted(set(a_by_index) & set(b_by_index))
    result: dict[str, Any] = {"indices": shared, "pairs": len(shared)}
    for field, higher in METRICS:
        pa: list[float] = []
        pb: list[float] = []
        for index in shared:
            a = a_by_index[index]
            b = b_by_index[index]
            if not a.success or not b.success:
                continue
            av = getattr(a, field, None)
            bv = getattr(b, field, None)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                pa.append(float(av))
                pb.append(float(bv))
        result[field] = _compare(pa, pb, higher, min_effect)
    return result


# --------------------------------------------------------------------------- #
# Measurement arms                                                             #
# --------------------------------------------------------------------------- #
def measure_unmanaged(
    *,
    requests: Sequence[ab.FixedRequest],
    model: Any,
    server_path: Path,
    settings: Settings,
    context: int,
    gpu_layers: int,
    stabilization_s: float,
    label: str,
    log_path: Path,
) -> tuple[list[ab.RequestOutcome], dict[str, Any]]:
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
                    test=label,
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
        "label": label,
        "server_pid": launched.pid,
        "server_identity": (launched.identity.model_dump() if launched.identity else None),
        "command": launched.command,
        "gpu_layers": gpu_layers,
        "startup_s": launched.startup_s,
        "ready_s": launched.ready_s,
        "stop": stop_report,
        "telemetry": asdict(telemetry),
    }
    return outcomes, identity


def run_managed(
    *,
    profile: str,
    mode: str,
    requests: Sequence[ab.FixedRequest],
    model: Any,
    server_path: Path,
    settings: Settings,
    context: int,
    gpu_layers: int,
    monitor_duration_s: float,
    stabilization_s: float,
) -> tuple[list[ab.RequestOutcome], Any, dict[str, Any]]:
    outcomes: list[ab.RequestOutcome] = []
    workload = ab.FixedSequenceResidentWorkload(
        fixed_requests=requests,
        outcomes=outcomes,
        test_label=profile,
        settings=settings,
        model=model,
        adapter=LlamaCppResidentAdapter(settings=settings, server=server_path),
        configuration=RuntimeConfiguration(context_length=context, gpu_offload_layers=gpu_layers),
        request_gap_s=0.05,
    )
    collector = ab.TelemetryCollector(
        jsonl_path=Path(settings.reports_dir) / f"profiles_{profile}_during.jsonl",
        interval_s=0.5,
        target_pid_provider=workload.target_pid,
    )
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=SystemTelemetrySampler(settings=settings, include_gpu=True),
        policy=build_runtime_policy(
            settings, budget=build_runtime_budget(settings, max_duration_s=monitor_duration_s)
        ),
        approval_provider=AutoApproveApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        detector=ControlledGpuUnderutilizationDetector(force=True),
        experiment_policy=build_experiment_policy(settings),
        settings=settings,
        session_notes=[
            f"Multi-profile effectiveness experiment arm {profile}.",
            "Production optimizer; optimization_mode is the only intentional difference.",
        ],
    )
    collector.start()
    monitor_run = None
    monitor_error = None
    try:
        time.sleep(max(stabilization_s, 0.0))
        result = monitor.run(duration_s=monitor_duration_s)
        monitor_run = result.run
    except Exception as exc:  # noqa: BLE001
        monitor_error = f"{type(exc).__name__}: {exc}"
    finally:
        telemetry = collector.stop()
        try:
            workload.stop()
        except Exception:  # noqa: BLE001
            pass
    extra = {"monitor_error": monitor_error, "telemetry": asdict(telemetry)}
    return outcomes, monitor_run, extra


def _economics_summary(monitor_run: Any) -> dict[str, Any]:
    """Derive the per-attempt economics from persisted run evidence.

    The runtime records the technical search (selections/comparisons), the resolved
    preference/performance state, the measured relative improvement and whether an
    intervention was applied; the required improvement for the observed state is
    computed from the recorded preference. This never fabricates a value.
    """
    if monitor_run is None:
        return {}
    preference = monitor_run.optimization_preference
    state = monitor_run.performance_state.state.value if monitor_run.performance_state else "unknown"
    required: float | None = None
    if preference is not None and monitor_run.performance_state is not None:
        from inferenceos.contracts.enums import PerformanceState

        required = preference.minimum_improvement_for(
            PerformanceState(monitor_run.performance_state.state)
        )
    attempts: list[dict[str, Any]] = []
    from inferenceos.contracts.enums import PerformanceState

    for search in monitor_run.gpu_searches:
        selected = search.selection.selected_layers
        measured: float | None = None
        if selected is not None:
            point = next((p for p in search.points if p.layers == int(selected)), None)
            if point is not None:
                measured = point.relative_improvement
        attempt_state = search.performance_state or state
        attempt_required: float | None = None
        if preference is not None:
            try:
                attempt_required = preference.minimum_improvement_for(
                    PerformanceState(attempt_state)
                )
            except ValueError:
                attempt_required = required
        attempts.append(
            {
                "search_id": search.search_id,
                "technical_decision": search.selection.kind.value,
                "selected_layers": selected,
                "measured_improvement": measured,
                "performance_state": attempt_state,
                "optimization_mode": search.optimization_mode,
                "required_improvement": attempt_required,
                "search_budget_s": search.search_budget_s,
                "estimated_search_s": search.estimated_search_s,
                "actual_search_s": search.overhead_s,
                "candidates_evaluated": search.candidates_evaluated,
                "retries": search.retries,
                "search_reason": search.selection.reason,
            }
        )
    economics_rejections = [c.trigger for c in monitor_run.cooldowns if c.trigger == "economics"]
    return {
        "mode": preference.mode.value if preference is not None else None,
        "performance_state": state,
        "required_improvement": required,
        "attempts": attempts,
        "economics_rejection_cooldowns": len(economics_rejections),
        "technical_selection": [s.selection.kind.value for s in monitor_run.gpu_searches],
        "applied": bool(monitor_run.interventions),
        "retained": bool(monitor_run.interventions and monitor_run.interventions[-1].retained),
    }


def _optimization_trace(monitor_run: Any) -> dict[str, Any]:
    if monitor_run is None:
        return {}
    return {
        "termination": monitor_run.termination.reason.value,
        "termination_detail": monitor_run.termination.detail,
        "safety": monitor_run.termination.safety,
        "errors": list(monitor_run.errors),
        "diagnosis_events": monitor_run.diagnosis_events,
        "opportunities": [
            {
                "kind": o.kind,
                "anomaly": o.anomaly.value,
                "diagnosis_kind": o.diagnosis_kind.value if o.diagnosis_kind else None,
                "confidence": o.confidence,
                "meaningful": o.meaningful,
                "persistence_samples": o.persistence_samples,
                "deviation_score": o.deviation_score,
            }
            for o in monitor_run.opportunities
        ],
        "searches": [s.model_dump(mode="json") for s in monitor_run.gpu_searches],
        "recommendations": len(monitor_run.recommendations),
        "approvals": [a.outcome.value for a in monitor_run.approvals],
        "interventions": [
            {
                "candidate_id": i.candidate_id,
                "action_id": i.action_id,
                "parameters": i.parameters,
                "decision": i.decision.value if i.decision else None,
                "retained": i.retained,
                "quality_status": i.quality_status.value if i.quality_status else None,
                "restoration_status": i.restoration_status.value,
                "validation_ok": i.validation_ok,
                "resume_verified": i.resume_verified,
            }
            for i in monitor_run.interventions
        ],
        "cooldowns": [{"trigger": c.trigger, "reason": c.reason} for c in monitor_run.cooldowns],
        "overhead": monitor_run.overhead.model_dump(mode="json"),
        "net_benefit": (
            json.loads(monitor_run.net_benefit.model_dump_json())
            if monitor_run.net_benefit is not None
            else None
        ),
        "performance_state": (
            monitor_run.performance_state.model_dump(mode="json")
            if monitor_run.performance_state is not None
            else None
        ),
        "optimization_preference": (
            monitor_run.optimization_preference.model_dump(mode="json")
            if monitor_run.optimization_preference is not None
            else None
        ),
        "resume_verifications": [r.model_dump(mode="json") for r in monitor_run.resume_verifications],
        "final_configuration": monitor_run.workload.configuration.model_dump(),
    }


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stabilization-seconds", type=float, default=30.0)
    parser.add_argument("--monitor-duration-seconds", type=float, default=240.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--order-index",
        type=int,
        default=0,
        help="starting index into the counterbalanced orders (for a single repetition)",
    )
    parser.add_argument(
        "--long-phase",
        action="store_true",
        help=(
            "append the project's genuine short->long workload-change phase, served "
            "identically by every arm (exercises the runtime workload-change path)"
        ),
    )
    parser.add_argument(
        "--arms",
        default=",".join(PROFILES),
        help="comma-separated subset of arms to run (in the given order)",
    )
    parser.add_argument("--benefit-scale-s", type=float, default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def _settings_for(mode: str | None, *, benefit_scale_s: float | None) -> Settings:
    settings = Settings()
    settings.ensure_directories()
    if mode is not None:
        settings.optimization_mode = mode
    if benefit_scale_s is not None:
        settings.phase10_benefit_scale_s = float(benefit_scale_s)
    # One bounded search per arm; allow the intervention to complete.
    settings.phase10_max_cumulative_intervention_s = 1800.0
    return settings


def main() -> int:
    args = _parse_args()
    model = load_model_profile(Path(args.model))
    server_path = Settings().llama_cpp_dir / "llama-server.exe"
    requests = ab.build_fixed_requests(
        rounds=args.rounds, n_predict=args.n_predict, seed=args.seed,
        temperature=0.0, ignore_eos=False, long_phase=args.long_phase,
    )
    min_effect = float(Settings().phase7_min_effect)
    started = datetime.now(timezone.utc)
    out_dir = Path(
        args.out_dir
        or ROOT / "artifacts" / "ab_effectiveness"
        / f"profiles_run_{started.strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {out_dir}")
    print(f"workload: {len(requests)} requests (rounds={args.rounds}, n_predict={args.n_predict}, seed={args.seed})")

    try:
        model_sha = ab.sha256_file(Path(model.path))
    except OSError:
        model_sha = None

    payload: dict[str, Any] = {
        "schema": "gpu_optimization_profiles.v1",
        "started_at": started.isoformat(),
        "environment": {
            "model": model.name,
            "model_path": model.path,
            "model_sha256": model_sha,
            "context_length": args.context,
            "starting_gpu_layers": args.gpu_layers,
            "rounds": args.rounds,
            "n_predict": args.n_predict,
            "seed": args.seed,
            "request_count": len(requests),
            "server_path": str(server_path),
            "stabilization_seconds": args.stabilization_seconds,
            "monitor_duration_seconds": args.monitor_duration_seconds,
            "benefit_scale_s": args.benefit_scale_s,
            "phase7_min_effect": min_effect,
            "phase7_noise_tolerance": Settings().phase7_noise_tolerance,
            "inferenceos_git": _git_revision(),
        },
        "repetitions": [],
    }

    for rep in range(max(args.repetitions, 1)):
        requested = {item.strip() for item in str(args.arms).split(",") if item.strip()}
        order = [
            arm for arm in COUNTERBALANCED_ORDERS[
                (args.order_index + rep) % len(COUNTERBALANCED_ORDERS)
            ]
            if arm in requested
        ]
        rep_record: dict[str, Any] = {"repetition": rep + 1, "arm_order": order, "arms": {}}
        print(f"\n===== repetition {rep + 1} order={order} =====")
        for arm in order:
            mode = PROFILES[arm]
            print(f"--- arm {arm} (mode={mode}) ---")
            if mode is None:
                settings = _settings_for(None, benefit_scale_s=args.benefit_scale_s)
                outcomes, identity = measure_unmanaged(
                    requests=requests, model=model, server_path=server_path,
                    settings=settings, context=args.context, gpu_layers=args.gpu_layers,
                    stabilization_s=args.stabilization_seconds, label="UNMANAGED",
                    log_path=Path(settings.logs_dir) / f"profiles_{arm}_server.log",
                )
                arm_record: dict[str, Any] = {
                    "profile": arm,
                    "mode": None,
                    "pre": _summary(outcomes),
                    "post": None,
                    "outcomes": [o.to_dict() for o in outcomes],
                    "identity": identity,
                    "optimization": None,
                    "economics": None,
                }
                print(f"  {arm}: {arm_record['pre']['decode_tokens_per_sec']['mean']:.2f} tok/s")
            else:
                settings = _settings_for(mode, benefit_scale_s=args.benefit_scale_s)
                pre_outcomes, monitor_run, extra = run_managed(
                    profile=arm, mode=mode, requests=requests, model=model,
                    server_path=server_path, settings=settings, context=args.context,
                    gpu_layers=args.gpu_layers,
                    monitor_duration_s=args.monitor_duration_seconds,
                    stabilization_s=args.stabilization_seconds,
                )
                trace = _optimization_trace(monitor_run)
                retained = bool(
                    monitor_run is not None
                    and monitor_run.interventions
                    and monitor_run.interventions[-1].retained
                )
                final_layers = (
                    monitor_run.workload.configuration.gpu_offload_layers
                    if monitor_run is not None
                    else args.gpu_layers
                )
                post_outcomes: list[ab.RequestOutcome] = []
                post_identity: dict[str, Any] | None = None
                if retained and final_layers is not None and int(final_layers) != args.gpu_layers:
                    print(f"  retained {final_layers} layers; measuring post")
                    post_outcomes, post_identity = measure_unmanaged(
                        requests=requests, model=model, server_path=server_path,
                        settings=settings, context=args.context, gpu_layers=int(final_layers),
                        stabilization_s=args.stabilization_seconds,
                        label=f"{arm}_POST",
                        log_path=Path(settings.logs_dir) / f"profiles_{arm}_post_server.log",
                    )
                arm_record = {
                    "profile": arm,
                    "mode": mode,
                    "pre": _summary(pre_outcomes),
                    "post": _summary(post_outcomes) if post_outcomes else None,
                    "post_identity": post_identity,
                    "outcomes": [o.to_dict() for o in pre_outcomes],
                    "post_outcomes": [o.to_dict() for o in post_outcomes],
                    "retained": retained,
                    "final_gpu_layers": final_layers,
                    "optimization": trace,
                    "economics": _economics_summary(monitor_run),
                    "telemetry": extra.get("telemetry"),
                    "monitor_error": extra.get("monitor_error"),
                }
                print(
                    f"  {arm}: termination={trace.get('termination')} "
                    f"searches={len(trace.get('searches', []))} retained={retained} "
                    f"final_layers={final_layers} gpu_search_s={trace.get('overhead', {}).get('gpu_search_s')}"
                )
            rep_record["arms"][arm] = arm_record
            (out_dir / f"rep{rep + 1}_{arm}.json").write_text(
                json.dumps(arm_record, indent=1, default=str), encoding="utf-8"
            )
        payload["repetitions"].append(rep_record)

    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    out = out_dir / "profiles_raw.json"
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"\nraw: {out}")
    return 0


def _git_revision() -> str | None:
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=10,
        )
        return completed.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    raise SystemExit(main())
