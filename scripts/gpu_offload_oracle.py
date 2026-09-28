"""Ground-truth oracle sweep for GPU-offload layer selection.

Independently evaluates EVERY feasible GPU-layer count on the live resident
``llama-server`` (via the same controlled-reload measurement seam the production
optimizer uses) and applies the project's existing comparison/decision/selection
machinery. Then runs the *unmodified* production search from the same baseline and
compares the two.

This is not a second optimizer: it reuses ``optimization.search`` evaluation,
``experiment.comparison``/``decision`` and ``select_configuration`` unchanged. The
only difference from the production search is the sampling policy (exhaustive vs the
configured sampling plan).

Run:
    .\\.venv\\Scripts\\python.exe scripts\\gpu_offload_oracle.py \\
        --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf \\
        --context 4096 --gpu-layers 8 --n-predict 64 --seed 1234
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.enums import CandidateEvaluationStatus, WorkloadKind
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.experiment.comparison import compare  # noqa: F401 (documented reuse)
from inferenceos.experiment.decision import decide  # noqa: F401 (documented reuse)
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.experiment.quality import validate_quality
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.policy import build_policy, OptimizationPolicy
from inferenceos.optimization.search.evaluation import (
    PointMeasurement,
    RobustPointEvaluator,
    RuntimeConfigurationPointEvaluator,
)
from inferenceos.optimization.search.resources import (
    build_resource_envelope,
    total_vram_from_hardware,
)
from inferenceos.optimization.search.runner import run_gpu_offload_search
from inferenceos.optimization.search.selection import select_configuration
from inferenceos.profiling.runner import BaselineProfiler
from inferenceos.profiling.workloads import BenchmarkWorkload
from inferenceos.runtime.gpu_search import (
    INCREASE_GPU_OFFLOAD_ACTION,
    ResidentMeasurementAdapter,
    make_applied_configuration_probe,
    make_contamination_probe,
)
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.telemetry.system import sample_system_state

_SHORT_PROMPT = "In one sentence, explain what a key value cache is."
_LONG_CORPUS = (
    "Inference operating systems schedule model execution across heterogeneous "
    "hardware. A key value cache stores attention state so that previously seen "
    "tokens do not require recomputation. Throughput depends on memory bandwidth, "
    "compute occupancy, and the size of the working set. The latency to the first "
    "token is dominated by prompt processing, while later tokens are memory bound. "
    "Profiling separates loading, prefill, and decode so that each phase can be "
    "compared under identical configurations. "
)
_LONG_PROMPT = (
    _LONG_CORPUS * 4
    + "Using the passage above, answer in one sentence: what does a key value cache store?"
)

_UNSAFE = {
    CandidateEvaluationStatus.OOM,
    CandidateEvaluationStatus.VRAM_EXCEEDED,
    CandidateEvaluationStatus.PAGING,
}


def build_live_envelope(
    *,
    hardware: object,
    model: object,
    context_length: int,
    current_layers: int,
    policy: object,
) -> object:
    state = sample_system_state(include_gpu=True, interval_s=0.5)
    used = None
    if state.gpu_memory:
        used = max((m.memory_used_bytes or 0) for m in state.gpu_memory)
    # now must be taken AFTER sampling so the telemetry age is non-negative.
    now = datetime.now(timezone.utc)
    return build_resource_envelope(
        total_vram_bytes=total_vram_from_hardware(hardware),  # type: ignore[arg-type]
        used_vram_bytes=used,
        telemetry_available=state.gpu_telemetry_available,
        telemetry_observed_at=state.gpu_observed_at,
        telemetry_source=state.gpu_source,
        now=now,
        model=model,  # type: ignore[arg-type]
        context_length=context_length,
        current_layers=current_layers,
        total_layers=model.n_layers,  # type: ignore[attr-defined]
        policy=policy,  # type: ignore[arg-type]
    )


def evaluate_sweep(
    *,
    evaluator: RuntimeConfigurationPointEvaluator,
    current: int,
    feasible_max: int,
) -> tuple[PointMeasurement, dict[int, PointMeasurement], list[int]]:
    baseline = evaluator.measure(current)
    points: dict[int, PointMeasurement] = {current: baseline}
    skipped: list[int] = []
    for layer in range(current + 1, feasible_max + 1):
        point = evaluator.measure(layer)
        points[layer] = point
        if point.status in _UNSAFE:
            # Do not evaluate higher layer counts past a genuine safety boundary.
            skipped = list(range(layer + 1, feasible_max + 1))
            break
    return baseline, points, skipped


def quality_for(
    *,
    baseline: PointMeasurement,
    point: PointMeasurement,
    benchmark: BenchmarkWorkload,
    policy: object,
    settings: Settings,
    model: object,
) -> object:
    if baseline.measurement is None or point.measurement is None:
        return None
    return validate_quality(
        baseline=baseline.measurement,
        candidate=point.measurement,
        workload=benchmark,
        policy=policy,  # type: ignore[arg-type]
        settings=settings,
        action_id=INCREASE_GPU_OFFLOAD_ACTION,
        diagnosed_bottleneck="gpu_underutilization",
        model=model,  # type: ignore[arg-type]
    )


def level_summary(point: PointMeasurement) -> dict[str, object]:
    return {
        "layers": point.layers,
        "status": point.status.value,
        "valid": point.valid,
        "safe": point.safe,
        "samples": point.samples,
        "warmup_runs": point.warmup_runs,
        "decode_tokens_per_sec": point.metric_mean("decode_tokens_per_sec"),
        "ttft_s": point.metric_mean("ttft_s"),
        "total_latency_s": point.metric_mean("total_latency_s"),
        "process_cpu_percent": point.metric_mean("process_cpu_percent"),
        "gpu_utilization_percent": point.metric_mean("gpu_utilization_percent"),
        "gpu_memory_used_bytes": point.metric_mean("gpu_memory_used_bytes"),
        "peak_rss_bytes": point.metric_mean("peak_rss_bytes"),
        "hard_page_faults_per_sec": point.metric_mean("hard_page_faults_per_sec"),
        "reasons": list(point.reasons[-3:]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=8)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--workloads", default="short,long")
    parser.add_argument("--per-candidate-time-limit-s", type=float, default=150.0)
    parser.add_argument("--out-prefix", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = Settings()
    settings.ensure_directories()
    model = load_model_profile(Path(args.model))
    experiment_policy = build_experiment_policy(settings)
    optimization_policy: OptimizationPolicy = build_policy(settings)
    search_policy = optimization_policy.search

    base_configuration = RuntimeConfiguration(
        context_length=args.context, gpu_offload_layers=args.gpu_layers
    )
    workload = ResidentActiveWorkload(
        settings=settings,
        model=model,
        workload_id="short-short",
        configuration=base_configuration,
        request_gap_s=0.05,
    )
    workload.start()
    # The production monitor pauses/drains the workload before any controlled reload, and
    # so must the oracle: the harness performs the measurements directly, and a live serve
    # thread would race the resident reloads.
    workload.pause()
    harness = BaselineProfiler(
        settings=settings,
        adapter=ResidentMeasurementAdapter(workload),
        hardware=None,
    )

    workloads: dict[str, BenchmarkWorkload] = {
        "short": BenchmarkWorkload(
            workload_id="oracle-decode", kind=WorkloadKind.SHORT,
            description="decode-sensitive oracle workload",
            prompt=_SHORT_PROMPT, n_predict=args.n_predict, seed=args.seed,
            temperature=0.0, min_context_tokens=64, ignore_eos=False, version="oracle.1",
        ),
        "long": BenchmarkWorkload(
            workload_id="oracle-prefill", kind=WorkloadKind.LONG,
            description="prefill-sensitive oracle workload",
            prompt=_LONG_PROMPT, n_predict=args.n_predict, seed=args.seed,
            temperature=0.0, min_context_tokens=640, ignore_eos=False, version="oracle.1",
        ),
    }
    selected_workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]

    started = datetime.now(timezone.utc)
    payload: dict[str, object] = {
        "schema": "gpu_offload_oracle.v1",
        "started_at": started.isoformat(),
        "model": {"name": model.name, "path": model.path, "n_layers": model.n_layers,
                  "weight_bytes": model.weight_bytes},
        "context_length": args.context,
        "starting_gpu_layers": args.gpu_layers,
        "n_predict": args.n_predict,
        "seed": args.seed,
        "model_file_size_bytes": Path(args.model).stat().st_size,
        "workloads": {},
    }

    try:
        for name in selected_workloads:
            benchmark = workloads[name]
            # Reset to the exact baseline and build the real envelope from live telemetry.
            workload.set_configuration(base_configuration)
            envelope = build_live_envelope(
                hardware=harness.hardware, model=model, context_length=args.context,
                current_layers=args.gpu_layers, policy=search_policy,
            )
            current = envelope.current_layers or args.gpu_layers
            feasible_max = envelope.feasible_max or current
            evaluator = RobustPointEvaluator(
                RuntimeConfigurationPointEvaluator(
                    profiler=harness, model=model, workload=benchmark,
                    experiment_policy=experiment_policy, context_length=args.context,
                    envelope=envelope, base_configuration=base_configuration, seed=args.seed,
                    paging_per_sec_ceiling=search_policy.paging_per_sec_ceiling,
                    max_primary_cv=search_policy.max_measure_cv,
                    contamination_probe=make_contamination_probe(workload),
                    applied_configuration_probe=make_applied_configuration_probe(workload),
                    deadline=time.perf_counter()
                    + args.per_candidate_time_limit_s * (feasible_max - current + 2),
                ),
                max_attempts=search_policy.max_measure_attempts,
            )
            print(f"=== oracle sweep: workload={name} feasible={current}-{feasible_max} ===")

            # --- exhaustive oracle sweep ---
            baseline, points, skipped = evaluate_sweep(
                evaluator=evaluator, current=current, feasible_max=feasible_max,
            )
            quality_map = {}
            for layer, point in points.items():
                if layer == current:
                    continue
                quality_map[layer] = quality_for(
                    baseline=baseline, point=point, benchmark=benchmark,
                    policy=experiment_policy, settings=settings, model=model,
                )
            candidates = [points[layer] for layer in sorted(points) if layer != current]
            outcome = select_configuration(
                baseline=baseline, candidates=candidates,
                baseline_configuration=base_configuration,
                candidate_configurations={
                    p.layers: evaluator.configuration_for(p.layers) for p in candidates
                },
                experiment_policy=experiment_policy, search_policy=search_policy,
                quality_by_layers={k: v for k, v in quality_map.items() if v is not None},
            )
            decisions = {
                item.point.layers: {
                    "status": item.point.status.value,
                    "valid": item.point.valid,
                    "safe": item.point.safe,
                    "decision": item.decision.kind.value,
                    "keep": item.decision.keep,
                    "relative_decode": (
                        item.comparison.deltas["decode_tokens_per_sec"].relative_delta
                        if "decode_tokens_per_sec" in item.comparison.deltas else None
                    ),
                    "relative_latency": (
                        item.comparison.deltas["total_latency_s"].relative_delta
                        if "total_latency_s" in item.comparison.deltas else None
                    ),
                    "quality": item.quality.status.value,
                    "reason": item.decision.reason,
                }
                for item in outcome.outcomes
            }
            acceptable = [
                layer for layer, d in decisions.items()
                if d["keep"] and d["safe"] and d["valid"]
            ]
            best_improvement_layer = None
            if acceptable:
                best_improvement_layer = max(
                    acceptable, key=lambda layer: decisions[layer]["relative_decode"] or 0.0
                )

            # --- production search (unmodified policy) from the same baseline ---
            workload.set_configuration(base_configuration)
            production = run_gpu_offload_search(
                envelope=envelope,
                evaluator=RobustPointEvaluator(
                    RuntimeConfigurationPointEvaluator(
                        profiler=harness, model=model, workload=benchmark,
                        experiment_policy=experiment_policy, context_length=args.context,
                        envelope=envelope, base_configuration=base_configuration, seed=args.seed,
                        paging_per_sec_ceiling=search_policy.paging_per_sec_ceiling,
                        max_primary_cv=search_policy.max_measure_cv,
                        contamination_probe=make_contamination_probe(workload),
                        applied_configuration_probe=make_applied_configuration_probe(workload),
                    ),
                    max_attempts=search_policy.max_measure_attempts,
                ),
                search_policy=search_policy,
                experiment_policy=experiment_policy,
                model_ref=model.name,
                quality_evaluator=lambda b, c: quality_for(
                    baseline=b, point=c, benchmark=benchmark,
                    policy=experiment_policy, settings=settings, model=model,
                ),
            )
            workload.set_configuration(base_configuration)

            payload["workloads"][name] = {
                "benchmark": {
                    "workload_id": benchmark.workload_id,
                    "prompt_sha256": benchmark.prompt_hash,
                    "n_predict": benchmark.n_predict,
                    "prompt_chars": len(benchmark.prompt),
                },
                "envelope": envelope.model_dump(mode="json"),
                "oracle": {
                    "evaluated_layers": sorted(points),
                    "skipped_after_unsafe": skipped,
                    "baseline": level_summary(baseline),
                    "levels": [level_summary(points[layer]) for layer in sorted(points)],
                    "decisions": decisions,
                    "acceptable_layers": sorted(acceptable),
                    "best_improvement_layer": best_improvement_layer,
                    "selection": outcome.selection.model_dump(mode="json"),
                },
                "production": production.model_dump(mode="json"),
            }
            print(
                f"    oracle selected={outcome.selection.selected_layers} "
                f"kind={outcome.selection.kind.value} best_improve={best_improvement_layer} "
                f"production selected={production.selection.selected_layers} "
                f"kind={production.selection.kind.value} sampled={production.sampled_layers}"
            )
    finally:
        try:
            workload.set_configuration(base_configuration)
        except Exception:  # noqa: BLE001 - restore is best-effort at shutdown
            pass
        workload.stop()

    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    prefix = args.out_prefix or f"artifacts/ab_effectiveness/gpu_offload_oracle_{started.strftime('%Y%m%dT%H%M%SZ')}"
    out_path = Path(prefix + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"json: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
