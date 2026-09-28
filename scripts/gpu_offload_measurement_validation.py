"""Real Windows validation of measurement robustness for the GPU-offload search.

Runs the *production* measurement machinery (resident controlled reload, real
telemetry, Phase 7 policy/quality) with the new bounded invalidity-only retries:

  A. Normal candidate  -> a clean candidate completes without unnecessary retry.
  B. Controlled invalidation -> one attempt of a chosen candidate is forced invalid
     through a controlled validation seam; the retry measures the real server and the
     invalid attempt is preserved in evidence.
  C. Long/prefill production search -> run the actual production search and report the
     previously-degraded layer-16/20 candidates.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\gpu_offload_measurement_validation.py \\
        --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf --context 4096 --gpu-layers 8
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.enums import CandidateEvaluationStatus, WorkloadKind
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.policy import build_policy, OptimizationPolicy
from inferenceos.optimization.search.evaluation import (
    PointEvaluator,
    PointMeasurement,
    RobustPointEvaluator,
    RuntimeConfigurationPointEvaluator,
)
from inferenceos.optimization.search.resources import (
    build_resource_envelope,
    total_vram_from_hardware,
)
from inferenceos.optimization.search.runner import run_gpu_offload_search
from inferenceos.profiling.runner import BaselineProfiler
from inferenceos.profiling.workloads import BenchmarkWorkload
from inferenceos.runtime.gpu_search import (
    make_applied_configuration_probe,
    make_contamination_probe,
    make_quality_evaluator,
    ResidentMeasurementAdapter,
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


class ForcedInvalidOnce:
    """Controlled validation seam: the first attempt at ``target_layer`` is recorded as
    an invalid/contaminated attempt (the real measurement still runs, but is marked
    non-authoritative), so the retry path can be demonstrated without faking results.
    """

    def __init__(self, inner: PointEvaluator, target_layer: int) -> None:
        self._inner = inner
        self._target = target_layer
        self._fired = False

    def configuration_for(self, layers: int) -> RuntimeConfiguration:
        return self._inner.configuration_for(layers)

    def measure(self, layers: int) -> PointMeasurement:
        point = self._inner.measure(layers)
        if layers == self._target and not self._fired:
            self._fired = True
            return replace(
                point,
                status=CandidateEvaluationStatus.INVALID,
                valid=False,
                safe=True,
                reasons=list(point.reasons) + ["controller_forced_invalid_attempt"],
                contamination_reasons=["controller_forced_invalid_attempt"],
            )
        return point


def build_live_envelope(*, hardware, model, context_length, current_layers, policy):
    state = sample_system_state(include_gpu=True, interval_s=0.5)
    used = None
    if state.gpu_memory:
        used = max((m.memory_used_bytes or 0) for m in state.gpu_memory)
    now = datetime.now(timezone.utc)
    return build_resource_envelope(
        total_vram_bytes=total_vram_from_hardware(hardware),
        used_vram_bytes=used,
        telemetry_available=state.gpu_telemetry_available,
        telemetry_observed_at=state.gpu_observed_at,
        telemetry_source=state.gpu_source,
        now=now,
        model=model,
        context_length=context_length,
        current_layers=current_layers,
        total_layers=model.n_layers,
        policy=policy,
    )


# The resident workload and search policy are created in ``main`` and referenced by the
# measurement helpers.
_RESIDENT: ResidentActiveWorkload | None = None
_SEARCH = None


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
    global _RESIDENT, _SEARCH
    args = parse_args()
    settings = Settings()
    settings.ensure_directories()
    model = load_model_profile(Path(args.model))
    experiment_policy = build_experiment_policy(settings)
    optimization_policy: OptimizationPolicy = build_policy(settings)
    _SEARCH = optimization_policy.search
    base = RuntimeConfiguration(context_length=args.context, gpu_offload_layers=args.gpu_layers)

    resident = ResidentActiveWorkload(
        settings=settings, model=model, workload_id="short-short",
        configuration=base, request_gap_s=0.05,
    )
    resident.start()
    resident.pause()
    _RESIDENT = resident
    harness = BaselineProfiler(settings=settings, adapter=ResidentMeasurementAdapter(resident), hardware=None)

    benchmarks = {
        "short": BenchmarkWorkload(
            workload_id="robust-decode", kind=WorkloadKind.SHORT,
            description="decode-sensitive", prompt=_SHORT_PROMPT, n_predict=args.n_predict,
            seed=args.seed, temperature=0.0, min_context_tokens=64, ignore_eos=False,
            version="robust.1",
        ),
        "long": BenchmarkWorkload(
            workload_id="robust-prefill", kind=WorkloadKind.LONG,
            description="prefill-sensitive", prompt=_LONG_PROMPT, n_predict=args.n_predict,
            seed=args.seed, temperature=0.0, min_context_tokens=640, ignore_eos=False,
            version="robust.1",
        ),
    }

    started = datetime.now(timezone.utc)
    payload: dict[str, object] = {
        "schema": "gpu_offload_measurement_robustness.v1",
        "started_at": started.isoformat(),
        "model": model.name,
        "context_length": args.context,
        "starting_gpu_layers": args.gpu_layers,
        "max_measure_attempts": _SEARCH.max_measure_attempts,
        "max_measure_cv": _SEARCH.max_measure_cv,
        "paging_per_sec_ceiling": _SEARCH.paging_per_sec_ceiling,
        "workloads": {},
    }

    def run_search(benchmark, *, seam_layer: int | None) -> object:
        resident.set_configuration(base)
        envelope = build_live_envelope(
            hardware=harness.hardware, model=model, context_length=args.context,
            current_layers=args.gpu_layers, policy=_SEARCH,
        )
        inner = RuntimeConfigurationPointEvaluator(
            profiler=harness, model=model, workload=benchmark,
            experiment_policy=experiment_policy, context_length=args.context,
            envelope=envelope, base_configuration=base, seed=args.seed,
            paging_per_sec_ceiling=_SEARCH.paging_per_sec_ceiling,
            max_primary_cv=_SEARCH.max_measure_cv,
            contamination_probe=make_contamination_probe(resident),
            applied_configuration_probe=make_applied_configuration_probe(resident),
        )
        evaluator: PointEvaluator = inner
        if seam_layer is not None:
            evaluator = ForcedInvalidOnce(inner, seam_layer)
        evaluator = RobustPointEvaluator(evaluator, max_attempts=_SEARCH.max_measure_attempts)
        result = run_gpu_offload_search(
            envelope=envelope, evaluator=evaluator, search_policy=_SEARCH,
            experiment_policy=experiment_policy, model_ref=model.name,
            quality_evaluator=make_quality_evaluator(
                workload=resident, settings=settings, experiment_policy=experiment_policy
            ),
        )
        resident.set_configuration(base)
        return result

    try:
        for name in [w.strip() for w in args.workloads.split(",") if w.strip()]:
            benchmark = benchmarks[name]
            # A: normal production search (no injected invalidation).
            normal = run_search(benchmark, seam_layer=None)
            # B: controlled invalidation on one sampled candidate (short workload only).
            seam = None
            if name == "short":
                sampled = [v for v in normal.sampled_layers if v != normal.resource.current_layers]
                if sampled:
                    forced = run_search(benchmark, seam_layer=sampled[0])
                    seam = {"target_layer": sampled[0], "result": forced.model_dump(mode="json")}
            payload["workloads"][name] = {
                "normal": normal.model_dump(mode="json"),
                "controlled_invalidation": seam,
            }
            print(
                f"workload={name} selected={normal.selection.selected_layers} "
                f"kind={normal.selection.kind.value} sampled={normal.sampled_layers}"
            )
            for point in normal.points:
                print(
                    f"    layer={point.layers:>3} status={point.status.value:<16} "
                    f"valid={point.valid} attempts={point.attempts} "
                    f"invalid_attempts={point.invalid_attempts} decode="
                    f"{point.metrics.get('decode_tokens_per_sec').mean if point.metrics.get('decode_tokens_per_sec') else None}"
                )
    finally:
        try:
            resident.set_configuration(base)
        except Exception:  # noqa: BLE001
            pass
        resident.stop()

    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    prefix = args.out_prefix or f"artifacts/ab_effectiveness/gpu_offload_measurement_robustness_{started.strftime('%Y%m%dT%H%M%SZ')}"
    out = Path(prefix + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"json: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
