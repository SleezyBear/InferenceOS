"""Is "as many graphics card layers as fit" really the fastest? (step 5 check)

Measures writing speed and time to the first word at a spread of graphics card layer
counts, from none up to every layer - including counts *above* what InferenceOS
estimates will fit - at one fixed context size. Each count is measured with the same
always-running model and the same test prompt that ``tune`` uses. Disturbed
measurements are repeated. With ``--passes 2`` (the default) the counts are measured
once going up and once going down, so slow drift (heat, background work) evens out.

Nothing is saved as a setup and nothing is kept: this only measures.

Run (from the repo folder, one line in PowerShell; please leave the computer alone):
    .\\.venv\\Scripts\\python.exe scripts\\check_layer_range.py --model models\\ab\\Qwen3.5-9B-Q4_K_M.gguf --context 4096
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.hardware.profiler import detect_hardware_profile
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.search.policy import build_search_policy
from inferenceos.profiling.runner import BaselineProfiler
from inferenceos.profiling.workloads import get_workload
from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.tune.fit import layers_that_fit
from inferenceos.tune.machine import check_machine, gib
from inferenceos.tune.measure import DisturbanceWatch, measure_steady


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model path")
    parser.add_argument("--context", type=int, required=True, help="fixed context size")
    parser.add_argument("--step", type=int, default=4, help="layer step (default 4)")
    parser.add_argument("--passes", type=int, default=2, help="1 = up only, 2 = up then down")
    parser.add_argument("--workload", default=None, help="test prompt (default: tune's)")
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="measured runs per layer count, always all of them (default: the usual policy)",
    )
    parser.add_argument("--out", default=None, help="report path")
    return parser.parse_args()


def _mean(measurement: object, name: str) -> float | None:
    stats = getattr(measurement, "metrics", {}).get(name)
    return float(stats.mean) if stats is not None and stats.mean is not None else None


def _range(item: dict[str, object]) -> str:
    low, high = item.get("slowest_run"), item.get("fastest_run")
    if low is None or high is None:
        return "-"
    return f"{float(str(low)):.2f}-{float(str(high)):.2f}"


def main() -> int:
    args = _parse_args()
    settings = Settings()
    settings.ensure_directories()
    model_path = Path(args.model)
    model = load_model_profile(model_path)
    hardware = detect_hardware_profile(settings)
    machine = check_machine(settings, hardware=hardware)
    search_policy = build_search_policy(settings)
    fit = layers_that_fit(
        model=model,
        context_length=args.context,
        total_graphics_memory_bytes=machine.total_graphics_memory_bytes,
        used_before_loading_bytes=machine.used_graphics_memory_bytes,
        policy=search_policy,
    )
    total = int(model.n_layers)
    counts = sorted(
        set(range(0, total + 1, max(args.step, 1)))
        | {total}
        | ({fit.layers} if fit.layers is not None else set())
    )
    plan = list(counts)
    if args.passes >= 2:
        plan += list(reversed(counts))

    print("=== Layer range check ===")
    print(f"model   : {model_path} ({total} layers)")
    print(f"context : {args.context} (fixed)")
    print(
        f"graphics: {gib(machine.free_graphics_memory_bytes)} free of "
        f"{gib(machine.total_graphics_memory_bytes)}; InferenceOS estimates "
        f"{fit.layers if fit.layers is not None else 'unknown'} layers fit"
    )
    print(f"counts  : {counts} x {min(args.passes, 2)} pass(es)")
    print("Please leave the computer alone until it finishes.")
    print()

    workload = get_workload(args.workload or settings.tune_workload)
    resident = ResidentActiveWorkload(
        settings=settings,
        model=model,
        adapter=LlamaCppResidentAdapter(settings=settings),
        workload=workload,
        configuration=RuntimeConfiguration(context_length=args.context, gpu_offload_layers=0),
    )
    harness = BaselineProfiler(
        settings=settings, adapter=ResidentMeasurementAdapter(resident), hardware=hardware
    )
    policy = build_experiment_policy(settings)
    if args.runs is not None and args.runs >= 1:
        # A fixed number of runs per count (no early stop), so every count is measured
        # the same way.
        policy = policy.model_copy(
            update={
                "measured_repetitions": args.runs,
                "max_repetitions": args.runs,
                "min_valid_samples": args.runs,
                "max_total_runs": args.runs + policy.warmup_repetitions + 2,
            }
        )
    watch = DisturbanceWatch(
        logical_cores=hardware.cpu.logical_cores,
        limit_percent=float(settings.tune_disturbance_cpu_percent),
        max_repeats=int(settings.tune_max_disturbed_repeats),
        on_repeat=lambda reason: print(f"    {reason}; repeating"),
    )

    rows: list[dict[str, object]] = []
    does_not_fit: int | None = None
    try:
        for index, layers in enumerate(plan, start=1):
            # Pass 1 walks up, pass 2 walks back down (only when --passes >= 2).
            pass_no = 1 if index <= len(counts) else 2
            if does_not_fit is not None and layers >= does_not_fit:
                print(f"[{index}/{len(plan)}] {layers} layers: skipped (does not fit)")
                rows.append(
                    {"layers": layers, "ok": False, "skipped": "does not fit", "pass": pass_no}
                )
                continue
            print(f"[{index}/{len(plan)}] {layers} layers ...", flush=True)
            config = RuntimeConfiguration(context_length=args.context, gpu_offload_layers=layers)
            try:
                measurement = measure_steady(
                    harness=harness,
                    model=model,
                    workload=workload,
                    config=config,
                    policy=policy,
                    phase=f"layer-range-{layers}",
                    watch=watch,
                )
            except Exception as exc:  # noqa: BLE001 - one bad count must not stop the check
                rows.append(
                    {
                        "layers": layers,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "pass": pass_no,
                    }
                )
                print(f"    failed: {type(exc).__name__}: {exc}")
                continue
            row: dict[str, object] = {
                "layers": layers,
                "pass": pass_no,
                "ok": measurement.valid_runs >= policy.min_valid_samples,
                "good_runs": measurement.valid_runs,
                "writing_speed": _mean(measurement, "decode_tokens_per_sec"),
                "writing_speed_runs": [
                    float(metrics.decode_tokens_per_sec)
                    for metrics in measurement.run_metrics
                    if metrics.decode_tokens_per_sec is not None
                ],
                "first_word_s": _mean(measurement, "ttft_s"),
                "prompt_speed": _mean(measurement, "prompt_tokens_per_sec"),
                "graphics_memory_bytes": _mean(measurement, "gpu_memory_used_bytes"),
                "memory_bytes": _mean(measurement, "peak_rss_bytes"),
                "notes": list(measurement.notes[-3:]),
            }
            rows.append(row)
            if any("out of memory" in note for note in measurement.notes):
                does_not_fit = layers if does_not_fit is None else min(does_not_fit, layers)
                print(
                    f"    does not fit on the graphics card; {layers} or more layers will be "
                    "skipped (putting the previous setup back takes a moment)"
                )
                continue
            graphics = row["graphics_memory_bytes"]
            each = ", ".join(f"{value:.2f}" for value in cast(list[float], row["writing_speed_runs"]))
            print(
                f"    writing speed {row['writing_speed']} (runs: {each}), "
                f"first word {row['first_word_s']} s, "
                f"graphics memory {gib(int(float(str(graphics)))) if graphics else 'unknown'}"
            )
    finally:
        resident.stop()

    # Average the passes per layer count.
    summary: list[dict[str, object]] = []
    for layers in counts:
        good = [row for row in rows if row["layers"] == layers and row.get("ok")]
        speeds = [float(str(row["writing_speed"])) for row in good if row.get("writing_speed") is not None]
        firsts = [float(str(row["first_word_s"])) for row in good if row.get("first_word_s") is not None]
        every_run = [
            value for row in good for value in cast(list[float], row.get("writing_speed_runs", []))
        ]
        summary.append(
            {
                "layers": layers,
                "measured": len(good),
                "writing_speed": statistics.fmean(speeds) if speeds else None,
                "first_word_s": statistics.fmean(firsts) if firsts else None,
                "spread_between_passes": (
                    (max(speeds) - min(speeds)) / statistics.fmean(speeds)
                    if len(speeds) >= 2
                    else None
                ),
                "slowest_run": min(every_run) if every_run else None,
                "fastest_run": max(every_run) if every_run else None,
                "runs_measured": len(every_run),
                "above_estimated_fit": fit.layers is not None and layers > fit.layers,
                "does_not_fit": does_not_fit is not None and layers >= does_not_fit,
            }
        )

    print()
    print(
        "layers | writing speed (tokens/s) | slowest-fastest run | first word (s) "
        "| passes differ by | above fit?"
    )
    for item in summary:
        speed = item["writing_speed"]
        first = item["first_word_s"]
        spread = item["spread_between_passes"]
        print(
            f"{item['layers']:>6} | "
            f"{(f'{float(str(speed)):.2f}' if speed is not None else ('does not fit' if item['does_not_fit'] else 'failed')):>24} | "
            f"{_range(item):>19} | "
            f"{(f'{float(str(first)):.3f}' if first is not None else '-'):>14} | "
            f"{(f'{float(str(spread)):.0%}' if spread is not None else '-'):>16} | "
            f"{'yes' if item['above_estimated_fit'] else 'no'}"
        )
    measured = [item for item in summary if item["writing_speed"] is not None]
    verdict = "no count could be measured"
    if measured:
        best = max(measured, key=lambda item: float(str(item["writing_speed"])))
        most = max(measured, key=lambda item: int(str(item["layers"])))
        best_speed = float(str(best["writing_speed"]))
        most_speed = float(str(most["writing_speed"]))
        verdict = (
            f"fastest: {best['layers']} layers ({best_speed:.2f} tokens/s); "
            f"most layers measured: {most['layers']} ({most_speed:.2f} tokens/s, "
            f"{(most_speed - best_speed) / best_speed:+.1%} vs the fastest)"
        )
        if fit.layers is not None:
            fitted = next((item for item in measured if item["layers"] == fit.layers), None)
            if fitted is not None:
                fit_speed = float(str(fitted["writing_speed"]))
                verdict += (
                    f"; InferenceOS's estimate ({fit.layers} layers): {fit_speed:.2f} tokens/s "
                    f"({(fit_speed - best_speed) / best_speed:+.1%} vs the fastest)"
                )
    print()
    print("RESULT: " + verdict)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else settings.reports_dir / f"layer_range_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "model": str(model_path),
                "context": args.context,
                "estimated_fit": fit.layers,
                "fit_reasons": fit.reasons,
                "machine": machine.model_dump(mode="json"),
                "plan": plan,
                "runs": rows,
                "summary": summary,
                "result": verdict,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
