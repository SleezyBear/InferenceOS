"""Measure how noisy the laptop is: the same setup compared with itself.

Runs the fair comparison (``tune/fair.py``) with the SAME setup under two names (A and
B), in independent blocks, each measurement from a fresh load, exactly like ``tune``
measures (``tune_path_runs`` runs after a warm-up). Any difference between A and B is
noise (or an order effect). Prints and saves, per question:

* each block's score and per-measure differences, and which setup went first;
* the spread of block scores (the noise the keep rule has to beat);
* the order effect (does whichever goes first in a block look better?);
* whether A and B differ at all (a 95% interval for the mean that leaves out 0 means a
  real bias such as an order or warm-up effect), and how many blocks a 2% / 3% / 5%
  gain needs to be found 80% of the time.

    .\\.venv\\Scripts\\python.exe scripts\\calibrate_noise.py --model M.gguf
        [--config gpu_offload_layers=1,threads=9] [--workloads short-short,long-question-9x-steady]
        [--blocks 10] --out _jobs\\out\\noise.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def _blocks_needed(sd: float, gain: float, alpha: float) -> int | None:
    """Blocks for an 80% chance to show ``gain`` (one look, normal approximation)."""
    from inferenceos.tune.stats import t_quantile

    if sd <= 0:
        return 2
    for blocks in range(2, 60):
        critical = t_quantile(1.0 - alpha, blocks - 1)
        se = sd / math.sqrt(blocks)
        if gain - critical * se > 0.84 * se:  # ~80% power
            return blocks
    return None


#: Per-run measures recorded; ``engine_first_word_s`` is the engine's own time to the
#: first word (reading the question plus one written word, no HTTP or streaming).
_RUN_FIELDS = ("ttft_s", "prefill_latency_s", "decode_tokens_per_sec", "total_latency_s")


def _run_values(run: object) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        name: (float(v) if (v := getattr(run, name, None)) is not None else None)
        for name in _RUN_FIELDS
    }
    prefill, decode = values["prefill_latency_s"], values["decode_tokens_per_sec"]
    values["engine_first_word_s"] = (
        prefill + 1.0 / decode if prefill is not None and decode else None
    )
    return values


def _run_table(fair: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side in ("start_runs", "candidate_runs"):
        for index, measurement in enumerate(getattr(fair, side)):
            for number, run in enumerate(measurement.run_metrics):
                rows.append(
                    {"side": side, "measurement": index, "run": number, **_run_values(run)}
                )
    return rows


def _noise_split(fair: object) -> dict[str, dict[str, float | None]]:
    """For each measure: typical spread between runs of one load, and between loads.

    Both as a share of the overall middle value. If the between-loads spread is much
    larger, more runs per measurement will not help; fresh loads differ.
    """
    out: dict[str, dict[str, float | None]] = {}
    for name in (*_RUN_FIELDS, "engine_first_word_s"):
        groups: list[list[float]] = []
        for side in ("start_runs", "candidate_runs"):
            for measurement in getattr(fair, side):
                values = [
                    v for run in measurement.run_metrics
                    if (v := _run_values(run)[name]) is not None
                ]
                if len(values) >= 2:
                    groups.append(values)
        if len(groups) < 2:
            out[name] = {"within": None, "between": None}
            continue
        middle = statistics.median(v for group in groups for v in group)
        within = statistics.fmean(statistics.stdev(group) for group in groups)
        means = [statistics.median(group) for group in groups]
        between = statistics.stdev(means)
        out[name] = {
            "within_share": within / middle if middle else None,
            "between_share": between / middle if middle else None,
            "runs_per_load": statistics.fmean(len(group) for group in groups),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--config", default="", help="setup, e.g. gpu_offload_layers=1,threads=9 (default: saved setup)")
    parser.add_argument("--workloads", default="short-short")
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--runs", type=int, default=0, help="runs per measurement (0 = tune's)")
    parser.add_argument("--effort", default="performance_first")
    parser.add_argument("--goal", default="balanced")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--warmup-seconds", type=float, default=0.0,
        help="answer questions with the setup for this long before the blocks (discarded)",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_tune_result import parse_config

    from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
    from inferenceos.config import Settings
    from inferenceos.contracts.enums import PowerState
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.experiment.policy import build_experiment_policy
    from inferenceos.hardware.profiler import detect_hardware_profile
    from inferenceos.logging_config import setup_logging
    from inferenceos.models.loader import load_model_profile
    from inferenceos.optimization.keep import ModeRules
    from inferenceos.profiling.runner import BaselineProfiler
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter
    from inferenceos.runtime.resident_workload import ResidentActiveWorkload
    from inferenceos.tune.fair import (
        CLOCK_METRIC,
        block_score,
        compare_fairly,
        fair_words,
        measurement_value,
    )
    from inferenceos.tune.measure import DisturbanceWatch, measure_steady
    from inferenceos.tune.paths import steady_workload
    from inferenceos.tune.stats import block_bounds
    from inferenceos.tune.profiles import SetupStore

    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "calibrate_noise.log")
    model_path = Path(args.model)
    model = load_model_profile(model_path)
    if args.config:
        config = parse_config(args.config, args.context)
    else:
        saved = SetupStore(settings).find(model_path, context_length=args.context, power=PowerState.AC)
        config = saved.configuration if saved is not None else RuntimeConfiguration(context_length=args.context)
    rules = ModeRules.from_settings(
        settings.model_copy(update={"effort_mode": args.effort, "goal_mode": args.goal})
    )
    base_policy = build_experiment_policy(settings)
    runs = max(int(args.runs or settings.tune_path_runs), 1)
    policy = base_policy.model_copy(
        update={
            "measured_repetitions": runs,
            "max_repetitions": runs,
            "min_valid_samples": runs,
            "max_total_runs": runs + base_policy.warmup_repetitions + 2,
        }
    )
    hardware = detect_hardware_profile(settings)
    looks = [int(settings.tune_fair_min_blocks), int(settings.tune_fair_max_blocks)]
    alpha = float(settings.tune_fair_alpha)
    print(f"model  : {model_path}")
    print(f"setup  : {config.model_dump(exclude_none=True)} (compared with itself)")
    print(f"blocks : {args.blocks}, {runs} runs per measurement")
    results: dict[str, object] = {"model": str(model_path), "setup": config.model_dump(exclude_none=True)}
    for workload_id in [item.strip() for item in args.workloads.split(",") if item.strip()]:
        workload = steady_workload(workload_id, default=get_workload("short-short"))
        resident = ResidentActiveWorkload(
            settings=settings,
            model=model,
            adapter=LlamaCppResidentAdapter(settings=settings),
            workload=workload,
            configuration=config,
        )
        harness = BaselineProfiler(
            settings=settings,
            adapter=ResidentMeasurementAdapter(resident, explicit_defaults=True),
            hardware=hardware,
        )
        watch = DisturbanceWatch(
            logical_cores=hardware.cpu.logical_cores,
            limit_percent=float(settings.tune_disturbance_cpu_percent),
            max_repeats=int(settings.tune_max_disturbed_repeats),
        )
        print(f"--- question {workload_id}", flush=True)
        # Which server process each measurement used (review 2: show that every
        # measurement, also across the A B | B A boundary, is a separate process).
        pids: list[int | None] = []

        def step(text: str) -> None:
            pid = resident.target_pid()
            pids.append(pid)
            print(f"  {text} (server before: {pid})", flush=True)

        warmup_record: dict[str, object] | None = None
        try:
            if args.warmup_seconds > 0:
                import time as _time

                began = _time.perf_counter()
                rounds = 0
                speeds: list[float] = []
                while _time.perf_counter() - began < args.warmup_seconds:
                    warm = measure_steady(
                        harness=harness, model=model, workload=workload, config=config,
                        policy=policy, phase=f"calibrate-warmup-{rounds}", watch=watch,
                    )
                    rounds += 1
                    clock = measurement_value(warm, CLOCK_METRIC)
                    if clock is not None:
                        speeds.append(round(clock, 1))
                warmup_record = {
                    "seconds": round(_time.perf_counter() - began, 1),
                    "measurements": rounds,
                    "processor_speed": speeds,
                }
                print(f"  warm-up: {warmup_record}", flush=True)
            fair = compare_fairly(
                harness=harness,
                model=model,
                workload=workload,
                start=config,
                candidate=config,
                policy=policy,
                settings=settings,
                rules=rules,
                watch=watch,
                card_id="tune.calibration",
                min_blocks=args.blocks,
                max_blocks=args.blocks,
                phase=f"calibrate-{workload_id}",
                on_step=step,
            )
            pids.append(resident.target_pid())
        finally:
            resident.stop()
        # Each counted block with its own score and order (review, 4 Oct 2026: never
        # pair a filtered score list with the unfiltered block list).
        scored = [
            (score, turn)
            for turn in fair.turns
            if (score := block_score(turn.improvements, rules)) is not None
        ]
        scores = [score for score, _ in scored]
        per_metric: dict[str, list[float]] = {}
        first_better: list[float] = []
        for turn in fair.turns:
            for metric, value in turn.improvements.items():
                per_metric.setdefault(metric, []).append(value)
        # Order effect: score of "B" when B went first minus when A went first.
        a_first = [score for score, turn in scored if turn.order.startswith("start")]
        b_first = [score for score, turn in scored if turn.order.startswith("candidate")]
        if a_first and b_first:
            first_better = [statistics.fmean(b_first) - statistics.fmean(a_first)]
        sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
        summary = {
            "blocks": len(scores),
            "block_scores": scores,
            "mean": statistics.fmean(scores) if scores else None,
            "sd_of_block_scores": sd,
            "sd_per_measure": {
                metric: (statistics.stdev(values) if len(values) > 1 else None)
                for metric, values in per_metric.items()
            },
            "order_effect_going_first": first_better[0] if first_better else None,
            # Same setup on both sides: a mean whose 95% interval leaves out 0 means a
            # real bias (order, label or warm-up effect), not noise.
            # Two one-sided 2.5% bounds = a two-sided 95% interval.
            "bias_interval_95": (
                [bias.lower, bias.upper] if (bias := block_bounds(scores, alpha=0.025)).known else None
            ),
            "blocks_needed_80pct": {
                f"{gain:.0%}": _blocks_needed(sd, gain, alpha / len(looks))
                for gain in (0.02, 0.03, 0.05)
            },
            # Every run of every measurement, to tell run-to-run noise (inside one
            # load) from load-to-load noise (between fresh loads).
            "runs": _run_table(fair),
            "noise_split": _noise_split(fair),
            # pids[k + 1] is the server that measurement k used.
            "server_pids": pids[1:],
            "separate_server_per_measurement": len(set(pids[1:])) == len(pids[1:]),
            "warmup": warmup_record,
            "words": fair_words(fair),
            "notes": fair.notes,
            "fair": fair.record(),
        }
        results[workload_id] = summary
        print(f"  {fair_words(fair)}")
        print(
            f"  noise: block-score spread {sd:.2%}; order effect "
            f"{summary['order_effect_going_first']}; A-vs-A difference 95% interval "
            f"{summary['bias_interval_95']}; blocks needed "
            f"{summary['blocks_needed_80pct']}",
            flush=True,
        )
        print(f"  noise split (spread inside one load vs between loads): {summary['noise_split']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
