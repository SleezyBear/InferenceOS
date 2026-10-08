"""Independent re-check of a tune result: starting setup vs saved setup, in turns.

Reads a tune report (JSON), then measures its starting setup and its final setup in
turns (A B, B A, ...) on one or more workloads, fresh loads every time, and prints
every reading. Nothing is saved or changed. Used to confirm that what ``tune`` kept
really is faster, with more turns than ``tune`` itself uses.

    .\\.venv\\Scripts\\python.exe scripts\\check_tune_result.py --report path\\to\\tune.json
        [--turns 4] [--workloads short-short,long-prefill] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inferenceos.contracts.experiment import RuntimeConfiguration


def parse_config(text: str, context: int) -> "RuntimeConfiguration":
    """``threads=8,flash_attention=true`` -> a configuration at ``context``."""
    from inferenceos.contracts.experiment import RuntimeConfiguration

    values: dict[str, object] = {"context_length": context}
    for part in [item.strip() for item in text.split(",") if item.strip()]:
        key, _, raw = part.partition("=")
        lowered = raw.strip().lower()
        value: object
        if lowered in ("true", "on"):
            value = True
        elif lowered in ("false", "off"):
            value = False
        elif lowered.lstrip("-").isdigit():
            value = int(lowered)
        else:
            value = raw.strip()
        values[key.strip()] = value
    return RuntimeConfiguration.model_validate(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=None, help="tune report to re-check")
    parser.add_argument("--model", default=None, help="model file (instead of --report)")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--start", default=None, help="e.g. gpu_offload_layers=1 (with --model)")
    parser.add_argument("--final", default=None, help="e.g. gpu_offload_layers=1,threads=8")
    parser.add_argument("--effort", default="performance_first")
    parser.add_argument("--goal", default="balanced")
    parser.add_argument("--turns", type=int, default=4, help="blocks; 0 = tune's own looks")
    parser.add_argument("--workloads", default="short-short,long-prefill")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
    from inferenceos.config import Settings
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.contracts.tune import TuneResult
    from inferenceos.experiment.policy import build_experiment_policy
    from inferenceos.hardware.profiler import detect_hardware_profile
    from inferenceos.logging_config import setup_logging
    from inferenceos.models.loader import load_model_profile
    from inferenceos.optimization.keep import ModeRules
    from inferenceos.profiling.runner import BaselineProfiler
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter
    from inferenceos.runtime.resident_workload import ResidentActiveWorkload
    from inferenceos.tune.fair import compare_fairly, fair_words
    from inferenceos.tune.measure import DisturbanceWatch
    from inferenceos.tune.paths import steady_workload

    if args.report:
        report = TuneResult.model_validate_json(Path(args.report).read_text(encoding="utf-8"))
        start = report.starting_configuration
        final = (
            report.saved_setup.configuration if report.saved_setup is not None
            else report.final_configuration
        )
        model_file, context, effort, goal = (
            report.model_path, report.context_length, report.effort_mode, report.goal_mode
        )
    elif args.model and args.start is not None and args.final is not None:
        start = parse_config(args.start, args.context)
        final = parse_config(args.final, args.context)
        model_file, context, effort, goal = args.model, args.context, args.effort, args.goal
    else:
        parser.error("give --report, or --model with --start and --final")
    if start == final:
        print("the report kept no change; nothing to check")
        return 0
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "check_tune_result.log")
    rules = ModeRules.from_settings(
        settings.model_copy(update={"effort_mode": effort, "goal_mode": goal})
    )
    model_path = Path(model_file)
    model = load_model_profile(model_path)
    hardware = detect_hardware_profile(settings)
    print(f"model  : {model_path}")
    print(f"start  : {start.model_dump(exclude_none=True)}")
    print(f"final  : {final.model_dump(exclude_none=True)}")
    print(f"turns  : {args.turns} each, modes effort {effort} goal {goal}")
    results: dict[str, object] = {}
    for workload_id in [item.strip() for item in args.workloads.split(",") if item.strip()]:
        workload = steady_workload(workload_id, default=get_workload("short-short"))
        resident = ResidentActiveWorkload(
            settings=settings,
            model=model,
            adapter=LlamaCppResidentAdapter(settings=settings),
            workload=workload,
            configuration=RuntimeConfiguration(
                context_length=context, gpu_offload_layers=0
            ),
        )
        harness = BaselineProfiler(
            settings=settings, adapter=ResidentMeasurementAdapter(resident, explicit_defaults=True), hardware=hardware
        )
        watch = DisturbanceWatch(
            logical_cores=hardware.cpu.logical_cores,
            limit_percent=float(settings.tune_disturbance_cpu_percent),
            max_repeats=int(settings.tune_max_disturbed_repeats),
        )
        print(f"--- workload {workload_id}")
        try:
            fair = compare_fairly(
                harness=harness,
                model=model,
                workload=workload,
                start=start,
                candidate=final,
                policy=build_experiment_policy(settings),
                settings=settings,
                rules=rules,
                watch=watch,
                card_id="tune.recheck",
                min_blocks=args.turns if args.turns > 0 else None,
                max_blocks=args.turns if args.turns > 0 else None,
                phase=f"recheck-{workload_id}",
                on_step=lambda step: print(f"  {step}", flush=True),
            )
        finally:
            resident.stop()
        for turn in fair.turns:
            start_speed = turn.start.get("decode_tokens_per_sec")
            new_speed = turn.candidate.get("decode_tokens_per_sec")
            start_ttft = turn.start.get("ttft_s")
            new_ttft = turn.candidate.get("ttft_s")
            print(
                f"  turn {turn.turn} ({turn.order}): writing {start_speed:.3f} -> {new_speed:.3f} "
                f"t/s, first word {start_ttft:.3f} -> {new_ttft:.3f} s"
                if None not in (start_speed, new_speed, start_ttft, new_ttft)
                else f"  turn {turn.turn}: {turn.improvements}"
            )
        print("  RESULT: " + fair_words(fair))
        print(
            "  answers: "
            + ("passed the answer check" if fair.quality_passed else "did NOT pass the answer check")
            + f"; failed runs: new {fair.candidate_failures}, start {fair.start_failures}"
            + (f"; blockers: {'; '.join(fair.acceptance_blockers())}" if fair.acceptance_blockers() else "")
        )
        if fair.heat_effect:
            print("  processor speed: " + fair.heat_effect)
        results[workload_id] = fair.record()
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
