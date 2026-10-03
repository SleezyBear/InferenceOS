"""Heat check on its own: one setup answers many questions in a row on one load.

Prints every answer's processor speed and writing speed, then the heat check's
verdict (the same analysis ``tune`` runs at its end). Nothing is saved or changed.

    .\\.venv\\Scripts\\python.exe scripts\\check_heat.py --model models\\ab\\Qwen3.5-9B-Q4_K_M.gguf
        [--setup gpu_offload_layers=8,threads=12,ubatch_size=256] [--context 4096]
        [--runs 30] [--workload short-long-decode] [--out result.json]

Without ``--setup`` the saved setup for the model (plugged in) is used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--setup", default=None)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--workload", default="short-long-decode")
    parser.add_argument("--out", default=None)
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
    from inferenceos.profiling.runner import BaselineProfiler
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter
    from inferenceos.runtime.resident_workload import ResidentActiveWorkload
    from inferenceos.tune.heat import analyse_heat
    from inferenceos.tune.measure import DisturbanceWatch, measure_steady
    from inferenceos.tune.paths import steady_workload
    from inferenceos.tune.profiles import SetupStore

    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "check_heat.log")
    model_path = Path(args.model)
    if args.setup:
        config = parse_config(args.setup, args.context)
    else:
        saved = SetupStore(settings).find(model_path, context_length=args.context, power=PowerState.AC)
        if saved is None:
            print("no saved setup for this model; give --setup")
            return 2
        config = saved.configuration
    model = load_model_profile(model_path)
    hardware = detect_hardware_profile(settings)
    workload = steady_workload(args.workload, default=get_workload("short-long-decode"))
    resident = ResidentActiveWorkload(
        settings=settings,
        model=model,
        adapter=LlamaCppResidentAdapter(settings=settings),
        workload=workload,
        configuration=RuntimeConfiguration(context_length=args.context, gpu_offload_layers=0),
    )
    harness = BaselineProfiler(
        settings=settings,
        adapter=ResidentMeasurementAdapter(resident, explicit_defaults=True),
        hardware=hardware,
    )
    policy = build_experiment_policy(settings).model_copy(
        update={
            "measured_repetitions": args.runs,
            "max_repetitions": args.runs,
            "min_valid_samples": args.runs,
            "max_total_runs": args.runs + 3,
        }
    )
    watch = DisturbanceWatch(
        logical_cores=hardware.cpu.logical_cores, limit_percent=101.0, max_repeats=0
    )
    print(f"model   : {model_path}")
    print(f"setup   : {config.model_dump(exclude_none=True)}")
    print(f"answers : {args.runs} in a row ({workload.workload_id})", flush=True)
    started = time.perf_counter()
    try:
        measurement = measure_steady(
            harness=harness,
            model=model,
            workload=workload,
            config=config,
            policy=policy,
            phase="heat-check",
            watch=watch,
        )
    finally:
        resident.stop()
    seconds = time.perf_counter() - started
    for index, run in enumerate(measurement.run_metrics, start=1):
        clock = run.cpu_performance_percent
        speed = run.decode_tokens_per_sec
        print(
            f"  answer {index:2d}: processor {clock if clock is None else round(clock, 1)}%, "
            f"writing {speed if speed is None else round(speed, 3)} words/s"
        )
    check = analyse_heat(
        measurement, tolerance=float(settings.tune_fair_clock_tolerance), seconds=seconds
    )
    print("RESULT: " + check.words)
    if args.out:
        Path(args.out).write_text(json.dumps(check.record(), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
