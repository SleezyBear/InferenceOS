"""Find the real graphics-card layer top by load-only tries (no speed tests).

Builds exactly what ``tune`` builds (the machine check, the model profile, the resident
workload and the measurement harness), loads the starting setup once, then runs only
:func:`inferenceos.tune.layer_top.find_layer_top` with the real load-only try. For every
try it prints the count, whether it loaded, the full error if any, the used and free
graphics memory, and whether it fits. Nothing is saved.

Run (from the repo folder, one line in PowerShell):
    .\\.venv\\Scripts\\python.exe scripts\\check_layer_top.py --model models\\study\\LFM2-1.2B-Q4_K_M.gguf --context 4096
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.config import Settings
from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.hardware.profiler import detect_hardware_profile
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.search.policy import SearchPolicy, build_search_policy
from inferenceos.optimization.search.resources import build_resource_envelope
from inferenceos.profiling.workloads import get_workload
from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.runtime.workload import runtime_params_from_config
from inferenceos.tune.commands import make_try_load
from inferenceos.tune.fit import layers_that_fit
from inferenceos.tune.layer_top import find_layer_top
from inferenceos.tune.machine import check_machine, gib
from inferenceos.tune.profiles import SetupStore


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model path")
    parser.add_argument("--context", type=int, required=True, help="fixed context size")
    parser.add_argument("--workload", default=None, help="test prompt (default: tune's)")
    return parser.parse_args()


def _starting_layers(
    *,
    settings: Settings,
    model_path: Path,
    model: ModelProfile,
    context_length: int,
    power: PowerState,
    total_bytes: int | None,
    used_bytes: int | None,
    policy: SearchPolicy,
) -> tuple[int, str]:
    store = SetupStore(settings)
    same = store.find(model_path, context_length=context_length, power=power)
    other_power = PowerState.BATTERY if power is PowerState.AC else PowerState.AC
    other = same or store.find(model_path, context_length=context_length, power=other_power)
    if other is not None:
        return int(other.configuration.gpu_offload_layers or 0), "your saved setup"
    fit = layers_that_fit(
        model=model,
        context_length=context_length,
        total_graphics_memory_bytes=total_bytes,
        used_before_loading_bytes=used_bytes,
        policy=policy,
    )
    return int(fit.layers or 0), "the model file estimate"


def main() -> int:
    args = _parse_args()
    settings = Settings()
    settings.ensure_directories()
    model_path = Path(args.model)
    model = load_model_profile(model_path)
    hardware = detect_hardware_profile(settings)
    machine = check_machine(settings, hardware=hardware)
    search_policy = build_search_policy(settings)
    workload = get_workload(args.workload or settings.tune_workload)

    def graphics_memory_in_use() -> int | None:
        from inferenceos.tune.machine import read_live_state

        used, _ = read_live_state()
        return used

    start_layers, source = _starting_layers(
        settings=settings,
        model_path=model_path,
        model=model,
        context_length=args.context,
        power=machine.power_source,
        total_bytes=machine.total_graphics_memory_bytes,
        used_bytes=machine.used_graphics_memory_bytes,
        policy=search_policy,
    )

    print("=== Layer top check ===")
    print(f"model   : {model_path} ({model.n_layers} layers)")
    print(f"context : {args.context} (fixed, never changed)")
    print(
        f"graphics: {gib(machine.total_graphics_memory_bytes)} total; "
        f"starting setup uses {gib(machine.used_graphics_memory_bytes)}"
    )
    print(f"starting: {start_layers} layers ({source})")

    resident = ResidentActiveWorkload(
        settings=settings,
        model=model,
        adapter=LlamaCppResidentAdapter(settings=settings),
        workload=workload,
        configuration=RuntimeConfiguration(
            context_length=args.context, gpu_offload_layers=0
        ),
    )
    measure_adapter = ResidentMeasurementAdapter(resident)

    try:
        # Bring the target up once at the starting setup, exactly as the tune
        # measurement does, so the load-only tries start from a live model.
        start_config = RuntimeConfiguration(
            context_length=args.context, gpu_offload_layers=start_layers
        )
        print()
        print("Loading the starting setup (no speed test) ...")
        measure_adapter.load(
            model,
            runtime_params_from_config(
                start_config,
                n_predict=workload.n_predict,
                seed=workload.seed,
                ignore_eos=workload.ignore_eos,
            ),
        )
        print(f"  loaded: {measure_adapter.is_loaded}")

        used_now = graphics_memory_in_use()
        now = datetime.now(timezone.utc)
        envelope = build_resource_envelope(
            total_vram_bytes=machine.total_graphics_memory_bytes,
            used_vram_bytes=used_now,
            telemetry_available=used_now is not None,
            telemetry_observed_at=now if used_now is not None else None,
            telemetry_source="check_layer_top: read with the starting setup loaded",
            now=now,
            model=model,
            context_length=args.context,
            current_layers=start_layers,
            total_layers=model.n_layers,
            policy=search_policy,
        )
        estimate: int | None = None
        if envelope.sufficient and envelope.feasible_max is not None:
            estimate = max(start_layers, int(envelope.feasible_max))
            if model.n_layers > 0:
                estimate = min(estimate, int(model.n_layers))
        print(f"  formula estimate: {estimate if estimate is not None else 'unknown'} layers")
        print()

        try_load = make_try_load(
            measure_adapter=measure_adapter,
            model=model,
            workload=workload,
            context_length=args.context,
            total_graphics_memory_bytes=machine.total_graphics_memory_bytes,
            search_policy=search_policy,
            graphics_memory_in_use=graphics_memory_in_use,
        )
        result = find_layer_top(
            n_layers=int(model.n_layers),
            estimate=estimate,
            try_load=try_load,
            memory_readable=machine.total_graphics_memory_bytes is not None,
            say=print,
        )
    finally:
        resident.stop()

    print()
    print(f"{'layers':>7} | {'loaded':>6} | {'fits':>4} | {'used':>10} | {'free':>10} | error")
    for entry in result.tries:
        used = entry.get("used_bytes")
        free = entry.get("free_bytes")
        print(
            f"{int(str(entry['layers'])):>7} | "
            f"{('yes' if entry.get('loaded') else 'no'):>6} | "
            f"{('yes' if entry.get('fits') else 'no'):>4} | "
            f"{(gib(int(str(used))) if used is not None else 'unknown'):>10} | "
            f"{(gib(int(str(free))) if free is not None else 'unknown'):>10} | "
            f"{entry.get('error') or ''}"
        )
    print()
    print(f"TOP: {result.top} layers (used the estimate: {result.used_estimate})")
    for reason in result.reasons:
        print(f"  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
