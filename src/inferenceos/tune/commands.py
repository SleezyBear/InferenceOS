"""The real-machine wiring for the ``tune`` and ``start`` commands (step 5)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.hardware.profiler import detect_hardware_profile
from inferenceos.logging_config import setup_logging
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.keep import ModeRules
from inferenceos.optimization.search.policy import build_search_policy
from inferenceos.profiling.workloads import get_workload
from inferenceos.tune.approval import ApproveAllChanges, ChangeApprover, ConsoleChangeApprover
from inferenceos.tune.machine import (
    check_machine,
    gib,
    load_last_check,
    machine_changes,
    read_live_state,
    save_check,
)
from inferenceos.tune.pipeline import Tuner
from inferenceos.tune.profiles import SetupStore, power_words
from inferenceos.tune.start import (
    LOOPBACK,
    describe_setup,
    port_is_free,
    quick_check,
    serve_until_stopped,
)


def run_tune(
    settings: Settings,
    *,
    model_path: Path,
    context_length: int,
    approve_all: bool = False,
    out: Path | None = None,
    as_json: bool = False,
) -> int:
    """Search, confirm, ask and save. Returns a process exit code."""
    from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
    from inferenceos.profiling.runner import BaselineProfiler
    from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter
    from inferenceos.runtime.resident_workload import ResidentActiveWorkload

    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "tune.log")
    if context_length < 1:
        print("error: --context must be a positive number", flush=True)
        return 2
    if not model_path.is_file():
        print(f"error: model file not found: {model_path}", flush=True)
        return 2

    rules = ModeRules.from_settings(settings)
    print("=== InferenceOS tune ===")
    print(f"model   : {model_path}")
    print(f"context : {context_length} (never changed)")
    print(
        f"modes   : effort {rules.effort.mode.value} "
        f"(up to {rules.effort.before_loading_search_time_s:.0f}s of searching, "
        f"smallest gain {rules.effort.minimum_improvement:.1%}), "
        f"goal {rules.goal.mode.value}"
    )

    print("Checking the machine (no model loaded) ...")
    hardware = detect_hardware_profile(settings)
    machine = check_machine(settings, hardware=hardware)
    previous = load_last_check(settings)
    changes = machine_changes(previous, machine) if previous is not None else []
    save_check(settings, machine)
    print(
        f"  graphics memory free: {gib(machine.free_graphics_memory_bytes)} of "
        f"{gib(machine.total_graphics_memory_bytes)}; {power_words(machine.power_source)}; "
        f"power plan {machine.power_plan.name or 'unknown'}; "
        f"llama.cpp {machine.program_version or 'unknown'}"
    )
    for change in changes:
        print(f"  changed since the last check: {change}")

    model = load_model_profile(model_path)
    workload = get_workload(settings.tune_workload)
    resident = ResidentActiveWorkload(
        settings=settings,
        model=model,
        adapter=LlamaCppResidentAdapter(settings=settings),
        workload=workload,
        configuration=RuntimeConfiguration(context_length=context_length, gpu_offload_layers=0),
    )
    harness = BaselineProfiler(
        settings=settings, adapter=ResidentMeasurementAdapter(resident), hardware=hardware
    )
    approver: ChangeApprover = ApproveAllChanges() if approve_all else ConsoleChangeApprover()
    if approve_all:
        print("  --yes given: every change found will be approved without asking")

    def graphics_memory_in_use() -> int | None:
        used, _ = read_live_state()
        return used

    tuner = Tuner(
        settings=settings,
        model=model,
        model_path=model_path,
        context_length=context_length,
        rules=rules,
        hardware=hardware,
        machine=machine,
        harness=harness,
        workload=workload,
        experiment_policy=build_experiment_policy(settings),
        search_policy=build_search_policy(settings),
        approver=approver,
        store=SetupStore(settings),
        graphics_memory_reader=graphics_memory_in_use,
        machine_changes=changes,
    )
    try:
        result = tuner.run()
    finally:
        # Every test undoes itself: the model is never left running by `tune`.
        resident.stop()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out or settings.reports_dir / f"tune_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json(), encoding="utf-8")
    if as_json:
        print(result.to_json())
    print(f"result : {result.outcome}")
    print(f"report : {path}")
    return 1 if result.outcome.startswith("failed") else 0


def run_start(
    settings: Settings,
    *,
    model_path: Path,
    context_length: int | None,
    port: int | None,
) -> int:
    """Quick check, then run the model with its saved setup until Ctrl+C."""
    from inferenceos.backends.llama_server_toolchain import resolve_server_executable
    from inferenceos.backends.resident_server import LlamaServerResidentBackend

    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "start.log")
    chosen_port = int(port if port is not None else settings.start_port)
    print("=== InferenceOS start ===")
    hardware = detect_hardware_profile(settings)
    machine = check_machine(settings, hardware=hardware)
    check = quick_check(
        store=SetupStore(settings),
        model_path=model_path,
        context_length=context_length,
        machine=machine,
    )
    for problem in check.problems:
        print(f"cannot start: {problem}")
    if not check.can_start or check.setup is None:
        return 1
    setup = check.setup
    print(f"setup  : {describe_setup(setup)}")
    for change in check.changes:
        print(f"changed: {change}")
    for note in check.notes:
        print(f"note   : {note}")
    if not port_is_free(chosen_port):
        print(
            f"cannot start: port {chosen_port} is already in use on this computer; "
            "choose another with --port"
        )
        return 1

    backend = LlamaServerResidentBackend(
        server=resolve_server_executable(settings.llama_cpp_dir),
        host=LOOPBACK,
        port=chosen_port,
        readiness_timeout_s=settings.resident_readiness_timeout_s,
        request_timeout_s=settings.resident_request_timeout_s,
    )
    model = load_model_profile(model_path)
    print("Starting the model ...")
    outcome = serve_until_stopped(setup=setup, model=model, backend=backend)
    if not outcome.started:
        print(f"cannot start: {outcome.reason}")
        return 1
    print(f"Stopped ({outcome.reason}); clean shutdown: {outcome.stopped_cleanly}.")
    return 0 if outcome.stopped_cleanly else 1


__all__ = ["run_start", "run_tune"]
