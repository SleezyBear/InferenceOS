"""The real-machine wiring for the ``tune`` and ``start`` commands (step 5)."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inferenceos.backends.engine_env import engine_settings, process_environment
from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.hardware.profiler import detect_hardware_profile
from inferenceos.logging_config import setup_logging
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.keep import ModeRules
from inferenceos.optimization.search.policy import SearchPolicy, build_search_policy
from inferenceos.profiling.workloads import BenchmarkWorkload, get_workload
from inferenceos.tune.approval import ApproveAllChanges, ChangeApprover, ConsoleChangeApprover
from inferenceos.tune.layer_top import LoadAttempt
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

if TYPE_CHECKING:
    from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter


def make_try_load(
    *,
    measure_adapter: ResidentMeasurementAdapter,
    model: ModelProfile,
    workload: BenchmarkWorkload,
    context_length: int,
    total_graphics_memory_bytes: int | None,
    search_policy: SearchPolicy,
    graphics_memory_in_use: Callable[[], int | None],
    say: Callable[[str], None] = print,
) -> Callable[[int], LoadAttempt]:
    """Build the load-only try used to find the real layer top.

    ``try_load(N)`` must bring the resident target up at N graphics card layers the same
    way a measurement does (the workload's controlled reload), wait until it is ready,
    and only then read the graphics memory. A failure to load -- including out of
    graphics memory -- is "does not fit", but the full reason is returned so it is never
    hidden.
    """
    from inferenceos.runtime.workload import runtime_params_from_config

    total = total_graphics_memory_bytes
    fractions = (
        search_policy.safety_headroom_fraction + search_policy.runtime_reserve_fraction
    )

    def try_load(layers: int) -> LoadAttempt:
        if total is None or total <= 0:
            return LoadAttempt(
                loaded=False, fits=False, error="total graphics memory is unknown"
            )
        config = RuntimeConfiguration(
            context_length=context_length, gpu_offload_layers=int(layers)
        )
        params = runtime_params_from_config(
            config,
            n_predict=workload.n_predict,
            seed=workload.seed,
            ignore_eos=workload.ignore_eos,
        )
        # Load through the measurement adapter (the resident workload's controlled
        # reload), exactly as a measurement does. Calling the raw adapter here would
        # skip that path and be rejected while the starting setup is still resident.
        try:
            measure_adapter.load(model, params)
            loaded = bool(measure_adapter.is_loaded)
        except Exception as exc:  # noqa: BLE001 - a failed load does not fit
            error = f"{type(exc).__name__}: {exc}"
            say(f"    load of {layers} layers failed: {error}")
            return LoadAttempt(loaded=False, fits=False, error=error)
        if not loaded:
            error = "the model did not report itself loaded after the controlled reload"
            say(f"    load of {layers} layers did not become ready: {error}")
            return LoadAttempt(loaded=False, fits=False, error=error)
        used = graphics_memory_in_use()
        if used is None:
            return LoadAttempt(
                loaded=True,
                fits=False,
                error="graphics memory in use could not be read",
            )
        free = max(total - used, 0)
        fits = (total - used) >= total * fractions
        return LoadAttempt(loaded=True, fits=fits, used_bytes=used, free_bytes=free)

    return try_load


def read_server_options(server: Path, *, timeout_s: float = 60.0) -> frozenset[str] | None:
    """The long options this llama-server build has, read from ``--help``.

    ``None`` when the help text cannot be read (then no path is skipped for it).
    """
    import re
    import subprocess

    try:
        completed = subprocess.run(
            [str(server), "--help"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    found = frozenset(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", text))
    return found or None


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
        + (
            "(no fixed time limit: runs until its planned comparisons are done, "
            if rules.effort.mode.value == "performance_first"
            else f"(up to {rules.effort.before_loading_search_time_s:.0f}s of searching, "
        )
        + f"smallest gain {rules.effort.minimum_improvement:.1%}), "
        f"goal {rules.goal.mode.value}"
    )

    inherited = engine_settings(process_environment(dict(os.environ), None))
    if inherited:
        # Engine variables set outside InferenceOS reach every test (diagnostic ones
        # such as engine timers are always removed): say so.
        print(
            "engine  : settings inherited from your environment, used in every test: "
            + ";".join(f"{k}={v}" for k, v in inherited.items())
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
    measure_adapter = ResidentMeasurementAdapter(resident, explicit_defaults=True)
    harness = BaselineProfiler(
        settings=settings, adapter=measure_adapter, hardware=hardware
    )
    approver: ChangeApprover = ApproveAllChanges() if approve_all else ConsoleChangeApprover()
    if approve_all:
        print("  --yes given: every change found will be approved without asking")

    def graphics_memory_in_use() -> int | None:
        used, _ = read_live_state()
        return used

    search_policy = build_search_policy(settings)
    try_load = make_try_load(
        measure_adapter=measure_adapter,
        model=model,
        workload=workload,
        context_length=context_length,
        total_graphics_memory_bytes=machine.total_graphics_memory_bytes,
        search_policy=search_policy,
        graphics_memory_in_use=graphics_memory_in_use,
    )

    from inferenceos.backends.llama_server_toolchain import resolve_server_executable

    try:
        supported = read_server_options(resolve_server_executable(settings.llama_cpp_dir))
    except Exception:  # noqa: BLE001 - unknown options just means nothing is skipped
        supported = None
    print(
        "  llama-server options: "
        + (f"{len(supported)} found" if supported is not None else "could not be read")
    )

    from inferenceos.tune.engine_probe import (
        ProbeNotSafe,
        ProbeResult,
        run_probe,
        run_switch_probe,
        wait_until_gone,
    )
    from inferenceos.tune.memory_speed import MemorySpeed, cached_memory_speed

    def probe_runner(config: RuntimeConfiguration, **options: Any) -> ProbeResult:
        # One timed run with its own engine process: the measuring server is stopped
        # first, and its exit is checked, so two copies of the model never share the
        # graphics card (review 4: a failed unload was only logged); the next
        # measurement starts it again.
        identity = resident.target_identity()
        resident.stop()
        if identity is not None and not wait_until_gone(identity, timeout_s=20.0):
            raise ProbeNotSafe(
                "the measuring server did not exit; no timed run, so two copies of the "
                "model never share the graphics card"
            )
        if options.pop("switch_check", False):
            return run_switch_probe(settings=settings, model=model, config=config, **options)
        return run_probe(settings=settings, model=model, config=config, **options)

    def memory_speed_reader() -> MemorySpeed | None:
        modules = ",".join(
            f"{module.capacity_bytes}@{module.speed_mhz}" for module in hardware.memory.modules
        )
        key = f"{hardware.cpu.name}|{hardware.memory.total_bytes}|{modules}"
        return cached_memory_speed(
            settings.data_dir / "memory_speed.json",
            key,
            max_threads=int(hardware.cpu.logical_cores),
        )

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
        search_policy=search_policy,
        approver=approver,
        store=SetupStore(settings),
        graphics_memory_reader=graphics_memory_in_use,
        machine_changes=changes,
        try_load=try_load,
        supported_options=supported,
        probe_runner=probe_runner,
        memory_speed_reader=memory_speed_reader,
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


__all__ = ["make_try_load", "read_server_options", "run_start", "run_tune"]
