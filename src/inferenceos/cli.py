"""InferenceOS command-line interface.

Phase 0's human-verifiable evidence command is ``inferenceos verify-phase0``: it
loads a real model, runs one prompt through the real Vulkan-enabled llama.cpp
binary, and prints (a) the generated text, (b) the backend/device that executed it,
and (c) GPU utilization observed via Windows GPU Engine counters while the call ran.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import psutil

from inferenceos import __version__
from inferenceos.backends.llama_cpp_toolchain import (
    backend_type_from_name,
    find_cli,
    parse_device_info,
    parse_generated_text,
    parse_timings,
    run_cli,
)
from inferenceos.config import Settings
from inferenceos.contracts.enums import BackendType, MeasurementMode, ObservationSource
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.system_state import SystemState
from inferenceos.hardware.profiler import classify_storage_path, detect_hardware_profile
from inferenceos.logging_config import get_logger, setup_logging
from inferenceos.selftest import run_reasoning_selftest, verify_contract_roundtrip
from inferenceos.telemetry.gpu import GpuUtilizationSampler, GpuUtilizationSummary
from inferenceos.telemetry.system import sample_system_state

logger = get_logger("cli")

DEFAULT_PHASE0_PROMPT = "Explain in one sentence why the sky appears blue."


def _find_model(models_dir: Path, override: str | None) -> Path:
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"model not found: {path}")
        return path
    candidates = sorted(models_dir.glob("*.gguf"))
    if not candidates:
        raise FileNotFoundError(
            f"no .gguf model found in {models_dir}; pass --model"
        )
    return candidates[0]


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"InferenceOS {__version__}")
    return 0


def _cmd_contracts_check(args: argparse.Namespace) -> int:
    passed = verify_contract_roundtrip()
    print(f"Contract round-trip: PASS ({len(passed)} contracts)")
    for name in passed:
        print(f"  - {name}")
    return 0


def _cmd_reasoning_selftest(args: argparse.Namespace) -> int:
    response = run_reasoning_selftest()
    print(
        "ReasoningEngine round-trip: PASS "
        f"(engine={response.engine.value}, status={response.status.value})"
    )
    print(f"  conclusion: {response.conclusion}")
    if response.bottleneck is not None:
        print(
            "  bottleneck: "
            f"{response.bottleneck.kind.value} "
            f"(meaningful={response.bottleneck.is_meaningful})"
        )
    return 0


def _cmd_verify_phase0(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase0.log")

    cli = find_cli(settings.llama_cpp_dir)
    model = _find_model(settings.models_dir, args.model)
    prompt = args.prompt

    print("=== InferenceOS Phase 0 Verification ===")
    print(f"timestamp : {datetime.now(timezone.utc).isoformat()}")
    print(f"host      : {socket.gethostname()}")
    print(f"llama-cli : {cli}")
    print(f"model     : {model} ({model.stat().st_size / 1024**2:.1f} MiB)")
    print(f"request   : -ngl {args.gpu_layers} -n {args.n_predict} -s {args.seed} --temp 0 -st")
    print()

    passed = verify_contract_roundtrip()
    reasoning = run_reasoning_selftest()
    print(f"[contracts] {len(passed)} contract types round-trip JSON: PASS")
    print(f"[contracts] stub ReasoningEngine round-trip: PASS (status={reasoning.status.value})")
    print()

    sampler = GpuUtilizationSampler(interval_s=settings.gpu_sample_interval_s)
    sampler.start()
    sampler.wait_until_ready(timeout_s=5.0)
    result = None
    summary: GpuUtilizationSummary | None = None
    try:
        result = run_cli(
            cli,
            model,
            prompt,
            n_predict=args.n_predict,
            seed=args.seed,
            temperature=0.0,
            n_gpu_layers=args.gpu_layers,
            threads=settings.default_threads,
            verbose=True,
            timeout_s=args.timeout,
        )
    finally:
        pid = result.pid if result is not None else None
        summary = sampler.stop(target_pid=pid)

    assert result is not None and summary is not None  # for type narrowing

    device = parse_device_info(result.stderr)
    text = parse_generated_text(result.stdout)
    prompt_tps, gen_tps = parse_timings(result.stdout)
    backend = backend_type_from_name(device.backend) if device else None

    print("--- Generated text (real llama.cpp output) ---")
    print(text if text else "<none>")
    print()
    print("--- Backend / device ---")
    if device is not None:
        print(f"backend          : {device.backend}")
        print(f"device           : {device.device}")
        if device.layers_offloaded is not None and device.layers_total is not None:
            print(f"layers offloaded : {device.layers_offloaded}/{device.layers_total} to GPU")
        if device.vram_total_bytes is not None:
            print(
                "VRAM             : "
                f"{device.vram_total_bytes / 1024**2:.0f} MiB total, "
                f"{(device.vram_free_bytes or 0) / 1024**2:.0f} MiB free"
            )
    else:
        print("no device information parsed from llama.cpp logs")

    print()
    print("--- Performance (real) ---")
    print(f"prompt eval     : {prompt_tps} tokens/s")
    print(f"generation      : {gen_tps} tokens/s")
    print(f"wall clock      : {result.duration_s:.2f} s")
    print()
    print("--- GPU utilization (observed via Windows GPU Engine counters) ---")
    print(f"samples captured: {len(summary.samples)} over {summary.duration_s:.2f} s")
    print(f"max overall     : {summary.max_overall_percent:.1f}% (engine: {summary.max_engine})")
    print(f"max for inference pid {summary.target_pid}: {summary.max_target_percent:.1f}%")
    if summary.engine_max_percent:
        engines = ", ".join(
            f"{name}={value:.1f}%" for name, value in sorted(summary.engine_max_percent.items())
        )
        print(f"per-engine max  : {engines}")

    offload_ok = (
        device is not None
        and device.layers_offloaded is not None
        and device.layers_total is not None
        and device.layers_offloaded == device.layers_total
        and device.layers_offloaded > 0
    )
    checks = [
        ("real llama.cpp inference produced non-empty text", bool(text)),
        ("Vulkan GPU offload confirmed (all layers)", offload_ok),
        ("backend is Vulkan", backend == BackendType.VULKAN),
        ("device is AMD Radeon", bool(device and "radeon" in device.device.lower())),
        ("GPU utilization > 0 observed during call", summary.max_overall_percent > 0),
    ]

    print()
    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    observation = Observation(
        observation_id=f"phase0-{int(time.time())}",
        captured_at=datetime.now(timezone.utc),
        source=ObservationSource.DIAGNOSTIC,
        mode=MeasurementMode.COLD,
        backend=backend,
        device=device.device if device else None,
        model_ref=model.name,
        workload_ref="phase0-smoke",
        metrics=RunMetrics(
            prompt_tokens_per_sec=prompt_tps,
            decode_tokens_per_sec=gen_tps,
            total_latency_s=result.duration_s,
            generated_tokens=args.n_predict,
            gpu_utilization_percent=summary.max_overall_percent,
        ),
        generated_text=text,
    )

    report = {
        "phase": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llama_cli": str(cli),
        "model": str(model),
        "command": result.command,
        "device": asdict(device) if device is not None else None,
        "generated_text": text,
        "timings": {
            "prompt_tokens_per_sec": prompt_tps,
            "decode_tokens_per_sec": gen_tps,
            "wall_seconds": result.duration_s,
        },
        "gpu": {
            "sample_count": len(summary.samples),
            "duration_s": summary.duration_s,
            "max_overall_percent": summary.max_overall_percent,
            "max_engine": summary.max_engine,
            "max_target_percent": summary.max_target_percent,
            "target_pid": summary.target_pid,
            "engine_max_percent": summary.engine_max_percent,
            "samples": [asdict(sample) for sample in summary.samples],
        },
        "observation": json.loads(observation.model_dump_json()),
        "contracts": passed,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    report_path = settings.reports_dir / (
        f"phase0_verification_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"=== RESULT: PHASE 0 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {report_path}")
    return 0 if ok else 1


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB ({value} B)"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MiB ({value} B)"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB ({value} B)"
    return f"{value} B"


def _print_hardware_profile(profile: HardwareProfile) -> None:
    cpu = profile.cpu
    mem = profile.memory
    print("=== CPU ===")
    print(f"name            : {cpu.name}")
    print(f"vendor          : {cpu.vendor}")
    print(f"architecture    : {cpu.architecture}")
    print(f"cores           : {cpu.physical_cores} physical / {cpu.logical_cores} logical")
    print(f"clock           : base={cpu.base_clock_mhz} MHz max={cpu.max_clock_mhz} MHz")
    print(
        "caches          : "
        f"L1={_fmt_bytes(cpu.caches.l1_bytes)} "
        f"L2={_fmt_bytes(cpu.caches.l2_bytes)} "
        f"L3={_fmt_bytes(cpu.caches.l3_bytes)}"
    )
    print(f"instruction sets: {', '.join(cpu.instruction_sets) or '<none detected>'}")
    print(f"detected via    : {cpu.detected_via}")
    print()
    print("=== GPUs (detected) and backends (usable) ===")
    if not profile.gpus:
        print("no GPUs detected")
    for index, gpu in enumerate(profile.gpus):
        backends = ", ".join(b.value for b in gpu.supported_backends) or "none"
        print(
            f"[{index}] {gpu.name} | vendor={gpu.vendor.value} type={gpu.device_type.value} "
            f"vram={_fmt_bytes(gpu.vram_bytes)}"
        )
        print(
            f"    driver={gpu.driver_version} ({gpu.driver_date}) pci={gpu.pci_id} "
            f"vulkan_index={gpu.vulkan_device_index} vulkan_api={gpu.vulkan_api_version} "
            f"backends=[{backends}]"
        )
    print()
    for availability in profile.backend_availability:
        print(
            f"backend {availability.backend.value:8s}: detected={availability.detected} "
            f"usable={availability.usable} devices={len(availability.devices)} "
            f"({availability.detail})"
        )
    print()
    print("=== RAM ===")
    print(f"total physical  : {_fmt_bytes(mem.total_bytes)}")
    print(f"available       : {_fmt_bytes(mem.available_bytes)}")
    print(f"committed       : {_fmt_bytes(mem.committed_bytes)} / limit {_fmt_bytes(mem.commit_limit_bytes)}")
    for module in mem.modules:
        print(
            f"module          : {_fmt_bytes(module.capacity_bytes)} "
            f"{module.speed_mhz} MHz {module.manufacturer}"
        )
    print()
    print("=== Storage ===")
    for device in profile.storage:
        print(
            f"{device.name} | media={device.media_type.value} bus={device.bus_type} "
            f"size={_fmt_bytes(device.size_bytes)} free={_fmt_bytes(device.free_bytes)} "
            f"removable={device.is_removable} volumes={device.volumes}"
        )
    print()
    print("=== OS ===")
    print(f"{profile.os.caption} {profile.os.version} build {profile.os.build} ({profile.os.architecture})")
    print(f"hostname: {profile.hostname} | captured_at: {profile.captured_at.isoformat()}")


def _print_system_state(state: SystemState) -> None:
    print("=== Live system state (real OS counters) ===")
    print(f"captured_at     : {state.captured_at.isoformat()}")
    print(f"RAM total       : {_fmt_bytes(state.total_ram_bytes)}")
    print(f"RAM available   : {_fmt_bytes(state.available_ram_bytes)}")
    print(
        f"committed       : {_fmt_bytes(state.committed_bytes)} / limit "
        f"{_fmt_bytes(state.commit_limit_bytes)}"
    )
    print(f"memory pressure : {state.memory_pressure:.3f} (commit ratio)")
    print(f"CPU total       : {state.cpu_percent_total:.1f}%")
    print(f"hard page faults: {state.hard_page_faults_per_sec:.1f} /s (\\Memory\\Pages Input/sec)")
    print("--- top processes by RSS ---")
    for process in state.processes:
        print(
            f"  pid {process.pid:>6} {process.name:<28.28} "
            f"cpu={process.cpu_percent:6.1f}% rss={_fmt_bytes(process.rss_bytes)}"
        )
    if state.target_process is not None:
        target = state.target_process
        print(
            f"target process  : pid {target.pid} {target.name} "
            f"cpu={target.cpu_percent:.1f}% rss={_fmt_bytes(target.rss_bytes)}"
        )
    print("--- GPU engines (per Windows GPU Engine counters) ---")
    if not state.gpu_engines:
        print("  no GPU engine utilization reported")
    for engine in state.gpu_engines:
        if engine.utilization_percent > 0:
            print(f"  device {engine.device_index} {engine.engine:<12} {engine.utilization_percent:6.1f}%")
    print("--- GPU adapter memory (dedicated usage) ---")
    if not state.gpu_memory:
        print("  no GPU adapter memory reported")
    for memory in state.gpu_memory:
        print(
            f"  device {memory.device_index} used={_fmt_bytes(memory.memory_used_bytes)} "
            f"total={_fmt_bytes(memory.memory_total_bytes)}"
        )


def _model_storage_line(settings: Settings, profile: HardwareProfile) -> tuple[str, dict[str, object] | None]:
    try:
        model = _find_model(settings.models_dir, None)
    except FileNotFoundError:
        return "<no model file found>", None
    info = classify_storage_path(model, profile)
    if info is None:
        return f"{model} (no matching storage device)", None
    line = (
        f"{model} -> {info.device_name} media={info.media_type.value} "
        f"bus={info.bus_type} volume={info.volume} external={info.is_external}"
    )
    payload: dict[str, object] = {
        "path": info.path,
        "volume": info.volume,
        "device_name": info.device_name,
        "media_type": info.media_type.value,
        "is_removable": info.is_removable,
        "is_external": info.is_external,
        "bus_type": info.bus_type,
        "free_bytes": info.free_bytes,
    }
    return line, payload


def _cmd_profile(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase1.log")

    profile = detect_hardware_profile(settings)
    state = sample_system_state(
        target_pid=None,
        interval_s=args.interval,
        top_processes=args.top,
        include_gpu=not args.no_gpu,
    )
    model_line, model_payload = _model_storage_line(settings, profile)

    print("=== InferenceOS Hardware / System Profile ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    _print_hardware_profile(profile)
    print()
    print("=== Model storage classification ===")
    print(model_line)
    print()
    _print_system_state(state)

    report = {
        "phase": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware_profile": json.loads(profile.model_dump_json()),
        "system_state": json.loads(state.model_dump_json()),
        "model_storage": model_payload,
    }
    report_path = settings.reports_dir / (
        f"hardware_profile_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"report: {report_path}")
    return 0


def _cmd_verify_phase1(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase1.log")

    print("=== InferenceOS Phase 1 Verification (real machine) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    profile = detect_hardware_profile(settings)
    _print_hardware_profile(profile)
    print()

    state = sample_system_state(
        target_pid=None,
        interval_s=args.interval,
        top_processes=settings.telemetry_top_processes,
        include_gpu=True,
    )
    _print_system_state(state)
    print()

    model_line, model_payload = _model_storage_line(settings, profile)
    print("=== Model storage classification ===")
    print(model_line)
    print()

    ps_logical = psutil.cpu_count(logical=True)
    ps_physical = psutil.cpu_count(logical=False)
    ps_total_ram = psutil.virtual_memory().total

    vulkan_devices = [
        device
        for availability in profile.backend_availability
        if availability.backend == BackendType.VULKAN
        for device in availability.devices
    ]
    cuda_availability = next(
        (a for a in profile.backend_availability if a.backend == BackendType.CUDA), None
    )
    ram_delta_ratio = (
        abs(profile.memory.total_bytes - ps_total_ram) / ps_total_ram if ps_total_ram else 1.0
    )
    cores_match = ps_logical is None or profile.cpu.logical_cores == ps_logical
    physical_match = ps_physical is None or profile.cpu.physical_cores == ps_physical
    storage_ok = bool(profile.storage) and model_payload is not None
    media_known = bool(model_payload and model_payload.get("media_type") in ("ssd", "hdd"))
    cuda_consistent = cuda_availability is not None and (
        cuda_availability.usable == (profile.cuda_available)
    )
    telemetry_ok = (
        state.total_ram_bytes > 0
        and state.available_ram_bytes <= state.total_ram_bytes
        and state.cpu_percent_total >= 0.0
        and bool(state.processes)
    )

    # Real GPU utilization must be observed under a real GPU workload.
    gpu_util_percent = 0.0
    gpu_inference_ran = False
    gpu_inference_note = ""
    gpu_sample_count = 0
    if not args.no_gpu_inference:
        try:
            cli = find_cli(settings.llama_cpp_dir)
            model = _find_model(settings.models_dir, args.model)
        except (FileNotFoundError, OSError) as exc:
            gpu_inference_note = f"inference unavailable: {exc}"
        else:
            for attempt in range(1, 3):
                sampler = GpuUtilizationSampler(interval_s=0.25)
                sampler.start()
                sampler.wait_until_ready(timeout_s=5.0)
                result = None
                try:
                    result = run_cli(
                        cli,
                        model,
                        "Name one color.",
                        n_predict=args.gpu_n_predict,
                        seed=args.seed,
                        temperature=0.0,
                        n_gpu_layers=args.gpu_layers,
                        threads=settings.default_threads,
                        verbose=True,
                        timeout_s=args.timeout,
                    )
                finally:
                    summary: GpuUtilizationSummary = sampler.stop(
                        target_pid=result.pid if result is not None else None
                    )
                if result is None:  # pragma: no cover - defensive
                    continue
                device = parse_device_info(result.stderr)
                compute_util = max(
                    (
                        value
                        for name, value in summary.engine_max_percent.items()
                        if name.lower() == "compute"
                    ),
                    default=0.0,
                )
                attempt_util = max(summary.max_target_percent, compute_util)
                gpu_sample_count = len(summary.samples)
                gpu_util_percent = max(gpu_util_percent, attempt_util)
                gpu_inference_ran = result.returncode == 0
                gpu_inference_note = (
                    f"device={device.device if device else '?'} "
                    f"layers_offloaded={device.layers_offloaded if device else '?'}/"
                    f"{device.layers_total if device else '?'} "
                    f"compute_util={attempt_util:.1f}% samples={len(summary.samples)} "
                    f"attempt={attempt}"
                )
                if gpu_util_percent > 0.0:
                    break
    else:
        gpu_inference_note = "skipped (--no-gpu-inference)"

    checks: list[tuple[str, bool]] = [
        ("CPU identity/cores detected and match psutil", cores_match and physical_match),
        ("CPU instruction sets detected (incl. SSE2)", "sse2" in profile.cpu.instruction_sets),
        ("RAM total matches psutil within 5%", ram_delta_ratio <= 0.05),
        ("Vulkan available and real device enumerated", profile.vulkan_available and bool(vulkan_devices)),
        ("CUDA availability consistent with detection", cuda_consistent),
        ("Storage device backing model identified", storage_ok),
        ("Storage media classified (SSD/HDD)", media_known),
        ("Live system/process telemetry captured", telemetry_ok),
        ("GPU utilization observed during real inference", gpu_util_percent > 0.0 and gpu_inference_ran),
    ]

    print("--- Cross-checks against independent Windows sources ---")
    print(
        f"CPU cores      : profile={profile.cpu.physical_cores}/{profile.cpu.logical_cores} "
        f"psutil={ps_physical}/{ps_logical}"
    )
    print(
        f"RAM total      : profile={profile.memory.total_bytes} psutil={ps_total_ram} "
        f"(delta {ram_delta_ratio * 100:.2f}%)"
    )
    print(f"GPU inference  : {gpu_inference_note}")
    print()
    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report = {
        "phase": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware_profile": json.loads(profile.model_dump_json()),
        "system_state": json.loads(state.model_dump_json()),
        "model_storage": model_payload,
        "cross_checks": {
            "psutil_physical_cores": ps_physical,
            "psutil_logical_cores": ps_logical,
            "psutil_total_ram_bytes": ps_total_ram,
            "ram_delta_ratio": ram_delta_ratio,
            "gpu_inference_ran": gpu_inference_ran,
            "gpu_utilization_percent": gpu_util_percent,
            "gpu_sample_count": gpu_sample_count,
            "gpu_inference_note": gpu_inference_note,
        },
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    report_path = settings.reports_dir / (
        f"phase1_verification_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"=== RESULT: PHASE 1 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {report_path}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferenceos",
        description="Autonomous adaptive local LLM optimization system",
    )
    parser.add_argument("--version", action="version", version=f"InferenceOS {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="print the version")
    version_parser.set_defaults(func=_cmd_version)

    contracts_parser = subparsers.add_parser(
        "contracts-check", help="instantiate and JSON round-trip every contract"
    )
    contracts_parser.set_defaults(func=_cmd_contracts_check)

    reasoning_parser = subparsers.add_parser(
        "reasoning-selftest", help="round-trip the stub ReasoningEngine interface"
    )
    reasoning_parser.set_defaults(func=_cmd_reasoning_selftest)

    verify_parser = subparsers.add_parser(
        "verify-phase0",
        help="run the real Phase 0 acceptance (llama.cpp + Vulkan + GPU utilization)",
    )
    verify_parser.add_argument("--model", default=None, help="path to a .gguf model")
    verify_parser.add_argument("--prompt", default=DEFAULT_PHASE0_PROMPT)
    verify_parser.add_argument("--n-predict", type=int, default=64)
    verify_parser.add_argument("--gpu-layers", type=int, default=99)
    verify_parser.add_argument("--seed", type=int, default=1234)
    verify_parser.add_argument("--timeout", type=float, default=900.0)
    verify_parser.set_defaults(func=_cmd_verify_phase0)

    profile_parser = subparsers.add_parser(
        "profile",
        help="print the real HardwareProfile + live SystemState for this machine",
    )
    profile_parser.add_argument("--interval", type=float, default=0.6)
    profile_parser.add_argument("--top", type=int, default=8)
    profile_parser.add_argument(
        "--no-gpu", action="store_true", help="skip GPU engine/adapter counters"
    )
    profile_parser.set_defaults(func=_cmd_profile)

    verify1_parser = subparsers.add_parser(
        "verify-phase1",
        help="run the real Phase 1 acceptance (hardware + live telemetry + GPU util)",
    )
    verify1_parser.add_argument("--model", default=None, help="path to a .gguf model")
    verify1_parser.add_argument("--interval", type=float, default=0.6)
    verify1_parser.add_argument("--gpu-layers", type=int, default=99)
    verify1_parser.add_argument("--gpu-n-predict", type=int, default=64)
    verify1_parser.add_argument("--seed", type=int, default=1234)
    verify1_parser.add_argument("--timeout", type=float, default=900.0)
    verify1_parser.add_argument(
        "--no-gpu-inference",
        action="store_true",
        help="do not run a real llama.cpp inference for GPU utilization evidence",
    )
    verify1_parser.set_defaults(func=_cmd_verify_phase1)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result: int = args.func(args)
    except (FileNotFoundError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
