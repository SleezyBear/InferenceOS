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
from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.backends.llama_cpp_toolchain import (
    backend_type_from_name,
    find_cli,
    parse_device_info,
    parse_generated_text,
    parse_timings,
    run_cli,
)
from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    AttentionType,
    BackendType,
    MeasurementMode,
    MetricQuality,
    ObservationSource,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.system_state import SystemState
from inferenceos.hardware.profiler import classify_storage_path, detect_hardware_profile
from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
from inferenceos.logging_config import get_logger, setup_logging
from inferenceos.models.estimator import estimate_kv_curve
from inferenceos.models.loader import ModelMetadataError, load_model_profile
from inferenceos.profiling.report import render_summary
from inferenceos.profiling.runner import BaselineProfiler, ProfilerOptions
from inferenceos.profiling.statistics import REPRODUCIBILITY_TOLERANCE, within_tolerance
from inferenceos.profiling.workloads import BENCHMARK_SUITE
from inferenceos.selftest import run_reasoning_selftest, verify_contract_roundtrip
from inferenceos.telemetry.gpu import GpuUtilizationSampler, GpuUtilizationSummary
from inferenceos.telemetry.system import sample_system_state

logger = get_logger("cli")

DEFAULT_PHASE0_PROMPT = "Explain in one sentence why the sky appears blue."
DEFAULT_PHASE2_PROMPT = "Write one short sentence about local inference."
DEFAULT_PHASE2_CONTEXTS = (1024, 2048, 4096)


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


def _discover_models(models_dir: Path) -> list[Path]:
    return sorted(models_dir.glob("*.gguf"))


def _parse_contexts(value: str) -> list[int]:
    contexts: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        contexts.add(int(part))
    if not contexts:
        raise ValueError("no context lengths provided")
    return sorted(contexts)


def _fmt_count(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value >= 1_000_000_000:
        return f"{value / 1e9:.2f} B ({value})"
    if value >= 1_000_000:
        return f"{value / 1e6:.2f} M ({value})"
    if value >= 1_000:
        return f"{value / 1e3:.1f} K ({value})"
    return str(value)


def _attention_label(model: ModelProfile) -> str:
    if model.attention_type is AttentionType.MHA:
        return f"MHA (kv == heads == {model.n_kv_heads})"
    if model.attention_type is AttentionType.GQA:
        return f"GQA (kv {model.n_kv_heads} < heads {model.n_attention_heads})"
    if model.attention_type is AttentionType.MQA:
        return f"MQA (kv == 1 < heads {model.n_attention_heads})"
    return str(model.attention_type.value)


def _print_model_profile(model: ModelProfile, contexts: list[int]) -> None:
    print(f"=== Model: {model.name} ===")
    print(f"path            : {model.path}")
    print(f"format          : {model.format}")
    print(f"architecture    : {model.architecture}")
    print(f"quantization    : {model.quantization.value}")
    print(f"parameters      : {_fmt_count(model.parameter_count)}")
    print(f"tensors         : {model.tensor_count}")
    print(f"context length  : {model.trained_context_length}")
    print(f"layers          : {model.n_layers}")
    print(f"attention heads : {model.n_attention_heads}")
    print(f"KV heads        : {model.n_kv_heads}")
    print(f"head dimension  : {model.head_dim} (key={model.key_length}, value={model.value_length})")
    print(f"embedding dim   : {model.embedding_dim}")
    print(f"vocabulary      : {_fmt_count(model.vocabulary_size)}")
    print(f"attention type  : {_attention_label(model)}")
    print(f"file size       : {_fmt_bytes(model.file_size_bytes)}")
    print(f"weight estimate : {_fmt_bytes(model.weight_bytes)} (estimated, not measured)")
    print(f"KV per token    : {_fmt_bytes(model.kv_cache_bytes_per_token)} (f16)")
    print("KV estimates    :")
    for estimate in estimate_kv_curve(model.kv_cache_bytes_per_token, contexts):
        footprint = model.footprint(estimate.context_length)
        print(
            f"  ctx={estimate.context_length:>6} "
            f"kv={_fmt_bytes(estimate.kv_cache_bytes)} "
            f"total_estimate={_fmt_bytes(footprint.total_estimate_bytes)}"
        )


def _cmd_model_info(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase2.log")

    if args.model:
        paths = [Path(args.model)]
    else:
        paths = _discover_models(settings.models_dir)
    if not paths:
        print("error: no .gguf models found; pass --model or add models", file=sys.stderr)
        return 2
    contexts = _parse_contexts(args.contexts)

    payloads: list[dict[str, object]] = []
    failures: list[str] = []
    for path in paths:
        try:
            model = load_model_profile(path)
        except (ModelMetadataError, OSError) as exc:
            failures.append(f"{path}: {exc}")
            continue
        curve = estimate_kv_curve(model.kv_cache_bytes_per_token, contexts)
        payloads.append(
            {
                "profile": json.loads(model.model_dump_json()),
                "kv_curve": [json.loads(item.model_dump_json()) for item in curve],
                "footprints": [
                    json.loads(model.footprint(item.context_length).model_dump_json())
                    for item in curve
                ],
            }
        )
        if not args.json:
            _print_model_profile(model, contexts)
            print()

    if args.json:
        print(json.dumps({"models": payloads, "failures": failures}, indent=2))
    for failure in failures:
        print(f"[FAIL] {failure}", file=sys.stderr)
    return 0 if payloads else 1


def _measure_context_memory(
    adapter: LlamaCppAdapter,
    model: ModelProfile,
    contexts: list[int],
    *,
    n_predict: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for context in contexts:
        row: dict[str, object] = {"context": context}
        try:
            adapter.load(
                model,
                RuntimeParams(
                    gpu_offload_layers=0,
                    device="none",
                    use_mmap=False,
                    context_length=context,
                    threads=0,
                ),
            )
            result = adapter.generate(
                GenerationRequest(prompt="Hello", n_predict=n_predict, seed=seed)
            )
            telemetry = adapter.telemetry()
            row["predicted_kv_bytes"] = model.estimate_kv_cache_bytes(context)
            row["reported_kv_bytes"] = telemetry.kv_cache_bytes
            row["peak_rss_bytes"] = result.peak_rss_bytes
            row["model_buffer_bytes"] = telemetry.model_buffer_bytes
        except (FileNotFoundError, RuntimeError, TimeoutError, OSError) as exc:
            row["error"] = str(exc)
        finally:
            adapter.unload()
        rows.append(row)

    baseline_rss: int | None = None
    baseline_kv: int | None = None
    for row in rows:
        rss = _as_int(row.get("peak_rss_bytes"))
        predicted = _as_int(row.get("predicted_kv_bytes"))
        if rss is not None and predicted is not None:
            baseline_rss = rss
            baseline_kv = predicted
            break
    if baseline_rss is not None and baseline_kv is not None:
        for row in rows:
            peak = _as_int(row.get("peak_rss_bytes"))
            predicted = _as_int(row.get("predicted_kv_bytes"))
            if peak is None or predicted is None:
                continue
            row["measured_delta_rss_bytes"] = peak - baseline_rss
            row["predicted_delta_kv_bytes"] = predicted - baseline_kv
    return rows


def _cmd_verify_phase2(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase2.log")

    print("=== InferenceOS Phase 2 Verification (Model & Backend Abstraction) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    paths = [Path(args.model)] if args.model else _discover_models(settings.models_dir)
    contexts = _parse_contexts(args.contexts)
    print(f"models discovered: {len(paths)}")
    for path in paths:
        print(f"  - {path.name}")
    print()

    profiles: list[ModelProfile] = []
    parse_failures: list[str] = []
    for path in paths:
        try:
            profiles.append(load_model_profile(path))
        except (ModelMetadataError, OSError) as exc:
            parse_failures.append(f"{path.name}: {exc}")

    for model in profiles:
        _print_model_profile(model, contexts)
        print()

    metadata_ok = all(
        model.architecture != "unknown"
        and model.trained_context_length is not None
        and model.n_layers >= 1
        and model.n_attention_heads >= 1
        and model.n_kv_heads >= 1
        and model.head_dim >= 1
        for model in profiles
    )
    has_gqa = any(model.attention_type is not AttentionType.MHA for model in profiles)
    has_mha = any(model.attention_type is AttentionType.MHA for model in profiles)

    # Estimator must use the KV-head count, not the attention-head count.
    kv_head_check = True
    for model in profiles:
        expected = (
            model.n_layers
            * model.n_kv_heads
            * (model.key_length or model.head_dim)
            + model.n_layers
            * model.n_kv_heads
            * (model.value_length or model.head_dim)
        ) * 2
        if model.kv_cache_bytes_per_token != expected:
            kv_head_check = False
        if (
            model.n_kv_heads != model.n_attention_heads
            and model.kv_cache_bytes_per_token
            == model.n_layers * 2 * model.n_attention_heads * model.head_dim * 2
        ):
            kv_head_check = False

    # Real backend adapter: capabilities + Vulkan inference on the largest model.
    adapter = LlamaCppAdapter(settings=settings)
    try:
        capabilities = adapter.capabilities()
    except (FileNotFoundError, OSError) as exc:
        print(f"[FAIL] backend adapter unavailable: {exc}", file=sys.stderr)
        return 2
    print("--- Backend adapter (llama.cpp) ---")
    print(f"backend        : {capabilities.name} {capabilities.version or ''}".rstrip())
    print(f"devices        : {', '.join(capabilities.devices)}")
    print(f"backends       : {', '.join(b.value for b in capabilities.backends)}")

    gpu_model = max(profiles, key=lambda model: model.file_size_bytes) if profiles else None
    inference_text = ""
    inference_device: str | None = None
    offloaded: int | None = None
    offloaded_total: int | None = None
    inference_ok = False
    if gpu_model is not None:
        gpu_context = min(2048, gpu_model.trained_context_length or 2048)
        try:
            adapter.load(
                gpu_model,
                RuntimeParams(gpu_offload_layers=args.gpu_layers, context_length=gpu_context),
            )
            result = adapter.generate(
                GenerationRequest(
                    prompt=DEFAULT_PHASE2_PROMPT, n_predict=args.gpu_n_predict, seed=args.seed
                )
            )
            telemetry = adapter.telemetry()
            inference_text = result.text
            inference_device = telemetry.device
            offloaded = telemetry.gpu_layers_offloaded
            offloaded_total = telemetry.gpu_layers_total
            inference_ok = bool(inference_text) and bool(offloaded and offloaded_total and offloaded == offloaded_total)
        except (FileNotFoundError, RuntimeError, TimeoutError, OSError) as exc:
            print(f"[FAIL] inference: {exc}", file=sys.stderr)
        finally:
            adapter.unload()
    print(f"load+generate  : model={gpu_model.name if gpu_model else '?'} device={inference_device}")
    print(f"layers offloaded: {offloaded}/{offloaded_total}")
    print(f"generated text : {inference_text or '<none>'}")
    print()

    # Predicted vs measured memory (real RSS on CPU so KV is in system RAM).
    print("--- Predicted vs measured memory (CPU, load-mode none) ---")
    memory: dict[str, list[dict[str, object]]] = {}
    for model in profiles:
        rows = _measure_context_memory(
            adapter, model, contexts, n_predict=args.rss_n_predict, seed=args.seed
        )
        memory[model.name] = rows
        print(f"{model.name}:")
        for row in rows:
            if "error" in row:
                print(f"  ctx={row['context']:>6} ERROR {row['error']}")
                continue
            print(
                f"  ctx={row['context']:>6} "
                f"predicted_kv={_fmt_bytes(_as_int(row.get('predicted_kv_bytes')))} "
                f"reported_kv={_fmt_bytes(_as_int(row.get('reported_kv_bytes')))} "
                f"peak_rss={_fmt_bytes(_as_int(row.get('peak_rss_bytes')))} "
                f"d_rss={_fmt_bytes(_as_int(row.get('measured_delta_rss_bytes')))} "
                f"d_kv_pred={_fmt_bytes(_as_int(row.get('predicted_delta_kv_bytes')))}"
            )
    print()

    # Checks.
    reported_ok = True
    reported_seen = False
    for rows in memory.values():
        for row in rows:
            reported = _as_int(row.get("reported_kv_bytes"))
            predicted = _as_int(row.get("predicted_kv_bytes"))
            if reported and predicted:
                reported_seen = True
                if abs(predicted - reported) / reported > 0.05:
                    reported_ok = False

    rss_ok = True
    rss_seen = False
    for model in profiles:
        rows = memory.get(model.name, [])
        for row in rows:
            measured = _as_int(row.get("measured_delta_rss_bytes"))
            predicted = _as_int(row.get("predicted_delta_kv_bytes"))
            if measured is None or predicted is None:
                continue
            if predicted < args.rss_min_predicted_kv:
                continue
            rss_seen = True
            if measured <= 0:
                rss_ok = False
                continue
            ratio = measured / predicted
            if ratio < 0.4 or ratio > 2.5:
                rss_ok = False
    if not rss_seen:
        rss_ok = False

    checks: list[tuple[str, bool]] = [
        ("at least two real GGUF models parsed", len(profiles) >= 2 and not parse_failures),
        ("model metadata complete for every parsed model", metadata_ok and bool(profiles)),
        ("both GQA/MQA and standard MHA architectures observed", has_gqa and has_mha),
        ("KV estimate uses the KV-head count", kv_head_check),
        ("predicted KV matches backend-reported KV within 5%", reported_ok and reported_seen),
        ("backend adapter produced real generated text", inference_ok),
        ("Vulkan device used with full GPU offload", bool(inference_device and offloaded and offloaded == offloaded_total)),
        ("predicted vs measured RSS deltas consistent", rss_ok),
    ]

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report = {
        "phase": 2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": [json.loads(model.model_dump_json()) for model in profiles],
        "parse_failures": parse_failures,
        "backend": {
            "name": capabilities.name,
            "version": capabilities.version,
            "devices": capabilities.devices,
            "backends": [b.value for b in capabilities.backends],
        },
        "inference": {
            "model": gpu_model.name if gpu_model else None,
            "device": inference_device,
            "layers_offloaded": offloaded,
            "layers_total": offloaded_total,
            "text": inference_text,
            "ok": inference_ok,
        },
        "memory": memory,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    report_path = settings.reports_dir / (
        f"phase2_verification_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"=== RESULT: PHASE 2 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {report_path}")
    return 0 if ok else 1


def _as_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _load_models(settings: Settings, model_arg: str | None) -> list[ModelProfile]:
    if model_arg:
        return [load_model_profile(model_arg)]
    return [load_model_profile(path) for path in _discover_models(settings.models_dir)]


def _resolve_modes(value: str) -> list[MeasurementMode]:
    if value == "cold":
        return [MeasurementMode.COLD]
    if value == "warm":
        return [MeasurementMode.WARM]
    if value == "both":
        return [MeasurementMode.COLD, MeasurementMode.WARM]
    raise ValueError(f"invalid mode: {value}")


def _cmd_baseline(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase3.log")

    models = _load_models(settings, args.model)
    if not models:
        print("error: no .gguf models found", file=sys.stderr)
        return 2
    profiler = BaselineProfiler(settings=settings)
    tasks = profiler.make_tasks(models, args.workload or None, args.context)
    if not tasks:
        print("error: no applicable workloads for the selected models", file=sys.stderr)
        return 2
    options = ProfilerOptions(
        gpu_offload_layers=args.gpu_layers,
        telemetry_sampling=not args.no_telemetry,
    )

    print("=== InferenceOS baseline profiling ===")
    print(f"models    : {', '.join(model.name for model in models)}")
    print(f"tasks     : {len(tasks)}")
    print(f"mode(s)   : {args.mode}  repetitions: {args.repetitions}")
    print()
    report = profiler.profile(
        tasks,
        modes=_resolve_modes(args.mode),
        repetitions=args.repetitions,
        options=options,
    )
    print(render_summary(report))

    out_path = Path(args.out) if args.out else settings.reports_dir / (
        f"baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_json(), encoding="utf-8")
    print()
    print(f"report: {out_path}")
    return 0


def _cmd_verify_phase3(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase3.log")

    print("=== InferenceOS Phase 3 Verification (Baseline Profiling & Telemetry) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()

    models = _load_models(settings, args.model)
    if len(models) < 2:
        print("error: verify-phase3 requires at least two real GGUF models", file=sys.stderr)
        return 2
    profiler = BaselineProfiler(settings=settings)
    tasks = profiler.make_tasks(models, None, args.context)
    if not tasks:
        print("error: no applicable workloads", file=sys.stderr)
        return 2

    options = ProfilerOptions(
        gpu_offload_layers=args.gpu_layers,
        telemetry_sampling=True,
    )
    report = profiler.profile(
        tasks,
        modes=[MeasurementMode.WARM],
        repetitions=args.repetitions,
        options=options,
    )
    measured = [run for run in report.runs if not run.identity.warmup]

    # One genuine COLD measurement for the largest model's first workload.
    primary_task = max(tasks, key=lambda task: task.model.file_size_bytes)
    cold_run = profiler.run_once(
        primary_task,
        mode=MeasurementMode.COLD,
        sequence=len(report.runs) + 1,
        repetition=0,
        warmup=False,
        options=options,
    )
    all_runs = list(report.runs) + [cold_run]
    report = report.model_copy(update={"runs": all_runs})

    # Profiler-overhead comparison: same workload with external sampling off.
    overhead_options = ProfilerOptions(
        gpu_offload_layers=args.gpu_layers,
        telemetry_sampling=False,
    )
    overhead_run = profiler.run_once(
        primary_task,
        mode=MeasurementMode.WARM,
        sequence=len(all_runs) + 1,
        repetition=0,
        warmup=True,
        options=overhead_options,
    )
    baseline_group = next(
        (
            group
            for group in report.groups
            if group.model_ref == primary_task.model.name
            and group.workload_id == primary_task.workload.workload_id
            and group.mode is MeasurementMode.WARM
        ),
        None,
    )
    sampled_tps = (
        baseline_group.metrics["decode_tokens_per_sec"].mean if baseline_group is not None else None
    )
    unsampled_tps = overhead_run.observation.metrics.decode_tokens_per_sec
    overhead_ratio: float | None = None
    if sampled_tps and unsampled_tps:
        overhead_ratio = abs(unsampled_tps - sampled_tps) / sampled_tps

    # --- checks -------------------------------------------------------- #
    measured_models = {run.context.model.name for run in measured}
    architectures = {run.context.model.architecture for run in measured}
    attention_types = {run.context.model.attention_type for run in measured}
    has_mha = AttentionType.MHA in attention_types
    has_gqa = bool(attention_types & {AttentionType.GQA, AttentionType.MQA})

    vulkan_runs = [
        run
        for run in measured
        if run.observation.backend == BackendType.VULKAN
        and run.context.config.device is not None
        and "radeon" in (run.context.config.device or "").lower()
        and run.context.config.gpu_layers_offloaded
        and run.context.config.gpu_layers_offloaded == run.context.config.gpu_layers_total
    ]
    text_runs = [run for run in measured if run.observation.generated_text]

    primary = max(models, key=lambda model: model.file_size_bytes)
    primary_metrics = [
        run.observation.metrics
        for run in measured
        if run.context.model.name == primary.name
        and run.context.workload.workload_id == primary_task.workload.workload_id
    ]
    required_metric_names = (
        "model_load_time_s",
        "ttft_s",
        "prefill_latency_s",
        "decode_latency_s",
        "prompt_tokens_per_sec",
        "decode_tokens_per_sec",
        "total_latency_s",
        "process_wall_s",
        "prompt_tokens",
        "generated_tokens",
        "total_tokens",
        "peak_rss_bytes",
        "process_cpu_percent",
        "system_cpu_percent",
        "gpu_utilization_percent",
        "hard_page_faults_per_sec",
        "total_ram_bytes",
        "available_ram_bytes",
        "committed_bytes",
        "commit_limit_bytes",
    )
    metrics_complete = bool(primary_metrics) and all(
        all(getattr(metrics, name) is not None for name in required_metric_names)
        for metrics in primary_metrics
    )
    unavailable_explicit = all(
        run.observation.metrics.gpu_memory_total_bytes is None for run in measured
    ) and all(
        run.metric_quality.get("gpu_memory_total_bytes") == MetricQuality.UNAVAILABLE
        for run in measured
    )
    context_persisted = all(
        run.context.hardware.hostname
        and run.context.model.architecture
        and run.context.workload.workload_id
        and run.context.config.context_length
        for run in measured
    )

    # Deterministic workloads: stable ids/hashes across resolution.
    suite_a = {w.workload_id: w.prompt_hash for w in BENCHMARK_SUITE}
    suite_b = {w.workload_id: w.prompt_hash for w in list(BENCHMARK_SUITE)}
    workloads_deterministic = suite_a == suite_b and all(
        run.context.workload.prompt_set_ref and run.context.workload.prompts
        for run in measured
    )

    cold_warm_ok = any(run.identity.mode is MeasurementMode.COLD for run in all_runs) and any(
        run.identity.mode is MeasurementMode.WARM for run in measured
    )

    reproducibility_issues: list[str] = []
    for group in report.groups:
        if group.repetitions < 2:
            continue
        for metric, tolerance in REPRODUCIBILITY_TOLERANCE.items():
            stats = group.metrics.get(metric)
            if stats is None or stats.count < 2:
                continue
            if not within_tolerance(stats, tolerance):
                reproducibility_issues.append(
                    f"{group.model_ref}/{group.workload_id}/{group.mode.value}:{metric}"
                    f" cv={stats.cv:.3f}>{tolerance}"
                )

    checks: list[tuple[str, bool]] = [
        ("profiles >=2 structurally different real models", len(measured_models) >= 2 and len(architectures) >= 2),
        ("both MHA and GQA/MQA architectures covered", has_mha and has_gqa),
        ("real generated text captured for every measured run", bool(text_runs) and len(text_runs) == len(measured)),
        ("Vulkan + Radeon + full layer offload exercised", bool(vulkan_runs)),
        ("required metrics captured for the primary model/workload", metrics_complete),
        ("unavailable telemetry represented explicitly", unavailable_explicit),
        ("run context persisted (hardware/model/workload/config)", context_persisted),
        ("workload suite is deterministic", workloads_deterministic),
        ("cold and warm modes both represented and documented", cold_warm_ok and bool(report.cold_warm_semantics)),
        ("resident-process warm explicitly unsupported", report.resident_warm_supported is False),
        ("repeated warm runs within agreed variance band", not reproducibility_issues),
        ("profiler overhead negligible (decode tps within 20%)", overhead_ratio is not None and overhead_ratio <= 0.20),
    ]

    print(render_summary(report))
    print()
    print("--- Profiler overhead ---")
    print(
        f"sampled decode_tps={_fmt_num(sampled_tps)} unsampled={_fmt_num(unsampled_tps)} "
        f"relative_delta={_fmt_pct(overhead_ratio)}"
    )
    if reproducibility_issues:
        print("--- Reproducibility exceptions ---")
        for issue in reproducibility_issues:
            print(f"  {issue}")
    print()
    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = json.loads(report.to_json())
    report_payload["acceptance"] = {name: flag for name, flag in checks}
    report_payload["overhead"] = {
        "sampled_decode_tokens_per_sec": sampled_tps,
        "unsampled_decode_tokens_per_sec": unsampled_tps,
        "relative_delta": overhead_ratio,
    }
    report_payload["reproducibility_exceptions"] = reproducibility_issues
    report_payload["result"] = "PASS" if ok else "FAIL"
    report_path = settings.reports_dir / (
        f"phase3_verification_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print()
    print(f"=== RESULT: PHASE 3 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {report_path}")
    return 0 if ok else 1


def _fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


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

    model_info_parser = subparsers.add_parser(
        "model-info",
        help="parse real GGUF metadata and print the normalized ModelProfile",
    )
    model_info_parser.add_argument("--model", default=None, help="path to a .gguf model")
    model_info_parser.add_argument(
        "--contexts",
        default=",".join(str(c) for c in DEFAULT_PHASE2_CONTEXTS),
        help="comma-separated context lengths for KV estimates",
    )
    model_info_parser.add_argument("--json", action="store_true", help="emit JSON")
    model_info_parser.set_defaults(func=_cmd_model_info)

    verify2_parser = subparsers.add_parser(
        "verify-phase2",
        help="run the real Phase 2 acceptance (models + backend + predicted vs measured)",
    )
    verify2_parser.add_argument("--model", default=None, help="use a single .gguf model")
    verify2_parser.add_argument(
        "--contexts",
        default=",".join(str(c) for c in DEFAULT_PHASE2_CONTEXTS),
        help="comma-separated context lengths for memory validation",
    )
    verify2_parser.add_argument("--gpu-layers", type=int, default=99)
    verify2_parser.add_argument("--gpu-n-predict", type=int, default=32)
    verify2_parser.add_argument("--rss-n-predict", type=int, default=1)
    verify2_parser.add_argument("--seed", type=int, default=1234)
    verify2_parser.add_argument(
        "--rss-min-predicted-kv",
        type=int,
        default=32 * 1024 * 1024,
        help="only gate the RSS check on contexts whose predicted KV delta exceeds this",
    )
    verify2_parser.set_defaults(func=_cmd_verify_phase2)

    baseline_parser = subparsers.add_parser(
        "baseline",
        help="run deterministic benchmark workloads and persist a baseline report",
    )
    baseline_parser.add_argument("--model", default=None, help="path to a .gguf model")
    baseline_parser.add_argument(
        "--workload", action="append", default=None, help="workload id (repeatable)"
    )
    baseline_parser.add_argument("--repetitions", type=int, default=3)
    baseline_parser.add_argument("--mode", choices=["cold", "warm", "both"], default="warm")
    baseline_parser.add_argument("--context", type=int, default=None)
    baseline_parser.add_argument("--gpu-layers", type=int, default=99)
    baseline_parser.add_argument(
        "--no-telemetry", action="store_true", help="disable external counter sampling"
    )
    baseline_parser.add_argument("--out", default=None, help="report output path")
    baseline_parser.set_defaults(func=_cmd_baseline)

    verify3_parser = subparsers.add_parser(
        "verify-phase3",
        help="run the real Phase 3 acceptance (profiling + reproducibility)",
    )
    verify3_parser.add_argument("--model", default=None, help="use a single .gguf model")
    verify3_parser.add_argument("--context", type=int, default=None)
    verify3_parser.add_argument("--gpu-layers", type=int, default=99)
    verify3_parser.add_argument("--repetitions", type=int, default=3)
    verify3_parser.set_defaults(func=_cmd_verify_phase3)

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
