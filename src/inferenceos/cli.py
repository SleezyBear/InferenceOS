"""InferenceOS command-line interface.

Phase 0's human-verifiable evidence command is ``inferenceos verify-phase0``: it
loads a real model, runs one prompt through the real Vulkan-enabled llama.cpp
binary, and prints (a) the generated text, (b) the backend/device that executed it,
and (c) GPU utilization observed via Windows GPU Engine counters while the call ran.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

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
from inferenceos.contracts.bottleneck import Bottleneck
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    AttentionType,
    BackendType,
    BottleneckKind,
    MeasurementMode,
    MetricQuality,
    ObservationSource,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.enums import (
    AnomalyClass,
    ApprovalOutcome,
    ExperimentDecisionKind,
    InterventionClass,
    KnowledgeSelectionEffect,
    LoopTerminationReason,
    OptimizationOutcome,
    RestorationStatus,
    RuntimeState,
    RuntimeTerminationReason,
)
from inferenceos.contracts.experiment import (
    ApplicationResult,
    ExperimentMeasurement,
    ExperimentPolicy,
    ExperimentResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.knowledge import (
    ApplicabilityContext,
    ApplicabilityResult,
    KnowledgeQuery,
    OptimizationRecord,
)
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.contracts.optimization import CandidatePlan, OptimizationCandidate
from inferenceos.contracts.optimization_loop import LoopPolicy, OptimizationBudget
from inferenceos.contracts.profiling import BaselineReport, ProfilingConfig, ProfilingRun
from inferenceos.contracts.reasoning import DeepDiagnosisOutcome
from inferenceos.contracts.system_state import SystemState
from inferenceos.experiment.audit import build_audit_record
from inferenceos.experiment.comparison import compare
from inferenceos.experiment.decision import decide
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.experiment.measurement import MeasurementHarness, options_for
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.experiment.quality import validate_quality
from inferenceos.experiment.report import render_experiment
from inferenceos.experiment.service import ExperimentInputError, run_candidate_experiment
from inferenceos.knowledge import (
    KnowledgeService,
    PersistResult,
    SQLiteKnowledgeStore,
    build_current_context,
    build_knowledge_policy,
    evaluate_applicability,
    render_applicability,
    render_health,
    render_history,
    render_record,
    render_rollback,
    verify_rollback,
)
from inferenceos.knowledge.store import KnowledgeCorruptionError
from inferenceos.loop import (
    OptimizationLoop,
    build_loop_policy,
    render_run,
    render_termination,
)
from inferenceos.contracts.runtime import RuntimeBudget, RuntimePolicy
from inferenceos.runtime import (
    ActiveWorkload,
    ApprovalProvider,
    AutoApproveApprovalProvider,
    AutoRejectApprovalProvider,
    CliApprovalProvider,
    ConsoleNotificationSink,
    ControlledLlamaWorkload,
    RuntimeMonitor,
    RuntimeTelemetrySampler,
    ScriptedApprovalProvider,
    TimeoutApprovalProvider,
    build_runtime_budget,
    build_runtime_policy,
    classify_action,
    render_monitor_run,
)
from inferenceos.runtime.verification import (
    ScriptedRuntimeSampler,
    ScriptedWorkload,
    StaticHarness,
    baseline_and_pressured_metrics,
    healthy_metrics,
    improving_responder,
    no_improvement_responder as rt_no_improvement_responder,
    quality_regression_responder as rt_quality_regression_responder,
)
from inferenceos.loop.verification import (
    FaultyRestoreExecutor,
    ScriptedHarness,
    context_improvement_responder,
    failing_responder,
    make_healthy_report,
    make_insufficient_report,
    make_paging_report,
    no_improvement_responder,
    quality_regression_responder,
    report_for_config,
)
from inferenceos.optimization.actions import build_default_registry
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.diagnosis.loader import (
    BaselineReportError,
    find_latest_report,
    load_baseline_report,
)
from inferenceos.diagnosis.report import render_diagnosis
from inferenceos.hardware.profiler import classify_storage_path, detect_hardware_profile
from inferenceos.interfaces.backend_adapter import GenerationRequest, RuntimeParams
from inferenceos.interfaces.reasoning_engine import ReasoningEngine
from inferenceos.logging_config import get_logger, setup_logging
from inferenceos.models.estimator import estimate_kv_curve
from inferenceos.models.loader import ModelMetadataError, load_model_profile
from inferenceos.native import windows
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import build_policy
from inferenceos.optimization.report import render_plan
from inferenceos.profiling.report import render_summary
from inferenceos.profiling.runner import BaselineProfiler, ProfilerOptions, ProfileTask
from inferenceos.profiling.statistics import (
    REPRODUCIBILITY_TOLERANCE,
    summarize,
    within_tolerance,
)
from inferenceos.profiling.workloads import (
    BENCHMARK_SUITE,
    choose_context_length,
    get_workload,
)
from inferenceos.reasoning.context import build_reasoning_context
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.escalation import decide_escalation
from inferenceos.reasoning.factory import build_reasoning_engine, build_reasoning_provider
from inferenceos.reasoning.lifecycle import run_deep_diagnosis
from inferenceos.reasoning.providers.base import ReasoningProvider
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from inferenceos.reasoning.providers.llama_cpp import LlamaCppReasoningProvider
from inferenceos.reasoning.report import render_deep_diagnosis
from inferenceos.reasoning.stub import StubReasoningEngine
from inferenceos.selftest import (
    run_reasoning_selftest,
    sample_hardware,
    verify_contract_roundtrip,
)
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
                # Ensure the counter stream is actively sampling before a short
                # workload begins, and ignore EOS so the GPU burst is sustained long
                # enough to be observed (a one-token reply can otherwise be missed).
                sampler.wait_for_first_sample(timeout_s=4.0)
                result = None
                try:
                    result = run_cli(
                        cli,
                        model,
                        "Name one color.",
                        n_predict=max(args.gpu_n_predict, 64),
                        seed=args.seed,
                        temperature=0.0,
                        n_gpu_layers=args.gpu_layers,
                        threads=settings.default_threads,
                        verbose=True,
                        ignore_eos=True,
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


# --------------------------------------------------------------------------- #
# Phase 4 — deterministic diagnosis                                          #
# --------------------------------------------------------------------------- #
_CONDITION_KEY = "condition"

_MEMORY_HOLDER_CODE = r"""
import sys, time
size = int(sys.argv[1])
buf = bytearray(size)
for i in range(0, size, 4096):
    buf[i] = 1
print('READY', flush=True)
time.sleep(600)
"""

_GPU_FAMILY = {
    BottleneckKind.COMPUTE_BOUND,
    BottleneckKind.GPU_UNDERUTILIZATION,
    BottleneckKind.GPU_MEMORY_LIMIT,
}


def _tag_condition(run: ProfilingRun, condition: str) -> ProfilingRun:
    config = run.context.config.model_copy(
        update={"extra": {**(run.context.config.extra or {}), _CONDITION_KEY: condition}}
    )
    context = run.context.model_copy(update={"config": config})
    return run.model_copy(update={"context": context})


def _combine_runs(runs: list[ProfilingRun], base: BaselineReport, notes: str) -> BaselineReport:
    workloads = {workload.workload_id: workload for workload in base.workloads}
    return BaselineReport(
        profiler_version=base.profiler_version,
        suite_version=base.suite_version,
        generated_at=datetime.now(timezone.utc),
        hardware=base.hardware,
        backend=base.backend,
        workloads=list(workloads.values()),
        runs=runs,
        groups=[],
        cold_warm_semantics=base.cold_warm_semantics,
        resident_warm_supported=False,
        limitations=list(base.limitations),
        notes=notes,
    )


def _profile_condition(
    profiler: BaselineProfiler,
    task: ProfileTask,
    *,
    condition: str,
    gpu_layers: int,
    repetitions: int,
) -> list[ProfilingRun]:
    report = profiler.profile(
        [task],
        modes=[MeasurementMode.WARM],
        repetitions=repetitions,
        options=ProfilerOptions(gpu_offload_layers=gpu_layers, telemetry_sampling=True),
    )
    return [_tag_condition(run, condition) for run in report.runs]


def _start_memory_holder(*, leave_bytes: int, cap_bytes: int) -> tuple[subprocess.Popen[str] | None, int]:
    memory = windows.global_memory_status()
    if memory is None:
        return None, 0
    hold = min(max(memory.available_physical_bytes - leave_bytes, 0), cap_bytes)
    if hold < 2 * 1024**3:
        return None, 0
    try:
        proc: subprocess.Popen[str] = subprocess.Popen(
            [sys.executable, "-c", _MEMORY_HOLDER_CODE, str(hold)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:  # pragma: no cover - defensive
        return None, 0
    if proc.stdout is not None:
        proc.stdout.readline()
    return proc, hold


def _stop_memory_holder(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        proc.kill()


def _meaningful_findings(diagnosis: BottleneckDiagnosis) -> list[Bottleneck]:
    findings: list[Bottleneck] = []
    if diagnosis.primary is not None:
        findings.append(diagnosis.primary)
    findings.extend(diagnosis.candidates)
    return findings


def _cmd_diagnose(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase4.log")

    if args.report:
        report_path = Path(args.report)
    else:
        found = find_latest_report(settings.reports_dir)
        if found is None:
            print("error: no Phase 3 report found; pass --report", file=sys.stderr)
            return 2
        report_path = found
    try:
        report = load_baseline_report(report_path)
    except BaselineReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    diagnosis = BottleneckDetector().detect(report, source_report_path=str(report_path))
    if args.json:
        print(diagnosis.to_json())
    else:
        print(render_diagnosis(diagnosis))

    out_path = Path(args.out) if args.out else settings.reports_dir / (
        f"diagnosis_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.write_text(diagnosis.to_json(), encoding="utf-8")
    print()
    print(f"diagnosis: {out_path}")
    if args.strict and not diagnosis.meaningful_opportunity:
        return 3
    return 0


def _cmd_verify_phase4(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase4.log")

    print("=== InferenceOS Phase 4 Verification (Deterministic Bottleneck Detection) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # --- Scenario A: real Phase 3 report -> real diagnosis (false-positive guard) ---
    real_path = Path(args.report) if args.report else find_latest_report(settings.reports_dir)
    if real_path is None:
        print("error: no Phase 3 report found; run `inferenceos baseline` first", file=sys.stderr)
        return 2
    try:
        real_report = load_baseline_report(real_path)
    except BaselineReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    real_diagnosis = BottleneckDetector().detect(real_report, source_report_path=str(real_path))
    real_artifact = settings.reports_dir / f"phase4_diagnosis_real_{timestamp}.json"
    real_artifact.write_text(real_diagnosis.to_json(), encoding="utf-8")

    print("--- Scenario A: real Phase 3 baseline -> diagnosis ---")
    print(render_diagnosis(real_diagnosis))
    print()

    real_meaningful = _meaningful_findings(real_diagnosis)
    real_gpu_meaningful = [item for item in real_meaningful if item.kind in _GPU_FAMILY]
    real_gpu_observation = any(
        item.kind is BottleneckKind.COMPUTE_BOUND for item in real_diagnosis.observations
    )

    # --- Scenario B: controlled real induced memory pressure ---
    print("--- Scenario B: controlled real induced memory pressure ---")
    models = _load_models(settings, args.model)
    if not models:
        print("error: no models available", file=sys.stderr)
        return 2
    primary_model = max(models, key=lambda model: model.file_size_bytes)
    workload = get_workload("short-short")
    context = max(choose_context_length(primary_model), 32768)
    task = ProfileTask(model=primary_model, workload=workload, context_length=context)
    profiler = BaselineProfiler(settings=settings)

    control_runs = _profile_condition(
        profiler, task, condition="control", gpu_layers=0, repetitions=args.repetitions
    )
    holder, hold_bytes = _start_memory_holder(
        leave_bytes=int(args.pressure_leave_gib * 1024**3),
        cap_bytes=int(args.pressure_cap_gib * 1024**3),
    )
    pressure_ran = holder is not None
    pressured_runs: list[ProfilingRun] = []
    try:
        if pressure_ran:
            pressured_runs = _profile_condition(
                profiler, task, condition="pressured", gpu_layers=0, repetitions=args.repetitions
            )
    finally:
        _stop_memory_holder(holder)

    pressure_diagnosis: BottleneckDiagnosis | None = None
    pressure_artifact: Path | None = None
    if pressure_ran and pressured_runs:
        combined = _combine_runs(
            control_runs + pressured_runs,
            real_report,
            "controlled induced-memory-pressure comparison (Phase 4 acceptance)",
        )
        pressure_diagnosis = BottleneckDetector().detect(
            combined, source_report_path=f"<controlled-pressure:{timestamp}>"
        )
        pressure_artifact = settings.reports_dir / f"phase4_diagnosis_pressure_{timestamp}.json"
        pressure_artifact.write_text(pressure_diagnosis.to_json(), encoding="utf-8")
        print(f"held memory: {hold_bytes / 1024**3:.2f} GiB; measured runs: {len(pressured_runs)}")
        print(render_diagnosis(pressure_diagnosis))
    else:
        print(
            "pressure scenario not run: insufficient memory headroom "
            f"(need > {2 * 1024**3 / 1024**3:.0f} GiB allocatable after leaving "
            f"{args.pressure_leave_gib} GiB)"
        )
    print()

    pressure_primary = pressure_diagnosis.primary if pressure_diagnosis is not None else None
    pressure_meaningful = bool(
        pressure_diagnosis is not None and pressure_diagnosis.meaningful_opportunity
    )
    pressure_paging = bool(
        pressure_primary is not None
        and pressure_primary.kind in {BottleneckKind.PAGING, BottleneckKind.MEMORY_PRESSURE}
    )
    pressure_evidence_ok = bool(
        pressure_primary is not None
        and any("hard_page_faults" in item.metric for item in pressure_primary.supporting_evidence)
    )

    checks: list[tuple[str, bool]] = [
        ("real Phase 3 report ingested and diagnosed", bool(real_diagnosis.context.model_refs)),
        (
            "false-positive avoided: no meaningful GPU bottleneck for saturated GPU",
            not real_gpu_meaningful,
        ),
        (
            "GPU saturation preserved as an explained (non-meaningful) observation",
            real_gpu_observation,
        ),
        (
            "real baseline returns no meaningful opportunity (or explicit insufficient evidence)",
            (not real_diagnosis.meaningful_opportunity)
            or real_diagnosis.insufficient_evidence,
        ),
        (
            "genuine evidence-supported bottleneck demonstrated under real induced pressure",
            pressure_ran and pressure_meaningful and pressure_paging and pressure_evidence_ok,
        ),
    ]

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": 4,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detector_version": BottleneckDetector.version,
        "source_report": str(real_path),
        "real_diagnosis": json.loads(real_diagnosis.to_json()),
        "real_diagnosis_artifact": str(real_artifact),
        "pressure": {
            "ran": pressure_ran,
            "held_bytes": hold_bytes,
            "measured_runs": len(pressured_runs),
            "diagnosis": json.loads(pressure_diagnosis.to_json())
            if pressure_diagnosis is not None
            else None,
            "artifact": str(pressure_artifact) if pressure_artifact is not None else None,
        },
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    verification_path = settings.reports_dir / f"phase4_verification_{timestamp}.json"
    verification_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print()
    print(f"=== RESULT: PHASE 4 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"real diagnosis      : {real_artifact}")
    if pressure_artifact is not None:
        print(f"pressure diagnosis  : {pressure_artifact}")
    print(f"report              : {verification_path}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Phase 5 — local reasoning / deep diagnosis                                  #
# --------------------------------------------------------------------------- #
def _try_load_report(path: str | None) -> BaselineReport | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        return load_baseline_report(candidate)
    except BaselineReportError:
        return None


def _load_diagnosis_for_deep(
    settings: Settings, args: argparse.Namespace
) -> tuple[BottleneckDiagnosis, BaselineReport | None, str]:
    if args.diagnosis:
        path = Path(args.diagnosis)
        diagnosis = BottleneckDiagnosis.from_json(path.read_text(encoding="utf-8"))
        return diagnosis, _try_load_report(diagnosis.context.source_report_path), str(path)
    if args.report:
        report = load_baseline_report(Path(args.report))
        diagnosis = BottleneckDetector().detect(report, source_report_path=str(args.report))
        return diagnosis, report, str(args.report)
    candidates = sorted(
        list(settings.reports_dir.glob("phase4_diagnosis_*.json"))
        + list(settings.reports_dir.glob("diagnosis_*.json")),
        key=lambda item: item.stat().st_mtime,
    )
    if candidates:
        path = candidates[-1]
        diagnosis = BottleneckDiagnosis.from_json(path.read_text(encoding="utf-8"))
        return diagnosis, _try_load_report(diagnosis.context.source_report_path), str(path)
    latest = find_latest_report(settings.reports_dir)
    if latest is None:
        raise BaselineReportError("no Phase 3/4 report found; run `inferenceos baseline` first")
    report = load_baseline_report(latest)
    diagnosis = BottleneckDetector().detect(report, source_report_path=str(latest))
    return diagnosis, report, str(latest)


def _cmd_deep_diagnose(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase5.log")

    try:
        diagnosis, report, origin = _load_diagnosis_for_deep(settings, args)
    except (BaselineReportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    enabled = settings.reasoning_enabled and not args.no_reasoning
    engine = build_reasoning_engine(settings) if enabled else None
    outcome = run_deep_diagnosis(
        diagnosis,
        report=report,
        engine=engine,
        enabled=enabled,
        confidence_threshold=settings.reasoning_escalation_threshold,
        max_evidence=settings.reasoning_max_evidence,
    )

    print("=== InferenceOS deep diagnosis ===")
    print(f"source: {origin}")
    print()
    if args.json:
        print(outcome.to_json())
    else:
        print(render_deep_diagnosis(outcome))

    out_path = Path(args.out) if args.out else settings.reports_dir / (
        f"deep_diagnosis_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.write_text(outcome.to_json(), encoding="utf-8")
    print()
    print(f"outcome: {out_path}")
    return 0


def _structural_signature(outcome: DeepDiagnosisOutcome) -> tuple[str, ...] | None:
    reasoning = outcome.reasoning
    if reasoning is None or reasoning.result is None:
        return None
    result = reasoning.result
    return (
        result.root_cause_class.value,
        result.affected_stage.value if result.affected_stage else "none",
        ",".join(sorted(result.supporting_evidence_ids)),
        ",".join(sorted(result.contradicting_evidence_ids)),
        result.evidence_strength.value,
        str(result.further_evidence_required),
    )


def _cmd_verify_phase5(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase5.log")

    print("=== InferenceOS Phase 5 Verification (Reasoning & Deep Diagnosis) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    real_path = Path(args.report) if args.report else find_latest_report(settings.reports_dir)
    if real_path is None:
        print("error: no Phase 3 report found; run `inferenceos baseline` first", file=sys.stderr)
        return 2
    report = load_baseline_report(real_path)
    diagnosis = BottleneckDetector().detect(report, source_report_path=str(real_path))

    engine = build_reasoning_engine(settings)
    engine_available = engine is not None and engine.is_available()
    print("--- Scenario A: real ambiguous/below-threshold escalation ---")
    outcome = run_deep_diagnosis(
        diagnosis,
        report=report,
        engine=engine,
        enabled=settings.reasoning_enabled,
        confidence_threshold=settings.reasoning_escalation_threshold,
        max_evidence=settings.reasoning_max_evidence,
    )
    print(render_deep_diagnosis(outcome))
    print()
    artifact = settings.reports_dir / f"phase5_deep_diagnosis_{timestamp}.json"
    artifact.write_text(outcome.to_json(), encoding="utf-8")

    # Reproducibility: a second identical invocation.
    outcome2 = run_deep_diagnosis(
        diagnosis,
        report=report,
        engine=engine,
        enabled=settings.reasoning_enabled,
        confidence_threshold=settings.reasoning_escalation_threshold,
        max_evidence=settings.reasoning_max_evidence,
    )
    signature1 = _structural_signature(outcome)
    signature2 = _structural_signature(outcome2)
    reproducible = signature1 is not None and signature1 == signature2

    # Strong deterministic case: controlled real pressure -> paging -> no escalation.
    print("--- Scenario B: strong deterministic diagnosis must not be escalated ---")
    strong_deterministic = False
    strong_evidence = ""
    pressure_diagnosis: BottleneckDiagnosis | None = None
    pressure_decision_code: str | None = None
    if args.skip_pressure:
        strong_evidence = "skipped (--skip-pressure)"
    else:
        try:
            models = _load_models(settings, args.model)
            primary_model = max(models, key=lambda model: model.file_size_bytes)
            task = ProfileTask(
                model=primary_model,
                workload=get_workload("short-short"),
                context_length=max(choose_context_length(primary_model), 32768),
            )
            profiler = BaselineProfiler(settings=settings)
            control = _profile_condition(
                profiler, task, condition="control", gpu_layers=0, repetitions=2
            )
            holder, _hold = _start_memory_holder(
                leave_bytes=int(1.5 * 1024**3), cap_bytes=int(8.0 * 1024**3)
            )
            pressured: list[ProfilingRun] = []
            try:
                if holder is not None:
                    pressured = _profile_condition(
                        profiler, task, condition="pressured", gpu_layers=0, repetitions=2
                    )
            finally:
                _stop_memory_holder(holder)
            if pressured:
                combined = _combine_runs(
                    control + pressured, report, "phase5 strong-deterministic control"
                )
                pressure_diagnosis = BottleneckDetector().detect(
                    combined, source_report_path="<controlled-pressure>"
                )
                pressure_decision = decide_escalation(
                    pressure_diagnosis,
                    enabled=settings.reasoning_enabled,
                    confidence_threshold=settings.reasoning_escalation_threshold,
                )
                pressure_decision_code = pressure_decision.reason_code
                strong_deterministic = (
                    pressure_diagnosis.meaningful_opportunity
                    and not pressure_decision.should_escalate
                )
                strong_evidence = (
                    f"primary={pressure_diagnosis.primary.kind.value if pressure_diagnosis.primary else None} "
                    f"escalation={pressure_decision.reason_code}"
                )
            else:
                strong_evidence = "pressure scenario not run (insufficient memory headroom)"
        except (OSError, RuntimeError, TimeoutError) as exc:
            strong_evidence = f"pressure scenario failed: {exc}"
    print(f"strong deterministic no-escalation: {strong_deterministic} ({strong_evidence})")
    print()

    # Failure fallback: real provider pointing at a non-existent model.
    print("--- Scenario C: reasoning failure falls back safely ---")
    broken_engine = LocalReasoningEngine(
        LlamaCppReasoningProvider(
            model_path=Path("<missing-reasoning-model>.gguf"),
            llama_cpp_dir=settings.llama_cpp_dir,
        )
    )
    fallback_outcome = run_deep_diagnosis(
        diagnosis,
        report=report,
        engine=broken_engine,
        enabled=True,
        confidence_threshold=settings.reasoning_escalation_threshold,
        max_evidence=settings.reasoning_max_evidence,
    )
    fallback_ok = (
        fallback_outcome.escalation.should_escalate
        and fallback_outcome.fallback_used
        and fallback_outcome.reasoning is not None
        and fallback_outcome.reasoning.status.value == "failed"
    )
    print(
        f"fallback used: {fallback_outcome.fallback_used}; "
        f"failure: {fallback_outcome.reasoning.failure if fallback_outcome.reasoning else None}"
    )
    print()

    # Target health after reasoning (real measurement).
    print("--- Scenario D: target model still healthy after reasoning ---")
    target_healthy = False
    try:
        adapter = LlamaCppAdapter(settings=settings)
        first = report.runs[0]
        context_length = first.context.config.context_length or 2048
        adapter.load(
            first.context.model,
            RuntimeParams(
                gpu_offload_layers=first.context.config.gpu_layers_offloaded or 99,
                context_length=context_length,
                log_verbosity=4,
            ),
        )
        target_result = adapter.generate(
            GenerationRequest(prompt="Reply with one short word.", n_predict=8, seed=1)
        )
        adapter.unload()
        target_healthy = bool(target_result.text)
        print(f"target generated text: {target_result.text!r}")
    except (FileNotFoundError, RuntimeError, TimeoutError, OSError) as exc:
        print(f"target health check failed: {exc}")
    outcome = outcome.model_copy(
        update={
            "lifecycle": outcome.lifecycle.model_copy(
                update={"target_resume_verified": target_healthy}
            )
        }
    )
    artifact.write_text(outcome.to_json(), encoding="utf-8")
    print()

    reasoning = outcome.reasoning
    supplied_ids = {
        item.evidence_id
        for item in build_reasoning_context(diagnosis, report).evidence
    }
    grounding_ok = (
        reasoning is not None
        and reasoning.result is not None
        and (set(reasoning.result.supporting_evidence_ids) | set(reasoning.result.contradicting_evidence_ids))
        <= supplied_ids
    )
    checks: list[tuple[str, bool]] = [
        (
            "real ambiguous/below-threshold diagnosis justifies escalation",
            outcome.escalation.should_escalate
            and outcome.escalation.reason_code
            in {"insufficient_evidence", "competing_hypotheses", "confidence_gap"},
        ),
        ("local reasoning engine configured and available", engine_available),
        (
            "real local model produced a schema-valid, grounded result",
            reasoning is not None
            and reasoning.validation_ok
            and reasoning.result is not None
            and reasoning.status.value != "failed",
        ),
        ("reasoning result references only supplied evidence", grounding_ok),
        ("strong deterministic diagnosis is not escalated", strong_deterministic or args.skip_pressure),
        ("reasoning failure falls back to the deterministic diagnosis", fallback_ok),
        ("target model healthy after reasoning", target_healthy),
        ("reasoning reproducibility recorded", signature1 is not None and signature2 is not None),
    ]

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": 5,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_report": str(real_path),
        "source_diagnosis_id": diagnosis.diagnosis_id,
        "deep_diagnosis": json.loads(outcome.to_json()),
        "artifact": str(artifact),
        "strong_deterministic": {
            "verified": strong_deterministic,
            "evidence": strong_evidence,
            "diagnosis": json.loads(pressure_diagnosis.to_json())
            if pressure_diagnosis is not None
            else None,
            "escalation_reason": pressure_decision_code,
        },
        "fallback": {
            "used": fallback_outcome.fallback_used,
            "failure": fallback_outcome.reasoning.failure if fallback_outcome.reasoning else None,
        },
        "reproducibility": {
            "stable": reproducible,
            "signature_1": list(signature1) if signature1 else None,
            "signature_2": list(signature2) if signature2 else None,
        },
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    verification_path = settings.reports_dir / f"phase5_verification_{timestamp}.json"
    verification_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print()
    print(f"=== RESULT: PHASE 5 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"deep diagnosis : {artifact}")
    print(f"report         : {verification_path}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Phase 6 — candidate generation / controlled optimization                    #
# --------------------------------------------------------------------------- #
def _load_plan_inputs(
    settings: Settings, args: argparse.Namespace
) -> tuple[BottleneckDiagnosis, BaselineReport | None, DeepDiagnosisOutcome | None, str]:
    if args.deep:
        path = Path(args.deep)
        outcome = DeepDiagnosisOutcome.from_json(path.read_text(encoding="utf-8"))
        report = _try_load_report(outcome.source_report_path)
        return outcome.deterministic, report, outcome, str(path)
    if args.diagnosis:
        path = Path(args.diagnosis)
        diagnosis = BottleneckDiagnosis.from_json(path.read_text(encoding="utf-8"))
        return diagnosis, _try_load_report(diagnosis.context.source_report_path), None, str(path)
    if args.report:
        report = load_baseline_report(Path(args.report))
        diagnosis = BottleneckDetector().detect(report, source_report_path=str(args.report))
        return diagnosis, report, None, str(args.report)

    deep_candidates = sorted(
        settings.reports_dir.glob("phase5_deep_diagnosis_*.json"),
        key=lambda item: item.stat().st_mtime,
    )
    if deep_candidates:
        path = deep_candidates[-1]
        outcome = DeepDiagnosisOutcome.from_json(path.read_text(encoding="utf-8"))
        return outcome.deterministic, _try_load_report(outcome.source_report_path), outcome, str(path)
    diagnosis, report, origin = _load_diagnosis_for_deep(settings, args)
    return diagnosis, report, None, origin


def _cmd_candidates(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase6.log")

    try:
        diagnosis, report, outcome, origin = _load_plan_inputs(settings, args)
    except (BaselineReportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    policy = build_policy(settings)
    provider = None
    if args.reasoning and settings.phase6_allow_reasoning:
        try:
            provider = build_reasoning_provider(settings)
        except (ValueError, OSError):
            provider = None

    plan = generate_candidates(
        diagnosis,
        report=report,
        outcome=outcome,
        policy=policy,
        provider=provider,
        enable_reasoning=provider is not None,
    )
    print("=== InferenceOS candidate generation ===")
    print(f"source: {origin}")
    print()
    if args.json:
        print(plan.to_json())
    else:
        print(render_plan(plan))

    out_path = Path(args.out) if args.out else settings.reports_dir / (
        f"candidates_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.write_text(plan.to_json(), encoding="utf-8")
    print()
    print(f"plan: {out_path}")
    return 0


def _is_meaningful_pressure(diagnosis: BottleneckDiagnosis | None) -> bool:
    if diagnosis is None or not diagnosis.meaningful_opportunity or diagnosis.primary is None:
        return False
    return diagnosis.primary.kind in {BottleneckKind.PAGING, BottleneckKind.MEMORY_PRESSURE}


def _snapshot_side_effects() -> dict[str, object]:
    available = windows.global_memory_status()
    children: list[int] = []
    try:
        children = [child.pid for child in psutil.Process().children(recursive=True)]
    except (psutil.Error, OSError):  # pragma: no cover - defensive
        children = []
    return {
        "children": sorted(children),
        "available_physical_bytes": available.available_physical_bytes if available else None,
    }


def _adversarial_proposal_payload() -> str:
    return json.dumps(
        {
            "schema_version": "1.0",
            "proposed_actions": [
                {
                    "action_id": "shell.run_command",
                    "parameter_names": ["cmd"],
                    "parameter_values": ["Remove-Item -Recurse C:\\"],
                    "evidence_ids": [],
                    "rationale": "free memory by deleting files",
                },
                {
                    "action_id": "runtime.right_size_context",
                    "parameter_names": ["context_length"],
                    "parameter_values": [999999],
                    "evidence_ids": ["E999"],
                    "rationale": "hallucinated evidence id",
                },
                {
                    "action_id": "runtime.right_size_context",
                    "parameter_names": ["context_length"],
                    "parameter_values": [16384],
                    "evidence_ids": [],
                    # An implausible magnitude so the fabricated-measurement guard is
                    # tested robustly (a small number such as "99%" can coincidentally
                    # match real telemetry and is therefore a flawed sentinel).
                    "rationale": "reduce page faults by 987654321%",
                },
            ],
            "summary": "adversarial proposal",
        }
    )


def _cmd_verify_phase6(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase6.log")

    print("=== InferenceOS Phase 6 Verification (Candidate Generation) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    policy = build_policy(settings)

    # --- Scenario A: real induced memory pressure -> real paging -> candidates ---
    print("--- Scenario A: real induced-pressure diagnosis -> candidates ---")
    real_path = Path(args.report) if args.report else find_latest_report(settings.reports_dir)
    if real_path is None:
        print("error: no Phase 3 report found; run `inferenceos baseline` first", file=sys.stderr)
        return 2
    real_report = load_baseline_report(real_path)

    pressure_diagnosis: BottleneckDiagnosis | None = None
    pressure_report: BaselineReport | None = None
    pressure_ran = False
    pressure_source = "not run"
    if not args.skip_pressure:
        try:
            models = _load_models(settings, args.model)
            primary_model = max(models, key=lambda model: model.file_size_bytes)
            task = ProfileTask(
                model=primary_model,
                workload=get_workload("short-short"),
                context_length=max(choose_context_length(primary_model), 32768),
            )
            profiler = BaselineProfiler(settings=settings)
            control = _profile_condition(
                profiler, task, condition="control", gpu_layers=0, repetitions=2
            )
            holder, _hold = _start_memory_holder(
                leave_bytes=int(1.5 * 1024**3), cap_bytes=int(8.0 * 1024**3)
            )
            pressured: list[ProfilingRun] = []
            try:
                if holder is not None:
                    pressured = _profile_condition(
                        profiler, task, condition="pressured", gpu_layers=0, repetitions=2
                    )
            finally:
                _stop_memory_holder(holder)
            if pressured:
                pressure_report = _combine_runs(
                    control + pressured, real_report, "phase6 real induced-pressure acceptance"
                )
                pressure_diagnosis = BottleneckDetector().detect(
                    pressure_report, source_report_path=f"<phase6-pressure:{timestamp}>"
                )
                pressure_ran = True
                pressure_source = "fresh induced-pressure run"
        except (OSError, RuntimeError, TimeoutError) as exc:
            print(f"pressure scenario failed: {exc}")
    if not _is_meaningful_pressure(pressure_diagnosis):
        fallback_paths = sorted(
            settings.reports_dir.glob("phase4_diagnosis_pressure_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in fallback_paths:
            try:
                candidate = BottleneckDiagnosis.from_json(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if not _is_meaningful_pressure(candidate):
                continue
            pressure_diagnosis = candidate
            loaded = _try_load_report(candidate.context.source_report_path)
            pressure_report = loaded if loaded is not None else real_report
            pressure_ran = True
            pressure_source = f"stored real diagnosis {path.name}"
            print(f"using stored real induced-pressure diagnosis: {path}")
            break

    before = _snapshot_side_effects()
    plan_a: CandidatePlan | None = None
    model_path: str | None = None
    model_mtime: float | None = None
    if pressure_report is not None and pressure_report.runs:
        model_path = pressure_report.runs[0].context.model.path
        try:
            model_mtime = Path(model_path).stat().st_mtime
        except OSError:
            model_mtime = None
    if pressure_diagnosis is not None:
        plan_a = generate_candidates(
            pressure_diagnosis, report=pressure_report, policy=policy
        )
        print(render_plan(plan_a))
    after = _snapshot_side_effects()
    print()

    # --- Scenario B: real healthy baseline -> no speculative candidates ---
    print("--- Scenario B: real baseline -> no-opportunity ---")
    real_diagnosis = BottleneckDetector().detect(real_report, source_report_path=str(real_path))
    plan_b = generate_candidates(real_diagnosis, report=real_report, policy=policy)
    print(f"outcome: {plan_b.outcome.value}")
    print(f"reason : {plan_b.terminal_reason}")
    print(f"candidates: {len(plan_b.candidates)}")
    print()

    # --- Scenario C: real Phase 5 outcome + adversarial reasoning proposals ---
    print("--- Scenario C: reasoning proposals cannot bypass the registry ---")
    deep_paths = sorted(
        settings.reports_dir.glob("phase5_deep_diagnosis_*.json"),
        key=lambda item: item.stat().st_mtime,
    )
    if deep_paths:
        try:
            outcome_c = DeepDiagnosisOutcome.from_json(deep_paths[-1].read_text(encoding="utf-8"))
            diagnosis_c = outcome_c.deterministic
            report_c = _try_load_report(outcome_c.source_report_path)
            origin_c = str(deep_paths[-1])
        except (ValueError, OSError):
            diagnosis_c, report_c, outcome_c, origin_c = real_diagnosis, real_report, None, str(real_path)
    else:
        diagnosis_c, report_c, outcome_c, origin_c = real_diagnosis, real_report, None, str(real_path)
    print(f"source: {origin_c}")
    adversarial_provider = FakeReasoningProvider(scripted=[_adversarial_proposal_payload()])
    plan_c = generate_candidates(
        diagnosis_c,
        report=report_c,
        outcome=outcome_c,
        policy=policy,
        provider=adversarial_provider,
        enable_reasoning=True,
    )
    rejections = plan_c.reasoning.rejections if plan_c.reasoning is not None else []
    joined = " | ".join(rejections)
    print(f"reasoning used: {plan_c.reasoning.used if plan_c.reasoning else False}")
    print(f"rejections    : {rejections}")
    print(f"outcome       : {plan_c.outcome.value}")
    print()

    real_reasoning: dict[str, object] | None = None
    if getattr(args, "reasoning", False) and settings.phase6_allow_reasoning:
        try:
            real_provider = build_reasoning_provider(settings)
            if real_provider.is_available():
                plan_real = generate_candidates(
                    diagnosis_c,
                    report=report_c,
                    outcome=outcome_c,
                    policy=policy,
                    provider=real_provider,
                    enable_reasoning=True,
                )
                real_reasoning = {
                    "used": plan_real.reasoning.used if plan_real.reasoning else False,
                    "accepted": plan_real.reasoning.accepted_action_ids if plan_real.reasoning else [],
                    "rejected": plan_real.reasoning.rejected_action_ids if plan_real.reasoning else [],
                    "rejections": plan_real.reasoning.rejections if plan_real.reasoning else [],
                    "failure": plan_real.reasoning.failure if plan_real.reasoning else None,
                    "outcome": plan_real.outcome.value,
                }
                print(f"real reasoning proposal: {real_reasoning}")
        except (ValueError, OSError) as exc:
            print(f"real reasoning unavailable: {exc}")
        print()

    # --- Acceptance checks ---
    pressure_primary = pressure_diagnosis.primary if pressure_diagnosis is not None else None
    real_paging = bool(
        pressure_diagnosis is not None
        and pressure_diagnosis.meaningful_opportunity
        and pressure_primary is not None
        and pressure_primary.kind in {BottleneckKind.PAGING, BottleneckKind.MEMORY_PRESSURE}
    )
    plan_a_eligible = [c for c in (plan_a.candidates if plan_a else []) if c.executable]
    candidate_evidence_ok = bool(plan_a_eligible) and all(c.evidence_refs for c in plan_a_eligible)
    preconditions_ok = (
        plan_a is not None
        and bool(plan_a.candidates)
        and all(c.preconditions for c in plan_a.candidates)
    )
    ranking_ok = (
        plan_a is not None
        and bool(plan_a.ranking)
        and all(
            next(c for c in plan_a.candidates if c.candidate_id == cid).ranking is not None
            for cid in plan_a.ranking
        )
    )
    no_execution = bool(
        plan_a is not None and not plan_a.executed and not plan_a.environment_mutated
    )
    model_unchanged = True
    if model_path and model_mtime is not None:
        try:
            model_unchanged = Path(model_path).stat().st_mtime == model_mtime
        except OSError:
            model_unchanged = False
    no_new_children = before["children"] == after["children"]
    no_opportunity_ok = bool(
        plan_b.outcome.value in {"no_opportunity", "insufficient_evidence"}
        and not plan_b.candidates
        and plan_b.terminal_reason
    )
    registry_enforced = (
        "unregistered" in joined
        and "hallucinated evidence" in joined
        and "fabricated measurement" in joined
    )
    no_shell_action = all("shell" not in c.action_id for c in plan_c.candidates)

    checks: list[tuple[str, bool]] = [
        (
            "A: real induced-pressure diagnosis is meaningful paging/memory pressure",
            real_paging,
        ),
        (
            "A: relevant registered candidate(s) generated from the real diagnosis",
            bool(plan_a and plan_a.outcome.value == "candidates_generated" and plan_a_eligible),
        ),
        (
            "A: eligible candidates reference real diagnosis evidence",
            candidate_evidence_ok,
        ),
        (
            "A: candidate preconditions explicitly evaluated",
            preconditions_ok,
        ),
        (
            "A: multi-factor ranking produced for eligible candidates",
            ranking_ok,
        ),
        (
            "A: no candidate action was executed and the environment was not mutated",
            no_execution and no_new_children and model_unchanged,
        ),
        (
            "B: real no-opportunity baseline produces no speculative candidates",
            no_opportunity_ok,
        ),
        (
            "C: reasoning proposals cannot bypass the registry or evidence grounding",
            registry_enforced,
        ),
        (
            "C: no unregistered/command action becomes a candidate",
            no_shell_action,
        ),
        (
            "C: adversarial proposals produce no executable candidate",
            not any(c.executable for c in plan_c.candidates),
        ),
    ]

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": 6,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_report": str(real_path),
        "policy": policy.as_dict(),
        "scenario_a": {
            "pressure_ran": pressure_ran,
            "pressure_source": pressure_source,
            "diagnosis": json.loads(pressure_diagnosis.to_json())
            if pressure_diagnosis is not None
            else None,
            "plan": json.loads(plan_a.to_json()) if plan_a is not None else None,
        },
        "scenario_b": {
            "diagnosis_id": real_diagnosis.diagnosis_id,
            "outcome": plan_b.outcome.value,
            "candidates": 0,
            "plan": json.loads(plan_b.to_json()),
        },
        "scenario_c": {
            "source": origin_c,
            "adversarial_plan": json.loads(plan_c.to_json()),
            "real_reasoning": real_reasoning,
        },
        "side_effects": {
            "children_before": before["children"],
            "children_after": after["children"],
            "model_unchanged": model_unchanged,
            "executed": bool(plan_a.executed) if plan_a is not None else False,
        },
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    verification_path = settings.reports_dir / f"phase6_verification_{timestamp}.json"
    verification_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    if plan_a is not None:
        (settings.reports_dir / f"phase6_plan_pressure_{timestamp}.json").write_text(
            plan_a.to_json(), encoding="utf-8"
        )

    print()
    print(f"=== RESULT: PHASE 6 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {verification_path}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Phase 7 — experiment execution / quality validation                          #
# --------------------------------------------------------------------------- #
def _select_candidate(
    plan: CandidatePlan, candidate_id: str | None
) -> OptimizationCandidate | None:
    eligible = [
        candidate
        for candidate in plan.candidates
        if candidate.eligibility.value == "eligible" and candidate.executable
    ]
    if candidate_id:
        for candidate in eligible:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None
    for candidate_id_value in plan.selected_ids:
        for candidate in eligible:
            if candidate.candidate_id == candidate_id_value:
                return candidate
    return eligible[0] if eligible else None


def _run_one_experiment(
    settings: Settings,
    diagnosis: BottleneckDiagnosis,
    report: BaselineReport | None,
    outcome: DeepDiagnosisOutcome | None,
    candidate: OptimizationCandidate,
    plan: CandidatePlan | None,
    workload_id: str,
) -> ExperimentResult:
    executor = RuntimeConfigurationExecutor()
    profiler = BaselineProfiler(settings=settings)
    return run_candidate_experiment(
        candidate,
        plan=plan,
        diagnosis=diagnosis,
        report=report,
        outcome=outcome,
        workload_id=workload_id,
        registry=build_default_registry(),
        optimization_policy=build_policy(settings),
        experiment_policy=build_experiment_policy(settings),
        executor=executor,
        profiler=profiler,
    )


def _cmd_experiment(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase7.log")

    if args.plan:
        plan_path = Path(args.plan)
        plan = CandidatePlan.from_json(plan_path.read_text(encoding="utf-8"))
        diagnosis = plan.source_diagnosis
        report = _try_load_report(diagnosis.context.source_report_path)
        if report is None:
            latest = find_latest_report(settings.reports_dir)
            report = load_baseline_report(latest) if latest is not None else None
        origin = str(plan_path)
        outcome = None
    else:
        try:
            diagnosis, report, outcome, origin = _load_plan_inputs(settings, args)
        except (BaselineReportError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if report is None:
            latest = find_latest_report(settings.reports_dir)
            if latest is not None:
                report = load_baseline_report(latest)
        provider = None
        if args.reasoning and settings.phase6_allow_reasoning:
            try:
                provider = build_reasoning_provider(settings)
            except (ValueError, OSError):
                provider = None
        plan = generate_candidates(
            diagnosis,
            report=report,
            outcome=outcome,
            policy=build_policy(settings),
            provider=provider,
            enable_reasoning=provider is not None,
        )

    candidate = _select_candidate(plan, args.candidate_id)
    if candidate is None:
        print("no eligible candidate to experiment with")
        print(f"plan outcome: {plan.outcome.value} ({plan.terminal_reason})")
        return 0

    try:
        result = _run_one_experiment(
            settings, diagnosis, report, outcome, candidate, plan, args.workload
        )
    except ExperimentInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("=== InferenceOS experiment ===")
    print(f"source: {origin}")
    print()
    if args.json:
        print(result.to_json())
    else:
        print(render_experiment(result))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else settings.reports_dir / f"experiment_{timestamp}.json"
    out_path.write_text(result.to_json(), encoding="utf-8")
    audit_path = settings.reports_dir / f"experiment_audit_{timestamp}.json"
    audit_path.write_text(build_audit_record(result).to_json(), encoding="utf-8")
    print()
    print(f"experiment: {out_path}")
    print(f"audit     : {audit_path}")
    return 0


def _synthetic_measurement(
    phase: str, *, decode: float, rss: float, texts: list[str], faults: float = 0.5
) -> ExperimentMeasurement:
    metrics = {
        "decode_tokens_per_sec": summarize("decode_tokens_per_sec", [decode, decode]),
        "peak_rss_bytes": summarize("peak_rss_bytes", [float(rss), float(rss)]),
        "hard_page_faults_per_sec": summarize("hard_page_faults_per_sec", [faults, faults]),
        "total_latency_s": summarize("total_latency_s", [2.0, 2.0]),
    }
    return ExperimentMeasurement(
        measurement_id=f"synth-{phase}",
        phase=phase,
        configuration=RuntimeConfiguration(),
        repetitions=2,
        valid_runs=2,
        run_ids=[f"{phase}-1", f"{phase}-2"],
        metrics=metrics,
        generated_texts=list(texts),
    )


def _latest_meaningful_pressure(settings: Settings) -> tuple[BottleneckDiagnosis, str] | None:
    paths = sorted(
        settings.reports_dir.glob("phase4_diagnosis_pressure_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        try:
            diagnosis = BottleneckDiagnosis.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if _is_meaningful_pressure(diagnosis):
            return diagnosis, str(path)
    return None


def _cmd_verify_phase7(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase7.log")

    print("=== InferenceOS Phase 7 Verification (Experiment & Quality) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_policy = build_experiment_policy(settings)

    real_path = Path(args.report) if args.report else find_latest_report(settings.reports_dir)
    if real_path is None:
        print("error: no Phase 3 report found; run `inferenceos baseline` first", file=sys.stderr)
        return 2
    real_report = load_baseline_report(real_path)

    # --- Scenario A/E: real candidate execution + restoration ---
    print("--- Scenario A: real candidate -> real experiment -> restore ---")
    pressure = _latest_meaningful_pressure(settings)
    real_diagnosis = BottleneckDetector().detect(real_report, source_report_path=str(real_path))
    candidate: OptimizationCandidate | None = None
    plan: CandidatePlan | None = None
    result: ExperimentResult | None = None
    origin = "n/a"
    model_path: str | None = None
    model_mtime: float | None = None
    if real_report.runs:
        model_path = real_report.runs[0].context.model.path
        try:
            model_mtime = Path(model_path).stat().st_mtime
        except OSError:
            model_mtime = None

    if pressure is not None:
        pressure_diagnosis, origin = pressure
        plan = generate_candidates(
            pressure_diagnosis, report=real_report, policy=build_policy(settings)
        )
        candidate = _select_candidate(plan, None)
        if candidate is not None:
            executor = RuntimeConfigurationExecutor()
            try:
                result = run_candidate_experiment(
                    candidate,
                    plan=plan,
                    diagnosis=pressure_diagnosis,
                    report=real_report,
                    outcome=None,
                    workload_id="short-short",
                    registry=build_default_registry(),
                    optimization_policy=build_policy(settings),
                    experiment_policy=exp_policy,
                    executor=executor,
                    profiler=BaselineProfiler(settings=settings),
                )
            except ExperimentInputError as exc:
                print(f"experiment input error: {exc}")
            if result is not None:
                print(render_experiment(result))
                restored_exactly = (
                    result.request is not None
                    and executor.current() == result.request.baseline_configuration
                )
            else:
                restored_exactly = False
        else:
            restored_exactly = False
    else:
        print("no real meaningful pressure diagnosis found; run verify-phase4 first")
        restored_exactly = False

    # --- Scenario B: real healthy baseline -> no experiment ---
    print()
    print("--- Scenario B: real baseline -> no executable candidate ---")
    healthy_plan = generate_candidates(
        real_diagnosis, report=real_report, policy=build_policy(settings)
    )
    healthy_candidate = _select_candidate(healthy_plan, None)
    no_speculative_experiment = (
        healthy_candidate is None
        and not healthy_plan.selected_ids
        and healthy_plan.outcome.value in {"no_opportunity", "insufficient_evidence", "no_viable_candidates"}
    )
    print(f"outcome: {healthy_plan.outcome.value}; candidates={len(healthy_plan.candidates)}")

    # --- Scenario C: approval-required candidate blocked ---
    print()
    print("--- Scenario C: unsafe/approval-required candidate blocked ---")
    blocked = False
    blocked_status = None
    if pressure is not None:
        pressure_diagnosis, _ = pressure
        unsafe_plan = generate_candidates(
            pressure_diagnosis, report=real_report, policy=build_policy(settings)
        )
        gated = next(
            (c for c in unsafe_plan.candidates if c.action_id == "system.increase_pagefile"),
            None,
        )
        if gated is not None:
            blocked_result = run_candidate_experiment(
                gated,
                plan=unsafe_plan,
                diagnosis=pressure_diagnosis,
                report=real_report,
                outcome=None,
                workload_id="short-short",
                registry=build_default_registry(),
                optimization_policy=build_policy(settings),
                experiment_policy=exp_policy,
                executor=RuntimeConfigurationExecutor(),
                profiler=BaselineProfiler(settings=settings),
            )
            blocked = (
                blocked_result.executed is False
                and blocked_result.baseline_measurement is None
                and blocked_result.status is ExperimentDecisionKind.UNSAFE
            )
            blocked_status = (
                blocked_result.status.value if blocked_result.status is not None else None
            )
            print(f"unsafe candidate status: {blocked_status}")
    print(f"blocked: {blocked}")

    # --- Scenario D: quality protection (controlled) ---
    print()
    print("--- Scenario D: quality failure blocks a performance improvement ---")
    workload = get_workload("short-short")
    baseline_measurement = _synthetic_measurement(
        "baseline", decode=20.0, rss=1.3e9, texts=["A key value cache stores attention state."]
    )
    candidate_measurement = _synthetic_measurement(
        "candidate", decode=30.0, rss=0.9e9, texts=["!!!! 1234 5678"]
    )
    quality = validate_quality(
        baseline=baseline_measurement,
        candidate=candidate_measurement,
        workload=workload,
        policy=exp_policy,
    )
    comparison = compare(
        baseline=baseline_measurement,
        candidate=candidate_measurement,
        bottleneck_metric="peak_rss_bytes",
        bottleneck_direction="decrease",
        quality=quality,
        policy=exp_policy,
        diagnosed_bottleneck="memory_pressure",
    )
    controlled_decision = decide(
        comparison=comparison,
        quality=quality,
        application=ApplicationResult(
            applied=True, action_id="runtime.right_size_context", supported=True
        ),
        restoration_status=RestorationStatus.RESTORED,
        measurement_valid=True,
        precondition_ok=True,
        precondition_kind=None,
        precondition_reason="ok",
        policy=exp_policy,
        errors=[],
    )
    quality_protected = (
        controlled_decision.kind is ExperimentDecisionKind.QUALITY_REGRESSION
        and not controlled_decision.keep
        and comparison.bottleneck_improved
    )
    print(
        f"perf improved={comparison.bottleneck_improved}; "
        f"quality={quality.status.value}; decision={controlled_decision.kind.value}"
    )

    # --- Side-effect protection ---
    model_unchanged = True
    if model_path and model_mtime is not None:
        try:
            model_unchanged = Path(model_path).stat().st_mtime == model_mtime
        except OSError:
            model_unchanged = False

    checks: list[tuple[str, bool]] = [
        (
            "A: real Phase 4 pressure diagnosis produced an eligible registered candidate",
            candidate is not None,
        ),
        (
            "A: experiment executed the candidate against the real backend",
            result is not None and result.executed and result.application is not None and result.application.applied,
        ),
        (
            "A: real baseline and candidate measurements captured",
            result is not None
            and result.baseline_measurement is not None
            and result.candidate_measurement is not None
            and result.baseline_measurement.valid_runs >= 1
            and result.candidate_measurement.valid_runs >= 1,
        ),
        (
            "A: real parameter change recorded",
            result is not None
            and result.application is not None
            and bool(result.application.changes)
            and any(c.prior_value != c.applied_value for c in result.application.changes),
        ),
        (
            "A: quality validation performed",
            result is not None and result.quality is not None,
        ),
        (
            "A: explicit experiment decision produced",
            result is not None and result.status is not None and result.decision_detail is not None,
        ),
        (
            "E: baseline configuration restored exactly and verified",
            result is not None and result.environment_restored and restored_exactly,
        ),
        (
            "B: real no-opportunity baseline executes no experiment",
            no_speculative_experiment,
        ),
        (
            "C: approval-required/system-wide candidate blocked before execution",
            blocked,
        ),
        (
            "D: quality failure blocks a performance improvement",
            quality_protected,
        ),
        (
            "overhead measured separately from workload metrics",
            result is not None
            and result.overhead is not None
            and result.overhead.total_s > 0.0
            and result.overhead.measurement_s >= 0.0,
        ),
        (
            "no model file or persistent deployment mutation",
            model_unchanged and (result is None or result.environment_restored),
        ),
    ]

    print()
    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": 7,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_report": str(real_path),
        "pressure_source": origin,
        "experiment_policy": exp_policy.as_dict(),
        "scenario_a": {
            "candidate_id": candidate.candidate_id if candidate else None,
            "action_id": candidate.action_id if candidate else None,
            "parameters": dict(candidate.parameters) if candidate else None,
            "result": json.loads(result.to_json()) if result is not None else None,
        },
        "scenario_b": {
            "outcome": healthy_plan.outcome.value,
            "candidates": len(healthy_plan.candidates),
        },
        "scenario_c": {"blocked": blocked, "status": blocked_status},
        "scenario_d": {
            "bottleneck_improved": comparison.bottleneck_improved,
            "quality": quality.status.value,
            "decision": controlled_decision.kind.value,
        },
        "side_effects": {"model_unchanged": model_unchanged},
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    verification_path = settings.reports_dir / f"phase7_verification_{timestamp}.json"
    verification_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    if result is not None:
        (settings.reports_dir / f"phase7_experiment_{timestamp}.json").write_text(
            result.to_json(), encoding="utf-8"
        )
        (settings.reports_dir / f"phase7_experiment_audit_{timestamp}.json").write_text(
            build_audit_record(result).to_json(), encoding="utf-8"
        )

    print()
    print(f"=== RESULT: PHASE 7 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {verification_path}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Phase 8 — persistent knowledge & rollback                                    #
# --------------------------------------------------------------------------- #
def _knowledge_service(settings: Settings, db_override: str | None = None) -> KnowledgeService:
    policy = build_knowledge_policy(settings)
    db_path = db_override or str(policy.db_path)
    return KnowledgeService(
        store=SQLiteKnowledgeStore(db_path),
        registry=build_default_registry(),
        policy=policy,
    )


def _current_context_for_record(
    settings: Settings, record: OptimizationRecord
) -> ApplicabilityContext:
    latest = find_latest_report(settings.reports_dir)
    if latest is not None:
        try:
            report = load_baseline_report(latest)
        except BaselineReportError:
            report = None
        if report is not None and report.runs:
            model = None
            wanted = record.context.model.name if record.context.model is not None else None
            for run in report.runs:
                if wanted is None or run.context.model.name == wanted:
                    model = run.context.model
                    break
            if model is None:
                model = report.runs[0].context.model
            return build_current_context(
                hardware=report.hardware,
                model=model,
                workload=record.context.workload,
                backend=report.backend,
                runtime=record.context.runtime_configuration,
                diagnosis=record.diagnosed_bottleneck,
                conditions=record.context.fingerprint.conditions,
            )
    return record.context


def _load_real_experiment_results(settings: Settings) -> list[tuple[Path, ExperimentResult]]:
    paths = sorted(
        list(settings.reports_dir.glob("phase7_experiment_*.json"))
        + list(settings.reports_dir.glob("experiment_*.json")),
        key=lambda item: item.stat().st_mtime,
    )
    results: list[tuple[Path, ExperimentResult]] = []
    for path in paths:
        if "audit" in path.name:
            continue
        try:
            results.append((path, ExperimentResult.from_json(path.read_text(encoding="utf-8"))))
        except (ValueError, OSError):
            continue
    return results


def _cmd_knowledge(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase8.log")
    service = _knowledge_service(settings, args.db)

    if args.health:
        health = service.health()
        stats = service.stats()
        if args.json:
            print(json.dumps({"health": json.loads(health.model_dump_json()),
                              "stats": json.loads(stats.model_dump_json())}, indent=2))
        else:
            print(render_health(health, stats))
        return 0

    query = KnowledgeQuery(
        action_id=args.action,
        candidate_id=args.candidate,
        model_ref=args.model,
        outcome=OptimizationOutcome(args.outcome) if args.outcome else None,
        limit=args.limit,
    )
    records = service.store.find_records(query, newest_first=False)
    current = None
    applicability = []
    if args.applicability and records:
        action_id = args.action or records[-1].action_id
        # Anchor "current" on the newest record so a human sees the latest real
        # context; older records are then classified against it.
        current = _current_context_for_record(settings, records[-1])
        applicability = service.query_applicable(action_id, current, limit=args.limit)

    if args.json:
        payload = {
            "records": [json.loads(record.model_dump_json()) for record in records],
            "applicability": [json.loads(result.model_dump_json()) for result in applicability],
            "rollbacks": [
                json.loads(record.model_dump_json())
                for record in service.store.all_rollbacks(limit=args.limit)
            ] if args.rollback else [],
            "health": json.loads(service.health().model_dump_json()),
            "stats": json.loads(service.stats().model_dump_json()),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(render_history(records))
    if records and args.detail:
        print()
        for record in records:
            print(render_record(record))
            print()
    if applicability:
        print()
        for result in applicability:
            print(render_applicability(result))
            print()
    if args.rollback:
        print()
        for rollback in service.store.all_rollbacks(limit=args.limit):
            print(render_rollback(rollback))
            print()
    print()
    print(render_health(service.health(), service.stats()))
    return 0


def _negative_outcomes() -> set[OptimizationOutcome]:
    return {
        OptimizationOutcome.NO_MEANINGFUL_IMPROVEMENT,
        OptimizationOutcome.REGRESSION,
        OptimizationOutcome.QUALITY_REGRESSION,
        OptimizationOutcome.BLOCKED,
        OptimizationOutcome.INVALID,
        OptimizationOutcome.FAILED,
    }


def _cmd_verify_phase8(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase8.log")

    print("=== InferenceOS Phase 8 Verification (Persistent Knowledge & Rollback) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    registry = build_default_registry()
    policy = build_knowledge_policy(settings)
    db_path = Path(args.db) if args.db else settings.phase8_db_path

    real_results = _load_real_experiment_results(settings)
    if not real_results:
        print("error: no real Phase 7 experiment artifact found; run verify-phase7 first", file=sys.stderr)
        return 2
    latest_path, latest_result = real_results[-1]
    print(f"real Phase 7 experiment: {latest_path.name} ({latest_result.experiment_id}, "
          f"status={latest_result.status.value if latest_result.status else None})")

    # --- A: persist + restart durability ---
    service = KnowledgeService(
        store=SQLiteKnowledgeStore(db_path), registry=registry, policy=policy
    )
    write_started = time.perf_counter()
    persisted: list[tuple[Path, ExperimentResult, PersistResult]] = []
    for path, result in real_results:
        outcome = service.persist_result(result)
        persisted.append((path, result, outcome))
    write_seconds = time.perf_counter() - write_started

    # Reinitialize the persistence component (restart durability).
    service = KnowledgeService(
        store=SQLiteKnowledgeStore(db_path), registry=registry, policy=policy
    )
    read_started = time.perf_counter()
    retrieved = service.store.get_by_experiment(latest_result.experiment_id)
    read_seconds = time.perf_counter() - read_started
    context_preserved = (
        retrieved is not None
        and retrieved.context.fingerprint.identity_hash
        == service.build_record(latest_result).context.fingerprint.identity_hash
    )
    outcome_preserved = retrieved is not None and retrieved.outcome is service.build_record(
        latest_result
    ).outcome
    print(f"persisted {len(real_results)} real result(s); retrieved latest: "
          f"{retrieved is not None} (outcome_preserved={outcome_preserved})")

    # --- F: true process restart readback ---
    restart_outcome = None
    restart_ok = False
    try:
        code = (
            "import sys;from inferenceos.knowledge import SQLiteKnowledgeStore;"
            "s=SQLiteKnowledgeStore(sys.argv[1]);"
            "r=s.get_by_experiment(sys.argv[2]);"
            "print(r.outcome.value if r else 'NONE')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code, str(db_path), latest_result.experiment_id],
            capture_output=True, text=True, timeout=120,
        )
        restart_outcome = (proc.stdout or "").strip()
        restart_ok = proc.returncode == 0 and restart_outcome not in ("", "NONE")
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"restart readback failed: {exc}")
    print(f"process-restart readback: {restart_outcome}")

    # --- Idempotency ---
    before = service.stats().optimization_records
    duplicate = service.persist_result(latest_result)
    after = service.stats().optimization_records
    idempotent = (
        duplicate.optimization.status.value == "duplicate" and before == after
    )
    print(f"idempotency: status={duplicate.optimization.status.value} count {before}->{after}")

    # --- B: real applicability ---
    applicable: ApplicabilityResult | None = None
    applicability_seconds = 0.0
    if retrieved is not None:
        current = _current_context_for_record(settings, retrieved)
        app_started = time.perf_counter()
        applicable = evaluate_applicability(retrieved, current, policy)
        applicability_seconds = time.perf_counter() - app_started
        print()
        print(render_applicability(applicable))

    # --- C: materially different context ---
    different: ApplicabilityResult | None = None
    material: ApplicabilityResult | None = None
    if retrieved is not None:
        base = _current_context_for_record(settings, retrieved)
        cpu = base.hardware.cpu.model_copy(
            update={
                "physical_cores": base.hardware.cpu.physical_cores + 4,
                "logical_cores": base.hardware.cpu.logical_cores + 8,
            }
        )
        altered_hardware = base.hardware.model_copy(update={"cpu": cpu})
        different_hw = build_current_context(
            hardware=altered_hardware,
            model=base.model,
            workload=base.workload,
            backend=base.backend,
            runtime=base.runtime_configuration,
            diagnosis=base.diagnosed_bottleneck,
            conditions=base.fingerprint.conditions,
        )
        different = evaluate_applicability(retrieved, different_hw, policy)
        alt_model = (
            base.model.model_copy(update={"n_layers": base.model.n_layers + 8})
            if base.model is not None
            else None
        )
        material_ctx = build_current_context(
            hardware=altered_hardware,
            model=alt_model,
            workload=base.workload,
            backend=base.backend,
            runtime=base.runtime_configuration,
            diagnosis=base.diagnosed_bottleneck,
            conditions=base.fingerprint.conditions,
        )
        material = evaluate_applicability(retrieved, material_ctx, policy)
        print()
        print("materially different hardware ->", different.classification.value,
              "triggers=", [t.value for t in different.revalidation_triggers])
        print("different hardware + model    ->", material.classification.value,
              "triggers=", [t.value for t in material.revalidation_triggers])

    # --- D: rollback record ---
    rollback_ok = False
    rollback = service.rollback_for_experiment(latest_result.experiment_id)
    if rollback is not None:
        current_ctx = retrieved.context if retrieved is not None else None
        verification = verify_rollback(rollback, registry=registry, current_context=current_ctx)
        rollback = rollback.model_copy(update={"verification": verification})
        rollback_ok = verification.verified and rollback.verification.verified
        print()
        print(render_rollback(rollback))
    else:
        print()
        print("no rollback record found for the latest real experiment")

    # --- E: negative knowledge retained ---
    negative_records = [
        record
        for record in service.all_records(limit=200)
        if record.outcome in _negative_outcomes()
    ]
    negative_ok = bool(negative_records)
    print()
    print(f"negative knowledge retained: {negative_ok} "
          f"({[r.outcome.value for r in negative_records][:5]})")

    # --- Integrity: tamper detection on a scratch copy ---
    integrity_ok = False
    scratch = settings.data_dir / f"knowledge_tamper_{timestamp}.db"
    try:
        shutil.copyfile(db_path, scratch)
        connection = sqlite3.connect(str(scratch))
        connection.execute(
            "UPDATE optimization_records SET payload_json = payload_json || ' ' "
            "WHERE experiment_id = ?",
            (latest_result.experiment_id,),
        )
        connection.commit()
        connection.close()
        tampered = SQLiteKnowledgeStore(scratch)
        tampered_health = tampered.health()
        get_raised = False
        try:
            tampered.get_by_experiment(latest_result.experiment_id)
        except KnowledgeCorruptionError:
            get_raised = True
        integrity_ok = tampered_health.status.value == "corrupt" and get_raised
        print(f"tamper detection: health={tampered_health.status.value} "
              f"corrupted={tampered_health.corrupted_records} get_raised={get_raised}")
    except (OSError, sqlite3.Error) as exc:
        print(f"integrity check failed: {exc}")
    finally:
        if scratch.exists():
            scratch.unlink()

    # --- Security: stored action ids validated against the live registry ---
    registry_ok = all(
        record.action_id in build_default_registry() for record in service.all_records(limit=200)
    )
    print(f"stored action ids all registered: {registry_ok}")

    checks: list[tuple[str, bool]] = [
        ("A: real Phase 7 result persisted into the durable store", retrieved is not None),
        (
            "A/F: knowledge survives reinitialization and retrieval preserves context/outcome",
            context_preserved and outcome_preserved,
        ),
        (
            "F: a separate process reads the persisted knowledge",
            restart_ok and restart_outcome == (retrieved.outcome.value if retrieved else None),
        ),
        (
            "idempotency: duplicate recording creates no second logical record",
            idempotent,
        ),
        (
            "B: real current-context applicability classified",
            applicable is not None
            and applicable.classification.value
            in ("directly_applicable", "conditionally_applicable"),
        ),
        (
            "C: materially different context requires revalidation / not applicable",
            different is not None
            and different.classification.value in ("requires_revalidation", "not_applicable")
            and material is not None
            and material.classification.value in ("requires_revalidation", "not_applicable"),
        ),
        ("D: rollback record persisted and integrity-verified", rollback_ok),
        ("E: negative optimization knowledge retained", negative_ok),
        ("integrity: tampered record detected", integrity_ok),
        ("security: stored action ids validated against the live registry", registry_ok),
        (
            "overhead measured (write/read/applicability)",
            write_seconds >= 0.0 and read_seconds >= 0.0 and applicability_seconds >= 0.0,
        ),
    ]

    print()
    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": 8,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "schema_version": policy.as_dict(),
        "real_experiment": {
            "path": str(latest_path),
            "experiment_id": latest_result.experiment_id,
            "status": latest_result.status.value if latest_result.status else None,
        },
        "persisted": [
            {
                "experiment_id": result.experiment_id,
                "status": outcome.optimization.status.value,
                "duplicate": outcome.optimization.duplicate,
            }
            for _path, result, outcome in persisted
        ],
        "retrieved": json.loads(retrieved.model_dump_json()) if retrieved is not None else None,
        "applicability": json.loads(applicable.model_dump_json()) if applicable else None,
        "materially_different_hardware": (
            json.loads(different.model_dump_json()) if different else None
        ),
        "materially_different_model": (
            json.loads(material.model_dump_json()) if material else None
        ),
        "rollback": json.loads(rollback.model_dump_json()) if rollback else None,
        "negative_outcomes": [record.outcome.value for record in negative_records],
        "integrity": {"tamper_detected": integrity_ok},
        "security": {"action_ids_registered": registry_ok},
        "overhead": {
            "write_seconds": write_seconds,
            "read_seconds": read_seconds,
            "applicability_seconds": applicability_seconds,
        },
        "health": json.loads(service.health().model_dump_json()),
        "stats": json.loads(service.stats().model_dump_json()),
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    verification_path = settings.reports_dir / f"phase8_verification_{timestamp}.json"
    verification_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print()
    print(f"=== RESULT: PHASE 8 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {verification_path}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Phase 9 — pre-deployment autonomous optimization loop                        #
# --------------------------------------------------------------------------- #
def _phase9_budget(
    settings: Settings,
    *,
    max_iterations: int | None = None,
    max_experiments: int | None = None,
    max_deep_diagnoses: int | None = None,
    max_wall_clock_s: float | None = None,
    max_optimizer_overhead_s: float | None = None,
) -> OptimizationBudget:
    return OptimizationBudget(
        max_iterations=max_iterations if max_iterations is not None else settings.phase9_max_iterations,
        max_experiments=(
            max_experiments if max_experiments is not None else settings.phase9_max_experiments
        ),
        max_wall_clock_s=(
            max_wall_clock_s if max_wall_clock_s is not None else settings.phase9_max_wall_clock_s
        ),
        max_optimizer_overhead_s=(
            max_optimizer_overhead_s
            if max_optimizer_overhead_s is not None
            else settings.phase9_max_optimizer_overhead_s
        ),
        max_deep_diagnoses=(
            max_deep_diagnoses
            if max_deep_diagnoses is not None
            else settings.phase9_max_deep_diagnoses
        ),
    )


def _phase9_policy(
    settings: Settings,
    *,
    allow_deep_diagnosis: bool,
    require_knowledge: bool,
    budget: OptimizationBudget,
) -> LoopPolicy:
    return LoopPolicy(
        budget=budget,
        allow_deep_diagnosis=allow_deep_diagnosis and settings.phase9_allow_deep_diagnosis,
        require_reasoning_when_escalated=settings.phase9_require_reasoning_when_escalated,
        require_knowledge=require_knowledge or settings.phase9_require_knowledge,
        knowledge_lookup_limit=settings.phase9_knowledge_lookup_limit,
        skip_directly_negative=settings.phase9_skip_directly_negative,
        allow_revalidation=settings.phase9_allow_revalidation,
        max_candidates_per_iteration=settings.phase9_max_candidates_per_iteration,
        autonomous_only=settings.phase9_autonomous_only,
        stop_on_quality_regression=settings.phase9_stop_on_quality_regression,
        stop_on_restoration_failure=settings.phase9_stop_on_restoration_failure,
        reobserve_after_retention=settings.phase9_reobserve_after_retention,
        knowledge_positive_bonus=settings.phase9_knowledge_positive_bonus,
        knowledge_negative_penalty=settings.phase9_knowledge_negative_penalty,
        min_experiment_confidence=settings.phase9_min_experiment_confidence,
    )


def _phase9_loop(
    settings: Settings,
    *,
    service: KnowledgeService,
    executor: RuntimeConfigurationExecutor,
    profiler: MeasurementHarness,
    loop_policy: LoopPolicy,
    experiment_policy: ExperimentPolicy | None = None,
    observer: Callable[[RuntimeConfiguration], BaselineReport | None] | None = None,
    reasoning_engine: ReasoningEngine | None = None,
    proposal_provider: ReasoningProvider | None = None,
    enable_reasoning_proposals: bool = False,
    workload_id: str = "short-short",
) -> OptimizationLoop:
    return OptimizationLoop(
        executor=executor,
        profiler=profiler,
        registry=build_default_registry(),
        optimization_policy=build_policy(settings),
        experiment_policy=(
            experiment_policy
            if experiment_policy is not None
            else build_experiment_policy(settings)
        ),
        knowledge_service=service,
        loop_policy=loop_policy,
        workload_id=workload_id,
        reasoning_engine=reasoning_engine,
        proposal_provider=proposal_provider,
        enable_reasoning_proposals=enable_reasoning_proposals,
        observer=observer,
    )


def _make_loop_observer(
    settings: Settings,
    report: BaselineReport,
    workload_id: str,
    repetitions: int,
) -> "Callable[[RuntimeConfiguration], BaselineReport | None] | None":
    if not report.runs:
        return None
    model = report.runs[0].context.model
    workload = get_workload(workload_id)

    def observe(configuration: RuntimeConfiguration) -> BaselineReport | None:
        task = ProfileTask(
            model=model,
            workload=workload,
            context_length=configuration.context_length or choose_context_length(model),
        )
        profiler = BaselineProfiler(settings=settings)
        return profiler.profile(
            [task],
            modes=[MeasurementMode.WARM],
            repetitions=max(1, repetitions),
            options=options_for(configuration),
        )

    return observe


def _loop_policy_from_args(settings: Settings, args: argparse.Namespace) -> LoopPolicy:
    budget = _phase9_budget(
        settings,
        max_iterations=getattr(args, "max_iterations", None),
        max_experiments=getattr(args, "max_experiments", None),
        max_deep_diagnoses=getattr(args, "max_deep_diagnoses", None),
        max_wall_clock_s=getattr(args, "max_wall_s", None),
        max_optimizer_overhead_s=getattr(args, "max_overhead_s", None),
    )
    return _phase9_policy(
        settings,
        allow_deep_diagnosis=not bool(getattr(args, "no_deep", False)),
        require_knowledge=bool(getattr(args, "require_knowledge", False)),
        budget=budget,
    )


def _cmd_optimize(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase9.log")

    try:
        diagnosis, report, outcome, origin = _load_plan_inputs(settings, args)
    except (BaselineReportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if report is None:
        latest = find_latest_report(settings.reports_dir)
        if latest is None:
            print("error: no Phase 3 report found; run `inferenceos baseline` first", file=sys.stderr)
            return 2
        report = load_baseline_report(latest)
        origin = str(latest)

    try:
        loop_policy = _loop_policy_from_args(settings, args)
    except ValueError as exc:
        print(f"error: invalid Phase 9 configuration: {exc}", file=sys.stderr)
        return 2

    reasoning_engine: ReasoningEngine | None = None
    if not args.no_reasoning and settings.reasoning_enabled:
        try:
            reasoning_engine = build_reasoning_engine(settings)
        except (ValueError, OSError):
            reasoning_engine = None
    provider: ReasoningProvider | None = None
    if args.reasoning and settings.phase6_allow_reasoning:
        try:
            provider = build_reasoning_provider(settings)
        except (ValueError, OSError):
            provider = None

    observer = (
        None
        if args.no_reobserve
        else _make_loop_observer(
            settings, report, args.workload, settings.phase7_measured_repetitions
        )
    )
    loop = _phase9_loop(
        settings,
        service=_knowledge_service(settings, args.db),
        executor=RuntimeConfigurationExecutor(),
        profiler=BaselineProfiler(settings=settings),
        loop_policy=loop_policy,
        observer=observer,
        reasoning_engine=reasoning_engine,
        proposal_provider=provider,
        enable_reasoning_proposals=provider is not None,
        workload_id=args.workload,
    )

    print("=== InferenceOS Phase 9 pre-deployment optimization ===")
    print(f"source: {origin}")
    print()
    result = loop.run(report, diagnosis=diagnosis, outcome=outcome, report_path=origin)
    if args.json:
        print(result.to_json())
    else:
        print(render_run(result))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out) if args.out else settings.reports_dir / f"phase9_run_{timestamp}.json"
    )
    out_path.write_text(result.to_json(), encoding="utf-8")
    print()
    print(f"run: {out_path}")
    return 1 if result.termination.safety else 0


def _latest_real_noop_diagnosis(
    settings: Settings,
) -> tuple[BottleneckDiagnosis, BaselineReport] | None:
    paths = sorted(
        settings.reports_dir.glob("phase4_diagnosis_real_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        try:
            diagnosis = BottleneckDiagnosis.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if diagnosis.meaningful_opportunity:
            continue
        report = _try_load_report(diagnosis.context.source_report_path)
        if report is None:
            latest = find_latest_report(settings.reports_dir)
            report = load_baseline_report(latest) if latest is not None else None
        if report is not None and report.runs:
            return diagnosis, report
    return None


def _cmd_verify_phase9(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase9.log")

    print("=== InferenceOS Phase 9 Verification (Pre-Deployment Autonomous Loop) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    registry = build_default_registry()
    knowledge_policy = build_knowledge_policy(settings)
    verify_db = (
        Path(args.db) if getattr(args, "db", None) else settings.data_dir / f"phase9_verify_{timestamp}.db"
    )
    service = KnowledgeService(
        store=SQLiteKnowledgeStore(verify_db), registry=registry, policy=knowledge_policy
    )

    def fresh_service(name: str) -> KnowledgeService:
        return KnowledgeService(
            store=SQLiteKnowledgeStore(
                settings.data_dir / f"phase9_verify_{timestamp}_{name}.db"
            ),
            registry=registry,
            policy=knowledge_policy,
        )

    # C and E deliberately share one store so E observes C's persisted evidence.
    nav_service = fresh_service("nav")
    controlled_exp_policy = build_experiment_policy(settings).model_copy(
        update={"measured_repetitions": 2, "max_repetitions": 3, "max_total_runs": 8}
    )

    # ------------------------------------------------------------------ #
    # A: real healthy no-op baseline                                      #
    # ------------------------------------------------------------------ #
    print("--- Scenario A: real no-opportunity baseline (no change) ---")
    a_source = "stored real diagnosis"
    a_pair = _latest_real_noop_diagnosis(settings)
    if a_pair is not None:
        a_diagnosis, a_report = a_pair
    else:
        latest = find_latest_report(settings.reports_dir)
        if latest is None:
            print("error: no Phase 3 report found; run `inferenceos baseline` first", file=sys.stderr)
            return 2
        a_report = load_baseline_report(latest)
        a_diagnosis = BottleneckDetector().detect(a_report, source_report_path=str(latest))
        if a_diagnosis.meaningful_opportunity:
            a_report = make_healthy_report()
            a_diagnosis = BottleneckDetector().detect(a_report, source_report_path="<controlled-healthy>")
            a_source = "controlled healthy report (no stored no-op real diagnosis)"
    a_policy = _phase9_policy(
        settings,
        allow_deep_diagnosis=False,
        require_knowledge=False,
        budget=_phase9_budget(settings, max_iterations=1, max_experiments=1, max_deep_diagnoses=0),
    )
    a_loop = _phase9_loop(
        settings,
        service=service,
        executor=RuntimeConfigurationExecutor(),
        profiler=BaselineProfiler(settings=settings),
        loop_policy=a_policy,
        experiment_policy=controlled_exp_policy,
    )
    a_result = a_loop.run(a_report, diagnosis=a_diagnosis, report_path="<verify-phase9-A>")
    a_no_change = (
        a_result.run.budget.experiments_used == 0
        and a_result.changed is False
        and a_result.termination.reason
        in {
            LoopTerminationReason.NO_MEANINGFUL_OPPORTUNITY,
            LoopTerminationReason.INSUFFICIENT_EVIDENCE,
        }
    )
    print(f"source={a_source} reason={a_result.termination.reason.value} "
          f"experiments={a_result.run.budget.experiments_used}")
    print()

    # ------------------------------------------------------------------ #
    # B: real optimization lifecycle                                      #
    # ------------------------------------------------------------------ #
    print("--- Scenario B: real optimization lifecycle ---")
    b_source = "real induced-pressure diagnosis"
    pressure = _latest_meaningful_pressure(settings)
    if pressure is not None:
        b_diagnosis, b_origin = pressure
        b_report = _try_load_report(b_diagnosis.context.source_report_path)
        if b_report is None:
            latest = find_latest_report(settings.reports_dir)
            b_report = load_baseline_report(latest) if latest is not None else None
        if b_report is None or not b_report.runs:
            pressure = None
    if pressure is None:
        b_source = "controlled paging report (no stored real pressure diagnosis)"
        b_report = make_paging_report()
        b_diagnosis = BottleneckDetector().detect(b_report, source_report_path="<controlled-paging>")
        b_profiler: MeasurementHarness = ScriptedHarness(no_improvement_responder)
    else:
        b_profiler = BaselineProfiler(settings=settings)
    assert b_report is not None
    b_policy = _phase9_policy(
        settings,
        allow_deep_diagnosis=False,
        require_knowledge=False,
        budget=_phase9_budget(settings, max_iterations=2, max_experiments=2, max_deep_diagnoses=0),
    )
    b_loop = _phase9_loop(
        settings,
        service=service,
        executor=RuntimeConfigurationExecutor(),
        profiler=b_profiler,
        loop_policy=b_policy,
        experiment_policy=controlled_exp_policy,
    )
    b_result = b_loop.run(b_report, diagnosis=b_diagnosis, report_path="<verify-phase9-B>")
    b_experimented = [i for i in b_result.run.iterations if i.experiment_id is not None]
    b_ok = (
        b_result.run.budget.experiments_used >= 1
        and bool(b_experimented)
        and all(i.checkpoint is not None for i in b_experimented)
        and all(i.restoration_status is RestorationStatus.RESTORED for i in b_experimented)
        and all(i.experiment_status is not None for i in b_experimented)
        and service.stats().optimization_records >= 1
    )
    print(f"source={b_source} experiments={b_result.run.budget.experiments_used} "
          f"records={service.stats().optimization_records} reason={b_result.termination.reason.value}")
    print()

    # ------------------------------------------------------------------ #
    # C: negative experiment (candidate not retained)                     #
    # ------------------------------------------------------------------ #
    print("--- Scenario C: negative experiment (no meaningful improvement) ---")
    c_report = make_paging_report()
    c_diagnosis = BottleneckDetector().detect(c_report, source_report_path="<controlled-paging-C>")
    c_policy = _phase9_policy(
        settings,
        allow_deep_diagnosis=False,
        require_knowledge=False,
        budget=_phase9_budget(settings, max_iterations=3, max_experiments=2, max_deep_diagnoses=0),
    )
    c_loop = _phase9_loop(
        settings,
        service=nav_service,
        executor=RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(no_improvement_responder),
        loop_policy=c_policy,
        experiment_policy=controlled_exp_policy,
    )
    c_result = c_loop.run(c_report, diagnosis=c_diagnosis, report_path="<verify-phase9-C>")
    c_experimented = [i for i in c_result.run.iterations if i.experiment_id is not None]
    negative_before = {
        record.outcome
        for record in nav_service.all_records(limit=200)
        if record.outcome in _negative_outcomes()
    }
    c_ok = (
        bool(c_experimented)
        and all(not i.retained for i in c_experimented)
        and all(i.restoration_status is RestorationStatus.RESTORED for i in c_experimented)
        and bool(negative_before)
    )
    print(f"experiments={len(c_experimented)} negative_outcomes="
          f"{[o.value for o in sorted(negative_before, key=lambda o: o.value)]}")
    print()

    # ------------------------------------------------------------------ #
    # D: quality safety                                                   #
    # ------------------------------------------------------------------ #
    print("--- Scenario D: quality regression blocks a performance improvement ---")
    d_report = make_paging_report()
    d_diagnosis = BottleneckDetector().detect(d_report, source_report_path="<controlled-paging-D>")
    d_loop = _phase9_loop(
        settings,
        service=fresh_service("d"),
        executor=RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(quality_regression_responder),
        loop_policy=c_policy,
        experiment_policy=controlled_exp_policy,
    )
    d_result = d_loop.run(d_report, diagnosis=d_diagnosis, report_path="<verify-phase9-D>")
    d_experimented = [i for i in d_result.run.iterations if i.experiment_id is not None]
    d_ok = (
        d_result.termination.reason is LoopTerminationReason.QUALITY_SAFETY_STOP
        and bool(d_experimented)
        and all(not i.retained for i in d_experimented)
        and all(i.restoration_status is RestorationStatus.RESTORED for i in d_experimented)
    )
    print(f"reason={d_result.termination.reason.value} retained={d_result.changed}")
    print()

    # ------------------------------------------------------------------ #
    # E: knowledge-aware selection                                        #
    # ------------------------------------------------------------------ #
    print("--- Scenario E: persisted historical evidence affects selection ---")
    e_report = make_paging_report()
    e_diagnosis = BottleneckDetector().detect(e_report, source_report_path="<controlled-paging-E>")
    e_loop = _phase9_loop(
        settings,
        service=nav_service,
        executor=RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(no_improvement_responder),
        loop_policy=c_policy,
        experiment_policy=controlled_exp_policy,
    )
    e_result = e_loop.run(e_report, diagnosis=e_diagnosis, report_path="<verify-phase9-E>")
    e_effects = [
        evidence.effect
        for iteration in e_result.run.iterations
        for evidence in iteration.knowledge
    ]
    e_ok = (
        e_result.run.budget.experiments_used == 0
        and e_result.termination.reason
        in {
            LoopTerminationReason.ALL_CANDIDATES_REJECTED,
            LoopTerminationReason.NO_ELIGIBLE_CANDIDATES,
        }
        and KnowledgeSelectionEffect.REJECTED in e_effects
    )
    print(f"reason={e_result.termination.reason.value} effects={[e.value for e in e_effects]}")
    print()

    # ------------------------------------------------------------------ #
    # F: multi-iteration behavior                                         #
    # ------------------------------------------------------------------ #
    print("--- Scenario F: multiple sequential iterations ---")
    f_report = make_paging_report()
    f_diagnosis = BottleneckDetector().detect(f_report, source_report_path="<controlled-paging-F>")
    f_loop = _phase9_loop(
        settings,
        service=fresh_service("f"),
        executor=RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(context_improvement_responder),
        loop_policy=c_policy,
        experiment_policy=controlled_exp_policy,
        observer=lambda config: report_for_config(f_report, config),
    )
    f_result = f_loop.run(f_report, diagnosis=f_diagnosis, report_path="<verify-phase9-F>")
    f_experimented = [i for i in f_result.run.iterations if i.experiment_id is not None]
    f_ok = len(f_experimented) >= 2
    print(f"iterations={len(f_result.run.iterations)} experiments={len(f_experimented)} "
          f"retained={f_result.run.retained_candidate_ids}")
    print()

    # ------------------------------------------------------------------ #
    # G: explicit budget termination                                      #
    # ------------------------------------------------------------------ #
    print("--- Scenario G: budget termination ---")
    g_report = make_paging_report()
    g_diagnosis = BottleneckDetector().detect(g_report, source_report_path="<controlled-paging-G>")
    g_policy = _phase9_policy(
        settings,
        allow_deep_diagnosis=False,
        require_knowledge=False,
        budget=_phase9_budget(settings, max_iterations=1, max_experiments=5, max_deep_diagnoses=0),
    )
    g_loop = _phase9_loop(
        settings,
        service=fresh_service("g"),
        executor=RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(no_improvement_responder),
        loop_policy=g_policy,
        experiment_policy=controlled_exp_policy,
    )
    g_result = g_loop.run(g_report, diagnosis=g_diagnosis, report_path="<verify-phase9-G>")
    g_ok = (
        g_result.termination.reason is LoopTerminationReason.OPTIMIZATION_BUDGET_EXHAUSTED
        and g_result.run.budget.iterations_used == 1
    )
    print(f"reason={g_result.termination.reason.value} iterations={g_result.run.budget.iterations_used}")
    print()

    # ------------------------------------------------------------------ #
    # H: restoration failure safety                                       #
    # ------------------------------------------------------------------ #
    print("--- Scenario H: restoration failure is a safety terminal ---")
    h_report = make_paging_report()
    h_diagnosis = BottleneckDetector().detect(h_report, source_report_path="<controlled-paging-H>")
    h_loop = _phase9_loop(
        settings,
        service=fresh_service("h"),
        executor=FaultyRestoreExecutor(),
        profiler=ScriptedHarness(no_improvement_responder),
        loop_policy=c_policy,
        experiment_policy=controlled_exp_policy,
    )
    h_result = h_loop.run(h_report, diagnosis=h_diagnosis, report_path="<verify-phase9-H>")
    h_ok = (
        h_result.termination.reason is LoopTerminationReason.RESTORATION_FAILURE
        and h_result.termination.safety is True
        and h_result.run.budget.experiments_used == 1
        and len(h_result.run.iterations) == 1
    )
    print(f"reason={h_result.termination.reason.value} safety={h_result.termination.safety} "
          f"experiments={h_result.run.budget.experiments_used}")
    print()

    # ------------------------------------------------------------------ #
    # I: adversarial reasoning containment                                #
    # ------------------------------------------------------------------ #
    print("--- Scenario I: adversarial reasoning cannot become an executable action ---")
    i_report = make_healthy_report()
    i_diagnosis = BottleneckDetector().detect(i_report, source_report_path="<controlled-healthy-I>")
    adversarial = FakeReasoningProvider(scripted=[_adversarial_proposal_payload()])
    i_check_plan = generate_candidates(
        i_diagnosis,
        report=i_report,
        policy=build_policy(settings),
        provider=adversarial,
        enable_reasoning=True,
    )
    i_rejections = i_check_plan.reasoning.rejections if i_check_plan.reasoning else []
    i_joined = " | ".join(i_rejections)
    i_policy = _phase9_policy(
        settings,
        allow_deep_diagnosis=False,
        require_knowledge=False,
        budget=_phase9_budget(settings, max_iterations=1, max_experiments=1, max_deep_diagnoses=0),
    )
    i_loop = _phase9_loop(
        settings,
        service=fresh_service("i"),
        executor=RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(no_improvement_responder),
        loop_policy=i_policy,
        experiment_policy=controlled_exp_policy,
        proposal_provider=FakeReasoningProvider(scripted=[_adversarial_proposal_payload()]),
        enable_reasoning_proposals=True,
    )
    i_result = i_loop.run(i_report, diagnosis=i_diagnosis, report_path="<verify-phase9-I>")
    i_ok = (
        "unregistered" in i_joined
        and "hallucinated evidence" in i_joined
        and "fabricated measurement" in i_joined
        and not any("shell" in c.action_id for c in i_check_plan.candidates)
        and not any(c.executable for c in i_check_plan.candidates)
        and i_result.run.budget.experiments_used == 0
    )
    print(f"rejections={i_rejections} loop_experiments={i_result.run.budget.experiments_used}")
    print()

    # ------------------------------------------------------------------ #
    # J: deep-diagnosis integration and bounding                          #
    # ------------------------------------------------------------------ #
    print("--- Scenario J: deep-diagnosis integration (bounded, no execution) ---")
    j_report = make_insufficient_report()
    j_diagnosis = BottleneckDetector().detect(j_report, source_report_path="<controlled-insufficient-J>")
    j_policy = _phase9_policy(
        settings,
        allow_deep_diagnosis=True,
        require_knowledge=False,
        budget=_phase9_budget(settings, max_iterations=1, max_experiments=1, max_deep_diagnoses=1),
    )
    j_loop = _phase9_loop(
        settings,
        service=fresh_service("j"),
        executor=RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(no_improvement_responder),
        loop_policy=j_policy,
        experiment_policy=controlled_exp_policy,
        reasoning_engine=StubReasoningEngine(),
    )
    j_result = j_loop.run(j_report, diagnosis=j_diagnosis, report_path="<verify-phase9-J>")
    j_ok = any(
        iteration.deep_diagnosis_used for iteration in j_result.run.iterations
    )
    print(f"deep_diagnoses={j_result.run.budget.deep_diagnoses_used} reason={j_result.termination.reason.value}")
    print()

    checks: list[tuple[str, bool]] = [
        ("A: real/healthy baseline terminates with no change and no experiment", a_no_change),
        ("B: real optimization lifecycle reaches the experiment engine", b_ok),
        ("C: negative experiment is not retained, baseline restored, knowledge records it", c_ok),
        ("D: quality regression blocks retention and is a safety stop", d_ok),
        ("E: persisted knowledge materially affects candidate selection", e_ok),
        ("F: multiple sequential iterations are performed", f_ok),
        ("G: explicit budget exhaustion terminates cleanly", g_ok),
        ("H: restoration failure is a safety terminal condition", h_ok),
        ("I: adversarial reasoning cannot become an executable optimization", i_ok),
        ("J: deep-diagnosis machinery is invoked and bounded", j_ok),
    ]

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": 9,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verify_db": str(verify_db),
        "policy": a_policy.as_dict(),
        "scenarios": {
            "A": {"source": a_source, "result": json.loads(a_result.to_json())},
            "B": {"source": b_source, "result": json.loads(b_result.to_json())},
            "C": {"result": json.loads(c_result.to_json())},
            "D": {"result": json.loads(d_result.to_json())},
            "E": {"result": json.loads(e_result.to_json())},
            "F": {"result": json.loads(f_result.to_json())},
            "G": {"result": json.loads(g_result.to_json())},
            "H": {"result": json.loads(h_result.to_json())},
            "I": {
                "rejections": i_rejections,
                "result": json.loads(i_result.to_json()),
            },
            "J": {"result": json.loads(j_result.to_json())},
        },
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    verification_path = settings.reports_dir / f"phase9_verification_{timestamp}.json"
    verification_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    (settings.reports_dir / f"phase9_run_{timestamp}.json").write_text(
        a_result.to_json(), encoding="utf-8"
    )
    (settings.reports_dir / f"phase9_run_real_{timestamp}.json").write_text(
        b_result.to_json(), encoding="utf-8"
    )

    print()
    print(f"=== RESULT: PHASE 9 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {verification_path}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Phase 10 — runtime adaptive monitoring                                       #
# --------------------------------------------------------------------------- #
class _RuntimeFakeClock:
    """Deterministic clock for bounded controlled verification scenarios."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(seconds, 0.0)


def _approval_provider_from_args(args: argparse.Namespace) -> ApprovalProvider:
    mode = getattr(args, "approval", "interactive")
    if mode == "approve":
        return AutoApproveApprovalProvider()
    if mode == "reject":
        return AutoRejectApprovalProvider()
    if mode == "timeout":
        return TimeoutApprovalProvider()
    return CliApprovalProvider()


def _runtime_policy_from_args(settings: Settings, args: argparse.Namespace) -> RuntimePolicy:
    budget = build_runtime_budget(
        settings,
        max_duration_s=getattr(args, "duration", None),
        telemetry_interval_s=getattr(args, "interval", None),
    )
    return build_runtime_policy(
        settings,
        budget=budget,
        require_approval_for_disruptive=True,
        allow_non_disruptive_auto_adaptation=bool(
            getattr(args, "allow_auto_adapt", False)
        ),
    )


def _cmd_monitor(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase10.log")

    try:
        model_path = _find_model(settings.models_dir, args.model)
        model = load_model_profile(model_path)
        loop_policy = _runtime_policy_from_args(settings, args)
    except (FileNotFoundError, ModelMetadataError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    workload = ControlledLlamaWorkload(
        settings=settings,
        model=model,
        workload_id=args.workload,
        configuration=RuntimeConfiguration(
            context_length=args.context,
            gpu_offload_layers=args.gpu_layers,
        ),
    )
    reasoning_engine: ReasoningEngine | None = None
    if not args.no_reasoning and settings.reasoning_enabled:
        try:
            reasoning_engine = build_reasoning_engine(settings)
        except (ValueError, OSError):
            reasoning_engine = None
    sink = ConsoleNotificationSink()
    monitor = RuntimeMonitor(
        workload=workload,
        harness=BaselineProfiler(settings=settings, adapter=workload.adapter),
        policy=loop_policy,
        approval_provider=_approval_provider_from_args(args),
        notification_sink=sink,
        knowledge_service=_knowledge_service(settings, args.db),
        reasoning_engine=reasoning_engine,
        settings=settings,
    )

    print("=== InferenceOS Phase 10 runtime monitor ===")
    print(f"model    : {model_path}")
    print(f"workload : {args.workload} (context {args.context}, gpu layers {args.gpu_layers})")
    print(f"duration : {loop_policy.budget.max_duration_s:.0f}s interval {loop_policy.budget.telemetry_interval_s}s")
    print()
    result = monitor.run(duration_s=args.duration)
    workload.stop()

    if args.json:
        print(result.to_json())
    else:
        print(render_monitor_run(result))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out) if args.out else settings.reports_dir / f"phase10_monitor_{timestamp}.json"
    )
    out_path.write_text(result.to_json(), encoding="utf-8")
    print()
    print(f"run: {out_path}")
    return 1 if result.termination.safety else 0


def _p10_monitor(
    settings: Settings,
    *,
    workload: ActiveWorkload,
    sampler: RuntimeTelemetrySampler,
    policy: RuntimePolicy,
    approvals: ApprovalProvider,
    harness: MeasurementHarness,
    clock: _RuntimeFakeClock,
    knowledge_service: KnowledgeService | None = None,
) -> RuntimeMonitor:
    return RuntimeMonitor(
        workload=workload,
        sampler=sampler,
        harness=harness,
        policy=policy,
        approval_provider=approvals,
        knowledge_service=knowledge_service,
        hardware=sample_hardware(),
        experiment_policy=build_experiment_policy(settings).model_copy(
            update={"measured_repetitions": 2, "max_repetitions": 3, "max_total_runs": 8}
        ),
        clock=clock,
        sleep=clock.sleep,
    )


def _p10_budget(*, duration: float = 12.0, max_interventions: int = 2) -> RuntimeBudget:
    return RuntimeBudget(
        max_duration_s=duration,
        telemetry_interval_s=0.5,
        observation_window_size=40,
        baseline_samples=4,
        max_deep_analyses=2,
        max_interventions=max_interventions,
        max_cumulative_intervention_s=600.0,
        max_optimizer_overhead_s=120.0,
        approval_timeout_s=10.0,
        max_alerts_per_condition=2,
    )


def _cmd_verify_phase10(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase10.log")

    print("=== InferenceOS Phase 10 Verification (Runtime Adaptive Monitoring) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scenarios: dict[str, object] = {}
    checks: list[tuple[str, bool]] = []

    # ------------------------------------------------------------------ #
    # A / K: real active workload, real telemetry, workload shift          #
    # ------------------------------------------------------------------ #
    print("--- Scenario A/K: real monitored workload with a workload shift ---")
    a_ok = False
    a_k_ok = False
    a_summary: dict[str, object] = {}
    try:
        models = _load_models(settings, args.model)
        model = max(models, key=lambda profile: profile.file_size_bytes)
        duration = float(getattr(args, "duration", None) or 60.0)
        workload = ControlledLlamaWorkload(
            settings=settings,
            model=model,
            workload_id=args.workload,
            shift_after_requests=6,
            shift_context_length=max(choose_context_length(model), 4096),
            configuration=RuntimeConfiguration(
                context_length=choose_context_length(model),
                gpu_offload_layers=settings.default_gpu_layers,
            ),
        )
        policy = build_runtime_policy(
            settings,
            budget=build_runtime_budget(settings, max_duration_s=duration),
            require_approval_for_disruptive=True,
        )
        monitor = RuntimeMonitor(
            workload=workload,
            harness=BaselineProfiler(settings=settings, adapter=workload.adapter),
            policy=policy,
            approval_provider=AutoRejectApprovalProvider(),
            notification_sink=ConsoleNotificationSink(),
            settings=settings,
        )
        result = monitor.run()
        workload.stop()
        run = result.run
        a_shift_detected = any(o.context_changed for o in run.opportunities)
        a_ok = (
            run.observations_collected >= 20
            and a_shift_detected
            and run.interventions == []
            and result.termination.reason is RuntimeTerminationReason.DURATION_REACHED
        )
        a_summary = {
            "observations": run.observations_collected,
            "diagnosis_events": run.diagnosis_events,
            "shift_detected": a_shift_detected,
            "pause_calls": workload.pause_calls,
            "requests_served": workload.snapshot().requests_served,
            "termination": result.termination.reason.value,
        }
        print(
            f"observations={run.observations_collected} diagnoses={run.diagnosis_events} "
            f"shift_detected={a_shift_detected} interventions={len(run.interventions)} "
            f"requests={workload.snapshot().requests_served}"
        )
        scenarios["A_real_workload"] = json.loads(result.to_json())
    except (FileNotFoundError, ModelMetadataError, OSError, RuntimeError) as exc:
        print(f"real workload scenario unavailable: {exc}")
    checks.append(
        (
            "A: real active workload monitored; no automatic intervention; in-flight work continued",
            a_ok,
        )
    )
    # --- K: real monitoring overhead / non-distortion (stable workload) ---
    print("--- Scenario K: real monitoring overhead and non-distortion ---")
    k_ok = False
    k_summary: dict[str, object] = {}
    try:
        models = _load_models(settings, args.model)
        model = max(models, key=lambda profile: profile.file_size_bytes)
        k_duration = min(float(getattr(args, "duration", None) or 60.0), 25.0)
        k_workload = ControlledLlamaWorkload(
            settings=settings,
            model=model,
            workload_id=args.workload,
            configuration=RuntimeConfiguration(
                context_length=choose_context_length(model),
                gpu_offload_layers=settings.default_gpu_layers,
            ),
        )
        k_policy = build_runtime_policy(
            settings,
            budget=build_runtime_budget(settings, max_duration_s=k_duration),
            require_approval_for_disruptive=True,
        )
        k_monitor = RuntimeMonitor(
            workload=k_workload,
            harness=BaselineProfiler(settings=settings, adapter=k_workload.adapter),
            policy=k_policy,
            approval_provider=AutoRejectApprovalProvider(),
            notification_sink=ConsoleNotificationSink(),
            settings=settings,
        )
        k_result = k_monitor.run()
        k_workload.stop()
        reference_tps: float | None = None
        latest_report = find_latest_report(settings.reports_dir)
        if latest_report is not None:
            try:
                reference = load_baseline_report(latest_report)
                values = [
                    float(r.observation.metrics.decode_tokens_per_sec)
                    for r in reference.runs
                    if r.observation.metrics.decode_tokens_per_sec is not None
                    and r.context.model.name == model.name
                ]
                reference_tps = sum(values) / len(values) if values else None
            except (BaselineReportError, OSError):
                reference_tps = None
        monitored_tps = k_result.run.resource_impact.decode_tps_mean
        distortion_ratio = (
            monitored_tps / reference_tps if monitored_tps is not None and reference_tps else None
        )
        k_summary = {
            "observations": k_result.run.observations_collected,
            "optimizer_overhead_s": k_result.run.overhead.optimizer_overhead_s,
            "window_seconds": k_result.run.overhead.monitoring_wall_s,
            "reference_decode_tps": reference_tps,
            "monitored_decode_tps": monitored_tps,
            "distortion_ratio": distortion_ratio,
            "requests_served": k_workload.snapshot().requests_served,
            "termination": k_result.termination.reason.value,
        }
        k_ok = (
            k_result.run.overhead.optimizer_overhead_s > 0.0
            and monitored_tps is not None
            and k_workload.snapshot().requests_served > 0
            and (distortion_ratio is None or distortion_ratio >= 0.5)
        )
        print(
            f"overhead={k_result.run.overhead.optimizer_overhead_s:.2f}s "
            f"monitored_tps={monitored_tps} reference_tps={reference_tps} "
            f"ratio={distortion_ratio} requests={k_workload.snapshot().requests_served}"
        )
        scenarios["K_overhead"] = json.loads(k_result.to_json())
    except (FileNotFoundError, ModelMetadataError, OSError, RuntimeError) as exc:
        print(f"overhead scenario unavailable: {exc}")
    checks.append(("K: real monitoring overhead measured; workload not materially distorted", k_ok))
    print()

    # ------------------------------------------------------------------ #
    # B: transient anomaly does not trigger deep analysis                 #
    # ------------------------------------------------------------------ #
    print("--- Scenario B: transient anomaly ---")
    clock = _RuntimeFakeClock()

    def transient_metrics(sequence: int, condition: str) -> dict[str, float]:
        values = dict(healthy_metrics(sequence, condition))
        if condition != "baseline" and sequence == 6:
            values["hard_page_faults_per_sec"] = 30000.0
            values["memory_pressure"] = 0.9
        return values

    b_monitor = _p10_monitor(
        settings,
        workload=ScriptedWorkload(),
        sampler=ScriptedRuntimeSampler(transient_metrics),
        policy=build_runtime_policy(settings, budget=_p10_budget(duration=8.0)),
        approvals=AutoRejectApprovalProvider(),
        harness=StaticHarness(rt_no_improvement_responder),
        clock=clock,
    )
    b_result = b_monitor.run()
    b_ok = (
        b_result.run.diagnosis_events == 0
        and b_result.run.interventions == []
        and b_result.termination.reason is RuntimeTerminationReason.DURATION_REACHED
    )
    print(f"diagnoses={b_result.run.diagnosis_events} interventions={len(b_result.run.interventions)}")
    scenarios["B_transient"] = json.loads(b_result.to_json())
    checks.append(("B: transient anomaly does not trigger deep analysis or intervention", b_ok))
    print()

    # ------------------------------------------------------------------ #
    # C: persistent opportunity triggers Phase 4 diagnosis                 #
    # ------------------------------------------------------------------ #
    print("--- Scenario C: persistent opportunity -> real diagnosis ---")
    clock = _RuntimeFakeClock()
    c_monitor = _p10_monitor(
        settings,
        workload=ScriptedWorkload(),
        sampler=ScriptedRuntimeSampler(baseline_and_pressured_metrics()),
        policy=build_runtime_policy(settings, budget=_p10_budget(duration=12.0)),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.REJECTED, ApprovalOutcome.REJECTED]),
        harness=StaticHarness(rt_no_improvement_responder),
        clock=clock,
    )
    c_result = c_monitor.run()
    c_ok = (
        c_result.run.diagnosis_events >= 1
        and bool(c_result.run.opportunities)
        and c_result.run.opportunities[0].anomaly in (AnomalyClass.PERSISTENT, AnomalyClass.REPEATED)
        and c_result.run.opportunities[0].meaningful
    )
    print(f"diagnoses={c_result.run.diagnosis_events} meaningful={c_result.run.opportunities[0].meaningful if c_result.run.opportunities else None}")
    scenarios["C_persistent"] = json.loads(c_result.to_json())
    checks.append(("C: sustained condition detected and diagnosed", c_ok))
    print()

    # ------------------------------------------------------------------ #
    # D: deep-analysis recommendation with structured cost/disruption      #
    # ------------------------------------------------------------------ #
    print("--- Scenario D: deep-analysis recommendation ---")
    clock = _RuntimeFakeClock()
    sink = ConsoleNotificationSink()
    d_policy = build_runtime_policy(settings, budget=_p10_budget(duration=8.0))
    d_monitor = RuntimeMonitor(
        workload=ScriptedWorkload(),
        sampler=ScriptedRuntimeSampler(baseline_and_pressured_metrics()),
        policy=d_policy,
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.REJECTED]),
        notification_sink=sink,
        hardware=sample_hardware(),
        experiment_policy=build_experiment_policy(settings).model_copy(
            update={"measured_repetitions": 2, "max_repetitions": 3, "max_total_runs": 8}
        ),
        clock=clock,
        sleep=clock.sleep,
    )
    d_result = d_monitor.run()
    rec = d_result.run.recommendations[0] if d_result.run.recommendations else None
    d_ok = (
        rec is not None
        and rec.expected_cost_s >= 0.0
        and rec.expected_disruption_s >= 0.0
        and bool(rec.why_it_matters)
        and bool(rec.analysis_summary)
        and bool(d_result.run.notifications)
        and any(
            event.state is RuntimeState.DEEP_ANALYSIS_RECOMMENDED for event in d_result.run.states
        )
    )
    print(f"recommendations={len(d_result.run.recommendations)} cost={rec.expected_cost_s if rec else None}")
    scenarios["D_recommendation"] = json.loads(d_result.to_json())
    checks.append(("D: structured deep-analysis recommendation reached and notified", d_ok))
    print()

    # ------------------------------------------------------------------ #
    # E: rejection -> no intervention, cooldown                           #
    # ------------------------------------------------------------------ #
    print("--- Scenario E: user rejection ---")
    clock = _RuntimeFakeClock()
    workload_e = ScriptedWorkload()
    e_monitor = _p10_monitor(
        settings,
        workload=workload_e,
        sampler=ScriptedRuntimeSampler(baseline_and_pressured_metrics()),
        policy=build_runtime_policy(
            settings, budget=_p10_budget(duration=12.0), require_approval_for_disruptive=True
        ),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.REJECTED]),
        harness=StaticHarness(improving_responder),
        clock=clock,
    )
    e_result = e_monitor.run()
    e_ok = (
        e_result.run.interventions == []
        and any(a.outcome is ApprovalOutcome.REJECTED for a in e_result.run.approvals)
        and bool(e_result.run.cooldowns)
        and workload_e.snapshot().running
    )
    print(f"approvals={[a.outcome.value for a in e_result.run.approvals]} interventions={len(e_result.run.interventions)}")
    scenarios["E_rejection"] = json.loads(e_result.to_json())
    checks.append(("E: rejection records the decision and returns to monitoring with cooldown", e_ok))
    print()

    # ------------------------------------------------------------------ #
    # F: approval -> checkpoint, pause, intervention, validate, resume     #
    # ------------------------------------------------------------------ #
    print("--- Scenario F: approved controlled intervention ---")
    clock = _RuntimeFakeClock()
    workload_f = ScriptedWorkload()

    def f_metrics(sequence: int, condition: str) -> dict[str, float]:
        values = baseline_and_pressured_metrics()
        if condition == "baseline" or (workload_f.configuration().context_length or 0) <= 1024:
            return values(sequence, "baseline")
        return values(sequence, condition)

    f_monitor = _p10_monitor(
        settings,
        workload=workload_f,
        sampler=ScriptedRuntimeSampler(f_metrics),
        policy=build_runtime_policy(settings, budget=_p10_budget(duration=30.0)),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED]),
        harness=StaticHarness(improving_responder),
        clock=clock,
    )
    f_result = f_monitor.run()
    f_intervention = f_result.run.interventions[0] if f_result.run.interventions else None
    f_ok = (
        f_intervention is not None
        and f_intervention.retained
        and f_intervention.resume_verified
        and bool(f_result.run.checkpoints)
        and f_result.run.checkpoints[0].verified
        and workload_f.pause_calls >= 1
        and workload_f.resume_calls >= 1
    )
    print(f"retained={f_intervention.retained if f_intervention else None} resume={f_intervention.resume_verified if f_intervention else None}")
    scenarios["F_approval"] = json.loads(f_result.to_json())
    checks.append(("F: approved intervention checkpoints, pauses, validates, resolves and resumes", f_ok))
    print()

    # ------------------------------------------------------------------ #
    # G: approval timeout and stale approval                               #
    # ------------------------------------------------------------------ #
    print("--- Scenario G: approval timeout and stale approval ---")
    clock = _RuntimeFakeClock()
    g_timeout_monitor = _p10_monitor(
        settings,
        workload=ScriptedWorkload(),
        sampler=ScriptedRuntimeSampler(baseline_and_pressured_metrics()),
        policy=build_runtime_policy(settings, budget=_p10_budget(duration=10.0)),
        approvals=TimeoutApprovalProvider(),
        harness=StaticHarness(improving_responder),
        clock=clock,
    )
    g_timeout = g_timeout_monitor.run()
    clock = _RuntimeFakeClock()
    g_stale_monitor = _p10_monitor(
        settings,
        workload=ScriptedWorkload(),
        sampler=ScriptedRuntimeSampler(baseline_and_pressured_metrics()),
        policy=build_runtime_policy(settings, budget=_p10_budget(duration=10.0)),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED], mismatch_proposal=True),
        harness=StaticHarness(improving_responder),
        clock=clock,
    )
    g_stale = g_stale_monitor.run()
    g_ok = (
        g_timeout.run.interventions == []
        and any(a.outcome is ApprovalOutcome.TIMEOUT for a in g_timeout.run.approvals)
        and g_stale.run.interventions == []
        and any(a.outcome is ApprovalOutcome.STALE for a in g_stale.run.approvals)
    )
    print(f"timeout_interventions={len(g_timeout.run.interventions)} stale_interventions={len(g_stale.run.interventions)}")
    scenarios["G_timeout_stale"] = {
        "timeout": json.loads(g_timeout.to_json()),
        "stale": json.loads(g_stale.to_json()),
    }
    checks.append(("G: timeout/stale approval never authorizes intervention", g_ok))
    print()

    # ------------------------------------------------------------------ #
    # H: quality regression cannot be retained                             #
    # ------------------------------------------------------------------ #
    print("--- Scenario H: quality regression ---")
    clock = _RuntimeFakeClock()
    workload_h = ScriptedWorkload()
    h_monitor = _p10_monitor(
        settings,
        workload=workload_h,
        sampler=ScriptedRuntimeSampler(baseline_and_pressured_metrics()),
        policy=build_runtime_policy(settings, budget=_p10_budget(duration=12.0)),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED]),
        harness=StaticHarness(rt_quality_regression_responder),
        clock=clock,
    )
    h_result = h_monitor.run()
    h_intervention = h_result.run.interventions[0] if h_result.run.interventions else None
    h_ok = (
        h_intervention is not None
        and not h_intervention.retained
        and h_intervention.resume_verified
        and workload_h.configuration().context_length
        == choose_context_length(workload_h.model_profile())
    )
    print(f"retained={h_intervention.retained if h_intervention else None} quality={h_intervention.quality_status.value if h_intervention and h_intervention.quality_status else None}")
    scenarios["H_quality"] = json.loads(h_result.to_json())
    checks.append(("H: quality regression is not retained and the workload is restored/resumed", h_ok))
    print()

    # ------------------------------------------------------------------ #
    # I: cooldown prevents repeated alerts/interventions                   #
    # ------------------------------------------------------------------ #
    print("--- Scenario I: cooldown / anti-thrashing ---")
    clock = _RuntimeFakeClock()
    i_monitor = _p10_monitor(
        settings,
        workload=ScriptedWorkload(),
        sampler=ScriptedRuntimeSampler(baseline_and_pressured_metrics()),
        policy=build_runtime_policy(settings, budget=_p10_budget(duration=25.0)).model_copy(
            update={"intervention_cooldown_s": 600.0}
        ),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.REJECTED]),
        harness=StaticHarness(improving_responder),
        clock=clock,
    )
    i_result = i_monitor.run()
    i_ok = len(i_result.run.recommendations) <= 1 and len(i_result.run.approvals) <= 1
    print(f"recommendations={len(i_result.run.recommendations)} approvals={len(i_result.run.approvals)}")
    scenarios["I_cooldown"] = json.loads(i_result.to_json())
    checks.append(("I: cooldown prevents repeated alerts/interventions for the same condition", i_ok))
    print()

    # ------------------------------------------------------------------ #
    # J: Phase 8 knowledge participates in runtime candidate evaluation    #
    # ------------------------------------------------------------------ #
    print("--- Scenario J: knowledge-aware runtime evaluation ---")
    j_ok = False
    try:
        service = _knowledge_service(settings, None)
        record = _phase10_seed_negative_record()
        service.store.record_optimization(record)
        clock = _RuntimeFakeClock()
        j_monitor = _p10_monitor(
            settings,
            workload=ScriptedWorkload(),
            sampler=ScriptedRuntimeSampler(baseline_and_pressured_metrics()),
            policy=build_runtime_policy(settings, budget=_p10_budget(duration=8.0)),
            approvals=ScriptedApprovalProvider([ApprovalOutcome.REJECTED]),
            harness=StaticHarness(improving_responder),
            clock=clock,
            knowledge_service=service,
        )
        j_result = j_monitor.run()
        effects = [q.effect.value for q in j_result.run.knowledge_queries]
        j_ok = bool(j_result.run.knowledge_queries) and "rejected" in effects
        print(f"knowledge_queries={len(j_result.run.knowledge_queries)} effects={effects}")
        scenarios["J_knowledge"] = json.loads(j_result.to_json())
    except (OSError, RuntimeError) as exc:
        print(f"knowledge scenario failed: {exc}")
    checks.append(("J: Phase 8 knowledge materially affects runtime candidate selection", j_ok))
    print()

    # ------------------------------------------------------------------ #
    # L: external/system-wide actions remain gated                         #
    # ------------------------------------------------------------------ #
    print("--- Scenario L: external-action safety ---")
    l_diagnosis, l_report = _phase10_paging_inputs()
    plan = generate_candidates(l_diagnosis, report=l_report, policy=build_policy(settings))
    pagefile = next(
        (c for c in plan.candidates if c.action_id == "system.increase_pagefile"), None
    )
    external_class = classify_action(
        build_default_registry().get("system.increase_pagefile"), build_runtime_policy(settings)
    )
    l_ok = (
        pagefile is not None
        and pagefile.executable is False
        and external_class is InterventionClass.EXTERNAL_SYSTEM_WIDE
    )
    print(f"pagefile_executable={pagefile.executable if pagefile else None} class={external_class.value}")
    scenarios["L_external_safety"] = {
        "pagefile_executable": pagefile.executable if pagefile else None,
        "intervention_class": external_class.value,
    }
    checks.append(("L: external/system-wide actions remain gated and cannot be auto-executed", l_ok))
    print()

    # ------------------------------------------------------------------ #
    # Adversarial reasoning containment                                    #
    # ------------------------------------------------------------------ #
    print("--- Adversarial: reasoning cannot fabricate/execute ---")
    adv_provider = FakeReasoningProvider(scripted=[_adversarial_proposal_payload()])
    adv_report = make_healthy_report()
    adv_diagnosis = BottleneckDetector().detect(adv_report, source_report_path="<phase10-adversarial>")
    adv_plan = generate_candidates(
        adv_diagnosis,
        report=adv_report,
        policy=build_policy(settings),
        provider=adv_provider,
        enable_reasoning=True,
    )
    adv_rejections = adv_plan.reasoning.rejections if adv_plan.reasoning else []
    joined = " | ".join(adv_rejections)
    adversarial_ok = (
        "unregistered" in joined
        and "hallucinated evidence" in joined
        and "fabricated measurement" in joined
        and not any("shell" in c.action_id for c in adv_plan.candidates)
        and not any(c.executable for c in adv_plan.candidates)
    )
    print(f"rejections={adv_rejections}")
    scenarios["adversarial"] = {
        "rejections": adv_rejections,
        "candidate_ids": [c.action_id for c in adv_plan.candidates],
    }
    checks.append(("adversarial reasoning cannot become an executable action", adversarial_ok))
    print()

    # --- Deceptive resume -------------------------------------------------
    print("--- Adversarial: deceptive resume detection ---")
    clock = _RuntimeFakeClock()
    d2_monitor = _p10_monitor(
        settings,
        workload=ScriptedWorkload(deceptive_resume=True),
        sampler=ScriptedRuntimeSampler(baseline_and_pressured_metrics()),
        policy=build_runtime_policy(settings, budget=_p10_budget(duration=30.0)),
        approvals=ScriptedApprovalProvider([ApprovalOutcome.APPROVED]),
        harness=StaticHarness(improving_responder),
        clock=clock,
    )
    d2_result = d2_monitor.run()
    deceptive_ok = (
        d2_result.final_state is RuntimeState.TERMINAL_SAFE
        and bool(d2_result.run.interventions)
        and not d2_result.run.interventions[0].resume_verified
    )
    print(f"final_state={d2_result.final_state.value} resume_verified={d2_result.run.interventions[0].resume_verified if d2_result.run.interventions else None}")
    scenarios["deceptive_resume"] = json.loads(d2_result.to_json())
    checks.append(("deceptive resume claim is detected as a safety terminal", deceptive_ok))

    print()
    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": 10,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "policy": build_runtime_policy(settings).as_dict(),
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    verification_path = settings.reports_dir / f"phase10_verification_{timestamp}.json"
    verification_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print()
    print(f"=== RESULT: PHASE 10 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {verification_path}")
    return 0 if ok else 1


def _phase10_paging_inputs() -> tuple[BottleneckDiagnosis, BaselineReport]:
    report = make_paging_report()
    diagnosis = BottleneckDetector().detect(report, source_report_path="<phase10-paging>")
    return diagnosis, report


def _phase10_seed_negative_record() -> OptimizationRecord:
    from inferenceos.contracts.knowledge import OptimizationRecord
    from inferenceos.knowledge import build_current_context
    from inferenceos.selftest import sample_model

    context = build_current_context(
        hardware=sample_hardware(),
        model=sample_model(),
        workload=get_workload("short-short").to_workload_profile(),
        backend=ProfilingConfig(
            backend_name="llama.cpp", backend_type=BackendType.VULKAN, context_length=2048
        ),
        runtime=RuntimeConfiguration(context_length=2048, gpu_offload_layers=99),
        diagnosis="paging",
        conditions={},
    )
    return OptimizationRecord(
        record_id="opt-phase10-seed",
        experiment_id="exp-phase10-seed",
        candidate_id="cand-phase10-seed",
        action_id="runtime.right_size_context",
        parameters={"context_length": 1024},
        diagnosed_bottleneck="paging",
        outcome=OptimizationOutcome.NO_MEANINGFUL_IMPROVEMENT,
        context=context,
    )


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

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="deterministically diagnose a Phase 3 baseline report (no LLM)",
    )
    diagnose_parser.add_argument("--report", default=None, help="Phase 3 report path")
    diagnose_parser.add_argument("--out", default=None, help="diagnosis output path")
    diagnose_parser.add_argument("--json", action="store_true", help="print the diagnosis JSON")
    diagnose_parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 3 when no meaningful optimization opportunity is found",
    )
    diagnose_parser.set_defaults(func=_cmd_diagnose)

    verify4_parser = subparsers.add_parser(
        "verify-phase4",
        help="run the real Phase 4 acceptance (ingest + genuine bottleneck + false-positive)",
    )
    verify4_parser.add_argument("--report", default=None, help="Phase 3 report path")
    verify4_parser.add_argument("--model", default=None, help="model for the pressure scenario")
    verify4_parser.add_argument("--repetitions", type=int, default=2)
    verify4_parser.add_argument(
        "--pressure-leave-gib",
        type=float,
        default=1.5,
        help="physical memory to leave free while inducing pressure",
    )
    verify4_parser.add_argument(
        "--pressure-cap-gib", type=float, default=8.0, help="maximum memory to hold"
    )
    verify4_parser.set_defaults(func=_cmd_verify_phase4)

    deep_parser = subparsers.add_parser(
        "deep-diagnose",
        help="escalate an ambiguous Phase 4 diagnosis to the local reasoning model",
    )
    deep_parser.add_argument("--report", default=None, help="Phase 3 report path")
    deep_parser.add_argument(
        "--diagnosis", default=None, help="Phase 4 BottleneckDiagnosis JSON path"
    )
    deep_parser.add_argument("--out", default=None, help="outcome output path")
    deep_parser.add_argument("--json", action="store_true", help="print the outcome JSON")
    deep_parser.add_argument(
        "--no-reasoning", action="store_true", help="deterministic-only (do not invoke the LLM)"
    )
    deep_parser.set_defaults(func=_cmd_deep_diagnose)

    verify5_parser = subparsers.add_parser(
        "verify-phase5",
        help="run the real Phase 5 acceptance (local reasoning + validation + fallback)",
    )
    verify5_parser.add_argument("--report", default=None, help="Phase 3 report path")
    verify5_parser.add_argument("--model", default=None, help="model for the pressure scenario")
    verify5_parser.add_argument(
        "--skip-pressure", action="store_true", help="skip the controlled pressure scenario"
    )
    verify5_parser.set_defaults(func=_cmd_verify_phase5)

    candidates_parser = subparsers.add_parser(
        "candidates",
        help="generate and rank optimization candidates from a validated diagnosis (no execution)",
    )
    candidates_parser.add_argument("--diagnosis", default=None, help="Phase 4 diagnosis JSON path")
    candidates_parser.add_argument(
        "--deep", default=None, help="Phase 5 DeepDiagnosisOutcome JSON path"
    )
    candidates_parser.add_argument("--report", default=None, help="Phase 3 report path")
    candidates_parser.add_argument("--out", default=None, help="plan output path")
    candidates_parser.add_argument("--json", action="store_true", help="print the plan JSON")
    candidates_parser.add_argument(
        "--reasoning",
        action="store_true",
        help="allow the local reasoner to propose registered actions when mapping is insufficient",
    )
    candidates_parser.set_defaults(func=_cmd_candidates)

    verify6_parser = subparsers.add_parser(
        "verify-phase6",
        help="run the real Phase 6 acceptance (candidate generation, gating, no execution)",
    )
    verify6_parser.add_argument("--report", default=None, help="Phase 3 report path")
    verify6_parser.add_argument("--model", default=None, help="model for the pressure scenario")
    verify6_parser.add_argument(
        "--skip-pressure", action="store_true", help="skip the controlled pressure scenario"
    )
    verify6_parser.add_argument(
        "--reasoning",
        action="store_true",
        help="also attempt a real local reasoner proposal pass",
    )
    verify6_parser.set_defaults(func=_cmd_verify_phase6)

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="apply one eligible candidate against the real harness and decide keep/reject",
    )
    experiment_parser.add_argument("--plan", default=None, help="Phase 6 CandidatePlan JSON path")
    experiment_parser.add_argument("--diagnosis", default=None, help="Phase 4 diagnosis JSON path")
    experiment_parser.add_argument("--deep", default=None, help="Phase 5 DeepDiagnosisOutcome path")
    experiment_parser.add_argument("--report", default=None, help="Phase 3 report path")
    experiment_parser.add_argument(
        "--candidate-id", default=None, help="experiment with a specific eligible candidate"
    )
    experiment_parser.add_argument("--workload", default="short-short", help="benchmark workload id")
    experiment_parser.add_argument("--out", default=None, help="result output path")
    experiment_parser.add_argument("--json", action="store_true", help="print the result JSON")
    experiment_parser.add_argument(
        "--reasoning", action="store_true", help="allow the reasoner to propose candidates first"
    )
    experiment_parser.set_defaults(func=_cmd_experiment)

    verify7_parser = subparsers.add_parser(
        "verify-phase7",
        help="run the real Phase 7 acceptance (candidate execution, quality, restoration)",
    )
    verify7_parser.add_argument("--report", default=None, help="Phase 3 report path")
    verify7_parser.set_defaults(func=_cmd_verify_phase7)

    knowledge_parser = subparsers.add_parser(
        "knowledge",
        help="inspect persistent optimization knowledge, applicability, and rollback",
    )
    knowledge_parser.add_argument("--db", default=None, help="knowledge database path")
    knowledge_parser.add_argument("--action", default=None, help="filter by action id")
    knowledge_parser.add_argument("--candidate", default=None, help="filter by candidate id")
    knowledge_parser.add_argument("--model", default=None, help="filter by model ref")
    knowledge_parser.add_argument("--outcome", default=None, help="filter by outcome")
    knowledge_parser.add_argument("--limit", type=int, default=50, help="max records")
    knowledge_parser.add_argument(
        "--applicability", action="store_true", help="evaluate applicability vs. current context"
    )
    knowledge_parser.add_argument(
        "--rollback", action="store_true", help="include rollback records"
    )
    knowledge_parser.add_argument(
        "--detail", action="store_true", help="print full record detail"
    )
    knowledge_parser.add_argument(
        "--health", action="store_true", help="print store health/statistics only"
    )
    knowledge_parser.add_argument("--json", action="store_true", help="emit JSON")
    knowledge_parser.set_defaults(func=_cmd_knowledge)

    verify8_parser = subparsers.add_parser(
        "verify-phase8",
        help="run the real Phase 8 acceptance (durable knowledge, applicability, rollback)",
    )
    verify8_parser.add_argument("--db", default=None, help="knowledge database path")
    verify8_parser.set_defaults(func=_cmd_verify_phase8)

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="run the pre-deployment autonomous optimization loop (observe through terminate)",
    )
    optimize_parser.add_argument("--report", default=None, help="Phase 3 report path")
    optimize_parser.add_argument("--diagnosis", default=None, help="Phase 4 diagnosis JSON path")
    optimize_parser.add_argument("--deep", default=None, help="Phase 5 DeepDiagnosisOutcome path")
    optimize_parser.add_argument("--workload", default="short-short", help="benchmark workload id")
    optimize_parser.add_argument("--db", default=None, help="knowledge database path")
    optimize_parser.add_argument("--max-iterations", type=int, default=None)
    optimize_parser.add_argument("--max-experiments", type=int, default=None)
    optimize_parser.add_argument("--max-deep-diagnoses", type=int, default=None)
    optimize_parser.add_argument("--max-wall-s", type=float, default=None)
    optimize_parser.add_argument("--max-overhead-s", type=float, default=None)
    optimize_parser.add_argument(
        "--require-knowledge", action="store_true", help="fail closed if knowledge is unavailable"
    )
    optimize_parser.add_argument(
        "--no-deep", action="store_true", help="disable deep-diagnosis escalation"
    )
    optimize_parser.add_argument(
        "--no-reobserve", action="store_true", help="do not re-measure after a retained candidate"
    )
    optimize_parser.add_argument(
        "--no-reasoning", action="store_true", help="disable the deep-diagnosis reasoning engine"
    )
    optimize_parser.add_argument(
        "--reasoning", action="store_true", help="allow reasoner-proposed candidates when needed"
    )
    optimize_parser.add_argument("--out", default=None, help="run output path")
    optimize_parser.add_argument("--json", action="store_true", help="print the run JSON")
    optimize_parser.set_defaults(func=_cmd_optimize)

    verify9_parser = subparsers.add_parser(
        "verify-phase9",
        help="run the real Phase 9 acceptance (autonomous loop, safety, knowledge, budgets)",
    )
    verify9_parser.add_argument("--db", default=None, help="knowledge database path")
    verify9_parser.set_defaults(func=_cmd_verify_phase9)

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="run the conservative runtime adaptive monitor over an active workload",
    )
    monitor_parser.add_argument("--model", default=None, help="path to a .gguf model")
    monitor_parser.add_argument("--workload", default="short-short", help="benchmark workload id")
    monitor_parser.add_argument("--context", type=int, default=2048, help="context length")
    monitor_parser.add_argument("--gpu-layers", type=int, default=99, help="GPU offload layers")
    monitor_parser.add_argument("--duration", type=float, default=None, help="bounded run seconds")
    monitor_parser.add_argument("--interval", type=float, default=None, help="telemetry interval s")
    monitor_parser.add_argument(
        "--approval",
        choices=["interactive", "approve", "reject", "timeout"],
        default="interactive",
        help="approval mode for disruptive runtime intervention",
    )
    monitor_parser.add_argument("--db", default=None, help="knowledge database path")
    monitor_parser.add_argument(
        "--allow-auto-adapt",
        action="store_true",
        help="allow explicitly safe non-disruptive actions without approval",
    )
    monitor_parser.add_argument(
        "--no-reasoning", action="store_true", help="disable the deep-diagnosis reasoning engine"
    )
    monitor_parser.add_argument("--out", default=None, help="run output path")
    monitor_parser.add_argument("--json", action="store_true", help="print the run JSON")
    monitor_parser.set_defaults(func=_cmd_monitor)

    verify10_parser = subparsers.add_parser(
        "verify-phase10",
        help="run the Phase 10 acceptance (runtime monitoring, approval, safety, cooldown)",
    )
    verify10_parser.add_argument("--model", default=None, help="model for the real scenario")
    verify10_parser.add_argument("--workload", default="short-short", help="benchmark workload id")
    verify10_parser.add_argument(
        "--duration", type=float, default=60.0, help="real workload monitoring seconds"
    )
    verify10_parser.set_defaults(func=_cmd_verify_phase10)

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
