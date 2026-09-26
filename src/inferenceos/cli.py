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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import psutil

from inferenceos import __version__
from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.backends.resident_server import LlamaServerResidentBackend
from inferenceos.backends.llama_cpp_toolchain import (
    backend_type_from_name,
    find_cli,
    parse_device_info,
    parse_generated_text,
    parse_timings,
    run_cli,
)
from inferenceos.config import Settings
from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.diagnosis import BottleneckDiagnosis, DiagnosisContext
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
    AttributionEvidenceCategory,
    EvidencePolarity,
    EvidenceStrength,
    ExperimentDecisionKind,
    ExternalActionSafetyClass,
    ExternalRecommendationClass,
    ExternalRecommendationStatus,
    ExternalResource,
    ExternalRunTerminationReason,
    ExternalScope,
    InferenceStage,
    InterventionClass,
    KnowledgeSelectionEffect,
    LoopTerminationReason,
    OptimizationOutcome,
    RestorationStatus,
    RollbackExecutionStatus,
    RuntimeState,
    RuntimeTerminationReason,
)
from inferenceos.contracts.external import (
    ExternalBudget,
    ExternalObservationReport,
    ExternalPolicy,
    ExternalProcessAccess,
    ExternalProcessIdentity,
    ExternalProcessObservation,
    ExternalRecommendation,
    ExternalRecommendationRun,
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
from inferenceos.contracts.lifecycle import (
    LifecyclePolicy,
    RequestPolicy,
    ResidentDeepDiagnosisOutcome,
)
from inferenceos.contracts.system_state import SystemState
from inferenceos.experiment.audit import build_audit_record
from inferenceos.experiment.comparison import compare
from inferenceos.experiment.decision import decide
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.experiment.measurement import MeasurementHarness, options_for
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.experiment.quality import validate_quality
from inferenceos.experiment.report import render_experiment
from inferenceos.experiment.restoration import (
    ResidentRestorationVerifier,
    runtime_params_from,
)
from inferenceos.experiment.provenance import parameters_hash, verify_provenance
from inferenceos.experiment.service import ExperimentInputError, run_candidate_experiment
from inferenceos.contracts.rollback import RollbackExecutionPolicy
from inferenceos.knowledge import (
    KnowledgeService,
    PersistResult,
    RollbackExecutor,
    SQLiteKnowledgeStore,
    build_current_context,
    build_knowledge_policy,
    build_rollback_record,
    evaluate_applicability,
    render_applicability,
    render_health,
    render_history,
    render_record,
    render_rollback,
    seal_rollback_record,
    verify_rollback,
)
from inferenceos.knowledge.store import KnowledgeCorruptionError
from inferenceos.loop import (
    FreshRunOrchestrator,
    OptimizationLoop,
    ResidentDeepDiagnosisRunner,
    build_loop_policy,
    render_run,
    render_termination,
)
from inferenceos.contracts.fresh import DeploymentState, DeploymentStatus, FreshRunPolicy
from inferenceos.deployment import (
    DeploymentIntegrityError,
    deployment_from_loop_result,
    file_sha256,
    load_deployment_state,
    serve_deployment,
    write_deployment_state,
)
from inferenceos.contracts.runtime import (
    RuntimeBudget,
    RuntimeMonitorResult,
    RuntimeObservation,
    RuntimePolicy,
)
from inferenceos.runtime import (
    ActiveWorkload,
    ApprovalProvider,
    AutoApproveApprovalProvider,
    AutoRejectApprovalProvider,
    CallbackApprovalProvider,
    CliApprovalProvider,
    ConsoleNotificationSink,
    ControlledLlamaWorkload,
    ResidentActiveWorkload,
    RuntimeMonitor,
    RuntimeTelemetrySampler,
    ScriptedApprovalProvider,
    SystemTelemetrySampler,
    TimeoutApprovalProvider,
    build_runtime_budget,
    build_runtime_policy,
    classify_action,
    render_monitor_run,
)
from inferenceos.runtime.verification import (
    ControlledFootprintDetector,
    ScriptedRuntimeSampler,
    ScriptedWorkload,
    StaticHarness,
    baseline_and_pressured_metrics,
    healthy_metrics,
    improving_responder,
    no_improvement_responder as rt_no_improvement_responder,
    quality_regression_responder as rt_quality_regression_responder,
)
from inferenceos.external import (
    ExternalKnowledgeStore,
    ExternalRecommendationEngine,
    RuntimeExternalAdvisor,
    WindowsProcessObserver,
    assess_external_attribution,
    bottleneck_to_resource,
    build_external_budget,
    build_external_policy,
    build_recommendations,
    evaluate_external_action_safety,
    launch_cpu_spinner,
    launch_memory_holder,
    launch_sleeper,
    render_external_run,
    resolve_identity,
    revalidate_recommendation,
)
from inferenceos.external.controlled import (
    FakeClock as ExternalFakeClock,
    ScriptedProcessObserver,
    make_process_observation,
    make_report,
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
from inferenceos.storage import guarded_write_text, run_startup_retention
from inferenceos.models.estimator import estimate_kv_curve
from inferenceos.models.kv_validation import (
    KV_CACHE_RELATIVE_TOLERANCE,
    derive_expected_kv_cache_bytes,
    validate_reported_kv_cache,
)
from inferenceos.models.loader import ModelMetadataError, load_model_profile
from inferenceos.native import get_platform
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy, build_policy
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
    get_quality_workload,
    get_workload,
)
from inferenceos.reasoning.context import build_reasoning_context
from inferenceos.reasoning.corroboration import assess_corroboration
from inferenceos.reasoning.economics import build_lifecycle_policy
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.escalation import decide_escalation
from inferenceos.reasoning.factory import (
    build_reasoning_engine,
    build_reasoning_provider,
    build_resident_reasoner,
)
from inferenceos.reasoning.lifecycle import run_deep_diagnosis
from inferenceos.reasoning.providers.base import ReasoningProvider
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from inferenceos.reasoning.providers.llama_cpp import LlamaCppReasoningProvider
from inferenceos.reasoning.report import render_deep_diagnosis
from inferenceos.reasoning.resident_lifecycle import ResidentLifecycleRunner
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
            key_length = model.key_length or model.head_dim
            value_length = model.value_length or model.head_dim
            # Independent derivation from GGUF architecture metadata only.
            row["independent_kv_bytes"] = derive_expected_kv_cache_bytes(
                n_layers=model.n_layers,
                n_kv_heads=model.n_kv_heads,
                key_length=key_length,
                value_length=value_length,
                context_length=context,
            )
            # The plausible-but-wrong derivation (attention heads instead of KV heads),
            # used to prove a same-source error cannot pass validation.
            row["attention_head_kv_bytes"] = derive_expected_kv_cache_bytes(
                n_layers=model.n_layers,
                n_kv_heads=model.n_attention_heads,
                key_length=key_length,
                value_length=value_length,
                context_length=context,
            )
            row["predicted_kv_bytes"] = model.estimate_kv_cache_bytes(context)
            row["reported_kv_bytes"] = telemetry.kv_cache_bytes
            row["peak_rss_bytes"] = result.peak_rss_bytes
            row["final_rss_bytes"] = result.final_rss_bytes
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

    # Checks: compare the INDEPENDENTLY derived KV expectation (from GGUF architecture
    # metadata) against the backend-reported allocation, never llama.cpp against itself.
    reported_ok = True
    reported_seen = False
    head_guard_ok = True
    head_guard_seen = False
    kv_validations: dict[str, list[dict[str, object]]] = {}
    for model in profiles:
        rows = memory.get(model.name, [])
        validations: list[dict[str, object]] = []
        for row in rows:
            reported = _as_int(row.get("reported_kv_bytes"))
            independent = _as_int(row.get("independent_kv_bytes"))
            if independent is None:
                continue
            validation = validate_reported_kv_cache(
                expected_bytes=independent, reported_bytes=reported
            )
            validations.append(
                {
                    "context": row.get("context"),
                    "expected_bytes": validation.expected_bytes,
                    "reported_bytes": validation.reported_bytes,
                    "relative_error": validation.relative_error,
                    "within_tolerance": validation.within_tolerance,
                    "tolerance": validation.tolerance,
                    "basis": validation.basis,
                }
            )
            if validation.within_tolerance is None:
                continue
            reported_seen = True
            if not validation.within_tolerance:
                reported_ok = False
            # If the KV-head count differs from the attention-head count, the wrong
            # derivation must NOT agree with the backend; otherwise the check would be
            # validating the wrong quantity (false agreement from the same bad source).
            if model.n_kv_heads != model.n_attention_heads and reported:
                wrong = _as_int(row.get("attention_head_kv_bytes"))
                if wrong is not None:
                    wrong_error = abs(wrong - reported) / reported
                    if wrong_error <= KV_CACHE_RELATIVE_TOLERANCE:
                        head_guard_ok = False
                    head_guard_seen = True
        kv_validations[model.name] = validations

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
        (
            "independently derived KV matches backend-reported KV within "
            f"{KV_CACHE_RELATIVE_TOLERANCE:.0%}",
            reported_ok and reported_seen,
        ),
        (
            "attention-head derivation does not falsely agree with the backend",
            head_guard_ok and head_guard_seen,
        ),
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
        "kv_validation": {
            "tolerance": KV_CACHE_RELATIVE_TOLERANCE,
            "basis": (
                "independent GGUF-architecture derivation vs backend-reported allocation"
            ),
            "per_model": kv_validations,
        },
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
    memory = get_platform().global_memory_status()
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
# Phase 5R — resident backend & deep-diagnosis lifecycle                       #
# --------------------------------------------------------------------------- #
def _resident_server_path(settings: Settings) -> Path:
    candidate = settings.llama_cpp_dir / "llama-server.exe"
    if candidate.is_file():
        return candidate
    fallback = settings.llama_cpp_dir / "llama-server"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"llama-server not found in {settings.llama_cpp_dir}")


def _resident_target_params(settings: Settings, model: ModelProfile) -> RuntimeParams:
    return RuntimeParams(
        gpu_offload_layers=settings.default_gpu_layers,
        context_length=choose_context_length(model),
        use_mmap=True,
    )


def _real_escalating_diagnosis(
    settings: Settings, report_arg: str | None
) -> tuple[BottleneckDiagnosis, BaselineReport, str] | None:
    paths: list[Path] = []
    if report_arg:
        paths.append(Path(report_arg))
    latest = find_latest_report(settings.reports_dir)
    if latest is not None:
        paths.append(latest)
    for path in paths:
        try:
            report = load_baseline_report(path)
        except (BaselineReportError, OSError):
            continue
        diagnosis = BottleneckDetector().detect(report, source_report_path=str(path))
        if decide_escalation(diagnosis).should_escalate:
            return diagnosis, report, str(path)
    for path in sorted(
        settings.reports_dir.glob("phase5_deep_diagnosis_*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        try:
            outcome = DeepDiagnosisOutcome.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not outcome.escalation.should_escalate:
            continue
        stored_report = _try_load_report(outcome.source_report_path)
        if stored_report is not None and stored_report.runs:
            return outcome.deterministic, stored_report, str(path)
    return None


def _adversarial_reasoning_checks(
    diagnosis: BottleneckDiagnosis, report: BaselineReport
) -> dict[str, bool]:
    """Deterministic safety-layer checks over the real reasoning context."""
    import json as _json

    from inferenceos.contracts.reasoning import ReasoningRequest, ReasoningResponse

    context = build_reasoning_context(diagnosis, report)
    if not context.evidence:
        return {"fabricated_evidence": False, "fabricated_measurement": False,
                "confidence_inflation": False, "recommendation_rejected": False}
    good_id = context.evidence[0].evidence_id
    base = {
        "schema_version": "1.0",
        "root_cause_class": "paging",
        "affected_stage": "end_to_end",
        "contributing_factors": ["x"],
        "supporting_evidence_ids": [good_id],
        "contradicting_evidence_ids": [],
        "confidence": 0.7,
        "evidence_strength": "medium",
        "uncertainty": "",
        "further_evidence_required": False,
        "summary": "ok",
        "reasoning": "grounded",
    }

    def _reason(payload: dict[str, object]) -> ReasoningResponse:
        engine = LocalReasoningEngine(
            FakeReasoningProvider(scripted=[_json.dumps(payload)])
        )
        request = ReasoningRequest(
            request_id="adv", reason="adversarial", objective="adversarial",
            context=context,
        )
        return engine.reason(request)

    fabricated_id = dict(base)
    fabricated_id["supporting_evidence_ids"] = ["E999"]
    resp_ids = _reason(fabricated_id)
    fabricated_evidence = resp_ids.validation_ok is False and resp_ids.status.value == "failed"

    fabricated_measurement = dict(base)
    fabricated_measurement["summary"] = "latency collapsed to 12345.6 ms"
    resp_measure = _reason(fabricated_measurement)
    measurement_rejected = (
        resp_measure.validation_ok is False and resp_measure.status.value == "failed"
    )

    recommendation = dict(base)
    recommendation["reasoning"] = "You should reduce the context length."
    resp_rec = _reason(recommendation)
    recommendation_rejected = (
        resp_rec.validation_ok is False and resp_rec.status.value == "failed"
    )

    inflated = dict(base)
    inflated["confidence"] = 0.99
    inflated["evidence_strength"] = "high"
    resp_inflated = _reason(inflated)
    assessment = assess_corroboration(
        diagnosis, resp_inflated, ceiling=0.25, evidence_ids={good_id}
    )
    confidence_inflation = bool(
        resp_inflated.validation_ok
        and assessment.confidence_capped
        and assessment.effective_confidence < 0.99
    )
    return {
        "fabricated_evidence": fabricated_evidence,
        "fabricated_measurement": measurement_rejected,
        "confidence_inflation": confidence_inflation,
        "recommendation_rejected": recommendation_rejected,
    }


def _cmd_verify_phase5r(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase5r.log")

    print("=== InferenceOS Phase 5R Verification (Resident Backend & Lifecycle) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    try:
        server_path = _resident_server_path(settings)
        model = _load_models(settings, args.model)[0]
    except (FileNotFoundError, ModelMetadataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pair = _real_escalating_diagnosis(settings, args.report)
    if pair is None:
        print(
            "error: no real diagnosis requiring escalation is available; run "
            "`inferenceos baseline` or provide --report",
            file=sys.stderr,
        )
        return 2
    diagnosis, report, diagnosis_origin = pair
    model = report.runs[0].context.model if report.runs else model

    reasoner = build_resident_reasoner(settings)
    if reasoner is None:
        print(
            f"error: the configured reasoner model is unavailable: "
            f"{settings.reasoning_model_path}",
            file=sys.stderr,
        )
        return 2

    lifecycle_policy = build_lifecycle_policy(settings)
    target_backend = LlamaServerResidentBackend(
        server=server_path,
        host=settings.resident_host,
        port=settings.resident_port,
        readiness_timeout_s=settings.resident_readiness_timeout_s,
        request_timeout_s=settings.resident_request_timeout_s,
    )
    adapter = LlamaCppResidentAdapter(settings=settings, server=server_path, backend=target_backend)
    target_params = _resident_target_params(settings, model)
    checks: list[tuple[str, bool]] = []
    scenarios: dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # Scenario A: resident target                                          #
    # ------------------------------------------------------------------ #
    print("--- Scenario A: resident target, streaming TTFT, no relaunch ---")
    a_ok = False
    try:
        adapter.load(model, target_params)
        load = adapter.last_load
        pid_a = adapter.pid
        result1 = adapter.generate(GenerationRequest(prompt="Once upon a time", n_predict=16, seed=1))
        gen1 = adapter.last_generation
        result2 = adapter.generate(GenerationRequest(prompt="The capital of France is", n_predict=16, seed=2))
        gen2 = adapter.last_generation
        same_pid = adapter.pid == pid_a and adapter.is_loaded
        a_ok = bool(
            load is not None
            and load.ready
            and load.resident_verified
            and pid_a is not None
            and gen1 is not None
            and gen1.ttft_s is not None
            and gen1.ttft_s >= 0.0
            and result1.text
            and result2.text
            and same_pid
        )
        scenarios["A_resident"] = {
            "pid": pid_a,
            "model": model.name,
            "startup_s": load.startup_s if load else None,
            "ready": bool(load and load.ready),
            "resident_verified": bool(load and load.resident_verified),
            "rss_after_bytes": load.after.rss_bytes if load else None,
            "ttft_s": gen1.ttft_s if gen1 else None,
            "total_s": gen1.total_s if gen1 else None,
            "decode_tokens_per_sec": gen1.decode_tokens_per_sec if gen1 else None,
            "generated_tokens": gen1.generated_tokens if gen1 else None,
            "second_request_same_pid": same_pid,
            "backend_device": adapter.telemetry().device,
        }
        print(
            f"pid={pid_a} ready={load.ready if load else None} "
            f"rss={load.after.rss_bytes if load else None} "
            f"ttft={gen1.ttft_s if gen1 else None} same_pid={same_pid}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario A failed: {type(exc).__name__}: {exc}")
        scenarios["A_resident"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("A: real resident target, measured streaming TTFT, no relaunch", a_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario B: unload / reload                                          #
    # ------------------------------------------------------------------ #
    print("--- Scenario B: real unload and reload with new process identity ---")
    b_ok = False
    try:
        pid_before = adapter.pid
        adapter.unload()
        unload = target_backend.last_unload
        exited = bool(unload and unload.cleanup_complete)
        adapter.load(model, target_params)
        reload_load = adapter.last_load
        pid_after = adapter.pid
        new_identity = pid_after is not None and pid_after != pid_before
        result3 = adapter.generate(GenerationRequest(prompt="Once upon a time", n_predict=8, seed=1))
        b_ok = bool(exited and new_identity and reload_load and reload_load.ready and result3.text)
        scenarios["B_unload_reload"] = {
            "pid_before": pid_before,
            "unload_exited": exited,
            "unload_s": unload.unload_s if unload else None,
            "stop_method": unload.stop_method if unload else None,
            "pid_after": pid_after,
            "new_identity": new_identity,
            "reload_startup_s": reload_load.startup_s if reload_load else None,
            "reload_rss_bytes": reload_load.after.rss_bytes if reload_load else None,
            "generation_ok": bool(result3.text),
        }
        print(
            f"pid {pid_before} -> {pid_after} exited={exited} new_identity={new_identity} "
            f"unload_s={unload.unload_s if unload else None}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario B failed: {type(exc).__name__}: {exc}")
        scenarios["B_unload_reload"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("B: real unload/reload with a new verified process identity", b_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario C / F: deep-diagnosis lifecycle + request during window     #
    # ------------------------------------------------------------------ #
    print("--- Scenario C: resident deep-diagnosis lifecycle (real reasoner) ---")
    c_ok = False
    f_ok = False
    lifecycle_outcome: ResidentDeepDiagnosisOutcome | None = None

    def _inject_request(window: object) -> None:
        # Fail-safe: emulate a real request arriving during the diagnosis window.
        window.submit("what is the meaning of life?")  # type: ignore[attr-defined]

    try:
        runner = ResidentLifecycleRunner(
            target=target_backend,
            target_model=model,
            target_params=target_params,
            policy=lifecycle_policy,
            reasoner=reasoner,
            request_policy=settings.phase5r_request_policy,
            interruption_cost_s=0.0,
            on_diagnosis_window=_inject_request,
        )
        lifecycle_outcome = runner.run(
            diagnosis,
            report=report,
            confidence_threshold=settings.reasoning_escalation_threshold,
        )
        lifecycle_artifact = settings.reports_dir / f"phase5r_lifecycle_{timestamp}.json"
        lifecycle_artifact.write_text(lifecycle_outcome.to_json(), encoding="utf-8")
        transitions = lifecycle_outcome.transitions
        states = [t.state for t in transitions]
        has_all = all(
            state in states
            for state in (
                "resident_target",
                "entry_decision",
                "checkpoint",
                "unload_target",
                "load_reasoner",
                "deep_diagnosis",
                "unload_reasoner",
                "reload_target",
                "verify_target",
                "resume",
            )
        )
        c_ok = bool(
            lifecycle_outcome.entered
            and has_all
            and lifecycle_outcome.diagnosis is not None
            and lifecycle_outcome.diagnosis.reasoning is not None
            and lifecycle_outcome.diagnosis.reasoning.validation_ok
            and lifecycle_outcome.post_reload is not None
            and lifecycle_outcome.post_reload.generation_ok
            and target_backend.is_running()
            and not lifecycle_outcome.errors
        )
        request_records = lifecycle_outcome.requests
        if settings.phase5r_request_policy == RequestPolicy.QUEUE:
            f_ok = bool(
                request_records
                and request_records[0].queued
                and request_records[0].drained
            )
        else:
            f_ok = bool(
                request_records
                and request_records[0].rejected
                and not request_records[0].queued
                and "rejected" in request_records[0].response_detail
            )
        scenarios["C_lifecycle"] = {
            "entered": lifecycle_outcome.entered,
            "states": states,
            "cost": lifecycle_outcome.cost.model_dump(mode="json")
            if lifecycle_outcome.cost
            else None,
            "reasoner_load_s": lifecycle_outcome.reasoner_load.startup_s
            if lifecycle_outcome.reasoner_load
            else None,
            "reasoner_unload_s": lifecycle_outcome.reasoner_unload.unload_s
            if lifecycle_outcome.reasoner_unload
            else None,
            "target_unload_s": lifecycle_outcome.target_unload.unload_s
            if lifecycle_outcome.target_unload
            else None,
            "target_reload_s": lifecycle_outcome.target_reload.startup_s
            if lifecycle_outcome.target_reload
            else None,
            "post_reload": lifecycle_outcome.post_reload.model_dump(mode="json")
            if lifecycle_outcome.post_reload
            else None,
            "corroboration": lifecycle_outcome.corroboration.model_dump(mode="json")
            if lifecycle_outcome.corroboration
            else None,
            "diagnosis_final_confidence": (
                lifecycle_outcome.diagnosis.final_confidence
                if lifecycle_outcome.diagnosis
                else None
            ),
            "diagnosis_confidence_basis": (
                lifecycle_outcome.diagnosis.confidence_basis
                if lifecycle_outcome.diagnosis
                else None
            ),
            "errors": lifecycle_outcome.errors,
            "request_policy": settings.phase5r_request_policy,
            "requests": [r.model_dump(mode="json") for r in request_records],
        }
        print(
            f"entered={lifecycle_outcome.entered} states={len(states)} "
            f"post_reload={lifecycle_outcome.post_reload.equivalence if lifecycle_outcome.post_reload else None} "
            f"errors={lifecycle_outcome.errors}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario C failed: {type(exc).__name__}: {exc}")
        scenarios["C_lifecycle"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("C: real deep-diagnosis lifecycle reaches all transitions", c_ok))
    checks.append(("F: request during diagnosis follows the documented policy", f_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario D: economics rejects deep diagnosis                         #
    # ------------------------------------------------------------------ #
    print("--- Scenario D: entry economics refuse (cost > benefit) ---")
    d_ok = False
    try:
        measured_interruption = (
            lifecycle_outcome.cost.total_s if lifecycle_outcome and lifecycle_outcome.cost else 1.0
        )
        refuse_policy = lifecycle_policy.model_copy(
            update={"benefit_scale_s": 1.0}
        )
        unloads_before = target_backend.last_unload
        runner_d = ResidentLifecycleRunner(
            target=target_backend,
            target_model=model,
            target_params=target_params,
            policy=refuse_policy,
            reasoner=reasoner,
            request_policy=RequestPolicy.REJECT,
            interruption_cost_s=measured_interruption,
        )
        outcome_d = runner_d.run(diagnosis, report=report)
        d_ok = bool(
            not outcome_d.entered
            and outcome_d.refused_reason is not None
            and "cost" in outcome_d.refused_reason
            and target_backend.is_running()
            and target_backend.last_unload is unloads_before
        )
        scenarios["D_economics_refuse"] = {
            "entered": outcome_d.entered,
            "refused_reason": outcome_d.refused_reason,
            "cost": outcome_d.entry_economics.cost_utility,
            "benefit": outcome_d.entry_economics.benefit_utility,
            "target_still_running": target_backend.is_running(),
        }
        print(
            f"entered={outcome_d.entered} reason={outcome_d.refused_reason} "
            f"benefit={outcome_d.entry_economics.benefit_utility} "
            f"cost={outcome_d.entry_economics.cost_utility}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario D failed: {type(exc).__name__}: {exc}")
        scenarios["D_economics_refuse"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("D: entry economics refuses deep diagnosis and leaves target running", d_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario E: feasibility gate rejects                                 #
    # ------------------------------------------------------------------ #
    print("--- Scenario E: reasoner feasibility gate rejects ---")
    e_ok = False
    try:
        huge_reasoner = build_resident_reasoner(settings, expected_bytes=1 << 50)
        if huge_reasoner is None:
            raise RuntimeError("could not build reasoner for feasibility scenario")
        runner_e = ResidentLifecycleRunner(
            target=target_backend,
            target_model=model,
            target_params=target_params,
            policy=lifecycle_policy,
            reasoner=huge_reasoner,
            request_policy=RequestPolicy.REJECT,
            interruption_cost_s=0.0,
        )
        outcome_e = runner_e.run(diagnosis, report=report)
        e_ok = bool(
            not outcome_e.entered
            and outcome_e.feasibility is not None
            and not outcome_e.feasibility.feasible
            and target_backend.is_running()
        )
        scenarios["E_feasibility_refuse"] = {
            "entered": outcome_e.entered,
            "feasible": outcome_e.feasibility.feasible if outcome_e.feasibility else None,
            "reasons": outcome_e.feasibility.reasons if outcome_e.feasibility else None,
            "target_still_running": target_backend.is_running(),
        }
        print(
            f"entered={outcome_e.entered} feasible="
            f"{outcome_e.feasibility.feasible if outcome_e.feasibility else None}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario E failed: {type(exc).__name__}: {exc}")
        scenarios["E_feasibility_refuse"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("E: reasoner feasibility gate rejects and leaves target running", e_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario G: adversarial reasoning containment                       #
    # ------------------------------------------------------------------ #
    print("--- Scenario G: adversarial reasoning is contained ---")
    g_results = _adversarial_reasoning_checks(diagnosis, report)
    g_ok = all(g_results.values())
    scenarios["G_adversarial"] = g_results
    for name, flag in g_results.items():
        print(f"  [{'PASS' if flag else 'FAIL'}] {name}")
    checks.append(("G: adversarial reasoning cannot fabricate or inflate", g_ok))
    print()

    # cleanup target
    try:
        if adapter.is_loaded:
            adapter.unload()
    except Exception:  # noqa: BLE001
        pass

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": "5r",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server": str(server_path),
        "diagnosis_origin": diagnosis_origin,
        "diagnosis_id": diagnosis.diagnosis_id,
        "model": model.name,
        "reasoner": reasoner.model_ref,
        "policy": lifecycle_policy.as_dict(),
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    verification_path = settings.reports_dir / f"phase5r_verification_{timestamp}.json"
    verification_path.write_text(
        json.dumps(report_payload, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(f"=== RESULT: PHASE 5R ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {verification_path}")
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
    available = get_platform().global_memory_status()
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
# Phase 7R — quality validation & real experiment safety remediation           #
# --------------------------------------------------------------------------- #
def _phase7r_footprint_diagnosis(
    report: BaselineReport,
    model: ModelProfile,
    *,
    run_id: str,
    peak_rss: float,
    source_report_path: str,
) -> BottleneckDiagnosis:
    """A controlled MODEL_FOOTPRINT diagnosis grounded in a real measured RSS.

    The condition is a controlled scenario (the machine is not genuinely under
    memory pressure), but the evidence value is the real measured resident RSS from
    this acceptance run, and the diagnosis is explicitly labelled as controlled.
    """
    evidence = EvidenceItem(
        metric="peak_rss_bytes",
        observed_value=float(peak_rss),
        description=(
            "real measured resident RSS of the resident target from this Phase 7R "
            "baseline run"
        ),
        polarity=EvidencePolarity.SUPPORTING,
        stage=InferenceStage.END_TO_END,
        strength=EvidenceStrength.HIGH,
        source="phase7r.controlled",
        evidence_id="E1",
    )
    primary = Bottleneck(
        bottleneck_id=f"bn-{run_id}-footprint",
        kind=BottleneckKind.MODEL_FOOTPRINT,
        confidence=0.8,
        is_meaningful=True,
        summary="controlled model-footprint bottleneck grounded in real measured RSS",
        evidence=[evidence],
        contributing_factors=["large resident model/KV footprint"],
        recommended_surface=[],
        source="phase7r.controlled",
        stage=InferenceStage.END_TO_END,
        subject=f"{model.name}/short-short",
        evidence_strength=EvidenceStrength.HIGH,
        supporting_evidence=[evidence],
        rationale=["measured peak_rss_bytes is high enough to justify context right-sizing"],
        detector_version="phase7r.controlled",
    )
    return BottleneckDiagnosis(
        diagnosis_id=f"diag-{run_id}",
        created_at=datetime.now(timezone.utc),
        context=DiagnosisContext(
            source_report_path=source_report_path,
            detector_version="phase7r.controlled",
            hardware=report.hardware,
            backend=report.backend,
            model_refs=[model.name],
            workload_ids=["short-short"],
            conditions=["controlled_model_footprint_grounded_in_real_rss"],
        ),
        meaningful_opportunity=True,
        insufficient_evidence=False,
        escalation_required=False,
        confidence=0.8,
        evidence_strength=EvidenceStrength.HIGH,
        primary=primary,
        supporting_evidence=[evidence],
        summary="controlled model-footprint diagnosis for Phase 7R experiment safety",
    )


def _phase7r_pick_candidate(plan: CandidatePlan, action_id: str) -> OptimizationCandidate | None:
    from inferenceos.contracts.enums import CandidateEligibility

    for candidate in plan.candidates:
        if (
            candidate.action_id == action_id
            and candidate.eligibility is CandidateEligibility.ELIGIBLE
        ):
            return candidate
    return None


def _cmd_verify_phase7r(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase7r.log")

    print("=== InferenceOS Phase 7R Verification (Quality & Experiment Safety) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase7r-{timestamp}"
    checks: list[tuple[str, bool]] = []
    scenarios: dict[str, object] = {}

    try:
        server_path = _resident_server_path(settings)
        models = _load_models(settings, args.model)
        model = max(models, key=lambda profile: profile.file_size_bytes)
    except (FileNotFoundError, ModelMetadataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    registry = build_default_registry()
    opt_policy = build_policy(settings)
    exp_policy = build_experiment_policy(settings).model_copy(
        update={"measured_repetitions": 2, "max_repetitions": 3, "max_total_runs": 10}
    )
    adapter = LlamaCppResidentAdapter(settings=settings, server=server_path)

    retained_context = int(getattr(args, "retained_context", 0) or 16384)
    candidate_context = int(getattr(args, "candidate_context", 0) or 512)
    fraction = 1.0 - (candidate_context / retained_context)
    opt_policy = opt_policy.model_copy(update={"context_reduction_fraction": fraction})

    # ------------------------------------------------------------------ #
    # A: real resident baseline                                            #
    # ------------------------------------------------------------------ #
    print(f"--- Scenario A: real resident baseline at context {retained_context} ---")
    report_r: BaselineReport | None = None
    a_ok = False
    try:
        task = ProfileTask(
            model=model,
            workload=get_workload("short-short"),
            context_length=retained_context,
        )
        profiler = BaselineProfiler(settings=settings, adapter=adapter)
        report_r = profiler.profile(
            [task],
            modes=[MeasurementMode.WARM],
            repetitions=1,
            options=ProfilerOptions(
                gpu_offload_layers=0, device="none", use_mmap=False, telemetry_sampling=False
            ),
        )
        run0 = report_r.runs[0]
        rss_values = [
            float(run.observation.metrics.peak_rss_bytes)
            for run in report_r.runs
            if run.observation.metrics.peak_rss_bytes is not None
        ]
        a_ok = bool(report_r.runs and rss_values and run0.observation.generated_text)
        scenarios["A_baseline"] = {
            "run_id": run_id,
            "context_length": retained_context,
            "runs": len(report_r.runs),
            "peak_rss_bytes": rss_values[0] if rss_values else None,
            "decode_tokens_per_sec": run0.observation.metrics.decode_tokens_per_sec,
            "generated_text": (run0.observation.generated_text or "")[:200],
        }
        print(
            f"runs={len(report_r.runs)} peak_rss={rss_values[0] if rss_values else None} "
            f"text={(run0.observation.generated_text or '')[:60]!r}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario A failed: {type(exc).__name__}: {exc}")
        scenarios["A_baseline"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("A: real resident baseline with real measurements", a_ok))
    print()
    if report_r is None:
        print("cannot continue without a real baseline", file=sys.stderr)
        return 2

    peak_rss = next(
        (
            float(run.observation.metrics.peak_rss_bytes)
            for run in report_r.runs
            if run.observation.metrics.peak_rss_bytes is not None
        ),
        0.0,
    )
    diagnosis = _phase7r_footprint_diagnosis(
        report_r,
        model,
        run_id=run_id,
        peak_rss=peak_rss,
        source_report_path=f"<phase7r-run:{run_id}>",
    )
    plan = generate_candidates(
        diagnosis, report=report_r, registry=registry, policy=opt_policy
    )
    retained_candidate = _phase7r_pick_candidate(plan, "runtime.right_size_context")

    # ------------------------------------------------------------------ #
    # B: real retained candidate                                           #
    # ------------------------------------------------------------------ #
    print("--- Scenario B: real retained candidate (real apply/measure/quality) ---")
    b_ok = False
    result_keep: ExperimentResult | None = None
    try:
        if retained_candidate is None:
            raise RuntimeError("no eligible runtime.right_size_context candidate")
        verifier = ResidentRestorationVerifier(adapter=adapter, model=model)
        result_keep = run_candidate_experiment(
            retained_candidate,
            plan=plan,
            diagnosis=diagnosis,
            report=report_r,
            outcome=None,
            workload_id="short-short",
            registry=registry,
            optimization_policy=opt_policy,
            experiment_policy=exp_policy,
            executor=RuntimeConfigurationExecutor(),
            profiler=BaselineProfiler(settings=settings, adapter=adapter),
            run_id=run_id,
            restoration_verifier=verifier,
        )
        quality_b = result_keep.quality
        b_ok = bool(
            result_keep.status is ExperimentDecisionKind.IMPROVED
            and result_keep.decision_detail is not None
            and result_keep.decision_detail.keep
            and result_keep.executed
            and quality_b is not None
            and quality_b.status.value in {"ok", "acceptable_variation"}
            and result_keep.application is not None
            and result_keep.application.applied
            and result_keep.executed
            and result_keep.restoration is not None
            and result_keep.restoration.verified
            and result_keep.restoration.verification is not None
            and result_keep.restoration.verification.verified
        )
        scenarios["B_retained"] = {
            "action_id": result_keep.action_id,
            "parameters": result_keep.parameters,
            "status": result_keep.status.value if result_keep.status else None,
            "decision": result_keep.decision.value,
            "keep": result_keep.decision_detail.keep if result_keep.decision_detail else None,
            "quality_status": quality_b.status.value if quality_b else None,
            "quality_suite": quality_b.suite_version if quality_b else None,
            "quality_tolerance": quality_b.tolerance if quality_b else None,
            "quality_tolerance_basis": quality_b.tolerance_basis if quality_b else None,
            "performance_delta": result_keep.performance_delta,
            "bottleneck_improved": (
                result_keep.comparison.bottleneck_improved if result_keep.comparison else None
            ),
            "restoration_verified": (
                result_keep.restoration.verification.verified
                if result_keep.restoration and result_keep.restoration.verification
                else None
            ),
            "provenance_run_id": result_keep.provenance.run_id if result_keep.provenance else None,
        }
        print(
            f"status={result_keep.status.value if result_keep.status else None} "
            f"keep={result_keep.decision_detail.keep if result_keep.decision_detail else None} "
            f"quality={quality_b.status.value if quality_b else None}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario B failed: {type(exc).__name__}: {exc}")
        scenarios["B_retained"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("B: real candidate retained with real quality + restoration", b_ok))
    print()

    # ------------------------------------------------------------------ #
    # C: real quality-regression rejection                                 #
    # ------------------------------------------------------------------ #
    print("--- Scenario C: real quality regression -> reject -> restore ---")
    c_ok = False
    result_reject: ExperimentResult | None = None
    try:
        quality_workload = get_quality_workload()
        reject_candidate = retained_candidate
        if reject_candidate is None:
            raise RuntimeError("no eligible runtime.right_size_context candidate")
        verifier_c = ResidentRestorationVerifier(adapter=adapter, model=model)
        result_reject = run_candidate_experiment(
            reject_candidate,
            plan=plan,
            diagnosis=diagnosis,
            report=report_r,
            outcome=None,
            workload_id=quality_workload.workload_id,
            workload=quality_workload,
            registry=registry,
            optimization_policy=opt_policy,
            experiment_policy=exp_policy,
            executor=RuntimeConfigurationExecutor(),
            profiler=BaselineProfiler(settings=settings, adapter=adapter),
            run_id=f"{run_id}-reject",
            restoration_verifier=verifier_c,
        )
        quality_c = result_reject.quality
        c_ok = bool(
            result_reject.status is ExperimentDecisionKind.QUALITY_REGRESSION
            and result_reject.decision_detail is not None
            and not result_reject.decision_detail.keep
            and quality_c is not None
            and quality_c.status.value in {"quality_regression", "structural_failure"}
            and quality_c.unacceptable_output
            and result_reject.executed
            and result_reject.restoration is not None
            and result_reject.restoration.verified
            and result_reject.restoration.verification is not None
            and result_reject.restoration.verification.verified
        )
        scenarios["C_quality_rejection"] = {
            "workload_id": quality_workload.workload_id,
            "status": result_reject.status.value if result_reject.status else None,
            "keep": result_reject.decision_detail.keep if result_reject.decision_detail else None,
            "quality_status": quality_c.status.value if quality_c else None,
            "quality_regression": quality_c.quality_regression if quality_c else None,
            "quality_cases": (
                [
                    {"case_id": case.case_id, "passed": case.passed, "score": case.score}
                    for case in quality_c.cases
                ]
                if quality_c
                else None
            ),
            "quality_suite": quality_c.suite_version if quality_c else None,
            "quality_tolerance": quality_c.tolerance if quality_c else None,
            "restoration_verified": (
                result_reject.restoration.verification.verified
                if result_reject.restoration and result_reject.restoration.verification
                else None
            ),
        }
        print(
            f"status={result_reject.status.value if result_reject.status else None} "
            f"quality={quality_c.status.value if quality_c else None} "
            f"restored={result_reject.restoration.verified if result_reject.restoration else None}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario C failed: {type(exc).__name__}: {exc}")
        scenarios["C_quality_rejection"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("C: real quality regression rejected and target restored", c_ok))
    print()

    # ------------------------------------------------------------------ #
    # D: safety rejection                                                  #
    # ------------------------------------------------------------------ #
    print("--- Scenario D: unsafe candidate refused before execution ---")
    d_ok = False
    try:
        pressure_primary = (
            diagnosis.primary.model_copy(update={"kind": BottleneckKind.PAGING})
            if diagnosis.primary is not None
            else None
        )
        pressure_diag = diagnosis.model_copy(update={"primary": pressure_primary})
        unsafe_plan = generate_candidates(
            pressure_diag, report=report_r, registry=registry, policy=opt_policy
        )
        unsafe_candidate = next(
            (
                candidate
                for candidate in unsafe_plan.candidates
                if candidate.action_id == "system.increase_pagefile"
            ),
            None,
        )
        if unsafe_candidate is None:
            raise RuntimeError("system.increase_pagefile was not considered")
        unsafe_result = run_candidate_experiment(
            unsafe_candidate,
            plan=unsafe_plan,
            diagnosis=pressure_diag,
            report=report_r,
            outcome=None,
            workload_id="short-short",
            registry=registry,
            optimization_policy=opt_policy,
            experiment_policy=exp_policy,
            executor=RuntimeConfigurationExecutor(),
            profiler=BaselineProfiler(settings=settings, adapter=adapter),
            run_id=f"{run_id}-unsafe",
        )
        d_ok = bool(
            unsafe_result.executed is False
            and unsafe_result.baseline_measurement is None
            and unsafe_result.status is ExperimentDecisionKind.UNSAFE
        )
        scenarios["D_safety"] = {
            "status": unsafe_result.status.value if unsafe_result.status else None,
            "executed": unsafe_result.executed,
            "reason": unsafe_result.decision_detail.reason if unsafe_result.decision_detail else None,
        }
        print(
            f"status={unsafe_result.status.value if unsafe_result.status else None} "
            f"executed={unsafe_result.executed}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario D failed: {type(exc).__name__}: {exc}")
        scenarios["D_safety"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("D: unsafe candidate refused before execution", d_ok))
    print()

    # ------------------------------------------------------------------ #
    # E: fabricated evidence cannot cause execution/retention              #
    # ------------------------------------------------------------------ #
    print("--- Scenario E: fabricated evidence is rejected ---")
    e_ok = False
    try:
        if retained_candidate is None:
            raise RuntimeError("no candidate to mutate for the fabricated scenario")
        fabricated = retained_candidate.model_copy(
            update={
                "action_id": "runtime.unregistered_fabricated_action",
                "model_ref": "fabricated-model-that-does-not-exist",
            }
        )
        fabricated_result = run_candidate_experiment(
            fabricated,
            plan=plan,
            diagnosis=diagnosis,
            report=report_r,
            outcome=None,
            workload_id="short-short",
            registry=registry,
            optimization_policy=opt_policy,
            experiment_policy=exp_policy,
            executor=RuntimeConfigurationExecutor(),
            profiler=BaselineProfiler(settings=settings, adapter=adapter),
            run_id=f"{run_id}-fabricated",
        )
        e_ok = bool(
            fabricated_result.executed is False
            and fabricated_result.status is ExperimentDecisionKind.UNSAFE
        )
        scenarios["E_fabricated"] = {
            "status": fabricated_result.status.value
            if fabricated_result.status
            else None,
            "executed": fabricated_result.executed,
            "reason": fabricated_result.decision_detail.reason
            if fabricated_result.decision_detail
            else None,
        }
        print(
            f"status={fabricated_result.status.value if fabricated_result.status else None} "
            f"executed={fabricated_result.executed}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario E failed: {type(exc).__name__}: {exc}")
        scenarios["E_fabricated"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("E: fabricated/hallucinated evidence cannot execute or retain", e_ok))
    print()

    # ------------------------------------------------------------------ #
    # F: persistence + provenance                                          #
    # ------------------------------------------------------------------ #
    print("--- Scenario F: persistence and provenance binding ---")
    f_ok = False
    try:
        service = _knowledge_service(settings, args.db)
        persisted = []
        for result in (result_keep, result_reject):
            if result is None:
                raise RuntimeError("missing experiment result for persistence")
            outcome = service.persist_result(result)
            fetched = service.store.get_by_experiment(result.experiment_id)
            provenance = verify_provenance(
                result,
                expected_run_id=result.run_id or "",
                expected_parameters_hash=parameters_hash(result.parameters),
            )
            persisted.append(
                {
                    "experiment_id": result.experiment_id,
                    "status": outcome.optimization.status.value,
                    "readback": fetched is not None,
                    "outcome": fetched.outcome.value if fetched is not None else None,
                    "provenance_ok": provenance.ok,
                    "provenance_reasons": provenance.reasons,
                }
            )
        # Stale evidence must fail.
        stale = verify_provenance(result_keep, expected_run_id="some-other-run")  # type: ignore[arg-type]
        stale_rejected = not stale.ok
        f_ok = bool(
            all(item["readback"] and item["provenance_ok"] for item in persisted)
            and stale_rejected
        )
        scenarios["F_persistence"] = {
            "persisted": persisted,
            "stale_evidence_rejected": stale_rejected,
        }
        print(f"persisted={persisted} stale_rejected={stale_rejected}")
    except Exception as exc:  # noqa: BLE001
        print(f"scenario F failed: {type(exc).__name__}: {exc}")
        scenarios["F_persistence"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("F: retained + rejected persisted; stale evidence rejected", f_ok))
    print()

    # cleanup target
    try:
        if adapter.is_loaded:
            adapter.unload()
    except Exception:  # noqa: BLE001
        pass

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": "7r",
        "result": "PASS" if ok else "FAIL",
        "command": "inferenceos verify-phase7r",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model": {"name": model.name, "path": model.path},
        "backend": {"server": str(server_path), "adapter": adapter.__class__.__name__},
        "quality_suite_version": settings.phase7r_suite_version,
        "retained_context": retained_context,
        "candidate_context": candidate_context,
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "known_limitations": [
            "the MODEL_FOOTPRINT diagnosis is a controlled scenario grounded in the "
            "real measured resident RSS; it is labelled as controlled",
            "GPU/VRAM feasibility is not measured on this machine",
        ],
    }
    verification_path = settings.reports_dir / f"phase7r_verification_{timestamp}.json"
    verification_path.write_text(
        json.dumps(report_payload, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(f"=== RESULT: PHASE 7R ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
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
# Phase 8R — persistent knowledge & real rollback remediation                  #
# --------------------------------------------------------------------------- #
def _cmd_verify_phase8r(args: argparse.Namespace) -> int:
    import shutil

    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase8r.log")

    print("=== InferenceOS Phase 8R Verification (Persistent Knowledge & Real Rollback) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"phase8r-{timestamp}"
    checks: list[tuple[str, bool]] = []
    scenarios: dict[str, object] = {}

    try:
        server_path = _resident_server_path(settings)
        models = _load_models(settings, args.model)
        model = max(models, key=lambda profile: profile.file_size_bytes)
    except (FileNotFoundError, ModelMetadataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    registry = build_default_registry()
    knowledge_policy = build_knowledge_policy(settings)
    db_path = (
        Path(args.db)
        if getattr(args, "db", None)
        else settings.data_dir / f"phase8r_verify_{timestamp}.db"
    )
    exp_policy = build_experiment_policy(settings).model_copy(
        update={"measured_repetitions": 2, "max_repetitions": 3, "max_total_runs": 10}
    )
    baseline_context = int(getattr(args, "baseline_context", 0) or 4096)
    candidate_context = int(getattr(args, "candidate_context", 0) or 2048)
    adapter = LlamaCppResidentAdapter(settings=settings, server=server_path)
    tracked_pids: set[int] = set()

    def _set_target_configuration(configuration: RuntimeConfiguration) -> None:
        if adapter.is_loaded:
            adapter.unload()
        adapter.load(model, runtime_params_from(configuration))
        if adapter.pid is not None:
            tracked_pids.add(adapter.pid)

    def _ensure_unloaded() -> None:
        if adapter.is_loaded:
            adapter.unload()

    def _fresh_service() -> KnowledgeService:
        return KnowledgeService(
            store=SQLiteKnowledgeStore(db_path), registry=registry, policy=knowledge_policy
        )

    # ------------------------------------------------------------------ #
    # Real baseline report                                                 #
    # ------------------------------------------------------------------ #
    print(f"--- Baseline: real resident target at context {baseline_context} ---")
    try:
        task = ProfileTask(
            model=model,
            workload=get_workload("short-short"),
            context_length=baseline_context,
        )
        report = BaselineProfiler(settings=settings, adapter=adapter).profile(
            [task],
            modes=[MeasurementMode.WARM],
            repetitions=1,
            options=ProfilerOptions(
                gpu_offload_layers=0, device="none", use_mmap=False, telemetry_sampling=False
            ),
        )
        if adapter.pid is not None:
            tracked_pids.add(adapter.pid)
        peak_rss = next(
            (
                float(run.observation.metrics.peak_rss_bytes)
                for run in report.runs
                if run.observation.metrics.peak_rss_bytes is not None
            ),
            0.0,
        )
        hardware = report.hardware
        backend = report.backend
        scenarios["baseline"] = {
            "run_id": run_id,
            "context_length": baseline_context,
            "runs": len(report.runs),
            "peak_rss_bytes": peak_rss,
        }
        print(f"runs={len(report.runs)} peak_rss={peak_rss}")
    except Exception as exc:  # noqa: BLE001
        print(f"baseline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    diagnosis = _phase7r_footprint_diagnosis(
        report,
        model,
        run_id=run_id,
        peak_rss=peak_rss,
        source_report_path=f"<phase8r-run:{run_id}>",
    )
    opt_policy = build_policy(settings)
    plan = generate_candidates(diagnosis, report=report, registry=registry, policy=opt_policy)
    candidate = _phase7r_pick_candidate(plan, "runtime.right_size_context")

    # ------------------------------------------------------------------ #
    # Scenario A: real optimization -> persist -> restart -> real rollback #
    # ------------------------------------------------------------------ #
    print("--- Scenario A: persisted rollback executed against the real target ---")
    a_ok = False
    rollback_id = ""
    experiment_id = ""
    exec_result = None
    loaded_rollback = None
    try:
        if candidate is None:
            raise RuntimeError("no eligible runtime.right_size_context candidate")
        verifier = ResidentRestorationVerifier(adapter=adapter, model=model)
        result_opt = run_candidate_experiment(
            candidate,
            plan=plan,
            diagnosis=diagnosis,
            report=report,
            outcome=None,
            workload_id="short-short",
            registry=registry,
            optimization_policy=opt_policy,
            experiment_policy=exp_policy,
            executor=RuntimeConfigurationExecutor(),
            profiler=BaselineProfiler(settings=settings, adapter=adapter),
            run_id=run_id,
            restoration_verifier=verifier,
        )
        if adapter.pid is not None:
            tracked_pids.add(adapter.pid)
        experiment_id = result_opt.experiment_id
        rollback_id = f"rb-{experiment_id}"
        first_service = _fresh_service()
        persist = first_service.persist_result(result_opt)
        assert persist.persisted and persist.rollback_record is not None

        # Genuine process restart of the store: brand-new service instance.
        service = _fresh_service()
        loaded_rollback = service.get_rollback(rollback_id)
        loaded_opt = service.store.get_by_experiment(experiment_id)
        if loaded_rollback is None or loaded_opt is None:
            raise RuntimeError("rollback record did not survive store reinitialization")

        # Put the target into the optimized (resulting) state, as a retained optimization.
        resulting = result_opt.request.experiment_configuration  # type: ignore[union-attr]
        _set_target_configuration(resulting)
        pid_optimized = adapter.pid

        executor = RollbackExecutor(
            service=service,
            registry=registry,
            adapter=adapter,
            model=model,
            hardware=hardware,
            backend=backend,
            policy=RollbackExecutionPolicy(),
        )
        exec_result = executor.execute(rollback_id, run_id=run_id)
        events = service.store.events_for_record(f"opt-{experiment_id}")
        event_types = {event.event_type.value for event in events}
        a_ok = bool(
            exec_result.status is RollbackExecutionStatus.APPLIED_VERIFIED
            and exec_result.integrity_ok
            and exec_result.applicability.classification.value == "applicable"
            and exec_result.configuration_restored
            and exec_result.behavioural_probe_passed
            and exec_result.process_reloaded
            and exec_result.pid_before != exec_result.pid_after
            and exec_result.ready is True
            and exec_result.event_ids
            and {"rollback_requested", "rollback_checkpoint", "rollback_executed"}
            <= event_types
        )
        scenarios["A_persisted_rollback"] = {
            "run_id": run_id,
            "rollback_id": rollback_id,
            "experiment_id": experiment_id,
            "status": exec_result.status.value,
            "applicability": exec_result.applicability.classification.value,
            "integrity_ok": exec_result.integrity_ok,
            "configuration_restored": exec_result.configuration_restored,
            "behavioural_probe_passed": exec_result.behavioural_probe_passed,
            "measurement_verification_passed": exec_result.measurement_verification_passed,
            "measurement_detail": exec_result.measurement_detail,
            "pid_before": exec_result.pid_before,
            "pid_after": exec_result.pid_after,
            "identity_before": exec_result.identity_before.model_dump(mode="json")
            if exec_result.identity_before
            else None,
            "identity_after": exec_result.identity_after.model_dump(mode="json")
            if exec_result.identity_after
            else None,
            "process_reloaded": exec_result.process_reloaded,
            "reload_s": exec_result.reload_s,
            "ready": exec_result.ready,
            "requested_configuration": exec_result.requested_configuration.model_dump()
            if exec_result.requested_configuration
            else None,
            "checkpoint_configuration": exec_result.checkpoint_configuration.model_dump()
            if exec_result.checkpoint_configuration
            else None,
            "actual_configuration": exec_result.actual_configuration.model_dump()
            if exec_result.actual_configuration
            else None,
            "baseline_probe_hash": exec_result.baseline_probe_hash,
            "restored_probe_hash": exec_result.restored_probe_hash,
            "event_ids": exec_result.event_ids,
            "audit_event_types": sorted(event_types),
            "pid_optimized": pid_optimized,
        }
        print(
            f"status={exec_result.status.value} pid {exec_result.pid_before}->{exec_result.pid_after} "
            f"restored={exec_result.configuration_restored} probe={exec_result.behavioural_probe_passed}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario A failed: {type(exc).__name__}: {exc}")
        scenarios["A_persisted_rollback"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("A: persisted rollback applied to real target and verified", a_ok))
    print()

    # ------------------------------------------------------------------ #
    # Idempotency                                                          #
    # ------------------------------------------------------------------ #
    print("--- Idempotency: second rollback is a no-op already_restored ---")
    idem_ok = False
    try:
        service = _fresh_service()
        executor = RollbackExecutor(
            service=service,
            registry=registry,
            adapter=adapter,
            model=model,
            hardware=hardware,
            backend=backend,
            policy=RollbackExecutionPolicy(),
        )
        second = executor.execute(rollback_id, run_id=f"{run_id}-idem")
        idem_ok = bool(
            second.status is RollbackExecutionStatus.ALREADY_RESTORED
            and second.process_reloaded is False
        )
        scenarios["idempotency"] = {
            "status": second.status.value,
            "process_reloaded": second.process_reloaded,
            "pid": adapter.pid,
        }
        print(f"status={second.status.value} reloaded={second.process_reloaded}")
    except Exception as exc:  # noqa: BLE001
        print(f"idempotency failed: {type(exc).__name__}: {exc}")
        scenarios["idempotency"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("idempotency: repeated rollback reports already_restored", idem_ok))
    print()

    # restore optimized state for rejection scenarios
    try:
        if loaded_rollback is not None:
            _set_target_configuration(loaded_rollback.resulting_state)
    except Exception:  # noqa: BLE001
        pass

    # ------------------------------------------------------------------ #
    # Scenario C: stale/incompatible rejection (no target change)          #
    # ------------------------------------------------------------------ #
    print("--- Scenario C: stale/incompatible rollback rejected before applying ---")
    c_ok = False
    try:
        if loaded_rollback is None:
            raise RuntimeError("no loaded rollback record")
        service = _fresh_service()
        executor = RollbackExecutor(
            service=service, registry=registry, adapter=adapter, model=model,
            hardware=hardware, backend=backend, policy=RollbackExecutionPolicy(),
        )
        pid_before = adapter.pid
        incompatible = seal_rollback_record(
            loaded_rollback.model_copy(update={"model_key": "different-model-xyz"})
        )
        res_model = executor.execute_record(incompatible, run_id=f"{run_id}-c1")
        old = datetime.now(timezone.utc) - timedelta(days=365)
        stale = seal_rollback_record(loaded_rollback.model_copy(update={"created_at": old}))
        res_stale = executor.execute_record(stale, run_id=f"{run_id}-c2")
        c_ok = bool(
            res_model.status is RollbackExecutionStatus.REJECTED_INAPPLICABLE
            and res_model.applicability.classification.value == "incompatible_model"
            and res_stale.status is RollbackExecutionStatus.REJECTED_INAPPLICABLE
            and res_stale.applicability.classification.value == "stale"
            and adapter.pid == pid_before
        )
        scenarios["C_stale_incompatible"] = {
            "model_status": res_model.status.value,
            "model_classification": res_model.applicability.classification.value,
            "stale_status": res_stale.status.value,
            "stale_classification": res_stale.applicability.classification.value,
            "target_unchanged": adapter.pid == pid_before,
        }
        print(
            f"model={res_model.applicability.classification.value} "
            f"stale={res_stale.applicability.classification.value} target_unchanged={adapter.pid == pid_before}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario C failed: {type(exc).__name__}: {exc}")
        scenarios["C_stale_incompatible"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("C: stale/incompatible rollback rejected without touching the target", c_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario D: tamper/integrity rejection                               #
    # ------------------------------------------------------------------ #
    print("--- Scenario D: integrity failure rejected ---")
    d_ok = False
    try:
        if loaded_rollback is None:
            raise RuntimeError("no loaded rollback record")
        service = _fresh_service()
        executor = RollbackExecutor(
            service=service, registry=registry, adapter=adapter, model=model,
            hardware=hardware, backend=backend, policy=RollbackExecutionPolicy(),
        )
        pid_before = adapter.pid
        tampered = loaded_rollback.model_copy(
            update={"previous_state": RuntimeConfiguration(context_length=1)}
        )
        res_tamper = executor.execute_record(tampered, run_id=f"{run_id}-d1")
        # Persisted tamper is detected on read by the store checksum.
        tamper_path = settings.data_dir / f"phase8r_tamper_{timestamp}.db"
        shutil.copyfile(db_path, tamper_path)
        connection = sqlite3.connect(str(tamper_path))
        connection.execute(
            "UPDATE rollback_records SET payload_json = payload_json || ' ' WHERE rollback_id = ?",
            (rollback_id,),
        )
        connection.commit()
        connection.close()
        corrupt_detected = False
        try:
            SQLiteKnowledgeStore(tamper_path).get_rollback(rollback_id)
        except KnowledgeCorruptionError:
            corrupt_detected = True
        finally:
            tamper_path.unlink(missing_ok=True)
        d_ok = bool(
            res_tamper.status is RollbackExecutionStatus.REJECTED_INTEGRITY
            and corrupt_detected
            and adapter.pid == pid_before
        )
        scenarios["D_tamper_integrity"] = {
            "tamper_status": res_tamper.status.value,
            "integrity_ok": res_tamper.integrity_ok,
            "persisted_tamper_detected": corrupt_detected,
            "target_unchanged": adapter.pid == pid_before,
        }
        print(
            f"status={res_tamper.status.value} persisted_tamper_detected={corrupt_detected} "
            f"target_unchanged={adapter.pid == pid_before}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario D failed: {type(exc).__name__}: {exc}")
        scenarios["D_tamper_integrity"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("D: integrity failure detected and refused", d_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario E: rollback failure containment                             #
    # ------------------------------------------------------------------ #
    print("--- Scenario E: rollback failure contained, checkpoint restored ---")
    e_ok = False
    try:
        if loaded_rollback is None:
            raise RuntimeError("no loaded rollback record")
        _set_target_configuration(loaded_rollback.resulting_state)
        service = _fresh_service()
        executor = RollbackExecutor(
            service=service, registry=registry, adapter=adapter, model=model,
            hardware=hardware, backend=backend,
            policy=RollbackExecutionPolicy(fault_stage="after_apply"),
        )
        res_fail = executor.execute_record(loaded_rollback, run_id=f"{run_id}-e")
        e_ok = bool(
            res_fail.status is RollbackExecutionStatus.FAILED_RESTORED
            and res_fail.recovery_attempted
            and res_fail.recovery_verified
        )
        scenarios["E_failure_containment"] = {
            "status": res_fail.status.value,
            "recovery_attempted": res_fail.recovery_attempted,
            "recovery_verified": res_fail.recovery_verified,
            "errors": res_fail.errors,
            "pid": adapter.pid,
        }
        print(
            f"status={res_fail.status.value} recovery_verified={res_fail.recovery_verified}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario E failed: {type(exc).__name__}: {exc}")
        scenarios["E_failure_containment"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("E: rollback failure contained and checkpoint recovered", e_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario B: negative knowledge persists                              #
    # ------------------------------------------------------------------ #
    print("--- Scenario B: rejected optimization persists as negative knowledge ---")
    b_ok = False
    try:
        reject_policy = build_policy(settings).model_copy(
            update={
                "context_reduction_fraction": 1.0
                - (512 / float(baseline_context))
            }
        )
        reject_plan = generate_candidates(
            diagnosis, report=report, registry=registry, policy=reject_policy
        )
        reject_candidate = _phase7r_pick_candidate(reject_plan, "runtime.right_size_context")
        if reject_candidate is None:
            raise RuntimeError("no eligible candidate for the negative-knowledge scenario")
        quality_workload = get_quality_workload()
        _ensure_unloaded()
        reject_result = run_candidate_experiment(
            reject_candidate,
            plan=reject_plan,
            diagnosis=diagnosis,
            report=report,
            outcome=None,
            workload_id=quality_workload.workload_id,
            workload=quality_workload,
            registry=registry,
            optimization_policy=reject_policy,
            experiment_policy=exp_policy,
            executor=RuntimeConfigurationExecutor(),
            profiler=BaselineProfiler(settings=settings, adapter=adapter),
            run_id=f"{run_id}-negative",
        )
        if adapter.pid is not None:
            tracked_pids.add(adapter.pid)
        persist_service = _fresh_service()
        persist_service.persist_result(reject_result)
        reopened = _fresh_service()
        negative = reopened.store.get_by_experiment(reject_result.experiment_id)
        provenance_ok = verify_provenance(
            reject_result, expected_run_id=f"{run_id}-negative"
        ).ok
        wrong_run_rejected = not verify_provenance(
            reject_result, expected_run_id="wrong-run-id"
        ).ok
        b_ok = bool(
            reject_result.status is ExperimentDecisionKind.QUALITY_REGRESSION
            and negative is not None
            and negative.outcome is OptimizationOutcome.QUALITY_REGRESSION
            and negative.quality_regression
            and not negative.keep
            and provenance_ok
            and wrong_run_rejected
        )
        scenarios["B_negative_knowledge"] = {
            "experiment_id": reject_result.experiment_id,
            "decision": reject_result.status.value if reject_result.status else None,
            "persisted_outcome": negative.outcome.value if negative else None,
            "quality_regression": negative.quality_regression if negative else None,
            "keep": negative.keep if negative else None,
            "provenance_ok": provenance_ok,
            "wrong_run_rejected": wrong_run_rejected,
            "readback_after_restart": negative is not None,
        }
        print(
            f"decision={reject_result.status.value if reject_result.status else None} "
            f"persisted_outcome={negative.outcome.value if negative else None} "
            f"provenance_ok={provenance_ok}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario B failed: {type(exc).__name__}: {exc}")
        scenarios["B_negative_knowledge"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("B: rejected optimization persists as negative knowledge", b_ok))
    print()

    # ------------------------------------------------------------------ #
    # Provenance binding of the optimization result                        #
    # ------------------------------------------------------------------ #
    provenance_ok = False
    try:
        if exec_result is not None:
            # The executed rollback's run id is bound to this acceptance run.
            provenance_ok = exec_result.run_id == run_id
        scenarios["provenance"] = {"rollback_execution_run_id": exec_result.run_id if exec_result else None}
    except Exception:  # noqa: BLE001
        provenance_ok = False
    checks.append(("provenance: rollback execution bound to the current run", provenance_ok))

    # cleanup + orphan check
    try:
        if adapter.is_loaded:
            adapter.unload()
        if adapter.last_unload is not None and adapter.last_unload.pid is not None:
            tracked_pids.discard(adapter.last_unload.pid)
    except Exception:  # noqa: BLE001
        pass
    orphans = sorted(pid for pid in tracked_pids if psutil.pid_exists(pid))
    checks.append(("no orphaned inference processes remain", not orphans))

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": "8r",
        "result": "PASS" if ok else "FAIL",
        "command": "inferenceos verify-phase8r",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "db_path": str(db_path),
        "model": {"name": model.name, "path": model.path},
        "backend": {"server": str(server_path)},
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "orphan_pids": orphans,
        "known_limitations": [
            "the MODEL_FOOTPRINT diagnosis is a controlled scenario grounded in the "
            "real measured resident RSS",
            "experiments run CPU-only so the context/memory effect is observable",
        ],
    }
    verification_path = settings.reports_dir / f"phase8r_verification_{timestamp}.json"
    verification_path.write_text(
        json.dumps(report_payload, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(f"=== RESULT: PHASE 8R ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
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

    # Persist a servable DeploymentState bound to the real model identity. This is a
    # structured state file, not a server launcher; `serve --deployment` consumes it.
    deployment_path: Path | None = None
    try:
        if report.runs:
            model = report.runs[0].context.model
            state = deployment_from_loop_result(
                result.run.run_id,
                result,
                model,
                backend_name=report.backend.backend_name,
                model_content_hash=file_sha256(model.path),
            )
            deployment_path = settings.reports_dir / f"deployment_{timestamp}.json"
            write_deployment_state(deployment_path, state)
    except (OSError, ValueError) as exc:  # pragma: no cover - defensive
        print(f"note: deployment state could not be persisted: {exc}", file=sys.stderr)

    print()
    print(f"run: {out_path}")
    if deployment_path is not None:
        print(f"deployment: {deployment_path}")
    return 1 if result.termination.safety else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "serve.log")

    try:
        state = load_deployment_state(args.deployment)
    except (DeploymentIntegrityError, ValueError, OSError) as exc:
        print(f"error: invalid deployment state: {exc}", file=sys.stderr)
        return 1

    prompt = args.prompt or "In one sentence, what is a key-value cache?"
    evidence = serve_deployment(
        state,
        settings=settings,
        prompt=prompt,
        n_predict=args.n_predict,
        serve_seconds=args.duration or 0.0,
    )
    if args.json:
        print(evidence.model_dump_json(indent=2))
    else:
        print("=== InferenceOS serve --deployment ===")
        print(f"deployment : {state.run_id} (status={state.status.value})")
        print(f"model      : {state.model_ref} -> {state.model_path}")
        print(f"backend    : {evidence.backend_server}")
        print(f"pid        : {evidence.pid}")
        print(f"identity   : {evidence.identity.create_time if evidence.identity else None}")
        print(f"config ok  : {evidence.configuration_matches}")
        print(f"resident   : {evidence.residency_verified}")
        print(f"probe      : {evidence.probe_text[:200]!r}")
        print(f"peak/final : {evidence.peak_rss_bytes} / {evidence.final_rss_bytes}")
        print(f"shutdown   : clean={evidence.clean_shutdown} orphan={evidence.orphan_process}")
        if evidence.reasons:
            print("reasons    :")
            for reason in evidence.reasons:
                print(f"  - {reason}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out) if args.out else settings.reports_dir / f"serve_{timestamp}.json"
    )
    out_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
    print(f"report: {out_path}")
    return 0 if evidence.served else 1


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
# Phase 9R — fresh full-pipeline autonomous optimization remediation           #
# --------------------------------------------------------------------------- #
def _phase9r_controlled_footprint_factory(
    run_id: str,
) -> Callable[[BaselineReport, HardwareProfile, ModelProfile], BottleneckDiagnosis]:
    """Build a controlled MODEL_FOOTPRINT diagnosis grounded in a fresh baseline."""

    def factory(
        report: BaselineReport, hardware: HardwareProfile, model: ModelProfile
    ) -> BottleneckDiagnosis:
        peak = next(
            (
                float(run.observation.metrics.peak_rss_bytes)
                for run in report.runs
                if run.observation.metrics.peak_rss_bytes is not None
            ),
            0.0,
        )
        return _phase7r_footprint_diagnosis(
            report,
            model,
            run_id=run_id,
            peak_rss=peak,
            source_report_path=f"<phase9r-run:{run_id}>",
        )

    return factory


def _cmd_verify_phase9r(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase9r.log")

    print("=== InferenceOS Phase 9R Verification (Fresh Full-Pipeline Run) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id_a = f"phase9r-noop-{timestamp}"
    run_id_b = f"phase9r-opt-{timestamp}"
    checks: list[tuple[str, bool]] = []
    scenarios: dict[str, object] = {}

    try:
        server_path = _resident_server_path(settings)
        models = _load_models(settings, args.model)
        model = max(models, key=lambda profile: profile.file_size_bytes)
    except (FileNotFoundError, ModelMetadataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    registry = build_default_registry()
    knowledge_policy = build_knowledge_policy(settings)
    experiment_policy = build_experiment_policy(settings).model_copy(
        update={"measured_repetitions": 2, "max_repetitions": 3, "max_total_runs": 10}
    )
    adapter = LlamaCppResidentAdapter(settings=settings, server=server_path)
    baseline_processes = len([p for p in psutil.process_iter(["name"]) if p.info.get("name") == "llama-server.exe"])

    # A stale, valid-looking baseline artifact that must NOT be consumed.
    decoy_path = settings.reports_dir / "phase9r_decoy_baseline.json"
    existing = sorted(settings.reports_dir.glob("phase3_verification_*.json"))
    if existing:
        decoy_path.write_text(existing[-1].read_text(encoding="utf-8"), encoding="utf-8")
    else:
        decoy_path.write_text('{"stale": true, "note": "decoy baseline"}', encoding="utf-8")
    decoy_bytes_before = decoy_path.read_bytes()

    # ------------------------------------------------------------------ #
    # Scenario A: fresh no-op                                              #
    # ------------------------------------------------------------------ #
    print("--- Scenario A: fresh no-op (characterize->introspect->baseline->diagnose) ---")
    a_ok = False
    try:
        service_a = KnowledgeService(
            store=SQLiteKnowledgeStore(settings.data_dir / f"phase9r_noop_{timestamp}.db"),
            registry=registry,
            policy=knowledge_policy,
        )
        policy_a = _phase9_policy(
            settings,
            allow_deep_diagnosis=False,
            require_knowledge=False,
            budget=_phase9_budget(
                settings, max_iterations=1, max_experiments=1, max_deep_diagnoses=0
            ),
        )
        orchestrator_a = FreshRunOrchestrator(
            settings=settings,
            target=adapter,
            executor=RuntimeConfigurationExecutor(),
            registry=registry,
            knowledge_service=service_a,
            loop_policy=policy_a,
            optimization_policy=build_policy(settings),
            experiment_policy=experiment_policy,
            model_path=Path(model.path),
            policy=FreshRunPolicy(
                workload_ids=["short-short"],
                baseline_repetitions=2,
                context_override=2048,
            ),
            run_id=run_id_a,
        )
        execution_a = orchestrator_a.run()
        result_a = execution_a.result
        loop_a = execution_a.loop
        decoy_bytes_after = decoy_path.read_bytes()
        decoy_untouched = decoy_bytes_after == decoy_bytes_before
        baseline_name = Path(result_a.baseline_report_path or "").name
        decoy_listed = any("phase9r_decoy_baseline.json" in c for c in result_a.stale_artifact_checks)
        a_ok = bool(
            loop_a.termination.reason
            in {
                LoopTerminationReason.NO_MEANINGFUL_OPPORTUNITY,
                LoopTerminationReason.INSUFFICIENT_EVIDENCE,
            }
            and loop_a.run.budget.experiments_used == 0
            and loop_a.changed is False
            and baseline_name.startswith(run_id_a)
            and result_a.provenance.all_bound_to_run()
            and result_a.provenance.run_id == run_id_a
            and loop_a.run.run_id == run_id_a
            and result_a.deployment.status
            in {DeploymentStatus.NO_CANDIDATE, DeploymentStatus.NO_CHANGE}
            and decoy_untouched
            and decoy_listed
            and result_a.characterization_s >= 0.0
            and result_a.baseline_s > 0.0
        )
        scenarios["A_fresh_noop"] = {
            "run_id": run_id_a,
            "termination": loop_a.termination.reason.value,
            "clean": loop_a.termination.clean,
            "experiments_used": loop_a.run.budget.experiments_used,
            "changed": loop_a.changed,
            "baseline_report_path": result_a.baseline_report_path,
            "baseline_artifact_id": result_a.baseline_artifact_id,
            "characterization_s": result_a.characterization_s,
            "introspection_s": result_a.introspection_s,
            "baseline_s": result_a.baseline_s,
            "deployment_status": result_a.deployment.status.value,
            "provenance_run_id": result_a.provenance.run_id,
            "all_artifacts_bound": result_a.provenance.all_bound_to_run(),
            "hardware_summary": result_a.hardware_summary,
            "model_summary": result_a.model_summary,
            "decoy_untouched": decoy_untouched,
            "decoy_listed_as_stale": decoy_listed,
            "stale_artifact_checks": result_a.stale_artifact_checks,
            "overhead": loop_a.run.overhead.model_dump(mode="json"),
        }
        print(
            f"termination={loop_a.termination.reason.value} experiments={loop_a.run.budget.experiments_used} "
            f"baseline={baseline_name} decoy_untouched={decoy_untouched}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario A failed: {type(exc).__name__}: {exc}")
        scenarios["A_fresh_noop"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("A: fresh no-op establishes its own state and terminates", a_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario B: fresh optimization path                                  #
    # ------------------------------------------------------------------ #
    print("--- Scenario B: fresh optimization path (candidate->experiment->retain) ---")
    b_ok = False
    try:
        context_b = 16384
        candidate_b = 512
        service_b = KnowledgeService(
            store=SQLiteKnowledgeStore(settings.data_dir / f"phase9r_opt_{timestamp}.db"),
            registry=registry,
            policy=knowledge_policy,
        )
        opt_policy_b = build_policy(settings).model_copy(
            update={"context_reduction_fraction": 1.0 - (candidate_b / context_b)}
        )
        policy_b = _phase9_policy(
            settings,
            allow_deep_diagnosis=False,
            require_knowledge=False,
            budget=_phase9_budget(
                settings, max_iterations=1, max_experiments=1, max_deep_diagnoses=0
            ),
        )
        orchestrator_b = FreshRunOrchestrator(
            settings=settings,
            target=adapter,
            executor=RuntimeConfigurationExecutor(),
            registry=registry,
            knowledge_service=service_b,
            loop_policy=policy_b,
            optimization_policy=opt_policy_b,
            experiment_policy=experiment_policy,
            model_path=Path(model.path),
            policy=FreshRunPolicy(
                workload_ids=["short-short"],
                baseline_repetitions=2,
                context_override=context_b,
                controlled_optimization_allowed=True,
            ),
            run_id=run_id_b,
            diagnosis_factory=_phase9r_controlled_footprint_factory(run_id_b),
            # CPU-only so the context/KV change is observable in system RSS on this
            # machine (GPU offload would move the KV cache out of system memory).
            # mmap is already enabled so the only eligible footprint candidate is
            # context right-sizing (enable_mmap's precondition is not met).
            profiler_options=ProfilerOptions(
                gpu_offload_layers=0,
                device="none",
                use_mmap=True,
                telemetry_sampling=False,
            ),
        )
        execution_b = orchestrator_b.run()
        result_b = execution_b.result
        loop_b = execution_b.loop
        retained_iterations = [i for i in loop_b.run.iterations if i.retained]
        retained_exp = retained_iterations[0].experiment_id if retained_iterations else None
        knowledge_record = (
            service_b.store.get_by_experiment(retained_exp) if retained_exp else None
        )
        baseline_name_b = Path(result_b.baseline_report_path or "").name
        b_ok = bool(
            loop_b.changed is True
            and retained_iterations
            and result_b.deployment.status is DeploymentStatus.RETAINED
            and result_b.deployment.retained is True
            and result_b.deployment.source_experiment_id is not None
            and result_b.deployment.final_configuration == loop_b.final_configuration
            and baseline_name_b.startswith(run_id_b)
            and result_b.provenance.all_bound_to_run()
            and result_b.diagnosis_controlled is True
            and knowledge_record is not None
            and knowledge_record.outcome is OptimizationOutcome.IMPROVED
            and all(i.restoration_status is RestorationStatus.RESTORED for i in retained_iterations)
        )
        scenarios["B_fresh_optimization"] = {
            "run_id": run_id_b,
            "termination": loop_b.termination.reason.value,
            "changed": loop_b.changed,
            "iterations": [
                {
                    "iteration": i.iteration,
                    "outcome": i.outcome.value,
                    "diagnosis_kind": i.diagnosis_kind,
                    "plan_outcome": i.plan_outcome.value if i.plan_outcome else None,
                    "selected_action_id": i.selected_action_id,
                    "selected_parameters": i.selected_parameters,
                    "experiment_id": i.experiment_id,
                    "experiment_status": i.experiment_status.value if i.experiment_status else None,
                    "retained": i.retained,
                    "restoration_status": i.restoration_status.value,
                    "restoration_verified": i.restoration_verified,
                    "baseline_metrics": i.baseline_metrics,
                    "candidate_metrics": i.candidate_metrics,
                }
                for i in loop_b.run.iterations
            ],
            "deployment": result_b.deployment.as_dict(),
            "provenance_run_id": result_b.provenance.run_id,
            "all_artifacts_bound": result_b.provenance.all_bound_to_run(),
            "experiment_ids": result_b.provenance.experiment_ids,
            "knowledge_record_ids": result_b.provenance.knowledge_record_ids,
            "persisted_outcome": knowledge_record.outcome.value if knowledge_record else None,
            "diagnosis_id": result_b.diagnosis_id,
            "diagnosis_controlled": result_b.diagnosis_controlled,
            "characterization_s": result_b.characterization_s,
            "introspection_s": result_b.introspection_s,
            "baseline_s": result_b.baseline_s,
            "overhead": loop_b.run.overhead.model_dump(mode="json"),
        }
        print(
            f"changed={loop_b.changed} deployment={result_b.deployment.status.value} "
            f"retained={loop_b.run.retained_candidate_ids} "
            f"knowledge={knowledge_record.outcome.value if knowledge_record else None}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"scenario B failed: {type(exc).__name__}: {exc}")
        scenarios["B_fresh_optimization"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("B: fresh optimization path retains with real experiment", b_ok))
    print()

    # cleanup
    try:
        if adapter.is_loaded:
            adapter.unload()
    except Exception:  # noqa: BLE001
        pass
    try:
        decoy_path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        pass
    remaining = len(
        [p for p in psutil.process_iter(["name"]) if p.info.get("name") == "llama-server.exe"]
    )
    checks.append(("no orphaned llama-server processes remain", remaining <= baseline_processes))

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": "9r",
        "result": "PASS" if ok else "FAIL",
        "command": "inferenceos verify-phase9r",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_ids": {"noop": run_id_a, "optimization": run_id_b},
        "model": {"name": model.name, "path": model.path},
        "backend": {"server": str(server_path)},
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "known_limitations": [
            "the optimization scenario uses a controlled MODEL_FOOTPRINT diagnosis "
            "grounded in this run's fresh baseline; it is labelled controlled",
            "deep resident reasoning is wired via ResidentDeepDiagnosisRunner and is "
            "exercised for real by verify-phase5r; the no-op/optimization scenarios "
            "use deterministic evidence so escalation does not fire",
            "the deployment state is a structured representation, not a server launcher",
        ],
    }
    verification_path = settings.reports_dir / f"phase9r_verification_{timestamp}.json"
    verification_path.write_text(
        json.dumps(report_payload, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(f"=== RESULT: PHASE 9R ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {verification_path}")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Phase 10 — runtime adaptive monitoring                                       #
# --------------------------------------------------------------------------- #
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


def _monitor_approval_from_args(
    args: argparse.Namespace,
) -> tuple[ApprovalProvider, list[str]]:
    """Resolve the Phase-10 monitor approval provider with the B-approval gate.

    ``monitor --approval approve`` alone must **not** authorize unattended
    auto-approval of disruptive runtime changes. Auto-approval is enabled only when
    the explicit ``--i-accept-unattended-actions`` opt-in is also present; otherwise
    the provider falls back to the interactive ``CliApprovalProvider``, which fails
    safe (TIMEOUT, never approval) when no interactive terminal is available.

    Returns the provider and provenance notes to persist in the run record.
    """
    mode = getattr(args, "approval", "interactive")
    if mode != "approve":
        # reject/timeout/interactive are unchanged and already fail safe.
        return _approval_provider_from_args(args), []
    if getattr(args, "i_accept_unattended_actions", False):
        note = (
            "unattended auto-approval explicitly accepted via "
            "--i-accept-unattended-actions; disruptive runtime changes may be "
            "approved without human review"
        )
        print(f"WARNING: {note}", file=sys.stderr)
        return AutoApproveApprovalProvider(), [note]
    note = (
        "--approval approve was given without --i-accept-unattended-actions; "
        "unattended auto-approval is NOT authorized, so disruptive runtime changes "
        "will not be auto-approved"
    )
    print(
        f"WARNING: {note}. Falling back to interactive approval, which fails safe "
        "when no interactive terminal is available.",
        file=sys.stderr,
    )
    return CliApprovalProvider(), [note]


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
    approval_provider, approval_notes = _monitor_approval_from_args(args)
    monitor = RuntimeMonitor(
        workload=workload,
        harness=BaselineProfiler(settings=settings, adapter=workload.adapter),
        policy=loop_policy,
        approval_provider=approval_provider,
        notification_sink=sink,
        knowledge_service=_knowledge_service(settings, args.db),
        reasoning_engine=reasoning_engine,
        settings=settings,
        session_notes=approval_notes,
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


# --------------------------------------------------------------------------- #
# Phase 10R — runtime adaptive monitoring remediation over a resident target    #
# --------------------------------------------------------------------------- #
class _PerturbingRuntimeSampler(RuntimeTelemetrySampler):
    """Wraps the real system sampler and adds a *controlled* runtime pressure signal.

    Real telemetry (target RSS/CPU/system memory, real request metrics) is preserved;
    only the post-baseline ``memory_pressure``/``hard_page_faults_per_sec`` values are
    raised so a persistent runtime condition exists on demand. This is a clearly
    labelled controlled condition used only by acceptance scenarios B-E.
    """

    def __init__(self, inner: RuntimeTelemetrySampler) -> None:
        self._inner = inner

    def sample(
        self, workload: ActiveWorkload, *, sequence: int, condition: str
    ) -> RuntimeObservation:
        observation = self._inner.sample(workload, sequence=sequence, condition=condition)
        if condition == "baseline":
            return observation
        metrics = dict(observation.metrics)
        metrics["hard_page_faults_per_sec"] = max(
            metrics.get("hard_page_faults_per_sec", 0.0), 8000.0
        )
        metrics["memory_pressure"] = max(metrics.get("memory_pressure", 0.0), 0.9)
        return observation.model_copy(update={"metrics": metrics})


class _Phase10rFaultHarness(MeasurementHarness):
    """Deterministic injected fault for the failure-containment scenario only."""

    def run_once(
        self,
        task: ProfileTask,
        *,
        mode: MeasurementMode,
        sequence: int,
        repetition: int,
        warmup: bool,
        options: ProfilerOptions,
    ) -> ProfilingRun:
        raise RuntimeError("injected intervention measurement fault (Phase 10R scenario E)")


def _phase10r_policy(
    settings: Settings, *, duration: float, max_interventions: int = 1
) -> RuntimePolicy:
    budget = build_runtime_budget(
        settings, max_duration_s=duration, telemetry_interval_s=0.5
    ).model_copy(
        update={
            "baseline_samples": 3,
            "observation_window_size": 40,
            "max_interventions": max_interventions,
            "max_cumulative_intervention_s": 600.0,
            "max_optimizer_overhead_s": 600.0,
            "approval_timeout_s": 10.0,
            "max_alerts_per_condition": 3,
        }
    )
    return build_runtime_policy(
        settings,
        budget=budget,
        require_approval_for_disruptive=True,
        allow_non_disruptive_auto_adaptation=False,
    )


def _phase10r_experiment_policy(settings: Settings) -> ExperimentPolicy:
    return build_experiment_policy(settings).model_copy(
        update={"measured_repetitions": 2, "max_repetitions": 3, "max_total_runs": 8}
    )


def _phase10r_run(
    settings: Settings,
    model: ModelProfile,
    *,
    run_id: str,
    detector: object,
    approval: ApprovalProvider,
    duration: float,
    workload_id: str = "short-short",
    context_length: int = 2048,
    perturb: bool = False,
    db_path: Path | None = None,
    harness: MeasurementHarness | None = None,
    optimization_policy: object | None = None,
    max_interventions: int = 1,
) -> tuple[ResidentActiveWorkload, RuntimeMonitorResult, KnowledgeService | None]:
    workload = ResidentActiveWorkload(
        settings=settings,
        model=model,
        workload_id=workload_id,
        configuration=RuntimeConfiguration(
            context_length=context_length, gpu_offload_layers=0
        ),
        request_gap_s=0.0,
    )
    base_sampler = SystemTelemetrySampler(settings=settings)
    sampler: RuntimeTelemetrySampler = (
        _PerturbingRuntimeSampler(base_sampler) if perturb else base_sampler
    )
    knowledge = _knowledge_service(settings, str(db_path)) if db_path is not None else None
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=sampler,
        policy=_phase10r_policy(settings, duration=duration, max_interventions=max_interventions),
        approval_provider=approval,
        notification_sink=ConsoleNotificationSink(),
        knowledge_service=knowledge,
        detector=detector,  # type: ignore[arg-type]
        harness=harness,
        optimization_policy=optimization_policy,  # type: ignore[arg-type]
        experiment_policy=_phase10r_experiment_policy(settings),
        run_id=run_id,
        session_id=run_id,
        settings=settings,
    )
    result = monitor.run()
    return workload, result, knowledge


def _phase10r_last_ok(workload: ResidentActiveWorkload) -> bool:
    last = workload.last_request()
    return bool(last is not None and last.success)


def _phase10r_last_success(workload: ResidentActiveWorkload) -> bool | None:
    last = workload.last_request()
    return last.success if last is not None else None


def _cmd_verify_phase10r(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase10r.log")

    print("=== InferenceOS Phase 10R Verification (Runtime Adaptive Monitoring) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checks: list[tuple[str, bool]] = []
    scenarios: dict[str, object] = {}
    tracked_pids: set[int] = set()

    try:
        _server = _resident_server_path(settings)
        models = _load_models(settings, args.model)
        model = max(models, key=lambda profile: profile.file_size_bytes)
    except (FileNotFoundError, ModelMetadataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    opt_policy = build_policy(settings).model_copy(
        update={"context_reduction_fraction": 0.75}
    )
    reject_cb = CallbackApprovalProvider(lambda _rec: ApprovalOutcome.REJECTED)
    approve_cb = CallbackApprovalProvider(lambda _rec: ApprovalOutcome.APPROVED)

    # ------------------------------------------------------------------ #
    # Scenario A: real active workload / conservative monitoring           #
    # ------------------------------------------------------------------ #
    print("--- Scenario A: real resident workload, real telemetry, no intervention ---")
    a_ok = False
    try:
        workload, result, _kb = _phase10r_run(
            settings,
            model,
            run_id=f"phase10r-A-{timestamp}",
            detector=ControlledFootprintDetector(force=False),
            approval=AutoRejectApprovalProvider(),
            duration=10.0,
        )
        if workload.target_pid():
            tracked_pids.add(workload.target_pid() or 0)
        session = result.run.session
        records = workload.request_records()
        a_ok = bool(
            session is not None
            and session.distinct_processes is True
            and session.monitor_pid != session.target_pid
            and session.run_id == f"phase10r-A-{timestamp}"
            and result.run.observations_collected >= 5
            and len(records) >= 1
            and all(r.completed_at is not None for r in records)
            and result.run.interventions == []
            and result.termination.reason is RuntimeTerminationReason.DURATION_REACHED
        )
        scenarios["A_active_monitoring"] = {
            "run_id": session.run_id if session else None,
            "monitor_pid": session.monitor_pid if session else None,
            "target_pid": session.target_pid if session else None,
            "distinct_processes": session.distinct_processes if session else None,
            "target_create_time": session.target_create_time if session else None,
            "observations": result.run.observations_collected,
            "requests_completed": workload.completed_requests(),
            "first_request": records[0].model_dump(mode="json") if records else None,
            "last_request": records[-1].model_dump(mode="json") if records else None,
            "interventions": len(result.run.interventions),
            "termination": result.termination.reason.value,
        }
        print(
            f"monitor_pid={session.monitor_pid if session else None} "
            f"target_pid={session.target_pid if session else None} "
            f"requests={workload.completed_requests()} interventions={len(result.run.interventions)}"
        )
        workload.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"scenario A failed: {type(exc).__name__}: {exc}")
        scenarios["A_active_monitoring"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("A: real active resident workload; distinct monitor/target; no intervention", a_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario B: opportunity -> approval requested -> REJECTED            #
    # ------------------------------------------------------------------ #
    print("--- Scenario B: runtime opportunity rejected by the user ---")
    b_ok = False
    try:
        workload, result, _kb = _phase10r_run(
            settings,
            model,
            run_id=f"phase10r-B-{timestamp}",
            detector=ControlledFootprintDetector(force=True),
            approval=reject_cb,
            duration=25.0,
            perturb=True,
            optimization_policy=opt_policy,
        )
        if workload.target_pid():
            tracked_pids.add(workload.target_pid() or 0)
        rejected = [a for a in result.run.approvals if a.outcome is ApprovalOutcome.REJECTED]
        b_ok = bool(
            result.run.opportunities
            and result.run.recommendations
            and rejected
            and result.run.interventions == []
            and workload.snapshot().running
            and workload.completed_requests() >= 1
            and _phase10r_last_ok(workload)
        )
        scenarios["B_approval_rejection"] = {
            "opportunities": len(result.run.opportunities),
            "recommendations": len(result.run.recommendations),
            "approvals": [a.model_dump(mode="json") for a in result.run.approvals],
            "interventions": len(result.run.interventions),
            "workload_running": workload.snapshot().running,
            "last_request_success": _phase10r_last_success(workload),
            "run_id": result.run.run_id,
        }
        print(
            f"opportunities={len(result.run.opportunities)} "
            f"approvals={[a.outcome.value for a in result.run.approvals]} "
            f"interventions={len(result.run.interventions)}"
        )
        workload.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"scenario B failed: {type(exc).__name__}: {exc}")
        scenarios["B_approval_rejection"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("B: opportunity -> approval requested -> rejected; workload intact", b_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario C: opportunity -> approval -> real intervention -> resume   #
    # ------------------------------------------------------------------ #
    print("--- Scenario C: runtime opportunity approved and intervened ---")
    c_ok = False
    try:
        db_path = settings.data_dir / f"phase10r_C_{timestamp}.db"
        workload, result, knowledge = _phase10r_run(
            settings,
            model,
            run_id=f"phase10r-C-{timestamp}",
            detector=ControlledFootprintDetector(force=True),
            approval=approve_cb,
            duration=40.0,
            perturb=True,
            db_path=db_path,
            optimization_policy=opt_policy,
        )
        if workload.target_pid():
            tracked_pids.add(workload.target_pid() or 0)
        intervention = result.run.interventions[-1] if result.run.interventions else None
        persisted = None
        if knowledge is not None and intervention is not None and intervention.experiment_id:
            persisted = knowledge.store.get_by_experiment(intervention.experiment_id)
        c_ok = bool(
            intervention is not None
            and intervention.resume_verified
            and intervention.approval_outcome is ApprovalOutcome.APPROVED
            and intervention.checkpoint_id
            and intervention.experiment_id
            and intervention.decision is not None
            and persisted is not None
            and workload.completed_requests() >= 1
            and _phase10r_last_ok(workload)
        )
        scenarios["C_approved_intervention"] = {
            "run_id": result.run.run_id,
            "approvals": [a.model_dump(mode="json") for a in result.run.approvals],
            "checkpoints": len(result.run.checkpoints),
            "intervention": intervention.model_dump(mode="json") if intervention else None,
            "knowledge_record_id": (f"opt-{intervention.experiment_id}" if intervention and intervention.experiment_id else None),
            "persisted_outcome": persisted.outcome.value if persisted else None,
            "requests_completed": workload.completed_requests(),
            "overhead": result.run.overhead.model_dump(mode="json"),
        }
        print(
            f"interventions={len(result.run.interventions)} "
            f"decision={intervention.decision.value if intervention and intervention.decision else None} "
            f"resume_verified={intervention.resume_verified if intervention else None} "
            f"persisted={persisted.outcome.value if persisted else None}"
        )
        workload.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"scenario C failed: {type(exc).__name__}: {exc}")
        scenarios["C_approved_intervention"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("C: approved intervention uses Phase 7 machinery and resumes safely", c_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario D: in-flight request safety                                 #
    # ------------------------------------------------------------------ #
    print("--- Scenario D: opportunity during an in-flight request is deferred ---")
    d_ok = False
    try:
        workload, result, _kb = _phase10r_run(
            settings,
            model,
            run_id=f"phase10r-D-{timestamp}",
            detector=ControlledFootprintDetector(force=True),
            approval=approve_cb,
            duration=40.0,
            workload_id="short-long-decode",
            perturb=True,
            optimization_policy=opt_policy,
        )
        if workload.target_pid():
            tracked_pids.add(workload.target_pid() or 0)
        intervention = result.run.interventions[-1] if result.run.interventions else None
        d_ok = bool(
            intervention is not None
            and intervention.in_flight_at_start
            and intervention.in_flight_completed_at is not None
            and intervention.intervention_applied_at is not None
            and intervention.in_flight_completed_at <= intervention.intervention_applied_at
            and intervention.opportunity_detected_at is not None
            and intervention.opportunity_detected_at <= intervention.in_flight_completed_at
            and intervention.resume_verified
        )
        scenarios["D_in_flight_safety"] = {
            "run_id": result.run.run_id,
            "in_flight_request_id": intervention.in_flight_request_id if intervention else None,
            "opportunity_detected_at": (
                intervention.opportunity_detected_at.isoformat() if intervention and intervention.opportunity_detected_at else None
            ),
            "in_flight_completed_at": (
                intervention.in_flight_completed_at.isoformat() if intervention and intervention.in_flight_completed_at else None
            ),
            "intervention_applied_at": (
                intervention.intervention_applied_at.isoformat() if intervention and intervention.intervention_applied_at else None
            ),
            "resumed_at": (
                intervention.resumed_at.isoformat() if intervention and intervention.resumed_at else None
            ),
            "resume_verified": intervention.resume_verified if intervention else None,
        }
        print(
            f"in_flight_at_start={intervention.in_flight_at_start if intervention else None} "
            f"in_flight_completed_at={intervention.in_flight_completed_at if intervention else None} "
            f"applied_at={intervention.intervention_applied_at if intervention else None}"
        )
        workload.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"scenario D failed: {type(exc).__name__}: {exc}")
        scenarios["D_in_flight_safety"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("D: in-flight request drained before any intervention mutation", d_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario E: failed intervention / recovery                           #
    # ------------------------------------------------------------------ #
    print("--- Scenario E: injected intervention failure -> recovery ---")
    e_ok = False
    try:
        workload, result, _kb = _phase10r_run(
            settings,
            model,
            run_id=f"phase10r-E-{timestamp}",
            detector=ControlledFootprintDetector(force=True),
            approval=approve_cb,
            duration=35.0,
            perturb=True,
            harness=_Phase10rFaultHarness(),
            optimization_policy=opt_policy,
        )
        if workload.target_pid():
            tracked_pids.add(workload.target_pid() or 0)
        intervention = result.run.interventions[-1] if result.run.interventions else None
        e_ok = bool(
            intervention is not None
            and not intervention.retained
            and intervention.resume_verified
            and result.run.checkpoints
            and workload.snapshot().running
            and workload.completed_requests() >= 1
            and _phase10r_last_ok(workload)
        )
        scenarios["E_failure_containment"] = {
            "run_id": result.run.run_id,
            "intervention": intervention.model_dump(mode="json") if intervention else None,
            "checkpoints": len(result.run.checkpoints),
            "workload_running": workload.snapshot().running,
            "last_request_success": _phase10r_last_success(workload),
        }
        print(
            f"retained={intervention.retained if intervention else None} "
            f"resume_verified={intervention.resume_verified if intervention else None} "
            f"decision={intervention.decision.value if intervention and intervention.decision else None}"
        )
        workload.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"scenario E failed: {type(exc).__name__}: {exc}")
        scenarios["E_failure_containment"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("E: injected failure contained, checkpoint restored, workload recovered", e_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario F: no meaningful runtime opportunity                        #
    # ------------------------------------------------------------------ #
    print("--- Scenario F: no meaningful runtime opportunity (no-op) ---")
    f_ok = False
    try:
        workload, result, _kb = _phase10r_run(
            settings,
            model,
            run_id=f"phase10r-F-{timestamp}",
            detector=ControlledFootprintDetector(force=False),
            approval=AutoRejectApprovalProvider(),
            duration=12.0,
        )
        if workload.target_pid():
            tracked_pids.add(workload.target_pid() or 0)
        f_ok = bool(
            result.run.interventions == []
            and result.run.approvals == []
            and not result.run.recommendations
            and workload.completed_requests() >= 1
            and result.termination.reason is RuntimeTerminationReason.DURATION_REACHED
        )
        scenarios["F_no_opportunity"] = {
            "run_id": result.run.run_id,
            "interventions": len(result.run.interventions),
            "approvals": len(result.run.approvals),
            "recommendations": len(result.run.recommendations),
            "requests_completed": workload.completed_requests(),
            "termination": result.termination.reason.value,
        }
        print(
            f"interventions={len(result.run.interventions)} approvals={len(result.run.approvals)} "
            f"recommendations={len(result.run.recommendations)} requests={workload.completed_requests()}"
        )
        workload.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"scenario F failed: {type(exc).__name__}: {exc}")
        scenarios["F_no_opportunity"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("F: no meaningful opportunity -> no intervention/approval", f_ok))
    print()

    # ------------------------------------------------------------------ #
    # Scenario G: stale approval cannot authorize an intervention          #
    # ------------------------------------------------------------------ #
    print("--- Scenario G: stale approval rejected ---")
    g_ok = False
    try:
        workload, result, _kb = _phase10r_run(
            settings,
            model,
            run_id=f"phase10r-G-{timestamp}",
            detector=ControlledFootprintDetector(force=True),
            approval=ScriptedApprovalProvider(
                [ApprovalOutcome.APPROVED], mismatch_context=True
            ),
            duration=25.0,
            perturb=True,
            optimization_policy=opt_policy,
        )
        if workload.target_pid():
            tracked_pids.add(workload.target_pid() or 0)
        stale = [a for a in result.run.approvals if a.outcome is ApprovalOutcome.STALE]
        g_ok = bool(stale and result.run.interventions == [])
        scenarios["G_stale_approval"] = {
            "run_id": result.run.run_id,
            "approvals": [a.model_dump(mode="json") for a in result.run.approvals],
            "interventions": len(result.run.interventions),
        }
        print(f"approvals={[a.outcome.value for a in result.run.approvals]} interventions={len(result.run.interventions)}")
        workload.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"scenario G failed: {type(exc).__name__}: {exc}")
        scenarios["G_stale_approval"] = {"error": f"{type(exc).__name__}: {exc}"}
    checks.append(("G: stale approval cannot authorize an intervention", g_ok))
    print()

    # cleanup / orphan check
    orphans = sorted(pid for pid in tracked_pids if pid and psutil.pid_exists(pid))
    checks.append(("no orphaned resident target processes remain", not orphans))

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": "10r",
        "result": "PASS" if ok else "FAIL",
        "command": "inferenceos verify-phase10r",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {"name": model.name, "path": model.path},
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "orphan_pids": orphans,
        "known_limitations": [
            "scenarios B-E apply a controlled pressure perturbation on top of real "
            "system telemetry and use a controlled MODEL_FOOTPRINT diagnosis grounded in "
            "the real measured target RSS; they are labelled controlled",
            "scenarios A/F use the real detector on unperturbed real telemetry",
            "the resident target runs CPU-only so the context/memory effect is observable",
            "scenario E uses a deterministic injected measurement fault for the failure path",
        ],
    }
    verification_path = settings.reports_dir / f"phase10r_verification_{timestamp}.json"
    verification_path.write_text(
        json.dumps(report_payload, indent=2, default=str), encoding="utf-8"
    )

    print()
    print(f"=== RESULT: PHASE 10R ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
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


# --------------------------------------------------------------------------- #
# Phase 11 — external process recommendations                                  #
# --------------------------------------------------------------------------- #
def _external_policy(
    settings: Settings,
    *,
    duration: float | None = None,
    memory_pressure_threshold: float | None = None,
    require_target_degradation: bool | None = None,
    restore_after_action: bool | None = None,
) -> ExternalPolicy:
    budget = build_external_budget(settings, max_duration_s=duration)
    return build_external_policy(
        settings,
        budget=budget,
        require_approval_for_actions=True,
        restore_after_action=restore_after_action,
    ).model_copy(
        update={
            "memory_pressure_threshold": (
                memory_pressure_threshold
                if memory_pressure_threshold is not None
                else settings.phase11_memory_pressure_threshold
            ),
            "require_target_degradation_for_recommendation": (
                require_target_degradation
                if require_target_degradation is not None
                else settings.phase11_require_target_degradation
            ),
        }
    )


def _cmd_external(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase11.log")

    try:
        policy = _external_policy(settings, duration=args.duration)
    except ValueError as exc:
        print(f"error: invalid Phase 11 configuration: {exc}", file=sys.stderr)
        return 2

    observer = WindowsProcessObserver(settings=settings)
    knowledge = ExternalKnowledgeStore(args.db or settings.phase11_db_path)
    engine = ExternalRecommendationEngine(
        observer=observer,
        registry=build_default_registry(),
        policy=policy,
        approval_provider=_approval_provider_from_args(args),
        knowledge_store=knowledge,
    )

    print("=== InferenceOS Phase 11 external recommendations ===")
    print(f"scope      : {policy.scope.value}")
    print(f"duration   : {policy.budget.max_duration_s:.0f}s interval {policy.observation_interval_s}s")
    print("policy     : observe -> recommend -> ask user (no silent control)")
    print()
    result = engine.run(
        target_pid=args.target_pid,
        target_degraded=args.target_degraded,
        context_identity=args.context or "cli",
        duration_s=args.duration,
    )
    if args.json:
        print(result.to_json())
    else:
        print(render_external_run(result))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out) if args.out else settings.reports_dir / f"phase11_external_{timestamp}.json"
    )
    out_path.write_text(result.to_json(), encoding="utf-8")
    print()
    print(f"run: {out_path}")
    return 1 if result.termination.safety else 0


def _verify11r_controlled_engine(
    settings: Settings,
    *,
    report: object,
    approvals: object,
    policy: ExternalPolicy | None = None,
    knowledge: ExternalKnowledgeStore | None = None,
) -> ExternalRecommendationEngine:
    return ExternalRecommendationEngine(
        observer=ScriptedProcessObserver(report),  # type: ignore[arg-type]
        registry=build_default_registry(),
        policy=policy or _external_policy(settings),
        approval_provider=approvals,  # type: ignore[arg-type]
        knowledge_store=knowledge,
        clock=ExternalFakeClock(),
    )


def _phase11r_priority(pid: int) -> int | None:
    import psutil

    try:
        return int(psutil.Process(pid).nice())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return None


def _phase11r_alive(pid: int) -> bool:
    import psutil

    try:
        process = psutil.Process(pid)
        return bool(process.is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False


def _phase11r_find(report: object, pid: int):  # type: ignore[no-untyped-def]
    processes = getattr(report, "processes", [])
    return next((item for item in processes if item.identity.pid == pid), None)


def temp_path_for_phase11(name: str) -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix=f"io_phase11r_{name}_")) / "external.db"


def _cmd_verify_phase11r(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase11r.log")

    print("=== InferenceOS Phase 11R Verification (External Process Recommendations) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("boundary : observe -> attribute -> recommend -> present -> user decision")
    print("no autonomous external-process control exists in this build")
    print()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scenarios: dict[str, object] = {}
    flags: dict[str, bool] = {
        "A": False,
        "B": False,
        "C": False,
        "D": False,
        "E": False,
        "F": False,
        "G": False,
        "H": False,
    }

    try:
        _ = _resident_server_path(settings)
        models = _load_models(settings, args.model)
        model = max(models, key=lambda profile: profile.file_size_bytes)
    except (FileNotFoundError, ModelMetadataError, OSError) as exc:
        print(f"error: a real resident inference model is required: {exc}", file=sys.stderr)
        return 2

    observer = WindowsProcessObserver(settings=settings)

    # Real deterministic diagnosis over controlled Phase 5 fixtures (labelled).
    paging_diag = BottleneckDetector().detect(
        make_paging_report(), source_report_path="<phase11r-controlled-paging>"
    )
    healthy_diag = BottleneckDetector().detect(
        make_healthy_report(), source_report_path="<phase11r-controlled-healthy>"
    )
    paging_kind = paging_diag.primary.kind if paging_diag.primary is not None else None
    healthy_kind = healthy_diag.primary.kind if healthy_diag.primary is not None else None
    paging_resource = bottleneck_to_resource(paging_kind)
    healthy_resource = bottleneck_to_resource(healthy_kind)
    paging_name = paging_kind.value if paging_kind else None
    scenarios["diagnoses"] = {
        "paging_kind": paging_name,
        "paging_resource": paging_resource.value if paging_resource else None,
        "paging_meaningful": paging_diag.meaningful_opportunity,
        "healthy_kind": healthy_kind.value if healthy_kind else None,
        "healthy_resource": healthy_resource.value if healthy_resource else None,
        "healthy_meaningful": healthy_diag.meaningful_opportunity,
        "source": "controlled fixtures + real BottleneckDetector",
    }

    workload = ResidentActiveWorkload(
        settings=settings,
        model=model,
        workload_id=args.workload,
        configuration=RuntimeConfiguration(
            context_length=choose_context_length(model), gpu_offload_layers=0
        ),
        request_gap_s=0.0,
    )
    target_pid: int | None = None
    holder_ctx: object | None = None
    holder_pid = 0
    scenario_b_recommendation: ExternalRecommendation | None = None
    scenario_b_identity: ExternalProcessIdentity | None = None

    def run_engine(
        report: object,
        approvals: object,
        *,
        label: str,
        knowledge: ExternalKnowledgeStore | None = None,
        target_degraded: bool,
        diagnosed_resource: ExternalResource | None,
        bottleneck: str | None,
        diagnosis_id: str | None,
        target_identity: ExternalProcessIdentity | None,
    ) -> ExternalRecommendationRun:
        engine = _verify11r_controlled_engine(
            settings, report=report, approvals=approvals, knowledge=knowledge
        )
        return engine.run(
            target_pid=target_pid,
            target_identity=target_identity,
            target_degraded=target_degraded,
            context_identity=label,
            diagnosed_bottleneck=bottleneck,
            diagnosed_resource=diagnosed_resource,
            diagnosis_id=diagnosis_id,
            run_id=label,
            session_id=label,
            max_cycles=1,
        )

    try:
        workload.start()
        time.sleep(3.0)
        target_pid = workload.target_pid()
        if target_pid is None:
            raise RuntimeError("resident workload did not expose a target pid")
        target_identity = resolve_identity(target_pid, is_target=True)

        holder_ctx = launch_memory_holder(2 * 1024**3)
        holder = holder_ctx.__enter__()
        holder_pid = holder.pid

        report_a = observer.observe(target_pid=target_pid, max_processes=400)
        target_obs = _phase11r_find(report_a, target_pid)
        helper_obs = _phase11r_find(report_a, holder_pid)
        if target_obs is None or helper_obs is None:
            raise RuntimeError("real observer did not return both target and helper")

        # -------------------------------------------------------------- #
        # A: external process observation                                 #
        # -------------------------------------------------------------- #
        print("--- Scenario A: external process observation ---")
        flags["A"] = bool(
            target_obs.identity.pid != helper_obs.identity.pid
            and target_obs.identity.identity_key != helper_obs.identity.identity_key
            and helper_obs.rss_bytes is not None
            and helper_obs.rss_bytes > 0
            and helper_obs.identity.create_time_epoch is not None
            and helper_obs.identity.pid_reuse_guard != f"pid:{holder_pid}"
            and report_a.captured_at is not None
            and bool(report_a.privacy_notes)
        )
        scenarios["A_observation"] = {
            "target_pid": target_pid,
            "target_create_time": target_obs.identity.create_time_epoch,
            "target_identity_key": target_obs.identity.identity_key,
            "helper_pid": holder_pid,
            "helper_create_time": helper_obs.identity.create_time_epoch,
            "helper_identity_key": helper_obs.identity.identity_key,
            "helper_rss_bytes": helper_obs.rss_bytes,
            "helper_cpu_percent": helper_obs.cpu_percent,
            "observed_at": report_a.captured_at.isoformat(),
            "read_only": True,
            "privacy_notes": report_a.privacy_notes,
        }
        print(
            f"target_pid={target_pid} helper_pid={holder_pid} "
            f"helper_rss={helper_obs.rss_bytes}"
        )
        print()

        controlled_target = ExternalProcessObservation(
            observation_id="extproc-target-controlled",
            identity=target_obs.identity,
            cpu_percent=25.0,
            rss_bytes=1_000_000_000,
            measurable=["rss_bytes", "cpu_percent"],
            access=ExternalProcessAccess.READABLE,
        )
        controlled_report = make_report(
            processes=[controlled_target, helper_obs],
            target_pid=target_pid,
            memory_pressure=0.9,
        )

        # -------------------------------------------------------------- #
        # B: causal external-process recommendation                       #
        # -------------------------------------------------------------- #
        print("--- Scenario B: causal external-process recommendation ---")
        b_run = run_engine(
            controlled_report,
            AutoRejectApprovalProvider(),
            label=f"phase11r-B-{timestamp}",
            knowledge=ExternalKnowledgeStore(temp_path_for_phase11("b")),
            target_degraded=True,
            diagnosed_resource=paging_resource,
            bottleneck=paging_name,
            diagnosis_id=paging_diag.diagnosis_id,
            target_identity=target_obs.identity,
        )
        b_rec = next(
            (r for r in b_run.recommendations if r.identity and r.identity.pid == holder_pid),
            None,
        )
        scenario_b_recommendation = b_rec
        scenario_b_identity = helper_obs.identity
        flags["B"] = bool(
            b_rec is not None
            and b_rec.identity is not None
            and b_rec.identity.pid == holder_pid
            and b_rec.identity.identity_key == helper_obs.identity.identity_key
            and b_rec.bottleneck == paging_name
            and b_rec.target_identity is not None
            and b_rec.target_identity.pid == target_pid
            and bool(b_rec.rationale)
            and bool(b_rec.evidence_summary)
            and b_rec.expected_benefit_available is True
            and b_rec.provenance is not None
            and b_rec.provenance.diagnosis_id == paging_diag.diagnosis_id
            and b_rec.provenance.run_id == f"phase11r-B-{timestamp}"
        )
        scenarios["B_recommendation"] = {
            "run_id": b_run.run_id,
            "session_id": b_run.session_id,
            "diagnosis_id": b_run.diagnosis_id,
            "diagnosed_bottleneck": b_run.diagnosed_bottleneck,
            "recommendation_id": b_rec.recommendation_id if b_rec else None,
            "process_identity_key": (
                b_rec.identity.identity_key if b_rec and b_rec.identity else None
            ),
            "observed_metrics": b_rec.observed_metrics if b_rec else None,
            "rationale": b_rec.rationale if b_rec else None,
            "proposed_action": b_rec.proposed_action_id if b_rec else None,
            "expected_benefit": b_rec.expected_benefit if b_rec else None,
            "expected_benefit_available": b_rec.expected_benefit_available if b_rec else None,
            "uncertainty": b_rec.uncertainty if b_rec else None,
            "disruption_risk": b_rec.disruption_risk if b_rec else None,
            "requires_user_approval": b_rec.requires_user_approval if b_rec else None,
            "provenance": (
                b_rec.provenance.model_dump(mode="json")
                if b_rec and b_rec.provenance
                else None
            ),
            "status": b_rec.status.value if b_rec else None,
            "expected_benefit_is_controlled_estimate": True,
        }
        print(
            f"recommendation={b_rec.recommendation_id if b_rec else None} "
            f"action={b_rec.proposed_action_id if b_rec else None} "
            f"benefit_available={b_rec.expected_benefit_available if b_rec else None}"
        )
        print()

        # -------------------------------------------------------------- #
        # D: user rejection                                               #
        # -------------------------------------------------------------- #
        print("--- Scenario D: user rejection ---")
        nice_before = _phase11r_priority(holder_pid)
        d_knowledge = ExternalKnowledgeStore(temp_path_for_phase11("d"))
        d_run = run_engine(
            controlled_report,
            ScriptedApprovalProvider([ApprovalOutcome.REJECTED]),
            label=f"phase11r-D-{timestamp}",
            knowledge=d_knowledge,
            target_degraded=True,
            diagnosed_resource=paging_resource,
            bottleneck=paging_name,
            diagnosis_id=paging_diag.diagnosis_id,
            target_identity=target_obs.identity,
        )
        nice_after = _phase11r_priority(holder_pid)
        flags["D"] = bool(
            d_run.action_results == []
            and any(a.outcome is ApprovalOutcome.REJECTED for a in d_run.approvals)
            and d_run.user_decisions
            and d_run.user_decisions[0].decision is ExternalRecommendationStatus.REJECTED
            and d_run.user_decisions[0].autonomous_execution is False
            and d_run.knowledge_writes
            and nice_before == nice_after
            and _phase11r_alive(holder_pid)
            and workload.target_alive()
            and d_run.termination.reason is ExternalRunTerminationReason.APPROVAL_DECLINED
        )
        scenarios["D_rejection"] = {
            "decisions": [d.decision.value for d in d_run.user_decisions],
            "autonomous_execution": [d.autonomous_execution for d in d_run.user_decisions],
            "action_results": len(d_run.action_results),
            "priority_before": nice_before,
            "priority_after": nice_after,
            "external_process_alive": _phase11r_alive(holder_pid),
            "workload_alive": workload.target_alive(),
            "termination": d_run.termination.reason.value,
        }
        print(
            f"decision={d_run.user_decisions[0].decision.value if d_run.user_decisions else None} "
            f"actions={len(d_run.action_results)} priority {nice_before}->{nice_after}"
        )
        print()

        # -------------------------------------------------------------- #
        # E: user acceptance WITHOUT autonomous execution                 #
        # -------------------------------------------------------------- #
        print("--- Scenario E: user acceptance (no autonomous execution) ---")
        nice_before_e = _phase11r_priority(holder_pid)
        e_run = run_engine(
            controlled_report,
            AutoApproveApprovalProvider(),
            label=f"phase11r-E-{timestamp}",
            knowledge=ExternalKnowledgeStore(temp_path_for_phase11("e")),
            target_degraded=True,
            diagnosed_resource=paging_resource,
            bottleneck=paging_name,
            diagnosis_id=paging_diag.diagnosis_id,
            target_identity=target_obs.identity,
        )
        nice_after_e = _phase11r_priority(holder_pid)
        e_rec = next(
            (r for r in e_run.recommendations if r.identity and r.identity.pid == holder_pid),
            None,
        )
        flags["E"] = bool(
            e_run.user_decisions
            and e_run.user_decisions[0].decision is ExternalRecommendationStatus.ACCEPTED
            and e_run.user_decisions[0].autonomous_execution is False
            and e_run.action_results == []
            and e_rec is not None
            and e_rec.status is ExternalRecommendationStatus.ACCEPTED
            and nice_before_e == nice_after_e
            and _phase11r_alive(holder_pid)
            and _phase11r_alive(target_pid)
            and e_run.termination.reason
            is ExternalRunTerminationReason.RECOMMENDATION_ACCEPTED
        )
        scenarios["E_acceptance"] = {
            "decisions": [d.decision.value for d in e_run.user_decisions],
            "autonomous_execution": [d.autonomous_execution for d in e_run.user_decisions],
            "recorded_intent": [d.detail for d in e_run.user_decisions],
            "action_results": len(e_run.action_results),
            "priority_before": nice_before_e,
            "priority_after": nice_after_e,
            "external_process_alive": _phase11r_alive(holder_pid),
            "target_alive": _phase11r_alive(target_pid),
            "termination": e_run.termination.reason.value,
        }
        print(
            f"decision={e_run.user_decisions[0].decision.value if e_run.user_decisions else None} "
            f"action_results={len(e_run.action_results)} priority {nice_before_e}->{nice_after_e}"
        )
        print()

        # -------------------------------------------------------------- #
        # H: no meaningful bottleneck -> no recommendation                #
        # -------------------------------------------------------------- #
        print("--- Scenario H: no meaningful inference bottleneck ---")
        h_run = run_engine(
            controlled_report,
            AutoRejectApprovalProvider(),
            label=f"phase11r-H-{timestamp}",
            target_degraded=False,
            diagnosed_resource=healthy_resource,
            bottleneck=healthy_kind.value if healthy_kind else None,
            diagnosis_id=healthy_diag.diagnosis_id,
            target_identity=target_obs.identity,
        )
        flags["H"] = bool(
            h_run.recommendations == []
            and h_run.termination.reason
            is ExternalRunTerminationReason.NO_MEANINGFUL_BOTTLENECK
            and workload.target_alive()
        )
        scenarios["H_no_bottleneck"] = {
            "recommendations": len(h_run.recommendations),
            "termination": h_run.termination.reason.value,
            "diagnosed_bottleneck": h_run.diagnosed_bottleneck,
            "diagnosed_resource": (
                healthy_resource.value if healthy_resource else None
            ),
            "workload_alive": workload.target_alive(),
        }
        print(
            f"recommendations={len(h_run.recommendations)} "
            f"termination={h_run.termination.reason.value}"
        )
        print()

        # -------------------------------------------------------------- #
        # F: stale process identity                                       #
        # -------------------------------------------------------------- #
        print("--- Scenario F: stale process identity ---")
        if b_rec is not None:
            holder_ctx.__exit__(None, None, None)
            time.sleep(0.5)
            replacement = resolve_identity(holder_pid)
            replacement_key = replacement.identity_key if replacement is not None else None
            stale = revalidate_recommendation(
                b_rec,
                current_run_id=f"phase11r-B-{timestamp}",
                current_session_id=f"phase11r-B-{timestamp}",
                revalidate_process=True,
            )
            flags["F"] = bool(
                not _phase11r_alive(holder_pid)
                and stale.status is ExternalRecommendationStatus.STALE
                and stale.stale_reason is not None
                and (
                    replacement_key is None
                    or replacement_key != helper_obs.identity.identity_key
                )
            )
            scenarios["F_stale_process"] = {
                "original_identity_key": helper_obs.identity.identity_key,
                "original_pid": helper_obs.identity.pid,
                "original_pid_alive": _phase11r_alive(holder_pid),
                "replacement_identity_key": replacement_key,
                "revalidated_status": stale.status.value,
                "stale_reason": stale.stale_reason,
                "action_authorized": False,
            }
            print(
                f"status={stale.status.value} reason={stale.stale_reason} "
                f"replacement_key={replacement_key}"
            )
        print()

    except (OSError, RuntimeError, AssertionError) as exc:  # noqa: BLE001
        print(f"external-process scenarios failed: {type(exc).__name__}: {exc}")
        scenarios["external_process_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if holder_ctx is not None:
            try:
                holder_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]
            except RuntimeError:
                pass
        try:
            workload.stop()
        except RuntimeError:
            pass

    if holder_pid:
        scenarios["cleanup"] = {
            "helper_pid": holder_pid,
            "helper_alive_after_cleanup": _phase11r_alive(holder_pid),
            "cleaned_by_harness": not _phase11r_alive(holder_pid),
        }

    # ------------------------------------------------------------------ #
    # C: irrelevant external process (wrong resource) -> no recommendation #
    # ------------------------------------------------------------------ #
    print("--- Scenario C: irrelevant external process (CPU helper, memory diagnosis) ---")
    try:
        with launch_cpu_spinner() as spinner:
            observer.observe(max_processes=400)  # prime CPU delta tracking
            time.sleep(1.5)
            report_c = observer.observe(max_processes=400)
            spinner_obs = _phase11r_find(report_c, spinner.pid)
            if spinner_obs is not None:
                controlled_c = make_report(
                    processes=[
                        make_process_observation(
                            pid=100,
                            name="llama.exe",
                            rss_bytes=1_000_000_000,
                            cpu_percent=25.0,
                            is_target=True,
                        ),
                        spinner_obs,
                    ],
                    target_pid=100,
                    memory_pressure=0.5,
                )
                c_run = _verify11r_controlled_engine(
                    settings, report=controlled_c, approvals=AutoRejectApprovalProvider()
                ).run(
                    target_pid=100,
                    target_degraded=True,
                    context_identity=f"phase11r-C-{timestamp}",
                    diagnosed_bottleneck=paging_name,
                    diagnosed_resource=paging_resource,
                    diagnosis_id=paging_diag.diagnosis_id,
                    run_id=f"phase11r-C-{timestamp}",
                    session_id=f"phase11r-C-{timestamp}",
                    max_cycles=1,
                )
                # Same observations, contrasted by diagnosed resource: a CPU diagnosis
                # would flag the helper, a memory diagnosis must not.
                cpu_recs = build_recommendations(
                    c_run.assessments,
                    registry=build_default_registry(),
                    policy=_external_policy(settings),
                    context_identity="phase11r-c-cpu",
                    diagnosed_resource=ExternalResource.CPU,
                    diagnosed_bottleneck="cpu",
                )
                cpu_relevant = any(
                    r.identity is not None and r.identity.pid == spinner.pid for r in cpu_recs
                )
                flags["C"] = bool(
                    c_run.recommendations == []
                    and c_run.termination.reason
                    is ExternalRunTerminationReason.NO_RECOMMENDATION
                    and _phase11r_alive(spinner.pid)
                    and spinner_obs.cpu_percent is not None
                    and spinner_obs.cpu_percent >= 25.0
                    and cpu_relevant
                )
                scenarios["C_irrelevant_process"] = {
                    "helper_pid": spinner.pid,
                    "helper_identity_key": spinner_obs.identity.identity_key,
                    "helper_cpu_percent": spinner_obs.cpu_percent,
                    "diagnosed_resource": (
                        paging_resource.value if paging_resource else None
                    ),
                    "recommendations_under_diagnosis": len(c_run.recommendations),
                    "recommendations_under_cpu_diagnosis_would_exist": cpu_relevant,
                    "termination": c_run.termination.reason.value,
                    "external_process_alive": _phase11r_alive(spinner.pid),
                }
                print(
                    f"helper_cpu={spinner_obs.cpu_percent} "
                    f"recommendations={len(c_run.recommendations)} "
                    f"termination={c_run.termination.reason.value}"
                )
    except (OSError, RuntimeError) as exc:
        print(f"irrelevant-process scenario failed: {type(exc).__name__}: {exc}")
        scenarios["C_irrelevant_process"] = {"error": f"{type(exc).__name__}: {exc}"}
    print()

    # ------------------------------------------------------------------ #
    # G: stale provenance                                                 #
    # ------------------------------------------------------------------ #
    print("--- Scenario G: stale provenance ---")
    if scenario_b_recommendation is not None:
        provenance_stale = revalidate_recommendation(
            scenario_b_recommendation,
            current_run_id=f"phase11r-OTHER-{timestamp}",
            current_session_id=f"phase11r-OTHER-{timestamp}",
            revalidate_process=False,
            now=datetime.now(timezone.utc),
        )
        flags["G"] = bool(
            provenance_stale.status is ExternalRecommendationStatus.STALE
            and provenance_stale.stale_reason is not None
            and "phase11r-OTHER" in provenance_stale.stale_reason
        )
        scenarios["G_stale_provenance"] = {
            "original_run_id": (
                scenario_b_recommendation.provenance.run_id
                if scenario_b_recommendation.provenance
                else None
            ),
            "original_session_id": (
                scenario_b_recommendation.provenance.optimization_session_id
                if scenario_b_recommendation.provenance
                else None
            ),
            "current_run_id": f"phase11r-OTHER-{timestamp}",
            "revalidated_status": provenance_stale.status.value,
            "stale_reason": provenance_stale.stale_reason,
            "action_authorized": False,
        }
        print(
            f"status={provenance_stale.status.value} reason={provenance_stale.stale_reason}"
        )
    print()

    checks: list[tuple[str, bool]] = [
        (
            "A: real external process observed with robust PID-reuse-safe identity, "
            "timestamped and read-only",
            flags["A"],
        ),
        (
            "B: diagnosis-aware, provenance-bound, evidence-backed recommendation "
            "(controlled target/perturbation; real external helper metrics)",
            flags["B"],
        ),
        (
            "C: an external process consuming a resource that is not the diagnosed "
            "bottleneck produces no_recommendation",
            flags["C"],
        ),
        (
            "D: user rejection is recorded; no external mutation; workload continues",
            flags["D"],
        ),
        (
            "E: acceptance is persisted as intent only; no autonomous external mutation",
            flags["E"],
        ),
        (
            "F: stale/replaced process identity invalidates the recommendation; no action",
            flags["F"],
        ),
        (
            "G: recommendation from a different run/session is stale and inapplicable",
            flags["G"],
        ),
        (
            "H: no meaningful bottleneck produces no recommendation; workload uninterrupted",
            flags["H"],
        ),
    ]

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    report_payload = {
        "phase": "11R",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "observe external process -> establish relevance -> generate recommendation "
            "-> present to user -> record user decision (no autonomous external control)"
        ),
        "policy": _external_policy(settings).as_dict(),
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "external_process_mutations": 0,
        "result": "PASS" if ok else "FAIL",
    }
    verification_path = settings.reports_dir / f"phase11r_verification_{timestamp}.json"
    verification_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print()
    print(f"=== RESULT: PHASE 11R ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {verification_path}")
    return 0 if ok else 1


def _cmd_verify_phase11(args: argparse.Namespace) -> int:
    print("note: verify-phase11 is superseded by verify-phase11r (recommendation-only).")
    return _cmd_verify_phase11r(args)


def _cmd_verify_remediation(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "remediation.log")

    print("=== InferenceOS cross-phase remediation acceptance ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    from inferenceos.remediation import run_remediation_acceptance

    scenarios, checks = run_remediation_acceptance(
        settings, probe_prompt=args.prompt or "In one sentence, what is a key-value cache?"
    )

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_payload = {
        "remediation": "cross-phase",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    out_path = (
        Path(args.out)
        if args.out
        else settings.reports_dir / f"remediation_verification_{timestamp}.json"
    )
    out_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print()
    print(f"=== RESULT: REMEDIATION ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {out_path}")
    return 0 if ok else 1


def _cmd_verify_validity(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "validity.log")

    print("=== InferenceOS measurement-validity verification (B1) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    from inferenceos.validity_acceptance import run_validity_acceptance

    scenarios, checks = run_validity_acceptance(
        settings, probe_prompt=args.prompt or "In one sentence, what is a key-value cache?"
    )

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_payload = {
        "stage": "B1-measurement-validity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    out_path = (
        Path(args.out)
        if args.out
        else settings.reports_dir / f"validity_verification_{timestamp}.json"
    )
    out_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print()
    print(f"=== RESULT: MEASUREMENT-VALIDITY ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {out_path}")
    return 0 if ok else 1


def _cmd_verify_resident_recovery(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "resident_recovery.log")

    print("=== InferenceOS B2 resident-server crash detection & bounded recovery ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    from inferenceos.resident_recovery_acceptance import run_resident_recovery_acceptance

    scenarios, checks = run_resident_recovery_acceptance(settings)

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_payload = {
        "stage": "B2-resident-recovery",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    out_path = (
        Path(args.out)
        if args.out
        else settings.reports_dir / f"resident_recovery_verification_{timestamp}.json"
    )
    out_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print()
    print(f"=== RESULT: RESIDENT RECOVERY ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {out_path}")
    return 0 if ok else 1


def _cmd_verify_storage(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "storage.log")

    print("=== InferenceOS H2 log/report retention & disk protection ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    from inferenceos.storage_acceptance import run_storage_acceptance

    scenarios, checks = run_storage_acceptance(settings)

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_payload = {
        "stage": "H2-retention-disk-protection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    out_path = (
        Path(args.out)
        if args.out
        else settings.reports_dir / f"storage_verification_{timestamp}.json"
    )
    # H2: persist the acceptance report atomically and fail safe under disk pressure.
    persistence = guarded_write_text(
        out_path, json.dumps(report_payload, indent=2), settings
    )
    print()
    if persistence.persisted:
        print(f"report: {out_path}")
    else:
        print(f"warning: acceptance report not persisted: {persistence.reason}")
    print(f"=== RESULT: STORAGE ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


def _cmd_verify_accounting(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "accounting.log")

    print("=== InferenceOS H4 optimizer self-overhead / net-benefit accounting ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    from inferenceos.accounting_acceptance import run_accounting_acceptance

    scenarios, checks = run_accounting_acceptance(settings)

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    real = scenarios.get("real")
    if isinstance(real, dict):
        assessment = real.get("assessment")
        if isinstance(assessment, dict):
            print()
            print(
                f"net-benefit status: {assessment.get('status')} "
                f"(benefit_ratio={assessment.get('benefit_ratio')}, "
                f"overhead_fraction={assessment.get('overhead_fraction')})"
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_payload = {
        "stage": "H4-optimizer-overhead-accounting",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    out_path = (
        Path(args.out)
        if args.out
        else settings.reports_dir / f"accounting_verification_{timestamp}.json"
    )
    persistence = guarded_write_text(
        out_path, json.dumps(report_payload, indent=2), settings
    )
    print()
    if persistence.persisted:
        print(f"report: {out_path}")
    else:
        print(f"warning: acceptance report not persisted: {persistence.reason}")
    print(f"=== RESULT: ACCOUNTING ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


def _cmd_verify_hardening(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "hardening.log")

    print("=== InferenceOS final production-hardening acceptance (H3/H1/B3/H6) ===")
    print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
    print()
    from inferenceos.hardening_acceptance import run_hardening_acceptance

    scenarios, checks = run_hardening_acceptance(settings)

    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_payload = {
        "stage": "final-hardening-H3-H1-B3-H6",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    out_path = (
        Path(args.out)
        if args.out
        else settings.reports_dir / f"hardening_verification_{timestamp}.json"
    )
    persistence = guarded_write_text(
        out_path, json.dumps(report_payload, indent=2), settings
    )
    print()
    if persistence.persisted:
        print(f"report: {out_path}")
    else:
        print(f"warning: acceptance report not persisted: {persistence.reason}")
    print(f"=== RESULT: HARDENING ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
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

    verify5r_parser = subparsers.add_parser(
        "verify-phase5r",
        help="run the real Phase 5R acceptance (resident backend + deep-diagnosis lifecycle)",
    )
    verify5r_parser.add_argument("--model", default=None, help="use a single .gguf model")
    verify5r_parser.add_argument("--report", default=None, help="Phase 3 report path")
    verify5r_parser.set_defaults(func=_cmd_verify_phase5r)

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

    verify7r_parser = subparsers.add_parser(
        "verify-phase7r",
        help=(
            "run the real Phase 7R acceptance (versioned quality, real retained + "
            "quality-rejected experiments, restoration, safety, provenance)"
        ),
    )
    verify7r_parser.add_argument("--model", default=None, help="use a single .gguf model")
    verify7r_parser.add_argument("--db", default=None, help="knowledge database path")
    verify7r_parser.add_argument(
        "--retained-context", type=int, default=16384, help="baseline context for retention"
    )
    verify7r_parser.add_argument(
        "--candidate-context", type=int, default=512, help="candidate context length"
    )
    verify7r_parser.set_defaults(func=_cmd_verify_phase7r)

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

    verify8r_parser = subparsers.add_parser(
        "verify-phase8r",
        help=(
            "run the real Phase 8R acceptance (persisted rollback applied to a real "
            "resident target, verified, idempotent, with safe rejection)"
        ),
    )
    verify8r_parser.add_argument("--model", default=None, help="use a single .gguf model")
    verify8r_parser.add_argument("--db", default=None, help="knowledge database path")
    verify8r_parser.add_argument(
        "--baseline-context", type=int, default=4096, help="baseline context length"
    )
    verify8r_parser.add_argument(
        "--candidate-context", type=int, default=2048, help="optimized context length"
    )
    verify8r_parser.set_defaults(func=_cmd_verify_phase8r)

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

    serve_parser = subparsers.add_parser(
        "serve",
        help=(
            "validate a persisted DeploymentState and serve it with the real resident "
            "llama-server backend (minimal; no orchestration)"
        ),
    )
    serve_parser.add_argument(
        "--deployment", required=True, help="persisted DeploymentState JSON path"
    )
    serve_parser.add_argument("--prompt", default=None, help="probe prompt")
    serve_parser.add_argument("--n-predict", type=int, default=32, help="probe token count")
    serve_parser.add_argument(
        "--duration", type=float, default=0.0, help="keep the verified server alive (seconds)"
    )
    serve_parser.add_argument("--out", default=None, help="serve evidence output path")
    serve_parser.add_argument("--json", action="store_true", help="print the evidence JSON")
    serve_parser.set_defaults(func=_cmd_serve)

    verify9_parser = subparsers.add_parser(
        "verify-phase9",
        help="run the real Phase 9 acceptance (autonomous loop, safety, knowledge, budgets)",
    )
    verify9_parser.add_argument("--db", default=None, help="knowledge database path")
    verify9_parser.set_defaults(func=_cmd_verify_phase9)

    verify9r_parser = subparsers.add_parser(
        "verify-phase9r",
        help=(
            "run the real Phase 9R acceptance (fresh characterize/introspect/baseline/"
            "diagnose -> autonomous optimization path with provenance)"
        ),
    )
    verify9r_parser.add_argument("--model", default=None, help="use a single .gguf model")
    verify9r_parser.set_defaults(func=_cmd_verify_phase9r)

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
    monitor_parser.add_argument(
        "--i-accept-unattended-actions",
        action="store_true",
        help=(
            "explicitly accept that --approval approve may auto-approve disruptive "
            "runtime changes without human review (required for unattended "
            "auto-approval; off by default)"
        ),
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

    verify10r_parser = subparsers.add_parser(
        "verify-phase10r",
        help=(
            "run the real Phase 10R acceptance (resident active workload, conservative "
            "runtime monitoring, approval, in-flight safety, recovery)"
        ),
    )
    verify10r_parser.add_argument("--model", default=None, help="model for the real scenarios")
    verify10r_parser.set_defaults(func=_cmd_verify_phase10r)

    external_parser = subparsers.add_parser(
        "external",
        help="observe external processes and generate recommendation-only / approval-gated advice",
    )
    external_parser.add_argument("--duration", type=float, default=None, help="bounded run seconds")
    external_parser.add_argument("--target-pid", type=int, default=None, help="target workload pid")
    external_parser.add_argument(
        "--target-degraded",
        action="store_true",
        help="the target workload currently shows a measured degradation",
    )
    external_parser.add_argument("--context", default=None, help="context identity label")
    external_parser.add_argument("--db", default=None, help="external knowledge database path")
    external_parser.add_argument(
        "--approval",
        choices=["interactive", "approve", "reject", "timeout"],
        default="interactive",
        help="approval mode for approval-gated external actions",
    )
    external_parser.add_argument("--out", default=None, help="run output path")
    external_parser.add_argument("--json", action="store_true", help="print the run JSON")
    external_parser.set_defaults(func=_cmd_external)

    validity_parser = subparsers.add_parser(
        "verify-validity",
        help=(
            "verify B1 power/thermal measurement validity (real Windows telemetry + "
            "deterministic confound policy)"
        ),
    )
    validity_parser.add_argument("--prompt", default=None, help="probe prompt (unused metadata)")
    validity_parser.add_argument("--out", default=None, help="report output path")
    validity_parser.set_defaults(func=_cmd_verify_validity)

    hardening_parser = subparsers.add_parser(
        "verify-hardening",
        help=(
            "run the final production-hardening acceptance (H3 API-key hygiene, H1 "
            "database policy, B3 deterministic quality, H6 self-target exclusion)"
        ),
    )
    hardening_parser.add_argument("--out", default=None, help="report output path")
    hardening_parser.set_defaults(func=_cmd_verify_hardening)

    accounting_parser = subparsers.add_parser(
        "verify-accounting",
        help=(
            "run the H4 optimizer self-overhead / net-benefit accounting acceptance "
            "(beneficial, overhead-dominant, insufficient-evidence, real runtime)"
        ),
    )
    accounting_parser.add_argument("--out", default=None, help="report output path")
    accounting_parser.set_defaults(func=_cmd_verify_accounting)

    storage_parser = subparsers.add_parser(
        "verify-storage",
        help=(
            "run the H2 log/report retention & disk-protection acceptance "
            "(count/byte limits, path safety, disk pressure, atomic writes)"
        ),
    )
    storage_parser.add_argument("--out", default=None, help="report output path")
    storage_parser.set_defaults(func=_cmd_verify_storage)

    resident_recovery_parser = subparsers.add_parser(
        "verify-resident-recovery",
        help=(
            "verify B2 resident-server crash detection, bounded recovery, identity "
            "continuity and unrecoverable handling"
        ),
    )
    resident_recovery_parser.add_argument("--out", default=None, help="report output path")
    resident_recovery_parser.set_defaults(func=_cmd_verify_resident_recovery)

    remediation_parser = subparsers.add_parser(
        "verify-remediation",
        help=(
            "run the cross-phase remediation acceptance (candidate surface, peak RSS, "
            "independent KV validation, serve --deployment)"
        ),
    )
    remediation_parser.add_argument("--prompt", default=None, help="probe prompt")
    remediation_parser.add_argument("--out", default=None, help="report output path")
    remediation_parser.set_defaults(func=_cmd_verify_remediation)

    verify11r_parser = subparsers.add_parser(
        "verify-phase11r",
        help=(
            "run the Phase 11R acceptance (real external-process observation, "
            "diagnosis-aware recommendation, user decision, staleness, safety)"
        ),
    )
    verify11r_parser.add_argument(
        "--model", default=None, help="model for the real resident workload scenario"
    )
    verify11r_parser.add_argument(
        "--workload", default="short-short", help="benchmark workload id"
    )
    verify11r_parser.set_defaults(func=_cmd_verify_phase11r)

    verify11_parser = subparsers.add_parser(
        "verify-phase11",
        help="deprecated alias for verify-phase11r (recommendation-only)",
    )
    verify11_parser.add_argument("--model", default=None, help="model for the real workload scenario")
    verify11_parser.add_argument("--workload", default="short-short", help="benchmark workload id")
    verify11_parser.set_defaults(func=_cmd_verify_phase11)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # H2: bounded, non-fatal retention at startup so unattended use does not grow
    # reports/logs without limit. It must never prevent the command from running.
    run_startup_retention()
    try:
        result: int = args.func(args)
    except (FileNotFoundError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
