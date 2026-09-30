"""InferenceOS command-line interface.

Everyday commands cover the single optimization steps (``baseline``, ``diagnose``,
``candidates``, ``experiment``), the autonomous paths (``optimize``, ``monitor``,
``serve``) and the inspection/acceptance commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


from inferenceos import __version__
from inferenceos.config import Settings
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    AttentionType,
    MeasurementMode,
    OptimizationOutcome,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.external import (
    ExternalPolicy,
)
from inferenceos.contracts.experiment import (
    ExperimentPolicy,
    ExperimentResult,
    QualityValidationResult,
    RuntimeConfiguration,
)
from inferenceos.contracts.knowledge import (
    ApplicabilityContext,
    KnowledgeQuery,
    OptimizationRecord,
)
from inferenceos.contracts.optimization import CandidatePlan, OptimizationCandidate
from inferenceos.contracts.optimization_loop import LoopPolicy, OptimizationBudget
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.search import GpuOffloadSearchResult
from inferenceos.contracts.reasoning import DeepDiagnosisOutcome
from inferenceos.contracts.system_state import SystemState
from inferenceos.experiment.audit import build_audit_record
from inferenceos.experiment.executor import RuntimeConfigurationExecutor
from inferenceos.experiment.measurement import MeasurementHarness, options_for
from inferenceos.experiment.policy import build_experiment_policy
from inferenceos.experiment.quality import validate_quality
from inferenceos.experiment.report import render_experiment
from inferenceos.experiment.service import ExperimentInputError, run_candidate_experiment
from inferenceos.knowledge import (
    KnowledgeService,
    SQLiteKnowledgeStore,
    build_current_context,
    build_knowledge_policy,
    render_applicability,
    render_health,
    render_history,
    render_record,
    render_rollback,
)
from inferenceos.loop import (
    OptimizationLoop,
    render_run,
)
from inferenceos.deployment import (
    DeploymentIntegrityError,
    deployment_from_loop_result,
    file_sha256,
    load_deployment_state,
    serve_deployment,
    write_deployment_state,
)
from inferenceos.contracts.runtime import (
    RuntimePolicy,
)
from inferenceos.runtime import (
    ApprovalProvider,
    AutoApproveApprovalProvider,
    AutoRejectApprovalProvider,
    CliApprovalProvider,
    ConsoleNotificationSink,
    ControlledLlamaWorkload,
    RuntimeMonitor,
    TimeoutApprovalProvider,
    build_runtime_budget,
    build_runtime_policy,
    render_monitor_run,
)
from inferenceos.external import (
    ExternalKnowledgeStore,
    ExternalRecommendationEngine,
    WindowsProcessObserver,
    build_external_budget,
    build_external_policy,
    render_external_run,
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
from inferenceos.interfaces.reasoning_engine import ReasoningEngine
from inferenceos.logging_config import get_logger, setup_logging
from inferenceos.storage import guarded_write_text, run_startup_retention
from inferenceos.models.estimator import estimate_kv_curve
from inferenceos.models.loader import ModelMetadataError, load_model_profile
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import build_policy
from inferenceos.optimization.preference import build_optimization_preference
from inferenceos.optimization.report import render_plan
from inferenceos.optimization.search.evaluation import (
    PointMeasurement,
    RuntimeConfigurationPointEvaluator,
)
from inferenceos.optimization.search.policy import build_search_policy
from inferenceos.optimization.search.resources import (
    build_resource_envelope,
    total_vram_from_hardware,
)
from inferenceos.optimization.search.runner import run_gpu_offload_search
from inferenceos.profiling.report import render_summary
from inferenceos.profiling.runner import BaselineProfiler, ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import (
    choose_context_length,
    get_workload,
)
from inferenceos.reasoning.factory import (
    build_reasoning_engine,
    build_reasoning_provider,
)
from inferenceos.reasoning.lifecycle import run_deep_diagnosis
from inferenceos.reasoning.providers.base import ReasoningProvider
from inferenceos.reasoning.report import render_deep_diagnosis
from inferenceos.selftest import (
    run_reasoning_selftest,
    verify_contract_roundtrip,
)
from inferenceos.telemetry.system import sample_system_state

logger = get_logger("cli")

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


# --------------------------------------------------------------------------- #
# Diagnosis                                                                     #
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Phase 5R — resident backend & deep-diagnosis lifecycle                       #
# --------------------------------------------------------------------------- #


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
                    "action_id": "runtime.reduce_threads",
                    "parameter_names": ["threads"],
                    "parameter_values": [2],
                    "evidence_ids": ["E999"],
                    "rationale": "hallucinated evidence id",
                },
                {
                    "action_id": "runtime.reduce_threads",
                    "parameter_names": ["threads"],
                    "parameter_values": [4],
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


def _cmd_gpu_offload_search(args: argparse.Namespace) -> int:
    """Resource-aware, empirically evaluated GPU-offload layer search."""
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase6.log")

    models = _load_models(settings, args.model)
    if not models:
        print("error: no .gguf models found", file=sys.stderr)
        return 2
    model = max(models, key=lambda candidate: candidate.file_size_bytes)
    context_length = args.context or choose_context_length(model)
    current_layers = int(args.current_layers)
    if current_layers < 0 or current_layers > model.n_layers:
        print(
            f"error: --current-layers must be within 0..{model.n_layers}",
            file=sys.stderr,
        )
        return 2

    experiment_policy = build_experiment_policy(settings)
    overrides: dict[str, object] = {}
    if args.warmup_repetitions is not None:
        overrides["warmup_repetitions"] = args.warmup_repetitions
    if args.measured_repetitions is not None:
        overrides["measured_repetitions"] = args.measured_repetitions
        overrides["min_valid_samples"] = min(
            int(args.measured_repetitions), experiment_policy.min_valid_samples
        )
    if overrides:
        experiment_policy = experiment_policy.model_copy(update=overrides)
    search_policy = build_search_policy(settings)

    hardware = detect_hardware_profile(settings)
    state = sample_system_state(include_gpu=True, interval_s=0.6)
    used_vram = (
        max((m.memory_used_bytes or 0) for m in state.gpu_memory)
        if state.gpu_memory
        else None
    )
    envelope = build_resource_envelope(
        total_vram_bytes=total_vram_from_hardware(hardware),
        used_vram_bytes=used_vram,
        telemetry_available=state.gpu_telemetry_available,
        telemetry_observed_at=state.gpu_observed_at,
        telemetry_source=state.gpu_source,
        now=datetime.now(timezone.utc),
        model=model,
        context_length=context_length,
        current_layers=current_layers,
        total_layers=model.n_layers,
        policy=search_policy,
    )

    print("=== InferenceOS resource-aware GPU-offload search ===")
    print(f"model          : {model.name} ({model.n_layers} layers, {model.architecture})")
    print(f"configured ctx : {context_length}")
    print(f"current layers : {current_layers}")
    print(f"total VRAM     : {envelope.total_vram_bytes}")
    print(f"used VRAM      : {envelope.used_vram_bytes}")
    print(f"free VRAM      : {envelope.free_vram_bytes}")
    print(f"other consumers: {envelope.other_consumers_bytes}")
    print(f"safety headroom: {envelope.safety_headroom_bytes} ({search_policy.safety_headroom_fraction:.0%})")
    print(f"KV cache       : {envelope.kv_cache_bytes}")
    print(f"per-layer      : {envelope.per_layer_weight_bytes}")
    print(f"feasible range : {envelope.feasible_min}-{envelope.feasible_max}")
    print(f"telemetry      : available={envelope.telemetry_available} fresh={envelope.telemetry_fresh} source={envelope.telemetry_source!r}")
    if not envelope.sufficient:
        print(f"insufficient   : {'; '.join(envelope.reasons)}")
    print()

    profiler = BaselineProfiler(settings=settings, hardware=hardware)
    workload = get_workload(args.workload)
    evaluator = RuntimeConfigurationPointEvaluator(
        profiler=profiler,
        model=model,
        workload=workload,
        experiment_policy=experiment_policy,
        context_length=context_length,
        envelope=envelope,
        base_configuration=RuntimeConfiguration(
            threads=(args.threads if args.threads else None),
            use_mmap=(True if args.mmap else None),
            device=args.device,
        ),
    )

    def _quality_for(
        baseline_point: PointMeasurement, candidate_point: PointMeasurement
    ) -> QualityValidationResult | None:
        if (
            baseline_point.measurement is None
            or candidate_point.measurement is None
        ):
            return None
        return validate_quality(
            baseline=baseline_point.measurement,
            candidate=candidate_point.measurement,
            workload=workload,
            policy=experiment_policy,
            settings=settings,
            action_id="runtime.increase_gpu_offload_layers",
            diagnosed_bottleneck="gpu_underutilization",
            model=model,
        )

    result = run_gpu_offload_search(
        envelope=envelope,
        evaluator=evaluator,
        search_policy=search_policy,
        experiment_policy=experiment_policy,
        model_ref=model.name,
        quality_evaluator=_quality_for,
    )

    _render_gpu_offload_search(result)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = (
        Path(args.out)
        if args.out
        else settings.reports_dir / f"gpu_offload_search_{timestamp}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.to_json(), encoding="utf-8")
    print()
    print(f"report: {out_path}")
    if args.json:
        print(result.to_json())
    return 0


def _render_gpu_offload_search(result: GpuOffloadSearchResult) -> None:
    from inferenceos.contracts.enums import SearchSelectionKind

    print("evaluated candidates:")
    for point in result.points:
        print(
            f"  layers={point.layers:>3} status={point.status.value:<18} "
            f"valid={point.valid} safe={point.safe} samples={point.samples} "
            f"decision={point.decision}"
        )
        for metric in ("decode_tokens_per_sec", "total_latency_s", "gpu_memory_used_bytes"):
            stats = point.metrics.get(metric)
            if stats is not None and stats.mean is not None:
                print(f"      {metric}={stats.mean:.4g}")
    print()
    selection = result.selection
    if selection.kind is SearchSelectionKind.SELECT:
        print(f"RECOMMEND {selection.selected_layers} GPU layers")
    else:
        print("NO CHANGE")
    print(f"reason: {selection.reason}")


# --------------------------------------------------------------------------- #
# Phase 7R — quality validation & real experiment safety remediation           #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Phase 8R — persistent knowledge & real rollback remediation                  #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Phase 9R — fresh full-pipeline autonomous optimization remediation           #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Phase 10 — runtime adaptive monitoring                                       #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


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


def _apply_optimization_args(settings: Settings, args: argparse.Namespace) -> None:
    """Apply explicit CLI optimization-policy overrides onto the persisted settings.

    CLI flags win over the program-level ``optimization_mode`` (env/``.env``); the
    persisted value remains the durable default when no flag is supplied.
    """
    mode = getattr(args, "optimization_mode", None)
    if mode is not None:
        settings.optimization_mode = str(mode)
    minimum = getattr(args, "minimum_improvement_percent", None)
    if minimum is not None:
        settings.optimization_minimum_improvement_percent = float(minimum)
    maximum = getattr(args, "max_search_time_s", None)
    if maximum is not None:
        settings.optimization_maximum_search_time_s = float(maximum)
    policy = getattr(args, "degraded_policy", None)
    if policy is not None:
        settings.optimization_degraded_performance_policy = str(policy)


def _cmd_optimization_policy(args: argparse.Namespace) -> int:
    """Print the resolved, program-level optimization policy (configuration surface)."""
    settings = Settings()
    _apply_optimization_args(settings, args)
    try:
        preference = build_optimization_preference(settings)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("=== InferenceOS optimization policy ===")
    print(f"mode                  : {preference.mode.value}")
    print(f"minimum_improvement   : {preference.minimum_improvement:.2%}")
    print(f"maximum_search_time_s : {preference.maximum_search_time_s:.0f}")
    print(
        "degraded_policy       : "
        f"{preference.degraded_performance_policy.value}"
    )
    print(f"source                : {preference.source}")
    if args.json:
        print(preference.model_dump_json(indent=2))
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase10.log")
    _apply_optimization_args(settings, args)

    try:
        model_path = _find_model(settings.models_dir, args.model)
        model = load_model_profile(model_path)
        loop_policy = _runtime_policy_from_args(settings, args)
        optimization_preference = build_optimization_preference(settings)
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
    print(
        f"optimize : mode={optimization_preference.mode.value} "
        f"min_improvement={optimization_preference.minimum_improvement:.2%} "
        f"max_search={optimization_preference.maximum_search_time_s:.0f}s"
    )
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


# --------------------------------------------------------------------------- #
# Phase 10R — runtime adaptive monitoring remediation over a resident target    #
# --------------------------------------------------------------------------- #


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

    gpu_search_parser = subparsers.add_parser(
        "gpu-offload-search",
        help=(
            "resource-aware GPU-offload layer search: feasible range, sampled points, "
            "real measurements, exactly-one-value selection or NO CHANGE"
        ),
    )
    gpu_search_parser.add_argument("--model", default=None, help="model path override")
    gpu_search_parser.add_argument(
        "--context", type=int, default=None, help="configured context length (fixed constraint)"
    )
    gpu_search_parser.add_argument(
        "--current-layers", type=int, default=0, help="current GPU-offload layer count (baseline)"
    )
    gpu_search_parser.add_argument(
        "--workload", default="short-short", help="benchmark workload id"
    )
    gpu_search_parser.add_argument("--threads", type=int, default=0, help="inference thread count")
    gpu_search_parser.add_argument(
        "--mmap", action="store_true", help="enable memory-mapped weights"
    )
    gpu_search_parser.add_argument("--device", default=None, help="backend device selector")
    gpu_search_parser.add_argument(
        "--warmup-repetitions", type=int, default=None, help="warm-up runs per candidate"
    )
    gpu_search_parser.add_argument(
        "--measured-repetitions", type=int, default=None, help="measured runs per candidate"
    )
    gpu_search_parser.add_argument("--out", default=None, help="search result JSON path")
    gpu_search_parser.add_argument("--json", action="store_true", help="print the result JSON")
    gpu_search_parser.set_defaults(func=_cmd_gpu_offload_search)


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
    monitor_parser.add_argument(
        "--optimization-mode",
        choices=[
            "balanced",
            "performance_first",
            "minimize_optimization_cost",
            "custom",
        ],
        default=None,
        help=(
            "user optimization policy: how aggressively to search for improvements "
            "(default: the persisted program-level optimization_mode, normally balanced)"
        ),
    )
    monitor_parser.add_argument(
        "--minimum-improvement-percent",
        type=float,
        default=None,
        help="custom mode: minimum worthwhile relative improvement percent",
    )
    monitor_parser.add_argument(
        "--max-search-time-s",
        type=float,
        default=None,
        help="custom mode: maximum optimization/search time in seconds",
    )
    monitor_parser.add_argument(
        "--degraded-policy",
        choices=["lower_requirement", "strict"],
        default=None,
        help="custom mode: whether degraded performance lowers the requirement",
    )
    monitor_parser.add_argument("--out", default=None, help="run output path")
    monitor_parser.add_argument("--json", action="store_true", help="print the run JSON")
    monitor_parser.set_defaults(func=_cmd_monitor)

    optimization_policy_parser = subparsers.add_parser(
        "optimization-policy",
        help="print the resolved program-level optimization policy",
    )
    optimization_policy_parser.add_argument(
        "--optimization-mode",
        choices=[
            "balanced",
            "performance_first",
            "minimize_optimization_cost",
            "custom",
        ],
        default=None,
        help="override the persisted optimization_mode for this printout",
    )
    optimization_policy_parser.add_argument(
        "--minimum-improvement-percent", type=float, default=None
    )
    optimization_policy_parser.add_argument("--max-search-time-s", type=float, default=None)
    optimization_policy_parser.add_argument(
        "--degraded-policy",
        choices=["lower_requirement", "strict"],
        default=None,
    )
    optimization_policy_parser.add_argument("--json", action="store_true")
    optimization_policy_parser.set_defaults(func=_cmd_optimization_policy)


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
