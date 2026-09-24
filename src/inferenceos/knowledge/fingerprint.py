"""Reproducible context fingerprinting (identity vs. transient conditions).

Identity dimensions must be stable across runs of the same hardware/model/backend/
workload/runtime configuration; transient operating conditions (available RAM, system
CPU, page faults) are kept separate so irrelevant telemetry never invalidates identity.
"""

from __future__ import annotations

import hashlib
import json

from inferenceos.contracts.experiment import ExperimentRequest, RuntimeConfiguration
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.knowledge import (
    ApplicabilityContext,
    ContextFingerprint,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import MetricStats, ProfilingConfig
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.signature import hardware_signature, workload_signature

_CONDITION_METRICS: tuple[str, ...] = (
    "available_ram_bytes",
    "system_cpu_percent",
    "hard_page_faults_per_sec",
    "peak_rss_bytes",
    "committed_bytes",
    "commit_limit_bytes",
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:32]


def runtime_key(configuration: RuntimeConfiguration) -> str:
    fields = {
        key: value
        for key, value in configuration.model_dump().items()
        if value is not None
    }
    return _canonical(fields)


def _cpu_key(hardware: HardwareProfile) -> str:
    cpu = hardware.cpu
    return (
        f"{cpu.vendor or '?'}:{cpu.name};"
        f"arch={cpu.architecture or '?'};"
        f"cores={cpu.physical_cores}/{cpu.logical_cores}"
    )


def _gpu_key(hardware: HardwareProfile) -> str:
    if not hardware.gpus:
        return "none"
    return "|".join(
        f"{gpu.vendor.value}:{gpu.name}:{gpu.device_type.value}:{gpu.vram_bytes}"
        for gpu in hardware.gpus
    )


def _os_key(hardware: HardwareProfile) -> str:
    os_profile = hardware.os
    return f"{os_profile.caption};{os_profile.version}.{os_profile.build};{os_profile.architecture}"


def _model_key(model: ModelProfile | None) -> str:
    if model is None:
        return "unknown"
    return (
        f"{model.name};arch={model.architecture};quant={model.quantization.value};"
        f"layers={model.n_layers};heads={model.n_attention_heads};kv_heads={model.n_kv_heads};"
        f"ctx={model.trained_context_length};size={model.file_size_bytes}"
    )


def _backend_key(backend: ProfilingConfig) -> str:
    return (
        f"{backend.backend_name or '?'}@{backend.backend_version or '?'};"
        f"{backend.backend_type.value if backend.backend_type else '?'}"
    )


def _workload_key(workload: WorkloadProfile | None) -> str:
    if workload is None:
        return "unknown"
    return f"{workload_signature(workload)};prompt={workload.prompt_set_ref or '?'}"


def conditions_from_metrics(metrics: dict[str, MetricStats]) -> dict[str, float]:
    conditions: dict[str, float] = {}
    for name in _CONDITION_METRICS:
        stats = metrics.get(name)
        if stats is not None and stats.mean is not None:
            conditions[name] = round(float(stats.mean), 6)
    committed = metrics.get("committed_bytes")
    limit = metrics.get("commit_limit_bytes")
    if (
        committed is not None
        and limit is not None
        and committed.mean is not None
        and limit.mean
    ):
        conditions["memory_pressure"] = round(float(committed.mean / limit.mean), 6)
    return conditions


def build_fingerprint(
    *,
    hardware: HardwareProfile,
    model: ModelProfile | None,
    workload: WorkloadProfile | None,
    backend: ProfilingConfig,
    runtime: RuntimeConfiguration,
    diagnosis: str | None,
    conditions: dict[str, float] | None = None,
    model_content_hash: str | None = None,
) -> ContextFingerprint:
    conditions = conditions or {}
    identity = {
        "hardware": hardware_signature(hardware),
        "cpu": _cpu_key(hardware),
        "gpu": _gpu_key(hardware),
        "os": _os_key(hardware),
        "backend": _backend_key(backend),
        "model": _model_key(model),
        "quantization": model.quantization.value if model is not None else "unknown",
        "workload": _workload_key(workload),
        "runtime": runtime_key(runtime),
        "diagnosis": diagnosis or "none",
    }
    return ContextFingerprint(
        identity_hash=_hash(identity),
        conditions_hash=_hash(conditions),
        hardware_key=identity["hardware"],
        cpu_key=identity["cpu"],
        gpu_key=identity["gpu"],
        os_key=identity["os"],
        backend_key=identity["backend"],
        model_key=identity["model"],
        quantization=identity["quantization"],
        model_size_bytes=model.file_size_bytes if model is not None else None,
        model_content_hash=model_content_hash,
        workload_key=identity["workload"],
        runtime_key=identity["runtime"],
        diagnosis=diagnosis,
        conditions=dict(conditions),
    )


def _assemble(
    *,
    hardware: HardwareProfile,
    model: ModelProfile | None,
    workload: WorkloadProfile | None,
    backend: ProfilingConfig,
    runtime: RuntimeConfiguration,
    diagnosis: str | None,
    conditions: dict[str, float] | None,
    model_content_hash: str | None = None,
) -> ApplicabilityContext:
    fingerprint = build_fingerprint(
        hardware=hardware,
        model=model,
        workload=workload,
        backend=backend,
        runtime=runtime,
        diagnosis=diagnosis,
        conditions=conditions,
        model_content_hash=model_content_hash,
    )
    return ApplicabilityContext(
        fingerprint=fingerprint,
        hardware=hardware,
        model=model,
        workload=workload,
        backend=backend,
        runtime_configuration=runtime,
        diagnosed_bottleneck=diagnosis,
    )


def fingerprint_context(context: ApplicabilityContext) -> ContextFingerprint:
    """Recompute the fingerprint from an :class:`ApplicabilityContext`."""
    return build_fingerprint(
        hardware=context.hardware,
        model=context.model,
        workload=context.workload,
        backend=context.backend,
        runtime=context.runtime_configuration,
        diagnosis=context.diagnosed_bottleneck,
        conditions=context.fingerprint.conditions,
        model_content_hash=context.fingerprint.model_content_hash,
    )


def build_context_from_request(
    request: ExperimentRequest,
    *,
    baseline_metrics: dict[str, MetricStats] | None = None,
    model_content_hash: str | None = None,
) -> ApplicabilityContext:
    conditions = conditions_from_metrics(baseline_metrics or {})
    return _assemble(
        hardware=request.hardware,
        model=request.model,
        workload=request.workload,
        backend=request.backend,
        runtime=request.baseline_configuration,
        diagnosis=request.diagnosed_bottleneck,
        conditions=conditions,
        model_content_hash=model_content_hash,
    )


def build_current_context(
    *,
    hardware: HardwareProfile,
    model: ModelProfile | None,
    workload: WorkloadProfile | None,
    backend: ProfilingConfig,
    runtime: RuntimeConfiguration,
    diagnosis: str | None = None,
    conditions: dict[str, float] | None = None,
    model_content_hash: str | None = None,
) -> ApplicabilityContext:
    return _assemble(
        hardware=hardware,
        model=model,
        workload=workload,
        backend=backend,
        runtime=runtime,
        diagnosis=diagnosis,
        conditions=conditions,
        model_content_hash=model_content_hash,
    )
