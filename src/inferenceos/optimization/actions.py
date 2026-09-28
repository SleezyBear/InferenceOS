"""Initial Phase 6 action vocabulary.

A deliberately small, real vocabulary covering the v1 optimization surfaces from the
plan (runtime parameters, context/KV management). It also includes a few
intentionally broad/destructive surfaces (external process, system pagefile, model
requantization) so the registry, approval gating, and rejection policy can be proven
against genuinely unsafe actions rather than toy ones. None of these actions is
executed in Phase 6.

Kernel/graph-level actions remain explicitly out of scope.
"""

from __future__ import annotations

from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    EvidenceStrength,
    ExternalActionSafetyClass,
    OptimizationScope,
    OptimizationSurface,
    ParameterType,
    PrivilegeRequirement,
    Reversibility,
    RiskLevel,
)
from inferenceos.contracts.optimization import CandidateAction, ParameterSpec
from inferenceos.optimization.registry import ActionRegistry

ACTION_REGISTRY_VERSION = "phase11r.remediation.actions.v2"


def build_default_registry() -> ActionRegistry:
    registry = ActionRegistry()
    for action in _ACTIONS:
        registry.register(action)
    return registry


_ACTIONS: tuple[CandidateAction, ...] = (
    CandidateAction(
        action_id="runtime.right_size_context",
        name="Right-size context length",
        description=(
            "Reduce the per-run context length so the KV cache footprint is smaller. "
            "Applies only when the diagnosed workload does not require the full context."
        ),
        surface=OptimizationSurface.CONTEXT_MANAGEMENT,
        subsystem="kv_cache",
        supported_bottlenecks=[
            BottleneckKind.PAGING,
            BottleneckKind.MEMORY_PRESSURE,
            BottleneckKind.MODEL_FOOTPRINT,
            BottleneckKind.GPU_MEMORY_LIMIT,
        ],
        causal_rationale=(
            "KV-cache memory scales linearly with context length; under memory pressure "
            "or paging, reducing context reduces the target workload's resident memory "
            "demand."
        ),
        parameters=[
            ParameterSpec(
                name="context_length",
                type=ParameterType.INTEGER,
                description="Target context length (<= current and <= trained context).",
                minimum=512,
                maximum=1_048_576,
                unit="tokens",
            )
        ],
        preconditions=[
            "current context length is known",
            "target context length is below the current value",
            "target context length does not exceed the model's trained context",
            "diagnosis evidence supports memory pressure/paging",
        ],
        expected_metrics=["peak_rss_bytes", "hard_page_faults_per_sec"],
        expected_direction="decrease",
        reversibility=Reversibility.REVERSIBLE,
        risk_level=RiskLevel.LOW,
        scope=OptimizationScope.TARGET_MODEL_RUNTIME,
        privilege=PrivilegeRequirement.STANDARD_USER,
        requires_human_approval=False,
        execution_cost_estimate=1.0,
        optimizer_overhead_estimate=0.02,
        min_evidence_strength=EvidenceStrength.MEDIUM,
    ),
    CandidateAction(
        action_id="runtime.enable_mmap",
        name="Enable memory-mapped weights",
        description=(
            "Use memory-mapped model weights so clean file-backed pages can be evicted "
            "under pressure instead of forcing anonymous resident memory."
        ),
        surface=OptimizationSurface.MEMORY_MANAGEMENT,
        subsystem="model_mapping",
        supported_bottlenecks=[
            BottleneckKind.PAGING,
            BottleneckKind.MEMORY_PRESSURE,
            BottleneckKind.MODEL_FOOTPRINT,
        ],
        causal_rationale=(
            "File-backed mappings let the OS reclaim clean weight pages, reducing "
            "anonymous commit pressure that drives hard page faults."
        ),
        parameters=[
            ParameterSpec(
                name="use_mmap",
                type=ParameterType.BOOLEAN,
                description="Whether mmap-backed weight loading is enabled.",
            )
        ],
        preconditions=[
            "current mmap configuration is known and is disabled",
            "diagnosis evidence supports memory pressure/paging",
        ],
        expected_metrics=["hard_page_faults_per_sec", "peak_rss_bytes"],
        expected_direction="decrease",
        reversibility=Reversibility.REVERSIBLE,
        risk_level=RiskLevel.LOW,
        scope=OptimizationScope.TARGET_MODEL_RUNTIME,
        privilege=PrivilegeRequirement.STANDARD_USER,
        requires_human_approval=False,
        execution_cost_estimate=1.0,
        optimizer_overhead_estimate=0.02,
        min_evidence_strength=EvidenceStrength.MEDIUM,
    ),
    CandidateAction(
        action_id="runtime.reduce_gpu_offload_layers",
        name="Reduce GPU offload layers",
        description=(
            "Move some transformer layers back to CPU so VRAM demand is reduced. Only "
            "relevant when VRAM is the diagnosed limiting resource."
        ),
        surface=OptimizationSurface.GPU_OFFLOAD,
        subsystem="gpu_memory",
        supported_bottlenecks=[BottleneckKind.GPU_MEMORY_LIMIT],
        causal_rationale=(
            "VRAM footprint scales with the number of offloaded layers; reducing the "
            "layer count lowers VRAM demand when VRAM is the diagnosed limit."
        ),
        parameters=[
            ParameterSpec(
                name="gpu_offload_layers",
                type=ParameterType.INTEGER,
                description="Number of layers to offload (lower than current).",
                minimum=0,
                maximum=1024,
                unit="layers",
            )
        ],
        preconditions=[
            "a usable GPU backend exists",
            "layers are currently offloaded",
            "diagnosis evidence supports a GPU memory limit",
        ],
        expected_metrics=["gpu_memory_used_bytes"],
        expected_direction="decrease",
        reversibility=Reversibility.REVERSIBLE,
        risk_level=RiskLevel.LOW,
        scope=OptimizationScope.TARGET_MODEL_RUNTIME,
        privilege=PrivilegeRequirement.STANDARD_USER,
        requires_human_approval=False,
        execution_cost_estimate=1.0,
        optimizer_overhead_estimate=0.02,
        min_evidence_strength=EvidenceStrength.MEDIUM,
    ),
    CandidateAction(
        action_id="runtime.increase_gpu_offload_layers",
        name="Increase GPU offload layers",
        description=(
            "Move more transformer layers onto the accelerator when offload is "
            "configured but the GPU is genuinely underutilized while execution is "
            "CPU-bound. Raises VRAM demand and requires a GPU with headroom; reversible "
            "by restoring the prior offload count."
        ),
        surface=OptimizationSurface.GPU_OFFLOAD,
        subsystem="gpu_offload",
        supported_bottlenecks=[BottleneckKind.GPU_UNDERUTILIZATION],
        causal_rationale=(
            "When offload is configured but the GPU is idle while process CPU is "
            "saturated, shifting more layers onto the accelerator can raise GPU "
            "utilization and throughput; this is an inference-level runtime parameter, "
            "not a model edit."
        ),
        parameters=[
            ParameterSpec(
                name="gpu_offload_layers",
                type=ParameterType.INTEGER,
                description=(
                    "Number of layers to offload (higher than current, up to the layer total)."
                ),
                minimum=0,
                maximum=1024,
                unit="layers",
            )
        ],
        preconditions=[
            "a usable GPU backend exists",
            "current offload count and layer total are known",
            "target offload is above the current value and does not exceed the layer total",
        ],
        expected_metrics=["gpu_utilization_percent", "decode_tokens_per_sec"],
        expected_direction="increase",
        reversibility=Reversibility.REVERSIBLE,
        risk_level=RiskLevel.LOW,
        scope=OptimizationScope.TARGET_MODEL_RUNTIME,
        privilege=PrivilegeRequirement.STANDARD_USER,
        requires_human_approval=False,
        execution_cost_estimate=1.0,
        optimizer_overhead_estimate=0.02,
        min_evidence_strength=EvidenceStrength.MEDIUM,
    ),
    CandidateAction(
        action_id="runtime.reduce_threads",
        name="Right-size CPU thread count",
        description=(
            "Reduce the inference thread count toward the physical core count to avoid "
            "oversubscription. Only relevant when CPU execution is the diagnosed limit."
        ),
        surface=OptimizationSurface.RUNTIME_PARAMS,
        subsystem="cpu_execution",
        supported_bottlenecks=[BottleneckKind.CPU],
        causal_rationale=(
            "When the target process oversubscribes physical cores, matching the thread "
            "count to physical cores reduces contention for the diagnosed CPU limit; the "
            "causal objective is throughput and CPU-seconds per token, with a reduction in "
            "context switches as supporting evidence (raw CPU utilization is secondary)."
        ),
        parameters=[
            ParameterSpec(
                name="threads",
                type=ParameterType.INTEGER,
                description="Thread count at or below the current value.",
                minimum=1,
                maximum=4096,
                unit="threads",
            )
        ],
        preconditions=[
            "current thread count is known",
            "target thread count is below the current value",
            "diagnosis evidence supports a CPU execution limit",
        ],
        expected_metrics=[
            "decode_tokens_per_sec",
            "cpu_seconds_per_token",
            "context_switches_per_token",
        ],
        expected_direction="increase",
        reversibility=Reversibility.REVERSIBLE,
        risk_level=RiskLevel.LOW,
        scope=OptimizationScope.TARGET_MODEL_RUNTIME,
        privilege=PrivilegeRequirement.STANDARD_USER,
        requires_human_approval=False,
        execution_cost_estimate=1.0,
        optimizer_overhead_estimate=0.02,
        min_evidence_strength=EvidenceStrength.MEDIUM,
    ),
    CandidateAction(
        action_id="external.suspend_process",
        name="Suspend an external memory-dominant process",
        description=(
            "Recommend pausing a non-target process that dominates memory, to relieve "
            "pressure without terminating it. Never applied autonomously."
        ),
        surface=OptimizationSurface.EXTERNAL_PROCESS,
        subsystem="memory_contention",
        supported_bottlenecks=[
            BottleneckKind.EXTERNAL_PROCESS,
            BottleneckKind.MEMORY_PRESSURE,
        ],
        causal_rationale=(
            "If an external process holds more resident memory than the model under "
            "pressure, pausing it can free memory for the target workload."
        ),
        parameters=[
            ParameterSpec(
                name="process_name",
                type=ParameterType.STRING,
                description="Name of the external process to suspend (explicit list).",
                choices=["memory_holder.exe", "python.exe"],
            )
        ],
        preconditions=[
            "an external process is identified as memory-dominant",
            "human approval is granted",
        ],
        expected_metrics=["available_ram_bytes"],
        expected_direction="increase",
        reversibility=Reversibility.REVERSIBLE,
        risk_level=RiskLevel.MEDIUM,
        scope=OptimizationScope.EXTERNAL_PROCESS,
        privilege=PrivilegeRequirement.STANDARD_USER,
        requires_human_approval=True,
        execution_cost_estimate=1.0,
        optimizer_overhead_estimate=0.02,
        min_evidence_strength=EvidenceStrength.MEDIUM,
        external_safety_class=ExternalActionSafetyClass.RECOMMENDATION_ONLY,
        implemented=False,
        phase11_allowed=True,
        verification_supported=False,
    ),
    CandidateAction(
        action_id="external.request_close",
        name="Recommend closing a competing process",
        description=(
            "Ask the user to close a non-target process that is a plausible dominant "
            "contributor to the observed contention. InferenceOS never performs the "
            "close itself; this is a user-action recommendation only."
        ),
        surface=OptimizationSurface.EXTERNAL_PROCESS,
        subsystem="user_action",
        supported_bottlenecks=[
            BottleneckKind.EXTERNAL_PROCESS,
            BottleneckKind.MEMORY_PRESSURE,
            BottleneckKind.CPU,
        ],
        causal_rationale=(
            "A process that is a plausible dominant contributor to a system resource "
            "under contention may be affecting the target workload if the user does not "
            "need it running; only the user can decide that."
        ),
        parameters=[
            ParameterSpec(
                name="process_name",
                type=ParameterType.STRING,
                description="Observed process name the user may consider closing.",
                choices=["memory_holder.exe", "python.exe", "pythonw.exe"],
            )
        ],
        preconditions=[
            "an external process is a plausible dominant contributor under contention",
            "human approval/decision is required",
        ],
        expected_metrics=["available_ram_bytes"],
        expected_direction="increase",
        reversibility=Reversibility.NON_REVERSIBLE,
        risk_level=RiskLevel.MEDIUM,
        scope=OptimizationScope.EXTERNAL_PROCESS,
        privilege=PrivilegeRequirement.STANDARD_USER,
        requires_human_approval=True,
        execution_cost_estimate=0.0,
        optimizer_overhead_estimate=0.0,
        min_evidence_strength=EvidenceStrength.MEDIUM,
        external_safety_class=ExternalActionSafetyClass.RECOMMENDATION_ONLY,
        implemented=False,
        phase11_allowed=True,
        verification_supported=False,
    ),
    CandidateAction(
        action_id="external.reduce_priority",
        name="Recommend reducing an external process's CPU priority",
        description=(
            "Recommend that the user consider lowering the scheduling priority of a "
            "non-target process that is a plausible dominant CPU contributor. Phase 11R "
            "is recommendation-only: InferenceOS never changes the priority itself and "
            "the user remains in control."
        ),
        surface=OptimizationSurface.EXTERNAL_PROCESS,
        subsystem="cpu_contention",
        supported_bottlenecks=[BottleneckKind.CPU, BottleneckKind.EXTERNAL_PROCESS],
        causal_rationale=(
            "When an external process contends for CPU and the target workload is "
            "degraded, lowering the external process's priority could increase the "
            "target's share without terminating the process; only the user can decide."
        ),
        parameters=[
            ParameterSpec(
                name="priority_class",
                type=ParameterType.STRING,
                description="Suggested priority class (lower than current), user-applied.",
                choices=["below_normal", "idle"],
            )
        ],
        preconditions=[
            "an external process is a plausible dominant CPU contributor",
            "human decision is required; InferenceOS performs no change",
        ],
        expected_metrics=["process_cpu_percent", "decode_tokens_per_sec"],
        expected_direction="decrease",
        reversibility=Reversibility.REVERSIBLE,
        risk_level=RiskLevel.LOW,
        scope=OptimizationScope.EXTERNAL_PROCESS,
        privilege=PrivilegeRequirement.STANDARD_USER,
        requires_human_approval=True,
        execution_cost_estimate=0.0,
        optimizer_overhead_estimate=0.0,
        min_evidence_strength=EvidenceStrength.MEDIUM,
        external_safety_class=ExternalActionSafetyClass.RECOMMENDATION_ONLY,
        implemented=False,
        phase11_allowed=True,
        verification_supported=False,
    ),
    CandidateAction(
        action_id="system.increase_pagefile",
        name="Increase system pagefile size",
        description=(
            "Enlarge the Windows pagefile to reduce hard page faults under sustained "
            "commit pressure. Requires elevation; never applied autonomously."
        ),
        surface=OptimizationSurface.SYSTEM_CONFIGURATION,
        subsystem="virtual_memory",
        supported_bottlenecks=[
            BottleneckKind.PAGING,
            BottleneckKind.MEMORY_PRESSURE,
        ],
        causal_rationale=(
            "A larger pagefile raises the commit limit and can reduce hard page-fault "
            "activity when commit pressure is the diagnosed mechanism."
        ),
        parameters=[
            ParameterSpec(
                name="increase_bytes",
                type=ParameterType.INTEGER,
                description="Additional pagefile capacity in bytes.",
                minimum=268435456,
                maximum=68719476736,
                unit="bytes",
            )
        ],
        preconditions=[
            "commit pressure is the diagnosed mechanism",
            "administrator elevation is available and approved",
        ],
        expected_metrics=["hard_page_faults_per_sec"],
        expected_direction="decrease",
        reversibility=Reversibility.REVERSIBLE_WITH_RESTART,
        risk_level=RiskLevel.HIGH,
        scope=OptimizationScope.SYSTEM_WIDE,
        privilege=PrivilegeRequirement.ELEVATION_REQUIRED,
        requires_human_approval=True,
        execution_cost_estimate=3.0,
        optimizer_overhead_estimate=0.02,
        min_evidence_strength=EvidenceStrength.MEDIUM,
        external_safety_class=ExternalActionSafetyClass.RECOMMENDATION_ONLY,
        implemented=False,
        phase11_allowed=False,
        verification_supported=False,
    ),
    CandidateAction(
        action_id="model.requantize",
        name="Requantize model weights",
        description=(
            "Rewrite the model to a smaller quantization. Destructive and "
            "non-reversible; rejected by policy, never autonomously selected."
        ),
        surface=OptimizationSurface.MODEL_REPRESENTATION,
        subsystem="weights",
        supported_bottlenecks=[
            BottleneckKind.MODEL_FOOTPRINT,
            BottleneckKind.MEMORY_PRESSURE,
            BottleneckKind.GPU_MEMORY_LIMIT,
        ],
        causal_rationale=(
            "A smaller weight representation reduces resident memory, but changes model "
            "quality and cannot be undone in place."
        ),
        parameters=[
            ParameterSpec(
                name="target_quantization",
                type=ParameterType.STRING,
                description="Target GGUF quantization.",
                choices=["q4_k_m", "q5_k_m", "q8_0"],
            )
        ],
        preconditions=[
            "a smaller quantization is available",
            "human approval is granted",
        ],
        expected_metrics=["peak_rss_bytes"],
        expected_direction="decrease",
        reversibility=Reversibility.NON_REVERSIBLE,
        risk_level=RiskLevel.HIGH,
        scope=OptimizationScope.TARGET_MODEL_RUNTIME,
        privilege=PrivilegeRequirement.STANDARD_USER,
        requires_human_approval=True,
        execution_cost_estimate=60.0,
        optimizer_overhead_estimate=0.02,
        min_evidence_strength=EvidenceStrength.HIGH,
        external_safety_class=ExternalActionSafetyClass.RECOMMENDATION_ONLY,
        implemented=False,
        phase11_allowed=False,
        verification_supported=False,
    ),
)
