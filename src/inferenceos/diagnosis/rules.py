"""Deterministic bottleneck rules.

Every rule is expressed as *relative-significance* reasoning plus comparative
evidence. Thresholds are heuristics that may raise a *candidate* signal; they are
never sufficient on their own to declare a meaningful bottleneck. A resource being
heavily used is explicitly not treated as a bottleneck unless corroborating,
mostly comparative, evidence supports it.

Rules never produce optimization recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inferenceos.contracts.bottleneck import EvidenceItem
from inferenceos.contracts.enums import (
    BottleneckKind,
    EvidencePolarity,
    EvidenceStrength,
    InferenceStage,
)
from inferenceos.diagnosis.groups import GroupObservation

# --- documented heuristic constants (never the sole basis for a conclusion) ---
FOOTPRINT_RATIO_HIGH = 0.5
PRESSURE_HIGH = 0.80
PRESSURE_LOW = 0.6
PRESSURE_DELTA = 0.10
PAGING_MIN_PER_SEC = 1.0
PAGING_RELATIVE = 3.0
CPU_SATURATION_HIGH = 0.70
GPU_UTIL_LOW = 40.0
GPU_UTIL_SATURATED = 85.0
GPU_CV_STABLE = 0.10
# GPU memory in use as a fraction of total device VRAM. A high ratio is only a
# candidate signal; it becomes a meaningful GPU-memory limit only with corroborating
# comparative evidence that less GPU-memory use performs materially better.
GPU_VRAM_PRESSURE_HIGH = 0.85
GPU_VRAM_PRESSURE_EXTREME = 0.95
COMPARABLE_THROUGHPUT_RATIO = 1.5
COMPARABLE_LATENCY_RATIO = 1.3
LOAD_SHARE_HIGH = 0.30
COLD_WARM_LOAD_RATIO = 2.0
STAGE_PREFILL_DOMINANT = 0.60
STAGE_DECODE_DOMINANT = 0.85

RULE_VERSION = "phase4.rules.v1"


@dataclass
class RuleContext:
    groups: list[GroupObservation]
    physical_cores: int
    logical_cores: int
    # Total device VRAM reported by the hardware/backend inventory (bytes), when known.
    # ``None`` means unavailable; the GPU-memory-limit rule then cannot fire.
    vram_bytes: int | None = None


@dataclass
class CandidateFinding:
    kind: BottleneckKind
    stage: InferenceStage
    subject: str
    supporting: list[EvidenceItem] = field(default_factory=list)
    contradicting: list[EvidenceItem] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    forced_non_meaningful: bool = False
    source: str = RULE_VERSION


def _evidence(
    metric: str,
    observed: float,
    reference: float | None,
    description: str,
    polarity: EvidencePolarity,
    stage: InferenceStage,
    strength: EvidenceStrength,
    source: str,
) -> EvidenceItem:
    ratio: float | None = None
    if reference not in (None, 0.0):
        ratio = observed / reference
    return EvidenceItem(
        metric=metric,
        observed_value=float(observed),
        reference_value=float(reference) if reference is not None else None,
        ratio=ratio,
        description=description,
        polarity=polarity,
        stage=stage,
        strength=strength,
        source=source,
    )


def _cpu_saturation(group: GroupObservation, physical_cores: int) -> float | None:
    cpu = group.mean("process_cpu_percent")
    if cpu is None:
        return None
    return cpu / (physical_cores * 100.0)


def _decode_share(group: GroupObservation) -> float | None:
    total = group.mean("total_latency_s")
    decode = group.mean("decode_latency_s")
    if not total or decode is None:
        return None
    return decode / total


def _comparables(ctx: RuleContext, group: GroupObservation) -> list[GroupObservation]:
    return [
        other
        for other in ctx.groups
        if other.model_ref == group.model_ref
        and other.workload_id == group.workload_id
        and other.condition != group.condition
    ]


def _best_higher(
    ctx: RuleContext, group: GroupObservation, metric: str
) -> tuple[GroupObservation, float] | None:
    value = group.mean(metric)
    if value is None or value <= 0:
        return None
    best: tuple[GroupObservation, float] | None = None
    for other in _comparables(ctx, group):
        other_value = other.mean(metric)
        if other_value is None or other_value <= 0:
            continue
        ratio = other_value / value
        if best is None or ratio > best[1]:
            best = (other, ratio)
    return best


def _best_lower(
    ctx: RuleContext, group: GroupObservation, metric: str
) -> tuple[GroupObservation, float] | None:
    value = group.mean(metric)
    if value is None or value <= 0:
        return None
    best: tuple[GroupObservation, float] | None = None
    for other in _comparables(ctx, group):
        other_value = other.mean(metric)
        if other_value is None or other_value <= 0:
            continue
        ratio = value / other_value  # >1 means group is slower than comparable
        if best is None or ratio > best[1]:
            best = (other, ratio)
    return best


def rule_model_footprint(ctx: RuleContext) -> list[CandidateFinding]:
    findings: list[CandidateFinding] = []
    for group in ctx.groups:
        ratio = _footprint_ratio(group)
        if ratio is None or ratio < FOOTPRINT_RATIO_HIGH:
            continue
        supporting = [
            _evidence(
                "model_rss_over_available_ram",
                ratio,
                FOOTPRINT_RATIO_HIGH,
                "model peak RSS is a large fraction of available physical RAM",
                EvidencePolarity.SUPPORTING,
                InferenceStage.END_TO_END,
                EvidenceStrength.MEDIUM,
                "rule.model_footprint.v1",
            )
        ]
        findings.append(
            CandidateFinding(
                kind=BottleneckKind.MODEL_FOOTPRINT,
                stage=InferenceStage.END_TO_END,
                subject=group.key,
                supporting=supporting,
                rationale=[
                    "Relative-significance: model footprint vs available RAM exceeds the "
                    "high-ratio reference; this alone is not a bottleneck and requires "
                    "pressure/paging corroboration."
                ],
                source="rule.model_footprint.v1",
            )
        )
    return findings


def rule_memory_pressure(ctx: RuleContext) -> list[CandidateFinding]:
    pressures = [g.memory_pressure_mean for g in ctx.groups if g.memory_pressure_mean is not None]
    reference = min(pressures) if pressures else None
    findings: list[CandidateFinding] = []
    for group in ctx.groups:
        pressure = group.memory_pressure_mean
        if pressure is None:
            continue
        supporting: list[EvidenceItem] = []
        contradicting: list[EvidenceItem] = []
        if (
            pressure >= PRESSURE_HIGH
            and reference is not None
            and pressure >= reference + PRESSURE_DELTA
        ):
            supporting.append(
                _evidence(
                    "memory_pressure",
                    pressure,
                    reference,
                    "commit pressure is high and materially above the report reference",
                    EvidencePolarity.SUPPORTING,
                    InferenceStage.END_TO_END,
                    EvidenceStrength.MEDIUM,
                    "rule.memory_pressure.v1",
                )
            )
        elif pressure < PRESSURE_LOW:
            contradicting.append(
                _evidence(
                    "memory_pressure",
                    pressure,
                    PRESSURE_LOW,
                    "commit pressure is low; no memory-capacity pressure",
                    EvidencePolarity.CONTRADICTING,
                    InferenceStage.END_TO_END,
                    EvidenceStrength.NONE,
                    "rule.memory_pressure.v1",
                )
            )
        if supporting:
            findings.append(
                CandidateFinding(
                    kind=BottleneckKind.MEMORY_PRESSURE,
                    stage=InferenceStage.END_TO_END,
                    subject=group.key,
                    supporting=supporting,
                    contradicting=contradicting,
                    rationale=[
                        "High commit pressure is a candidate signal; it becomes meaningful "
                        "only with paging or latency-degradation corroboration."
                    ],
                    source="rule.memory_pressure.v1",
                )
            )
    return findings


def rule_paging(ctx: RuleContext) -> list[CandidateFinding]:
    faults = [g.mean("hard_page_faults_per_sec") for g in ctx.groups]
    present = [value for value in faults if value is not None]
    reference = min(present) if present else None
    pressures = [g.memory_pressure_mean for g in ctx.groups if g.memory_pressure_mean is not None]
    pressure_reference = min(pressures) if pressures else None
    findings: list[CandidateFinding] = []
    for group in ctx.groups:
        value = group.mean("hard_page_faults_per_sec")
        if value is None:
            continue
        supporting: list[EvidenceItem] = []
        contradicting: list[EvidenceItem] = []
        threshold = max(PAGING_RELATIVE * (reference or 0.0), PAGING_MIN_PER_SEC)
        if value >= threshold and value >= PAGING_MIN_PER_SEC:
            ratio = value / max(reference or 0.0, 1e-9)
            evidence_reference = max(reference or 0.0, PAGING_MIN_PER_SEC)
            supporting.append(
                _evidence(
                    "hard_page_faults_per_sec",
                    value,
                    evidence_reference,
                    f"hard page-fault rate is {value / evidence_reference:,.1f}x the "
                    f"report reference floor",
                    EvidencePolarity.SUPPORTING,
                    InferenceStage.END_TO_END,
                    EvidenceStrength.HIGH if ratio >= 5.0 else EvidenceStrength.MEDIUM,
                    "rule.paging.v1",
                )
            )
            if (
                group.memory_pressure_mean is not None
                and pressure_reference is not None
                and group.memory_pressure_mean >= pressure_reference + PRESSURE_DELTA
            ):
                supporting.append(
                    _evidence(
                        "memory_pressure",
                        group.memory_pressure_mean,
                        pressure_reference,
                        "elevated commit pressure co-occurs with paging",
                        EvidencePolarity.SUPPORTING,
                        InferenceStage.END_TO_END,
                        EvidenceStrength.MEDIUM,
                        "rule.paging.v1",
                    )
                )
            faster_tps = _best_higher(ctx, group, "decode_tokens_per_sec")
            if faster_tps is not None and faster_tps[1] >= COMPARABLE_LATENCY_RATIO:
                supporting.append(
                    _evidence(
                        "decode_tokens_per_sec",
                        group.mean("decode_tokens_per_sec") or 0.0,
                        faster_tps[0].mean("decode_tokens_per_sec"),
                        f"decode throughput is {faster_tps[1]:.2f}x lower than comparable "
                        f"condition {faster_tps[0].condition or 'reference'}",
                        EvidencePolarity.SUPPORTING,
                        InferenceStage.DECODE,
                        EvidenceStrength.HIGH,
                        "rule.paging.v1",
                    )
                )
            slower_latency = _best_lower(ctx, group, "total_latency_s")
            if slower_latency is not None and slower_latency[1] >= COMPARABLE_LATENCY_RATIO:
                supporting.append(
                    _evidence(
                        "total_latency_s",
                        group.mean("total_latency_s") or 0.0,
                        slower_latency[0].mean("total_latency_s"),
                        f"total latency is {slower_latency[1]:.2f}x higher than comparable "
                        f"condition {slower_latency[0].condition or 'reference'}",
                        EvidencePolarity.SUPPORTING,
                        InferenceStage.END_TO_END,
                        EvidenceStrength.MEDIUM,
                        "rule.paging.v1",
                    )
                )
        else:
            contradicting.append(
                _evidence(
                    "hard_page_faults_per_sec",
                    value,
                    PAGING_MIN_PER_SEC,
                    "hard page-fault rate is at/near the report reference",
                    EvidencePolarity.CONTRADICTING,
                    InferenceStage.END_TO_END,
                    EvidenceStrength.NONE,
                    "rule.paging.v1",
                )
            )
        if supporting:
            findings.append(
                CandidateFinding(
                    kind=BottleneckKind.PAGING,
                    stage=InferenceStage.END_TO_END,
                    subject=group.key,
                    supporting=supporting,
                    contradicting=contradicting,
                    rationale=[
                        "Paging is meaningful only when faults rise materially, pressure is "
                        "elevated, and comparable runs perform differently (causal evidence)."
                    ],
                    source="rule.paging.v1",
                )
            )
    return findings


def rule_cpu_execution(ctx: RuleContext) -> list[CandidateFinding]:
    findings: list[CandidateFinding] = []
    for group in ctx.groups:
        saturation = _cpu_saturation(group, ctx.physical_cores)
        if saturation is None or saturation < CPU_SATURATION_HIGH:
            continue
        gpu = group.mean("gpu_utilization_percent")
        if gpu is not None and gpu >= GPU_UTIL_LOW:
            continue  # GPU is actively sharing the work; not a clean CPU limitation
        if group.offload_configured:
            continue  # handled by the GPU-underutilization rule
        supporting = [
            _evidence(
                "process_cpu_saturation",
                saturation,
                CPU_SATURATION_HIGH,
                f"process CPU is {saturation * 100:.0f}% of physical-core capacity",
                EvidencePolarity.SUPPORTING,
                InferenceStage.DECODE,
                EvidenceStrength.MEDIUM,
                "rule.cpu_execution.v1",
            )
        ]
        contradicting: list[EvidenceItem] = []
        faster = _best_higher(ctx, group, "decode_tokens_per_sec")
        if faster is not None and faster[1] >= COMPARABLE_THROUGHPUT_RATIO:
            supporting.append(
                _evidence(
                    "decode_tokens_per_sec",
                    group.mean("decode_tokens_per_sec") or 0.0,
                    faster[0].mean("decode_tokens_per_sec"),
                    f"decode throughput is {faster[1]:.2f}x lower than comparable "
                    f"condition {faster[0].condition or 'reference'}",
                    EvidencePolarity.SUPPORTING,
                    InferenceStage.DECODE,
                    EvidenceStrength.HIGH,
                    "rule.cpu_execution.v1",
                )
            )
        else:
            contradicting.append(
                _evidence(
                    "comparative_throughput",
                    0.0,
                    COMPARABLE_THROUGHPUT_RATIO,
                    "no materially faster comparable condition demonstrates a better "
                    "attainable operating point",
                    EvidencePolarity.CONTRADICTING,
                    InferenceStage.DECODE,
                    EvidenceStrength.NONE,
                    "rule.cpu_execution.v1",
                )
            )
        findings.append(
            CandidateFinding(
                kind=BottleneckKind.CPU,
                stage=InferenceStage.DECODE,
                subject=group.key,
                supporting=supporting,
                contradicting=contradicting,
                rationale=[
                    "CPU saturation with an idle GPU is meaningful only when a comparable "
                    "condition demonstrates materially higher throughput."
                ],
                source="rule.cpu_execution.v1",
            )
        )
    return findings


def rule_gpu_underutilization(ctx: RuleContext) -> list[CandidateFinding]:
    findings: list[CandidateFinding] = []
    for group in ctx.groups:
        if not group.offload_configured:
            continue
        gpu = group.mean("gpu_utilization_percent")
        saturation = _cpu_saturation(group, ctx.physical_cores)
        if gpu is None or gpu >= GPU_UTIL_LOW:
            continue
        if saturation is None or saturation < CPU_SATURATION_HIGH:
            continue
        supporting = [
            _evidence(
                "gpu_utilization_percent",
                gpu,
                GPU_UTIL_LOW,
                "GPU is configured to hold all layers but is not meaningfully utilized",
                EvidencePolarity.SUPPORTING,
                InferenceStage.DECODE,
                EvidenceStrength.MEDIUM,
                "rule.gpu_underutilization.v1",
            ),
            _evidence(
                "process_cpu_saturation",
                saturation,
                CPU_SATURATION_HIGH,
                "process CPU is saturated while the GPU is idle",
                EvidencePolarity.SUPPORTING,
                InferenceStage.DECODE,
                EvidenceStrength.MEDIUM,
                "rule.gpu_underutilization.v1",
            ),
        ]
        findings.append(
            CandidateFinding(
                kind=BottleneckKind.GPU_UNDERUTILIZATION,
                stage=InferenceStage.DECODE,
                subject=group.key,
                supporting=supporting,
                rationale=[
                    "Offload is configured but execution is CPU-bound (the plan's "
                    "'GPU utilization vs offload configuration state' rule)."
                ],
                source="rule.gpu_underutilization.v1",
            )
        )
    return findings


def rule_gpu_saturation(ctx: RuleContext) -> list[CandidateFinding]:
    findings: list[CandidateFinding] = []
    for group in ctx.groups:
        gpu = group.mean("gpu_utilization_percent")
        if gpu is None or gpu < GPU_UTIL_SATURATED:
            continue
        cv = group.cv("gpu_utilization_percent")
        if cv is not None and cv > GPU_CV_STABLE:
            continue  # not sustained
        share = _decode_share(group)
        generated = group.mean("generated_tokens")
        if share is None or share < 0.5 or generated is None or generated < 32:
            continue
        supporting = [
            _evidence(
                "gpu_utilization_percent",
                gpu,
                GPU_UTIL_SATURATED,
                "GPU compute utilization is sustained near saturation during decode",
                EvidencePolarity.SUPPORTING,
                InferenceStage.DECODE,
                EvidenceStrength.MEDIUM,
                "rule.gpu_saturation.v1",
            ),
            _evidence(
                "decode_latency_share",
                share,
                STAGE_DECODE_DOMINANT,
                "decode dominates end-to-end latency for this workload",
                EvidencePolarity.SUPPORTING,
                InferenceStage.DECODE,
                EvidenceStrength.LOW,
                "rule.gpu_saturation.v1",
            ),
        ]
        contradicting = [
            _evidence(
                "comparative_opportunity",
                0.0,
                1.0,
                "no better attainable operating point is established by the available "
                "evidence; high utilization alone is not an optimization opportunity",
                EvidencePolarity.CONTRADICTING,
                InferenceStage.DECODE,
                EvidenceStrength.NONE,
                "rule.gpu_saturation.v1",
            )
        ]
        findings.append(
            CandidateFinding(
                kind=BottleneckKind.COMPUTE_BOUND,
                stage=InferenceStage.DECODE,
                subject=group.key,
                supporting=supporting,
                contradicting=contradicting,
                rationale=[
                    "GPU is the limiting execution resource for decode, but sustained "
                    "utilization is expected behaviour and is not by itself a meaningful "
                    "optimization opportunity."
                ],
                forced_non_meaningful=True,
                source="rule.gpu_saturation.v1",
            )
        )
    return findings


def _less_vram_faster(
    ctx: RuleContext, group: GroupObservation
) -> tuple[GroupObservation, float] | None:
    """Fastest comparable that uses *less* GPU memory than ``group`` (or ``None``).

    Only same-model/same-workload conditions that observed a lower ``gpu_memory_used``
    are eligible, so an unrelated faster condition cannot be mistaken for evidence that
    GPU memory is the limiting resource.
    """
    used = group.mean("gpu_memory_used_bytes")
    value = group.mean("total_latency_s")
    if used is None or not value:
        return None
    best: tuple[GroupObservation, float] | None = None
    for other in ctx.groups:
        if other is group or other.model_ref != group.model_ref or other.workload_id != group.workload_id:
            continue
        other_used = other.mean("gpu_memory_used_bytes")
        other_latency = other.mean("total_latency_s")
        if other_used is None or other_used >= used or not other_latency or other_latency <= 0:
            continue
        ratio = value / other_latency  # >1 means this group is slower than the comparable
        if best is None or ratio > best[1]:
            best = (other, ratio)
    return best


def rule_gpu_memory_limit(ctx: RuleContext) -> list[CandidateFinding]:
    """GPU VRAM exhaustion for the configured offload.

    A high VRAM-usage ratio is only a *candidate* signal. It becomes a meaningful limit
    only when a comparable same-model/workload condition that uses **less** GPU memory
    performs materially better (comparative, causal evidence). On its own, high VRAM
    usage is recorded as an observation.
    """
    findings: list[CandidateFinding] = []
    if not ctx.vram_bytes or ctx.vram_bytes <= 0:
        return findings
    for group in ctx.groups:
        if not group.offload_configured:
            continue
        used = group.mean("gpu_memory_used_bytes")
        if used is None or used <= 0:
            continue
        ratio = used / ctx.vram_bytes
        if ratio < GPU_VRAM_PRESSURE_HIGH:
            continue
        strength = (
            EvidenceStrength.HIGH
            if ratio >= GPU_VRAM_PRESSURE_EXTREME
            else EvidenceStrength.MEDIUM
        )
        supporting = [
            _evidence(
                "gpu_memory_used_over_vram",
                ratio,
                GPU_VRAM_PRESSURE_HIGH,
                "GPU memory in use is at/near device VRAM capacity while layers are offloaded",
                EvidencePolarity.SUPPORTING,
                InferenceStage.END_TO_END,
                strength,
                "rule.gpu_memory_limit.v1",
            )
        ]
        faster = _less_vram_faster(ctx, group)
        if faster is not None and faster[1] >= COMPARABLE_LATENCY_RATIO:
            supporting.append(
                _evidence(
                    "total_latency_s",
                    group.mean("total_latency_s") or 0.0,
                    faster[0].mean("total_latency_s"),
                    f"total latency is {faster[1]:.2f}x higher than comparable "
                    f"{faster[0].condition or 'reference'} which uses less GPU memory",
                    EvidencePolarity.SUPPORTING,
                    InferenceStage.END_TO_END,
                    EvidenceStrength.HIGH,
                    "rule.gpu_memory_limit.v1",
                )
            )
        findings.append(
            CandidateFinding(
                kind=BottleneckKind.GPU_MEMORY_LIMIT,
                stage=InferenceStage.END_TO_END,
                subject=group.key,
                supporting=supporting,
                rationale=[
                    "VRAM is exhausted for the configured offload, and a comparable "
                    "condition that uses less GPU memory performs materially better. "
                    "Reducing the offloaded layer count frees VRAM for the working set."
                ],
                forced_non_meaningful=len(supporting) < 2,
                source="rule.gpu_memory_limit.v1",
            )
        )
    return findings


def rule_storage_load(ctx: RuleContext) -> list[CandidateFinding]:
    findings: list[CandidateFinding] = []
    for group in ctx.groups:
        load = group.mean("model_load_time_s")
        total = group.mean("total_latency_s")
        if load is None or not total:
            continue
        share = load / (load + total)
        if share < LOAD_SHARE_HIGH:
            continue
        supporting = [
            _evidence(
                "model_load_share",
                share,
                LOAD_SHARE_HIGH,
                "model load time is a substantial fraction of the run",
                EvidencePolarity.SUPPORTING,
                InferenceStage.LOAD,
                EvidenceStrength.LOW,
                "rule.storage_load.v1",
            )
        ]
        contradicting: list[EvidenceItem] = []
        warm = next(
            (
                other
                for other in ctx.groups
                if other.model_ref == group.model_ref
                and other.workload_id == group.workload_id
                and other.mode.value != group.mode.value
            ),
            None,
        )
        warm_load = warm.mean("model_load_time_s") if warm is not None else None
        if group.mode.value == "cold" and warm_load and load >= COLD_WARM_LOAD_RATIO * warm_load:
            supporting.append(
                _evidence(
                    "cold_vs_warm_load_time_s",
                    load,
                    warm_load,
                    "cold load is materially slower than the warm comparable (OS cache / "
                    "storage effect)",
                    EvidencePolarity.SUPPORTING,
                    InferenceStage.LOAD,
                    EvidenceStrength.HIGH,
                    "rule.storage_load.v1",
                )
            )
        else:
            contradicting.append(
                _evidence(
                    "comparative_load",
                    0.0,
                    1.0,
                    "no materially slower cold-load comparable demonstrates a storage/IO "
                    "limitation for startup",
                    EvidencePolarity.CONTRADICTING,
                    InferenceStage.LOAD,
                    EvidenceStrength.NONE,
                    "rule.storage_load.v1",
                )
            )
        findings.append(
            CandidateFinding(
                kind=BottleneckKind.STORAGE,
                stage=InferenceStage.LOAD,
                subject=group.key,
                supporting=supporting,
                contradicting=contradicting,
                rationale=[
                    "Load time is attributed to the load stage; it is meaningful only when "
                    "comparative evidence isolates storage/OS-cache from steady-state execution."
                ],
                forced_non_meaningful=len(supporting) < 2,
                source="rule.storage_load.v1",
            )
        )
    return findings


def rule_stage_dominance(ctx: RuleContext) -> list[CandidateFinding]:
    findings: list[CandidateFinding] = []
    for group in ctx.groups:
        total = group.mean("total_latency_s")
        prefill = group.mean("prefill_latency_s")
        decode = group.mean("decode_latency_s")
        if not total or total <= 0:
            continue
        if prefill is not None and prefill / total >= STAGE_PREFILL_DOMINANT:
            findings.append(
                _stage_finding(group, InferenceStage.PREFILL, prefill / total)
            )
        elif decode is not None and decode / total >= STAGE_DECODE_DOMINANT:
            findings.append(_stage_finding(group, InferenceStage.DECODE, decode / total))
    return findings


def _stage_finding(group: GroupObservation, stage: InferenceStage, share: float) -> CandidateFinding:
    return CandidateFinding(
        kind=BottleneckKind.LATENCY,
        stage=stage,
        subject=group.key,
        supporting=[
            _evidence(
                "stage_latency_share",
                share,
                STAGE_PREFILL_DOMINANT if stage is InferenceStage.PREFILL else STAGE_DECODE_DOMINANT,
                f"{stage.value} dominates end-to-end latency for this workload",
                EvidencePolarity.SUPPORTING,
                stage,
                EvidenceStrength.LOW,
                "rule.stage_dominance.v1",
            )
        ],
        contradicting=[
            _evidence(
                "comparative_degradation",
                0.0,
                1.0,
                "stage dominance is expected for this workload shape and no comparative "
                "degradation demonstrates an opportunity",
                EvidencePolarity.CONTRADICTING,
                stage,
                EvidenceStrength.NONE,
                "rule.stage_dominance.v1",
            )
        ],
        rationale=[
            "Stage attribution only: it records where latency is spent, not that it is "
            "improvable."
        ],
        forced_non_meaningful=True,
        source="rule.stage_dominance.v1",
    )


def rule_external_process(ctx: RuleContext) -> list[CandidateFinding]:
    findings: list[CandidateFinding] = []
    for group in ctx.groups:
        pressure = group.memory_pressure_mean
        if pressure is None or pressure < PRESSURE_HIGH:
            continue
        for run in group.runs:
            state = run.observation.system_state
            if state is None or not state.processes:
                continue
            model_rss = run.observation.metrics.peak_rss_bytes
            if model_rss is None:
                continue
            dominant = max(state.processes, key=lambda process: process.rss_bytes)
            if dominant.rss_bytes > model_rss:
                findings.append(
                    CandidateFinding(
                        kind=BottleneckKind.EXTERNAL_PROCESS,
                        stage=InferenceStage.END_TO_END,
                        subject=group.key,
                        supporting=[
                            _evidence(
                                "external_process_rss_bytes",
                                float(dominant.rss_bytes),
                                float(model_rss),
                                f"process '{dominant.name}' holds more RSS than the model "
                                "under memory pressure",
                                EvidencePolarity.SUPPORTING,
                                InferenceStage.END_TO_END,
                                EvidenceStrength.MEDIUM,
                                "rule.external_process.v1",
                            ),
                            _evidence(
                                "memory_pressure",
                                pressure,
                                PRESSURE_HIGH,
                                "memory pressure is high while the external process dominates RSS",
                                EvidencePolarity.SUPPORTING,
                                InferenceStage.END_TO_END,
                                EvidenceStrength.MEDIUM,
                                "rule.external_process.v1",
                            ),
                        ],
                        rationale=[
                            "An external process dominates memory while pressure is high."
                        ],
                        source="rule.external_process.v1",
                    )
                )
                break
    return findings


def _footprint_ratio(group: GroupObservation) -> float | None:
    rss = group.mean("peak_rss_bytes")
    available = group.mean("available_ram_bytes")
    if rss is None or not available:
        return None
    return rss / available


RULES = (
    rule_model_footprint,
    rule_memory_pressure,
    rule_paging,
    rule_cpu_execution,
    rule_gpu_underutilization,
    rule_gpu_memory_limit,
    rule_gpu_saturation,
    rule_storage_load,
    rule_stage_dominance,
    rule_external_process,
)


def run_rules(ctx: RuleContext) -> list[CandidateFinding]:
    findings: list[CandidateFinding] = []
    for rule in RULES:
        findings.extend(rule(ctx))
    return findings
