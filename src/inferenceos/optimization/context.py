"""Build the real context a candidate generation run reasons over.

The context deliberately reuses the Phase 1-5 artifacts (hardware profile, model
profile, profiling config, workload, diagnosis, optional deep-diagnosis outcome) and
derives an effective bottleneck plus an id-labelled evidence index. Nothing is
fabricated: unavailable metrics are simply absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    BottleneckKind,
    EvidenceStrength,
    ReasoningStatus,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import BaselineReport, ProfilingConfig
from inferenceos.contracts.reasoning import DeepDiagnosisOutcome
from inferenceos.contracts.workload import WorkloadProfile
from inferenceos.diagnosis.groups import GroupObservation, extract_groups
from inferenceos.measurement_validity import validity_blocks
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.safety import strength_rank

INSUFFICIENT_KINDS = {
    BottleneckKind.NONE,
    BottleneckKind.AMBIGUOUS,
    BottleneckKind.INSUFFICIENT_EVIDENCE,
}

OBSERVED_METRICS: tuple[str, ...] = (
    "peak_rss_bytes",
    "available_ram_bytes",
    "total_ram_bytes",
    "process_cpu_percent",
    "system_cpu_percent",
    "gpu_utilization_percent",
    "gpu_memory_used_bytes",
    "hard_page_faults_per_sec",
    "model_load_time_s",
    "decode_tokens_per_sec",
    "total_latency_s",
    "generated_tokens",
    "prompt_tokens",
)


@dataclass(frozen=True)
class OptimizationContext:
    diagnosis: BottleneckDiagnosis
    outcome: DeepDiagnosisOutcome | None
    hardware: HardwareProfile
    report: BaselineReport | None
    group: GroupObservation | None
    model: ModelProfile | None
    config: ProfilingConfig
    workload: WorkloadProfile | None
    effective_kind: BottleneckKind | None
    effective_confidence: float
    effective_strength: EvidenceStrength
    effective_basis: str
    source_bottleneck: Bottleneck | None
    evidence_ids: tuple[str, ...] = ()
    evidence_by_id: dict[str, EvidenceItem] = field(default_factory=dict)
    observed_metrics: dict[str, float] = field(default_factory=dict)
    reasoning_performed: bool = False
    reasoning_cost_s: float | None = None

    def metric(self, name: str) -> float | None:
        return self.observed_metrics.get(name)

    def parameter(self, name: str) -> object:
        return getattr(self.config, name, None)

    @property
    def effective_threads(self) -> int | None:
        """The observed *effective* runtime thread count, or ``None`` when unknown.

        This is authoritative runtime state captured from backend evidence. It is
        deliberately distinct from the *requested* thread configuration and is never
        inferred from CPU topology.
        """
        value = getattr(self.config, "effective_threads", None)
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value)
        return None


def build_optimization_context(
    diagnosis: BottleneckDiagnosis,
    *,
    report: BaselineReport | None,
    outcome: DeepDiagnosisOutcome | None,
    policy: OptimizationPolicy,
) -> OptimizationContext:
    groups = extract_groups(report) if report is not None else []
    group = _select_group(diagnosis, groups)

    kind, confidence, strength, basis, source = _effective_bottleneck(diagnosis, outcome, policy)
    evidence_ids, evidence_by_id = _evidence_index(source)
    metrics = _observed_metrics(group)

    config = diagnosis.context.backend
    model: ModelProfile | None = None
    workload: WorkloadProfile | None = None
    if group is not None:
        model = group.model
        workload = group.workload
        if group.runs:
            config = group.runs[0].context.config
    elif report is not None and report.runs:
        model = report.runs[0].context.model
        config = report.backend

    reasoning = outcome.reasoning if outcome is not None else None
    reasoning_performed = bool(reasoning is not None and reasoning.status is not ReasoningStatus.FAILED)
    reasoning_cost = outcome.lifecycle.reasoning_latency_s if outcome is not None else None

    return OptimizationContext(
        diagnosis=diagnosis,
        outcome=outcome,
        hardware=diagnosis.context.hardware,
        report=report,
        group=group,
        model=model,
        config=config,
        workload=workload,
        effective_kind=kind,
        effective_confidence=confidence,
        effective_strength=strength,
        effective_basis=basis,
        source_bottleneck=source,
        evidence_ids=tuple(evidence_ids),
        evidence_by_id=evidence_by_id,
        observed_metrics=metrics,
        reasoning_performed=reasoning_performed,
        reasoning_cost_s=reasoning_cost,
    )


def _effective_bottleneck(
    diagnosis: BottleneckDiagnosis,
    outcome: DeepDiagnosisOutcome | None,
    policy: OptimizationPolicy,
) -> tuple[BottleneckKind | None, float, EvidenceStrength, str, Bottleneck | None]:
    if validity_blocks(diagnosis.measurement_validity):
        # B1: a confounded power/performance measurement must never become an
        # optimization signal, regardless of which findings the rules produced.
        return (
            None,
            diagnosis.confidence,
            diagnosis.evidence_strength,
            "measurement_confounded",
            None,
        )

    if outcome is not None:
        reasoning = outcome.reasoning
        if (
            reasoning is not None
            and reasoning.status is not ReasoningStatus.FAILED
            and reasoning.validation_ok
            and reasoning.result is not None
            and outcome.final_root_cause is not None
            and outcome.final_root_cause not in INSUFFICIENT_KINDS
            and outcome.final_confidence >= policy.min_diagnosis_confidence
            and strength_rank(outcome.final_evidence_strength)
            >= strength_rank(policy.min_evidence_strength)
        ):
            return (
                outcome.final_root_cause,
                outcome.final_confidence,
                outcome.final_evidence_strength,
                "reasoning",
                reasoning.bottleneck,
            )

    if (
        diagnosis.meaningful_opportunity
        and diagnosis.primary is not None
        and diagnosis.primary.kind not in INSUFFICIENT_KINDS
        and diagnosis.primary.confidence >= policy.min_diagnosis_confidence
        and strength_rank(diagnosis.primary.evidence_strength)
        >= strength_rank(policy.min_evidence_strength)
    ):
        return (
            diagnosis.primary.kind,
            diagnosis.primary.confidence,
            diagnosis.primary.evidence_strength,
            "deterministic",
            diagnosis.primary,
        )
    return None, diagnosis.confidence, diagnosis.evidence_strength, "none", None


def _select_group(
    diagnosis: BottleneckDiagnosis, groups: list[GroupObservation]
) -> GroupObservation | None:
    if not groups:
        return None
    subject = None
    if diagnosis.primary is not None:
        subject = diagnosis.primary.subject
    if subject:
        for group in groups:
            if group.key == subject:
                return group
    for group in groups:
        if group.model_ref in diagnosis.context.model_refs and (
            not diagnosis.context.workload_ids
            or group.workload_id in diagnosis.context.workload_ids
        ):
            return group
    return groups[0]


def _evidence_index(source: Bottleneck | None) -> tuple[list[str], dict[str, EvidenceItem]]:
    ids: list[str] = []
    by_id: dict[str, EvidenceItem] = {}
    if source is None:
        return ids, by_id
    seen: set[tuple[str, float, str, str]] = set()
    items = list(source.supporting_evidence) + list(source.contradicting_evidence)
    for item in items:
        key = (item.metric, round(item.observed_value, 6), item.polarity.value, item.description)
        if key in seen:
            continue
        seen.add(key)
        evidence_id = f"E{len(ids) + 1}"
        ids.append(evidence_id)
        by_id[evidence_id] = item
    return ids, by_id


def _observed_metrics(group: GroupObservation | None) -> dict[str, float]:
    if group is None:
        return {}
    metrics: dict[str, float] = {}
    for name in OBSERVED_METRICS:
        value = group.mean(name)
        if value is not None:
            metrics[name] = float(value)
    if group.memory_pressure_mean is not None:
        metrics["memory_pressure"] = float(group.memory_pressure_mean)
    return metrics
