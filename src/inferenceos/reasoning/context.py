"""Build a compact, structured deep-diagnosis context from Phase 4 + Phase 3.

The reasoning model is given structured fields and a bounded, id-labelled evidence
list — not a telemetry dump. Metrics that are unavailable are listed explicitly and
must not be invented by the model.
"""

from __future__ import annotations

from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    EvidencePolarity,
    InferenceStage,
    MeasurementMode,
    WorkloadKind,
)
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.reasoning import (
    ReasoningBackendContext,
    ReasoningDiagnosisContext,
    ReasoningEvidence,
    ReasoningMetric,
    ReasoningModelContext,
    ReasoningWorkloadContext,
)
from inferenceos.diagnosis.groups import GroupObservation, extract_groups

_METRIC_UNITS: dict[str, str] = {
    "model_load_time_s": "s",
    "ttft_s": "s",
    "prefill_latency_s": "s",
    "decode_latency_s": "s",
    "total_latency_s": "s",
    "process_wall_s": "s",
    "prompt_tokens_per_sec": "tokens/s",
    "decode_tokens_per_sec": "tokens/s",
    "prompt_tokens": "tokens",
    "generated_tokens": "tokens",
    "peak_rss_bytes": "bytes",
    "process_cpu_percent": "percent",
    "system_cpu_percent": "percent",
    "gpu_utilization_percent": "percent",
    "gpu_memory_used_bytes": "bytes",
    "hard_page_faults_per_sec": "faults/s",
}

_CONTEXT_METRICS: tuple[str, ...] = (
    "model_load_time_s",
    "ttft_s",
    "prefill_latency_s",
    "decode_latency_s",
    "prompt_tokens_per_sec",
    "decode_tokens_per_sec",
    "total_latency_s",
    "peak_rss_bytes",
    "process_cpu_percent",
    "system_cpu_percent",
    "gpu_utilization_percent",
    "hard_page_faults_per_sec",
)

_VARIANCE_METRICS: tuple[str, ...] = (
    "decode_tokens_per_sec",
    "total_latency_s",
    "gpu_utilization_percent",
)

DEFAULT_OBJECTIVE = (
    "Identify the most likely root cause of the observed inference behaviour and "
    "state the uncertainty. Do not recommend any configuration change."
)


def _ordered_findings(diagnosis: BottleneckDiagnosis) -> list[Bottleneck]:
    findings: list[Bottleneck] = []
    if diagnosis.primary is not None:
        findings.append(diagnosis.primary)
    findings.extend(diagnosis.candidates)
    findings.extend(diagnosis.observations)
    return findings


def _focus_subjects(diagnosis: BottleneckDiagnosis) -> list[str]:
    subjects: list[str] = []
    for finding in _ordered_findings(diagnosis):
        if finding.subject and finding.subject not in subjects:
            subjects.append(finding.subject)
    return subjects


def _select_groups(groups: list[GroupObservation], subjects: list[str]) -> list[GroupObservation]:
    matched = [group for group in groups if group.key in subjects]
    if matched:
        involved = {group.key for group in matched}
        # Include same-model/workload comparables that the diagnosis reasoned over.
        for group in groups:
            for subject in subjects:
                model_ref, _sep, rest = subject.partition("/")
                workload_id = rest.partition("/")[0]
                if group.model_ref == model_ref and group.workload_id == workload_id:
                    involved.add(group.key)
        matched = [group for group in groups if group.key in involved]
    return matched or groups[:3]


def build_reasoning_context(
    diagnosis: BottleneckDiagnosis,
    report: BaselineReport | None,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    max_evidence: int = 12,
) -> ReasoningDiagnosisContext:
    groups = extract_groups(report) if report is not None else []
    subjects = _focus_subjects(diagnosis)
    involved = _select_groups(groups, subjects)

    model_profile = None
    config = diagnosis.context.backend
    if involved:
        model_profile = involved[0].model
        if involved[0].runs:
            config = involved[0].runs[0].context.config
    elif report is not None and report.runs:
        model_profile = report.runs[0].context.model
        config = report.backend

    workload_profile = involved[0].workload if involved else None

    focus_stage = _focus_stage(diagnosis)
    model_context = ReasoningModelContext(
        model_ref=model_profile.name if model_profile is not None else _first(diagnosis.context.model_refs),
        architecture=model_profile.architecture if model_profile is not None else "unknown",
        quantization=(
            model_profile.quantization.value if model_profile is not None else "unknown"
        ),
        format=model_profile.format if model_profile is not None else "unknown",
        n_layers=model_profile.n_layers if model_profile is not None else None,
        n_attention_heads=model_profile.n_attention_heads if model_profile is not None else None,
        n_kv_heads=model_profile.n_kv_heads if model_profile is not None else None,
        context_length=(
            model_profile.trained_context_length if model_profile is not None else None
        ),
        parameter_count=model_profile.parameter_count if model_profile is not None else None,
    )
    backend_context = ReasoningBackendContext(
        backend_name=config.backend_name,
        backend_version=config.backend_version,
        backend_type=config.backend_type,
        device=config.device,
        gpu_layers_offloaded=config.gpu_layers_offloaded,
        gpu_layers_total=config.gpu_layers_total,
        offload_configured=bool(
            config.gpu_layers_offloaded
            and config.gpu_layers_total
            and config.gpu_layers_offloaded == config.gpu_layers_total
        ),
    )
    workload_context = ReasoningWorkloadContext(
        workload_id=workload_profile.workload_id if workload_profile is not None else _first(diagnosis.context.workload_ids),
        kind=workload_profile.kind if workload_profile is not None else WorkloadKind.UNKNOWN,
        objective=objective,
        mode=(
            workload_profile.modes[0]
            if (workload_profile is not None and workload_profile.modes)
            else MeasurementMode.WARM
        ),
        stage=focus_stage,
        prompt_tokens=_prompt_tokens(involved),
        generated_tokens=_generated_tokens(involved),
        summary=_workload_summary(involved),
    )

    metrics = _build_metrics(involved)
    evidence = _build_evidence(diagnosis, involved, max_evidence=max_evidence)
    variance = _build_variance(involved)

    return ReasoningDiagnosisContext(
        source_diagnosis_id=diagnosis.diagnosis_id,
        source_report_path=diagnosis.context.source_report_path,
        hardware=diagnosis.context.hardware,
        model=model_context,
        backend=backend_context,
        workload=workload_context,
        metrics=metrics,
        evidence=evidence,
        deterministic=diagnosis,
        unavailable_metrics=list(diagnosis.unavailable_measurements),
        variance=variance,
    )


def _first(values: list[str]) -> str:
    return values[0] if values else "unknown"


def _focus_stage(diagnosis: BottleneckDiagnosis) -> InferenceStage:
    for finding in _ordered_findings(diagnosis):
        if finding.stage is not None:
            return finding.stage
    return InferenceStage.END_TO_END


def _prompt_tokens(groups: list[GroupObservation]) -> int | None:
    for group in groups:
        value = group.mean("prompt_tokens")
        if value is not None:
            return int(value)
    return None


def _generated_tokens(groups: list[GroupObservation]) -> int | None:
    for group in groups:
        value = group.mean("generated_tokens")
        if value is not None:
            return int(value)
    return None


def _workload_summary(groups: list[GroupObservation]) -> str:
    if not groups:
        return ""
    group = groups[0]
    return f"{group.workload.kind.value} workload, mode={group.mode.value}"


def _build_metrics(groups: list[GroupObservation]) -> list[ReasoningMetric]:
    metrics: list[ReasoningMetric] = []
    for group in groups[:2]:
        for name in _CONTEXT_METRICS:
            value = group.mean(name)
            if value is None:
                continue
            metrics.append(
                ReasoningMetric(
                    metric=f"{group.key}:{name}",
                    value=value,
                    unit=_METRIC_UNITS.get(name, "value"),
                    description=f"mean over {group.n} run(s)",
                )
            )
        if group.memory_pressure_mean is not None:
            metrics.append(
                ReasoningMetric(
                    metric=f"{group.key}:memory_pressure",
                    value=group.memory_pressure_mean,
                    unit="ratio",
                    description=f"mean over {group.n} run(s)",
                )
            )
    return metrics


def _build_variance(groups: list[GroupObservation]) -> dict[str, float]:
    variance: dict[str, float] = {}
    for group in groups[:2]:
        for name in _VARIANCE_METRICS:
            cv = group.cv(name)
            if cv is not None:
                variance[f"{group.key}:{name}"] = cv
    return variance


def _build_evidence(
    diagnosis: BottleneckDiagnosis,
    groups: list[GroupObservation],
    *,
    max_evidence: int,
) -> list[ReasoningEvidence]:
    raw: list[tuple[EvidenceItem, str]] = []
    for finding in _ordered_findings(diagnosis):
        for item in finding.supporting_evidence:
            raw.append((item, finding.subject or finding.summary))
        for item in finding.contradicting_evidence:
            raw.append((item, finding.subject or finding.summary))

    seen: set[tuple[str, float, str, str]] = set()
    evidence: list[ReasoningEvidence] = []
    for item, subject in raw:
        key = (item.metric, round(item.observed_value, 6), item.polarity.value, item.description)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            ReasoningEvidence(
                evidence_id=f"E{len(evidence) + 1}",
                metric=item.metric,
                observed_value=item.observed_value,
                reference_value=item.reference_value,
                polarity=item.polarity,
                stage=item.stage,
                source=item.source,
                description=f"{item.description} [{subject}]" if subject else item.description,
            )
        )
        if len(evidence) >= max_evidence:
            return evidence

    for group in groups[:1]:
        for name in _CONTEXT_METRICS:
            value = group.mean(name)
            if value is None:
                continue
            evidence.append(
                ReasoningEvidence(
                    evidence_id=f"E{len(evidence) + 1}",
                    metric=name,
                    observed_value=value,
                    polarity=EvidencePolarity.NEUTRAL,
                    stage=InferenceStage.END_TO_END,
                    source="group_metrics",
                    description=f"{group.key} mean {name}",
                )
            )
            if len(evidence) >= max_evidence:
                return evidence
    return evidence
