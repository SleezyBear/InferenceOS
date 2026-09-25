"""Human-readable rendering of a deterministic bottleneck diagnosis."""

from __future__ import annotations

from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.diagnosis import BottleneckDiagnosis


def render_diagnosis(diagnosis: BottleneckDiagnosis) -> str:
    lines: list[str] = []
    context = diagnosis.context
    lines.append("=== Deterministic bottleneck diagnosis ===")
    lines.append(f"detector      : {context.detector_version}")
    lines.append(f"source report : {context.source_report_path}")
    if context.source_report_generated_at is not None:
        lines.append(f"report time   : {context.source_report_generated_at.isoformat()}")
    lines.append(
        f"hardware      : {context.hardware.cpu.name} | "
        f"{context.hardware.cpu.physical_cores}c/{context.hardware.cpu.logical_cores}t"
    )
    lines.append(f"models        : {', '.join(context.model_refs) or 'n/a'}")
    lines.append(f"workloads     : {', '.join(context.workload_ids) or 'n/a'}")
    if context.conditions:
        lines.append(f"conditions    : {', '.join(context.conditions)}")
    lines.append("")
    lines.append(
        f"meaningful opportunity : {diagnosis.meaningful_opportunity} "
        f"(confidence={diagnosis.confidence:.2f}, strength={diagnosis.evidence_strength.value})"
    )
    lines.append(f"insufficient evidence  : {diagnosis.insufficient_evidence}")
    lines.append(f"escalation required    : {diagnosis.escalation_required}")
    validity = diagnosis.measurement_validity
    if validity is not None:
        lines.append(
            f"measurement validity   : {validity.validity.value} "
            f"(power={validity.power_state.value}, "
            f"cpu_ratio={validity.cpu_performance_ratio})"
        )
    lines.append(f"summary                : {diagnosis.summary}")
    if validity is not None and validity.validity.value != "valid":
        for reason in validity.reasons:
            lines.append(f"  validity: {reason}")
    lines.append("")

    if diagnosis.primary is not None:
        lines.append("--- Primary finding ---")
        _render_bottleneck(lines, diagnosis.primary)
    if diagnosis.candidates:
        lines.append("--- Other candidate findings ---")
        for candidate in diagnosis.candidates:
            _render_bottleneck(lines, candidate)
    if diagnosis.observations:
        lines.append("--- Non-meaningful observations (preserved evidence) ---")
        for observation in diagnosis.observations:
            _render_bottleneck(lines, observation)

    lines.append("--- Reasoning chain ---")
    for step in diagnosis.reasoning:
        lines.append(f"  - {step}")

    if diagnosis.unavailable_measurements:
        lines.append("--- Unavailable measurements ---")
        for item in diagnosis.unavailable_measurements:
            lines.append(f"  - {item}")
    if diagnosis.limitations:
        lines.append("--- Limitations ---")
        for item in diagnosis.limitations:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def _render_bottleneck(lines: list[str], bottleneck: Bottleneck) -> None:
    stage = bottleneck.stage.value if bottleneck.stage else "unknown"
    lines.append(
        f"[{bottleneck.kind.value}] stage={stage} "
        f"confidence={bottleneck.confidence:.2f} strength={bottleneck.evidence_strength.value} "
        f"meaningful={bottleneck.is_meaningful}"
    )
    lines.append(f"    {bottleneck.summary}")
    for rationale in bottleneck.rationale:
        lines.append(f"    rationale: {rationale}")
    for item in bottleneck.supporting_evidence:
        lines.append(f"    + {_evidence_line(item)}")
    for item in bottleneck.contradicting_evidence:
        lines.append(f"    - {_evidence_line(item)}")


def _evidence_line(item: EvidenceItem) -> str:
    reference = (
        f" ref={item.reference_value:g}" if item.reference_value is not None else ""
    )
    ratio = f" ratio={item.ratio:.2f}" if item.ratio is not None else ""
    return (
        f"{item.metric}={item.observed_value:g}{reference}{ratio} "
        f"({item.polarity.value}, {item.source or 'n/a'}): {item.description}"
    )
