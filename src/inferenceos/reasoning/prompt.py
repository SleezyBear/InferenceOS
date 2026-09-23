"""Render the reasoning prompt and build the per-request JSON schema.

The prompt is compact and structured; the schema enumerates the supplied evidence
ids so constrained decoding cannot emit an unknown reference.
"""

from __future__ import annotations

from inferenceos.contracts.enums import (
    BottleneckKind,
    EvidenceStrength,
    InferenceStage,
)
from inferenceos.contracts.reasoning import REASONING_SCHEMA_VERSION, ReasoningDiagnosisContext

SYSTEM_PROMPT = (
    "You are the deep-diagnosis reasoner of InferenceOS, a local LLM inference "
    "optimizer. You receive structured evidence about one or more real inference "
    "runs and must identify the most likely root cause of the observed behaviour.\n"
    "Rules:\n"
    "- Reason only from the supplied structured fields and evidence. Never invent "
    "measurements or metrics; reference evidence ONLY by its id (for example E1).\n"
    "- Do not recommend any configuration change or optimization; describe what "
    "appears to be happening and why.\n"
    "- If the evidence cannot distinguish between explanations, use the root cause "
    "class 'insufficient_evidence' and set further_evidence_required to true.\n"
    "- Respond with a single JSON object that matches the required schema."
)

# JSON object keys the model is allowed to return.
REASONING_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "root_cause_class",
        "affected_stage",
        "contributing_factors",
        "supporting_evidence_ids",
        "contradicting_evidence_ids",
        "confidence",
        "evidence_strength",
        "uncertainty",
        "further_evidence_required",
        "summary",
        "reasoning",
    }
)

# Recommendation-like keys are stripped before validation (never canonical).
RECOMMENDATION_KEYS: frozenset[str] = frozenset(
    {
        "recommendations",
        "recommended_actions",
        "recommended_candidates",
        "actions",
        "optimizations",
        "suggestions",
        "mitigations",
        "fix",
        "fixes",
        "next_steps",
        "configuration_changes",
        "parameter_changes",
        "tuning",
    }
)


def render_system_prompt() -> str:
    return SYSTEM_PROMPT


def render_user_prompt(context: ReasoningDiagnosisContext) -> str:
    lines: list[str] = []
    lines.append("EXECUTION IDENTITY")
    lines.append(
        f"  cpu={context.hardware.cpu.name} cores={context.hardware.cpu.physical_cores}/"
        f"{context.hardware.cpu.logical_cores} ram_bytes={context.hardware.memory.total_bytes}"
    )
    lines.append(
        f"  os={context.hardware.os.caption} {context.hardware.os.version} "
        f"build={context.hardware.os.build}"
    )
    lines.append(
        f"  backend={context.backend.backend_name} {context.backend.backend_version or ''} "
        f"type={_value(context.backend.backend_type)} device={context.backend.device}"
    )
    lines.append(
        f"  offload={context.backend.gpu_layers_offloaded}/{context.backend.gpu_layers_total} "
        f"offload_configured={context.backend.offload_configured}"
    )
    lines.append(
        f"  model={context.model.model_ref} arch={context.model.architecture} "
        f"quant={context.model.quantization} layers={context.model.n_layers} "
        f"heads={context.model.n_attention_heads} kv_heads={context.model.n_kv_heads} "
        f"ctx={context.model.context_length}"
    )
    lines.append("")
    lines.append("WORKLOAD")
    lines.append(
        f"  id={context.workload.workload_id} kind={context.workload.kind.value} "
        f"mode={context.workload.mode.value} stage={context.workload.stage.value}"
    )
    lines.append(
        f"  prompt_tokens={context.workload.prompt_tokens} "
        f"generated_tokens={context.workload.generated_tokens}"
    )
    lines.append(f"  objective={context.workload.objective}")
    lines.append("")
    lines.append("OBSERVED METRICS (mean over runs)")
    for metric in context.metrics:
        lines.append(
            f"  {metric.metric} = {_num(metric.value)} {metric.unit} ({metric.quality.value})"
        )
    lines.append("")
    lines.append("DETERMINISTIC DIAGNOSIS (authoritative; do not contradict without evidence)")
    deterministic = context.deterministic
    lines.append(f"  meaningful_opportunity={deterministic.meaningful_opportunity}")
    lines.append(f"  insufficient_evidence={deterministic.insufficient_evidence}")
    lines.append(f"  summary={deterministic.summary}")
    if deterministic.no_opportunity_reason:
        lines.append(f"  reason={deterministic.no_opportunity_reason}")
    primary = deterministic.primary
    lines.append(
        "  primary="
        + (
            f"{primary.kind.value} stage={_value(primary.stage)} "
            f"confidence={primary.confidence:.2f} strength={primary.evidence_strength.value}"
            if primary is not None
            else "none"
        )
    )
    lines.append("  signals:")
    for finding in list(deterministic.candidates) + list(deterministic.observations):
        lines.append(
            f"    - {finding.kind.value} stage={_value(finding.stage)} "
            f"confidence={finding.confidence:.2f} strength={finding.evidence_strength.value} "
            f"meaningful={finding.is_meaningful}"
        )
    lines.append("")
    lines.append("EVIDENCE (reference by id only)")
    for item in context.evidence:
        lines.append(
            f"  {item.evidence_id} [{item.polarity.value}] {item.metric}="
            f"{_num(item.observed_value)} stage={_value(item.stage)}: {item.description}"
        )
    lines.append("")
    if context.unavailable_metrics:
        lines.append(f"UNAVAILABLE MEASUREMENTS (must not be invented): {', '.join(context.unavailable_metrics)}")
    if context.variance:
        lines.append(
            "RUN-TO-RUN VARIANCE (coefficient of variation): "
            + ", ".join(f"{key}={value:.3f}" for key, value in sorted(context.variance.items()))
        )
    lines.append("")
    lines.append("TASK")
    lines.append(
        "Return one JSON object with keys: schema_version, root_cause_class, "
        "affected_stage, contributing_factors, supporting_evidence_ids, "
        "contradicting_evidence_ids, confidence, evidence_strength, uncertainty, "
        "further_evidence_required, summary, reasoning."
    )
    lines.append(
        "root_cause_class must be one of: "
        + ", ".join(kind.value for kind in BottleneckKind)
    )
    lines.append(
        "affected_stage must be one of: "
        + ", ".join(stage.value for stage in InferenceStage)
    )
    lines.append(
        "evidence_strength must be one of: "
        + ", ".join(strength.value for strength in EvidenceStrength)
    )
    lines.append(
        "supporting_evidence_ids and contradicting_evidence_ids may only contain the "
        "listed evidence ids."
    )
    return "\n".join(lines)


def build_json_schema(allowed_evidence_ids: list[str]) -> dict[str, object]:
    evidence_items: dict[str, object]
    if allowed_evidence_ids:
        evidence_items = {"type": "string", "enum": allowed_evidence_ids}
    else:
        evidence_items = {"type": "string"}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": REASONING_SCHEMA_VERSION},
            "root_cause_class": {"enum": [kind.value for kind in BottleneckKind]},
            "affected_stage": {"enum": [stage.value for stage in InferenceStage]},
            "contributing_factors": {
                "type": "array",
                "items": {"type": "string", "maxLength": 160},
                "maxItems": 5,
            },
            "supporting_evidence_ids": {
                "type": "array",
                "items": evidence_items,
                "maxItems": 6,
            },
            "contradicting_evidence_ids": {
                "type": "array",
                "items": evidence_items,
                "maxItems": 6,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "evidence_strength": {"enum": [strength.value for strength in EvidenceStrength]},
            "uncertainty": {"type": "string", "maxLength": 400},
            "further_evidence_required": {"type": "boolean"},
            "summary": {"type": "string", "maxLength": 800},
            "reasoning": {"type": "string", "maxLength": 1200},
        },
        "required": sorted(REASONING_OUTPUT_KEYS),
    }


def _num(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4g}"


def _value(value: object) -> str:
    if value is None:
        return "none"
    inner = getattr(value, "value", None)
    return str(inner) if inner is not None else str(value)
