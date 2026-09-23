"""Deterministic grounding and safety validation of a reasoning result.

This is the safety boundary: the model may only reference supplied evidence and may
not claim measurements that were not observed or that were explicitly unavailable.
Optimization-type claims are rejected (Phase 5 is diagnosis only).
"""

from __future__ import annotations

import re

from inferenceos.contracts.reasoning import ReasoningDiagnosisContext, ReasoningResult

# Numbers with a decimal point, or integers carrying a measurement unit.
_NUMBER_RE = re.compile(
    r"\d+\.\d+|\d+(?:\.\d+)?\s*(?:%|ms|MiB|GiB|KiB|tokens?|tok/s|t/s)"
)

# Multi-word imperative patterns that indicate an optimization recommendation.
_RECOMMENDATION_PATTERNS: tuple[str, ...] = (
    "reduce the",
    "increase the",
    "decrease the",
    "change the",
    "set the",
    "lower the",
    "raise the",
    "adjust the",
    "tune the",
    "move the",
    "close the",
    "kill the",
    "terminate the",
    "switch to",
    "enable the",
    "disable the",
    "recommend",
    "you should",
    "try to",
    "consider using",
    "consider reducing",
    "consider increasing",
)


class ReasoningValidationError(ValueError):
    """Raised when a reasoning result is not grounded or violates constraints."""


def _grounded(value: float, allowed: list[float]) -> bool:
    for candidate in allowed:
        tolerance = max(1e-9, abs(candidate) * 0.01)
        if abs(value - candidate) <= tolerance:
            return True
    return False


def _expanded(values: list[float]) -> list[float]:
    """Add the rounded forms the prompt actually displays, so citations match."""
    out: list[float] = []
    for value in values:
        out.append(value)
        out.append(round(value, 2))
        try:
            out.append(float(f"{value:.4g}"))
        except (ValueError, OverflowError):  # pragma: no cover - defensive
            continue
    return out


def _leading_float(token: str) -> float | None:
    match = re.match(r"\d+(?:\.\d+)?", token)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:  # pragma: no cover - defensive
        return None


def validate_reasoning_result(
    result: ReasoningResult,
    context: ReasoningDiagnosisContext,
) -> None:
    allowed_ids = {item.evidence_id for item in context.evidence}
    referenced = set(result.supporting_evidence_ids) | set(result.contradicting_evidence_ids)
    unknown = referenced - allowed_ids
    if unknown:
        raise ReasoningValidationError(f"unknown evidence ids: {sorted(unknown)}")

    allowed_values: list[float] = [result.confidence, 0.0, 1.0]
    if context.workload.prompt_tokens is not None:
        allowed_values.append(float(context.workload.prompt_tokens))
    if context.workload.generated_tokens is not None:
        allowed_values.append(float(context.workload.generated_tokens))
    for item in context.evidence:
        allowed_values.append(item.observed_value)
        if item.reference_value is not None:
            allowed_values.append(item.reference_value)
    for metric in context.metrics:
        if metric.value is not None:
            allowed_values.append(metric.value)
    allowed_values.extend(context.variance.values())
    # Identity / configuration numbers the context supplied.
    hardware = context.hardware
    allowed_values.extend(
        float(value)
        for value in (
            hardware.cpu.physical_cores,
            hardware.cpu.logical_cores,
            hardware.memory.total_bytes,
            hardware.memory.available_bytes,
        )
    )
    model = context.model
    allowed_values.extend(
        float(value)
        for value in (
            model.n_layers,
            model.n_attention_heads,
            model.n_kv_heads,
            model.context_length,
            model.parameter_count,
        )
        if value is not None
    )
    backend = context.backend
    allowed_values.extend(
        float(value)
        for value in (backend.gpu_layers_offloaded, backend.gpu_layers_total)
        if value is not None
    )
    # Deterministic signal confidences are supplied and may be cited.
    deterministic = context.deterministic
    findings = list(deterministic.candidates) + list(deterministic.observations)
    if deterministic.primary is not None:
        findings.append(deterministic.primary)
    allowed_values.extend(finding.confidence for finding in findings)

    allowed_values = _expanded(allowed_values)
    text = " ".join([result.summary, result.reasoning, *result.contributing_factors])
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        value = _leading_float(token)
        if value is None:
            continue
        if not _grounded(value, allowed_values):
            raise ReasoningValidationError(f"ungrounded numeric claim: {token!r}")

    lowered = text.lower()
    for pattern in _RECOMMENDATION_PATTERNS:
        if pattern in lowered:
            raise ReasoningValidationError(
                f"optimization recommendation detected: {pattern!r}"
            )
