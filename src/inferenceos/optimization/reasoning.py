"""Optional reasoning-assisted candidate proposal.

The local reasoning engine may propose *registered* actions for cases where
deterministic mapping is insufficient. The LLM is never authoritative over safety:
it cannot invent an action, parameter, command or measurement, and every proposal is
validated against the registry, the parameter schema, and the supplied evidence
before it is considered. Any failure falls back to deterministic generation.

This module uses ``ReasoningProvider`` directly; it never constructs a command from
model output.
"""

from __future__ import annotations

import json
import re
from typing import Any

from inferenceos.contracts.enums import (
    OptimizationScope,
    ParameterType,
)
from inferenceos.contracts.optimization import (
    CandidateAction,
    ReasoningProposalOutcome,
)
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.registry import ActionRegistry, UnknownActionError
from inferenceos.reasoning.parse import _balanced_objects
from inferenceos.reasoning.providers.base import ReasoningProvider

PROPOSAL_SCHEMA_VERSION = "1.0"
PROPOSAL_KEYS = frozenset({"schema_version", "proposed_actions", "summary"})
PROPOSAL_ACTION_KEYS = frozenset(
    {"action_id", "parameter_names", "parameter_values", "evidence_ids", "rationale"}
)
_OFFERED_SCOPES = {
    OptimizationScope.TARGET_MODEL_RUNTIME,
    OptimizationScope.TARGET_INFERENCE_PROCESS,
    OptimizationScope.CURRENT_WORKLOAD,
}
_NUMBER_RE = re.compile(
    r"\d+\.\d+|\d+(?:\.\d+)?\s*(?:%|ms|MiB|GiB|KiB|tokens?|tok/s|t/s)"
)


class ProposalParseError(ValueError):
    """Raised when a proposal payload is structurally invalid."""


def offered_actions(registry: ActionRegistry) -> list[CandidateAction]:
    return [
        action
        for action in registry.all()
        if action.enabled and action.scope in _OFFERED_SCOPES
    ]


def build_proposal_schema(
    actions: list[CandidateAction], evidence_ids: list[str]
) -> dict[str, object]:
    action_ids = [action.action_id for action in actions]
    parameter_names = sorted(
        {spec.name for action in actions for spec in action.parameters}
    )
    evidence_items: dict[str, object] = (
        {"type": "string", "enum": evidence_ids} if evidence_ids else {"type": "string"}
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": PROPOSAL_SCHEMA_VERSION},
            "proposed_actions": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "action_id": {"enum": action_ids},
                        "parameter_names": {
                            "type": "array",
                            "items": {"type": "string", "enum": parameter_names},
                            "maxItems": 4,
                        },
                        "parameter_values": {
                            "type": "array",
                            "items": {
                                "type": ["integer", "number", "boolean", "string"]
                            },
                            "maxItems": 4,
                        },
                        "evidence_ids": {
                            "type": "array",
                            "items": evidence_items,
                            "maxItems": 6,
                        },
                        "rationale": {"type": "string", "maxLength": 400},
                    },
                    "required": sorted(PROPOSAL_ACTION_KEYS),
                },
            },
            "summary": {"type": "string", "maxLength": 600},
        },
        "required": sorted(PROPOSAL_KEYS),
    }


def render_system_prompt() -> str:
    return (
        "You are the candidate-proposal component of InferenceOS, a local LLM "
        "inference optimizer. You may ONLY choose action_id values from the supplied "
        "registered action list, and ONLY bind parameters declared for that action. "
        "Never invent actions, parameters, shell/PowerShell commands, executables, "
        "registry edits, or measurements. Reference only the supplied evidence ids. "
        "If the evidence does not support any action, return an empty proposed_actions "
        "list. Respond with a single JSON object matching the required schema."
    )


def render_user_prompt(ctx: OptimizationContext, actions: list[CandidateAction]) -> str:
    lines: list[str] = []
    lines.append("CONTEXT")
    lines.append(
        f"  hardware cpu={ctx.hardware.cpu.name} cores={ctx.hardware.cpu.physical_cores}/"
        f"{ctx.hardware.cpu.logical_cores} ram_bytes={ctx.hardware.memory.total_bytes}"
    )
    model = ctx.model
    if model is not None:
        lines.append(
            f"  model={model.name} arch={model.architecture} quant={model.quantization.value} "
            f"layers={model.n_layers} ctx={model.trained_context_length}"
        )
    if ctx.workload is not None:
        lines.append(f"  workload={ctx.workload.workload_id} kind={ctx.workload.kind.value}")
    lines.append(f"  backend={ctx.config.backend_name} device={ctx.config.device}")
    lines.append(
        f"  config context_length={ctx.config.context_length} threads={ctx.config.threads} "
        f"use_mmap={ctx.config.use_mmap} gpu_layers={ctx.config.gpu_layers_offloaded}"
        f"/{ctx.config.gpu_layers_total}"
    )
    lines.append("")
    lines.append("DIAGNOSIS")
    lines.append(f"  meaningful_opportunity={ctx.diagnosis.meaningful_opportunity}")
    lines.append(f"  insufficient_evidence={ctx.diagnosis.insufficient_evidence}")
    lines.append(f"  summary={ctx.diagnosis.summary}")
    if ctx.effective_kind is not None:
        lines.append(
            f"  effective={ctx.effective_kind.value} confidence={ctx.effective_confidence:.2f} "
            f"strength={ctx.effective_strength.value} basis={ctx.effective_basis}"
        )
    lines.append("")
    lines.append("OBSERVED METRICS (mean over runs)")
    for name, value in sorted(ctx.observed_metrics.items()):
        lines.append(f"  {name}={value:.4g}")
    lines.append("")
    lines.append("EVIDENCE (reference by id only)")
    for evidence_id in ctx.evidence_ids:
        item = ctx.evidence_by_id[evidence_id]
        lines.append(
            f"  {evidence_id} [{item.polarity.value}] {item.metric}={item.observed_value:.4g}: "
            f"{item.description}"
        )
    lines.append("")
    lines.append("REGISTERED ACTIONS (choose action_id only from this list)")
    for action in actions:
        params = ", ".join(
            f"{spec.name}:{spec.type.value}"
            + (f"[{spec.minimum},{spec.maximum}]" if spec.minimum is not None or spec.maximum is not None else "")
            + (f"={spec.choices}" if spec.choices else "")
            for spec in action.parameters
        )
        lines.append(
            f"  {action.action_id}: {action.description} (surface={action.surface.value}, "
            f"params={params or 'none'})"
        )
    lines.append("")
    lines.append("TASK")
    lines.append(
        "Return one JSON object with keys schema_version, proposed_actions, summary. "
        "Each proposed action has keys action_id, parameter_names, parameter_values, "
        "evidence_ids, rationale. proposed_actions may be empty."
    )
    return "\n".join(lines)


def _extract_top_object(text: str) -> dict[str, Any] | None:
    """Return the outermost/largest balanced JSON object (proposals nest objects)."""
    best: dict[str, Any] | None = None
    best_len = -1
    for candidate in _balanced_objects(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and len(candidate) > best_len:
            best = parsed
            best_len = len(candidate)
    return best


def parse_proposals(
    text: str,
    *,
    registry: ActionRegistry,
    allowed_evidence_ids: list[str],
    allowed_values: list[float],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Strictly parse and validate proposals.

    Returns ``(accepted, rejections, proposed_action_ids)`` where
    ``proposed_action_ids`` records every action id the model emitted (accepted or
    not) for audit.
    """
    obj = _extract_top_object(text)
    if obj is None:
        raise ProposalParseError("no JSON object found in model output")
    unknown = sorted(set(obj) - PROPOSAL_KEYS)
    if unknown:
        raise ProposalParseError(f"unexpected output keys: {unknown}")
    missing = sorted(PROPOSAL_KEYS - set(obj))
    if missing:
        raise ProposalParseError(f"missing required output keys: {missing}")
    if obj.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
        raise ProposalParseError(f"schema_version mismatch: {obj.get('schema_version')!r}")
    raw_actions = obj.get("proposed_actions")
    if not isinstance(raw_actions, list):
        raise ProposalParseError("proposed_actions must be a list")

    accepted: list[dict[str, Any]] = []
    rejections: list[str] = []
    proposed_ids: list[str] = []
    for index, raw in enumerate(raw_actions):
        label = f"proposal[{index}]"
        if not isinstance(raw, dict):
            rejections.append(f"{label}: not an object")
            continue
        unknown = sorted(set(raw) - PROPOSAL_ACTION_KEYS)
        if unknown:
            rejections.append(f"{label}: unexpected keys {unknown}")
            continue
        if set(raw) != PROPOSAL_ACTION_KEYS:
            rejections.append(f"{label}: missing keys")
            continue
        action_id = raw.get("action_id")
        if not isinstance(action_id, str):
            rejections.append(f"{label}: action_id must be a string")
            continue
        proposed_ids.append(action_id)
        try:
            action = registry.get(action_id)
        except UnknownActionError:
            rejections.append(f"{label}: unregistered action {action_id!r} rejected")
            continue

        names = raw.get("parameter_names")
        values = raw.get("parameter_values")
        if not isinstance(names, list) or not isinstance(values, list) or len(names) != len(values):
            rejections.append(f"{label}: parameter_names/parameter_values mismatch")
            continue
        params, error = _bind_parameters(action, names, values)
        if error is not None:
            rejections.append(f"{label}: {error}")
            continue

        evidence_ids = raw.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not all(
            isinstance(item, str) for item in evidence_ids
        ):
            rejections.append(f"{label}: evidence_ids must be a string list")
            continue
        unknown_evidence = sorted(set(evidence_ids) - set(allowed_evidence_ids))
        if unknown_evidence:
            rejections.append(f"{label}: hallucinated evidence ids {unknown_evidence}")
            continue

        rationale = raw.get("rationale")
        if not isinstance(rationale, str):
            rejections.append(f"{label}: rationale must be a string")
            continue
        if _has_ungrounded_number(rationale, allowed_values):
            rejections.append(f"{label}: fabricated measurement in rationale")
            continue

        accepted.append(
            {
                "action_id": action_id,
                "parameters": params,
                "evidence_ids": list(evidence_ids),
                "rationale": rationale,
            }
        )
    return accepted, rejections, proposed_ids


def _bind_parameters(
    action: CandidateAction, names: list[object], values: list[object]
) -> tuple[dict[str, object], str | None]:
    specs = {spec.name: spec for spec in action.parameters}
    params: dict[str, object] = {}
    for name, value in zip(names, values):
        if not isinstance(name, str) or name not in specs:
            return {}, f"undeclared parameter {name!r}"
        spec = specs[name]
        coerced, error = _coerce(spec.type, value, spec.minimum, spec.maximum, spec.choices)
        if error is not None:
            return {}, f"parameter {name}: {error}"
        params[name] = coerced
    return params, None


def _coerce(
    ptype: ParameterType,
    value: object,
    minimum: float | None,
    maximum: float | None,
    choices: list[str],
) -> tuple[object, str | None]:
    if ptype is ParameterType.BOOLEAN:
        if not isinstance(value, bool):
            return None, "expected boolean"
        return value, None
    if ptype is ParameterType.INTEGER:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, "expected integer"
        number = float(value)
        if number != int(number):
            return None, "expected integer"
        integer = int(number)
        if minimum is not None and integer < minimum:
            return None, f"below minimum {minimum:g}"
        if maximum is not None and integer > maximum:
            return None, f"above maximum {maximum:g}"
        return integer, None
    if ptype is ParameterType.FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, "expected number"
        number = float(value)
        if minimum is not None and number < minimum:
            return None, f"below minimum {minimum:g}"
        if maximum is not None and number > maximum:
            return None, f"above maximum {maximum:g}"
        return number, None
    if not isinstance(value, str):
        return None, "expected string"
    if choices and value not in choices:
        return None, f"not in allowed choices {choices}"
    return value, None


def _has_ungrounded_number(text: str, allowed_values: list[float]) -> bool:
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0)
        leading = re.match(r"\d+(?:\.\d+)?", token)
        if leading is None:
            continue
        value = float(leading.group(0))
        if not _grounded(value, allowed_values):
            return True
    return False


def _grounded(value: float, allowed: list[float]) -> bool:
    for candidate in allowed:
        tolerance = max(1e-9, abs(candidate) * 0.01)
        if abs(value - candidate) <= tolerance:
            return True
    return False


def allowed_numeric_values(ctx: OptimizationContext) -> list[float]:
    values: list[float] = [ctx.effective_confidence, ctx.diagnosis.confidence, 0.0, 1.0]
    values.extend(ctx.observed_metrics.values())
    for item in ctx.evidence_by_id.values():
        values.append(item.observed_value)
        if item.reference_value is not None:
            values.append(item.reference_value)
    values.extend(
        float(value)
        for value in (
            ctx.hardware.cpu.physical_cores,
            ctx.hardware.cpu.logical_cores,
            ctx.hardware.memory.total_bytes,
            ctx.hardware.memory.available_bytes,
        )
    )
    if ctx.config.context_length is not None:
        values.append(float(ctx.config.context_length))
    if ctx.config.gpu_layers_offloaded is not None:
        values.append(float(ctx.config.gpu_layers_offloaded))
    if ctx.config.gpu_layers_total is not None:
        values.append(float(ctx.config.gpu_layers_total))
    # Allow the rounded forms actually displayed in the prompt.
    expanded = list(values)
    for value in values:
        expanded.append(round(value, 2))
    return expanded


def propose_actions(
    ctx: OptimizationContext,
    registry: ActionRegistry,
    provider: ReasoningProvider,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    seed: int = 1234,
    timeout_s: float = 300.0,
    raw_output_limit: int = 6000,
) -> tuple[list[dict[str, Any]], ReasoningProposalOutcome]:
    actions = offered_actions(registry)
    schema = build_proposal_schema(actions, list(ctx.evidence_ids))
    outcome = ReasoningProposalOutcome(
        used=True,
        provider=provider.name,
        model=provider.model_id,
    )
    try:
        result = provider.complete(
            system_prompt=render_system_prompt(),
            user_prompt=render_user_prompt(ctx, actions),
            json_schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - provider failures must not crash generation
        outcome = outcome.model_copy(
            update={"status": "failed", "failure": f"provider raised: {exc}"}
        )
        return [], outcome

    raw = result.text if len(result.text) <= raw_output_limit else result.text[:raw_output_limit]
    outcome = outcome.model_copy(update={"latency_s": result.latency_s, "raw_output": raw})
    if not result.success:
        outcome = outcome.model_copy(
            update={"status": "failed", "failure": result.error or "provider failure"}
        )
        return [], outcome

    try:
        accepted, rejections, proposed_ids = parse_proposals(
            result.text,
            registry=registry,
            allowed_evidence_ids=list(ctx.evidence_ids),
            allowed_values=allowed_numeric_values(ctx),
        )
    except ProposalParseError as exc:
        outcome = outcome.model_copy(
            update={"status": "failed", "failure": f"{type(exc).__name__}: {exc}"}
        )
        return [], outcome

    accepted_ids = [str(item["action_id"]) for item in accepted]
    accepted_set = set(accepted_ids)
    rejected_ids = sorted({item for item in proposed_ids if item not in accepted_set})
    outcome = outcome.model_copy(
        update={
            "status": "ok",
            "validation_ok": True,
            "proposed_action_ids": proposed_ids,
            "accepted_action_ids": accepted_ids,
            "rejected_action_ids": rejected_ids,
            "rejections": rejections,
        }
    )
    return accepted, outcome
