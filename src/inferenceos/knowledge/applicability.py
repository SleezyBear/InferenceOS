"""Explainable applicability of historical evidence to the current context.

Prior knowledge is conditional evidence, not universal truth. Each identity dimension
is compared explicitly and the verdict is rule-based; an internal weighted score is
exposed with its per-dimension contributions rather than presented as a probability.
"""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.contracts.enums import (
    ApplicabilityClass,
    DimensionStatus,
    OptimizationOutcome,
    RevalidationTrigger,
)
from inferenceos.contracts.knowledge import (
    ApplicabilityContext,
    ApplicabilityResult,
    DimensionMatch,
    OptimizationRecord,
)
from inferenceos.knowledge.policy import KnowledgePolicy

_MEMORY_DIAGNOSES = {"paging", "memory_pressure", "model_footprint"}
# A difference in any of these means historical evidence must be re-validated before
# reuse; two or more concurrent differences mean the prior result is not usable at all.
_CRITICAL = ("hardware", "backend", "model", "runtime", "diagnosis")
_WEIGHTED = ("hardware", "backend", "model", "quantization", "workload", "runtime", "diagnosis", "os")


def _dim(
    dimension: str,
    historical: str | None,
    current: str | None,
    *,
    conflict: bool = False,
) -> DimensionMatch:
    if historical is None and current is None:
        status = DimensionStatus.UNKNOWN
    elif historical is None or current is None:
        status = DimensionStatus.UNKNOWN
    elif conflict:
        status = DimensionStatus.CONFLICT
    elif historical == current:
        status = DimensionStatus.MATCH
    else:
        status = DimensionStatus.DIFFER
    return DimensionMatch(
        dimension=dimension,
        status=status,
        historical=historical,
        current=current,
    )


def evaluate_applicability(
    record: OptimizationRecord,
    current: ApplicabilityContext,
    policy: KnowledgePolicy,
    *,
    now: datetime | None = None,
) -> ApplicabilityResult:
    now = now or datetime.now(timezone.utc)
    historical = record.context.fingerprint
    current_fp = current.fingerprint

    same_model_key = historical.model_key == current_fp.model_key
    model_conflict = bool(
        same_model_key
        and historical.model_content_hash
        and current_fp.model_content_hash
        and historical.model_content_hash != current_fp.model_content_hash
    )
    model_dim = _dim(
        "model", historical.model_key, current_fp.model_key, conflict=model_conflict
    )

    dimensions: list[DimensionMatch] = [
        _dim("hardware", historical.hardware_key, current_fp.hardware_key),
        _dim("backend", historical.backend_key, current_fp.backend_key),
        model_dim,
        _dim("quantization", historical.quantization, current_fp.quantization),
        _dim("workload", historical.workload_key, current_fp.workload_key),
        _dim("runtime", historical.runtime_key, current_fp.runtime_key),
        _dim("diagnosis", historical.diagnosis, current_fp.diagnosis),
        _dim("os", historical.os_key, current_fp.os_key),
    ]
    conditions_status = _conditions_match(historical.conditions, current_fp.conditions, policy)
    dimensions.append(
        DimensionMatch(
            dimension="conditions",
            status=conditions_status,
            historical=str(historical.conditions) or None,
            current=str(current_fp.conditions) or None,
        )
    )

    by_name = {dimension.dimension: dimension for dimension in dimensions}
    matches = [d for d in dimensions if d.status is DimensionStatus.MATCH]
    differences = [d for d in dimensions if d.status is DimensionStatus.DIFFER]
    unknowns = [d for d in dimensions if d.status is DimensionStatus.UNKNOWN]
    conflicts = [d for d in dimensions if d.status is DimensionStatus.CONFLICT]

    triggers = _triggers(by_name, model_conflict)
    if _is_stale(record, now, policy):
        triggers.append(RevalidationTrigger.STALE)
    if not record.action_registered:
        triggers.append(RevalidationTrigger.ACTION_UNREGISTERED)
    if by_name["conditions"].status is DimensionStatus.DIFFER:
        triggers.append(RevalidationTrigger.SYSTEM_CONDITIONS_CHANGE)

    components = _score_components(by_name, policy)
    total_weight = sum(policy.weights.as_dict()[name] for name in _WEIGHTED)
    score = (
        sum(policy.weights.as_dict()[name] * components[name] for name in _WEIGHTED)
        / total_weight
        if total_weight
        else None
    )

    classification = _classify(
        record=record,
        current=current,
        by_name=by_name,
        differences=differences,
        unknowns=unknowns,
        policy=policy,
        model_conflict=model_conflict,
        stale=_is_stale(record, now, policy),
    )

    reasons = _reasons(classification, by_name, differences, unknowns, conflicts, triggers)
    return ApplicabilityResult(
        classification=classification,
        record_id=record.record_id,
        experiment_id=record.experiment_id,
        action_id=record.action_id,
        record_outcome=record.outcome,
        score=round(score, 4) if score is not None else None,
        components=components,
        matches=matches,
        differences=differences,
        unknown=unknowns,
        conflicts=conflicts,
        revalidation_triggers=_dedupe(triggers),
        reasons=reasons,
        rationale=_rationale(classification, by_name),
    )


def _conditions_match(
    historical: dict[str, float], current: dict[str, float], policy: KnowledgePolicy
) -> DimensionStatus:
    if not historical or not current:
        return DimensionStatus.UNKNOWN
    tolerance = policy.condition_tolerance
    for key in ("memory_pressure",):
        if key in historical and key in current:
            if abs(historical[key] - current[key]) > tolerance:
                return DimensionStatus.DIFFER
    for key in ("available_ram_bytes", "peak_rss_bytes"):
        if key in historical and key in current and historical[key]:
            relative = abs(historical[key] - current[key]) / abs(historical[key])
            if relative > tolerance:
                return DimensionStatus.DIFFER
    return DimensionStatus.MATCH


def _triggers(
    by_name: dict[str, DimensionMatch], model_conflict: bool
) -> list[RevalidationTrigger]:
    triggers: list[RevalidationTrigger] = []
    if model_conflict:
        triggers.append(RevalidationTrigger.MODEL_VERSION_CHANGE)
    mapping = {
        "hardware": RevalidationTrigger.HARDWARE_CHANGE,
        "backend": RevalidationTrigger.BACKEND_CHANGE,
        "model": RevalidationTrigger.MODEL_CHANGE,
        "quantization": RevalidationTrigger.QUANTIZATION_CHANGE,
        "workload": RevalidationTrigger.WORKLOAD_CHANGE,
        "runtime": RevalidationTrigger.RUNTIME_CHANGE,
        "diagnosis": RevalidationTrigger.DIAGNOSIS_CHANGE,
    }
    for dimension, trigger in mapping.items():
        if by_name[dimension].status in (DimensionStatus.DIFFER, DimensionStatus.CONFLICT):
            if not (dimension == "model" and model_conflict):
                triggers.append(trigger)
    return triggers


def _is_stale(record: OptimizationRecord, now: datetime, policy: KnowledgePolicy) -> bool:
    if policy.revalidation_max_age_days <= 0:
        return False
    age_days = (now - record.created_at).total_seconds() / 86400.0
    return age_days > policy.revalidation_max_age_days


def _score_components(
    by_name: dict[str, DimensionMatch], policy: KnowledgePolicy
) -> dict[str, float]:
    components: dict[str, float] = {}
    for name in _WEIGHTED:
        status = by_name[name].status
        if status is DimensionStatus.MATCH:
            components[name] = 1.0
        elif status is DimensionStatus.UNKNOWN:
            components[name] = 0.5
        else:
            components[name] = 0.0
    return components


def _classify(
    *,
    record: OptimizationRecord,
    current: ApplicabilityContext,
    by_name: dict[str, DimensionMatch],
    differences: list[DimensionMatch],
    unknowns: list[DimensionMatch],
    policy: KnowledgePolicy,
    model_conflict: bool,
    stale: bool,
) -> ApplicabilityClass:
    if not record.action_registered:
        return ApplicabilityClass.NOT_APPLICABLE
    if current.model is None or current.hardware is None:
        return ApplicabilityClass.INSUFFICIENT_CONTEXT
    critical_diffs = [
        name
        for name in _CRITICAL
        if by_name[name].status is DimensionStatus.DIFFER or name == "model" and model_conflict
    ]
    important_diffs = [
        d
        for d in differences
        if d.dimension not in _CRITICAL and d.dimension != "conditions"
    ]
    if len(critical_diffs) >= policy.not_applicable_critical_threshold:
        return ApplicabilityClass.NOT_APPLICABLE
    if critical_diffs:
        return ApplicabilityClass.REQUIRES_REVALIDATION
    condition_downgrade = by_name["conditions"].status is DimensionStatus.DIFFER
    if important_diffs or condition_downgrade or stale:
        return ApplicabilityClass.CONDITIONALLY_APPLICABLE
    return ApplicabilityClass.DIRECTLY_APPLICABLE


def _reasons(
    classification: ApplicabilityClass,
    by_name: dict[str, DimensionMatch],
    differences: list[DimensionMatch],
    unknowns: list[DimensionMatch],
    conflicts: list[DimensionMatch],
    triggers: list[RevalidationTrigger],
) -> list[str]:
    reasons = [f"classification={classification.value}"]
    for difference in differences:
        reasons.append(f"differs: {difference.dimension}")
    for conflict in conflicts:
        reasons.append(f"conflict: {conflict.dimension}")
    for unknown in unknowns:
        reasons.append(f"unknown: {unknown.dimension}")
    for trigger in _dedupe(triggers):
        reasons.append(f"revalidation: {trigger.value}")
    return reasons


def _rationale(
    classification: ApplicabilityClass, by_name: dict[str, DimensionMatch]
) -> str:
    if classification is ApplicabilityClass.DIRECTLY_APPLICABLE:
        return "identity dimensions match; historical evidence is directly relevant"
    if classification is ApplicabilityClass.CONDITIONALLY_APPLICABLE:
        return "core identity matches but some dimensions differ; evidence is conditional"
    if classification is ApplicabilityClass.REQUIRES_REVALIDATION:
        return "a critical identity dimension differs; re-validate before reuse"
    if classification is ApplicabilityClass.NOT_APPLICABLE:
        return "materially different or unregistered; prior result is not directly usable"
    return "current context is missing required identity dimensions"


def _dedupe(triggers: list[RevalidationTrigger]) -> list[RevalidationTrigger]:
    seen: set[RevalidationTrigger] = set()
    ordered: list[RevalidationTrigger] = []
    for trigger in triggers:
        if trigger in seen:
            continue
        seen.add(trigger)
        ordered.append(trigger)
    return ordered


