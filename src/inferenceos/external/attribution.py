"""Deterministic external resource attribution.

Estimates how external processes relate to the target workload using proportional
comparison (never absolute-size ranking) and explicit evidence categories. Correlation
is never represented as causation: ``causal`` is only true with a controlled
comparison or convergent resource-pressure + target-degradation evidence.
"""

from __future__ import annotations

import uuid

from inferenceos.contracts.enums import (
    AttributionEvidenceCategory,
    EvidenceStrength,
    ExternalProcessAccess,
    ExternalResource,
)
from inferenceos.contracts.external import (
    AttributionAssessment,
    AttributionEvidence,
    ExternalControlledComparison,
    ExternalKnowledgeRecord,
    ExternalObservationReport,
    ExternalPolicy,
    ExternalProcessObservation,
    ExternalResourceImpact,
)

def _relative(part: float | None, other: float | None) -> float | None:
    if part is None or other is None:
        return None
    total = part + other
    if total <= 0:
        return None
    return part / total


def assess_external_attribution(
    report: ExternalObservationReport,
    policy: ExternalPolicy,
    *,
    target_degraded: bool = False,
    comparisons: list[ExternalControlledComparison] | None = None,
    knowledge: list[ExternalKnowledgeRecord] | None = None,
) -> list[AttributionAssessment]:
    comparisons = comparisons or []
    knowledge = knowledge or []
    comparison_by_key = {item.identity_key: item for item in comparisons}
    knowledge_keys = {item.identity_key for item in knowledge}

    target = _target_observation(report)
    target_rss = float(target.rss_bytes) if target is not None and target.rss_bytes else None
    target_cpu = target.cpu_percent if target is not None else None
    total_ram = float(report.total_ram_bytes) if report.total_ram_bytes else None
    memory_pressure = report.memory_pressure
    pressure = memory_pressure is not None and memory_pressure >= policy.memory_pressure_threshold

    assessments: list[AttributionAssessment] = []
    for observation in report.processes:
        identity = observation.identity
        if (
            identity.is_target
            or identity.pid == report.own_pid
            or (report.target_pid is not None and identity.pid == report.target_pid)
        ):
            continue
        if observation.access is ExternalProcessAccess.DENIED:
            continue
        assessment = _assess_process(
            observation,
            policy=policy,
            target_rss=target_rss,
            target_cpu=target_cpu,
            total_ram=total_ram,
            memory_pressure=memory_pressure,
            pressure=pressure,
            target_degraded=target_degraded,
            comparison=comparison_by_key.get(identity.identity_key),
            historical=identity.identity_key in knowledge_keys,
        )
        if assessment is not None:
            assessments.append(assessment)

    assessments.sort(key=lambda item: (-item.confidence, item.identity.pid))
    return assessments


def _target_observation(
    report: ExternalObservationReport,
) -> ExternalProcessObservation | None:
    for observation in report.processes:
        if observation.identity.is_target:
            return observation
    if report.target_pid is not None:
        for observation in report.processes:
            if observation.identity.pid == report.target_pid:
                return observation
    return None


def _assess_process(
    observation: ExternalProcessObservation,
    *,
    policy: ExternalPolicy,
    target_rss: float | None,
    target_cpu: float | None,
    total_ram: float | None,
    memory_pressure: float | None,
    pressure: bool,
    target_degraded: bool,
    comparison: ExternalControlledComparison | None,
    historical: bool,
) -> AttributionAssessment | None:
    identity = observation.identity
    impacts: list[ExternalResourceImpact] = []
    evidence: list[AttributionEvidence] = []
    categories: list[AttributionEvidenceCategory] = []

    proc_rss = float(observation.rss_bytes) if observation.rss_bytes else None
    memory_relative = _relative(proc_rss, target_rss)
    memory_share = (
        (proc_rss / total_ram) if proc_rss is not None and total_ram else None
    )
    memory_dominant = bool(
        proc_rss is not None
        and proc_rss >= policy.min_process_rss_bytes
        and memory_relative is not None
        and memory_relative >= policy.memory_dominance_threshold
    )
    if proc_rss is not None:
        impacts.append(
            ExternalResourceImpact(
                resource=ExternalResource.MEMORY,
                unit="bytes",
                process_contribution=proc_rss,
                target_contribution=target_rss,
                system_total=total_ram,
                share_of_system=memory_share,
                relative_significance=memory_relative,
                dominant=memory_dominant,
                within_threshold=not memory_dominant,
                explanation=(
                    f"process rss {proc_rss:.0f} vs target rss "
                    f"{target_rss if target_rss is not None else 'unknown'}"
                ),
            )
        )
        evidence.append(
            AttributionEvidence(
                category=AttributionEvidenceCategory.DIRECT_OBSERVATION,
                metric="rss_bytes",
                value=proc_rss,
                reference=target_rss,
                description="external process resident memory observed read-only",
            )
        )
        categories.append(AttributionEvidenceCategory.DIRECT_OBSERVATION)

    proc_cpu = observation.cpu_percent
    # CPU contention is only meaningful when the target itself uses CPU; a target at
    # ~0% CPU is not contended, so an external CPU consumer is not flagged on that basis.
    cpu_relative = (
        _relative(proc_cpu, target_cpu) if target_cpu is not None and target_cpu > 0.0 else None
    )
    cpu_dominant = bool(
        proc_cpu is not None
        and proc_cpu >= policy.min_process_cpu_percent
        and cpu_relative is not None
        and cpu_relative >= policy.cpu_dominance_threshold
    )
    if proc_cpu is not None:
        impacts.append(
            ExternalResourceImpact(
                resource=ExternalResource.CPU,
                unit="percent",
                process_contribution=proc_cpu,
                target_contribution=target_cpu,
                system_total=None,
                share_of_system=None,
                relative_significance=cpu_relative,
                dominant=cpu_dominant,
                within_threshold=not cpu_dominant,
                explanation=(
                    f"process cpu {proc_cpu:.1f}% vs target cpu "
                    f"{target_cpu if target_cpu is not None else 'unknown'}%"
                ),
            )
        )
        evidence.append(
            AttributionEvidence(
                category=AttributionEvidenceCategory.DIRECT_OBSERVATION,
                metric="cpu_percent",
                value=proc_cpu,
                reference=target_cpu,
                description="external process CPU utilization observed read-only",
            )
        )

    if proc_rss is None and proc_cpu is None:
        return None

    dominant = memory_dominant or cpu_dominant
    if pressure and memory_dominant:
        evidence.append(
            AttributionEvidence(
                category=AttributionEvidenceCategory.RESOURCE_PRESSURE,
                metric="memory_pressure",
                value=proc_rss,
                reference=policy.memory_pressure_threshold,
                description="system memory pressure is elevated and this process is a dominant contributor",
            )
        )
        categories.append(AttributionEvidenceCategory.RESOURCE_PRESSURE)
    if cpu_dominant and observation.cpu_percent is not None:
        evidence.append(
            AttributionEvidence(
                category=AttributionEvidenceCategory.RESOURCE_PRESSURE,
                metric="process_cpu_percent",
                value=observation.cpu_percent,
                reference=policy.min_process_cpu_percent,
                description="external process is a dominant CPU consumer",
            )
        )
        if AttributionEvidenceCategory.RESOURCE_PRESSURE not in categories:
            categories.append(AttributionEvidenceCategory.RESOURCE_PRESSURE)

    if target_degraded and dominant:
        evidence.append(
            AttributionEvidence(
                category=AttributionEvidenceCategory.TEMPORAL_ALIGNMENT,
                metric="co_occurrence",
                value=None,
                reference=None,
                description="external resource use co-occurs with target degradation (not proof of cause)",
            )
        )
        categories.append(AttributionEvidenceCategory.TEMPORAL_ALIGNMENT)
    if target_degraded:
        evidence.append(
            AttributionEvidence(
                category=AttributionEvidenceCategory.TARGET_DEGRADATION,
                metric="target_degraded",
                value=None,
                reference=None,
                description="target workload shows a measured deviation from its baseline",
            )
        )
        categories.append(AttributionEvidenceCategory.TARGET_DEGRADATION)

    controlled = bool(comparison is not None and comparison.improved_when_inactive)
    if controlled and comparison is not None:
        evidence.append(
            AttributionEvidence(
                category=AttributionEvidenceCategory.CONTROLLED_COMPARISON,
                metric=comparison.metric,
                value=comparison.inactive_mean,
                reference=comparison.active_mean,
                description=(
                    "target performance improved when the external process became "
                    "naturally inactive (observational comparison)"
                ),
            )
        )
        categories.append(AttributionEvidenceCategory.CONTROLLED_COMPARISON)
    if historical:
        evidence.append(
            AttributionEvidence(
                category=AttributionEvidenceCategory.HISTORICAL_KNOWLEDGE,
                metric="historical_record",
                value=None,
                reference=None,
                description="a contextual historical record exists for this process identity",
            )
        )
        categories.append(AttributionEvidenceCategory.HISTORICAL_KNOWLEDGE)

    causal = bool(controlled or (pressure and target_degraded and dominant))
    correlation_only = not causal
    if correlation_only:
        evidence.append(
            AttributionEvidence(
                category=AttributionEvidenceCategory.CORRELATION_ONLY,
                metric="attribution",
                value=None,
                reference=None,
                description="resource use is observed but causal evidence is not established",
            )
        )
        categories.append(AttributionEvidenceCategory.CORRELATION_ONLY)

    requires_degradation = policy.require_target_degradation_for_recommendation
    memory_sufficient = memory_dominant and pressure and (target_degraded or not requires_degradation)
    cpu_sufficient = cpu_dominant and (target_degraded or not requires_degradation)
    sufficient = bool((memory_sufficient or cpu_sufficient or (controlled and dominant)))
    if proc_rss is None and proc_cpu is None:
        sufficient = False

    confidence = _confidence(categories, controlled, dominant)
    strength = _strength(confidence)
    resource = (
        ExternalResource.MEMORY
        if memory_dominant
        else ExternalResource.CPU
        if cpu_dominant
        else (ExternalResource.MEMORY if proc_rss is not None else ExternalResource.CPU)
    )

    rationale = [
        f"relative memory significance={memory_relative}",
        f"relative cpu significance={cpu_relative}",
        f"memory dominant={memory_dominant} cpu dominant={cpu_dominant}",
        f"system memory pressure={memory_pressure} (threshold {policy.memory_pressure_threshold})",
        f"target degraded={target_degraded} controlled comparison={controlled}",
    ]
    return AttributionAssessment(
        assessment_id=f"extattr-{uuid.uuid4().hex[:10]}",
        identity=identity,
        resource=resource,
        evidence=evidence,
        categories=list(dict.fromkeys(categories)),
        impacts=impacts,
        causal=causal,
        correlation_only=correlation_only,
        target_degraded=target_degraded,
        controlled_comparison=controlled,
        historical=historical,
        confidence=confidence,
        strength=strength,
        sufficient_for_recommendation=sufficient,
        rationale=rationale,
        insufficient_reason=(
            None
            if sufficient
            else "no convergent pressure/degradation/comparison evidence for this process"
        ),
    )


def _confidence(
    categories: list[AttributionEvidenceCategory], controlled: bool, dominant: bool
) -> float:
    if not dominant:
        return 0.0
    weights = {
        AttributionEvidenceCategory.DIRECT_OBSERVATION: 0.2,
        AttributionEvidenceCategory.RESOURCE_PRESSURE: 0.2,
        AttributionEvidenceCategory.TARGET_DEGRADATION: 0.25,
        AttributionEvidenceCategory.CONTROLLED_COMPARISON: 0.3,
        AttributionEvidenceCategory.TEMPORAL_ALIGNMENT: 0.1,
        AttributionEvidenceCategory.HISTORICAL_KNOWLEDGE: 0.15,
        AttributionEvidenceCategory.CORRELATION_ONLY: 0.1,
    }
    present = set(categories)
    score = sum(
        weight for category, weight in weights.items() if category in present
    )
    if controlled:
        score += 0.1
    return max(0.0, min(1.0, score))


def _strength(confidence: float) -> EvidenceStrength:
    if confidence >= 0.75:
        return EvidenceStrength.HIGH
    if confidence >= 0.45:
        return EvidenceStrength.MEDIUM
    if confidence > 0.0:
        return EvidenceStrength.LOW
    return EvidenceStrength.NONE
