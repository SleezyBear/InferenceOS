"""Build structured, auditable, diagnosis-aware external recommendations.

Recommendations are never execution. They are generated only when an observed external
process is a plausible contributor to the **diagnosed** bottleneck resource (not merely
a large consumer), and they always terminate at a user decision. Phase 11R proposes only
recommendation-only actions from the existing registry; no action here is executable.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from inferenceos.contracts.enums import (
    ExternalActionSafetyClass,
    ExternalRecommendationClass,
    ExternalRecommendationStatus,
    ExternalResource,
    ExternalScope,
)
from inferenceos.contracts.external import (
    AttributionAssessment,
    ExternalPolicy,
    ExternalProcessIdentity,
    ExternalProvenance,
    ExternalRecommendation,
)
from inferenceos.contracts.optimization import CandidateAction
from inferenceos.optimization.registry import ActionRegistry

_CPU_ACTIONS = ("external.reduce_priority", "external.request_close", "external.suspend_process")
_MEMORY_ACTIONS = ("external.request_close", "external.suspend_process")
_STORAGE_ACTIONS = ("external.request_close",)

_DISRUPTION_RISK = {
    "external.request_close": "Closing the process may discard unsaved work.",
    "external.reduce_priority": (
        "A priority change is temporary but may slow the external process while applied."
    ),
    "external.suspend_process": "Suspending the process pauses its work until the user resumes it.",
}


def _candidate_action(
    resource: ExternalResource, registry: ActionRegistry
) -> CandidateAction | None:
    order: tuple[str, ...]
    if resource is ExternalResource.CPU:
        order = _CPU_ACTIONS
    elif resource is ExternalResource.MEMORY:
        order = _MEMORY_ACTIONS
    elif resource is ExternalResource.STORAGE_IO:
        order = _STORAGE_ACTIONS
    else:
        order = _MEMORY_ACTIONS
    for action_id in order:
        action = registry.get_optional(action_id)
        if action is not None and action.phase11_allowed:
            return action
    return None


def _parameters_for(action: CandidateAction, process_name: str) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for spec in action.parameters:
        if spec.name == "priority_class":
            parameters["priority_class"] = "below_normal"
        elif spec.name == "process_name":
            choices = list(spec.choices or [])
            if process_name in choices:
                parameters["process_name"] = process_name
    return parameters


def _resource_matches(actual: ExternalResource, diagnosed: ExternalResource) -> bool:
    if diagnosed is ExternalResource.SYSTEM:
        return True
    return actual is diagnosed


def _observed_metrics(assessment: AttributionAssessment) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for impact in assessment.impacts:
        if impact.process_contribution is not None:
            metrics[f"process_{impact.resource.value}_{impact.unit}"] = float(
                impact.process_contribution
            )
        if impact.target_contribution is not None:
            metrics[f"target_{impact.resource.value}_{impact.unit}"] = float(
                impact.target_contribution
            )
        if impact.relative_significance is not None:
            metrics[f"relative_{impact.resource.value}"] = float(impact.relative_significance)
    return metrics


def _grounded_benefit(
    resource: ExternalResource, metrics: dict[str, float]
) -> tuple[str, bool]:
    if resource is ExternalResource.MEMORY:
        rss = metrics.get("process_memory_bytes")
        if rss is not None and rss > 0:
            mib = rss / (1024 * 1024)
            return (
                f"up to approximately {mib:.0f} MiB of the observed resident memory could "
                "be freed if the process releases it (observed RSS; actual release is not "
                "guaranteed)",
                True,
            )
    # No grounded numerical estimate is available for this resource; do not invent one.
    return (
        "expected benefit is uncertain: it cannot be grounded from the observed evidence",
        False,
    )


def build_recommendations(
    assessments: list[AttributionAssessment],
    *,
    registry: ActionRegistry,
    policy: ExternalPolicy,
    context_identity: str,
    diagnosed_resource: ExternalResource | None = None,
    diagnosed_bottleneck: str | None = None,
    target_identity: ExternalProcessIdentity | None = None,
    target_degraded: bool = False,
    run_id: str = "",
    session_id: str = "",
    diagnosis_id: str | None = None,
    observation_report_id: str | None = None,
    observed_at: datetime | None = None,
    now: datetime | None = None,
) -> list[ExternalRecommendation]:
    """Build recommendations from assessments.

    A recommendation is produced only when the diagnosed bottleneck maps onto the
    external process's resource domain. With no meaningful/diagnosable bottleneck this
    returns no recommendations — raw resource consumption alone is never sufficient.
    """

    if diagnosed_resource is None:
        return []

    now = now or datetime.now(timezone.utc)
    recommendations: list[ExternalRecommendation] = []
    for assessment in assessments:
        if not _resource_matches(assessment.resource, diagnosed_resource):
            continue
        dominant = any(impact.dominant for impact in assessment.impacts)
        if not dominant and not assessment.sufficient_for_recommendation:
            continue
        identity = assessment.identity
        resource_label = assessment.resource.value.replace("_", " ")
        evidence_summary = [item.description for item in assessment.evidence]
        sufficient = assessment.sufficient_for_recommendation
        action = _candidate_action(assessment.resource, registry) if sufficient else None
        classification = (
            ExternalRecommendationClass.USER_ACTION
            if sufficient
            else ExternalRecommendationClass.INFORMATIONAL
        )
        supported = bool(action is not None and action.implemented and action.phase11_allowed)
        safety_class = (
            action.external_safety_class
            if action is not None
            else ExternalActionSafetyClass.RECOMMENDATION_ONLY
        )
        parameters = _parameters_for(action, identity.name) if action is not None else {}
        metrics = _observed_metrics(assessment)
        benefit_text, benefit_available = _grounded_benefit(assessment.resource, metrics)
        disruption = (
            _DISRUPTION_RISK.get(action.action_id, "The user decides whether any action is taken.")
            if action is not None
            else "No action is proposed; this is an observation only."
        )
        if sufficient and assessment.causal:
            uncertainty = (
                "causal attribution supported by convergent pressure/degradation evidence; "
                "still not a controlled measurement of the benefit"
            )
        elif sufficient:
            uncertainty = (
                "correlation only: the process is dominant in the diagnosed resource "
                "domain, but causality is not established"
            )
        else:
            uncertainty = "correlation only; no causal claim is made"

        if sufficient:
            neutral_description = (
                f"{identity.name} (pid {identity.pid}, identity {identity.identity_key}) is a "
                f"dominant {resource_label} consumer observed while the diagnosed bottleneck "
                f"was {diagnosed_bottleneck or 'unknown'}."
            )
            why = (
                f"The diagnosed inference bottleneck is {diagnosed_bottleneck or 'unknown'}; "
                "this external process is a plausible dominant contributor to that same "
                "resource domain. Reducing its footprint could plausibly help."
            )
        else:
            neutral_description = (
                f"{identity.name} (pid {identity.pid}, identity {identity.identity_key}) is a "
                f"high {resource_label} consumer co-occurring with the diagnosed bottleneck."
            )
            why = "This is an observation of correlation only; no causal claim is made."

        recommendations.append(
            ExternalRecommendation(
                recommendation_id=f"extrec-{uuid.uuid4().hex[:10]}",
                created_at=now,
                classification=classification,
                scope=ExternalScope.EXTERNAL_PROCESS,
                resource=assessment.resource,
                identity=identity,
                neutral_description=neutral_description,
                why_it_matters=why,
                evidence_summary=evidence_summary,
                attribution_id=assessment.assessment_id,
                proposed_action_id=action.action_id if action is not None else None,
                proposed_parameters=parameters,
                expected_benefit=benefit_text,
                expected_benefit_available=benefit_available,
                expected_cost_s=0.0,
                expected_disruption_s=0.0,
                requires_approval=True,
                requires_user_approval=True,
                supported=supported,
                reversible=False,
                verification_possible=False,
                safety_class=safety_class,
                context_identity=context_identity,
                expires_at=now + timedelta(seconds=policy.approval_timeout_s),
                limitations=[
                    "external processes belong to the user/system; InferenceOS never "
                    "silently controls them",
                    "correlation is not causation unless a controlled comparison exists",
                    "an accepted recommendation records user intent only; InferenceOS does "
                    "not perform the action",
                ],
                run_id=run_id,
                session_id=session_id,
                target_identity=target_identity,
                bottleneck=diagnosed_bottleneck,
                observed_metrics=metrics,
                observed_at=observed_at or now,
                rationale=list(assessment.rationale) + evidence_summary,
                confidence=assessment.confidence,
                uncertainty=uncertainty,
                disruption_risk=disruption,
                provenance=ExternalProvenance(
                    optimization_session_id=session_id,
                    run_id=run_id,
                    diagnosis_id=diagnosis_id,
                    observation_report_id=observation_report_id,
                    process_observation_id=None,
                    attribution_id=assessment.assessment_id,
                    target_pid=target_identity.pid if target_identity is not None else None,
                    target_identity_key=(
                        target_identity.identity_key if target_identity is not None else None
                    ),
                    context_identity=context_identity,
                    generated_at=now,
                ),
                status=ExternalRecommendationStatus.GENERATED,
            )
        )
    return recommendations
