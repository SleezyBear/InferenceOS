"""Build structured, auditable external recommendations from attribution assessments.

Recommendations are never execution. The default is informational or user-action; an
approval-gated registered action is only proposed when the registry explicitly marks it
implemented, Phase 11 allowed, approval-required, and verifiable.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from inferenceos.contracts.enums import (
    ExternalActionSafetyClass,
    ExternalRecommendationClass,
    ExternalResource,
    ExternalScope,
)
from inferenceos.contracts.external import (
    AttributionAssessment,
    ExternalPolicy,
    ExternalRecommendation,
)
from inferenceos.optimization.registry import ActionRegistry
from inferenceos.contracts.optimization import CandidateAction

_CPU_ACTIONS = ("external.reduce_priority", "external.request_close", "external.suspend_process")
_MEMORY_ACTIONS = ("external.request_close", "external.suspend_process")
_STORAGE_ACTIONS = ("external.request_close",)


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


def _classification(action: CandidateAction | None) -> ExternalRecommendationClass:
    if action is None:
        return ExternalRecommendationClass.INFORMATIONAL
    if action.implemented and action.phase11_allowed:
        return ExternalRecommendationClass.APPROVAL_GATED_ACTION
    if action.phase11_allowed:
        return ExternalRecommendationClass.USER_ACTION
    return ExternalRecommendationClass.UNSUPPORTED_ACTION


def build_recommendations(
    assessments: list[AttributionAssessment],
    *,
    registry: ActionRegistry,
    policy: ExternalPolicy,
    context_identity: str,
    now: datetime | None = None,
) -> list[ExternalRecommendation]:
    now = now or datetime.now(timezone.utc)
    recommendations: list[ExternalRecommendation] = []
    for assessment in assessments:
        dominant = any(impact.dominant for impact in assessment.impacts)
        if not dominant and not assessment.sufficient_for_recommendation:
            continue
        identity = assessment.identity
        resource_label = assessment.resource.value.replace("_", " ")
        evidence_summary = [item.description for item in assessment.evidence]
        sufficient = assessment.sufficient_for_recommendation
        action = _candidate_action(assessment.resource, registry) if sufficient else None
        classification = _classification(action) if sufficient else ExternalRecommendationClass.INFORMATIONAL
        supported = bool(action is not None and action.implemented and action.phase11_allowed)
        safety_class = (
            action.external_safety_class
            if action is not None
            else ExternalActionSafetyClass.RECOMMENDATION_ONLY
        )
        parameters = _parameters_for(action, identity.name) if action is not None else {}
        reversible = bool(action is not None and action.reversibility.value == "reversible")
        verification_possible = bool(action is not None and action.verification_supported)
        requires_approval = bool(
            sufficient
            and (
                policy.require_approval_for_actions
                or action is None
                or action.requires_human_approval
            )
        )
        action_cost = action.execution_cost_estimate if action is not None else 0.0
        disruption = (
            policy.cost_model.action_cost_s + policy.cost_model.disruption_cost_s
            if supported
            else 0.0
        )
        if sufficient:
            expected_benefit = (
                f"reduce target {resource_label} contention from {identity.name}"
                if supported
                else f"potentially reduce target {resource_label} contention if the user acts"
            )
            neutral_description = (
                f"{identity.name} (pid {identity.pid}) is a high {resource_label} consumer "
                "observed while the target workload shows a measured deviation."
            )
            why = (
                "A plausible dominant external contributor under current contention may be "
                "affecting target inference; only evidence from this context supports it."
            )
        else:
            expected_benefit = "no action proposed; correlation only"
            neutral_description = (
                f"{identity.name} (pid {identity.pid}) is a high {resource_label} consumer "
                "co-occurring with target activity; causality is not established."
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
                expected_benefit=expected_benefit,
                expected_cost_s=action_cost + policy.cost_model.verification_cost_s,
                expected_disruption_s=disruption,
                requires_approval=requires_approval,
                supported=supported,
                reversible=reversible,
                verification_possible=verification_possible,
                safety_class=safety_class,
                context_identity=context_identity,
                expires_at=now + timedelta(seconds=policy.approval_timeout_s),
                limitations=[
                    "external processes belong to the user/system; InferenceOS never "
                    "silently controls them",
                    "correlation is not causation unless a controlled comparison exists",
                ],
            )
        )
    return recommendations
