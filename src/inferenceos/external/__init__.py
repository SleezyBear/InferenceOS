"""Phase 11R external process recommendations (staged, minimal, recommendation-only).

InferenceOS may observe and recommend actions concerning external processes and system
resources, but it must not silently control them. The flow is strictly:

    observe -> attribute -> recommend -> present -> user decision

Even an accepted recommendation only records user intent; there is no executor and no
autonomous external-process mutation. Observation is read-only. The layer reuses the
Phase 6 action registry and the Phase 10 approval abstraction; no parallel
arbitrary-action system exists.
"""

from __future__ import annotations

from inferenceos.external.actions import evaluate_external_action_safety
from inferenceos.external.attribution import assess_external_attribution
from inferenceos.external.controlled import (
    ScriptedProcessObserver,
    launch_cpu_spinner,
    launch_memory_holder,
    launch_sleeper,
)
from inferenceos.external.engine import ExternalRecommendationEngine
from inferenceos.external.identity import (
    build_identity,
    identity_from_process,
    resolve_identity,
    verify_identity,
)
from inferenceos.external.integration import RuntimeExternalAdvisor
from inferenceos.external.knowledge import ExternalKnowledgeError, ExternalKnowledgeStore
from inferenceos.external.observer import ProcessObserver, WindowsProcessObserver
from inferenceos.external.policy import build_external_budget, build_external_policy
from inferenceos.external.recommendation import (
    bottleneck_to_resource,
    revalidate_recommendation,
)
from inferenceos.external.recommender import build_recommendations
from inferenceos.external.report import render_external_run, render_recommendation

__all__ = [
    "ExternalKnowledgeError",
    "ExternalKnowledgeStore",
    "ExternalRecommendationEngine",
    "ProcessObserver",
    "RuntimeExternalAdvisor",
    "ScriptedProcessObserver",
    "WindowsProcessObserver",
    "assess_external_attribution",
    "bottleneck_to_resource",
    "build_external_budget",
    "build_external_policy",
    "build_identity",
    "build_recommendations",
    "evaluate_external_action_safety",
    "identity_from_process",
    "launch_cpu_spinner",
    "launch_memory_holder",
    "launch_sleeper",
    "render_external_run",
    "render_recommendation",
    "resolve_identity",
    "revalidate_recommendation",
    "verify_identity",
]
