"""Phase 11 external process recommendations (staged, minimal).

InferenceOS may observe and recommend actions concerning external processes and system
resources, but it must not silently control them. The default is observe -> recommend
-> ask user, with explicit contextual approval required before any supported action.

The layer is strictly read-only for observation and reuses the Phase 6 action registry
and the Phase 10 approval abstraction; no parallel arbitrary-action system exists.
"""

from __future__ import annotations

from inferenceos.external.actions import (
    ExternalActionExecutor,
    capture_external_state,
    evaluate_external_action_safety,
)
from inferenceos.external.attribution import assess_external_attribution
from inferenceos.external.checkpoint import capture_external_checkpoint
from inferenceos.external.controlled import (
    ScriptedProcessObserver,
    StubExternalActionExecutor,
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
from inferenceos.external.recommender import build_recommendations
from inferenceos.external.report import render_external_run
from inferenceos.external.verification import verify_external_action

__all__ = [
    "ExternalActionExecutor",
    "ExternalKnowledgeError",
    "ExternalKnowledgeStore",
    "ExternalRecommendationEngine",
    "ProcessObserver",
    "RuntimeExternalAdvisor",
    "ScriptedProcessObserver",
    "StubExternalActionExecutor",
    "WindowsProcessObserver",
    "assess_external_attribution",
    "build_external_budget",
    "build_external_policy",
    "build_identity",
    "build_recommendations",
    "capture_external_checkpoint",
    "capture_external_state",
    "evaluate_external_action_safety",
    "identity_from_process",
    "launch_cpu_spinner",
    "launch_memory_holder",
    "launch_sleeper",
    "render_external_run",
    "resolve_identity",
    "verify_external_action",
    "verify_identity",
]
