"""Phase 10 runtime adaptive monitoring (conservative mode).

Continuously observes an active, InferenceOS-controlled workload, temporally qualifies
deviations, reuses the Phase 4 diagnosis engine, evaluates intervention economics,
*recommends* deep analysis, requires explicit approval for disruptive active-runtime
changes, checkpoints before pausing, reuses Phase 5/6/7/8 machinery for the controlled
intervention, and returns the workload safely to a verified state.
"""

from __future__ import annotations

from inferenceos.runtime.approval import (
    ApprovalProvider,
    AutoApproveApprovalProvider,
    AutoRejectApprovalProvider,
    CliApprovalProvider,
    ScriptedApprovalProvider,
    TimeoutApprovalProvider,
)
from inferenceos.runtime.checkpoint import (
    capture_runtime_checkpoint,
    restore_runtime_checkpoint,
    verify_runtime_checkpoint,
)
from inferenceos.runtime.economics import classify_action, evaluate_intervention_economics
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.notification import (
    CollectingNotificationSink,
    ConsoleNotificationSink,
    NotificationSink,
)
from inferenceos.runtime.policy import build_runtime_budget, build_runtime_policy
from inferenceos.runtime.report import render_monitor_run, render_state
from inferenceos.runtime.telemetry import RuntimeTelemetrySampler, SystemTelemetrySampler
from inferenceos.runtime.window import ObservationBuffer
from inferenceos.runtime.workload import (
    ActiveWorkload,
    ControlledLlamaWorkload,
    RuntimeWorkloadExecutor,
    config_key,
)

__all__ = [
    "ActiveWorkload",
    "ApprovalProvider",
    "AutoApproveApprovalProvider",
    "AutoRejectApprovalProvider",
    "CliApprovalProvider",
    "CollectingNotificationSink",
    "ConsoleNotificationSink",
    "ControlledLlamaWorkload",
    "NotificationSink",
    "ObservationBuffer",
    "RuntimeMonitor",
    "RuntimeTelemetrySampler",
    "RuntimeWorkloadExecutor",
    "ScriptedApprovalProvider",
    "SystemTelemetrySampler",
    "TimeoutApprovalProvider",
    "build_runtime_budget",
    "build_runtime_policy",
    "capture_runtime_checkpoint",
    "classify_action",
    "config_key",
    "evaluate_intervention_economics",
    "render_monitor_run",
    "render_state",
    "restore_runtime_checkpoint",
    "verify_runtime_checkpoint",
]
