"""Phase 11 identity / safety / executor / config / privacy / reporting tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferenceos.config import Settings
from inferenceos.contracts.enums import ExternalActionSafetyClass, OptimizationScope
from inferenceos.contracts.external import ExternalBudget
from inferenceos.external.actions import (
    ExternalActionExecutor,
    capture_external_state,
    evaluate_external_action_safety,
)
from inferenceos.external.checkpoint import capture_external_checkpoint
from inferenceos.external.identity import (
    build_identity,
    resolve_identity,
    verify_identity,
)
from inferenceos.external.observer import WindowsProcessObserver
from inferenceos.external.policy import build_external_budget, build_external_policy
from inferenceos.external.report import render_external_run
from inferenceos.optimization.actions import build_default_registry
from tests.external_helpers import ScriptedProcessObserver, launch_sleeper, make_report, make_process_observation
from inferenceos.contracts.enums import ExternalProcessAccess
from tests.external_helpers import StubExternalActionExecutor
from inferenceos.external.engine import ExternalRecommendationEngine
from inferenceos.runtime.approval import AutoRejectApprovalProvider


def test_identity_rejects_recycled_pid() -> None:
    with launch_sleeper() as proc:
        identity = resolve_identity(proc.pid)
        assert identity is not None
        assert verify_identity(identity) is True
        # A tampered create time simulates a recycled PID.
        assert identity.create_time_epoch is not None
        stale = identity.model_copy(
            update={"create_time_epoch": identity.create_time_epoch - 999.0}
        )
        assert verify_identity(stale) is False


def test_identity_marks_unavailable_fields_potentially_stale() -> None:
    identity = build_identity(
        pid=999999,
        name="ghost.exe",
        executable_path=None,
        owner=None,
        create_time_epoch=None,
        access=ExternalProcessAccess.DENIED,
    )
    assert identity.potentially_stale is True
    assert "stale" in " ".join(identity.notes).lower()


def test_registry_external_invariants() -> None:
    registry = build_default_registry()
    reduce_priority = registry.get("external.reduce_priority")
    assert reduce_priority.implemented is True
    assert reduce_priority.phase11_allowed is True
    assert reduce_priority.verification_supported is True
    assert reduce_priority.external_safety_class is ExternalActionSafetyClass.SAFE_REVERSIBLE
    assert reduce_priority.scope is OptimizationScope.EXTERNAL_PROCESS
    assert reduce_priority.requires_human_approval is True
    # System-wide/elevated actions must remain recommendation-only.
    pagefile = registry.get("system.increase_pagefile")
    assert pagefile.implemented is False
    assert pagefile.phase11_allowed is False


def test_safety_blocks_elevation_and_unsupported_actions() -> None:
    registry = build_default_registry()
    policy = build_external_policy(Settings())
    pagefile = evaluate_external_action_safety(
        "system.increase_pagefile", registry=registry, policy=policy, identity=None
    )
    assert pagefile.allowed is False
    assert pagefile.elevation_required is True
    assert any("elevation" in reason for reason in pagefile.reasons)

    user_action = evaluate_external_action_safety(
        "external.request_close", registry=registry, policy=policy, identity=None
    )
    assert user_action.allowed is False
    assert any("recommendation-only" in reason for reason in user_action.reasons)

    internal = evaluate_external_action_safety(
        "runtime.right_size_context", registry=registry, policy=policy, identity=None
    )
    assert internal.allowed is False

    unknown = evaluate_external_action_safety(
        "external.invented_action", registry=registry, policy=policy, identity=None
    )
    assert unknown.allowed is False
    assert any("not registered" in reason for reason in unknown.reasons)


def test_config_validation_fails_before_observation() -> None:
    with pytest.raises(ValidationError):
        ExternalBudget(max_duration_s=0.0)
    with pytest.raises(ValidationError):
        ExternalBudget(max_processes=0)
    with pytest.raises(ValidationError):
        build_external_policy(Settings(phase11_memory_pressure_threshold=1.5))
    with pytest.raises(ValidationError):
        build_external_budget(Settings(phase11_max_recommendations=0))


def test_real_executor_reduces_priority_and_restores() -> None:
    with launch_sleeper() as proc:
        identity = resolve_identity(proc.pid)
        assert identity is not None
        prior = capture_external_state(identity, "external.reduce_priority")
        checkpoint = capture_external_checkpoint(identity, "external.reduce_priority")
        assert checkpoint.verified is True
        executor = ExternalActionExecutor(own_user=identity.owner)
        result = executor.execute(
            action_id="external.reduce_priority",
            identity=identity,
            parameters={"priority_class": "below_normal"},
            checkpoint=checkpoint,
        )
        assert result.executed is True, result.reason
        assert executor.restore(
            action_id="external.reduce_priority", identity=identity, checkpoint=checkpoint
        )
        assert capture_external_state(identity, "external.reduce_priority").get("nice") == prior.get("nice")


def test_executor_rejects_stale_process() -> None:
    with launch_sleeper() as proc:
        identity = resolve_identity(proc.pid)
        assert identity is not None
    # Process has been terminated; the identity is now stale.
    executor = ExternalActionExecutor(own_user=identity.owner)
    result = executor.execute(
        action_id="external.reduce_priority",
        identity=identity,
        parameters={"priority_class": "below_normal"},
        checkpoint=None,
    )
    assert result.executed is False
    assert "stale" in result.reason or "changed" in result.reason or "disappeared" in result.reason


def test_observer_is_read_only_and_private() -> None:
    observer = WindowsProcessObserver()
    report = observer.observe(max_processes=5)
    assert report.privacy_notes
    assert any("no process memory" in note for note in report.privacy_notes)
    for observation in report.processes:
        assert "cmdline" not in observation.measurable
        assert "environ" not in observation.measurable


def test_external_run_report_is_structured() -> None:
    target = make_process_observation(pid=1, name="llama.exe", is_target=True)
    external = make_process_observation(pid=2, name="heavy.exe", rss_bytes=8 * 1024**3, cpu_percent=170.0)
    report = make_report(processes=[target, external], target_pid=1, memory_pressure=0.9)
    engine = ExternalRecommendationEngine(
        observer=ScriptedProcessObserver(report),
        registry=build_default_registry(),
        policy=build_external_policy(Settings()),
        approval_provider=AutoRejectApprovalProvider(),
        executor=StubExternalActionExecutor(),
    )
    run = engine.run(target_pid=1, target_degraded=True, context_identity="ctx", max_cycles=1)
    text = render_external_run(run)
    for expected in ("Phase 11 external recommendation run", "Attribution assessments", "Recommendations", "termination"):
        assert expected in text
