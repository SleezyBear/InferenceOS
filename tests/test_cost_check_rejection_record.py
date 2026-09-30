"""Step 2 bug: a change the after-test cost check rejects must not be recorded as kept."""

from __future__ import annotations

import pytest

from inferenceos.contracts.enums import ApprovalOutcome, ExperimentDecisionKind, OptimizationOutcome
from inferenceos.runtime import monitor as monitor_module
from inferenceos.runtime.approval import ScriptedApprovalProvider
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.selftest import sample_hardware
from tests.experiment_helpers import ScriptedHarness, build_test_policy
from tests.knowledge_helpers import base_result, make_service, with_decision
from tests.runtime_helpers import ScriptedSampler, ScriptedWorkload, baseline_and_pressured, make_runtime_policy
from tests.test_runtime_intervention import FakeClock, _budget, _improving_responder


def test_build_record_marks_economically_rejected_improvement_as_not_kept() -> None:
    service = make_service()
    result = with_decision(base_result(), ExperimentDecisionKind.IMPROVED)
    assert result.decision_detail is not None
    result = result.model_copy(
        update={"decision_detail": result.decision_detail.model_copy(update={"keep": True})}
    )
    kept = service.build_record(result)
    assert kept.outcome is OptimizationOutcome.IMPROVED
    assert kept.keep is True
    rejected = service.build_record(result, retained=False, not_retained_reason="cost check said no")
    assert rejected.outcome is OptimizationOutcome.REVERTED
    assert rejected.keep is False
    assert any("cost check said no" in item for item in rejected.evidence)
    assert rejected.decision is ExperimentDecisionKind.IMPROVED


def test_monitor_records_cost_check_rejection_as_not_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    real = monitor_module.evaluate_retention_economics

    def always_unaffordable(**kwargs):  # type: ignore[no-untyped-def]
        economics = real(**kwargs)
        return economics.model_copy(
            update={"beneficial": False, "rationale": [*economics.rationale, "forced: not worth keeping"]}
        )

    monkeypatch.setattr(monitor_module, "evaluate_retention_economics", always_unaffordable)
    clock = FakeClock()
    workload = ScriptedWorkload()
    service = make_service()
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=ScriptedSampler(baseline_and_pressured()),
        harness=ScriptedHarness(_improving_responder),
        policy=make_runtime_policy(budget=_budget()),
        approval_provider=ScriptedApprovalProvider([ApprovalOutcome.APPROVED, ApprovalOutcome.APPROVED]),
        knowledge_service=service,
        hardware=sample_hardware(),
        experiment_policy=build_test_policy(),
        clock=clock,
        sleep=clock.sleep,
    )
    run = monitor.run().run
    assert run.interventions
    intervention = run.interventions[0]
    assert intervention.decision is ExperimentDecisionKind.IMPROVED
    assert intervention.retained is False
    assert workload.configuration().use_mmap is False
    records = [r for r in service.history("runtime.enable_mmap") if r.experiment_id == intervention.experiment_id]
    assert len(records) == 1
    assert records[0].keep is False
    # Step 4 decision: measured better but not worth keeping is remembered as
    # "worked, not worth it", never as a bad result.
    assert records[0].outcome is OptimizationOutcome.NOT_WORTH_IT
    assert any("post-measurement cost check" in item for item in records[0].evidence)
