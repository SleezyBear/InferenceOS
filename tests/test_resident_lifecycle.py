"""Phase 5R resident deep-diagnosis lifecycle (deterministic fakes)."""

from __future__ import annotations

from typing import Callable

from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.lifecycle import LifecyclePolicy, RequestPolicy
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.contracts.reasoning import ReasoningDiagnosisContext
from inferenceos.interfaces.backend_adapter import RuntimeParams
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from inferenceos.reasoning.resident_lifecycle import RequestWindow, ResidentLifecycleRunner
from tests.reasoning_helpers import insufficient_bundle, valid_result_json
from tests.resident_helpers import FakeReasoner, FakeResidentBackend, synthetic_model

_Setup = tuple[
    ReasoningDiagnosisContext,
    BottleneckDiagnosis,
    BaselineReport,
    ModelProfile,
    RuntimeParams,
    FakeResidentBackend,
    FakeReasoner,
]


def _setup(*, expected_bytes: int = 256 * 1024 * 1024) -> _Setup:
    context, diagnosis, report = insufficient_bundle()
    engine = LocalReasoningEngine(
        FakeReasoningProvider(
            scripted=[valid_result_json(context, root_cause_class="memory_pressure")]
        )
    )
    model = synthetic_model()
    params = RuntimeParams(context_length=1024)
    target = FakeResidentBackend()
    target.start(model, params)
    reasoner = FakeReasoner(engine=engine, expected_bytes=expected_bytes)
    return context, diagnosis, report, model, params, target, reasoner


def _runner(
    target: FakeResidentBackend,
    model: ModelProfile,
    params: RuntimeParams,
    reasoner: FakeReasoner,
    *,
    request_policy: str = RequestPolicy.REJECT,
    on_diagnosis_window: Callable[[RequestWindow], None] | None = None,
) -> ResidentLifecycleRunner:
    return ResidentLifecycleRunner(
        target=target,
        target_model=model,
        target_params=params,
        policy=LifecyclePolicy(),
        reasoner=reasoner,
        request_policy=request_policy,
        on_diagnosis_window=on_diagnosis_window,
    )


def test_successful_lifecycle_reaches_all_states() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup()
    outcome = _runner(target, model, params, reasoner).run(diagnosis, report=report)
    assert outcome.entered is True
    states = [t.state for t in outcome.transitions]
    for state in (
        "resident_target",
        "entry_decision",
        "checkpoint",
        "unload_target",
        "load_reasoner",
        "deep_diagnosis",
        "unload_reasoner",
        "reload_target",
        "verify_target",
        "resume",
    ):
        assert state in states
    assert outcome.diagnosis is not None
    assert outcome.post_reload is not None and outcome.post_reload.generation_ok
    assert target.is_running() is True
    assert not outcome.errors
    # durations and wall clocks are real, not synthesized
    assert all(t.wall_clock is not None for t in outcome.transitions)
    assert outcome.cost is not None and outcome.cost.total_s > 0.0


def test_economics_refuses_and_leaves_target_running() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup()
    policy = LifecyclePolicy(benefit_scale_s=0.0)
    runner = ResidentLifecycleRunner(
        target=target,
        target_model=model,
        target_params=params,
        policy=policy,
        reasoner=reasoner,
    )
    outcome = runner.run(diagnosis, report=report)
    assert outcome.entered is False
    assert outcome.refused_reason is not None and "cost" in outcome.refused_reason
    assert target.is_running() is True
    assert target.last_unload is None


def test_feasibility_refuses_when_reasoner_cannot_fit() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup(expected_bytes=1 << 50)
    outcome = _runner(target, model, params, reasoner).run(diagnosis, report=report)
    assert outcome.entered is False
    assert outcome.feasibility is not None and not outcome.feasibility.feasible
    assert target.is_running() is True


def _submit(window: RequestWindow) -> None:
    window.submit("hello")


def test_request_reject_policy() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup()
    runner = _runner(
        target,
        model,
        params,
        reasoner,
        request_policy=RequestPolicy.REJECT,
        on_diagnosis_window=_submit,
    )
    outcome = runner.run(diagnosis, report=report)
    assert outcome.requests
    assert outcome.requests[0].rejected is True
    assert outcome.requests[0].queued is False


def test_request_queue_policy() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup()
    runner = _runner(
        target,
        model,
        params,
        reasoner,
        request_policy=RequestPolicy.QUEUE,
        on_diagnosis_window=_submit,
    )
    outcome = runner.run(diagnosis, report=report)
    assert outcome.requests
    assert outcome.requests[0].queued is True
    assert outcome.requests[0].drained is True


def test_target_unload_failure_is_surfaced_and_recovered() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup()
    target.fail_stop = True
    outcome = _runner(target, model, params, reasoner).run(diagnosis, report=report)
    assert outcome.errors
    assert target.is_running() is True


def test_reasoner_load_failure_is_surfaced_and_target_restored() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup()
    reasoner.fail_start = True
    outcome = _runner(target, model, params, reasoner).run(diagnosis, report=report)
    assert outcome.errors
    assert target.is_running() is True


def test_reasoner_diagnosis_failure_is_surfaced() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup()
    reasoner.engine_error = "boom"
    outcome = _runner(target, model, params, reasoner).run(diagnosis, report=report)
    assert outcome.errors
    assert target.is_running() is True


def test_reasoner_unload_failure_is_surfaced() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup()
    reasoner.fail_stop = True
    outcome = _runner(target, model, params, reasoner).run(diagnosis, report=report)
    assert outcome.errors
    # The reasoner remains resident: do not start a second model on top of it.
    assert reasoner.is_running() is True
    assert target.is_running() is False
    assert any("release unconfirmed" in error for error in outcome.errors)


def test_target_reload_failure_is_surfaced_and_retried() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup()
    target.fail_start_times = 1
    outcome = _runner(target, model, params, reasoner).run(diagnosis, report=report)
    assert outcome.errors
    assert target.is_running() is True


def test_post_reload_verification_failure_is_not_success() -> None:
    _, diagnosis, report, model, params, target, reasoner = _setup()
    target.fail_generate = True
    outcome = _runner(target, model, params, reasoner).run(diagnosis, report=report)
    assert outcome.post_reload is not None
    assert outcome.post_reload.generation_ok is False
    assert outcome.errors


def test_request_window_contract_directly() -> None:
    reject = RequestWindow(RequestPolicy.REJECT)
    record = reject.submit("x")
    assert record.rejected and not record.accepted
    queue = RequestWindow(RequestPolicy.QUEUE)
    queued = queue.submit("x")
    assert queued.queued and queued.accepted
    drained = queue.drain()
    assert drained and drained[0].drained
