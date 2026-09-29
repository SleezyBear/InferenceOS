"""Context-safety contract and corrected memory-retention semantics.

Covers the hard context requirement contract, its propagation into the optimization
context, the safe candidate gate (including unknown-requirement refusal), request-level
safety, and the corrected Phase 9 memory-retention rule (resource improvement alone is
not an optimization benefit).
"""

from __future__ import annotations

from typing import Any, cast

from inferenceos.config import Settings
from inferenceos.contracts.enums import (
    CandidateEligibility,
    ExperimentDecisionKind,
    OptimizationSurface,
    ParameterType,
    WorkloadKind,
)
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.optimization import ParameterSpec
from inferenceos.contracts.workload import ContextRequirement
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.context import build_optimization_context
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.optimization.registry import ActionRegistry
from inferenceos.profiling.workloads import BenchmarkWorkload
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.runtime.workload import context_requirement_violation
from inferenceos.selftest import sample_model
from tests.diagnosis_fixtures import make_report
from tests.experiment_helpers import SampleResult
from tests.loop_helpers import (
    context_improvement_responder,
    make_loop,
    patched_paging_report,
    report_for_config,
)
from tests.reasoning_helpers import paging_report

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _paging_report(
    *,
    requirement: ContextRequirement | None,
    context_length: int = 2048,
    trained_context_length: int | None = None,
) -> Any:
    base = paging_report()
    runs = []
    for run in base.runs:
        config = run.context.config.model_copy(update={"context_length": context_length})
        model = run.context.model
        if trained_context_length is not None:
            model = model.model_copy(update={"trained_context_length": trained_context_length})
        workload = run.context.workload.model_copy(update={"context_requirement": requirement})
        runs.append(
            run.model_copy(
                update={
                    "context": run.context.model_copy(
                        update={"config": config, "model": model, "workload": workload}
                    )
                }
            )
        )
    return make_report(runs)


def _context_candidate(report: Any, *, registry: ActionRegistry | None = None) -> Any:
    diagnosis = BottleneckDetector().detect(report, source_report_path="<test>")
    plan = generate_candidates(
        diagnosis, report=report, registry=registry or build_default_registry()
    )
    return next(
        (c for c in plan.candidates if c.action_id == "runtime.right_size_context"), None
    )


def _clamped_registry(maximum: int) -> ActionRegistry:
    action = build_default_registry().get("runtime.right_size_context")
    clamped = action.model_copy(
        update={
            "parameters": [
                ParameterSpec(
                    name="context_length",
                    type=ParameterType.INTEGER,
                    minimum=512,
                    maximum=maximum,
                    unit="tokens",
                )
            ]
        }
    )
    registry = ActionRegistry()
    registry.register(clamped)
    return registry


# --------------------------------------------------------------------------- #
# 1. Representation                                                            #
# --------------------------------------------------------------------------- #
def test_context_requirement_distinguishes_unknown_hard_expected_observed() -> None:
    unknown = ContextRequirement()
    assert unknown.required_context_length() is None and not unknown.known

    hard = ContextRequirement(hard_required_context_tokens=1024)
    assert hard.known and hard.required_context_length() == 1024

    # expected/observed are lower bounds, never the hard requirement.
    soft = ContextRequirement(expected_context_tokens=9999, observed_context_tokens=500)
    assert soft.required_context_length() is None and not soft.known


def test_context_requirement_decomposition_is_hard() -> None:
    decomposed = ContextRequirement(max_prompt_tokens=700, required_generation_tokens=128)
    assert decomposed.known and decomposed.required_context_length() == 828


# --------------------------------------------------------------------------- #
# 2. Propagation                                                               #
# --------------------------------------------------------------------------- #
def test_hard_requirement_propagates_into_optimization_context() -> None:
    requirement = ContextRequirement(hard_required_context_tokens=700)
    report = _paging_report(requirement=requirement)
    diagnosis = BottleneckDetector().detect(report, source_report_path="<test>")
    context = build_optimization_context(
        diagnosis, report=report, outcome=None, policy=OptimizationPolicy()
    )
    assert context.context_requirement is not None
    assert context.context_requirement.required_context_length() == 700


# --------------------------------------------------------------------------- #
# 3 / 4 / 5. Candidate safety gate                                             #
# --------------------------------------------------------------------------- #
def test_candidate_rejected_when_target_below_hard_requirement() -> None:
    requirement = ContextRequirement(hard_required_context_tokens=1024)
    report = _paging_report(requirement=requirement)
    # maximum clamps the target to 512, which is below the 1024 requirement.
    candidate = _context_candidate(report, registry=_clamped_registry(512))
    assert candidate is not None
    assert candidate.eligibility is CandidateEligibility.REJECTED
    assert candidate.executable is False
    safety = next(
        p for p in candidate.preconditions if p.check_id == "target_meets_context_requirement"
    )
    assert safety.satisfied is False


def test_candidate_allowed_when_target_meets_hard_requirement() -> None:
    requirement = ContextRequirement(hard_required_context_tokens=900)
    report = _paging_report(requirement=requirement)
    candidate = _context_candidate(report)
    assert candidate is not None
    assert candidate.eligibility is CandidateEligibility.ELIGIBLE
    assert candidate.executable is True
    # current 2048 * (1 - 0.5) = 1024 >= 900.
    assert candidate.parameters["context_length"] == 1024
    assert all(p.satisfied is not False for p in candidate.preconditions)


def test_unknown_requirement_prevents_autonomous_reduction() -> None:
    report = _paging_report(requirement=None)
    candidate = _context_candidate(report)
    assert candidate is not None
    assert candidate.eligibility is CandidateEligibility.INSUFFICIENT_EVIDENCE
    assert candidate.executable is False
    reason = next(
        p for p in candidate.preconditions if p.check_id == "context_requirement_known"
    )
    assert reason.satisfied is None
    assert candidate.parameters == {}


def test_prompt_plus_generation_requirement_is_enforced() -> None:
    # max_prompt 700 + generation 128 = 828 hard requirement.
    requirement = ContextRequirement(max_prompt_tokens=700, required_generation_tokens=128)
    allowed = _context_candidate(_paging_report(requirement=requirement))
    assert allowed is not None and allowed.eligibility is CandidateEligibility.ELIGIBLE
    # A clamped 512 target is below 828 and must be rejected.
    rejected = _context_candidate(
        _paging_report(requirement=requirement), registry=_clamped_registry(512)
    )
    assert rejected is not None and rejected.eligibility is CandidateEligibility.REJECTED


# --------------------------------------------------------------------------- #
# 7. Model/trained-context constraint remains active                           #
# --------------------------------------------------------------------------- #
def test_trained_context_check_remains_active() -> None:
    requirement = ContextRequirement(hard_required_context_tokens=512)
    # target 1024 exceeds the model's trained context of 768.
    report = _paging_report(requirement=requirement, trained_context_length=768)
    candidate = _context_candidate(report)
    assert candidate is not None
    assert candidate.eligibility is CandidateEligibility.REJECTED
    trained = next(
        p for p in candidate.preconditions if p.check_id == "within_trained_context"
    )
    assert trained.satisfied is False


# --------------------------------------------------------------------------- #
# 8. Request-level safety                                                      #
# --------------------------------------------------------------------------- #
def test_context_requirement_violation_helper() -> None:
    known = ContextRequirement(hard_required_context_tokens=900)
    assert (
        context_requirement_violation(RuntimeConfiguration(context_length=512), known)
        is not None
    )
    assert (
        context_requirement_violation(RuntimeConfiguration(context_length=1024), known)
        is None
    )
    # Unknown requirement never blocks serving.
    assert (
        context_requirement_violation(
            RuntimeConfiguration(context_length=64), ContextRequirement()
        )
        is None
    )
    assert context_requirement_violation(RuntimeConfiguration(context_length=64), None) is None


def test_runtime_request_below_requirement_is_refused() -> None:
    workload = BenchmarkWorkload(
        workload_id="declared",
        kind=WorkloadKind.SHORT,
        description="declared bound",
        prompt="hello world",
        n_predict=64,
        seed=1,
        temperature=0.0,
        min_context_tokens=900,
        context_requirement=ContextRequirement(hard_required_context_tokens=900),
    )

    class _FakeResidentAdapter:
        pid = None
        identity = None
        last_generation = None
        last_param_application = None

        @property
        def is_loaded(self) -> bool:
            return True

        def generate(self, request: Any) -> Any:  # pragma: no cover - must not be called
            raise AssertionError("request must be refused before generation")

    resident = ResidentActiveWorkload(
        settings=Settings(),
        model=sample_model(),
        adapter=cast(Any, _FakeResidentAdapter()),
        workload=workload,
        configuration=RuntimeConfiguration(context_length=512, gpu_offload_layers=0),
    )
    resident._serve_one()  # noqa: SLF001 - exercising the refusal seam directly
    record = resident.last_request()
    assert record is not None and record.success is False
    assert record.error is not None and "hard context requirement" in record.error


# --------------------------------------------------------------------------- #
# 9 / 10. Corrected Phase 9 retention semantics                                #
# --------------------------------------------------------------------------- #
def _rss_only_responder(task: Any, options: Any, call: int) -> SampleResult:
    if (task.context_length or 0) >= 2048:
        return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)
    # Resource improves, performance does not.
    return SampleResult(rss_bytes=900_000_000, hard_faults_per_sec=1.0, decode_tps=20.0)


def test_memory_optimization_not_retained_from_rss_alone() -> None:
    report = patched_paging_report()
    stack = make_loop(responder=_rss_only_responder, report=report)
    result = stack.loop.run(report, report_path="<test-rss-only>")
    assert result.changed is False
    experimented = [i for i in result.run.iterations if i.experiment_id is not None]
    assert experimented
    # The technical experiment improved the resource metric, but the loop must not
    # retain a memory-surfaced change without a measured performance outcome.
    assert all(not i.retained for i in experimented)
    gated = [
        i
        for i in experimented
        if any("no measured performance outcome" in note for note in i.notes)
    ]
    assert gated
    assert all(i.experiment_status is ExperimentDecisionKind.IMPROVED for i in gated)


def test_memory_optimization_retained_with_genuine_performance_gain() -> None:
    report = patched_paging_report()
    stack = make_loop(
        responder=context_improvement_responder,
        report=report,
        observer=lambda config: report_for_config(report, config),
    )
    result = stack.loop.run(report, report_path="<test-performance>")
    assert result.changed is True
    assert result.run.retained_candidate_ids


# --------------------------------------------------------------------------- #
# 11 / 12. Retention gate is scoped to resource surfaces                       #
# --------------------------------------------------------------------------- #
def test_gpu_and_threads_surfaces_are_not_resource_gated() -> None:
    registry = build_default_registry()
    assert registry.get("runtime.reduce_threads").surface is OptimizationSurface.RUNTIME_PARAMS
    assert (
        registry.get("runtime.increase_gpu_offload_layers").surface
        is OptimizationSurface.GPU_OFFLOAD
    )
    assert (
        registry.get("runtime.right_size_context").surface
        is OptimizationSurface.CONTEXT_MANAGEMENT
    )
