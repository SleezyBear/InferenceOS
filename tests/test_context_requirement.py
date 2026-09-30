"""Context-safety contract and corrected memory-retention semantics.

Covers the hard context requirement contract, its propagation into the optimization
context, the rule that no context-size candidate is ever generated, request-level
safety, and the corrected Phase 9 memory-retention semantics for the executable mmap
candidate (a measured memory objective is a legitimate retention basis).
"""

from __future__ import annotations

from typing import Any, cast

from inferenceos.config import Settings
from inferenceos.contracts.enums import (
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
    make_loop,
    no_improvement_responder,
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
# 3 / 4 / 5. Context candidates are never generated                             #
# --------------------------------------------------------------------------- #
def test_no_context_candidate_when_target_would_be_below_hard_requirement() -> None:
    requirement = ContextRequirement(hard_required_context_tokens=1024)
    report = _paging_report(requirement=requirement)
    # Even a clamped target below the hard requirement must not produce a candidate.
    assert _context_candidate(report, registry=_clamped_registry(512)) is None


def test_no_context_candidate_when_target_would_meet_hard_requirement() -> None:
    requirement = ContextRequirement(hard_required_context_tokens=900)
    report = _paging_report(requirement=requirement)
    assert _context_candidate(report) is None


def test_no_context_candidate_when_requirement_is_unknown() -> None:
    report = _paging_report(requirement=None)
    assert _context_candidate(report) is None


def test_no_context_candidate_for_prompt_plus_generation_requirement() -> None:
    # max_prompt 700 + generation 128 = 828 hard requirement.
    requirement = ContextRequirement(max_prompt_tokens=700, required_generation_tokens=128)
    assert _context_candidate(_paging_report(requirement=requirement)) is None
    assert (
        _context_candidate(
            _paging_report(requirement=requirement), registry=_clamped_registry(512)
        )
        is None
    )


# --------------------------------------------------------------------------- #
# 7. Trained-context constraint does not resurrect a context candidate          #
# --------------------------------------------------------------------------- #
def test_no_context_candidate_with_trained_context_constraint() -> None:
    requirement = ContextRequirement(hard_required_context_tokens=512)
    report = _paging_report(requirement=requirement, trained_context_length=768)
    assert _context_candidate(report) is None


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
def test_loop_never_creates_a_context_candidate() -> None:
    # No plan surfaced by the loop contains a context candidate, and the loop only
    # experiments with the executable memory action (here, mmap).
    report = patched_paging_report()
    stack = make_loop(responder=no_improvement_responder, report=report)
    result = stack.loop.run(report, report_path="<test-no-context-candidate>")
    assert result.changed is False
    experimented = [i for i in result.run.iterations if i.experiment_id is not None]
    assert experimented
    assert all(
        iteration.selected_action_id != "runtime.right_size_context"
        for iteration in result.run.iterations
    )
    assert {i.selected_action_id for i in experimented} == {"runtime.enable_mmap"}
    assert all(
        candidate.action_id != "runtime.right_size_context"
        for iteration in result.run.iterations
        if iteration.plan is not None
        for candidate in iteration.plan.candidates
    )


def _memory_only_responder(task: Any, options: Any, call: int) -> SampleResult:
    # Memory improves (RSS and page faults) while throughput is unchanged.
    if options.use_mmap:
        return SampleResult(rss_bytes=900_000_000, hard_faults_per_sec=1.0, decode_tps=20.0)
    return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def _speed_up_responder(task: Any, options: Any, call: int) -> SampleResult:
    # Throughput and the diagnosed page-fault bottleneck improve while resident memory
    # is unchanged (a genuine speed-up, not a memory-only result).
    if options.use_mmap:
        return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=1.0, decode_tps=30.0)
    return SampleResult(rss_bytes=1_300_000_000, hard_faults_per_sec=5.0, decode_tps=20.0)


def test_mmap_memory_only_improvement_is_retained() -> None:
    report = patched_paging_report()
    stack = make_loop(
        responder=_memory_only_responder,
        report=report,
        observer=lambda config: report_for_config(report, config),
    )
    result = stack.loop.run(report, report_path="<test-mmap-memory-only>")
    assert result.changed is True
    retained = [i for i in result.run.iterations if i.retained]
    assert retained
    assert retained[0].selected_action_id == "runtime.enable_mmap"
    # Retained on the memory objective, not a fabricated performance claim.
    assert any("memory" in note for note in retained[0].notes)
    assert result.final_configuration.use_mmap is True


def test_mmap_real_speed_up_is_retained() -> None:
    report = patched_paging_report()
    stack = make_loop(
        responder=_speed_up_responder,
        report=report,
        observer=lambda config: report_for_config(report, config),
    )
    result = stack.loop.run(report, report_path="<test-mmap-speed-up>")
    assert result.changed is True
    retained = [i for i in result.run.iterations if i.retained]
    assert retained
    assert retained[0].selected_action_id == "runtime.enable_mmap"
    assert result.final_configuration.use_mmap is True


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
