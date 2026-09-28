"""`runtime.reduce_threads` eligibility wired to observed *effective* runtime threads.

Proves the existing oversubscription-remediation action uses authoritative runtime state
(``effective_threads``) rather than the requested value, that a no-op is represented as a
distinct ``ALREADY_AT_DESIRED_STATE`` NO_CHANGE, and that unknown effective state never
becomes autonomously eligible. No thread search is added.
"""

from __future__ import annotations

from inferenceos.contracts.enums import (
    ApprovalRequirement,
    BackendType,
    BottleneckKind,
    CandidateEligibility,
    Reversibility,
    RiskLevel,
)
from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.optimization import CandidatePlan, OptimizationCandidate
from inferenceos.contracts.profiling import ProfilingRun
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.experiment.executor import apply_to_configuration
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.strategies import ALREADY_AT_DESIRED_STATE
from tests.diagnosis_fixtures import make_report, make_run
from tests.optimization_helpers import paging_bundle

_THREAD_ACTION = "runtime.reduce_threads"


def _run(
    condition: str, decode: float, cpu: float, requested: int | None, effective: int | None
) -> ProfilingRun:
    run = make_run(
        condition=condition,
        backend=BackendType.CPU,
        gpu_offload_layers=0,
        gpu_offload_total=28,
        metrics={
            "decode_tokens_per_sec": decode,
            "total_latency_s": 2.0,
            "process_cpu_percent": cpu,
            "gpu_utilization_percent": 5.0,
            "peak_rss_bytes": 1_300_000_000,
            "available_ram_bytes": float(8 * 1024**3),
            "total_ram_bytes": float(16 * 1024**3),
        },
    )
    config = run.context.config.model_copy(
        update={"requested_threads": requested, "effective_threads": effective}
    )
    return run.model_copy(
        update={"context": run.context.model_copy(update={"config": config})}
    )


def _cpu_setup(
    *, requested: int | None, effective: int | None, physical: int = 6
) -> tuple[BottleneckDiagnosis, CandidatePlan, list[OptimizationCandidate]]:
    """A CPU-diagnosing report (no GPU offload) with the observed thread relationship."""
    runs = []
    for _ in range(2):
        runs.append(_run("control", 20.0, 600.0, requested, effective))
        runs.append(_run("contended", 10.0, 500.0, requested, effective))
    report = make_report(runs)
    hardware = report.hardware.model_copy(
        update={
            "cpu": report.hardware.cpu.model_copy(
                update={"physical_cores": physical, "logical_cores": 12}
            )
        }
    )
    report = report.model_copy(update={"hardware": hardware})
    diagnosis = BottleneckDetector().detect(report)
    plan = generate_candidates(diagnosis, report=report)
    thread_candidates = [c for c in plan.candidates if c.action_id == _THREAD_ACTION]
    return diagnosis, plan, thread_candidates


# --------------------------------------------------------------------------- #
# 1 / 2 / 6  Effective oversubscription prepares the physical-core target        #
# --------------------------------------------------------------------------- #
def test_explicit_request_oversubscribed_prepares_physical() -> None:
    diagnosis, _, candidates = _cpu_setup(requested=12, effective=12, physical=6)
    assert diagnosis.primary is not None and diagnosis.primary.kind is BottleneckKind.CPU
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.eligibility is CandidateEligibility.ELIGIBLE
    assert candidate.executable is True
    assert candidate.parameters == {"threads": 6}


def test_backend_selected_runtime_oversubscribed_prepares_physical() -> None:
    _, _, candidates = _cpu_setup(requested=None, effective=12, physical=6)
    assert len(candidates) == 1
    assert candidates[0].eligibility is CandidateEligibility.ELIGIBLE
    assert candidates[0].parameters == {"threads": 6}


def test_eligibility_is_based_on_effective_not_requested() -> None:
    # requested unset, effective oversubscribed -> eligible (based on effective).
    _, _, from_effective = _cpu_setup(requested=None, effective=12, physical=6)
    assert from_effective and from_effective[0].parameters == {"threads": 6}
    # requested oversubscribed but effective unknown -> NOT eligible.
    _, _, unknown = _cpu_setup(requested=12, effective=None, physical=6)
    assert unknown and unknown[0].eligibility is not CandidateEligibility.ELIGIBLE


# --------------------------------------------------------------------------- #
# 3 / 4  Already at (or below) the desired state -> NO_CHANGE                    #
# --------------------------------------------------------------------------- #
def test_already_at_desired_state_is_no_change() -> None:
    _, plan, candidates = _cpu_setup(requested=None, effective=6, physical=6)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.eligibility is not CandidateEligibility.ELIGIBLE
    assert candidate.executable is False
    assert candidate.parameters == {}  # no target_threads prepared
    assert candidate.rejection_reason == ALREADY_AT_DESIRED_STATE
    detail = next(
        p.detail for p in candidate.preconditions if p.check_id == "threads_exceed_physical_cores"
    )
    assert ALREADY_AT_DESIRED_STATE in detail
    assert plan.outcome.value == "no_viable_candidates"


def test_below_physical_cores_is_no_change() -> None:
    _, _, candidates = _cpu_setup(requested=4, effective=4, physical=6)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.eligibility is not CandidateEligibility.ELIGIBLE
    assert candidate.parameters == {}
    assert candidate.rejection_reason == ALREADY_AT_DESIRED_STATE


# --------------------------------------------------------------------------- #
# 5  Unknown effective state -> never autonomously eligible                     #
# --------------------------------------------------------------------------- #
def test_unknown_effective_state_is_not_eligible() -> None:
    _, _, candidates = _cpu_setup(requested=None, effective=None, physical=6)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.eligibility is CandidateEligibility.INSUFFICIENT_EVIDENCE
    assert candidate.executable is False
    assert candidate.parameters == {}
    known = next(p for p in candidate.preconditions if p.check_id == "thread_count_known")
    assert known.satisfied is None


# --------------------------------------------------------------------------- #
# 7 / 10  Existing diagnosis gates and unrelated candidates unchanged           #
# --------------------------------------------------------------------------- #
def test_thread_candidate_only_from_cpu_diagnosis() -> None:
    # A non-CPU diagnosis must not produce a thread candidate.
    _, diagnosis, report = paging_bundle()
    plan = generate_candidates(diagnosis, report=report)
    assert all(c.action_id != _THREAD_ACTION for c in plan.candidates)
    # The paging surface is unaffected.
    assert any(c.action_id == "runtime.right_size_context" for c in plan.candidates)


def test_thread_candidate_requires_supported_evidence() -> None:
    # No effective diagnosis -> evidence precondition is not established -> not eligible.
    from tests.optimization_helpers import synthetic_diagnosis

    diagnosis = synthetic_diagnosis(BottleneckKind.PAGING)
    plan = generate_candidates(diagnosis, report=None)
    assert all(
        c.eligibility is not CandidateEligibility.ELIGIBLE
        for c in plan.candidates
        if c.action_id == _THREAD_ACTION
    )


# --------------------------------------------------------------------------- #
# 8  Existing safety/resource gates apply to the prepared candidate             #
# --------------------------------------------------------------------------- #
def test_prepared_candidate_keeps_existing_safety_properties() -> None:
    _, _, candidates = _cpu_setup(requested=None, effective=12, physical=6)
    candidate = candidates[0]
    assert candidate.risk.reversibility is Reversibility.REVERSIBLE
    assert candidate.risk.risk_level is RiskLevel.LOW
    assert candidate.risk.rollback_available is True
    assert candidate.approval is ApprovalRequirement.NOT_REQUIRED
    assert candidate.executable is True


# --------------------------------------------------------------------------- #
# 9  Existing restoration/executor mapping unchanged                            #
# --------------------------------------------------------------------------- #
def test_executor_mapping_applies_only_threads() -> None:
    prior = RuntimeConfiguration(context_length=4096, gpu_offload_layers=8, threads=12)
    updated = apply_to_configuration(_THREAD_ACTION, {"threads": 6}, prior)
    assert updated.threads == 6
    assert updated.context_length == 4096
    assert updated.gpu_offload_layers == 8
    # Restoring the prior configuration is an exact inverse.
    restored = apply_to_configuration(_THREAD_ACTION, {"threads": 12}, updated)
    assert restored == prior
