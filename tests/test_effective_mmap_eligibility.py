"""Effective mmap-state eligibility.

Phase 17 added ``BackendTelemetry.mmap_effective``. These tests prove the optimizer now
uses that *effective* backend evidence for ``runtime.enable_mmap`` eligibility (falling
back to the configured value only when effective state is unknown), that unknown never
becomes "disabled", and that the effective value is propagated into the runtime report.
"""

from __future__ import annotations

from typing import Any

from inferenceos.contracts.diagnosis import BottleneckDiagnosis
from inferenceos.contracts.enums import (
    BackendType,
    BottleneckKind,
    CandidateEligibility,
    WorkloadState,
)
from inferenceos.contracts.optimization import CandidatePlan
from inferenceos.contracts.profiling import BaselineReport, ProfilingRun
from inferenceos.contracts.runtime import RuntimeObservation
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.optimization.generator import generate_candidates
from tests.diagnosis_fixtures import make_report, make_run


def _run(
    *,
    condition: str,
    use_mmap: bool | None,
    effective_mmap: bool | None,
    faults: float,
    pressure: float,
    latency: float,
) -> ProfilingRun:
    run = make_run(
        condition=condition,
        backend=BackendType.CPU,
        gpu_offload_layers=0,
        gpu_offload_total=28,
        hard_required_context_tokens=None,
        memory_pressure=pressure,
        metrics={
            "decode_tokens_per_sec": 24.0 if faults < 1.0 else 9.0,
            "total_latency_s": latency,
            "hard_page_faults_per_sec": faults,
            "peak_rss_bytes": 1_300_000_000,
            "available_ram_bytes": float(8 * 1024**3),
            "total_ram_bytes": float(16 * 1024**3),
            "process_cpu_percent": 120.0,
        },
    )
    config = run.context.config.model_copy(
        update={"context_length": 2048, "use_mmap": use_mmap, "effective_mmap": effective_mmap}
    )
    return run.model_copy(update={"context": run.context.model_copy(update={"config": config})})


def _paging_plan(use_mmap: bool | None, effective_mmap: bool | None) -> tuple[BottleneckDiagnosis, CandidatePlan]:
    runs: list[ProfilingRun] = []
    for _ in range(2):
        runs.append(_run(condition="control", use_mmap=use_mmap, effective_mmap=effective_mmap,
                         faults=0.5, pressure=0.6, latency=1.4))
        runs.append(_run(condition="pressured", use_mmap=use_mmap, effective_mmap=effective_mmap,
                         faults=18000.0, pressure=0.85, latency=2.6))
    report = make_report(runs)
    diag = BottleneckDetector().detect(report, source_report_path="<test>")
    return diag, generate_candidates(diag, report=report)


def _mmap_candidate(plan: CandidatePlan) -> Any:
    return next(c for c in plan.candidates if c.action_id == "runtime.enable_mmap")


def test_paging_diagnosis_is_the_effective_bottleneck() -> None:
    diag, _ = _paging_plan(use_mmap=False, effective_mmap=False)
    assert diag.primary is not None and diag.primary.kind is BottleneckKind.PAGING


def test_effective_disabled_is_eligible() -> None:
    _, plan = _paging_plan(use_mmap=None, effective_mmap=False)
    candidate = _mmap_candidate(plan)
    assert candidate.eligibility is CandidateEligibility.ELIGIBLE
    assert candidate.executable is True
    detail = next(p.detail for p in candidate.preconditions if p.check_id == "mmap_config_known")
    assert "effective" in detail


def test_effective_enabled_is_a_noop_regardless_of_configured() -> None:
    # Effective evidence wins over the configured value.
    _, plan = _paging_plan(use_mmap=False, effective_mmap=True)
    candidate = _mmap_candidate(plan)
    assert candidate.eligibility is not CandidateEligibility.ELIGIBLE
    assert candidate.parameters == {"use_mmap": True}  # prepared but not executable
    pre = next(p for p in candidate.preconditions if p.check_id == "mmap_currently_disabled")
    assert pre.satisfied is False
    assert "already enabled" in pre.detail


def test_configured_false_is_fallback_when_effective_unknown() -> None:
    _, plan = _paging_plan(use_mmap=False, effective_mmap=None)
    candidate = _mmap_candidate(plan)
    assert candidate.eligibility is CandidateEligibility.ELIGIBLE
    detail = next(p.detail for p in candidate.preconditions if p.check_id == "mmap_config_known")
    assert "configured" in detail


def test_unknown_state_is_insufficient_evidence_never_disabled() -> None:
    _, plan = _paging_plan(use_mmap=None, effective_mmap=None)
    candidate = _mmap_candidate(plan)
    assert candidate.eligibility is CandidateEligibility.INSUFFICIENT_EVIDENCE
    known = next(p for p in candidate.preconditions if p.check_id == "mmap_config_known")
    assert known.satisfied is None
    disabled = next(p for p in candidate.preconditions if p.check_id == "mmap_currently_disabled")
    assert disabled.satisfied is None


def test_runtime_report_carries_effective_mmap() -> None:
    from datetime import datetime, timezone

    from inferenceos.runtime.approval import AutoRejectApprovalProvider
    from inferenceos.runtime.monitor import RuntimeMonitor
    from inferenceos.runtime.notification import CollectingNotificationSink
    from inferenceos.runtime.verification import ScriptedRuntimeSampler, ScriptedWorkload
    from inferenceos.selftest import sample_hardware
    from tests.runtime_helpers import make_runtime_policy

    workload = ScriptedWorkload(effective_mmap=True)
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=ScriptedRuntimeSampler(lambda sequence, condition: {}),
        policy=make_runtime_policy(),
        approval_provider=AutoRejectApprovalProvider(),
        notification_sink=CollectingNotificationSink(),
        hardware=sample_hardware(),
    )
    observation = RuntimeObservation(
        observation_id="obs-mmap",
        captured_at=datetime.now(timezone.utc),
        workload_state=WorkloadState.RUNNING,
        effective_mmap=True,
        metrics={"available_ram_bytes": 8_000.0},
    )
    run = monitor._to_run(
        observation,
        condition="baseline",
        model=workload.model_profile(),
        workload=workload.workload_profile(),
        hardware=sample_hardware(),
    )
    assert run.context.config.effective_mmap is True
