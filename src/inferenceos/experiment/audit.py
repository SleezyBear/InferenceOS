"""Map a Phase 7 experiment onto the existing persistent audit format.

Phase 8 owns durable knowledge/rollback, so Phase 7 does not invent a parallel store.
It emits an :class:`AuditRecord` (the Phase 0 audit schema) covering candidate →
experiment → decision → applied changes, which a later phase can persist and index.
"""

from __future__ import annotations

from inferenceos.audit.schema import AppliedChange, AuditRecord
from inferenceos.contracts.candidate import Candidate
from inferenceos.contracts.enums import CandidateKind
from inferenceos.contracts.experiment import ExperimentResult
from inferenceos.signature import Signature, make_signature


def build_audit_record(
    result: ExperimentResult, *, signature: Signature | None = None
) -> AuditRecord:
    request = result.request
    if signature is None and request is not None:
        signature = make_signature(request.hardware, request.model, request.workload)
    if signature is None:  # pragma: no cover - result always carries a request
        signature = Signature(hardware="unknown", model="unknown", workload="unknown")

    baseline = None
    if result.baseline_measurement is not None and result.baseline_measurement.runs:
        baseline = result.baseline_measurement.runs[0].observation

    candidates: list[Candidate] = []
    if request is not None:
        candidates.append(
            Candidate(
                candidate_id=request.candidate_id,
                kind=_surface_to_kind(request.action_id),
                description=f"Phase 6 candidate {request.action_id}",
                parameters=dict(request.parameters),
                expected_benefit=0.0,
                confidence=0.0,
                cost=0.0,
                expected_value=0.0,
                rationale="Phase 7 experiment candidate",
            )
        )

    changes: list[AppliedChange] = []
    if result.application is not None and result.application.applied:
        changes = [
            AppliedChange(
                parameter=change.parameter,
                prior_value=change.prior_value,
                applied_value=change.applied_value,
            )
            for change in result.application.changes
        ]

    return AuditRecord(
        record_id=f"audit-{result.experiment_id}",
        signature=signature,
        baseline=baseline,
        candidates=candidates,
        experiment=result,
        decision=result.decision,
        applied_changes=changes,
        notes=result.summary,
    )


def _surface_to_kind(action_id: str) -> CandidateKind:
    prefix = {
        "runtime.right_size_context": CandidateKind.CONTEXT_LENGTH,
        "runtime.enable_mmap": CandidateKind.MMAP,
        "runtime.reduce_gpu_offload_layers": CandidateKind.GPU_OFFLOAD_LAYERS,
        "runtime.reduce_threads": CandidateKind.THREADS,
    }
    return prefix.get(action_id, CandidateKind.CONTEXT_LENGTH)
