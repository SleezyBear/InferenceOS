"""Experiment provenance binding and stale-evidence rejection.

A result must be tied to the exact run/report/diagnosis/parameters that produced it.
This prevents an acceptance gate from silently consuming a stale prior experiment or
an unrelated report: ``verify_provenance`` fails when the run id or parameter hash
does not match the current run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.contracts.experiment import (
    ExperimentProvenance,
    ExperimentRequest,
    ExperimentResult,
)


def _hash_file(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    try:
        size = candidate.stat().st_size
        digest.update(str(size).encode("ascii"))
        with candidate.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
            if size > 2 * 1024 * 1024:
                import os

                handle.seek(-1024 * 1024, os.SEEK_END)
                digest.update(handle.read(1024 * 1024))
    except OSError:
        return None
    return digest.hexdigest()[:32]


def parameters_hash(parameters: dict[str, object]) -> str:
    encoded = json.dumps(parameters, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def build_provenance(
    request: ExperimentRequest,
    *,
    run_id: str,
    source_report_path: str | None = None,
    source_report_hash: str | None = None,
    model_content_hash: str | None = None,
) -> ExperimentProvenance:
    report_hash = source_report_hash
    if report_hash is None and source_report_path is not None:
        report_hash = _hash_file(source_report_path)
    return ExperimentProvenance(
        run_id=run_id,
        created_at=datetime.now(timezone.utc),
        source_report_path=source_report_path or request.model.path,
        source_report_hash=report_hash,
        diagnosis_id=request.source_diagnosis_id,
        plan_id=request.source_plan_id,
        candidate_id=request.candidate_id,
        action_id=request.action_id,
        parameters_hash=parameters_hash(request.parameters),
        model_path=request.model.path,
        model_content_hash=model_content_hash,
    )


@dataclass
class ProvenanceCheck:
    ok: bool
    run_id: str
    reasons: list[str] = field(default_factory=list)


def verify_provenance(
    result: ExperimentResult,
    *,
    expected_run_id: str,
    expected_parameters_hash: str | None = None,
    expected_report_hash: str | None = None,
) -> ProvenanceCheck:
    """Confirm a result belongs to the current run (rejects stale/unrelated results)."""
    reasons: list[str] = []
    provenance = result.provenance
    if provenance is None:
        return ProvenanceCheck(
            ok=False, run_id=expected_run_id, reasons=["result carries no provenance"]
        )
    if provenance.run_id != expected_run_id:
        reasons.append(
            f"run id mismatch: result={provenance.run_id!r} current={expected_run_id!r}"
        )
    if result.run_id is not None and result.run_id != expected_run_id:
        reasons.append(
            f"result.run_id mismatch: {result.run_id!r} != {expected_run_id!r}"
        )
    if (
        expected_parameters_hash is not None
        and provenance.parameters_hash != expected_parameters_hash
    ):
        reasons.append(
            "parameters hash mismatch: result provenance does not match the applied "
            "candidate parameters"
        )
    if expected_report_hash is not None and provenance.source_report_hash != expected_report_hash:
        reasons.append("source report hash mismatch: possible stale evidence")
    return ProvenanceCheck(ok=not reasons, run_id=expected_run_id, reasons=reasons)
