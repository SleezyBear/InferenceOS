"""Provenance binding and stale-evidence rejection."""

from __future__ import annotations

from inferenceos.contracts.experiment import ExperimentResult
from inferenceos.experiment.provenance import (
    build_provenance,
    parameters_hash,
    verify_provenance,
)
from tests.experiment_helpers import make_request


def _result(run_id: str) -> ExperimentResult:
    request = make_request().model_copy(
        update={
            "run_id": run_id,
            "parameters": {"context_length": 1024},
            "source_report_path": "reports/example.json",
        }
    )
    provenance = build_provenance(request, run_id=run_id)
    return ExperimentResult(
        experiment_id=f"exp-{run_id}",
        run_id=run_id,
        provenance=provenance,
        candidate_id=request.candidate_id,
        parameters=dict(request.parameters),
    )


def test_provenance_binds_run_and_parameters() -> None:
    result = _result("run-a")
    check = verify_provenance(
        result,
        expected_run_id="run-a",
        expected_parameters_hash=parameters_hash({"context_length": 1024}),
    )
    assert check.ok is True
    assert check.reasons == []


def test_wrong_run_id_is_rejected() -> None:
    result = _result("run-a")
    check = verify_provenance(result, expected_run_id="run-b")
    assert check.ok is False
    assert any("run id mismatch" in reason for reason in check.reasons)


def test_wrong_parameters_hash_is_rejected() -> None:
    result = _result("run-a")
    check = verify_provenance(
        result,
        expected_run_id="run-a",
        expected_parameters_hash=parameters_hash({"context_length": 4096}),
    )
    assert check.ok is False


def test_missing_provenance_is_rejected() -> None:
    result = ExperimentResult(experiment_id="exp-x", candidate_id="cand-x")
    check = verify_provenance(result, expected_run_id="run-a")
    assert check.ok is False
    assert any("no provenance" in reason for reason in check.reasons)
