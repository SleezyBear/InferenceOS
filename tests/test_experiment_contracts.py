"""Phase 7 experiment contracts: existence, round-trip, strictness."""

from __future__ import annotations

import pytest

from inferenceos.contracts.experiment import (
    ExperimentMeasurement,
    ExperimentRequest,
    ExperimentResult,
)
from inferenceos.selftest import (
    sample_experiment_measurement,
    sample_experiment_request,
    sample_experiment_result,
    verify_contract_roundtrip,
)

REQUIRED = {
    "ExperimentRequest",
    "BaselineSnapshot",
    "ExperimentMeasurement",
    "QualityValidationResult",
    "ExperimentComparison",
    "ExperimentDecision",
    "ApplicationResult",
    "RestorationResult",
    "ExperimentResult",
    "ExperimentPolicy",
    "RuntimeConfiguration",
}


def test_phase7_contracts_roundtrip() -> None:
    passed = verify_contract_roundtrip()
    assert REQUIRED.issubset(set(passed))


def test_experiment_result_roundtrips_with_lifecycle() -> None:
    result = sample_experiment_result()
    restored = ExperimentResult.from_json(result.to_json())
    assert restored == result
    assert restored.executed is True
    assert restored.environment_restored is True
    assert restored.restoration is not None and restored.restoration.verified


def test_experiment_request_carries_identity_and_config() -> None:
    request = sample_experiment_request()
    assert request.model
    assert request.workload
    assert request.baseline_configuration.context_length == 32768
    assert request.experiment_configuration.context_length == 8192
    assert request.bottleneck_metric == "peak_rss_bytes"


def test_contracts_reject_unknown_fields() -> None:
    request = sample_experiment_request()
    with pytest.raises(ValueError):
        ExperimentRequest.model_validate({**request.model_dump(), "unexpected": 1})
    measurement = sample_experiment_measurement()
    with pytest.raises(ValueError):
        ExperimentMeasurement.model_validate({**measurement.model_dump(), "unexpected": 1})


def test_measurement_exposes_metrics_and_text() -> None:
    measurement = sample_experiment_measurement()
    assert measurement.valid_runs == 2
    assert measurement.metrics["decode_tokens_per_sec"].mean == pytest.approx(22.0)
    assert measurement.generated_texts
