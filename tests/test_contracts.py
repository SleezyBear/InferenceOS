"""Contract-level tests: required types exist, round-trip, and behave correctly."""

from __future__ import annotations

import pytest

from inferenceos.contracts.enums import AttentionType, BackendType
from inferenceos.contracts.model import ModelProfile
from inferenceos.selftest import sample_model, verify_contract_roundtrip

REQUIRED_CONTRACTS = {
    "HardwareProfile",
    "ModelProfile",
    "WorkloadProfile",
    "SystemState",
    "Observation",
    "Bottleneck",
    "Candidate",
    "ExperimentResult",
    "ReasoningRequest",
    "ReasoningResponse",
}


def test_all_required_contracts_roundtrip() -> None:
    passed = verify_contract_roundtrip()
    assert REQUIRED_CONTRACTS.issubset(set(passed))
    assert "AuditRecord" in passed


def test_contracts_reject_unknown_fields() -> None:
    profile = sample_model()
    with pytest.raises(ValueError):
        ModelProfile.model_validate({**profile.model_dump(), "unexpected": 1})


def test_model_derives_gqa() -> None:
    profile = sample_model()
    assert profile.attention_type is AttentionType.GQA


def _with_kv_heads(n_kv_heads: int) -> ModelProfile:
    data = sample_model().model_dump()
    data["n_kv_heads"] = n_kv_heads
    data["attention_type"] = AttentionType.UNKNOWN
    return ModelProfile.model_validate(data)


def test_model_derives_mqa() -> None:
    assert _with_kv_heads(1).attention_type is AttentionType.MQA


def test_model_derives_mha() -> None:
    assert _with_kv_heads(16).attention_type is AttentionType.MHA


def test_kv_estimate_scales_linearly_with_context() -> None:
    profile = sample_model()
    per_token = profile.kv_cache_bytes_per_token
    assert profile.estimate_kv_cache_bytes(4096) == per_token * 4096
    assert profile.estimate_kv_cache_bytes(8192) == 2 * profile.estimate_kv_cache_bytes(4096)


def test_kv_estimate_rejects_bad_context() -> None:
    with pytest.raises(ValueError):
        sample_model().estimate_kv_cache_bytes(0)


def test_backend_type_values() -> None:
    assert BackendType.VULKAN.value == "vulkan"
    assert BackendType.CPU.value == "cpu"
