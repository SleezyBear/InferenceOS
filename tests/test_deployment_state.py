"""R5: persisted DeploymentState integrity and fail-closed validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferenceos.config import Settings
from inferenceos.contracts.enums import QuantizationType
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.fresh import DeploymentState, DeploymentStatus
from inferenceos.contracts.model import ModelProfile
from inferenceos.deployment import (
    DeploymentIntegrityError,
    file_sha256,
    load_deployment_state,
    validate_deployment,
    write_deployment_state,
)


def _settings(tmp_path: Path) -> Settings:
    backend_dir = tmp_path / "llama"
    backend_dir.mkdir(exist_ok=True)
    (backend_dir / "llama-server.exe").write_bytes(b"")
    return Settings().model_copy(update={"llama_cpp_dir": backend_dir})


def _profile(tmp_path: Path, *, trained_context: int = 4096) -> ModelProfile:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUFdummy")
    return ModelProfile(
        path=str(model_path),
        name="dummy",
        architecture="llama",
        quantization=QuantizationType.Q4_K_M,
        file_size_bytes=model_path.stat().st_size,
        n_layers=28,
        n_attention_heads=12,
        n_kv_heads=2,
        head_dim=128,
        key_length=128,
        value_length=128,
        trained_context_length=trained_context,
        weight_bytes=model_path.stat().st_size,
        kv_cache_bytes_per_token=28 * 2 * 256 * 2,
    )


def _state(profile: ModelProfile, **overrides: object) -> DeploymentState:
    base: dict[str, object] = {
        "run_id": "run-1",
        "status": DeploymentStatus.NO_CHANGE,
        "model_ref": profile.name,
        "model_path": profile.path,
        "model_content_hash": file_sha256(profile.path),
        "backend_name": "llama.cpp (resident)",
        "baseline_configuration": RuntimeConfiguration(context_length=1024),
        "final_configuration": RuntimeConfiguration(context_length=1024, gpu_offload_layers=0),
        "provenance_hash": "provenance-1",
    }
    base.update(overrides)
    return DeploymentState.model_validate(base)


def test_valid_state_round_trips_and_validates(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = _profile(tmp_path)
    path = tmp_path / "deployment.json"
    write_deployment_state(path, _state(profile))
    loaded = load_deployment_state(path)
    assert loaded.integrity_hash is not None
    validation = validate_deployment(loaded, settings=settings, model_profile=profile)
    assert validation.valid is True
    assert validation.model_hash_verified is True


def test_tampered_state_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = _profile(tmp_path)
    path = tmp_path / "deployment.json"
    write_deployment_state(path, _state(profile))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_configuration"]["context_length"] = 4096
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DeploymentIntegrityError):
        load_deployment_state(path)


def test_unsigned_state_is_rejected(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    path = tmp_path / "unsigned.json"
    _ = _settings(tmp_path)
    path.write_text(_state(profile).model_dump_json(), encoding="utf-8")
    with pytest.raises(DeploymentIntegrityError):
        load_deployment_state(path)


def test_non_servable_status_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = _profile(tmp_path)
    state = _state(profile, status=DeploymentStatus.SAFETY_STOP)
    validation = validate_deployment(state, settings=settings, model_profile=profile)
    assert validation.valid is False
    assert any("not servable" in reason for reason in validation.reasons)


def test_missing_model_and_hash_mismatch_are_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = _profile(tmp_path)
    missing = _state(profile, model_path=str(tmp_path / "gone.gguf"))
    validation = validate_deployment(missing, settings=settings, model_profile=profile)
    assert validation.valid is False

    mismatch = _state(profile, model_content_hash="deadbeef")
    validation2 = validate_deployment(mismatch, settings=settings, model_profile=profile)
    assert validation2.valid is False
    assert any("content hash" in reason for reason in validation2.reasons)


def test_incompatible_configuration_is_rejected(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    profile = _profile(tmp_path, trained_context=2048)
    state = _state(
        profile,
        final_configuration=RuntimeConfiguration(
            context_length=8192, gpu_offload_layers=999
        ),
    )
    validation = validate_deployment(state, settings=settings, model_profile=profile)
    assert validation.valid is False
    assert any("context_length" in reason for reason in validation.reasons)
    assert any("gpu_offload_layers" in reason for reason in validation.reasons)
