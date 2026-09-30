"""Resident adapter behaviour over a deterministic fake backend."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.backends.resident_server import ResidentBackendError
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    GenerationRequest,
    RuntimeParams,
)
from tests.resident_helpers import FakeResidentBackend, synthetic_model


def _adapter(tmp_path: Path) -> tuple[LlamaCppResidentAdapter, FakeResidentBackend]:
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"\x00")
    backend = FakeResidentBackend()
    adapter = LlamaCppResidentAdapter(server=Path("unused"), backend=backend)
    return adapter, backend


def _model(tmp_path: Path) -> ModelProfile:
    return synthetic_model(str(tmp_path / "model.gguf"))


def test_resident_adapter_implements_interface(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    assert isinstance(adapter, BackendAdapter)
    assert adapter.is_loaded is False


def test_load_and_unload_are_real_calls(tmp_path: Path) -> None:
    adapter, backend = _adapter(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(context_length=1024))
    assert adapter.is_loaded is True
    assert adapter.last_load is not None and adapter.last_load.resident_verified
    assert adapter.pid is not None
    adapter.unload()
    assert adapter.is_loaded is False
    assert backend.last_unload is not None and backend.last_unload.cleanup_complete


def test_generate_delegates_and_records_ttft(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(context_length=1024))
    result = adapter.generate(GenerationRequest(prompt="hi", n_predict=4))
    assert result.text == "ok"
    assert result.ttft_s == pytest.approx(0.02)
    assert adapter.last_generation is not None
    assert adapter.last_generation.ttft_s == pytest.approx(0.02)


def test_set_runtime_params_restores_none(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(threads=4, context_length=1024))
    adapter.set_runtime_params(RuntimeParams(threads=None))
    assert adapter.params.threads is None
    assert adapter.params.context_length == 1024
    application = adapter.last_param_application
    assert application is not None
    assert application.changed is True
    assert application.requires_reload is True
    assert application.reloaded is True
    assert application.verified is True


def test_set_runtime_params_omitted_field_is_unchanged(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(threads=4, context_length=1024))
    adapter.set_runtime_params(RuntimeParams(context_length=2048))
    assert adapter.params.threads == 4
    assert adapter.params.context_length == 2048


def test_set_runtime_params_reload_failure_preserves_prior(tmp_path: Path) -> None:
    adapter, backend = _adapter(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(threads=4, context_length=1024))
    backend.fail_start_times = 1
    with pytest.raises(ResidentBackendError):
        adapter.set_runtime_params(RuntimeParams(context_length=4096))
    assert backend.is_running()
    assert adapter.params.context_length == 1024
    application = adapter.last_param_application
    assert application is not None
    assert application.reloaded is False
    assert application.error is not None


def test_set_runtime_params_no_fields_is_noop(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(context_length=1024))
    adapter.set_runtime_params(RuntimeParams())
    assert adapter.params.context_length == 1024
    assert adapter.last_param_application is not None
    assert adapter.last_param_application.changed is False


def test_generate_without_load_raises(tmp_path: Path) -> None:
    adapter, _ = _adapter(tmp_path)
    with pytest.raises(RuntimeError):
        adapter.generate(GenerationRequest(prompt="hi"))


# --------------------------------------------------------------------------- #
# Step 2 bug: a failed reload must report honestly whether the old settings  #
# came back, and must never claim the new settings are running.              #
# --------------------------------------------------------------------------- #
def test_failed_reload_with_successful_restore_reports_restored(tmp_path: Path) -> None:
    adapter, backend = _adapter(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(threads=4, context_length=1024))
    backend.fail_start_times = 1  # the new settings fail; putting the old ones back works
    with pytest.raises(ResidentBackendError):
        adapter.set_runtime_params(RuntimeParams(threads=2))
    assert adapter.is_loaded is True
    assert adapter.params.threads == 4
    application = adapter.last_param_application
    assert application is not None
    assert application.verified is False
    assert "prior configuration restored" in application.reason


def test_failed_reload_with_failed_restore_is_not_labelled_restored(tmp_path: Path) -> None:
    adapter, backend = _adapter(tmp_path)
    adapter.load(_model(tmp_path), RuntimeParams(threads=4, context_length=1024))
    backend.fail_start_times = 2  # the new settings fail AND the old ones fail too
    with pytest.raises(ResidentBackendError) as excinfo:
        adapter.set_runtime_params(RuntimeParams(threads=2))
    assert adapter.is_loaded is False
    # The failed new settings must not be reported as the current configuration.
    assert adapter.params.threads != 2
    application = adapter.last_param_application
    assert application is not None
    assert application.verified is False
    assert "restored" not in application.reason.replace("NOT restored", "")
    assert "NOT restored" in application.reason
    assert "NOT restored" in str(excinfo.value)
