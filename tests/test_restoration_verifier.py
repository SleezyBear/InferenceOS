"""Real restoration verification over a fake backend (deterministic)."""

from __future__ import annotations

from pathlib import Path

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.experiment.restoration import ResidentRestorationVerifier
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendCapabilities,
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from tests.resident_helpers import synthetic_model


class _FakeAdapter(BackendAdapter):
    def __init__(self) -> None:
        self._loaded = False
        self._params = RuntimeParams()
        self.text = "OK"
        self.fail = False

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name="fake")

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        self._loaded = True
        self._params = params

    def unload(self) -> None:
        self._loaded = False

    def set_runtime_params(self, params: RuntimeParams) -> None:
        updates = {name: getattr(params, name) for name in params.model_fields_set}
        self._params = self._params.model_copy(update=updates)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.fail:
            raise RuntimeError("injected failure")
        return GenerationResult(text=self.text)

    def telemetry(self) -> BackendTelemetry:
        return BackendTelemetry()

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def _verifier(adapter: _FakeAdapter) -> ResidentRestorationVerifier:
    return ResidentRestorationVerifier(adapter=adapter, model=synthetic_model())


def test_verifies_when_probe_reproduced() -> None:
    adapter = _FakeAdapter()
    baseline = RuntimeConfiguration(context_length=2048)
    verifier = _verifier(adapter)
    verifier.capture(baseline)
    verification = verifier.verify(baseline)
    assert verification.verified is True
    assert verification.probe_match is True
    assert verification.configuration_match is True
    assert verification.restored_probe_hash == verification.baseline_probe_hash


def test_detects_divergent_probe() -> None:
    adapter = _FakeAdapter()
    baseline = RuntimeConfiguration(context_length=2048)
    verifier = _verifier(adapter)
    verifier.capture(baseline)
    adapter.text = "REGRESSED OUTPUT"
    verification = verifier.verify(baseline)
    assert verification.probe_match is False
    assert verification.verified is False


def test_detects_generation_failure() -> None:
    adapter = _FakeAdapter()
    baseline = RuntimeConfiguration(context_length=2048)
    verifier = _verifier(adapter)
    verifier.capture(baseline)
    adapter.fail = True
    verification = verifier.verify(baseline)
    assert verification.verified is False
    assert "failed" in verification.reason


def test_detects_configuration_mismatch() -> None:
    adapter = _FakeAdapter()
    baseline = RuntimeConfiguration(context_length=2048)
    verifier = _verifier(adapter)
    verifier.capture(baseline)
    verification = verifier.verify(RuntimeConfiguration(context_length=512))
    assert verification.configuration_match is False
    assert verification.verified is False
