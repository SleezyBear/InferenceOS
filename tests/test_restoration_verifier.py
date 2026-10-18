"""Real restoration verification over a fake backend (deterministic)."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.backends.observed import observed_configuration
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
        # Explicit synthetic engine evidence, independent overrides use their
        # own telemetry rather than treating request equality as verification.
        return BackendTelemetry(observed={"context_length": self._params.context_length})

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


@pytest.mark.parametrize("threads,layers,problems", [
    (0, None, ()), (None, -1, ()), (0, -1, ()), (6, 8, ()),
    (4, 0, ("gpu_offload_layers", "threads")), (0, 0, ("gpu_offload_layers",)),
    (4, -1, ("threads",)),
])
def test_automatic_requests_are_not_literal_settings_but_explicit_requests_still_are(
    threads: int | None, layers: int | None, problems: tuple[str, ...],
) -> None:
    requested = RuntimeConfiguration(threads=threads, gpu_offload_layers=layers)
    telemetry = BackendTelemetry(effective_threads=6, gpu_layers_offloaded=8, gpu_layers_total=29)
    seen, contradictions, unknown = observed_configuration(requested, telemetry)
    assert tuple(item.split(":", 1)[0] for item in contradictions) == problems
    assert not unknown
    assert seen.threads == 6 and seen.gpu_offload_layers == 8


@pytest.mark.parametrize("threads,layers", [(0, None), (None, -1), (0, -1)])
def test_supported_automatic_requests_do_not_make_a_healthy_restoration_fail(
    threads: int | None, layers: int | None,
) -> None:
    class Evidence(_FakeAdapter):
        def telemetry(self) -> BackendTelemetry:
            return BackendTelemetry(effective_threads=6, gpu_layers_offloaded=8,
                gpu_layers_total=29, observed={"context_length": 2048})

    adapter = Evidence()
    config = RuntimeConfiguration(threads=threads, gpu_offload_layers=layers, context_length=2048)
    verifier = _verifier(adapter)
    verifier.capture(config)
    result = verifier.verify(config)
    assert result.verified and result.configuration_match and result.probe_match
    assert not adapter.is_loaded


@pytest.mark.parametrize("setting,changed", [
    ("effective_threads", 8), ("effective_threads", None), ("mmap_effective", False),
    ("flash_attention", True), ("cache_type_k", "q8_0"), ("cache_type_v", None),
    ("context_length", 4096),
])
def test_restoration_requires_the_engine_settings_observed_in_the_baseline(
    setting: str, changed: object,
) -> None:
    class Evidence(_FakeAdapter):
        shown = BackendTelemetry(effective_threads=6, mmap_effective=True,
            observed={"context_length": 2048, "flash_attention": False,
                      "cache_type_k": "f16", "cache_type_v": "f16"})

        def telemetry(self) -> BackendTelemetry:
            return self.shown

    adapter = Evidence()
    config = RuntimeConfiguration(context_length=1024)
    verifier = _verifier(adapter)
    verifier.capture(config)
    if setting in ("effective_threads", "mmap_effective"):
        adapter.shown = adapter.shown.model_copy(update={setting: changed})
    else:
        adapter.shown = adapter.shown.model_copy(update={
            "observed": {**adapter.shown.observed, setting: changed}})
    result = verifier.verify(config)
    assert result.probe_match
    assert not result.configuration_match and not result.verified
    assert any("baseline" in note for note in result.notes)
    assert not adapter.is_loaded


@pytest.mark.parametrize("before", [None, 8])
def test_later_valid_observations_cannot_repair_an_unverified_baseline_capture(before: int | None) -> None:
    class Evidence(_FakeAdapter):
        effective: int | None = before

        def telemetry(self) -> BackendTelemetry:
            return BackendTelemetry(effective_threads=self.effective, observed={"context_length": 2048})

    adapter = Evidence()
    config = RuntimeConfiguration(context_length=2048, threads=6)
    verifier = _verifier(adapter)
    verifier.capture(config)
    adapter.effective = 6
    result = verifier.verify(config)
    assert result.probe_match and not result.configuration_match and not result.verified
    assert any("baseline" in note for note in result.notes)
    assert not adapter.is_loaded
