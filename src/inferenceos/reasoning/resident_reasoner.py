"""A resident reasoner: a real local model loaded on demand behind a resident server.

The deep-diagnosis lifecycle needs the reasoner to be genuinely loaded and unloaded
(so the target can own the device/VRAM exclusively). This wraps a second
``llama-server`` (the reasoner model) and exposes a :class:`ReasoningEngine` that talks
to it, while remaining behind the existing reasoning interfaces.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from inferenceos.backends.resident_server import (
    LlamaServerResidentBackend,
    ResidentBackendError,
)
from inferenceos.backends.release import ReleaseEvidence, release_evidence
from inferenceos.contracts.lifecycle import (
    ResidentLoadMeasurement,
    ResidentUnloadMeasurement,
)
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import RuntimeParams
from inferenceos.interfaces.reasoning_engine import ReasoningEngine
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.providers.resident import ResidentReasoningProvider


@runtime_checkable
class ResidentReasoner(Protocol):
    """Structural interface the lifecycle uses to load/unload the reasoner."""

    @property
    def model_ref(self) -> str:
        ...

    @property
    def expected_bytes(self) -> int | None:
        ...

    def start(self) -> ResidentLoadMeasurement:
        ...

    def stop(self) -> ResidentUnloadMeasurement:
        ...

    def engine(self) -> ReasoningEngine:
        ...

    def is_running(self) -> bool:
        ...


class LlamaServerReasoner:
    """Concrete resident reasoner over a real ``llama-server``."""

    def __init__(
        self,
        *,
        backend: LlamaServerResidentBackend,
        model: ModelProfile,
        params: RuntimeParams,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int = 1234,
        timeout_s: float = 300.0,
        meaningful_confidence_threshold: float = 0.5,
        expected_bytes: int | None = None,
    ) -> None:
        self._backend = backend
        self._model = model
        self._params = params
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seed = seed
        self._timeout_s = timeout_s
        self._meaningful_threshold = meaningful_confidence_threshold
        self._expected_override = expected_bytes
        self._engine: LocalReasoningEngine | None = None

    @property
    def model_ref(self) -> str:
        return self._model.name

    @property
    def expected_bytes(self) -> int | None:
        if self._expected_override is not None:
            return self._expected_override
        weights = self._model.file_size_bytes or 0
        context = self._params.context_length or self._model.trained_context_length or 2048
        try:
            kv = self._model.estimate_kv_cache_bytes(int(context))
        except Exception:  # pragma: no cover - defensive
            kv = 0
        total = int(weights * 1.2) + int(kv)
        return total or None

    def start(self) -> ResidentLoadMeasurement:
        measurement = self._backend.start(self._model, self._params)
        client = self._backend.client
        if client is None:  # pragma: no cover - start() guarantees a client
            raise ResidentBackendError("reasoner backend started without a client")
        provider = ResidentReasoningProvider(
            client=client,
            model_id=self._model.name,
            request_timeout_s=self._timeout_s,
        )
        self._engine = LocalReasoningEngine(
            provider,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            seed=self._seed,
            timeout_s=self._timeout_s,
            meaningful_confidence_threshold=self._meaningful_threshold,
        )
        return measurement

    def stop(self) -> ResidentUnloadMeasurement:
        self._engine = None
        return self._backend.stop()

    def cleanup(self) -> None:
        self._engine = None
        self._backend.cleanup()

    def release_evidence(self) -> ReleaseEvidence:
        return release_evidence(self._backend)

    def engine(self) -> ReasoningEngine:
        if self._engine is None:
            raise ResidentBackendError("reasoner is not loaded")
        return self._engine

    def is_running(self) -> bool:
        return self._backend.is_running()
