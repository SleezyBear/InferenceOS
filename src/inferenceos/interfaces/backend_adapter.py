"""The abstract BackendAdapter interface.

llama.cpp is the only implementation in v1, but the optimizer must never call the
backend directly — it always goes through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import BackendType
from inferenceos.contracts.model import ModelProfile


class RuntimeParams(ContractBase):
    """Runtime knobs exposed by the backend (v1 candidate surfaces)."""

    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    gpu_offload_layers: int | None = None
    use_mmap: bool | None = None
    context_length: int | None = None
    flash_attention: bool | None = None


class BackendCapabilities(ContractBase):
    name: str
    backends: list[BackendType] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    supports_runtime_params: bool = True


class GenerationRequest(ContractBase):
    prompt: str
    n_predict: int = 128
    seed: int = -1
    temperature: float = 0.0
    stop: list[str] = Field(default_factory=list)


class GenerationResult(ContractBase):
    text: str
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    decode_tokens_per_sec: float | None = None
    prompt_tokens_per_sec: float | None = None


class BackendTelemetry(ContractBase):
    device: str | None = None
    gpu_layers_offloaded: int | None = None
    gpu_layers_total: int | None = None
    vram_used_bytes: int | None = None


class BackendAdapter(ABC):
    """Abstract backend interface used by every optimizer component."""

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Report available backends/devices without loading a model."""

    @abstractmethod
    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        """Load a model with the given runtime parameters."""

    @abstractmethod
    def unload(self) -> None:
        """Unload the currently loaded model."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run a real generation against the loaded model."""

    @abstractmethod
    def set_runtime_params(self, params: RuntimeParams) -> None:
        """Apply runtime parameters to the loaded model."""

    @abstractmethod
    def telemetry(self) -> BackendTelemetry:
        """Read back real backend/device telemetry."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
