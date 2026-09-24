"""The abstract BackendAdapter interface.

llama.cpp is the only implementation in v1, but the optimizer must never call the
backend directly — it always goes through this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import Field

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import BackendType
from inferenceos.contracts.model import ModelProfile

if TYPE_CHECKING:
    from inferenceos.backends.llama_cpp_toolchain import DeviceInfo


class RuntimeParams(ContractBase):
    """Runtime knobs exposed by the backend (v1 candidate surfaces)."""

    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    gpu_offload_layers: int | None = None
    use_mmap: bool | None = None
    context_length: int | None = None
    flash_attention: bool | None = None
    # Backend-defined device selector (e.g. a device name, or "none" for CPU-only);
    # ``None`` means "let the backend choose". Interpreted by the adapter.
    device: str | None = None
    # KV-cache element type exposed by the backend (e.g. "f16", "q8_0").
    kv_cache_type: str | None = None
    # Backend log verbosity (backend-defined scale). When set, the backend should
    # use this instead of its maximum-verbosity debug mode.
    log_verbosity: int | None = None
    # Sampling control: continue generating to the requested length instead of
    # stopping at the model's end-of-sequence token.
    ignore_eos: bool | None = None


class BackendCapabilities(ContractBase):
    name: str
    version: str | None = None
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
    # Phase timings, in seconds. ``ttft_s`` is model-ready -> first generated token.
    model_load_s: float | None = None
    prefill_s: float | None = None
    decode_s: float | None = None
    ttft_s: float | None = None
    total_s: float | None = None
    process_wall_s: float | None = None
    # Real measured peak RSS of the backend process during this generation.
    peak_rss_bytes: int | None = None
    peak_process_cpu_percent: float | None = None


class BackendTelemetry(ContractBase):
    device: str | None = None
    gpu_layers_offloaded: int | None = None
    gpu_layers_total: int | None = None
    vram_used_bytes: int | None = None
    # Values reported by the backend for its most recent context allocation.
    kv_cache_bytes: int | None = None
    model_buffer_bytes: int | None = None
    context_length: int | None = None


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
    def last_device(self) -> "DeviceInfo | None":
        """Device actually used by the most recent generation (optional)."""
        return None

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
