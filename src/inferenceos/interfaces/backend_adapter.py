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
    # Threads used to read the prompt (llama.cpp ``-tb``); None = same as ``threads``.
    threads_batch: int | None = None
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
    # Engine switches: environment variables set for the engine process only,
    # canonical "NAME=VALUE;NAME=VALUE" (``tune/engine_switches.py``).
    engine_switches: str | None = None
    # Explicit placement of model pieces ("PATTERN=BUFFER;..."), see
    # ``RuntimeConfiguration.placement_overrides``.
    placement_overrides: str | None = None
    # Per-side cache formats; a per-side value wins over ``kv_cache_type``.
    kv_cache_type_k: str | None = None
    kv_cache_type_v: str | None = None
    # Prompt reuse (16 Oct 2026, development evaluation only): the engine's minimum chunk
    # size for reusing cached work after a mismatch by shifting it (llama.cpp
    # ``--cache-reuse``). None = the engine default (off). Not a tune setting: it has no
    # RuntimeConfiguration field, so tune and saved setups never set it.
    cache_reuse: int | None = None


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
    # Keep writing to ``n_predict`` even past the end-of-answer token (per request), so
    # a fixed-length workload measures the same amount of writing on every model.
    ignore_eos: bool = False
    # Ask the engine for the probabilities of the chosen word and its closest
    # alternatives (0 = none). Used only by correctness checks of engine switches.
    n_probs: int = 0
    # Let the engine reuse already-processed work for the start of this prompt (llama.cpp
    # ``cache_prompt``). False by default: every tune measurement is a fresh reading of
    # the whole prompt. True only in the prompt-reuse development evaluation.
    cache_prompt: bool = False


class TokenChoice(ContractBase):
    """One written word with the engine's probability for it and for the runner-up."""

    token: str
    prob: float
    runner_up_prob: float | None = None


class GenerationResult(ContractBase):
    text: str
    # Per written word, when the request asked for probabilities (``n_probs``).
    token_choices: list[TokenChoice] | None = None
    #: Prompt pieces the engine actually processed for this answer (llama.cpp
    #: ``timings.prompt_n``). With reuse this is fewer than the whole prompt.
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    decode_tokens_per_sec: float | None = None
    prompt_tokens_per_sec: float | None = None
    #: Prompt reuse evidence, as the engine reported it (None: not reported): the pieces
    #: taken from earlier work instead of being processed (``timings.cache_n``), and the
    #: whole prompt's length (``tokens_evaluated``). ``cache_prompt`` is what was asked.
    prompt_tokens_cached: int | None = None
    prompt_tokens_total: int | None = None
    cache_prompt: bool | None = None
    # Phase timings, in seconds. ``ttft_s`` is model-ready -> first generated token.
    model_load_s: float | None = None
    prefill_s: float | None = None
    decode_s: float | None = None
    ttft_s: float | None = None
    total_s: float | None = None
    process_wall_s: float | None = None
    # Real measured peak RSS of the backend process during this generation interval
    # (sampled + OS peak working set where available), distinct from the final RSS.
    peak_rss_bytes: int | None = None
    final_rss_bytes: int | None = None
    peak_process_cpu_percent: float | None = None
    # Bounded CPU efficiency/contention telemetry over the generation interval (unknown
    # when unavailable; never fabricated): total process CPU-seconds consumed, and
    # process context switches (voluntary/involuntary).
    cpu_seconds: float | None = None
    context_switches_voluntary: int | None = None
    context_switches_involuntary: int | None = None
    context_switches_total: int | None = None


class BackendTelemetry(ContractBase):
    device: str | None = None
    gpu_layers_offloaded: int | None = None
    gpu_layers_total: int | None = None
    vram_used_bytes: int | None = None
    # Values reported by the backend for its most recent context allocation.
    kv_cache_bytes: int | None = None
    model_buffer_bytes: int | None = None
    context_length: int | None = None
    # The *effective* generation thread count the backend actually initialized (observed
    # from backend evidence, never inferred from CPU topology). ``None`` means unknown.
    # This is distinct from the *requested* ``RuntimeParams.threads``.
    effective_threads: int | None = None
    # The *effective* weight-load mode the backend actually used, observed from backend
    # evidence (the startup log's ``load_mode`` line). ``True`` = mmap, ``False`` =
    # non-mmap, ``None`` = unknown/unavailable. This is deliberately distinct from the
    # *configured* ``RuntimeParams.use_mmap`` and is never inferred from configuration,
    # launch arguments, or OS memory statistics; missing evidence stays unknown.
    mmap_effective: bool | None = None
    # What the engine's own output showed about how it was set up (backend-defined
    # keys, e.g. "cache_type_k", "model_buffers"); a missing key means not shown.
    observed: dict[str, object] = Field(default_factory=dict)


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
