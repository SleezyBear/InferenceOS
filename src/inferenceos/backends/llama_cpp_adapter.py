"""Concrete llama.cpp :class:`BackendAdapter`.

This is the only place the optimizer-visible adapter knows how to map the generic
:class:`RuntimeParams` onto real llama.cpp command-line flags. The rest of the
system never sees llama.cpp details.

Execution model: the acquired llama.cpp build is a one-shot CLI, so a *generation*
runs a real ``llama-cli`` process. ``load`` therefore validates and pins the model +
runtime configuration (including a real device capability probe), and ``unload``
releases it; residency is per invocation rather than a long-lived process. This is
documented rather than hidden — a resident-server adapter can implement the same
interface later without changing callers.
"""

from __future__ import annotations

from pathlib import Path

from inferenceos.backends.llama_cpp_toolchain import (
    parse_effective_settings,
    DeviceInfo,
    LlamaMemoryReport,
    LlamaRunResult,
    cli_version,
    find_cli,
    list_devices,
    parse_device_info,
    parse_generated_text,
    parse_load_mode,
    parse_timings,
    run_cli,
)
from inferenceos.config import Settings
from inferenceos.contracts.enums import BackendType
from inferenceos.contracts.experiment import effective_kv_types
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendCapabilities,
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.logging_config import get_logger

logger = get_logger("backends.llama_cpp_adapter")


class LlamaCppAdapter(BackendAdapter):
    """Real llama.cpp adapter; preserves the Phase 0 Vulkan execution path."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        cli: Path | None = None,
        timeout_s: float = 900.0,
    ) -> None:
        self._settings = settings or Settings()
        self._cli = cli or find_cli(self._settings.llama_cpp_dir)
        self._timeout_s = timeout_s
        self._model: ModelProfile | None = None
        self._params = RuntimeParams()
        self._loaded = False
        self._last_run: LlamaRunResult | None = None
        self._last_device: DeviceInfo | None = None

    # ------------------------------------------------------------------ #
    # Introspection helpers (beyond the abstract interface, for evidence)  #
    # ------------------------------------------------------------------ #
    @property
    def cli_path(self) -> Path:
        return self._cli

    @property
    def params(self) -> RuntimeParams:
        return self._params

    @property
    def last_run(self) -> LlamaRunResult | None:
        return self._last_run

    @property
    def last_device(self) -> DeviceInfo | None:
        return self._last_device

    @property
    def last_memory(self) -> LlamaMemoryReport | None:
        return self._last_run.memory if self._last_run is not None else None

    # ------------------------------------------------------------------ #
    # BackendAdapter                                                       #
    # ------------------------------------------------------------------ #
    def capabilities(self) -> BackendCapabilities:
        devices = list_devices(self._cli)
        backends: list[BackendType] = [BackendType.CPU]
        if any("vulkan" in device.lower() for device in devices):
            backends.insert(0, BackendType.VULKAN)
        return BackendCapabilities(
            name="llama.cpp",
            version=cli_version(self._cli),
            backends=backends,
            devices=devices or ["CPU"],
            supports_runtime_params=True,
        )

    def load(self, model: ModelProfile, params: RuntimeParams) -> None:
        path = Path(model.path)
        if not path.is_file():
            raise FileNotFoundError(f"model file not found: {path}")
        self._model = model
        self._params = params
        self._loaded = True
        logger.info("Loaded model configuration: %s (%s)", model.name, path)

    def unload(self) -> None:
        self._model = None
        self._params = RuntimeParams()
        self._loaded = False
        logger.info("Unloaded model configuration")

    def set_runtime_params(self, params: RuntimeParams) -> None:
        # Only fields explicitly provided change: an explicit ``None`` clears the
        # value, while omitted fields keep their current value.
        if not params.model_fields_set:
            return
        updates = {name: getattr(params, name) for name in params.model_fields_set}
        self._params = self._params.model_copy(update=updates)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self._loaded or self._model is None:
            raise RuntimeError("no model loaded; call load() first")
        params = self._params
        n_gpu_layers = params.gpu_offload_layers if params.gpu_offload_layers is not None else 99
        load_mode: str | None = None
        if params.use_mmap is not None:
            load_mode = "mmap" if params.use_mmap else "none"
        result = run_cli(
            self._cli,
            Path(self._model.path),
            request.prompt,
            n_predict=request.n_predict,
            seed=request.seed,
            temperature=request.temperature,
            n_gpu_layers=n_gpu_layers,
            threads=params.threads or 0,
            context_length=params.context_length,
            batch_size=params.batch_size,
            ubatch_size=params.ubatch_size,
            flash_attention=params.flash_attention,
            device=params.device,
            load_mode=load_mode,
            cache_type_k=effective_kv_types(
                params.kv_cache_type, params.kv_cache_type_k, params.kv_cache_type_v
            )[0],
            cache_type_v=effective_kv_types(
                params.kv_cache_type, params.kv_cache_type_k, params.kv_cache_type_v
            )[1],
            tensor_overrides=params.placement_overrides,
            verbose=True,
            log_verbosity=params.log_verbosity,
            ignore_eos=params.ignore_eos,
            timeout_s=self._timeout_s,
            track_rss=True,
        )
        self._last_run = result
        self._last_device = parse_device_info(result.stderr)
        text = parse_generated_text(result.stdout)
        prompt_tps, decode_tps = parse_timings(result.stdout)
        timings = result.timings
        if timings is not None:
            prompt_tps = timings.prefill_tokens_per_sec or prompt_tps
            decode_tps = timings.decode_tokens_per_sec or decode_tps
        return GenerationResult(
            text=text or "",
            prompt_tokens=timings.prompt_tokens if timings is not None else None,
            generated_tokens=timings.generated_tokens if timings is not None else None,
            prompt_tokens_per_sec=prompt_tps,
            decode_tokens_per_sec=decode_tps,
            model_load_s=timings.model_load_s if timings is not None else None,
            prefill_s=timings.prompt_eval_s if timings is not None else None,
            decode_s=timings.eval_s if timings is not None else None,
            ttft_s=timings.ttft_s if timings is not None else None,
            total_s=timings.total_s if timings is not None else None,
            process_wall_s=result.duration_s,
            peak_rss_bytes=result.peak_rss_bytes,
            final_rss_bytes=result.final_rss_bytes,
            peak_process_cpu_percent=result.peak_process_cpu_percent,
        )

    def telemetry(self) -> BackendTelemetry:
        device = self._last_device
        memory = self.last_memory
        return BackendTelemetry(
            device=device.device if device is not None else None,
            gpu_layers_offloaded=device.layers_offloaded if device is not None else None,
            gpu_layers_total=device.layers_total if device is not None else None,
            kv_cache_bytes=memory.kv_cache_bytes if memory is not None else None,
            model_buffer_bytes=memory.model_buffer_bytes if memory is not None else None,
            context_length=self._params.context_length,
            # Effective load mode from the most recent real run's log, if it carried it.
            mmap_effective=(
                parse_load_mode(self._last_run.stderr)
                if self._last_run is not None
                else None
            ),
            observed=(
                parse_effective_settings(self._last_run.stderr)
                if self._last_run is not None
                else {}
            ),
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded
