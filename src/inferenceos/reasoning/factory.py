"""Factory building the configured reasoning engine from settings."""

from __future__ import annotations

from pathlib import Path

from inferenceos.backends.resident_server import LlamaServerResidentBackend
from inferenceos.config import Settings
from inferenceos.interfaces.backend_adapter import RuntimeParams
from inferenceos.logging_config import get_logger
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.providers.llama_cpp import LlamaCppReasoningProvider
from inferenceos.reasoning.resident_reasoner import LlamaServerReasoner

logger = get_logger("reasoning.factory")


def build_reasoning_provider(settings: Settings) -> LlamaCppReasoningProvider:
    """Build the configured local reasoning provider (no model substitution)."""
    if settings.reasoning_provider != "llama_cpp":
        raise ValueError(
            f"unsupported reasoning provider: {settings.reasoning_provider!r} "
            "(only 'llama_cpp' is implemented)"
        )
    return LlamaCppReasoningProvider(
        model_path=settings.reasoning_model_path,
        cli_path=settings.reasoning_cli_path,
        llama_cpp_dir=settings.llama_cpp_dir,
        context_length=settings.reasoning_context_length,
        n_gpu_layers=settings.reasoning_gpu_layers,
    )


def build_reasoning_engine(settings: Settings) -> LocalReasoningEngine | None:
    """Build the local reasoning engine, or ``None`` when disabled/unsupported."""
    if not settings.reasoning_enabled:
        return None
    provider = build_reasoning_provider(settings)
    return LocalReasoningEngine(
        provider,
        max_tokens=settings.reasoning_max_tokens,
        temperature=settings.reasoning_temperature,
        seed=settings.reasoning_seed,
        timeout_s=settings.reasoning_timeout_s,
        meaningful_confidence_threshold=0.5,
    )


def build_resident_reasoner(
    settings: Settings, *, expected_bytes: int | None = None
) -> LlamaServerReasoner | None:
    """Build a resident reasoner over a real ``llama-server``, or ``None`` if absent.

    The current default model is preserved: no model substitution and no silent
    upgrade. If the configured reasoner is not present, the caller reports
    unavailability rather than substituting a different model.
    """
    model_path = settings.reasoning_model_path
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.is_file():
        return None
    from inferenceos.models.loader import load_model_profile

    model = load_model_profile(path)
    server = settings.llama_cpp_dir / "llama-server.exe"
    if not server.is_file():
        server = settings.llama_cpp_dir / "llama-server"
    backend = LlamaServerResidentBackend(
        server=server,
        host=settings.resident_host,
        port=settings.resident_port,
        readiness_timeout_s=settings.resident_readiness_timeout_s,
        request_timeout_s=settings.resident_request_timeout_s,
    )
    return LlamaServerReasoner(
        backend=backend,
        model=model,
        params=RuntimeParams(
            gpu_offload_layers=settings.reasoning_gpu_layers,
            context_length=settings.reasoning_context_length,
        ),
        max_tokens=settings.reasoning_max_tokens,
        temperature=settings.reasoning_temperature,
        seed=settings.reasoning_seed,
        timeout_s=settings.reasoning_timeout_s,
        expected_bytes=expected_bytes if expected_bytes is not None else settings.phase5r_reasoner_expected_bytes,
    )

