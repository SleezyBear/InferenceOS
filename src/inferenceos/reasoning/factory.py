"""Factory building the configured reasoning engine from settings."""

from __future__ import annotations

from inferenceos.config import Settings
from inferenceos.logging_config import get_logger
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.providers.llama_cpp import LlamaCppReasoningProvider

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
