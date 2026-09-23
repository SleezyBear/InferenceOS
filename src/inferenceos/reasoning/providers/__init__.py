"""Reasoning providers (local runtimes behind the ReasoningEngine interface)."""

from __future__ import annotations

from inferenceos.reasoning.providers.base import ProviderResult, ReasoningProvider
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from inferenceos.reasoning.providers.llama_cpp import LlamaCppReasoningProvider

__all__ = [
    "FakeReasoningProvider",
    "LlamaCppReasoningProvider",
    "ProviderResult",
    "ReasoningProvider",
]
