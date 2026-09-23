"""Reasoning provider abstraction.

A provider turns a (system, user, JSON-schema) request into raw completion text.
The core reasoning contracts and engine never depend on a specific runtime — the
initial implementation is local llama.cpp; Ollama or any other local runtime can be
added by implementing this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ProviderResult:
    """Raw outcome of one provider completion call."""

    success: bool
    text: str = ""
    error: str | None = None
    latency_s: float = 0.0
    model_id: str | None = None
    model_version: str | None = None
    output_tokens: int | None = None
    peak_rss_bytes: int | None = None
    command: list[str] = field(default_factory=list)


@runtime_checkable
class ReasoningProvider(Protocol):
    """Local completion provider interface."""

    @property
    def name(self) -> str:
        """Provider identity (e.g. ``llama.cpp``)."""

    @property
    def model_id(self) -> str:
        """Configured model identity (path or name)."""

    def is_available(self) -> bool:
        """Whether the provider and its configured model are usable right now."""

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, object],
        max_tokens: int,
        temperature: float,
        seed: int,
        timeout_s: float,
    ) -> ProviderResult:
        """Produce a constrained completion for the request."""
