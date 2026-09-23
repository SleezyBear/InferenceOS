"""Fake in-process reasoning provider for deterministic tests.

Never used by the final acceptance path; it exists so escalation, parsing,
validation and failure handling can be tested without a model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from inferenceos.reasoning.providers.base import ProviderResult


@dataclass
class FakeReasoningProvider:
    """Returns scripted text; can simulate unavailability/failure/timeout."""

    scripted: list[str] = field(default_factory=list)
    available: bool = True
    fail_with: str | None = None
    latency_s: float = 0.0
    model: str = "fake-reasoner"
    calls: int = 0

    name = "fake"

    @property
    def model_id(self) -> str:
        return self.model

    def is_available(self) -> bool:
        return self.available

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
        _ = (system_prompt, user_prompt, json_schema, max_tokens, temperature, seed, timeout_s)
        index = min(self.calls, len(self.scripted) - 1) if self.scripted else 0
        self.calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        if self.fail_with is not None:
            return ProviderResult(
                success=False, error=self.fail_with, model_id=self.model, latency_s=self.latency_s
            )
        text = self.scripted[index] if self.scripted else ""
        return ProviderResult(
            success=True, text=text, model_id=self.model, latency_s=self.latency_s
        )
