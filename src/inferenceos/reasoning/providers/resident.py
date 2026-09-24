"""Resident reasoning provider.

Unlike :class:`~inferenceos.reasoning.providers.llama_cpp.LlamaCppReasoningProvider`
(which spawns ``llama-cli`` per call), this provider talks to an already-running
resident ``llama-server`` over its authenticated chat-completions API. The lifecycle
owns loading/unloading the reasoner server around the invocation.
"""

from __future__ import annotations

import time
from typing import Any

from inferenceos.backends.llama_server_toolchain import (
    LlamaServerClient,
    ServerRequestError,
)
from inferenceos.logging_config import get_logger
from inferenceos.reasoning.providers.base import ProviderResult

logger = get_logger("reasoning.providers.resident")


class ResidentReasoningProvider:
    """Provider backed by a live resident server; no per-call process spawning."""

    name = "llama.cpp-resident"

    def __init__(
        self,
        *,
        client: LlamaServerClient,
        model_id: str,
        request_timeout_s: float = 300.0,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._request_timeout_s = request_timeout_s

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def client(self) -> LlamaServerClient:
        return self._client

    def is_available(self) -> bool:
        return self._client.health()

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
        started = time.perf_counter()
        try:
            payload: dict[str, Any] = self._client.chat_complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                timeout_s=timeout_s or self._request_timeout_s,
            )
        except ServerRequestError as exc:
            return ProviderResult(
                success=False,
                error=f"resident reasoner request failed: {exc}",
                latency_s=time.perf_counter() - started,
                model_id=self._model_id,
            )
        latency = time.perf_counter() - started
        choices = payload.get("choices")
        text = ""
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    text = content
        usage = payload.get("usage")
        output_tokens: int | None = None
        if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
            output_tokens = usage["completion_tokens"]
        model_used = payload.get("model")
        if not text:
            return ProviderResult(
                success=False,
                error="resident reasoner returned no content",
                latency_s=latency,
                model_id=str(model_used) if model_used else self._model_id,
                output_tokens=output_tokens,
            )
        return ProviderResult(
            success=True,
            text=text.strip(),
            latency_s=latency,
            model_id=str(model_used) if model_used else self._model_id,
            output_tokens=output_tokens,
        )
