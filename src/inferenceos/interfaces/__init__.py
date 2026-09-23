"""Abstract interfaces every later phase builds against.

Phase 0 defines interfaces only; real implementations land in later phases
(``BackendAdapter`` -> Phase 2, LLM ``ReasoningEngine`` -> Phase 5).
"""

from __future__ import annotations

from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    BackendCapabilities,
    BackendTelemetry,
    GenerationRequest,
    GenerationResult,
    RuntimeParams,
)
from inferenceos.interfaces.reasoning_engine import ReasoningEngine

__all__ = [
    "BackendAdapter",
    "BackendCapabilities",
    "BackendTelemetry",
    "GenerationRequest",
    "GenerationResult",
    "ReasoningEngine",
    "RuntimeParams",
]
