"""The abstract ReasoningEngine interface.

Both the deterministic engine and the (later) local-LLM engine conform to this
interface from day one, so callers never depend on which implementation is active.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from inferenceos.contracts.enums import ReasoningEngineKind
from inferenceos.contracts.reasoning import ReasoningRequest, ReasoningResponse


class ReasoningEngine(ABC):
    """Abstract reasoning interface (deterministic or LLM-backed)."""

    @property
    @abstractmethod
    def kind(self) -> ReasoningEngineKind:
        """Which class of engine this is."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier (e.g. model name)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the engine can currently serve a request."""

    @abstractmethod
    def reason(self, request: ReasoningRequest) -> ReasoningResponse:
        """Produce a structured response for the request."""
