"""InferenceOS — autonomous adaptive local LLM optimization system.

Phase 0 defines the vocabulary (contracts) and interfaces every later phase builds
against, plus the environment baseline (llama.cpp + Vulkan, model, logging, config,
audit schema).
"""

from __future__ import annotations

from inferenceos.version import __version__

__all__ = ["__version__"]
