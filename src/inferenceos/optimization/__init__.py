"""Phase 6 candidate generation & controlled optimization engine.

Turns a validated Phase 4/5 diagnosis into a bounded, auditable set of possible
optimization interventions drawn only from a registered action vocabulary. It
inspects context and ranks candidates; it never executes an intervention.
"""

from __future__ import annotations

from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.policy import OptimizationPolicy, build_policy
from inferenceos.optimization.registry import (
    ActionRegistry,
    ActionRegistryError,
    UnknownActionError,
)

__all__ = [
    "ActionRegistry",
    "ActionRegistryError",
    "OptimizationPolicy",
    "UnknownActionError",
    "build_default_registry",
    "build_policy",
    "generate_candidates",
]
