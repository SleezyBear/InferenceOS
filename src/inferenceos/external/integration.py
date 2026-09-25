"""Runtime integration: an ``ExternalAdvisor`` implemented by the Phase 11R engine.

The Phase 10 runtime monitor consults this advisor when a persistent condition is
diagnosed. The advisor performs read-only external observation, attribution, and
recommendation, and records the user decision. It never executes external actions and
never duplicates runtime monitoring.
"""

from __future__ import annotations

from inferenceos.contracts.external import (
    ExternalProcessIdentity,
    ExternalRecommendationRun,
)
from inferenceos.external.engine import ExternalRecommendationEngine
from inferenceos.external.recommendation import bottleneck_to_resource


class RuntimeExternalAdvisor:
    """Adapter from the external engine to the Phase 10 runtime hook."""

    def __init__(
        self,
        engine: ExternalRecommendationEngine,
        *,
        max_cycles: int = 1,
    ) -> None:
        self._engine = engine
        self._max_cycles = max_cycles

    def advise(
        self,
        *,
        target_pid: int | None,
        target_identity: ExternalProcessIdentity | None,
        target_degraded: bool,
        target_metrics: dict[str, float],
        context_identity: str,
        diagnosed_bottleneck: str | None = None,
        approval_available: bool = True,
    ) -> ExternalRecommendationRun:
        _ = approval_available
        return self._engine.run(
            target_pid=target_pid,
            target_identity=target_identity,
            target_degraded=target_degraded,
            target_metrics=target_metrics,
            context_identity=context_identity,
            diagnosed_bottleneck=diagnosed_bottleneck,
            diagnosed_resource=bottleneck_to_resource(diagnosed_bottleneck),
            max_cycles=self._max_cycles,
        )
