"""Phase 11R recommendation context and staleness helpers.

These pure helpers keep the recommendation layer subordinate to the diagnosed
inference bottleneck and make staleness explicit. A recommendation is only current
while its provenance, context, expiry and external process identity all still hold; a
stale recommendation is never silently treated as current.
"""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.contracts.enums import (
    BottleneckKind,
    ExternalRecommendationStatus,
    ExternalResource,
)
from inferenceos.contracts.external import ExternalRecommendation
from inferenceos.external.identity import verify_identity

# Diagnosed bottlenecks that map onto an external-process resource domain. A
# bottleneck that an external process cannot plausibly relieve (e.g. GPU VRAM,
# latency, compute-bound) maps to ``None``: no external recommendation is produced.
_BOTTLENECK_RESOURCE: dict[str, ExternalResource] = {
    BottleneckKind.MEMORY_PRESSURE.value: ExternalResource.MEMORY,
    BottleneckKind.MODEL_FOOTPRINT.value: ExternalResource.MEMORY,
    BottleneckKind.PAGING.value: ExternalResource.MEMORY,
    BottleneckKind.CPU.value: ExternalResource.CPU,
    BottleneckKind.STORAGE.value: ExternalResource.STORAGE_IO,
    BottleneckKind.IO_BOUND.value: ExternalResource.STORAGE_IO,
    BottleneckKind.EXTERNAL_PROCESS.value: ExternalResource.SYSTEM,
}


def bottleneck_to_resource(bottleneck: str | BottleneckKind | None) -> ExternalResource | None:
    """Map a diagnosed bottleneck to the external resource it may relate to.

    Returns ``None`` when the bottleneck is absent, not meaningful, or cannot plausibly
    be influenced by an external process. This is what keeps recommendations
    diagnosis-aware rather than raw-usage based.
    """

    if bottleneck is None:
        return None
    value = bottleneck.value if isinstance(bottleneck, BottleneckKind) else str(bottleneck)
    return _BOTTLENECK_RESOURCE.get(value)


def recommendation_is_current(recommendation: ExternalRecommendation) -> bool:
    return recommendation.status in (
        ExternalRecommendationStatus.GENERATED,
        ExternalRecommendationStatus.PRESENTED,
        ExternalRecommendationStatus.ACCEPTED,
        ExternalRecommendationStatus.REJECTED,
    )


def revalidate_recommendation(
    recommendation: ExternalRecommendation,
    *,
    current_run_id: str | None = None,
    current_session_id: str | None = None,
    current_context_identity: str | None = None,
    revalidate_process: bool = True,
    now: datetime | None = None,
) -> ExternalRecommendation:
    """Return the recommendation with an updated status if its evidence is no longer valid.

    Staleness reasons (checked in order): changed/disappeared process identity,
    missing provenance, changed originating run/session, changed context, and expiry.
    A stale recommendation must never authorize (or be acted upon as) a current one.
    """

    now = now or datetime.now(timezone.utc)

    if recommendation.expires_at is not None and now > recommendation.expires_at:
        return recommendation.model_copy(
            update={
                "status": ExternalRecommendationStatus.EXPIRED,
                "stale_reason": (
                    f"recommendation expired at {recommendation.expires_at.isoformat()}"
                ),
            }
        )

    if revalidate_process and recommendation.identity is not None:
        if not verify_identity(recommendation.identity):
            return recommendation.model_copy(
                update={
                    "status": ExternalRecommendationStatus.STALE,
                    "stale_reason": (
                        "external process identity no longer matches the observed "
                        "process (terminated or PID reused)"
                    ),
                }
            )

    provenance = recommendation.provenance
    if provenance is None:
        return recommendation.model_copy(
            update={
                "status": ExternalRecommendationStatus.STALE,
                "stale_reason": "recommendation provenance is missing",
            }
        )

    if current_run_id and recommendation.run_id and recommendation.run_id != current_run_id:
        return recommendation.model_copy(
            update={
                "status": ExternalRecommendationStatus.STALE,
                "stale_reason": (
                    f"originating run {recommendation.run_id!r} is not the current run "
                    f"{current_run_id!r}"
                ),
            }
        )
    if (
        current_session_id
        and provenance.optimization_session_id
        and provenance.optimization_session_id != current_session_id
    ):
        return recommendation.model_copy(
            update={
                "status": ExternalRecommendationStatus.STALE,
                "stale_reason": (
                    f"originating session {provenance.optimization_session_id!r} is not "
                    f"the current session {current_session_id!r}"
                ),
            }
        )
    if (
        current_context_identity
        and recommendation.context_identity
        and current_context_identity != recommendation.context_identity
    ):
        return recommendation.model_copy(
            update={
                "status": ExternalRecommendationStatus.STALE,
                "stale_reason": (
                    f"context identity {recommendation.context_identity!r} does not match "
                    f"the current context {current_context_identity!r}"
                ),
            }
        )
    return recommendation
