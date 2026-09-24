"""Adaptive quality tolerance derived from the diagnosed/optimization context.

The tolerance is no longer a single universal string-similarity threshold: a
runtime-only candidate on a memory-bound workload is treated differently from a
semantics-changing candidate under a high-severity diagnosis. The derivation is
explicit, deterministic, versioned, and every contributing reason is recorded.
"""

from __future__ import annotations

from inferenceos.config import Settings
from inferenceos.contracts.enums import EvidenceStrength, WorkloadKind
from inferenceos.contracts.quality import AdaptiveQualityPolicy

# Actions that can plausibly change the *generated content* (not just resource use).
_SEMANTIC_ACTIONS = frozenset(
    {
        "runtime.right_size_context",
        "runtime.kv_cache_type",
        "runtime.flash_attention",
        "model.requantize",
    }
)
_MODEL_MUTATING_ACTIONS = frozenset({"model.requantize"})

_STRENGTH_SEVERITY: dict[EvidenceStrength, float] = {
    EvidenceStrength.NONE: 0.0,
    EvidenceStrength.LOW: 0.25,
    EvidenceStrength.MEDIUM: 0.6,
    EvidenceStrength.HIGH: 0.9,
}


def build_adaptive_quality_policy(
    settings: Settings,
    *,
    action_id: str | None = None,
    bottleneck_kind: str | None = None,
    bottleneck_confidence: float = 0.0,
    evidence_strength: EvidenceStrength = EvidenceStrength.NONE,
    optimization_magnitude: float = 0.0,
    workload_kind: WorkloadKind = WorkloadKind.UNKNOWN,
    baseline_cv: float | None = None,
) -> AdaptiveQualityPolicy:
    """Derive a context-aware tolerance band with a transparent basis."""
    base = settings.phase7r_base_similarity
    effective = base
    basis = [f"base similarity floor={base:.2f}"]

    affects_semantics = action_id in _SEMANTIC_ACTIONS
    changes_model = action_id in _MODEL_MUTATING_ACTIONS

    severity = max(
        float(bottleneck_confidence or 0.0),
        _STRENGTH_SEVERITY.get(evidence_strength, 0.0),
    )

    if action_id is not None and not affects_semantics:
        effective -= settings.phase7r_runtime_only_relaxation
        basis.append(
            "runtime-only candidate (no generation-semantics change): tolerance "
            f"relaxed by {settings.phase7r_runtime_only_relaxation:.2f}"
        )
    if affects_semantics:
        effective += settings.phase7r_semantic_tightening
        basis.append(
            "candidate may change generation semantics: tolerance tightened by "
            f"{settings.phase7r_semantic_tightening:.2f}"
        )
    if changes_model:
        effective += settings.phase7r_semantic_tightening
        basis.append("candidate mutates model behaviour: tolerance tightened further")
    if severity >= 0.6:
        effective += settings.phase7r_semantic_tightening
        basis.append(f"high severity ({severity:.2f}): tolerance tightened")
    if baseline_cv is not None and baseline_cv > 0.15:
        effective -= settings.phase7r_semantic_tightening
        basis.append(
            f"baseline variability high (CV {baseline_cv:.3f}): tolerance relaxed"
        )

    effective = max(
        settings.phase7r_min_similarity,
        min(settings.phase7r_max_similarity, effective),
    )
    basis.append(f"effective similarity tolerance={effective:.3f}")

    content_min = settings.phase7r_content_overlap_min
    if affects_semantics:
        content_min = min(0.9, content_min + 0.10)
        basis.append("task-relevant content-coverage floor raised for semantic candidate")

    return AdaptiveQualityPolicy(
        base_similarity=base,
        effective_similarity=round(effective, 4),
        content_overlap_min=round(content_min, 4),
        length_ratio_min=settings.phase7r_length_ratio_min,
        length_ratio_max=settings.phase7r_length_ratio_max,
        affects_generation_semantics=affects_semantics,
        changes_model_behavior=changes_model,
        bottleneck_kind=bottleneck_kind,
        bottleneck_severity=round(severity, 4),
        optimization_magnitude=round(float(optimization_magnitude or 0.0), 4),
        workload_kind=workload_kind.value,
        baseline_cv=baseline_cv,
        tolerance_basis=basis,
    )


def optimization_magnitude(baseline: object, experiment: object) -> float:
    """Relative magnitude of the primary runtime change (0..1), context-length based."""
    base_ctx = getattr(baseline, "context_length", None)
    cand_ctx = getattr(experiment, "context_length", None)
    if not base_ctx or not cand_ctx or base_ctx <= 0:
        return 0.0
    return max(0.0, min(1.0, abs(float(base_ctx) - float(cand_ctx)) / float(base_ctx)))
