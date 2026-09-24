"""Deterministic safety orderings and constants.

Pure helpers used by the eligibility policy, ranking, and validation layers. The
policy is explicit and inspectable — safety-critical behaviour is never buried in a
magic constant at a call site.
"""

from __future__ import annotations

from inferenceos.contracts.enums import (
    EvidenceStrength,
    OptimizationScope,
    PrivilegeRequirement,
    Reversibility,
    RiskLevel,
)

RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.NONE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}

STRENGTH_ORDER: dict[EvidenceStrength, int] = {
    EvidenceStrength.NONE: 0,
    EvidenceStrength.LOW: 1,
    EvidenceStrength.MEDIUM: 2,
    EvidenceStrength.HIGH: 3,
}

# Narrower scopes score higher (a smaller blast radius is preferred).
SCOPE_ORDER: dict[OptimizationScope, int] = {
    OptimizationScope.TARGET_MODEL_RUNTIME: 6,
    OptimizationScope.TARGET_INFERENCE_PROCESS: 5,
    OptimizationScope.CURRENT_WORKLOAD: 4,
    OptimizationScope.USER_PROCESS: 3,
    OptimizationScope.EXTERNAL_PROCESS: 2,
    OptimizationScope.SYSTEM_WIDE: 1,
}

REVERSIBILITY_SCORE: dict[Reversibility, float] = {
    Reversibility.REVERSIBLE: 1.0,
    Reversibility.REVERSIBLE_WITH_RESTART: 0.6,
    Reversibility.NON_REVERSIBLE: 0.0,
}

# Scopes that can never be autonomously applied in Phase 6/7 (approval required).
BROAD_SCOPES = {OptimizationScope.EXTERNAL_PROCESS, OptimizationScope.SYSTEM_WIDE}


def risk_rank(level: RiskLevel) -> int:
    return RISK_ORDER.get(level, RISK_ORDER[RiskLevel.CRITICAL])


def strength_rank(strength: EvidenceStrength) -> int:
    return STRENGTH_ORDER.get(strength, 0)


def scope_score(scope: OptimizationScope) -> float:
    return SCOPE_ORDER.get(scope, 1) / len(SCOPE_ORDER)


def requires_approval(
    *,
    scope: OptimizationScope,
    privilege: PrivilegeRequirement,
    flag: bool,
) -> bool:
    """External/system-wide/elevated work is always approval-required."""
    return (
        flag
        or scope in BROAD_SCOPES
        or privilege is PrivilegeRequirement.ELEVATION_REQUIRED
    )
