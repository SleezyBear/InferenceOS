"""Shared fixtures for Phase 6 optimization tests."""

from __future__ import annotations

from datetime import datetime, timezone

from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.diagnosis import BottleneckDiagnosis, DiagnosisContext
from inferenceos.contracts.enums import (
    ApprovalRequirement,
    BottleneckKind,
    CandidateEligibility,
    EvidencePolarity,
    EvidenceStrength,
    InferenceStage,
    OptimizationScope,
    PreconditionKind,
    Reversibility,
    RiskLevel,
)
from inferenceos.contracts.hardware import HardwareProfile
from inferenceos.contracts.optimization import (
    BenefitEstimate,
    CandidatePrecondition,
    CostEstimate,
    OptimizationCandidate,
    RiskAssessment,
)
from inferenceos.contracts.profiling import BaselineReport, ProfilingConfig
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.context import (
    OptimizationContext,
    build_optimization_context,
)
from inferenceos.optimization.policy import OptimizationPolicy
from inferenceos.selftest import sample_hardware
from tests.reasoning_helpers import (  # noqa: F401 - re-exported fixtures
    healthy_bundle,
    insufficient_bundle,
    paging_bundle,
)

_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def synthetic_diagnosis(
    kind: BottleneckKind,
    *,
    meaningful: bool = True,
    confidence: float = 0.7,
    strength: EvidenceStrength = EvidenceStrength.MEDIUM,
    insufficient: bool = False,
    hardware: HardwareProfile | None = None,
    subject: str = "synthetic-model/short-short/warm",
) -> BottleneckDiagnosis:
    evidence = [
        EvidenceItem(
            metric="synthetic_evidence",
            observed_value=0.9,
            reference_value=0.5,
            ratio=1.8,
            description="synthetic supporting evidence",
            polarity=EvidencePolarity.SUPPORTING,
            stage=InferenceStage.END_TO_END,
            strength=strength,
            source="test",
        )
    ]
    primary = Bottleneck(
        bottleneck_id="bn-synth",
        kind=kind,
        confidence=confidence,
        is_meaningful=meaningful,
        summary=f"synthetic {kind.value}",
        evidence=evidence,
        source="test",
        stage=InferenceStage.END_TO_END,
        subject=subject,
        evidence_strength=strength,
        supporting_evidence=evidence,
        detector_version="test",
    )
    return BottleneckDiagnosis(
        diagnosis_id="diag-synth",
        created_at=_NOW,
        context=DiagnosisContext(
            source_report_path="<synthetic>",
            detector_version="test",
            hardware=hardware or sample_hardware(),
            backend=ProfilingConfig(backend_name="test", context_length=32768),
            model_refs=["synthetic-model"],
            workload_ids=["short-short"],
        ),
        meaningful_opportunity=meaningful,
        insufficient_evidence=insufficient,
        escalation_required=insufficient,
        confidence=confidence,
        evidence_strength=strength,
        primary=primary if meaningful else None,
        summary=f"synthetic {kind.value}",
        no_opportunity_reason=None if meaningful else "synthetic no-opportunity",
    )


def synthetic_context(
    kind: BottleneckKind, *, policy: OptimizationPolicy | None = None
) -> OptimizationContext:
    policy = policy or OptimizationPolicy()
    return build_optimization_context(
        synthetic_diagnosis(kind), report=None, outcome=None, policy=policy
    )


def make_candidate(
    ctx: OptimizationContext,
    *,
    action_id: str = "runtime.right_size_context",
    known: bool = True,
    magnitude: float = 200_000_000.0,
    cost: float = 3.0,
    overhead: float = 0.02,
    risk_level: RiskLevel = RiskLevel.LOW,
    reversibility: Reversibility = Reversibility.REVERSIBLE,
    scope: OptimizationScope = OptimizationScope.TARGET_MODEL_RUNTIME,
    approval: ApprovalRequirement = ApprovalRequirement.NOT_REQUIRED,
    eligibility: CandidateEligibility = CandidateEligibility.ELIGIBLE,
    confidence: float = 0.7,
    strength: EvidenceStrength = EvidenceStrength.MEDIUM,
    preconditions_satisfied: bool = True,
) -> OptimizationCandidate:
    action = build_default_registry().get(action_id)
    return OptimizationCandidate(
        candidate_id=f"cand-{action_id}",
        action_id=action.action_id,
        action_name=action.name,
        surface=action.surface,
        subsystem=action.subsystem,
        parameters={"context_length": 8192},
        source_diagnosis_id=ctx.diagnosis.diagnosis_id,
        source_bottleneck_id="bn-synth",
        diagnosis_kind=ctx.effective_kind or BottleneckKind.INSUFFICIENT_EVIDENCE,
        diagnosis_confidence=confidence,
        diagnosis_strength=strength,
        causal_rationale=action.causal_rationale,
        evidence_refs=list(ctx.evidence_ids),
        hardware=ctx.hardware,
        model_ref="synthetic-model",
        workload_id="short-short",
        observed_metrics=dict(ctx.observed_metrics),
        preconditions=[
            CandidatePrecondition(
                check_id="evidence_supported",
                kind=PreconditionKind.EVIDENCE,
                description="evidence supports",
                satisfied=preconditions_satisfied,
            )
        ],
        benefit=BenefitEstimate(
            metric="peak_rss_bytes",
            direction="decrease",
            unit="bytes",
            known=known,
            magnitude_low=magnitude if known else None,
            magnitude_high=magnitude if known else None,
            confidence=confidence,
            evidence_basis=list(ctx.evidence_ids),
        ),
        cost=CostEstimate(
            execution_cost_s=cost,
            optimizer_overhead_s=overhead,
            total_cost_s=cost + overhead,
        ),
        risk=RiskAssessment(
            risk_level=risk_level,
            reversibility=reversibility,
            scope=scope,
            privilege=action.privilege,
            requires_approval=approval is ApprovalRequirement.REQUIRED,
        ),
        approval=approval,
        eligibility=eligibility,
        executable=eligibility is CandidateEligibility.ELIGIBLE
        and approval is ApprovalRequirement.NOT_REQUIRED,
    )


def detector_diagnosis(report: BaselineReport) -> BottleneckDiagnosis:
    return BottleneckDetector().detect(report, source_report_path="<test>")
