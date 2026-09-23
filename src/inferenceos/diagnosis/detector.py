"""Deterministic bottleneck detector.

Consumes a Phase 3 :class:`BaselineReport`, groups comparative observations,
runs the deterministic rules, and resolves them into a structured
:class:`BottleneckDiagnosis`.

The confidence model is *explainable and heuristic*, not a calibrated probability:
``confidence = clamp((support - 0.5*contradict) / 3, 0, 1) * quality_factor`` where
the quality factor is reduced by few repetitions, high variance, or unavailable
measurements. The category thresholds are documented in :mod:`diagnosis.rules`.

No optimization is performed or recommended.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from inferenceos import __version__
from inferenceos.contracts.bottleneck import Bottleneck, EvidenceItem
from inferenceos.contracts.diagnosis import BottleneckDiagnosis, DiagnosisContext
from inferenceos.contracts.enums import (
    BottleneckKind,
    EvidenceStrength,
    InferenceStage,
)
from inferenceos.contracts.profiling import BaselineReport
from inferenceos.diagnosis.groups import GroupObservation, extract_groups
from inferenceos.diagnosis.rules import CandidateFinding, RuleContext, run_rules

DETECTOR_VERSION = f"{__version__}+phase4"

SUPPORT_NORMALIZER = 3.0
EVIDENCE_STRENGTH_LOW = 0.20
EVIDENCE_STRENGTH_MEDIUM = 0.50
EVIDENCE_STRENGTH_HIGH = 0.75
MEANINGFUL_THRESHOLD = 0.50
ESCALATION_THRESHOLD = 0.25
AMBIGUITY_MARGIN = 0.15
HIGH_VARIANCE_CV = 0.25

_RESOURCE_FAMILY: dict[BottleneckKind, str] = {
    BottleneckKind.CPU: "cpu",
    BottleneckKind.COMPUTE_BOUND: "gpu",
    BottleneckKind.GPU_UNDERUTILIZATION: "gpu",
    BottleneckKind.GPU_MEMORY_LIMIT: "gpu",
    BottleneckKind.MEMORY_PRESSURE: "memory",
    BottleneckKind.MODEL_FOOTPRINT: "memory",
    BottleneckKind.PAGING: "memory",
    BottleneckKind.STORAGE: "storage",
    BottleneckKind.IO_BOUND: "storage",
    BottleneckKind.EXTERNAL_PROCESS: "external",
    BottleneckKind.LATENCY: "latency",
}


class BottleneckDetector:
    """Deterministic detector; contains no state beyond the detector version."""

    version = DETECTOR_VERSION

    def detect(
        self,
        report: BaselineReport,
        *,
        source_report_path: str = "<in-memory>",
    ) -> BottleneckDiagnosis:
        groups = extract_groups(report)
        physical_cores = max(report.hardware.cpu.physical_cores, 1)
        logical_cores = max(report.hardware.cpu.logical_cores, 1)
        context = RuleContext(
            groups=groups,
            physical_cores=physical_cores,
            logical_cores=logical_cores,
        )
        findings = run_rules(context)
        by_subject = {group.key: group for group in groups}

        scored = [_ScoredFinding(finding, by_subject.get(finding.subject)) for finding in findings]
        for item in scored:
            item.confidence = _confidence(item.finding, item.group)
            item.strength = _strength(item.confidence, item.finding)
            item.is_meaningful = (
                not item.finding.forced_non_meaningful
                and item.strength in (EvidenceStrength.MEDIUM, EvidenceStrength.HIGH)
            )

        meaningful = sorted(
            [item for item in scored if item.is_meaningful],
            key=lambda item: item.confidence,
            reverse=True,
        )
        observations = sorted(
            [item for item in scored if not item.is_meaningful],
            key=lambda item: item.confidence,
            reverse=True,
        )

        ambiguous = _is_ambiguous(meaningful)
        escalation_candidates = [
            item
            for item in scored
            if not item.is_meaningful and not item.finding.forced_non_meaningful
        ]
        best_escalation = max((item.confidence for item in escalation_candidates), default=0.0)
        measurement_ok = _measurement_sufficient(groups)

        if meaningful and not ambiguous:
            primary = meaningful[0]
            meaningful_opportunity = True
            insufficient_evidence = False
            escalation_required = False
            no_opportunity_reason = None
        elif ambiguous:
            primary = None
            meaningful_opportunity = False
            insufficient_evidence = True
            escalation_required = True
            no_opportunity_reason = (
                "multiple candidate resource limitations are comparably supported; "
                "deterministic evidence cannot isolate one root cause"
            )
        elif not measurement_ok:
            primary = None
            meaningful_opportunity = False
            insufficient_evidence = True
            escalation_required = True
            no_opportunity_reason = "measurement/telemetry coverage is insufficient to diagnose"
        elif best_escalation >= ESCALATION_THRESHOLD:
            primary = None
            meaningful_opportunity = False
            insufficient_evidence = True
            escalation_required = True
            no_opportunity_reason = (
                "a candidate signal exists but its evidence strength is below the "
                "meaningful-opportunity threshold"
            )
        else:
            primary = None
            meaningful_opportunity = False
            insufficient_evidence = False
            escalation_required = False
            no_opportunity_reason = (
                "no resource or execution signal is supported by meaningful, "
                "comparatively-corroborated evidence; resource utilization alone "
                "(including explained saturation) is not an optimization opportunity"
            )

        primary_bottleneck = _to_bottleneck(primary) if primary is not None else None
        candidate_bottlenecks = [
            _to_bottleneck(item) for item in meaningful if item is not primary
        ]
        observation_bottlenecks = [_to_bottleneck(item) for item in observations]

        if primary is not None:
            overall_strength = primary.strength
            overall_confidence = primary.confidence
        elif ambiguous and meaningful:
            overall_strength = EvidenceStrength.MEDIUM
            overall_confidence = max(item.confidence for item in meaningful)
        else:
            overall_strength = EvidenceStrength.NONE
            overall_confidence = 0.0

        unavailable = _unavailable_measurements(report, groups)
        limitations = list(report.limitations)
        limitations.append(
            "Deterministic diagnosis uses relative-significance heuristics; confidence "
            "categories are evidence-derived, not calibrated probabilities."
        )
        reasoning = _reasoning_steps(
            groups, scored, primary, ambiguous, meaningful_opportunity, no_opportunity_reason
        )
        summary = _summary(
            primary_bottleneck, meaningful_opportunity, insufficient_evidence, no_opportunity_reason
        )

        return BottleneckDiagnosis(
            diagnosis_id=f"diag-{uuid.uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc),
            context=DiagnosisContext(
                source_report_path=source_report_path,
                source_report_generated_at=report.generated_at,
                profiler_version=report.profiler_version,
                detector_version=self.version,
                hardware=report.hardware,
                backend=report.backend,
                model_refs=sorted({group.model_ref for group in groups}),
                workload_ids=sorted({group.workload_id for group in groups}),
                conditions=sorted({group.condition for group in groups if group.condition}),
            ),
            meaningful_opportunity=meaningful_opportunity,
            insufficient_evidence=insufficient_evidence,
            escalation_required=escalation_required,
            confidence=overall_confidence,
            evidence_strength=overall_strength,
            primary=primary_bottleneck,
            candidates=candidate_bottlenecks,
            observations=observation_bottlenecks,
            supporting_evidence=[item for finding in scored for item in finding.supporting],
            contradicting_evidence=[item for finding in scored for item in finding.contradicting],
            unavailable_measurements=unavailable,
            limitations=limitations,
            reasoning=reasoning,
            no_opportunity_reason=no_opportunity_reason,
            summary=summary,
        )


class _ScoredFinding:
    def __init__(self, finding: CandidateFinding, group: GroupObservation | None) -> None:
        self.finding = finding
        self.group = group
        self.confidence = 0.0
        self.strength = EvidenceStrength.NONE
        self.is_meaningful = False

    @property
    def supporting(self) -> list[EvidenceItem]:
        return self.finding.supporting

    @property
    def contradicting(self) -> list[EvidenceItem]:
        return self.finding.contradicting

    @property
    def kind(self) -> BottleneckKind:
        return self.finding.kind

    @property
    def stage(self) -> InferenceStage:
        return self.finding.stage

    @property
    def subject(self) -> str:
        return self.finding.subject


def _confidence(finding: CandidateFinding, group: GroupObservation | None) -> float:
    support = len(finding.supporting)
    contradict = len(finding.contradicting)
    raw = max(0.0, min(1.0, (support - 0.5 * contradict) / SUPPORT_NORMALIZER))
    quality = 1.0
    if group is not None:
        if group.n < 2:
            quality = min(quality, 0.5)
        # High-magnitude evidence (e.g. extreme paging) is robust to run-to-run
        # variance, so the variance penalty is skipped when such evidence is present.
        has_high_strength = any(
            item.strength is EvidenceStrength.HIGH for item in finding.supporting
        )
        cv = _group_variance(group)
        if cv is not None and cv > HIGH_VARIANCE_CV and not has_high_strength:
            quality *= 0.7
    if finding.unavailable:
        quality *= 0.7
    return max(0.0, min(1.0, raw * quality))


def _group_variance(group: GroupObservation) -> float | None:
    cvs = [
        group.cv("decode_tokens_per_sec"),
        group.cv("total_latency_s"),
    ]
    present = [cv for cv in cvs if cv is not None]
    return max(present) if present else None


def _strength(confidence: float, finding: CandidateFinding) -> EvidenceStrength:
    support = len(finding.supporting)
    if confidence < EVIDENCE_STRENGTH_LOW:
        return EvidenceStrength.NONE
    if confidence >= EVIDENCE_STRENGTH_HIGH and support >= 2:
        return EvidenceStrength.HIGH
    if confidence >= EVIDENCE_STRENGTH_MEDIUM and support >= 2:
        return EvidenceStrength.MEDIUM
    return EvidenceStrength.LOW


def _is_ambiguous(meaningful: list[_ScoredFinding]) -> bool:
    families: dict[str, float] = {}
    for item in meaningful:
        family = _RESOURCE_FAMILY.get(item.kind, item.kind.value)
        families[family] = max(families.get(family, 0.0), item.confidence)
    if len(families) < 2:
        return False
    ordered = sorted(families.values(), reverse=True)
    return ordered[0] - ordered[1] <= AMBIGUITY_MARGIN


def _measurement_sufficient(groups: list[GroupObservation]) -> bool:
    if not groups:
        return False
    if not any(group.n >= 2 for group in groups):
        return False
    gpu_seen = any(group.mean("gpu_utilization_percent") is not None for group in groups)
    cpu_seen = any(group.mean("process_cpu_percent") is not None for group in groups)
    mem_seen = any(group.mean("available_ram_bytes") is not None for group in groups)
    return (gpu_seen or cpu_seen) and mem_seen


def _unavailable_measurements(
    report: BaselineReport, groups: list[GroupObservation]
) -> list[str]:
    unavailable: set[str] = set()
    if not any(group.mean("gpu_utilization_percent") is not None for group in groups):
        unavailable.add("gpu_utilization_percent")
    if not any(group.mean("hard_page_faults_per_sec") is not None for group in groups):
        unavailable.add("hard_page_faults_per_sec")
    if not any(group.mean("gpu_memory_used_bytes") is not None for group in groups):
        unavailable.add("gpu_memory_used_bytes")
    unavailable.add("gpu_memory_total_bytes")
    if not any(
        run.observation.system_state is not None
        and bool(run.observation.system_state.processes)
        for run in report.runs
    ):
        unavailable.add("system_state.processes")
    if not any(group.condition for group in groups):
        unavailable.add("comparative_conditions")
    return sorted(unavailable)


def _reasoning_steps(
    groups: list[GroupObservation],
    scored: list[_ScoredFinding],
    primary: _ScoredFinding | None,
    ambiguous: bool,
    meaningful_opportunity: bool,
    no_opportunity_reason: str | None,
) -> list[str]:
    steps: list[str] = []
    steps.append(
        f"Observed {len(groups)} comparative group(s) across "
        f"{len({group.model_ref for group in groups})} model(s)."
    )
    steps.append(
        "Applied relative-significance and comparative-evidence rules; thresholds only "
        "raise candidate signals, never conclusions."
    )
    for item in scored:
        status = "meaningful" if item.is_meaningful else "observation"
        steps.append(
            f"[{status}] {item.kind.value} on {item.subject} "
            f"(stage={item.stage.value}, confidence={item.confidence:.2f}, "
            f"strength={item.strength.value}, support={len(item.supporting)}, "
            f"contradict={len(item.contradicting)})"
        )
    if ambiguous:
        steps.append("Conflicting resource signals are comparably supported -> ambiguous.")
    if primary is not None:
        steps.append(
            f"Selected primary limitation '{primary.kind.value}' for stage "
            f"'{primary.stage.value}'."
        )
    elif not meaningful_opportunity:
        steps.append(f"No meaningful optimization opportunity: {no_opportunity_reason}")
    return steps


def _summary(
    primary: Bottleneck | None,
    meaningful_opportunity: bool,
    insufficient_evidence: bool,
    no_opportunity_reason: str | None,
) -> str:
    if primary is not None and meaningful_opportunity:
        return (
            f"Meaningful {primary.kind.value} limitation on stage "
            f"{(primary.stage.value if primary.stage else 'unknown')} "
            f"(confidence {primary.confidence:.2f}, strength {primary.evidence_strength.value})."
        )
    if insufficient_evidence:
        return f"Insufficient evidence for a confident bottleneck: {no_opportunity_reason}"
    return f"No meaningful optimization opportunity: {no_opportunity_reason}"


def _to_bottleneck(item: _ScoredFinding) -> Bottleneck:
    evidence = item.supporting + item.contradicting
    summary = (
        f"{'Meaningful' if item.is_meaningful else 'Observation'}: "
        f"{item.kind.value} on {item.subject} during stage {item.stage.value}"
    )
    return Bottleneck(
        bottleneck_id=f"bn-{uuid.uuid4().hex[:8]}",
        kind=item.kind,
        confidence=item.confidence,
        is_meaningful=item.is_meaningful,
        summary=summary,
        evidence=evidence,
        contributing_factors=list(item.finding.rationale),
        recommended_surface=[],
        source=item.finding.source,
        stage=item.stage,
        subject=item.subject,
        evidence_strength=item.strength,
        supporting_evidence=item.supporting,
        contradicting_evidence=item.contradicting,
        unavailable_measurements=list(item.finding.unavailable),
        rationale=list(item.finding.rationale),
        limitations=[],
        detector_version=DETECTOR_VERSION,
    )
