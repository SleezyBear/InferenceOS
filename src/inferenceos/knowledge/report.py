"""Human-readable rendering of persistent knowledge, applicability, and health."""

from __future__ import annotations

from inferenceos.contracts.knowledge import (
    ApplicabilityResult,
    KnowledgeHealth,
    OptimizationRecord,
    RollbackRecord,
    StorageStats,
)


def render_record(record: OptimizationRecord) -> str:
    lines: list[str] = []
    lines.append(f"record      : {record.record_id}")
    lines.append(f"experiment  : {record.experiment_id}")
    lines.append(f"created     : {record.created_at.isoformat()}")
    lines.append(
        f"candidate   : {record.candidate_id}  action={record.action_id}  "
        f"registered={record.action_registered}"
    )
    lines.append(f"parameters  : {record.parameters}")
    lines.append(
        f"outcome     : {record.outcome.value}  decision={record.decision.value if record.decision else None} "
        f"keep={record.keep}"
    )
    lines.append(
        f"flags       : regression={record.regression} quality_regression={record.quality_regression} "
        f"bottleneck_improved={record.bottleneck_improved}"
    )
    lines.append(
        f"restoration : {record.restoration_status.value}  environment_restored={record.environment_restored}"
    )
    lines.append(
        f"quality     : {record.quality_status.value if record.quality_status else None} "
        f"validated={record.quality_validated} similarity={record.quality_similarity}"
    )
    lines.append(
        f"diagnosis   : {record.diagnosed_bottleneck} (source {record.diagnosis_id})"
    )
    lines.append(
        f"context     : identity={record.context.fingerprint.identity_hash} "
        f"conditions={record.context.fingerprint.conditions_hash}"
    )
    lines.append(
        f"              hardware={record.context.fingerprint.hardware_key}"
    )
    lines.append(f"              model={record.context.fingerprint.model_key}")
    lines.append(f"              workload={record.context.fingerprint.workload_key}")
    lines.append(f"              runtime={record.context.fingerprint.runtime_key}")
    if record.measured_benefit:
        lines.append(f"benefit     : {record.measured_benefit}")
    if record.measured_cost:
        lines.append(f"cost        : {record.measured_cost}")
    if record.uncertainty:
        lines.append(f"uncertainty : {record.uncertainty}")
    if record.rollback_id:
        lines.append(f"rollback    : {record.rollback_id}")
    if record.evidence:
        lines.append("evidence    :")
        for item in record.evidence:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def render_history(records: list[OptimizationRecord]) -> str:
    lines = ["=== Optimization history (oldest first) ==="]
    if not records:
        lines.append("  (no records)")
    for record in records:
        lines.append(
            f"  {record.created_at.isoformat()}  {record.action_id}  "
            f"outcome={record.outcome.value}  keep={record.keep}  "
            f"model={record.context.fingerprint.model_key.split(';')[0]}"
        )
    return "\n".join(lines)


def render_applicability(result: ApplicabilityResult) -> str:
    lines: list[str] = []
    lines.append("=== Applicability ===")
    lines.append(f"record        : {result.record_id} ({result.experiment_id})")
    lines.append(f"action        : {result.action_id}  historical outcome={result.record_outcome.value}")
    lines.append(f"classification: {result.classification.value}")
    lines.append(f"score         : {result.score}")
    lines.append(f"components    : {result.components}")
    lines.append(f"rationale     : {result.rationale}")
    for label, items in (
        ("match", result.matches),
        ("differ", result.differences),
        ("unknown", result.unknown),
        ("conflict", result.conflicts),
    ):
        for item in items:
            lines.append(
                f"  [{label}] {item.dimension}: historical={item.historical} "
                f"current={item.current}"
            )
    if result.revalidation_triggers:
        lines.append(
            "triggers      : " + ", ".join(t.value for t in result.revalidation_triggers)
        )
    for reason in result.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def render_rollback(record: RollbackRecord) -> str:
    lines: list[str] = []
    lines.append("=== Rollback record ===")
    lines.append(f"rollback    : {record.rollback_id} (experiment {record.experiment_id})")
    lines.append(f"action      : {record.action_id}  scope={record.target_scope}")
    lines.append(f"previous    : {record.previous_state.as_dict()}")
    lines.append(f"resulting   : {record.resulting_state.as_dict()}")
    lines.append(f"restore     : {record.restore_parameters}")
    lines.append(
        f"restored    : experiment_restored={record.experiment_restored} "
        f"status={record.restoration_status.value}"
    )
    verification = record.verification
    lines.append(
        f"verification: verified={verification.verified} "
        f"action_registered={verification.action_registered} "
        f"parameters_match={verification.parameters_match} "
        f"expected_matches_previous={verification.expected_matches_previous} "
        f"context_compatible={verification.context_compatible}"
    )
    for reason in verification.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def render_health(health: KnowledgeHealth, stats: StorageStats | None = None) -> str:
    lines = ["=== Knowledge store health ==="]
    lines.append(f"status          : {health.status.value}")
    lines.append(
        f"schema          : {health.schema_version} (expected {health.expected_schema_version})"
    )
    lines.append(
        f"records         : {health.record_count}  rollbacks={health.rollback_count} "
        f"events={health.event_count}"
    )
    lines.append(f"db size bytes   : {health.db_size_bytes}")
    if health.corrupted_records:
        lines.append(f"corrupted       : {health.corrupted_records}")
    lines.append(f"detail          : {health.detail}")
    if stats is not None:
        lines.append(f"db path         : {stats.db_path}")
    return "\n".join(lines)
