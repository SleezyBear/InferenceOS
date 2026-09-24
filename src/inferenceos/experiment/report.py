"""Human-readable rendering of a Phase 7 experiment result."""

from __future__ import annotations

from inferenceos.contracts.experiment import ExperimentMeasurement, ExperimentResult


def render_experiment(result: ExperimentResult) -> str:
    lines: list[str] = []
    lines.append("=== Experiment result ===")
    lines.append(f"experiment   : {result.experiment_id}")
    lines.append(f"candidate    : {result.candidate_id} ({result.action_id})")
    lines.append(f"parameters   : {result.parameters}")
    lines.append(f"status       : {result.status.value if result.status else None}")
    lines.append(f"executed     : {result.executed} (environment restored: {result.environment_restored})")
    lines.append(f"summary      : {result.summary}")
    lines.append("")

    request = result.request
    if request is not None:
        lines.append("--- Request / reproducibility ---")
        lines.append(f"diagnosis    : {request.diagnosed_bottleneck} (source {request.source_diagnosis_id})")
        lines.append(f"bottleneck   : {request.bottleneck_metric} ({request.bottleneck_direction})")
        lines.append(f"model        : {request.model.path} [{request.model.architecture}/{request.model.quantization.value}]")
        lines.append(f"workload     : {request.workload.workload_id} seed={request.seed}")
        lines.append(f"baseline cfg : {request.baseline_configuration.as_dict()}")
        lines.append(f"experiment   : {request.experiment_configuration.as_dict()}")
        lines.append("")

    lines.append("--- Application (what changed) ---")
    if result.application is None:
        lines.append("  not applied")
    else:
        lines.append(f"  applied={result.application.applied} supported={result.application.supported} reason={result.application.reason}")
        for change in result.application.changes:
            lines.append(f"  {change.parameter}: {change.prior_value} -> {change.applied_value}")
    lines.append("")

    lines.append("--- Baseline measurement ---")
    _render_measurement(lines, result.baseline_measurement)
    lines.append("--- Candidate measurement ---")
    _render_measurement(lines, result.candidate_measurement)
    lines.append("")

    if result.comparison is not None:
        lines.append("--- Comparison (baseline -> candidate) ---")
        for metric, delta in result.comparison.deltas.items():
            if delta.baseline_mean is None or delta.candidate_mean is None:
                continue
            rel = f"{delta.relative_delta:+.1%}" if delta.relative_delta is not None else "n/a"
            lines.append(
                f"  [{delta.role}] {metric}: {delta.baseline_mean:.4g} -> "
                f"{delta.candidate_mean:.4g} ({rel}, significant={delta.significant})"
            )
        lines.append(
            f"  bottleneck improved={result.comparison.bottleneck_improved} "
            f"regressed={result.comparison.bottleneck_regressed} "
            f"persistent={result.comparison.bottleneck_persistent}"
        )
        lines.append(
            f"  objective improved={result.comparison.objective_improved} "
            f"regressed={result.comparison.objective_regressed} "
            f"resource_regressed={result.comparison.resource_regressed}"
        )
        lines.append(f"  uncertainty: {result.comparison.uncertainty}")
        lines.append("")

    if result.quality is not None:
        lines.append("--- Quality validation ---")
        lines.append(
            f"  status={result.quality.status.value} validated={result.quality.validated} "
            f"execution_ok={result.quality.execution_ok} structural_ok={result.quality.structural_ok} "
            f"similarity={result.quality.similarity} exact={result.quality.deterministic_match}"
        )
        for check in result.quality.checks:
            lines.append(f"  [{'PASS' if check.passed else 'FAIL'}] {check.description}: {check.detail}")
        for item in result.quality.rationale:
            lines.append(f"  - {item}")
        lines.append("")

    if result.decision_detail is not None:
        decision = result.decision_detail
        lines.append("--- Decision ---")
        lines.append(f"  kind        : {decision.kind.value} (keep={decision.keep})")
        lines.append(f"  reason      : {decision.reason}")
        lines.append(f"  confidence  : {decision.confidence:.2f}")
        lines.append(f"  regression  : {decision.regression} quality_regression={decision.quality_regression}")
        for item in decision.evidence:
            lines.append(f"  evidence    : {item}")
        lines.append("")

    lines.append("--- Restoration ---")
    if result.restoration is None:
        lines.append("  not attempted")
    else:
        lines.append(
            f"  status={result.restoration.status.value} verified={result.restoration.verified} "
            f"reason={result.restoration.reason}"
        )
    lines.append("")

    if result.overhead is not None:
        o = result.overhead
        lines.append("--- Experiment overhead (separate from workload metrics) ---")
        lines.append(f"  baseline setup : {o.baseline_setup_s:.2f}s")
        lines.append(f"  application    : {o.application_s:.4f}s")
        lines.append(f"  measurement    : {o.measurement_s:.2f}s")
        lines.append(f"  validation     : {o.validation_s:.4f}s")
        lines.append(f"  restoration    : {o.restoration_s:.4f}s")
        lines.append(f"  total          : {o.total_s:.2f}s")
        lines.append("")

    if result.errors:
        lines.append("--- Errors ---")
        for error in result.errors:
            lines.append(f"  - {error}")
    return "\n".join(lines)


def _render_measurement(lines: list[str], measurement: ExperimentMeasurement | None) -> None:
    if measurement is None:
        lines.append("  not measured")
        return
    lines.append(
        f"  config={measurement.configuration.as_dict()} "
        f"valid={measurement.valid_runs} invalid={measurement.invalid_runs}"
    )
    for metric in (
        "decode_tokens_per_sec",
        "total_latency_s",
        "peak_rss_bytes",
        "hard_page_faults_per_sec",
    ):
        stats = measurement.metrics.get(metric)
        if stats is not None and stats.mean is not None:
            cv = f" cv={stats.cv:.3f}" if stats.cv is not None else ""
            lines.append(f"  {metric}: mean={stats.mean:.4g} n={stats.count}{cv}")
    for note in measurement.notes:
        lines.append(f"  note: {note}")
