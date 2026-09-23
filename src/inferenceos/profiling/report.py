"""Human-readable rendering of a baseline report (no analysis, just numbers)."""

from __future__ import annotations

from inferenceos.contracts.profiling import BaselineReport, MetricStats, RunGroupStats

_KEY_METRICS: tuple[str, ...] = (
    "model_load_time_s",
    "ttft_s",
    "prefill_latency_s",
    "decode_latency_s",
    "prompt_tokens_per_sec",
    "decode_tokens_per_sec",
    "total_latency_s",
    "process_wall_s",
    "peak_rss_bytes",
    "gpu_utilization_percent",
    "hard_page_faults_per_sec",
)


def render_summary(report: BaselineReport) -> str:
    lines: list[str] = []
    lines.append("=== Baseline report ===")
    lines.append(f"profiler      : {report.profiler_version} (schema {report.schema_version})")
    lines.append(f"suite version : {report.suite_version}")
    lines.append(f"generated at  : {report.generated_at.isoformat()}")
    lines.append(
        f"hardware      : {report.hardware.cpu.name} | "
        f"{report.hardware.cpu.physical_cores}c/{report.hardware.cpu.logical_cores}t | "
        f"{report.hardware.memory.total_bytes / 1024**3:.1f} GiB"
    )
    lines.append(
        f"backend       : {report.backend.backend_name} {report.backend.backend_version or ''} "
        f"device={report.backend.device} offload={report.backend.gpu_layers_offloaded}"
        f"/{report.backend.gpu_layers_total}"
    )
    lines.append(
        f"repo runs     : {len(report.runs)} "
        f"({sum(1 for run in report.runs if not run.identity.warmup)} measured, "
        f"{sum(1 for run in report.runs if run.identity.warmup)} warm-up)"
    )
    lines.append("")
    lines.append("--- Per-run measurements ---")
    for run in report.runs:
        metrics = run.observation.metrics
        tag = "warm-up" if run.identity.warmup else f"rep {run.identity.repetition}"
        lines.append(
            f"[{run.identity.mode.value:4}] {run.context.model.name} / "
            f"{run.context.workload.workload_id} ({tag}): "
            f"load={_fmt(metrics.model_load_time_s)}s ttft={_fmt(metrics.ttft_s)}s "
            f"prefill={_fmt(metrics.prefill_latency_s)}s decode={_fmt(metrics.decode_latency_s)}s "
            f"decode_tps={_fmt(metrics.decode_tokens_per_sec)} "
            f"tokens={metrics.prompt_tokens}/{metrics.generated_tokens} "
            f"peak_rss={_fmt_gib(metrics.peak_rss_bytes)} gpu={_fmt(metrics.gpu_utilization_percent)}% "
            f"hardfaults/s={_fmt(metrics.hard_page_faults_per_sec)}"
        )
    lines.append("")
    lines.append("--- Aggregated groups (warm-up excluded) ---")
    for group in report.groups:
        lines.append(
            f"{group.model_ref} / {group.workload_id} [{group.mode.value}] "
            f"n={group.repetitions}"
        )
        for metric in _KEY_METRICS:
            stats = group.metrics.get(metric)
            if stats is None or stats.count == 0:
                lines.append(f"    {metric:<24} unavailable")
                continue
            lines.append(f"    {metric:<24} {_stat_line(stats)}")
    return "\n".join(lines)


def _stat_line(stats: MetricStats) -> str:
    mean = _fmt(stats.mean)
    stdev = _fmt(stats.stdev)
    cv = f"{stats.cv * 100:.1f}%" if stats.cv is not None else "n/a"
    return (
        f"mean={mean} stdev={stdev} min={_fmt(stats.min)} max={_fmt(stats.max)} cv={cv} "
        f"(n={stats.count})"
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.4f}"


def _fmt_gib(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1024**3:.2f}GiB"


def group_lookup(report: BaselineReport) -> dict[tuple[str, str, str], RunGroupStats]:
    return {
        (group.model_ref, group.workload_id, group.mode.value): group for group in report.groups
    }
