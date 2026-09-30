"""Uncertainty-aware benefit estimation.

The optimizer does not know real speedups it has not measured. Benefit estimates are
therefore evidence-derived where possible, bounded, and explicit about unknown
magnitudes: ``known=False`` means "Phase 7 must measure this", never a fabricated
number.
"""

from __future__ import annotations

from inferenceos.contracts.optimization import BenefitEstimate
from inferenceos.optimization.context import OptimizationContext

# Normalization scales used only to place an *estimated* magnitude on a 0..1 axis
# for ranking. They are heuristics, not measured effects.
_METRIC_SCALES: dict[str, str] = {
    "peak_rss_bytes": "total_ram_bytes",
    "available_ram_bytes": "total_ram_bytes",
    "gpu_memory_used_bytes": "vram_bytes",
    "hard_page_faults_per_sec": "page_fault_scale",
    "process_cpu_percent": "cpu_scale",
    "decode_tokens_per_sec": "decode_scale",
    "total_latency_s": "latency_scale",
}

_SCALE_DEFAULTS = {
    "ram_bytes": 16 * 1024**3,
    "vram_bytes": 4 * 1024**3,
    "page_fault_scale": 1000.0,
    "cpu_scale": 100.0,
    "decode_scale": 100.0,
    "latency_scale": 10.0,
}


def external_process_benefit(
    *,
    process_rss_bytes: float | None,
    confidence: float,
    evidence_basis: list[str],
) -> BenefitEstimate:
    if process_rss_bytes is None:
        return unknown_benefit(
            "available_ram_bytes",
            "increase",
            confidence=confidence,
            evidence_basis=evidence_basis,
            notes="external process resident size unavailable; magnitude unknown",
        )
    return BenefitEstimate(
        metric="available_ram_bytes",
        direction="increase",
        unit="bytes",
        known=True,
        magnitude_low=float(process_rss_bytes),
        magnitude_high=float(process_rss_bytes),
        confidence=confidence,
        evidence_basis=evidence_basis,
        notes="bounded by the observed resident memory of the dominant external process",
    )


def unknown_benefit(
    metric: str,
    direction: str,
    *,
    confidence: float,
    evidence_basis: list[str],
    notes: str,
) -> BenefitEstimate:
    return BenefitEstimate(
        metric=metric,
        direction=direction,
        unit="value",
        known=False,
        magnitude_low=None,
        magnitude_high=None,
        confidence=confidence,
        evidence_basis=evidence_basis,
        notes=notes or "magnitude not responsibly estimable before measurement",
    )


def benefit_score(benefit: BenefitEstimate, ctx: OptimizationContext) -> float:
    """0..1 benefit axis; unknown magnitudes get a deliberately modest score."""
    if not benefit.known or benefit.magnitude_high is None:
        return 0.35
    return min(1.0, max(0.0, benefit.magnitude_high / _reference_scale(benefit.metric, ctx)))


def _reference_scale(metric: str, ctx: OptimizationContext) -> float:
    key = _METRIC_SCALES.get(metric)
    if key == "total_ram_bytes":
        value = ctx.observed_metrics.get("total_ram_bytes") or ctx.hardware.memory.total_bytes
        return float(value) if value else _SCALE_DEFAULTS["ram_bytes"]
    if key == "vram_bytes":
        for gpu in ctx.hardware.gpus:
            if gpu.vram_bytes:
                return float(gpu.vram_bytes)
        return _SCALE_DEFAULTS["vram_bytes"]
    if key is not None:
        return _SCALE_DEFAULTS.get(key, 1.0)
    return 1.0
