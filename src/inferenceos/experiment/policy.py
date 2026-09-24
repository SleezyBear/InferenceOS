"""Explicit, inspectable Phase 7 experiment policy.

Safety-critical behaviour (repetition counts, time/resource limits, effect and noise
thresholds, quality policy, restoration policy) lives here as named, validated fields
rather than as magic constants inside the engine.
"""

from __future__ import annotations

from inferenceos.config import Settings
from inferenceos.contracts.experiment import ExperimentPolicy


def build_experiment_policy(settings: Settings) -> ExperimentPolicy:
    return ExperimentPolicy(
        warmup_repetitions=settings.phase7_warmup_repetitions,
        measured_repetitions=settings.phase7_measured_repetitions,
        max_repetitions=settings.phase7_max_repetitions,
        min_valid_samples=settings.phase7_min_valid_samples,
        max_total_runs=settings.phase7_max_total_runs,
        time_limit_s=settings.phase7_time_limit_s,
        run_timeout_s=settings.phase7_run_timeout_s,
        min_effect=settings.phase7_min_effect,
        noise_tolerance=settings.phase7_noise_tolerance,
        quality_similarity_min=settings.phase7_quality_similarity_min,
        require_quality_validation=settings.phase7_require_quality_validation,
        restore_baseline=settings.phase7_restore_baseline,
    )
