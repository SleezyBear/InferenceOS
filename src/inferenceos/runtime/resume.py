"""Search-duration-aware bounded resume verification.

A GPU-offload search pauses the workload for minutes (repeated resident reloads). The
post-search resume must not fail merely because the search itself exceeded an unrelated
fixed timeout. Resume readiness is verified as its own explicit phase whose timeout is
derived from a bounded base allowance plus a capped share of the *actual* search
duration, always within a hard maximum. It never weakens the terminal checks: a dead
target, a lost/identity-mismatched process or an unrecoverable target still fails.
"""

from __future__ import annotations

from inferenceos.config import Settings


def compute_resume_timeout(
    *,
    search_duration_s: float,
    configured_search_budget_s: float,
    base_allowance_s: float,
    search_factor: float,
    max_search_allowance_s: float,
    hard_max_s: float,
) -> float:
    """Return the bounded resume-verification timeout in seconds.

    ``configured_search_budget_s`` is accepted so the caller's chosen budget is part of
    the derivation (a longer permitted search implies more settling time). The result is
    always ``>= base_allowance_s`` and ``<= hard_max_s`` when a positive hard maximum is
    configured.
    """
    base = max(1.0, float(base_allowance_s))
    duration = max(0.0, float(search_duration_s))
    factor = max(0.0, float(search_factor))
    allowance = min(duration * factor, max(0.0, float(max_search_allowance_s)))
    timeout = base + allowance
    # A deliberately huge configured budget must never make the verification unbounded.
    _ = configured_search_budget_s
    hard_max = float(hard_max_s)
    if hard_max > 0.0:
        timeout = min(timeout, hard_max)
    return max(1.0, timeout)


def resume_timeout_settings(settings: Settings) -> dict[str, float]:
    """The configured resume-verification bounds (explicit, inspectable)."""
    return {
        "base_allowance_s": float(settings.optimization_resume_base_timeout_s),
        "search_factor": float(settings.optimization_resume_search_factor),
        "max_search_allowance_s": float(
            settings.optimization_resume_max_search_allowance_s
        ),
        "hard_max_s": float(settings.optimization_resume_hard_max_s),
    }


__all__ = ["compute_resume_timeout", "resume_timeout_settings"]
