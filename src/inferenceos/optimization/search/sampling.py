"""Deterministic candidate sampling across a feasible layer range.

The planner never tests every integer layer count. It produces a bounded, deterministic
set that always contains the current configuration (the baseline) and, when a higher
safe configuration exists, the safe upper boundary. The exact step and cap come from
:class:`SearchPolicy`, so the strategy is inspectable and reproducible.
"""

from __future__ import annotations

from inferenceos.optimization.search.policy import SearchPolicy


def sample_layers(current: int, feasible_max: int, policy: SearchPolicy) -> list[int]:
    """Deterministic, de-duplicated, ascending sample points in ``[current, max]``.

    Always includes ``current`` (the baseline) and — when ``max > current`` — the upper
    boundary. Points are spread evenly across the range at a granularity no finer than
    ``policy.sample_step_layers``, bounded by ``policy.max_samples``.
    """
    if feasible_max <= current:
        return [int(current)]

    span = int(feasible_max) - int(current)
    by_step = span // policy.sample_step_layers + 1
    count = max(2, min(policy.max_samples, by_step))
    points = {int(current), int(feasible_max)}
    for index in range(1, count - 1):
        points.add(int(current) + (index * span) // (count - 1))

    return sorted(points)


def planned_probe(current: int, feasible_max: int, policy: SearchPolicy) -> int:
    """The single exploratory value a Phase 6 planner proposes.

    This is a bounded step above the current configuration — deliberately *not* the
    upper boundary. The empirical search may explore further; the planner never
    blindly proposes the maximum.
    """
    if feasible_max <= current:
        return int(current)
    return min(int(feasible_max), int(current) + policy.probe_step_layers)
