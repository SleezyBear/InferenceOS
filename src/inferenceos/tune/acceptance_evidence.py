"""Evidence an accepted setup must carry, beyond speed (review 7, 6 Oct 2026).

On the fresh loads that supply acceptance evidence:

* **Resources of a memory-safety correction.** A correction makes no speed claim and is
  not compared with the unsafe start for speed, but it exists to keep the machine within
  its resources. So it must show, freshly: a load check that it fits with the safety
  margins; every resource reading present on every run (paging and memory always, plus
  whatever the goal guards); no severe paging; and paging not above the starting setup's
  by more than the usual tolerance.
* **Applied settings** are checked in ``tune/pipeline.py`` with
  ``paths.verify_every_run`` on every fresh measurement used as acceptance evidence
  (final check, partial-approval recheck, trade-off, known answers, safety check), on
  both setups. Approval can waive a speed loss, never missing configuration evidence.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field

from inferenceos.contracts.experiment import ExperimentMeasurement
from inferenceos.optimization.goals import plain_name
from inferenceos.tune.fair import COUNT_GUARDS
from inferenceos.tune.layer_top import LoadAttempt

#: Resource readings a memory-safety correction must always carry, whatever the goal.
SAFETY_READINGS: tuple[str, ...] = ("hard_page_faults_per_sec", "peak_rss_bytes")


@dataclass
class ResourceEvidence:
    """Whether a memory-safety correction was shown to stay within its resources."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    readings: dict[str, object] = field(default_factory=dict)


def _values(measurement: ExperimentMeasurement, metric: str) -> list[float | None]:
    return [
        None if getattr(run, metric, None) is None else float(getattr(run, metric))
        for run in measurement.run_metrics
    ]


def safety_resource_evidence(
    *,
    measurement: ExperimentMeasurement,
    reference: ExperimentMeasurement | None,
    guarded: Iterable[str],
    load_check: LoadAttempt | None,
) -> ResourceEvidence:
    """The resource evidence a memory-safety correction needs (no speed claim).

    ``measurement`` is the fresh measurement of the corrected setup (its valid runs);
    ``reference`` the starting setup's measurement from the start of this tune (used only
    for the paging tolerance, never for speed); ``load_check`` a fresh load-only try of
    the corrected setup.
    """
    reasons: list[str] = []
    readings: dict[str, object] = {}
    passed = True
    if load_check is None:
        passed = False
        reasons.append("no fresh load check could be made, so fitting in graphics memory is not shown")
    else:
        readings["load_check"] = {
            "loaded": load_check.loaded,
            "fits": load_check.fits,
            "used_bytes": load_check.used_bytes,
            "free_bytes": load_check.free_bytes,
            "error": load_check.error,
        }
        if not load_check.loaded:
            passed = False
            reasons.append(f"a fresh load did not come up ({load_check.error or 'no reason given'})")
        elif not load_check.fits:
            passed = False
            reasons.append("a fresh load check says it does not leave enough graphics memory free")
    runs = len(measurement.run_metrics)
    required = list(dict.fromkeys([*SAFETY_READINGS, *guarded]))
    if runs == 0:
        passed = False
        reasons.append("no run of the memory-safe setup could be used")
    for metric in required:
        values = _values(measurement, metric)
        present = [value for value in values if value is not None]
        readings[metric] = {"read_in": len(present), "runs": runs, "values": present}
        if runs and len(present) < runs:
            passed = False
            reasons.append(
                f"{plain_name(metric)} was read in {len(present)} of {runs} run(s); every "
                "run must have it"
            )
            continue
        if metric in COUNT_GUARDS and present:
            tolerance, severe = COUNT_GUARDS[metric]
            worst = max(present)
            if worst >= severe:
                passed = False
                reasons.append(
                    f"severe {plain_name(metric)} ({worst:,.0f}/s, limit {severe:,.0f}/s)"
                )
            middle = float(statistics.median(present))
            start_values = (
                [value for value in _values(reference, metric) if value is not None]
                if reference is not None
                else []
            )
            start_middle = float(statistics.median(start_values)) if start_values else 0.0
            if middle > start_middle + tolerance:
                passed = False
                reasons.append(
                    f"{plain_name(metric)} {middle:,.0f}/s, more than the starting setup's "
                    f"{start_middle:,.0f}/s plus {tolerance:,.0f}/s"
                )
    if passed:
        reasons.append(
            "fresh load check fits; every resource reading present on every run; no "
            "severe paging"
        )
    return ResourceEvidence(passed=passed, reasons=reasons, readings=readings)


__all__ = [
    "ResourceEvidence",
    "SAFETY_READINGS",
    "safety_resource_evidence",
]
