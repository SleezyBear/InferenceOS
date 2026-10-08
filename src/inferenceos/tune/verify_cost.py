"""Where measurement time goes, and what verification will cost (handoff, 6 Oct 2026;
attribution corrected after review 8).

Every run records its phases (``ProfilingRun.phase_timings_s``: engine load, monitor
start including the wait for its first reading, the request, monitor stop, the reads
afterwards, unload, and the engine's own first-word time). Warm-ups are never scored, so
their runs are not in ``ExperimentMeasurement.runs``; their phases are kept separately
in ``warmup_phase_timings`` (the warm-up is where the fresh load happens). A
measurement's elapsed time minus every recorded phase (warm-ups included) is what
happens between runs: the fresh start, failed runs, disturbed repeats, bookkeeping.

``CostLedger`` keeps every timed operation of a tune, final-check measurements included,
and answers "how long does one measurement on this question take here", from this
tune's own measurements only.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field

from inferenceos.contracts.experiment import ExperimentMeasurement

PHASES = ("load_s", "monitor_start_s", "request_s", "monitor_stop_s", "after_reads_s", "unload_s")
MEASUREMENT_KINDS = ("measure", "final-side")


def run_phases(measurement: ExperimentMeasurement) -> list[dict[str, float]]:
    """The recorded phases of each scored run, in order (warm-ups not included)."""
    return [dict(run.phase_timings_s) for run in measurement.runs if run.phase_timings_s]


def measurement_breakdown(measurement: ExperimentMeasurement, wall_s: float | None = None) -> dict[str, object]:
    """One measurement's elapsed time split into phases (seconds).

    Phase totals include the warm-up runs (which carry the load); ``warmup_s`` is their
    total and ``scored_s`` the scored runs' total. ``between_runs_s`` is elapsed time no
    recorded run accounts for; ``None`` when nothing was recorded or the elapsed time is
    unknown."""
    wall = wall_s if wall_s is not None else measurement.elapsed_s
    scored = run_phases(measurement)
    warm = [dict(item) for item in measurement.warmup_phase_timings]
    every = scored + warm
    totals = {name: round(sum(item.get(name, 0.0) for item in every), 3) for name in PHASES}
    first_word = [item["first_word_s"] for item in scored if "first_word_s" in item]
    accounted = sum(totals.values())
    return {
        "runs_with_phases": len(scored),
        "warmups_with_phases": len(warm),
        "runs": measurement.valid_runs + measurement.warmup_runs + measurement.invalid_runs,
        "wall_s": None if wall is None else round(wall, 3),
        **totals,
        "warmup_s": round(sum(sum(item.get(n, 0.0) for n in PHASES) for item in warm), 3),
        "scored_s": round(sum(sum(item.get(n, 0.0) for n in PHASES) for item in scored), 3),
        "first_word_s_total": round(sum(first_word), 3) if first_word else None,
        "monitor_overhead_s": round(totals["monitor_start_s"] + totals["monitor_stop_s"], 3),
        "between_runs_s": round(max(wall - accounted, 0.0), 3) if (wall is not None and every) else None,
    }


@dataclass
class Operation:
    kind: str  # "measure", "final-side", "load-try", "final", "save", "cleanup", ...
    question: str | None
    wall_s: float
    breakdown: dict[str, object] = field(default_factory=dict)
    started_at: float = 0.0


@dataclass
class CostLedger:
    operations: list[Operation] = field(default_factory=list)

    def add(self, kind: str, question: str | None, wall_s: float, *, started_at: float = 0.0,
            measurement: ExperimentMeasurement | None = None) -> Operation:
        breakdown = measurement_breakdown(measurement, wall_s) if measurement is not None else {}
        op = Operation(kind, question, max(float(wall_s), 0.0), breakdown, started_at)
        self.operations.append(op)
        return op

    def _walls(self, question: str) -> list[float]:
        return [op.wall_s for op in self.operations if op.question == question and op.kind in MEASUREMENT_KINDS]

    def measurement_cost(self, question: str) -> float | None:
        """Median wall time of one fresh-load measurement on ``question`` in this tune."""
        walls = self._walls(question)
        return float(statistics.median(walls)) if walls else None

    def worst_measurement_cost(self, question: str) -> float | None:
        walls = self._walls(question)
        return max(walls) if walls else None

    def answer_cost(self, question: str) -> float | None:
        """Median seconds per scored answer (request plus monitor and reads) on
        ``question``, from recorded phases."""
        per: list[float] = []
        for op in self.operations:
            if op.question != question or op.kind not in MEASUREMENT_KINDS:
                continue
            runs = op.breakdown.get("runs_with_phases")
            scored = op.breakdown.get("scored_s")
            if isinstance(runs, int) and runs and isinstance(scored, (int, float)):
                per.append(float(scored) / runs)
        return float(statistics.median(per)) if per else None

    def load_cost(self) -> float | None:
        """Median seconds of a fresh load (warm-up phases' load), from recorded phases."""
        loads: list[float] = []
        for op in self.operations:
            value = op.breakdown.get("load_s")
            if op.kind in MEASUREMENT_KINDS and op.breakdown.get("warmups_with_phases") and isinstance(value, (int, float)):
                loads.append(float(value))
        return float(statistics.median(loads)) if loads else None

    def total(self, kinds: Iterable[str] | None = None) -> float:
        wanted = set(kinds) if kinds is not None else None
        return sum(op.wall_s for op in self.operations if wanted is None or op.kind in wanted)

    def summary(self) -> dict[str, object]:
        by_kind: dict[str, float] = {}
        for op in self.operations:
            by_kind[op.kind] = round(by_kind.get(op.kind, 0.0) + op.wall_s, 2)
        names = (*PHASES, "monitor_overhead_s", "between_runs_s", "warmup_s", "scored_s")
        phase_totals = {name: 0.0 for name in names}
        measured_runs = 0
        for op in self.operations:
            for name in names:
                value = op.breakdown.get(name)
                if isinstance(value, (int, float)):
                    phase_totals[name] += float(value)
            runs = op.breakdown.get("runs_with_phases")
            warm = op.breakdown.get("warmups_with_phases")
            measured_runs += (runs if isinstance(runs, int) else 0) + (warm if isinstance(warm, int) else 0)
        return {
            "seconds_by_kind": by_kind,
            "phase_seconds": {k: round(v, 2) for k, v in phase_totals.items()},
            "runs_with_phases": measured_runs,
            "monitor_overhead_per_run_s": (
                round(phase_totals["monitor_overhead_s"] / measured_runs, 3) if measured_runs else None
            ),
            "measurement_cost_s": {
                question: round(cost, 2)
                for question in sorted({op.question for op in self.operations if op.question})
                if (cost := self.measurement_cost(question)) is not None
            },
        }


__all__ = ["CostLedger", "MEASUREMENT_KINDS", "Operation", "PHASES", "measurement_breakdown", "run_phases"]
