"""Request-specific cost forecasts. Historical speed/quality is never evidence.

The initial window/count/quantile are provisional engineering limits. A forecast is
not a deadline or a calibrated guarantee. No final-side or legacy request is reused.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import math
import re
import statistics
from typing import Any
from copy import deepcopy

from inferenceos.contracts.tune_learning import TuneHistoryRun
from inferenceos.tune.learning import _hash, request_identity_problems

MAX_AGE = timedelta(days=30)
MIN_RUNS = 3
MAX_RUNS = 20


def compatible_request(old: dict[str, Any], new: dict[str, Any], wall_s: float) -> bool:
    """Same request, with either identical caps or demonstrably nonbinding caps.

    Total completed measurement wall time bounds every individual answer's time.
    Require it to fit below ALL old/new watchdog allowances before permitting a
    difference. Never erase deadlines or infer missing limits. The watchdog rule and
    every other request field still match exactly. This is cost compatibility only.
    """
    if request_identity_problems(old) or request_identity_problems(new):
        return False
    if old == new:
        return True
    return (0 < wall_s < min(*old["watchdog_limits"].values(), *new["watchdog_limits"].values())
            and {k: v for k, v in old.items() if k != "watchdog_limits"}
            == {k: v for k, v in new.items() if k != "watchdog_limits"})


def envelope_key(identity: dict[str, Any]) -> str | None:
    """Match measured host/conditions; engine/device is checked per request instead.

    Free memory uses conservative 256 MiB bands; exact power plan/source and driver
    strings are required. Missing information is never supplied from current defaults.
    """
    try:
        model, machine = identity["model"], identity["machine"]
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (model["sha256"], identity["environment_sha256"])):
            return None
        if any(type(value) is not int or value <= 0 for value in
               (model["size_bytes"], identity["context_length"], machine["total_graphics_memory_bytes"])):
            return None
        # These two run-wide unknowns are superseded ONLY by the required observed
        # per-request runtime identity. Any other missing condition disables reuse.
        missing = identity.get("missing", [])
        if not isinstance(missing, list) or any(not isinstance(value, str) or value not in
                ("engine bundle/dependency identity", "effective backend/device identity") for value in missing):
            return None
        required = [machine["hostname"], machine["hardware_signature"], machine["program_version"],
                    machine["power_plan"]["plan_id"], identity["os"], identity["environment_sha256"]]
        if (not all(isinstance(value, str) and value.strip() and value.strip().lower() != "unknown"
                    for value in required) or machine["power_source"] not in ("ac", "battery")):
            return None
        drivers = machine["graphics_drivers"]
        if not isinstance(drivers, list) or not drivers or not all(
                isinstance(d, str) and d.strip() and d.strip().lower() != "unknown" for d in drivers):
            return None
        free = [machine["free_graphics_memory_bytes"], identity["available_ram_bytes"]]
        if any(type(v) is not int or v < 0 for v in free):
            return None
        if free[0] > machine["total_graphics_memory_bytes"]:
            return None
        return _hash({"model": (model["sha256"], model["size_bytes"]),
                      "context": identity["context_length"], "host": required,
                      "drivers": drivers, "power": machine["power_source"],
                      "total_vram": machine["total_graphics_memory_bytes"],
                      "free_memory_bands": [v // (256 * 1024**2) for v in free]})
    except (KeyError, TypeError, ValueError):
        return None


class HistoricalCosts:
    """One read-only snapshot; no SQLite, engine calls or acceptance decisions here."""

    def __init__(self, records: tuple[TuneHistoryRun, ...], identity: dict[str, Any], *,
                 origin: str, now: datetime | None = None, excluded_run_id: str | None = None) -> None:
        self.now = now or datetime.now(timezone.utc)
        self.key = envelope_key(identity)
        self.context = identity.get("context_length")
        self.runs: list[TuneHistoryRun] = []
        self.rejected: Counter[str] = Counter()
        seen: set[str] = set()
        for run in deepcopy(records):
            if run.run_id in seen:
                self.rejected["duplicate run"] += 1
                continue
            seen.add(run.run_id)
            if run.run_id == excluded_run_id:
                self.rejected["current run"] += 1
            elif len({o.operation_id for o in run.observations}) != len(run.observations):
                self.rejected["duplicate operation"] += 1
            elif origin not in ("hardware", "simulation") or run.origin != origin:
                self.rejected["different or unknown origin"] += 1
            elif run.started_at is None or run.started_at.tzinfo is None:
                self.rejected["missing date"] += 1
            elif not timedelta(0) <= self.now - run.started_at <= MAX_AGE:
                self.rejected["stale or future date"] += 1
            elif self.key is None or envelope_key(run.identity) != self.key:
                self.rejected["incomplete or incompatible host/model/conditions"] += 1
            else:
                self.runs.append(run)
        self.runs.sort(key=lambda r: (r.started_at, r.run_id), reverse=True)

    def estimate(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if request_identity_problems(request):
            return None
        if request["configuration"]["context_length"] != self.context:
            return None
        samples: list[tuple[str, float, int]] = []
        for run in self.runs:
            costs = [o.wall_s for o in run.observations
                     if o.kind == "screening" and o.status == "completed" and o.failure is None
                     and compatible_request(o.request, request, o.wall_s) and o.valid_runs is not None
                     and o.valid_runs >= request["policy"]["min_valid_samples"]
                     and o.retries == 0 and isinstance(o.phases.get("raw"), dict)
                     and o.phases["raw"].get("invalid_runs") == 0
                     and math.isfinite(o.wall_s) and o.wall_s > 0]
            if costs:
                samples.append((run.run_id, float(statistics.median(costs)), len(costs)))
            if len(samples) == MAX_RUNS:
                break
        if len(samples) < MIN_RUNS:
            return None
        values = sorted(v for _, v, _ in samples)
        return {"source": "history", "rule": "cost-prior-1", "request_sha256": _hash(request),
                "seconds": values[math.ceil(0.75 * len(values)) - 1],
                "run_ids": [r for r, _, _ in samples], "run_count": len(samples),
                "sample_count": sum(n for _, _, n in samples), "range_s": [values[0], values[-1]],
                "watchdog_match": "identical or total wall time below all recorded and requested limits",
                "maximum_age_days": MAX_AGE.days, "provisional": True}

    def summary(self) -> dict[str, Any]:
        return {"eligible_runs": len(self.runs), "rejected_runs": dict(self.rejected),
                "rule": "cost-prior-1", "minimum_runs": MIN_RUNS, "maximum_runs": MAX_RUNS,
                "maximum_age_days": MAX_AGE.days, "provisional": True}
