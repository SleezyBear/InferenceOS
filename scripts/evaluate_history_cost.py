"""C4-2 paired deterministic simulations; no hardware, model loads or real stores.

Three synthetic cost profiles x five targets. Each profile trains three independent
complete tunes in scratch, round-trips the actual observations through SQLite, then
runs the same evaluation world with/without history. These are engineered cases,
not a calibration of hardware benefit or a proof about path selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inferenceos.contracts.tune_learning import TuneHistoryRun
from inferenceos.tune.learning_store import TuneHistoryStore
from tests.test_budgeted_tune import Clock, _practical, _world
from tests.test_history_cost import identity, runtime


def run_case(root: Path, clock: Clock, records: tuple[TuneHistoryRun, ...], target: float,
             profile: str, *, training: bool = False) -> tuple[Any, Any, Any]:
    root.mkdir(parents=True, exist_ok=True)
    def extra(task: Any, options: Any) -> float:
        if profile == "uniform":
            return 0.0
        long = task.workload.workload_id.startswith("long")
        duration = (4.0 if (options.gpu_offload_layers or 0) < 20 else 0.0)
        duration += 8.0 if long and (options.ubatch_size or 512) <= 512 else 0.0
        if profile == "slower-now" and not training:
            duration += 3.0
        return duration
    def factor(layers: int, threads: int | None, batch: int | None, long: bool) -> float:
        return (1.2 if layers >= 20 else 1.0) * (1.15 if threads == 10 else 1.0) * (
            1.1 if long and (batch or 512) >= 1024 else 1.0)
    tuner, _, driver, _ = _practical(root, _world(clock, answer_s=3, long_answer_s=8,
                                               hang=extra, factor=factor), target, clock=clock)
    tuner.history_identity = identity()
    tuner.history_origin = "simulation"
    tuner.history_records = records
    tuner.history_runtime_reader = runtime
    tuner.history_runtime_valid = lambda observed: True
    return tuner, driver, driver.run()


def describe(tuner: Any, driver: Any, result: Any) -> dict[str, Any]:
    forecasts = []
    for event in driver._forecast_events:
        if event["source"] != "history":
            continue
        actual = next((o for o in tuner._learning.observations if o.kind == "screening"
                       and o.operation_id > event["before_operation"]
                       and o.request["configuration"] == event["configuration"]
                       and o.request["workload"] == event["request"]["workload"]
                       and o.request["policy"] == event["request"]["policy"]), None)
        forecasts.append({**event, "actual_s": None if actual is None else actual.wall_s,
                          "absolute_error_s": None if actual is None else abs(actual.wall_s - event["seconds"]),
                          "fallback_absolute_error_s": None if actual is None else abs(actual.wall_s - event["fallback_s"])})
    budget = next(r for r in result.rounds if r.get("round") == "time budget")
    final = [r for r in result.rounds if r.get("round") == "final check"]
    return {"outcome": result.outcome, "saved": result.saved_setup is not None,
            "chosen": result.final_configuration.model_dump(mode="json"),
            "used_s": budget["used_s"], "target_s": budget["target_s"], "overrun_s": budget["extension_s"],
            "steps": len(driver.steps), "screenings": len(driver.screenings),
            "confirmation_count": len(final), "confirmation": None if result.confirmation is None else result.confirmation.model_dump(mode="json"),
            "final_checks": final, "forecasts": forecasts,
            "current_request_forecasts": sum(e["source"] == "current request" for e in driver._forecast_events)}


def evaluate(out: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    clock = Clock()
    with tempfile.TemporaryDirectory(prefix="c4-2-cost-") as directory, patch("time.perf_counter", clock):
        root = Path(directory)
        for profile in ("uniform", "different-costs", "slower-now"):
            store = TuneHistoryStore(root / profile / "history")
            for n in range(3):
                tuner, driver, result = run_case(root / profile / f"training-{n}", clock, (), 15000, profile, training=True)
                record = TuneHistoryRun(run_id=result.run_id, started_at=datetime.now(timezone.utc),
                    origin="simulation", identity=identity(), observations=tuple(tuner._learning.observations),
                    comparisons=tuple(tuner._learning.comparisons), disposition={"outcome": result.outcome},
                    source_report_sha256=hashlib.sha256(result.to_json().encode("utf-8")).hexdigest())
                store.append(record)
                evidence.append({"training": profile, "record": record.model_dump(mode="json"),
                                 "report": result.model_dump(mode="json")})
            records = store.recent()
            for target in (800, 1600, 2200, 3000, 6000):
                pair: dict[str, Any] = {"profile": profile, "target_s": target}
                for enabled in (False, True):
                    tuner, driver, result = run_case(root / profile / f"{target}-{enabled}", clock,
                                                     records if enabled else (), target, profile)
                    key = "history" if enabled else "control"
                    pair[key] = describe(tuner, driver, result)
                    evidence.append({"case": f"{profile}/{target}/{key}", "report": result.model_dump(mode="json"),
                                     "observations": [o.model_dump(mode="json") for o in tuner._learning.observations]})
                rows.append(pair)
    paired_errors = [e for row in rows for e in row["history"]["forecasts"] if e["actual_s"] is not None]
    summary = {"origin": "deterministic simulation", "pairs": len(rows), "training_tunes": 9,
               "forecast_count_with_observed_cost": len(paired_errors),
               "history_mean_absolute_error_s": statistics.mean(e["absolute_error_s"] for e in paired_errors) if paired_errors else None,
               "fallback_mean_absolute_error_s": statistics.mean(e["fallback_absolute_error_s"] for e in paired_errors) if paired_errors else None,
               "saved_outcome_changes": sum(r["control"]["saved"] != r["history"]["saved"] for r in rows),
               "chosen_configuration_changes": sum(r["control"]["chosen"] != r["history"]["chosen"] for r in rows),
               "target_overruns": {k: sum(r[k]["overrun_s"] > 0 for r in rows) for k in ("control", "history")},
               "cases": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with gzip.open(out.with_suffix(".evidence.json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(evidence, stream, allow_nan=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    data = evaluate(parser.parse_args().out)
    print(json.dumps({k: v for k, v in data.items() if k != "cases"}, indent=2))
