"""Paired C4-3 simulation with actual stored training runs; no hardware or real stores.

The useful coarse value is deliberately third in the default order. Compare the
same cost-history-enabled tune with ordering disabled, explicitly enabled for
development, and in the default shadow mode, isolating C4-3.
Reversed evaluation performance checks that old preferences never replace evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
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
             *, ordering: bool | None = None, reversed_world: bool = False,
             slower_drift: bool = False) -> tuple[Any, Any, Any]:
    root.mkdir(parents=True, exist_ok=True)
    def factor(layers: int, threads: int | None, batch: int | None, long: bool) -> float:
        if reversed_world:
            return 0.8 if layers == 1 else (1.2 if layers >= 20 else 1.0)
        return 1.5 if layers == 1 else (1.1 if layers >= 20 else 1.0)
    world = _world(clock) if slower_drift else _world(clock, answer_s=5, long_answer_s=10, factor=factor)
    tuner, _, driver, _ = _practical(root, world, target, clock=clock)
    tuner.history_identity, tuner.history_origin = identity(), "simulation"
    tuner.history_records = records
    tuner.history_runtime_reader = runtime
    tuner.history_runtime_valid = lambda observed: True
    if ordering is True:
        driver.apply_candidate_hints = True  # explicit development-only evaluation
    elif ordering is False:
        driver._candidate_order = lambda step: step.values  # type: ignore[method-assign]
    return tuner, driver, driver.run()


def train(root: Path, clock: Clock, reports: list[dict[str, Any]] | None = None) -> tuple[TuneHistoryRun, ...]:
    store = TuneHistoryStore(root / "history")
    for n in range(3):
        tuner, driver, result = run_case(root / f"train-{n}", clock, (), 12000)
        report_json = result.to_json()
        if reports is not None:
            reports.append({"run_id": result.run_id, "report_json": report_json})
        store.append(TuneHistoryRun(run_id=result.run_id, started_at=datetime.now(timezone.utc), origin="simulation",
            identity=identity(), observations=tuple(tuner._learning.observations),
            comparisons=tuple(tuner._learning.comparisons),
            disposition={"outcome": result.outcome, "saved": result.saved_setup is not None},
            source_report_sha256=hashlib.sha256(report_json.encode()).hexdigest()))
    return store.recent()


def describe(tuner: Any, driver: Any, result: Any) -> dict[str, Any]:
    return {"outcome": result.outcome, "saved": result.saved_setup is not None,
            "configuration": result.final_configuration.model_dump(mode="json"),
            "steps": driver.steps, "screenings": driver.screenings, "hints": driver._candidate_hint_events,
            "final_checks": [r for r in result.rounds if r.get("round") == "final check"],
            "confirmation": None if result.confirmation is None else result.confirmation.model_dump(mode="json"),
            "budget": next(r for r in result.rounds if r.get("round") == "time budget"),
            "operations": [o.model_dump(mode="json") for o in tuner._learning.observations],
            "report": result.model_dump(mode="json")}


def evaluate(out: Path) -> dict[str, Any]:
    clock = Clock()
    evidence: dict[str, Any] = {"origin": "simulation; synthetic identity; real temporary history store", "cases": []}
    with tempfile.TemporaryDirectory(prefix="c4-3-hints-") as directory, patch("time.perf_counter", clock):
        root = Path(directory)
        evidence["training_reports"] = []
        records = train(root, clock, evidence["training_reports"])
        evidence["training"] = [r.model_dump(mode="json") for r in records]
        for reversed_world in (False, True):
            for target in (700, 1100, 2200, 6000):
                row: dict[str, Any] = {"target": target, "reversed_world": reversed_world}
                for ordering in (False, True, None):
                    t, d, result = run_case(root / f"eval-{reversed_world}-{target}-{ordering}", clock, records,
                                           target, ordering=ordering, reversed_world=reversed_world)
                    row["shadow" if ordering is None else "history" if ordering else "control"] = describe(t, d, result)
                evidence["cases"].append(row)
        # Retain the adverse cases discovered while adapting the old tight-target
        # tests: the old favourite is flat now, and fresh answers take longer.
        for target in (760, 800):
            row = {"target": target, "reversed_world": False, "slower_drift": True}
            for ordering in (False, True, None):
                t, d, result = run_case(root / f"drift-{target}-{ordering}", clock, records,
                                       target, ordering=ordering, slower_drift=True)
                row["shadow" if ordering is None else "history" if ordering else "control"] = describe(t, d, result)
            evidence["cases"].append(row)
    summary: dict[str, Any] = {"pairs": len(evidence["cases"]), "training_runs": len(evidence["training"]), "cases": []}
    for row in evidence["cases"]:
        item = {k: row[k] for k in ("target", "reversed_world")}
        item["slower_drift"] = row.get("slower_drift", False)
        for side in ("control", "history", "shadow"):
            run = row[side]
            item[side] = {"first_values": run["steps"][0]["values"], "hint_steps": len(run["hints"]),
                "saved": run["saved"], "configuration": run["configuration"],
                "confirmation_count": len(run["final_checks"]), "outcome": run["outcome"],
                "used_s": run["budget"]["used_s"], "overrun_s": run["budget"]["extension_s"],
                "preferred_candidate_operation": next((e["candidate"]["measurement"] for e in run["screenings"]
                    if e["category"] == "placement" and e["value"] == 1), None)}
        summary["cases"].append(item)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with gzip.open(out.with_suffix(".evidence.json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(evidence, stream, allow_nan=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    result = evaluate(parser.parse_args().out)
    print(json.dumps(result, indent=2))
