"""Practical observation collector. No history is used to make decisions in C4-1."""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from inferenceos.contracts.tune import TuneResult
from inferenceos.contracts.tune_learning import DisturbanceControls, MeasurementObservation, TuneHistoryRun
from inferenceos.tune.coordinator import PRACTICAL_PATH_IDS
from inferenceos.tune.learning_store import TuneHistoryStore


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False, default=lambda v: v.model_dump(mode="json")).encode("utf-8")).hexdigest()


def workload_identity(workload: Any) -> dict[str, Any]:
    full = asdict(workload)
    # Benchmark enum values are strings; include every field, not just workload_id.
    return {"id": workload.workload_id, "sha256": _hash(full), "prompt_sha256": workload.prompt_hash}


def request_identity(config: Any, workload: Any, policy: Any, *, watch: Any = None,
                     runtime_controls: dict[str, Any] | None = None) -> dict[str, Any]:
    disturbance = None if watch is None else DisturbanceControls(
        logical_cores=max(int(watch.logical_cores or 1), 1), limit_percent=watch.limit_percent,
        max_repeats=max(watch.max_repeats, 0)).model_dump(mode="json")
    missing = []
    if policy is None:
        missing.append("measurement policy")
    if disturbance is None:
        missing.append("disturbance controls")
    if runtime_controls is None:
        missing.append("runtime controls")
    # The watchdog derives individual deadlines from the live ledger. Recording its
    # settings alone cannot reconstruct those limits. Never upgrade old data later.
    missing.append("effective per-answer watchdog limits")
    return {"format_version": "tune-request-2", "configuration": config.model_dump(mode="json"),
            "workload": workload_identity(workload), "policy": None if policy is None else policy.model_dump(mode="json"),
            "disturbance": disturbance, "runtime_controls": deepcopy(runtime_controls), "missing": missing}


def request_identity_problems(request: dict[str, Any]) -> list[str]:
    """Conservative request gate, not a model/build/age compatibility decision.

    Older stored dictionaries are left untouched; no current defaults fill their
    missing controls. There is no history consumer in C4-1.
    """
    if request.get("format_version") != "tune-request-2":
        return ["unsupported or legacy request identity"]
    missing = request.get("missing")
    problems = (list(missing) if isinstance(missing, list) and all(isinstance(v, str) for v in missing)
                else ["invalid completeness metadata"])
    controls = request.get("disturbance")
    try:
        if not isinstance(controls, dict) or "rule_version" not in controls:
            raise ValueError("unversioned disturbance controls")
        DisturbanceControls.model_validate(controls)
    except ValueError:
        problems.append("missing or invalid disturbance controls")
    if request.get("policy") is None:
        problems.append("measurement policy")
    if request.get("runtime_controls") is None:
        problems.append("runtime controls")
    # This format does not yet record the actual dynamically chosen limits.
    problems.append("effective per-answer watchdog limits")
    return list(dict.fromkeys(problems))


def capture_identity(model_path: Path, machine: Any, context_length: int, settings: Any) -> dict[str, Any]:
    """No engine starts. Unknown build dependencies remain explicitly unverified.

    KeyboardInterrupt is not swallowed; hashing is outside the timed search. A later
    stat mismatch invalidates the recorded model identity rather than blessing it.
    """
    from inferenceos.tune.profiles import model_identity

    began = time.perf_counter()
    identity: dict[str, Any] = {"context_length": context_length, "machine": machine.model_dump(mode="json"),
                               "engine_bundle_verified": False,
                               "missing": ["engine bundle/dependency identity", "effective backend/device identity"],
                               "monitor_scoped": bool(settings.tune_experimental_measurement_monitor)}
    try:
        before = model_path.stat()
        model = model_identity(model_path).model_dump(mode="json")
        after = model_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise OSError("model changed while hashing")
        identity["model"] = model
    except (OSError, ValueError) as exc:
        identity["missing"].append(f"model content identity: {type(exc).__name__}")
    identity["capture_s"] = time.perf_counter() - began
    return identity


@dataclass
class PracticalHistory:
    observations: list[MeasurementObservation] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    blocked_cleanup: bool = False

    def observe(self, operation_id: int, config: Any, workload: Any, policy: Any, wall_s: float,
                *, watch: Any = None, runtime_controls: dict[str, Any] | None = None, **details: Any) -> None:
        try:
            self.measurement(operation_id, request_identity(config, workload, policy, watch=watch,
                             runtime_controls=runtime_controls), wall_s, **details)
        except Exception as exc:
            self.diagnostics.append(f"operation {operation_id} request unavailable: {type(exc).__name__}")

    def measurement(self, operation_id: int, request: dict[str, Any], wall_s: float, *,
                    kind: Literal["screening", "final-side"] = "screening", valid_runs: int | None = None,
                    status: Literal["completed", "failed", "incomplete"] = "completed",
                    retries: int | None = None, phases: dict[str, Any] | None = None,
                    failure: dict[str, str] | None = None, measurement_notes: tuple[str, ...] = ()) -> None:
        try:
            self.observations.append(MeasurementObservation(operation_id=operation_id, kind=kind,
                request=deepcopy(request), wall_s=wall_s, status=status, valid_runs=valid_runs,
                retries=retries, phases=deepcopy(phases or {}), failure=deepcopy(failure),
                measurement_notes=measurement_notes))
        except Exception as exc:
            self.diagnostics.append(f"operation {operation_id} not recorded: {type(exc).__name__}")

    def comparison(self, record: dict[str, Any], *, noise: float | None = None,
                   aggregate_noise: float | None = None) -> None:
        try:
            self.comparisons.append({**deepcopy(record), "path": PRACTICAL_PATH_IDS[str(record["category"])],
                                     "raw_noise": noise, "raw_aggregate_noise": aggregate_noise})
        except Exception as exc:
            self.diagnostics.append(f"comparison not recorded: {type(exc).__name__}")

    def record(self, result: TuneResult, report_bytes: bytes, *, identity: dict[str, Any] | None = None,
               origin: Literal["hardware", "simulation", "unknown"] = "unknown") -> TuneHistoryRun:
        envelope = deepcopy(identity or {"missing": ["identity not captured"]})
        model = envelope.get("model")
        if model:
            try:
                stat = Path(model["path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (model["size_bytes"], model["modified_ns"]):
                    raise OSError("model changed during tune")
            except OSError:
                envelope.setdefault("missing", []).append("model changed or unavailable after tune")
                envelope["model"] = {**model, "sha256": None}
        # Full final disposition is retained separately; screening never implies saved.
        disposition = {"outcome": result.outcome, "starting": result.starting_configuration.model_dump(mode="json"),
            "final": result.final_configuration.model_dump(mode="json"),
            "saved": result.saved_setup is not None,
            "saved_configuration": None if result.saved_setup is None else result.saved_setup.configuration.model_dump(mode="json"),
            "decisions": [d.model_dump(mode="json") for d in result.decisions],
            "confirmation": None if result.confirmation is None else result.confirmation.model_dump(mode="json"),
            "checks": deepcopy([r for r in result.rounds if r.get("round") == "final check"]),
            "not_reached": list(result.not_reached)}
        return TuneHistoryRun(run_id=result.run_id, started_at=result.started_at, origin=origin, identity=envelope,
            observations=tuple(self.observations), comparisons=tuple(self.comparisons), disposition=disposition,
            source_report_sha256=hashlib.sha256(report_bytes).hexdigest(), diagnostics=tuple(self.diagnostics))


def persist_completed(tuner: Any, result: TuneResult, report_path: Path, *,
                      identity: dict[str, Any] | None = None,
                      origin: Literal["hardware", "simulation", "unknown"] = "unknown") -> str:
    """Called only after existing cleanup and primary report publication.

    Returns a separate diagnostic: history failure cannot change a saved tune's outcome.
    """
    collector = tuner._learning
    if collector is None:
        return "not applicable"
    if collector.blocked_cleanup:
        return "not written: monitor cleanup unresolved"
    try:
        raw = report_path.read_bytes()
        published = TuneResult.model_validate_json(raw)
        if published != result:
            raise ValueError("published report does not match this tune")
        record = collector.record(result, raw, identity=identity, origin=origin)
        added = TuneHistoryStore(tuner.settings.data_dir).append(record)
        return "recorded" if added else "already recorded"
    except Exception as exc:
        return f"not written: {type(exc).__name__}: {exc}"
