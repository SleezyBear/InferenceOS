"""Practical observations, with explicit request identity for cost forecasting."""
from __future__ import annotations

import hashlib
import json
import time
import os
import platform
import math
import re
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
                     runtime_controls: dict[str, Any] | None = None,
                     watchdog_limits: dict[str, float] | None = None,
                     runtime_identity: dict[str, Any] | None = None) -> dict[str, Any]:
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
    if watchdog_limits is None:
        missing.append("effective per-answer watchdog limits")
    result = {"format_version": "tune-request-2", "configuration": config.model_dump(mode="json"),
            "workload": workload_identity(workload), "policy": None if policy is None else policy.model_dump(mode="json"),
            "disturbance": disturbance, "runtime_controls": deepcopy(runtime_controls), "missing": missing}
    if watchdog_limits is not None:
        result.update(format_version="tune-request-3", watchdog_limits=deepcopy(watchdog_limits),
                      runtime_identity=deepcopy(runtime_identity))
    return result


def request_identity_problems(request: dict[str, Any]) -> list[str]:
    """Conservative request gate, not a model/build/age compatibility decision.

    Older stored dictionaries are left untouched; no current defaults fill their
    missing controls. There is no history consumer in C4-1.
    """
    version = request.get("format_version")
    if version not in ("tune-request-2", "tune-request-3"):
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
    if version == "tune-request-2":
        problems.append("effective per-answer watchdog limits")
    else:
        from inferenceos.contracts.experiment import ExperimentPolicy, RuntimeConfiguration

        for name, contract in (("policy", ExperimentPolicy), ("configuration", RuntimeConfiguration)):
            value = request.get(name)
            try:
                # No defaults may fill an incomplete stored request.
                if not isinstance(value, dict) or set(value) != set(contract.model_fields):
                    raise ValueError("incomplete request")
                contract.model_validate(value)
            except ValueError:
                problems.append(f"missing or invalid {name}")
        workload = request.get("workload")
        if (not isinstance(workload, dict) or not workload.get("id")
                or not all(isinstance(workload.get(k), str) and re.fullmatch(r"[0-9a-f]{64}", workload[k])
                           for k in ("sha256", "prompt_sha256"))):
            problems.append("invalid workload identity")
        control = request.get("runtime_controls")
        numbers = ("watchdog_factor", "watchdog_grace_s", "resident_request_timeout_s", "resident_readiness_timeout_s")
        if (not isinstance(control, dict) or type(control.get("monitor_scoped")) is not bool
                or "measurement_deadline" not in control or control["measurement_deadline"] is not None
                or any(type(control.get(k)) not in (int, float) or not math.isfinite(control[k]) or control[k] <= 0
                       for k in numbers)):
            problems.append("invalid runtime controls")
        limits = request.get("watchdog_limits")
        if (not isinstance(limits, dict) or set(limits) != {"warmup_s", "answer_s"}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in limits.values())):
            problems.append("invalid watchdog limits")
        runtime = request.get("runtime_identity")
        if (not isinstance(runtime, dict) or runtime.get("version") != "loaded-runtime-1"
                or any(not isinstance(runtime.get(k), str) or not runtime[k].strip()
                       or runtime[k].strip().lower() == "unknown" for k in ("backend", "device"))
                or not isinstance(runtime.get("modules"), dict) or len(runtime["modules"]) < 2
                or runtime.get("executable") not in runtime["modules"]
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
                       for h in runtime["modules"].values())
                or not isinstance(runtime.get("bundle"), dict) or runtime.get("executable") not in runtime["bundle"]
                or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h)
                       for h in runtime["bundle"].values())):
            problems.append("missing or invalid loaded engine identity")
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
    # Per-request observed loaded modules supersede the run-wide unknown engine
    # fields for format-3 cost matching only. Legacy requests stay ineligible.
    from inferenceos.backends.engine_env import process_environment
    import psutil

    identity["os"] = platform.platform()
    identity["environment_sha256"] = _hash(process_environment(dict(os.environ), None))
    try:
        identity["available_ram_bytes"] = int(psutil.virtual_memory().available)
    except Exception:
        identity["available_ram_bytes"] = None
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


def model_identity_unchanged(identity: dict[str, Any]) -> bool:
    """Recheck the captured content identity's file observation before using priors."""
    try:
        model = identity["model"]
        stat = Path(model["path"]).stat()
        return bool(model.get("sha256")) and (stat.st_size, stat.st_mtime_ns) == (model["size_bytes"], model["modified_ns"])
    except (KeyError, OSError, TypeError):
        return False


@dataclass
class PracticalHistory:
    observations: list[MeasurementObservation] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    blocked_cleanup: bool = False

    def observe(self, operation_id: int, config: Any, workload: Any, policy: Any, wall_s: float,
                *, watch: Any = None, runtime_controls: dict[str, Any] | None = None,
                watchdog_limits: dict[str, float] | None = None,
                runtime_identity: dict[str, Any] | None = None, **details: Any) -> None:
        try:
            self.measurement(operation_id, request_identity(config, workload, policy, watch=watch,
                             runtime_controls=runtime_controls, watchdog_limits=watchdog_limits,
                             runtime_identity=runtime_identity), wall_s, **details)
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
                   aggregate_noise: float | None = None, goal: dict[str, Any] | None = None) -> None:
        try:
            self.comparisons.append({**deepcopy(record), "path": PRACTICAL_PATH_IDS[str(record["category"])],
                                     "raw_noise": noise, "raw_aggregate_noise": aggregate_noise,
                                     "hint_version": "candidate-hint-1", "goal": deepcopy(goal)})
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
