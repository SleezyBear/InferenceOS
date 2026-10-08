"""Reviewer follow-up, item 2: a probe case whose phase is never observed and whose worker
survives its bounded wait blocks everything after it.

Runs the real ``scripts/check_verify_cost.py`` ``main()`` against a scripted engine; the
first case's worker is made to hang inside the model load (before the backend ever
reports "starting"), so the loading phase is never observed and the worker outlives
``WORKER_S``. Proves: the case is blocked, recovery is never entered, no later case
runs, the results file is written, and the exit status is 2.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_verify_cost.py"


def _load_probe() -> Any:
    spec = importlib.util.spec_from_file_location("check_verify_cost_survivor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unobserved_phase_with_surviving_worker_blocks_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inferenceos.backends.llama_cpp_resident_adapter as lra
    import inferenceos.models.loader as loader
    import inferenceos.profiling.runner as runner
    import inferenceos.runtime.resident_workload as rw
    import inferenceos.telemetry.system as system
    import inferenceos.tune.engine_probe as engine_probe
    from inferenceos.selftest import sample_hardware
    from tests.resident_helpers import synthetic_model
    from tests.test_measurement_monitor import _Platform

    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"\x00")
    monkeypatch.setattr(lra, "resolve_server_executable", lambda d: Path("unused"))
    monkeypatch.setattr(loader, "load_model_profile", lambda p: synthetic_model(str(model_file)))
    monkeypatch.setattr(runner, "get_platform", lambda: _Platform())
    monkeypatch.setattr(system, "get_platform", lambda: _Platform())
    monkeypatch.setattr(runner, "detect_hardware_profile", lambda s: sample_hardware())
    monkeypatch.setattr(engine_probe, "wait_until_gone", lambda identity, timeout_s=20.0: True)

    hang = threading.Event()
    entered_load = threading.Event()

    def hanging_load(self: Any, model: Any, params: Any) -> None:
        entered_load.set()
        hang.wait(30.0)  # survives WORKER_S; released at the end of the test
        raise RuntimeError("released")

    monkeypatch.setattr(lra.LlamaCppResidentAdapter, "load", hanging_load)
    calls: dict[str, int] = {"allow_restart": 0, "interrupt_cases": 0}
    real_allow = rw.ResidentActiveWorkload.allow_restart

    def counting_allow(self: Any) -> None:
        calls["allow_restart"] += 1
        real_allow(self)

    monkeypatch.setattr(rw.ResidentActiveWorkload, "allow_restart", counting_allow)

    probe = _load_probe()
    probe.PHASE_WAIT_S = 0.5
    probe.WORKER_S = 0.5
    out = tmp_path / "probe.json"
    try:
        code = probe.main(["--model", str(model_file), "--layers", "1", "--interrupt-rounds", "3",
                           "--rounds", "1", "--out", str(out)])
    finally:
        hang.set()
    data = json.loads(out.read_text())
    rows = data["interruptions"]
    print("\nexit:", code, "| status:", data["status"])
    print("rows:", json.dumps([{k: r.get(k) for k in ("label", "classification", "worker_returned", "recovered", "blocked")}
                               for r in rows]))
    print("allow_restart calls:", calls["allow_restart"], "| final_stop:", data.get("final_stop"))
    assert entered_load.is_set()
    assert code == 2
    assert len(rows) == 1, "a later case ran"
    row = rows[0]
    assert row["label"] == "C-loading-1" and row["classification"] == "phase not exercised"
    assert row["worker_returned"] is False and "did not return" in row["blocked"]
    assert row["recovered"] is None and "recovery_s" not in row  # recovery never entered
    assert calls["allow_restart"] == 0
    assert data["status"].startswith("failed: C failed:") and data["measurements"] == []
    assert data["parts"]["C"].startswith("failed:")
    assert all(data["parts"][p] == "not run: Part C did not pass" for p in ("A0", "A", "B"))
    assert data["final_stop"] == "ok" and data["parts"]["cleanup"] == "passed"
