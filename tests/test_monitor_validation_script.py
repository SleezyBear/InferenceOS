"""The prepared hardware validation of the monitor's attribution
(``scripts/check_monitor_attribution.py``): its schedule, how it decides where a burst
really fell, its pass criteria, and an end-to-end plumbing smoke run with a simulated
perfect counter (the real counter and uncached reads only exist on the laptop)."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import pytest

from inferenceos.telemetry.attribution import PAGING_SPEC, RawReading

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_monitor_attribution.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("check_monitor_attribution_t", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_schedule_shape_and_regions() -> None:
    m = _load()
    windows, bursts = m.build_schedule(1000.0)
    counts: dict[str, int] = {}
    for w in windows:
        counts[w["placement"]] = counts.get(w["placement"], 0) + 1
    assert counts == {"quiet": 9, "inside": 9, "before": 4, "after": 4, "edge": 7}
    assert len(bursts) == 24
    offsets = sorted(b["offset"] for b in bursts)
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    assert min(gaps) == (2048 + 1024) * 4096  # fresh, separated regions
    assert offsets[-1] + 2048 * 4096 <= 512 << 20


@pytest.mark.parametrize(("began", "ended", "expected"), [
    (9.0, 9.2, "before"), (10.1, 10.2, "near"), (11.0, 11.2, "inside"), (12.95, 13.1, "edge"),
    (9.9, 10.1, "edge"), (13.3, 13.4, "after"), (12.8, 12.9, "near"),
])
def test_actual_placement(began: float, ended: float, expected: str) -> None:
    m = _load()
    window = {"start": 10.0, "end": 13.0, "actual_start": 10.0, "actual_end": 13.0}
    assert m.actual_placement(window, {"began": began, "ended": ended}) == expected


def _windows() -> list[dict[str, Any]]:
    return [{"id": f"w{i}", "placement": p, "start": 0.0, "end": 1.0}
            for i, p in enumerate(["quiet", "inside", "before", "after", "edge"])]


def _full() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, bool]]:
    """Windows meeting every coverage minimum, with right verdicts (edges unknown)."""
    plan = ["quiet"] * 3 + ["inside"] * 3 + ["before"] * 2 + ["after"] * 2 + ["edge"] * 2
    windows: list[dict[str, Any]] = [{"id": f"w{i}", "placement": p, "start": 0.0, "end": 1.0}
                                     for i, p in enumerate(plan)]
    verdicts: dict[str, dict[str, Any]] = {}
    for w in windows:
        p = w["placement"]
        verdicts[w["id"]] = ({"status": "unknown", "pages": None} if p == "edge" else
                             {"status": "measured", "pages": 2048 if p == "inside" else 0})
    return windows, verdicts, {w["id"]: True for w in windows}


def test_evaluate_passes_only_with_real_coverage() -> None:
    m = _load()
    windows, verdicts, confirmed = _full()
    ok, report = m.evaluate(windows, verdicts, confirmed, 2048)
    assert ok and report["outcome"] == "passed" and report["table"]["edge"]["unknown"] == 2


def test_evaluate_all_unknown_is_inconclusive_not_success() -> None:
    """Reviewer, monitor round 2: every answer unknown used to pass."""
    m = _load()
    windows, verdicts, confirmed = _full()
    unknown = {k: {"status": "unknown", "pages": None} for k in verdicts}
    ok, report = m.evaluate(windows, unknown, confirmed, 2048)
    print("\nall unknown ->", report["outcome"], report["shortfalls"])
    assert not ok and report["outcome"] == "inconclusive" and not report["failures"]
    assert len(report["shortfalls"]) == 4  # quiet, inside, before, after each lack usable evidence


def test_evaluate_needs_edges_really_straddled() -> None:
    m = _load()
    windows, verdicts, confirmed = _full()
    bursts = {w["id"]: {"began": 0.5, "ended": 0.6} for w in windows if w["placement"] == "inside"}
    bursts |= {w["id"]: {"began": -0.9, "ended": -0.8} for w in windows if w["placement"] == "before"}
    bursts |= {w["id"]: {"began": 1.6, "ended": 1.7} for w in windows if w["placement"] == "after"}
    # Scheduled at an edge, but finished 0.1 s before it: "near", not an exercised edge.
    bursts |= {w["id"]: {"began": -0.15, "ended": -0.1} for w in windows if w["placement"] == "edge"}
    ok, report = m.evaluate(windows, verdicts, confirmed, 2048, bursts)
    assert not ok and report["outcome"] == "inconclusive"
    assert any(s.startswith("edge: 0 confirmed") for s in report["shortfalls"]) and report["table"]["near"]["windows"] == 2


@pytest.mark.parametrize(("placement", "pages", "words"), [
    ("inside", 0, "hidden"), ("before", 2048, "wrong answer"), ("after", 2048, "wrong answer"),
    ("edge", 0, "no paging"), ("quiet", 2048, "quiet"),
])
def test_evaluate_fails_each_wrong_attribution(placement: str, pages: int, words: str) -> None:
    m = _load()
    windows, verdicts, confirmed = _full()
    wid = next(w["id"] for w in windows if w["placement"] == placement)
    verdicts[wid] = {"status": "measured", "pages": pages}
    ok, report = m.evaluate(windows, verdicts, confirmed, 2048)
    assert not ok and report["outcome"] == "failed" and words in report["failures"][0]


def test_evaluate_unconfirmed_bursts_are_inconclusive() -> None:
    m = _load()
    windows, verdicts, _ = _full()
    ok, report = m.evaluate(windows, verdicts, {}, 2048)
    assert not ok and report["outcome"] == "inconclusive" and "confirmed" in report["shortfalls"][-1]


def test_quick_run_keeps_its_reference_evidence_through_real_retention(tmp_path: Path,
                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end with the sampler's REAL retention (a RawCounterSampler subclass that
    only replaces the PowerShell process with a simulated counter): finalize trims the
    sampler's history, yet every burst is still confirmed from the readings the answers
    were judged on. The quick plan (one window per placement) is too small to prove
    anything, so the outcome must be inconclusive -- never a success."""
    import threading

    import inferenceos.telemetry.counters as counters
    from inferenceos.telemetry.counters import RawCounterSampler

    m = _load()
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.setattr(m.tempfile, "mkdtemp", lambda prefix="": str(workdir))
    monkeypatch.setattr(m, "write_uncached_file", lambda path, size: path.write_bytes(b"\x5a" * size))
    made: list[Any] = []

    class SimulatedCounter(RawCounterSampler):
        """Real reading storage and real discard_before; a thread instead of PowerShell."""

        def __init__(self, specs: Any, interval_s: float = 0.1) -> None:
            super().__init__(specs, interval_s)
            self._halt = threading.Event()
            made.append(self)

        def start(self) -> bool:
            def tick() -> None:
                log_path = workdir / "bursts.json"
                while not self._halt.is_set():
                    try:
                        log = json.loads(log_path.read_text()) if log_path.exists() else []
                    except ValueError:
                        log = []
                    t = time.time()
                    raw = sum(e["pages"] for e in log if e["ended"] < t)
                    with self._lock:
                        self._readings.append(RawReading(spec=PAGING_SPEC, read_before=t, read_after=t + 0.001,
                                                         raw=raw, base=0, ts100=int(t * 1e7), counter_type=272696320,
                                                         calculated=None, arrival_wall=t + 0.002, arrival_perf=0.0))
                    time.sleep(0.05)

            self._thread = threading.Thread(target=tick, daemon=True)
            self._thread.start()
            return True

        def stop(self) -> dict[str, object]:
            self._halt.set()
            assert self._thread is not None
            self._thread.join(2.0)
            return {"exit_confirmed": True, "reader_joined": True, "pipe_closed": True, "resolved": True, "errors": []}

    real = counters.MeasurementMonitor
    monkeypatch.setattr(counters, "MeasurementMonitor",
                        lambda paths, interval_s=0.1, **k: real(paths, interval_s,
                                                                sampler_factory=SimulatedCounter, **k))
    out = tmp_path / "attr.json"
    code = m.main(["--out", str(out), "--run-id", "rid-1", "--pages", "64", "--scratch-mb", "16", "--quick"])
    data = json.loads(out.read_text())
    kept_after = len(made[0].readings())
    print(f"\nexit {code} | {data['status']} | readings judged {data['readings_judged']}, kept by the sampler "
          f"after retention {kept_after} | bursts confirmed {data['evaluation']['bursts_confirmed']}")
    assert kept_after < data["readings_judged"] / 4  # retention really trimmed the history
    assert all(b["confirmed"] for b in data["bursts"])  # ...yet the reference check still had its readings
    assert code == 2 and data["status"].startswith("inconclusive:") and data["run_id"] == "rid-1"
    assert data["evaluation"]["outcome"] == "inconclusive" and not data["evaluation"]["failures"]
