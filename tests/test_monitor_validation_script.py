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


@pytest.mark.parametrize(("placement", "pages", "words", "reads"), [
    ("inside", 0, "hidden", 2048), ("before", 2048, "wrong answer", None), ("after", 2048, "wrong answer", None),
    ("edge", 0, "hidden", 2048), ("quiet", 2048, "quiet", None),
])
def test_evaluate_fails_each_wrong_attribution(placement: str, pages: int, words: str, reads: int | None) -> None:
    """A hidden burst is a failure only with a burst-specific read total (reviewer,
    after the bounds patch); paging credited to the wrong answer needs none."""
    m = _load()
    windows, verdicts, confirmed = _full()
    wid = next(w["id"] for w in windows if w["placement"] == placement)
    verdicts[wid] = {"status": "measured", "pages": pages}
    burst_reads = {wid: reads} if reads is not None else None
    ok, report = m.evaluate(windows, verdicts, confirmed, 2048, burst_reads=burst_reads)
    assert not ok and report["outcome"] == "failed" and words in report["failures"][0]


@pytest.mark.parametrize("placement", ["inside", "edge"])
def test_a_burst_seemingly_hidden_without_a_burst_specific_total_is_only_suspected(placement: str) -> None:
    """Without a burst-specific read total nothing says the touched pages were read from
    disk: a reading below the touches is suspected, never a failure, never a pass."""
    m = _load()
    windows, verdicts, confirmed = _full()
    wid = next(w["id"] for w in windows if w["placement"] == placement)
    verdicts[wid] = {"status": "measured", "pages": 0}
    ok, report = m.evaluate(windows, verdicts, confirmed, 2048)
    print(f"\n{placement} 0 pages, no burst-specific total -> {report['outcome']}: {report['shortfalls'][-1][:90]}")
    assert not ok and report["outcome"] == "inconclusive" and not report["failures"]
    assert "suspected hidden" in report["shortfalls"][-1]


def test_evaluate_unconfirmed_bursts_are_inconclusive() -> None:
    m = _load()
    windows, verdicts, _ = _full()
    ok, report = m.evaluate(windows, verdicts, {}, 2048)
    assert not ok and report["outcome"] == "inconclusive" and "confirmed" in report["shortfalls"][-1]


def simulated_reading(log_path: Path, t: float) -> RawReading | None:
    """The simulated paging counter at ``t``: pages of every burst chunk the generator
    has logged as finished before ``t``. ``None`` when the log cannot be read completely
    -- the generator rewrites it in place, so a read can catch it empty or half written.
    That sample is skipped: a cumulative counter must never read a fabricated zero (it
    would look like the count went backwards, then like a burst of every page so far)."""
    if not log_path.exists():
        return _counter_reading(t, 0)  # the generator has not logged anything yet: nothing finished
    try:
        log = json.loads(log_path.read_text())
        raw = sum(c["pages"] for e in log for c in e.get("chunks") or [e]
                  if c["ended"] is not None and c["ended"] < t)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    return _counter_reading(t, raw)


def _counter_reading(t: float, raw: int) -> RawReading:
    return RawReading(spec=PAGING_SPEC, read_before=t, read_after=t + 0.001, raw=raw, base=0,
                      ts100=int(t * 1e7), counter_type=272696320, calculated=None,
                      arrival_wall=t + 0.002, arrival_perf=0.0)


def test_an_incomplete_log_read_is_skipped_not_read_as_zero(tmp_path: Path) -> None:
    """The flaky case, made deterministic: the simulated counter read the generator's
    log while it was being rewritten (empty or half written) and fell back to an empty
    log, i.e. a reading of 0 pages after earlier readings of 1,024 -- a counter going
    backwards, which the attribution then sees as a reset or as a fresh burst."""
    log_path = tmp_path / "bursts.json"
    complete: list[dict[str, Any]] = [{"id": "b1", "chunks": [{"pages": 512, "began": 9.0, "ended": 9.5},
                                        {"pages": 512, "began": 9.5, "ended": 10.0}]},
                {"id": "b2", "pages": 256, "began": 10.5, "ended": None}]
    assert simulated_reading(log_path, 9.0) is not None  # no log yet: truly nothing finished
    text = json.dumps(complete)
    log_path.write_text(text)
    first = simulated_reading(log_path, 10.2)
    assert first is not None and first.raw == 1024
    raws = [first.raw]
    for partial in ("", text[: len(text) // 2], text[:-1], "[{\"id\": \"b1\"}]"):
        log_path.write_text(partial)  # caught mid-rewrite (or a record without its pages yet)
        assert simulated_reading(log_path, 10.3) is None, partial
    log_path.write_text(json.dumps(complete[:1] + [{**complete[1], "ended": 10.6}]))
    later = simulated_reading(log_path, 10.7)
    assert later is not None and later.raw == 1280
    raws.append(later.raw)
    print(f"\nreadings kept {raws}; 4 incomplete reads skipped")
    assert raws == sorted(raws)  # never backwards


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
                    reading = simulated_reading(log_path, time.time())
                    if reading is not None:  # an incomplete log read is skipped, never a zero count
                        with self._lock:
                            self._readings.append(reading)
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
    # Evidence for an exact replay (reviewer, after job 121): every judged reading, unrounded,
    # and each answer's wall-clock and perf-counter brackets; each burst's chunk clock reads.
    raw = data["raw_readings"]
    assert len(raw) == data["readings_judged"] and {"read_before", "read_after", "raw", "ts100"} <= set(raw[0])
    assert all(isinstance(w["perf_start"], float) and w["actual_end"] > w["actual_start"] for w in data["windows"])
    edge = next(b for b in data["bursts"] if b["id"] == next(w["id"] for w in data["windows"]
                                                              if w["placement"] == "edge"))
    assert len(edge["chunks"]) == 8 and all(c["ended"] >= c["began"] for c in edge["chunks"])
    verdicts = data["verdicts"].values()
    assert {v["evidence"] for v in verdicts} <= {"exact", "bounded", "unknown"}


# --------------------------------------------- reviewer, after job 121: bounded evidence
def _bounded(low: int, high: int) -> dict[str, Any]:
    return {"status": "measured", "evidence": "exact" if low == high else "bounded", "pages": low if low == high else None,
            "pages_lower": low, "pages_upper": high}


@pytest.mark.parametrize(("placement", "low", "high", "counts"), [
    ("quiet", 0, 78, "usable"),         # job 121's w25: certainly not a burst's worth
    ("before", 0, 300, "usable"),
    ("inside", 2040, 2300, "usable"),   # the burst certainly credited
    ("inside", 500, 2300, "inconclusive"),  # may or may not include the burst: proves nothing
    ("after", 300, 1500, "inconclusive"),
])
def test_bounded_evidence_counts_only_when_it_settles_the_question(placement: str, low: int, high: int,
                                                                   counts: str) -> None:
    m = _load()
    windows, verdicts, confirmed = _full()
    wid = next(w["id"] for w in windows if w["placement"] == placement)
    verdicts[wid] = _bounded(low, high)
    ok, report = m.evaluate(windows, verdicts, confirmed, 2048)
    row = report["table"][placement]
    assert row["bounded"] == 1 and not report["failures"]
    assert (row["usable"], row["inconclusive"]) == ((REQUIRED[placement], 0) if counts == "usable"
                                                     else (REQUIRED[placement] - 1, 1))
    assert ok is (counts == "usable")  # a window that does not settle leaves its placement short


REQUIRED = {"quiet": 3, "inside": 3, "before": 2, "after": 2}


@pytest.mark.parametrize(("placement", "low", "high", "words", "reads"), [
    ("before", 1500, 2048, "wrong answer", None),  # certainly credited to the wrong answer
    ("inside", 0, 500, "hidden", 2048),            # cannot contain the burst's certain reads
    ("edge", 0, 600, "hidden", 2048),              # at least half the edge burst was read inside
])
def test_bounded_evidence_that_misses_the_truth_fails(placement: str, low: int, high: int, words: str,
                                                      reads: int | None) -> None:
    m = _load()
    windows, verdicts, confirmed = _full()
    wid = next(w["id"] for w in windows if w["placement"] == placement)
    verdicts[wid] = _bounded(low, high)
    burst_reads = {wid: reads} if reads is not None else None
    ok, report = m.evaluate(windows, verdicts, confirmed, 2048, burst_reads=burst_reads)
    assert not ok and report["outcome"] == "failed" and words in report["failures"][0]


def test_none_is_never_read_as_zero() -> None:
    m = _load()
    windows, verdicts, confirmed = _full()
    for w in windows:
        if w["placement"] in ("quiet", "before"):
            verdicts[w["id"]] = {"status": "measured", "pages": None}  # no count at all
    ok, report = m.evaluate(windows, verdicts, confirmed, 2048)
    assert not ok and report["outcome"] == "inconclusive"
    assert report["table"]["quiet"]["unknown"] == 3 and report["table"]["quiet"]["usable"] == 0


def _edge_burst() -> tuple[dict[str, Any], dict[str, Any]]:
    window = {"id": "w", "placement": "edge", "start": 10.0, "end": 12.0, "actual_start": 10.0, "actual_end": 12.0}
    chunks = [{"began": 9.8 + 0.057 * k, "ended": 9.8 + 0.057 * k + 0.004, "pages": 256} for k in range(8)]
    burst = {"began": 9.8, "ended": chunks[-1]["ended"], "pages": 2048, "chunks": chunks,
             "counter_pages_across_burst": 2048}
    return window, burst


def test_edge_truth_comes_from_the_chunk_clock_reads() -> None:
    m = _load()
    window, burst = _edge_burst()
    truth = m.burst_truth(window, burst, 2048)
    # Chunks 4..7 (1,024 pages) began after 10.0. With no burst-specific read total, no
    # positive share is certain; the touches bound it from above.
    assert (truth["low"], truth["high"], truth["premise_low"]) == (0, 1024, 1024)
    assert "no burst-specific read total" in truth["basis"]
    # A trustworthy burst-specific total: every page read -> the certain share is 1,024.
    assert m.burst_truth(window, burst, 2048, 2048)["low"] == 1024


def test_the_reviewers_counterexample_gives_no_certain_interior_reads() -> None:
    """1,024 pages touched before the answer and 1,024 inside it, 1,024 disk reads in
    all: every read may have come before the answer (the interior touches served from
    memory). The old proportional split claimed 512 certainly inside; it is 0."""
    m = _load()
    window, burst = _edge_burst()
    burst["counter_pages_across_burst"] = 1024  # system-wide: not burst-specific, ignored
    truth = m.burst_truth(window, burst, 2048)
    assert truth["low"] == 0 and truth["high"] == 1024
    # Even if that 1,024 WERE a burst-specific total, the deficit goes to the inside chunks.
    adverse = m.burst_truth(window, burst, 2048, 1024)
    print(f"\nreviewer's counterexample: certain inside {adverse['low']} (was 512), possible {adverse['high']}")
    assert adverse["low"] == 0 and adverse["high"] == 1024
    assert m.burst_truth(window, burst, 2048, 1536)["low"] == 512  # 512 reads cannot be placed outside
