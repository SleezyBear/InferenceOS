"""Hardware validation of the measurement-scoped monitor's attribution (step 1 of jobs
121 and ``_jobs/later/122_boundary_and_model_paging.ps1``; 122 is prepared, not
queued). No model is loaded.

Independent reference. A separate process causes hard page faults at scheduled times:
it memory-maps a scratch file that was written with ``FILE_FLAG_NO_BUFFERING |
FILE_FLAG_WRITE_THROUGH`` (so its pages are not in the file cache) and touches one byte
in each 4 KiB page of a fresh region. It logs, with its own clock reads, when each
burst really began and ended and how many pages it touched. The burst's page reads are
also confirmed from the raw counter itself (the count rose by at least half the pages
touched across the burst), so a burst that was satisfied from memory is reported, not
used.

Pseudo-answers. Answer windows of 2 s (and some of 6 s) are timed exactly as the
profiler times an answer (``time.time()`` / ``perf_counter()`` at both ends) and noted
in a real :class:`MeasurementMonitor`; the windows are finalized together at the end,
like a measurement's answers. Each window gets one placement of a burst:

* ``quiet``  -- no burst;
* ``inside`` -- a burst in the middle of the window;
* ``before`` -- a burst ending 0.5 s before the window starts;
* ``after``  -- a burst starting 0.5 s after the window ends;
* ``edge``   -- a burst spread over 0.4 s from 0.2 s before the window's start.

Evidence. Each answer's verdict is ``exact`` (its page count is known), ``bounded``
(valid evidence, but the count is only known to lie in ``pages_lower .. pages_upper``,
e.g. pages in an interval crossing the answer's edge) or ``unknown`` -- ``None`` is
never read as zero. What the burst's disk reads inside each answer can be comes from
the generator's own clock reads around every chunk it touched (``burst_truth``):
``truth_upper`` = pages of chunks that possibly fell inside (a read needs a touch);
``truth_lower`` = 0 unless a trustworthy **burst-specific** read total exists, in which
case any shortfall is assigned adversarially to the chunks inside -- a touched page may
have been served from memory, and nothing says cached pages were spread evenly. The
system-wide counter's delta across a burst includes unrelated paging, so it is not such
a total (none exists today). ``premise_low`` = pages of chunks certainly inside: the
share if every touch was a disk read (the uncached scratch file's premise, unproven).

Outcome, with ``N`` = pages per burst, a tolerance of N/2 for background paging and
N/8 for hiding; placements are where each burst really fell, and only
bursts confirmed by the counter count.

**failed** if any attribution is wrong -- the reported bounds miss the truth:

* ``pages_upper`` below ``truth_lower - N/8`` (paging hidden; needs a burst-specific
  read total);
* ``pages_lower`` at or above ``truth_upper + N/2`` (paging attributed to the wrong
  answer: ``before`` / ``after`` / ``quiet``, or an ``edge`` burst over-counted);
* or the cleanup is unresolved.

A window whose ``pages_upper`` is below ``premise_low - N/8`` is a **suspected** hidden
burst: never a failure (the premise is unproven), never a pass -- it makes the outcome
inconclusive.

**inconclusive** (not a success) when nothing was wrong but the placements were not
really exercised with usable evidence -- evidence that settles the question, exact or
bounded: fewer than 3 ``quiet``, 2 ``before`` and 2 ``after`` windows whose
``pages_upper`` is below N/2 (the burst certainly not credited), or fewer than 3
``inside`` windows whose ``pages_lower`` is at least N/2 (the burst certainly
credited); fewer than 2 confirmed bursts really straddling an edge; or fewer than 90%
of bursts confirmed. Bounded readings that neither settle nor contradict are counted
as "inconclusive" for their placement. Unknown verdicts alone never make a success.

Evidence format. The results keep the full-precision raw readings the answers were
judged on (``raw_readings``) and every answer's wall-clock and perf-counter brackets,
so the attribution can be replayed exactly later.

**passed** otherwise. The reference check reads the exact readings the answers were
judged on (``MeasurementMonitor.last_finalize_readings``), captured before retention
trims the sampler's history. Results are written to ``--out`` (with ``--run-id``);
exit 0 only when passed, 2 when failed or inconclusive, 3 on an unexpected error.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PAGE = 4096
PLACEMENTS = ("quiet", "inside", "before", "after", "edge")

GENERATOR = r"""
import json, mmap, sys, time
path, schedule_path, log_path = sys.argv[1:4]
schedule = json.load(open(schedule_path))
log = []
with open(path, "rb") as handle:
    view = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    for burst in schedule:
        while time.time() < burst["at"]:
            time.sleep(0.001)
        began = time.time()
        total = 0
        chunks = 8 if burst.get("spread_s") else 1
        per = burst["pages"] // chunks
        parts = []
        for c in range(chunks):
            first, last = c * per, (burst["pages"] if c == chunks - 1 else (c + 1) * per)
            c_began = time.time()
            for i in range(first, last):
                total += view[burst["offset"] + i * 4096]
            parts.append({"began": c_began, "ended": time.time(), "pages": last - first})
            if chunks > 1 and c < chunks - 1:
                # The chunks so far, in the log during the pause (a replay or a simulated
                # counter can follow the burst as it happens); the pause keeps its length.
                pause_until = time.time() + burst["spread_s"] / (chunks - 1)
                with open(log_path, "w") as out:
                    json.dump(log + [{"id": burst["id"], "began": began, "ended": None, "pages": burst["pages"],
                                      "sum": total, "chunks": parts}], out)
                time.sleep(max(0.0, pause_until - time.time()))
        ended = time.time()
        log.append({"id": burst["id"], "began": began, "ended": ended, "pages": burst["pages"], "sum": total,
                    "chunks": parts})
        with open(log_path, "w") as out:
            json.dump(log, out)
    view.close()
"""


def build_schedule(t0: float, *, short_s: float = 2.0, long_s: float = 6.0, gap_s: float = 2.5,
                   pages: int = 2048, spacing_pages: int = 1024,
                   quick: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Answer windows and bursts. Short windows: every placement 4 times (quiet 6, inside
    6); long windows: quiet, inside and edge 3 times each. ``quick``: one short window of
    each placement (a plumbing smoke test, not a validation). Every burst gets a fresh
    file region, separated from the next by ``spacing_pages`` (read-ahead never reaches it)."""
    plan = ([("quiet", short_s)] * 6 + [("inside", short_s)] * 6 + [("before", short_s)] * 4
            + [("after", short_s)] * 4 + [("edge", short_s)] * 4
            + [("quiet", long_s)] * 3 + [("inside", long_s)] * 3 + [("edge", long_s)] * 3)
    if quick:
        plan = [(placement, short_s) for placement in PLACEMENTS]
    windows: list[dict[str, Any]] = []
    bursts: list[dict[str, Any]] = []
    t = t0
    region = 0
    for index, (placement, length) in enumerate(plan):
        start = t + gap_s
        windows.append({"id": f"w{index:02d}", "placement": placement, "start": start, "end": start + length})
        # Edge bursts are spread over 0.4 s from 0.2 s before the start, so they really
        # straddle it however fast the disk is; the others are as fast as possible.
        at = {"quiet": None, "inside": start + length / 2 - 0.1, "before": start - 0.5 - 0.3,
              "after": start + length + 0.5, "edge": start - 0.2}[placement]
        if at is not None:
            bursts.append({"id": f"w{index:02d}", "at": at, "pages": pages,
                           "spread_s": 0.4 if placement == "edge" else 0.0,
                           "offset": region * (pages + spacing_pages) * PAGE})
            region += 1
        t = start + length
    return windows, bursts


MARGIN_S = 0.25  # closer than this to an edge (but not overlapping): "near", no criterion
#: Windows that must end with usable evidence (exact or bounded, settling the question)
#: for a success, per placement.
REQUIRED_USABLE = {"quiet": 3, "inside": 3, "before": 2, "after": 2}
REQUIRED_MEASURED = REQUIRED_USABLE  # earlier name
#: Confirmed bursts that must really straddle an answer edge (verdict may be unknown).
REQUIRED_EDGE_EXERCISED = 2


def actual_placement(window: dict[str, Any], burst: dict[str, Any] | None) -> str:
    """Where the burst really fell, from the generator's own clock reads and the
    window's actual start and end (the schedule is only the intent)."""
    if burst is None:
        return "quiet"
    start, end = window.get("actual_start", window["start"]), window.get("actual_end", window["end"])
    began, ended = burst["began"], burst["ended"]
    if ended < start - MARGIN_S:
        return "before"
    if began > end + MARGIN_S:
        return "after"
    if began >= start + MARGIN_S and ended <= end - MARGIN_S:
        return "inside"
    if began < start < ended or began < end < ended:
        return "edge"
    return "near"


def evidence_bounds(verdict: dict[str, Any] | None) -> tuple[str, int | None, int | None]:
    """(evidence, pages_lower, pages_upper) of one verdict record. ``None`` is never
    read as zero: without a usable count the evidence is ``unknown``."""
    verdict = verdict or {}
    if verdict.get("status") != "measured":
        return str(verdict.get("status") or "unknown"), None, None
    low, high = verdict.get("pages_lower"), verdict.get("pages_upper")
    if low is None and high is None and isinstance(verdict.get("pages"), int):
        low = high = verdict["pages"]  # an older record: an exact count
    if not isinstance(low, int) or not isinstance(high, int):
        return "unknown", None, None
    return ("exact" if low == high else "bounded"), low, high


def burst_truth(window: dict[str, Any], burst: dict[str, Any] | None, pages: int,
                burst_reads: int | None = None) -> dict[str, Any]:
    """What the burst's disk reads inside the answer can be, from the generator's clock
    reads around each chunk it touched (the whole burst when no chunks).

    * ``high``: pages of chunks that possibly fell inside the answer -- a disk read needs
      a touch, so no more of the burst's reads can be inside.
    * ``low``: what is certain. A touched page may have been served from memory, and
      nothing says the cached ones were spread evenly (reviewer, after the bounds patch:
      1,024 pages touched before the answer and 1,024 inside, 1,024 reads in all -- every
      read may have come before the answer). So with a trustworthy **burst-specific**
      read total ``burst_reads`` the deficit is assigned adversarially -- to the chunks
      inside: ``low = max(0, reads - pages touched not certainly inside)``; without one
      (the system-wide counter's delta across the burst includes unrelated paging and
      is not burst-specific) no positive share is certain: ``low = 0``.
    * ``premise_low``: pages of chunks certainly inside -- the share if every touch was
      a disk read (the uncached scratch file's premise, not established). Used only to
      flag a *suspected* hidden burst (inconclusive), never to fail."""
    if burst is None:
        return {"low": 0, "high": 0, "premise_low": 0, "basis": "no burst"}
    start, end = window.get("actual_start", window["start"]), window.get("actual_end", window["end"])
    touched = int(burst.get("pages", pages))
    parts = burst.get("chunks") or [{"began": burst["began"], "ended": burst["ended"], "pages": touched}]
    certain = sum(p["pages"] for p in parts if p["began"] >= start and p["ended"] <= end)
    possible = sum(p["pages"] for p in parts if p["ended"] >= start and p["began"] <= end)
    if isinstance(burst_reads, int) and burst_reads >= 0:
        low = max(0, min(burst_reads, touched) - (touched - certain))
        basis = "burst-specific read total; any shortfall assigned to the chunks inside"
    else:
        low = 0
        basis = "no burst-specific read total: no positive share is certain"
    return {"low": low, "high": possible, "premise_low": certain, "basis": basis}


#: Allowance below the certain share before a reading counts as hiding paging.
HIDDEN_TOLERANCE = 1 / 8


def evaluate(windows: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]], confirmed: dict[str, bool],
             pages: int, bursts: dict[str, dict[str, Any]] | None = None,
             burst_reads: dict[str, int] | None = None) -> tuple[bool, dict[str, Any]]:
    """Apply the pass criteria (module docs); returns (passed, report). Placements and
    the truth come from where each burst actually fell when the burst log is given;
    otherwise from the schedule. ``burst_reads``: trustworthy burst-specific disk-read
    totals, when any exist (none does today: the system-wide counter is not one)."""
    half = pages / 2
    failures: list[str] = []
    suspected: list[str] = []
    table: dict[str, dict[str, int]] = {p: {"windows": 0, "exact": 0, "bounded": 0, "unknown": 0, "usable": 0,
                                            "inconclusive": 0, "unconfirmed": 0, "measured": 0}
                                        for p in (*PLACEMENTS, "near")}
    for window in windows:
        wid = window["id"]
        burst = (bursts or {}).get(wid)
        placement = actual_placement(window, burst) if bursts is not None else window["placement"]
        window["actual_placement"] = placement
        row = table[placement]
        row["windows"] += 1
        if placement != "quiet" and not confirmed.get(wid, False):
            row["unconfirmed"] += 1
            continue
        reads = (burst_reads or {}).get(wid)
        if bursts is not None:
            truth = burst_truth(window, burst, pages, reads)
        else:
            # No burst log (unit use): the schedule's intent -- an edge burst spreads its
            # 8 chunks over 0.4 s from 0.2 s before the start, so at least half follow it.
            touched_in = pages if placement == "inside" else (pages // 2 if placement == "edge" else 0)
            touched_max = pages if placement in ("inside", "edge") else 0
            low_ = max(0, min(reads, pages) - (pages - touched_in)) if isinstance(reads, int) else 0
            truth = {"low": low_, "high": touched_max, "premise_low": touched_in,
                     "basis": "schedule (no burst log)"}
        truth_low, truth_high = int(truth["low"]), int(truth["high"])
        window["truth_pages"] = [truth_low, truth_high]
        window["truth"] = truth
        evidence, low, high = evidence_bounds(verdicts.get(wid))
        row[evidence if evidence in ("exact", "bounded") else "unknown"] += 1
        if low is None or high is None:
            continue
        row["measured"] += 1
        label = f"{wid} {placement}: {evidence} {low}..{high} pages, truth {truth_low}..{truth_high}"
        if high < truth_low - pages * HIDDEN_TOLERANCE:
            failures.append(f"{label} (paging hidden)")
            continue
        if low >= truth_high + half:
            failures.append(f"{label} (paging attributed to the wrong answer)")
            continue
        if high < int(truth["premise_low"]) - pages * HIDDEN_TOLERANCE:
            # Below what the burst put inside if every touch was a disk read -- but no
            # evidence says every touch was: suspected only, never a failure, never a pass.
            suspected.append(f"{label} (suspected hidden: below the {truth['premise_low']} pages touched "
                             "certainly inside, but no burst-specific read total shows they were read)")
            row["inconclusive"] += 1
            continue
        settles = (low >= pages - half) if placement == "inside" else (high < half)
        if placement in ("quiet", "before", "after", "inside") and settles:
            row["usable"] += 1
        elif placement != "edge":
            row["inconclusive"] += 1
    scheduled = [w for w in windows if w["placement"] != "quiet"]
    confirmed_n = sum(1 for w in scheduled if confirmed.get(w["id"], False))
    # Coverage (reviewer, monitor round 2): success needs the placements to have been
    # really exercised with evidence that settles them -- positive (inside: the burst
    # certainly credited) and negative (quiet / before / after: certainly not) -- and
    # edges really straddled by a confirmed burst. Unknowns alone prove nothing.
    shortfalls: list[str] = []
    for placement, needed in REQUIRED_USABLE.items():
        got = table[placement]["usable"]
        if got < needed:
            shortfalls.append(f"{placement}: {got} usable window(s), need {needed}")
    edges = table["edge"]["windows"] - table["edge"]["unconfirmed"]
    if edges < REQUIRED_EDGE_EXERCISED:
        shortfalls.append(f"edge: {edges} confirmed burst(s) straddling an edge, need {REQUIRED_EDGE_EXERCISED}")
    if scheduled and confirmed_n < 0.9 * len(scheduled):
        shortfalls.append(f"only {confirmed_n} of {len(scheduled)} bursts were confirmed by the counter")
    shortfalls += suspected
    outcome = "failed" if failures else ("inconclusive" if shortfalls else "passed")
    return outcome == "passed", {"outcome": outcome, "failures": failures, "shortfalls": shortfalls,
                                 "table": table, "bursts_confirmed": f"{confirmed_n} of {len(scheduled)}"}


def write_uncached_file(path: Path, size: int) -> None:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.VirtualAlloc.restype = wintypes.LPVOID
    kernel32.VirtualAlloc.argtypes = [wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
    kernel32.WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                   ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    generic_write, create_always = 0x40000000, 2
    no_buffering, write_through = 0x20000000, 0x80000000
    handle = kernel32.CreateFileW(str(path), generic_write, 0, None, create_always, no_buffering | write_through, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), "CreateFileW failed")
    chunk = 1 << 20
    buffer = kernel32.VirtualAlloc(None, chunk, 0x3000, 0x04)  # page-aligned, as unbuffered I/O requires
    ctypes.memset(buffer, 0x5A, chunk)
    written = wintypes.DWORD(0)
    try:
        for _ in range(size // chunk):
            if not kernel32.WriteFile(handle, buffer, chunk, ctypes.byref(written), None) or written.value != chunk:
                raise OSError(ctypes.get_last_error(), "WriteFile failed")
    finally:
        kernel32.CloseHandle(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--pages", type=int, default=2048)
    parser.add_argument("--scratch-mb", type=int, default=512)
    parser.add_argument("--quick", action="store_true", help="one window per placement (smoke test only)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    data: dict[str, Any] = {"run_id": args.run_id, "status": "running", "args": vars(args)}

    def save() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".partial")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, out)

    try:
        return _run(args, data, save)
    except BaseException as exc:  # noqa: BLE001 - recorded, then reported by exit status
        data["status"] = f"error: {type(exc).__name__}: {exc}"
        save()
        print(data["status"])
        return 3


def _run(args: argparse.Namespace, data: dict[str, Any], save: Any) -> int:
    from inferenceos.profiling.runner import COUNTER_PATHS, SCOPED_INTERVAL_S
    from inferenceos.telemetry.attribution import PAGING_SPEC
    from inferenceos.telemetry.counters import MeasurementMonitor

    workdir = Path(tempfile.mkdtemp(prefix="inferenceos-attr-"))
    scratch = workdir / "scratch.bin"
    began = time.perf_counter()
    write_uncached_file(scratch, args.scratch_mb << 20)
    data["scratch_write_s"] = round(time.perf_counter() - began, 2)
    monitor = MeasurementMonitor(COUNTER_PATHS, SCOPED_INTERVAL_S, finalize_wait_s=3.0)
    data["monitor_open_s"] = round(monitor.open(first_sample_timeout_s=10.0), 3)
    generator: subprocess.Popen[bytes] | None = None
    try:
        t0 = time.time() + 1.0
        windows, bursts = build_schedule(t0, pages=args.pages, quick=args.quick)
        data["quick"] = bool(args.quick)
        needed = max(b["offset"] for b in bursts) + args.pages * 4096
        if needed > args.scratch_mb << 20:
            raise ValueError(f"scratch file too small: need {needed >> 20} MB")
        schedule_path, log_path = workdir / "schedule.json", workdir / "bursts.json"
        schedule_path.write_text(json.dumps(bursts), encoding="utf-8")
        generator = subprocess.Popen([sys.executable, "-c", GENERATOR, str(scratch), str(schedule_path), str(log_path)])
        for window in windows:
            while time.time() < window["start"]:
                time.sleep(0.0005)
            wall_start, perf_start = time.time(), time.perf_counter()
            window["actual_start"], window["perf_start"] = wall_start, perf_start
            while time.time() < window["end"]:
                time.sleep(0.0005)
            wall_end, perf_end = time.time(), time.perf_counter()
            window["actual_end"], window["perf_end"] = wall_end, perf_end
            monitor.note_answer(window["id"], wall_start=wall_start, wall_end=wall_end,
                                perf_start=perf_start, perf_end=perf_end)
        generator.wait(timeout=30)
        found, waited = monitor.finalize([w["id"] for w in windows])
        data["finalize_wait_s"] = round(waited, 3)
        log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
        # The readings finalize judged the answers on, captured before retention trimmed
        # the sampler's history (the reference check needs readings before each burst).
        readings = [r for r in monitor.last_finalize_readings if r.spec == PAGING_SPEC]
        data["readings_judged"] = len(readings)
        # Full precision, for an exact replay later (json keeps every float digit).
        data["raw_readings"] = [dataclasses.asdict(r) for r in monitor.last_finalize_readings]
        data["timing_note"] = ("windows: actual_start/actual_end are time.time() and perf_start/perf_end "
                               "perf_counter() at both ends of each answer; raw_readings are every reading "
                               "the answers were judged on, unrounded")
        confirmed: dict[str, bool] = {}
        burst_rows = []
        for entry in log:
            before = [r for r in readings if r.read_after <= entry["began"]]
            after = [r for r in readings if r.read_before >= entry["ended"]]
            delta = (after[0].raw - before[-1].raw) if before and after else None
            ok = delta is not None and delta >= entry["pages"] / 2
            confirmed[entry["id"]] = ok
            burst_rows.append({**entry, "counter_pages_across_burst": delta, "confirmed": ok,
                               "duration_s": round(entry["ended"] - entry["began"], 4)})
        verdicts = {wid: t.paging.record() for wid, t in found.items()}
        passed, report = evaluate(windows, verdicts, confirmed, args.pages, {e["id"]: e for e in burst_rows})
        data.update(windows=windows, bursts=burst_rows, verdicts=verdicts, evaluation=report,
                    sampler_errors=monitor.errors)
    finally:
        if generator is not None and generator.poll() is None:
            generator.kill()
            try:
                generator.wait(timeout=10)
            except subprocess.TimeoutExpired:
                data["generator_cleanup"] = "the generator did not exit"
        monitor.close()
        data["monitor_close"] = monitor.stop_record
        try:
            scratch.unlink()
        except OSError:
            pass
    cleanup_ok = bool(monitor.stop_record.get("resolved")) and "generator_cleanup" not in data
    if not cleanup_ok:
        report["failures"].append(f"cleanup unresolved: {monitor.stop_record} {data.get('generator_cleanup', '')}")
        report["outcome"], passed = "failed", False
    if passed:
        data["status"] = "done"
    elif report["outcome"] == "inconclusive":
        data["status"] = "inconclusive: " + "; ".join(report["shortfalls"])
    else:
        data["status"] = "failed: " + "; ".join(report["failures"] + report["shortfalls"])
    save()
    print(json.dumps({"status": data["status"], "table": report["table"],
                      "bursts_confirmed": report["bursts_confirmed"],
                      "finalize_wait_s": data.get("finalize_wait_s"), "monitor_open_s": data.get("monitor_open_s")},
                     indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
