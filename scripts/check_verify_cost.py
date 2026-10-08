"""Where one measurement's time goes, and whether stopping works, on the real engine
(proposed probe; prepared as ``_jobs/later/120_verify_cost_probe.ps1``, not queued).

Revised after review 10. Nothing is tuned or saved; production settings are untouched.

* **Part C, interruption** (first, because a failure here blocks everything else). For
  each round, three cases: stop the engine while it is **loading** (the backend is
  "starting" and owns its process), during **request dispatched / pre-first-chunk** (the
  HTTP POST to llama-server was sent and no data chunk has arrived yet, 2 s in; called
  that rather than "prefill" until hardware shows what the server is doing then -- the
  recorded ``opened`` time may later justify the stronger name) and during **decode** (tokens are
  streaming, 0.5 s after the first). The phases are observed at the HTTP client (stream
  dispatched / response opened / first chunk / done), never inferred from a higher
  adapter, and the owned process identity is recorded. Latency is measured from the
  stop request. Each case records separately: the stop call, the worker's return, the
  process's exit, and recovery (a fresh load and one answer). A case counts as
  exercised only if its phase still held when the stop was asked; a phase that is not
  exercised and passed in every round **fails** Part C (exit 2) -- it never counts as a
  pass. A worker or process that survives blocks recovery and every later case; every
  wait is bounded, so an unresponsive stop cannot hang the probe.
* **Part A0, raw counter lines** (about 40 s): the Windows counters the monitor reads,
  captured raw with each sample's own source timestamp, while the engine is idle, then
  during one short and one long answer. Records each line's Python arrival time and
  wall clock, so the counter's cadence, what interval a ``Pages Input/sec`` value covers,
  and the lag from source timestamp to arrival can be established (reviewer question 8;
  unverified until this runs).
* **Part A, the monitor** (diagnostic only; never an acceptance path): the same
  configuration and short question measured under three conditions -- the per-answer
  monitor (what tune uses today), the experimental measurement-scoped monitor, and no
  monitor -- in rotating order. Records each run's phases, the monitor's open/close
  costs and per-answer collection, and answer speed.
* **Part B, both questions in one load**: separate fresh loads vs one fresh load
  answering both questions, alternating short-then-long and long-then-short. Loads are
  counted from recorded events -- each successful backend start, with a monotonically
  increasing load id and the new process's pid and creation time -- never inferred from
  a duration; every answer records the process that gave it. "One process, one load"
  is "shown" only when exactly one load happened and every answer came from that
  process; it is "unverified" when the identity evidence is missing. Shared-load
  acceptance is not implemented.

Results are written to ``--out`` after every case and measurement (failures included).
Every requested part ends "passed", "disabled" (not requested: ``--interrupt-rounds 0``
or ``--skip-part-b``), "failed: ..." or "not run: ..."; so does the final cleanup (the
sticky stop returned and the last engine process is gone). Exit status: 0 only when
every requested part and the cleanup passed; 2 otherwise, listing each failing part
(Part C failing skips the later parts; a counter sampler whose exit is not confirmed
skips Parts A and B; a measurement-scoped sampler whose cleanup is unresolved stops
Part A there -- the measurements left are "not run" -- and skips Part B); 3 on an
unexpected error.

    .\\.venv\\Scripts\\python.exe -u scripts\\check_verify_cost.py --model models\\study\\Llama-3.2-3B-Instruct-Q4_K_M.gguf --layers 29 --long-passages 18 --rounds 3 --interrupt-rounds 3 --out _jobs\\out\\120_verify_cost_llama.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PHASE_WAIT_S = 90.0      # longest wait for the phase to be observed
STOP_CALL_S = 60.0       # longest wait for the stop call to return
WORKER_S = 90.0          # longest wait for the interrupted worker to return
GONE_S = 30.0            # longest wait for the process to exit
DISPATCHED_DELAY_S = 2.0  # pre-first-chunk case: stop this long after the POST, if no chunk yet
DECODE_DELAY_S = 0.5     # decode case: stop this long after the first token
#: Every round must exercise each of these phases (and pass) for exit status 0.
PHASES = ("loading", "dispatched_pre_first_chunk", "decode")


def classify_case(*, phase: str, completed: bool, returned_before_stop: bool, start_cancelled: bool,
                  at_stop: dict[str, Any]) -> str:
    """'interrupted' only when the stop hit the intended phase and the request did not
    complete; 'completed despite the stop' is a failure; otherwise 'phase not exercised'
    (which also fails the phase, see ``interruption_verdict``)."""
    if completed and returned_before_stop:
        return "phase not exercised"
    if completed:
        return "completed despite the stop"
    if phase == "loading":
        held = at_stop.get("backend_state") == "starting" and start_cancelled
    elif phase == "dispatched_pre_first_chunk":
        held = at_stop.get("dispatched") is not None and at_stop.get("first") is None
    else:  # decode
        held = at_stop.get("first") is not None and at_stop.get("done") is None
    return "interrupted" if held else "phase not exercised"


def case_passed(row: dict[str, Any]) -> bool:
    return bool(row.get("classification") == "interrupted" and row.get("stop_error") is None
                and row.get("stop_call_s") is not None and row.get("worker_returned")
                and row.get("process_gone") is True and row.get("recovered") is True)


def interruption_verdict(rows: list[dict[str, Any]], *, rounds: int,
                         phases: tuple[str, ...] = PHASES) -> tuple[int, str]:
    """Exit status for Part C: 0 only if, for every required phase, every round was
    exercised and passed. A phase never (or not always) exercised is a failure, never
    a pass; so is a blocked case or one that completed despite the stop."""
    problems: list[str] = []
    for row in rows:
        if row.get("blocked"):
            problems.append(str(row['blocked']))
        elif row.get("classification") == "completed despite the stop":
            problems.append(f"{row['label']}: completed despite the stop")
    for phase in phases:
        mine = [r for r in rows if r.get("phase") == phase]
        passed = [r for r in mine if case_passed(r)]
        if len(passed) < rounds:
            problems.append(f"{phase}: {len(passed)} of {rounds} rounds exercised and passed "
                            f"({sum(r.get('classification') == 'phase not exercised' for r in mine)} not exercised)")
    return (2, "; ".join(problems)) if problems else (0, f"all {len(phases)} phases exercised and passed in {rounds} rounds")


class Results:
    """Everything recorded so far, rewritten (atomically) after every case."""

    def __init__(self, path: Path, args: dict[str, Any]) -> None:
        self.path = path
        self.data: dict[str, Any] = {"args": args, "run_id": args.get("run_id"), "status": "running",
                                     "interruptions": [],
                                     "measurements": [], "monitor_events": [], "summary": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".partial")
        temporary.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, self.path)


def bounded(fn: Callable[[], Any], timeout_s: float) -> tuple[bool, Any, str | None]:
    """Run ``fn`` in a daemon thread; (returned in time, value, error)."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - recorded
            box["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout_s)
    return (not thread.is_alive()), box.get("value"), box.get("error")


RAW_COUNTER_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$paths = @(__PATHS__)
while ($true) {
    $before = [DateTime]::UtcNow.ToString('o')
    $set = Get-Counter -Counter $paths
    $after = [DateTime]::UtcNow.ToString('o')
    foreach ($s in $set.CounterSamples) {
        '{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}' -f $before, $after, $set.Timestamp.ToUniversalTime().ToString('o'), $s.Timestamp100NSec, $s.Path, $s.InstanceName, $s.CookedValue, $s.RawValue
    }
    [Console]::Out.Flush()
    Start-Sleep -Milliseconds 250
}
"""


COUNTER_EXIT_S = 10.0  # longest wait for the counter sampler process to exit (each of two waits)
READER_JOIN_S = 5.0    # longest wait for the reader thread after the process wait
PIPE_CLOSE_S = 2.0     # longest wait for closing the pipe (only after the reader returned)


def capture_raw_counters(paths: tuple[str, ...], seconds: float, during: Callable[[], None] | None = None,
                         marks: dict[str, float] | None = None) -> dict[str, Any]:
    """Run the same Get-Counter loop shape as the sampler, but keep every raw line with
    the counter set's own timestamp, the PowerShell UTC time before/after Get-Counter,
    and Python's arrival time (perf_counter and wall clock).

    The sampler process is always cleaned up (reviewer follow-up 3): terminate, a
    bounded wait for its exit (then kill and wait again), the reader thread joined and
    the pipe closed -- also when ``during()`` raises. Lines captured so far are kept.
    Returns ``{"lines", "during_error", "exit_confirmed", "reader_joined",
    "pipe_closed", "cleanup_resolved", "cleanup_errors"}``. The whole cleanup is
    bounded; ``cleanup_resolved`` False means a sampler or its reader may still be
    alive (the caller must not measure anything after that)."""
    import subprocess

    escaped = ",".join("'" + p.replace("'", "''") + "'" for p in paths)
    proc = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                             RAW_COUNTER_SCRIPT.replace("__PATHS__", escaped)],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    lines: list[dict[str, Any]] = []
    record: dict[str, Any] = {"lines": lines, "during_error": None, "exit_confirmed": None,
                              "reader_joined": None, "pipe_closed": None, "cleanup_resolved": None,
                              "cleanup_errors": []}

    def read() -> None:
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                lines.append({"perf": time.perf_counter(), "wall": time.time(), "line": raw.rstrip("\n")})
        except (OSError, ValueError):  # the pipe was closed under the reader
            pass

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    began = time.perf_counter()
    try:
        if during is not None:
            time.sleep(min(10.0, seconds / 3))
            if marks is not None:
                marks["answers_started"] = time.perf_counter()
            try:
                during()
            except Exception as exc:  # noqa: BLE001 - recorded; the lines so far are kept
                record["during_error"] = f"{type(exc).__name__}: {exc}"
            if marks is not None:
                marks["answers_ended"] = time.perf_counter()
        while record["during_error"] is None and time.perf_counter() - began < seconds:
            time.sleep(0.1)
    finally:
        _stop_sampler(proc, reader, record)
    return record


def _bounded_call(fn: Callable[[], Any], timeout_s: float) -> tuple[bool, str | None]:
    """Run ``fn`` in a daemon thread and wait at most ``timeout_s``: (finished, error)."""
    box: dict[str, str] = {}

    def run() -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - recorded
            box["error"] = f"{type(exc).__name__}: {exc}"

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout_s)
    return (not worker.is_alive()), box.get("error")


def _stop_sampler(proc: Any, reader: threading.Thread, record: dict[str, Any]) -> None:
    """Cleanup of the counter sampler. Never raises and never blocks beyond its bounds
    (reviewer follow-up 4, item 1): terminate; wait at most ``COUNTER_EXIT_S`` for the
    exit, then kill and wait again; join the reader at most ``READER_JOIN_S``. The pipe
    is closed only once the reader has returned (closing a stream another thread is
    blocked reading can itself block) and even then in a bounded helper thread. If the
    reader is still blocked the pipe is left to it -- the reader is a daemon thread --
    and the cleanup is reported unresolved, so no later measurement runs."""
    import subprocess

    errors: list[str] = record["cleanup_errors"]
    try:
        proc.terminate()
    except Exception as exc:  # noqa: BLE001 - recorded
        errors.append(f"terminate: {type(exc).__name__}: {exc}")
    confirmed = False
    try:
        proc.wait(timeout=COUNTER_EXIT_S)
        confirmed = True
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=COUNTER_EXIT_S)
            confirmed = True
        except Exception as exc:  # noqa: BLE001 - recorded
            errors.append(f"kill: {type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - recorded
        errors.append(f"wait: {type(exc).__name__}: {exc}")
    record["exit_confirmed"] = confirmed
    reader.join(READER_JOIN_S)
    joined = not reader.is_alive()
    record["reader_joined"] = joined
    record["pipe_closed"] = False
    if joined and getattr(proc, "stdout", None) is not None:
        closed, error = _bounded_call(proc.stdout.close, PIPE_CLOSE_S)
        record["pipe_closed"] = closed and error is None
        if error is not None:
            errors.append(f"close: {error}")
        elif not closed:
            errors.append(f"close: did not return within {PIPE_CLOSE_S:g} s")
    elif not joined:
        errors.append(f"reader: still blocked on the pipe after {READER_JOIN_S:g} s (pipe left open)")
    record["cleanup_resolved"] = bool(confirmed and joined and record["pipe_closed"])


def identity_key(identity: Any) -> tuple[int, float] | None:
    """(pid, creation time) -- enough to tell processes apart even if a pid is reused;
    None when either is unknown."""
    pid, created = getattr(identity, "pid", None), getattr(identity, "create_time", None)
    return None if pid is None or created is None else (int(pid), float(created))


class LoadLog:
    """Every successful engine start, recorded where it happens: the backend's
    ``start`` returned (reviewer follow-up 5). The backend refuses to start while a
    server is running, so each event is a genuine new load in a new process. Each
    event has a monotonically increasing ``load_id`` and the process identity (pid and
    creation time). Load counts come from these events, never from a duration."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.last_identity: Any = None
        self._lock = threading.Lock()

    def attach(self, backend: Any) -> Callable[[], None]:
        """Wrap ``backend.start`` (on the instance); returns the function that undoes it."""
        original = backend.start

        def start(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.record(getattr(backend, "identity", None))
            return result

        backend.start = start

        def detach() -> None:
            if getattr(backend, "start", None) is start:
                try:
                    del backend.start
                except AttributeError:
                    backend.start = original
        return detach

    def record(self, identity: Any) -> dict[str, Any]:
        with self._lock:
            key = identity_key(identity)
            event = {"load_id": len(self.events) + 1, "at": time.perf_counter(),
                     "pid": None if key is None else key[0], "create_time": None if key is None else key[1]}
            self.events.append(event)
            if identity is not None:
                self.last_identity = identity
            return event


def load_evidence(load_events: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether one measurement was answered by one process from one load.

    "shown": exactly one successful load during the measurement and every answer came
    from that process (same pid and creation time); "not shown": the events contradict
    it; "unverified": the identity evidence needed is missing."""
    loads = [e["load_id"] for e in load_events]
    if not runs:
        return {"verdict": "unverified", "why": "no answers were recorded", "load_ids": loads}
    if any(e.get("pid") is None or e.get("create_time") is None for e in load_events) or any(
            r.get("pid") is None or r.get("create_time") is None for r in runs):
        return {"verdict": "unverified", "why": "process identity (pid and creation time) not available",
                "load_ids": loads}
    if len(load_events) != 1:
        return {"verdict": "not shown", "why": f"{len(load_events)} successful loads during the measurement",
                "load_ids": loads}
    owner = (load_events[0]["pid"], load_events[0]["create_time"])
    others = sorted({(r["pid"], r["create_time"]) for r in runs} - {owner})
    if others:
        return {"verdict": "not shown", "why": f"answers came from another process: {others}", "load_ids": loads}
    return {"verdict": "shown", "why": f"one load (id {loads[0]}), every answer from pid {owner[0]}",
            "load_ids": loads}


def measurement_problems(row: dict[str, Any], *, expected_runs: int, require_sharing: bool) -> list[str]:
    """Why one Part A/B measurement does not count (empty list: it does)."""
    problems: list[str] = []
    if row.get("error"):
        problems.append(f"{row['label']}: {row['error']}")
    if len(row.get("runs", [])) != expected_runs:
        problems.append(f"{row['label']}: {len(row.get('runs', []))} of {expected_runs} answers recorded")
    if row.get("monitor") == "measurement":
        kinds = {e.get("event") for e in row.get("monitor_events", [])}
        if not {"open", "close"} <= kinds:
            problems.append(f"{row['label']} (measurement-scoped monitor): no open/close event recorded")
    for event in row.get("monitor_events", []):
        if event.get("event") == "close" and event.get("resolved") is False:
            problems.append(f"{row['label']}: sampler cleanup unresolved ("
                            + "; ".join(str(e) for e in (event.get("errors") or ["no detail"])) + ")")
    if require_sharing and (row.get("sharing") or {}).get("verdict") != "shown":
        sharing = row.get("sharing") or {}
        problems.append(f"{row['label']}: one process / one load {sharing.get('verdict', 'unverified')} "
                        f"({sharing.get('why', 'no evidence')})")
    return problems


def counter_problems(record: dict[str, Any]) -> list[str]:
    """Why the Part A0 raw-counter capture does not count (empty list: it does)."""
    problems: list[str] = []
    if record.get("error"):
        problems.append(str(record["error"]))
    if record.get("during_error"):
        problems.append(f"an answer failed during the capture ({record['during_error']})")
    if record.get("exit_confirmed") is False:
        problems.append("the counter sampler's exit was not confirmed")
    if record.get("reader_joined") is False:
        problems.append("the counter reader was still blocked on the pipe")
    if record.get("cleanup_resolved") is False and record.get("exit_confirmed") is not False \
            and record.get("reader_joined") is not False:
        problems.append("the counter sampler's cleanup was not completed (" +
                        "; ".join(record.get("cleanup_errors") or ["no detail"]) + ")")
    lines = record.get("lines") or []
    if not lines:
        problems.append("no counter line was captured")
    else:
        marks = record.get("marks_perf") or {}
        start, end = marks.get("answers_started"), marks.get("answers_ended")
        if start is not None and end is not None and not any(start <= ln["perf"] <= end + 2.0 for ln in lines):
            problems.append("no counter line arrived while the answers ran")
    return problems


def overall_verdict(parts: dict[str, str]) -> tuple[int, str]:
    """Exit status for the whole probe (reviewer follow-up 2): 0 only when every
    requested part passed and cleanup succeeded. A part is "passed", "disabled"
    (intentionally not requested; not a failure), "failed: ..." or "not run: ..."
    (requested but not reached -- a failure)."""
    failed = [f"{name} {state}" for name, state in parts.items() if state not in ("passed", "disabled")]
    if failed:
        return 2, "; ".join(failed)
    disabled = [name for name, state in parts.items() if state == "disabled"]
    return 0, "every requested part passed" + (f" ({', '.join(disabled)} disabled)" if disabled else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--long-passages", type=int, default=18)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--short-runs", type=int, default=5)
    parser.add_argument("--long-runs", type=int, default=3)
    parser.add_argument("--skip-part-b", action="store_true")
    parser.add_argument("--interrupt-rounds", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", default=None,
                        help="written into the results so a wrapper can tell this run's file from an older one")
    args = parser.parse_args(argv)
    results = Results(Path(args.out), vars(args))
    try:
        code = _probe(args, results)
    except BaseException as exc:  # noqa: BLE001 - recorded, then reported by exit status
        results.data["status"] = f"error: {type(exc).__name__}: {exc}"
        results.save()
        print(results.data["status"])
        return 3
    results.save()
    print(json.dumps(results.data["summary"], indent=2, default=str))
    return code


def _probe(args: argparse.Namespace, results: Results) -> int:  # noqa: C901 - one linear script
    from inferenceos.backends import llama_server_toolchain as srv
    from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
    from inferenceos.config import Settings
    from inferenceos.contracts.enums import MeasurementMode
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.experiment.measurement import options_for
    from inferenceos.models.loader import load_model_profile
    from inferenceos.profiling.runner import BaselineProfiler, ProfileTask
    from inferenceos.profiling.workloads import get_workload
    from inferenceos.runtime.gpu_search import ResidentMeasurementAdapter
    from inferenceos.runtime.resident_workload import ResidentActiveWorkload
    from inferenceos.telemetry.counters import MonitorCleanupUnresolved
    from inferenceos.tune.engine_probe import wait_until_gone
    from inferenceos.tune.paths import steady_workload

    settings = Settings()
    model = load_model_profile(Path(args.model))
    short = get_workload("short-short")
    long = steady_workload(f"long-question-{args.long_passages}x-steady", default=short)
    config = RuntimeConfiguration(context_length=args.context, gpu_offload_layers=args.layers)
    engine = LlamaCppResidentAdapter(settings=settings)
    backend: Any = engine._backend  # noqa: SLF001 - the probe observes the lifecycle
    resident = ResidentActiveWorkload(settings=settings, model=model, adapter=engine, workload=short,
                                      configuration=config)
    adapter = ResidentMeasurementAdapter(resident, explicit_defaults=True)
    profiler = BaselineProfiler(settings=settings, adapter=adapter)
    # Where an answer really is (review 10 follow-up, reviewer question 7): the HTTP
    # stream to llama-server is instrumented at the client, not at a higher adapter.
    #   dispatched: the client's generator started (the POST is about to be sent);
    #   opened:     the HTTP response object exists (status and headers received);
    #   first:      the first streamed data chunk arrived (prompt read, generation on);
    #   done:       the stream ended.
    stream: dict[str, float | None] = {"dispatched": None, "opened": None, "first": None, "done": None}
    original_stream = srv.LlamaServerClient.stream_completion
    original_open = srv.LlamaServerClient._open  # noqa: SLF001

    def observed_open(self: Any, *a: Any, **k: Any) -> Any:
        response = original_open(self, *a, **k)
        if k.get("method") == "POST":
            stream["opened"] = time.perf_counter()
        return response

    def observed_stream(self: Any, **kwargs: Any) -> Any:
        stream.update(dispatched=time.perf_counter(), opened=None, first=None, done=None)
        for index, chunk in enumerate(original_stream(self, **kwargs)):
            if index == 0:
                stream["first"] = time.perf_counter()
            yield chunk
        stream["done"] = time.perf_counter()

    srv.LlamaServerClient._open = observed_open  # noqa: SLF001
    srv.LlamaServerClient.stream_completion = observed_stream
    loads = LoadLog()
    detach_loads = loads.attach(backend)
    sequence = 0

    last_runs: list[Any] = []  # the ProfilingRun of every answer, for finalize_runs

    def answer(workload: Any, *, warmup: bool, monitor: bool = True) -> dict[str, Any]:
        nonlocal sequence
        sequence += 1
        options = dataclasses.replace(options_for(config), telemetry_sampling=monitor)
        loads_before = len(loads.events)
        began = time.perf_counter()
        run = profiler.run_once(ProfileTask(model=model, workload=workload, context_length=args.context),
                                mode=MeasurementMode.WARM, sequence=sequence, repetition=sequence,
                                warmup=warmup, options=options)
        last_runs.append(run)
        metrics = run.observation.metrics
        phases = dict(run.phase_timings_s)
        key = identity_key(engine.identity)
        new_loads = [e["load_id"] for e in loads.events[loads_before:]]
        return {
            "workload": workload.workload_id, "warmup": warmup, "monitor": monitor,
            "wall_s": round(time.perf_counter() - began, 4), "phases": phases,
            # Process and load evidence (reviewer follow-up 5): which process answered
            # (pid + creation time) and which successful loads happened during this call.
            "pid": engine.pid, "create_time": None if key is None else key[1],
            "load_ids": new_loads, "loaded_now": bool(new_loads),
            "ttft_s": metrics.ttft_s, "decode_tps": metrics.decode_tokens_per_sec,
            "prompt_tokens": metrics.prompt_tokens, "hard_faults_per_sec": metrics.hard_page_faults_per_sec,
        }

    # ------------------------------------------------------------------ Part C
    blocked: str | None = None

    def interrupt(label: str, phase: str) -> None:
        """One interruption case. ``phase`` is "loading" (the backend is starting and
        owns its process), "dispatched_pre_first_chunk" (the POST was sent, no data chunk
        yet) or "decode"
        (tokens are streaming). The stop is sent only once that phase is observed, and
        the case is "exercised" only if the phase still held when the stop was asked."""
        nonlocal blocked
        profiler.request_fresh_start()
        if phase != "loading":
            answer(long, warmup=True)  # loaded and warm before the case
        stream.update(dispatched=None, opened=None, first=None, done=None)
        box: dict[str, Any] = {}

        def work() -> None:
            try:
                box["run"] = answer(long if phase != "loading" else short, warmup=phase == "loading")
            except BaseException as exc:  # noqa: BLE001 - recorded
                box["error"] = f"{type(exc).__name__}: {exc}"
            box["returned"] = time.perf_counter()

        if phase == "loading":
            profiler.request_fresh_start()
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        row: dict[str, Any] = {"label": label, "phase": phase}
        identity = None
        deadline = time.monotonic() + PHASE_WAIT_S
        seen = False
        while time.monotonic() < deadline and worker.is_alive():
            now = time.perf_counter()
            if phase == "loading":
                owned = getattr(backend, "_owned", None)
                if backend.state == "starting" and owned is not None and getattr(owned, "pid", None):
                    identity, seen = owned.identity(), True
                    break
            elif phase == "dispatched_pre_first_chunk":
                sent = stream["dispatched"]
                if sent is not None and stream["first"] is None and now - sent >= DISPATCHED_DELAY_S:
                    identity, seen = engine.identity, True
                    break
            elif stream["first"] is not None and stream["done"] is None and now - stream["first"] >= DECODE_DELAY_S:
                identity, seen = engine.identity, True
                break
            time.sleep(0.01)
        asked = time.perf_counter()
        # What held at the moment of the stop request (decides "exercised").
        at_stop = {"backend_state": backend.state, **{k: (None if v is None else round(v - asked, 3))
                                                      for k, v in stream.items()}}
        row["at_stop"] = at_stop
        if not seen:
            worker.join(WORKER_S)
            row["classification"] = "phase not exercised"
            row["why"] = ("the request finished before the phase was observed" if "returned" in box
                          else "the phase was not observed in time")
            row["answer_error"] = box.get("error")
            row["worker_returned"] = not worker.is_alive()
            if worker.is_alive():
                # The worker survived its bounded wait: nothing new may start while it
                # lives (no recovery, no later case).
                blocked = (f"{label}: the phase was not observed and the worker did not return "
                           f"within {WORKER_S:g} s")
        else:
            row["pid"] = getattr(identity, "pid", None)
            stop_ok, _, stop_error = bounded(resident.cancel, STOP_CALL_S)  # the user-stop path
            row["stop_call_s"] = round(time.perf_counter() - asked, 3) if stop_ok else None
            row["stop_error"] = stop_error
            worker.join(WORKER_S)
            row["worker_returned"] = not worker.is_alive()
            row["worker_returned_s"] = round(box["returned"] - asked, 3) if "returned" in box else None
            gone = wait_until_gone(identity, timeout_s=GONE_S) if identity is not None else None
            row["process_gone"] = gone
            row["process_gone_s"] = round(time.perf_counter() - asked, 3)
            row["answer_error"] = box.get("error")
            unload = getattr(backend, "last_unload", None)
            row["start_cancelled"] = bool(unload is not None and unload.notes
                                          and str(unload.notes[0]).startswith("start cancelled"))
            row["classification"] = classify_case(
                phase=phase, completed="run" in box, returned_before_stop=box.get("returned", asked + 1) < asked,
                start_cancelled=row["start_cancelled"], at_stop=at_stop)
            if not stop_ok:
                blocked = f"{label}: the stop call did not return within {STOP_CALL_S:.0f} s"
            elif not row["worker_returned"]:
                blocked = f"{label}: the interrupted worker did not return within {WORKER_S:g} s"
            elif gone is False:
                blocked = f"{label}: the engine process (pid {row['pid']}) was still running after {GONE_S:.0f} s"
        # Recovery only once the worker has returned and the process is gone.
        if blocked is None:
            began = time.perf_counter()
            try:
                resident.allow_restart()
                profiler.request_fresh_start()
                answer(short, warmup=True)
                answer(short, warmup=False)
                row["recovered"] = True
            except Exception as exc:  # noqa: BLE001 - recorded
                row["recovered"] = False
                row["recover_error"] = f"{type(exc).__name__}: {exc}"
                blocked = f"{label}: recovery failed ({row['recover_error']})"
            row["recovery_s"] = round(time.perf_counter() - began, 3)
        else:
            row["recovered"] = None
            row["blocked"] = blocked
        results.data["interruptions"].append(row)
        results.save()
        print(json.dumps(row))

    # Every requested part gets a state; the exit status comes from all of them and the
    # cleanup (reviewer follow-up 2), never from Part C alone.
    parts: dict[str, str] = {"C": "disabled" if args.interrupt_rounds <= 0 else "not run: not reached",
                             "A0": "not run: not reached", "A": "not run: not reached",
                             "B": "disabled" if args.skip_part_b else "not run: not reached",
                             "cleanup": "not run: not reached"}
    results.data["parts"] = parts
    try:
        if args.interrupt_rounds > 0:
            for index in range(args.interrupt_rounds):
                for phase in PHASES:
                    if blocked is None:
                        interrupt(f"C-{phase}-{index + 1}", phase)
            code, verdict = interruption_verdict(results.data["interruptions"], rounds=args.interrupt_rounds)
            results.data["interruption_verdict"] = verdict
            parts["C"] = "passed" if code == 0 else f"failed: {verdict}"
        stop_reason: str | None = None
        if parts["C"] not in ("passed", "disabled"):
            # A phase that was never exercised has not passed: nothing else runs.
            stop_reason = "Part C did not pass"

        # -------------------------------------------------------------- Part A / B
        #: Set once a counter sampler's cleanup is unresolved (reviewer, monitor round 2):
        #: from then on no measurement starts, and every one left is reported "not run".
        halted: dict[str, str | None] = {"reason": None}

        def measurement(label: str, parts_: list[tuple[Any, int]], *, monitor: str) -> dict[str, Any]:
            profiler.measurement_monitor = monitor == "measurement"
            profiler.request_fresh_start()
            events_before = len(profiler.monitor_events)
            loads_before = len(loads.events)
            began = time.perf_counter()
            runs: list[dict[str, Any]] = []
            error = None
            del last_runs[:]
            try:
                for workload, count in parts_:
                    runs.append(answer(workload, warmup=True, monitor=monitor != "off"))
                    for _ in range(count):
                        runs.append(answer(workload, warmup=False, monitor=monitor != "off"))
                # As measure_configuration does: the scoped monitor attributes its
                # readings to the scored answers now (inside the timed measurement).
                finalize = getattr(profiler, "finalize_runs", None)
                if callable(finalize):
                    scored = [(i, r) for i, r in enumerate(last_runs) if not runs[i]["warmup"]]
                    done = list(finalize([r for _, r in scored]))
                    for (i, _), run_ in zip(scored, done):
                        runs[i]["hard_faults_per_sec"] = run_.observation.metrics.hard_page_faults_per_sec
                        runs[i]["phases"] = dict(run_.phase_timings_s)
            except MonitorCleanupUnresolved as exc:
                # Not generic retry material: recorded, and nothing further starts.
                error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:  # noqa: BLE001 - recorded
                error = f"{type(exc).__name__}: {exc}"
            finally:
                profiler.end_measurement()
            if getattr(profiler, "monitor_blocked", None):
                halted["reason"] = f"sampler cleanup unresolved: {profiler.monitor_blocked}"
            load_events = list(loads.events[loads_before:])
            sharing = load_evidence(load_events, runs)
            row: dict[str, Any] = {"label": label, "monitor": monitor, "order": [w.workload_id for w, _ in parts_],
                   "wall_s": round(time.perf_counter() - began, 3), "runs": runs, "error": error,
                   "pids": sorted({r["pid"] for r in runs if r["pid"]}),
                   "load_events": load_events, "loads": len(load_events),
                   "sharing": sharing,
                   "monitor_events": list(profiler.monitor_events)[events_before:]}
            results.data["measurements"].append(row)
            results.save()
            print(json.dumps({k: row[k] for k in ("label", "monitor", "order", "wall_s", "pids", "loads", "error")}
                             | {"sharing": sharing["verdict"]}))
            return row

        # Part A0: raw counter lines (reviewer question 8).
        if stop_reason is not None:
            parts["A0"] = f"not run: {stop_reason}"
        else:
            from inferenceos.profiling.runner import COUNTER_PATHS

            marks: dict[str, float] = {}

            def two_answers() -> None:
                for workload in (short, long):
                    marks[f"{workload.workload_id}_start"] = time.perf_counter()
                    answer(workload, warmup=False, monitor=False)
                    marks[f"{workload.workload_id}_end"] = time.perf_counter()

            record: dict[str, Any]
            try:
                profiler.request_fresh_start()
                answer(short, warmup=True, monitor=False)  # loaded; the capture then sees idle, then answers
                record = capture_raw_counters(COUNTER_PATHS, 40.0, during=two_answers, marks=marks)
            except Exception as exc:  # noqa: BLE001 - recorded
                record = {"error": f"{type(exc).__name__}: {exc}", "lines": [], "exit_confirmed": None}
            record["marks_perf"] = marks
            results.data["counter_raw"] = record
            problems = counter_problems(record)
            parts["A0"] = "passed" if not problems else "failed: " + "; ".join(problems)
            results.save()
            if record.get("cleanup_resolved") is False or record.get("exit_confirmed") is False:
                # A sampler (or its reader) may still be running: it would disturb every
                # later answer, so nothing else is measured.
                stop_reason = ("the counter sampler's exit was not confirmed" if record.get("exit_confirmed") is False
                               else "the counter sampler's cleanup was unresolved")

        # Part A: the monitor conditions.
        if stop_reason is not None:
            parts["A"] = f"not run: {stop_reason}"
        else:
            problems = []
            conditions = ["answer", "measurement", "off"]
            plan_a = [monitor for index in range(args.rounds)
                      for monitor in conditions[index % 3:] + conditions[: index % 3]]
            for done_a, monitor in enumerate(plan_a):
                if halted["reason"] is not None:
                    problems.append(f"{len(plan_a) - done_a} measurement(s) not run: {halted['reason']}")
                    break
                row = measurement("A-short", [(short, args.short_runs)], monitor=monitor)
                problems += measurement_problems(row, expected_runs=args.short_runs + 1, require_sharing=False)
            parts["A"] = "passed" if not problems else "failed: " + "; ".join(problems)
            if halted["reason"] is not None:
                stop_reason = halted["reason"]

        # Part B: both questions in one load.
        if args.skip_part_b:
            pass
        elif stop_reason is not None:
            parts["B"] = f"not run: {stop_reason}"
        else:
            problems = []
            for index in range(args.rounds):
                # Odd rounds reverse everything: long before short inside the shared load,
                # and the shared load before the separate ones.
                pair = [(short, args.short_runs), (long, args.long_runs)]
                steps: list[tuple[str, list[tuple[Any, int]]]] = [
                    ("B-separate-short", [pair[0]]), ("B-separate-long", [pair[1]]),
                    ("B-shared", pair if index % 2 == 0 else pair[::-1]),
                ]
                for label, chosen in (steps if index % 2 == 0 else steps[::-1]):
                    if halted["reason"] is not None:
                        problems.append(f"{label}: not run: {halted['reason']}")
                        continue
                    row = measurement(label, chosen, monitor="answer")
                    expected = sum(count + 1 for _, count in chosen)
                    problems += measurement_problems(row, expected_runs=expected, require_sharing=True)
            parts["B"] = "passed" if not problems else "failed: " + "; ".join(problems)
    finally:
        # Bounded cleanup with the sticky user stop: it never re-allows starts, so a
        # surviving worker cannot start new work afterwards. Then the last engine
        # process must be confirmed gone.
        last = engine.identity or loads.last_identity
        done, _, error = bounded(resident.cancel, STOP_CALL_S)
        results.data["final_stop"] = "ok" if done and error is None else (error or "did not return")
        try:
            gone = True if last is None else bool(wait_until_gone(last, timeout_s=GONE_S))
        except Exception as exc:  # noqa: BLE001 - recorded
            gone, results.data["final_gone_error"] = False, f"{type(exc).__name__}: {exc}"
        results.data["final_process_gone"] = gone
        cleanup_problems = []
        if results.data["final_stop"] != "ok":
            cleanup_problems.append(f"the final stop failed ({results.data['final_stop']})")
        if not gone:
            cleanup_problems.append(f"the engine process (pid {getattr(last, 'pid', None)}) was not confirmed gone")
        parts["cleanup"] = "passed" if not cleanup_problems else "failed: " + "; ".join(cleanup_problems)
        detach_loads()
        srv.LlamaServerClient._open = original_open  # noqa: SLF001
        srv.LlamaServerClient.stream_completion = original_stream

    results.data["summary"] = _summary(results.data, short.workload_id, long.workload_id)
    code, verdict = overall_verdict(parts)
    results.data["verdict"] = verdict
    results.data["status"] = "done" if code == 0 else f"failed: {verdict}"
    results.save()
    print(results.data["status"])
    return code


def _summary(data: dict[str, Any], short_id: str, long_id: str) -> dict[str, Any]:
    measurements = data["measurements"]

    def median(values: list[float]) -> float | None:
        values = [v for v in values if isinstance(v, (int, float))]
        return round(statistics.median(values), 4) if values else None

    summary: dict[str, Any] = {"interruptions": [
        {k: row.get(k) for k in ("label", "classification", "stop_call_s", "worker_returned_s",
                                 "process_gone", "process_gone_s", "recovered", "recovery_s")}
        for row in data["interruptions"]]}
    for monitor in ("answer", "measurement", "off"):
        rows = [m for m in measurements if m["label"] == "A-short" and m["monitor"] == monitor]
        scored = [r for m in rows for r in m["runs"] if not r["warmup"]]
        events = [e for m in rows for e in m["monitor_events"]]
        summary[f"monitor_{monitor}"] = {
            "measurement_wall_s": [m["wall_s"] for m in rows],
            "monitor_start_s_median": median([r["phases"].get("monitor_start_s") for r in scored]),
            "monitor_stop_s_median": median([r["phases"].get("monitor_stop_s") for r in scored]),
            "monitor_collect_s_median": median([r["phases"].get("monitor_collect_s") for r in scored]),
            "scope_open_s": [e["s"] for e in events if e["event"] == "open"],
            "scope_close_s": [e["s"] for e in events if e["event"] == "close"],
            "answers_without_samples": sum(1 for e in events if e["event"] == "answer" and not e.get("samples", 1)),
            # The scoped monitor's one bounded wait per measurement, and how often its
            # paging stayed unknown (missing, incomplete or ambiguous evidence).
            "finalize_wait_s": [e["wait_s"] for e in events if e["event"] == "finalize"],
            "paging_unknown_measurements": sum(1 for e in events if e["event"] == "finalize"
                                               and e.get("measurement_paging") == "unknown"),
            "request_s_median": median([r["phases"].get("request_s") for r in scored]),
            "decode_tps_median": median([r["decode_tps"] for r in scored]),
            "note": "diagnostic only; not an acceptance path" if monitor == "off" else None,
        }
    for label in ("B-separate-short", "B-separate-long", "B-shared"):
        rows = [m for m in measurements if m["label"] == label]
        summary[label] = {
            "wall_s": [m["wall_s"] for m in rows], "orders": [m["order"] for m in rows],
            "one_process_one_load": [m["sharing"]["verdict"] for m in rows],
            "long_first_word_median_s": median([r["ttft_s"] for m in rows for r in m["runs"]
                                                if not r["warmup"] and r["workload"] == long_id]),
            "short_first_word_median_s": median([r["ttft_s"] for m in rows for r in m["runs"]
                                                 if not r["warmup"] and r["workload"] == short_id]),
        }
    return summary


if __name__ == "__main__":
    sys.exit(main())
