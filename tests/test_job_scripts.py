"""Prepared laptop jobs (reviewer follow-up 3, item 1): every prepared ``.ps1`` must parse,
and the 120 wrapper must never print "all steps passed" when a step failed.

Python tests never exercise a PowerShell file, so a script that cannot even be parsed
(job 120 had ``$probe:`` inside a string) would only fail on the laptop. These tests use
the PowerShell parser itself: Windows PowerShell 5.1 (what ``agent-runner.ps1`` uses) when
present, otherwise PowerShell 7 (``pwsh``). They are skipped only when neither exists.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LATER = ROOT / "_jobs" / "later"
JOB_120 = LATER / "120_verify_cost_probe.ps1"

PARSE = r"""
param([string]$File)
$tokens = $null; $errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($File, [ref]$tokens, [ref]$errors)
foreach ($e in $errors) { "{0}: {1}" -f $e.Extent.StartLineNumber, $e.Message }
exit $errors.Count
"""


def _shell() -> str | None:
    names = ("powershell", "pwsh") if sys.platform == "win32" else ("pwsh",)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


SHELL = _shell()
needs_shell = pytest.mark.skipif(SHELL is None, reason="no PowerShell on this machine")


def _parse_errors(path: Path, tmp_path: Path) -> tuple[int, str]:
    script = tmp_path / "parse_check.ps1"
    script.write_text(PARSE, encoding="ascii")
    done = subprocess.run([str(SHELL), "-NoProfile", "-NonInteractive", "-File", str(script), "-File", str(path)],
                          capture_output=True, text=True, timeout=120)
    return done.returncode, (done.stdout + done.stderr).strip()


@needs_shell
@pytest.mark.parametrize("job", sorted(LATER.glob("*.ps1")), ids=lambda p: p.name)
def test_prepared_job_parses(job: Path, tmp_path: Path) -> None:
    code, errors = _parse_errors(job, tmp_path)
    print(f"\n{job.name}: {'parsed, no errors' if code == 0 else errors} ({Path(str(SHELL)).name})")
    assert code == 0, errors


@needs_shell
def test_parser_check_catches_the_old_job_120_error(tmp_path: Path) -> None:
    """The check itself would have caught the reviewer's finding."""
    bad = tmp_path / "bad.ps1"
    bad.write_text('$probe = 2\n"probe (exit $probe: 2 = failed)"\n', encoding="ascii")
    code, errors = _parse_errors(bad, tmp_path)
    print("\nold form:", errors)
    assert code != 0 and "Variable reference is not valid" in errors


def test_prepared_jobs_are_ascii() -> None:
    for job in LATER.glob("*.ps1"):
        job.read_bytes().decode("ascii")  # raises on any non-ASCII byte


FAKE_PY = r"""
import json, os, sys
args = sys.argv[1:]
if os.environ.get("FAKE_CALL_LOG"):
    with open(os.environ["FAKE_CALL_LOG"], "a", encoding="utf-8") as log:
        log.write(" ".join(a for a in args if a.endswith(".py") or a == "pytest") + "\n")
if "pytest" in args:
    sys.exit(int(os.environ.get("FAKE_LOCK_EXIT", "0")))
if any(a.endswith("time_model_hash.py") for a in args):
    if os.environ.get("FAKE_REMOVE_LAUNCHER"):
        os.remove(os.environ["FAKE_REMOVE_LAUNCHER"])  # the next launch fails to start
    sys.exit(int(os.environ.get("FAKE_HASH_EXIT", "0")))
out = args[args.index("--out") + 1].replace("\\", os.sep)
run_id = args[args.index("--run-id") + 1] if "--run-id" in args else None
status = os.environ.get("FAKE_STATUS")
scripted = os.environ.get("FAKE_SCRIPT_STATUS", "")  # "<script>=<status>": that script reports it
if scripted and any(a.endswith(scripted.split("=", 1)[0]) for a in args):
    status = scripted.split("=", 1)[1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"status": status, "run_id": run_id}, handle)
    print("probe status:", status)
    sys.exit(int(os.environ.get("FAKE_SCRIPT_EXIT", "2")))
failing = os.environ.get("FAKE_FAIL_SCRIPT")
if failing and any(a.endswith(failing) for a in args):
    status = "failed: scripted"
if status:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"status": status, "run_id": os.environ.get("FAKE_RUN_ID", run_id)}, handle)
print("probe status:", status)
if failing and any(a.endswith(failing) for a in args):
    sys.exit(2)
sys.exit(int(os.environ.get("FAKE_PROBE_EXIT", "0")))
"""


def _fake_python(tmp_path: Path) -> Path:
    script = tmp_path / "fake_python.py"
    script.write_text(FAKE_PY, encoding="utf-8")
    if sys.platform == "win32":
        launcher = tmp_path / "fake_python.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n@exit /b %ERRORLEVEL%\r\n', encoding="ascii")
    else:
        launcher = tmp_path / "fake_python.sh"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="ascii")
        launcher.chmod(0o755)
    return launcher


def _run_wrapper(tmp_path: Path, *, stale: bool = False, remove_launcher: bool = False,
                 job: Path = JOB_120, python: str | None = None, **env: str) -> tuple[int, str]:
    root = tmp_path / "repo"
    out = root / "_jobs" / "out"
    out.mkdir(parents=True)
    if stale:
        # Successful results left by earlier runs, under the old fixed name and others.
        for name in ("120_verify_cost_llama.json", "120_verify_cost_llama_20261001_000000_deadbeef.json"):
            (out / name).write_text(json.dumps({"status": "done", "run_id": "old-run"}), encoding="utf-8")
    launcher = _fake_python(tmp_path)
    if python is not None:
        launcher = Path(python)  # e.g. a missing interpreter: nothing can start
    if remove_launcher:
        env = {**env, "FAKE_REMOVE_LAUNCHER": str(launcher)}
    done = subprocess.run(
        [str(SHELL), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(job),
         "-Root", str(root), "-Python", str(launcher), "-SkipPowerWait"]
        if sys.platform == "win32" else
        [str(SHELL), "-NoProfile", "-NonInteractive", "-File", str(job),
         "-Root", str(root), "-Python", str(launcher), "-SkipPowerWait"],
        capture_output=True, text=True, timeout=120, env={**os.environ, **env})
    return done.returncode, done.stdout + done.stderr


CASES = {
    "probe failed (exit 2)": dict(FAKE_PROBE_EXIT="2", FAKE_STATUS="failed: A failed: A-short: RuntimeError"),
    "probe error (exit 3)": dict(FAKE_PROBE_EXIT="3", FAKE_STATUS="error: RuntimeError: boom"),
    "probe exit 0 but status not done": dict(FAKE_PROBE_EXIT="0", FAKE_STATUS="failed: cleanup failed"),
    "probe exit 0, no results file": dict(FAKE_PROBE_EXIT="0"),
    "lock regression failed": dict(FAKE_LOCK_EXIT="1", FAKE_STATUS="done"),
    "hash timing failed": dict(FAKE_HASH_EXIT="1", FAKE_STATUS="done"),
    # Reviewer follow-up 4, item 2: an older successful file must never count for this run.
    "stale success file, probe wrote nothing": dict(FAKE_PROBE_EXIT="0", stale="1"),
    "results file of another run": dict(FAKE_PROBE_EXIT="0", FAKE_STATUS="done", FAKE_RUN_ID="old-run"),
    # A probe that cannot even start must not inherit step 2's exit code 0.
    "probe launch failed after a successful step": dict(FAKE_STATUS="done", remove_launcher="1"),
}


@needs_shell
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_wrapper_never_reports_success_when_a_step_failed(case: str, tmp_path: Path) -> None:
    env = dict(CASES[case])
    stale, remove = bool(env.pop("stale", "")), bool(env.pop("remove_launcher", ""))
    code, output = _run_wrapper(tmp_path, stale=stale, remove_launcher=remove, **env)  # type: ignore[arg-type]
    last = [line for line in output.splitlines() if line.strip()][-1]
    print(f"\n{case}: exit {code}; last line: {last}")
    if case == "probe launch failed after a successful step":
        assert "probe (did not start" in output
    assert code == 2
    assert "all steps passed" not in output
    assert "FAILED:" in output


@needs_shell
def test_wrapper_reports_success_only_when_everything_passed(tmp_path: Path) -> None:
    code, output = _run_wrapper(tmp_path, stale=True, FAKE_PROBE_EXIT="0", FAKE_STATUS="done")
    print("\nall passing: exit", code, "|", [ln for ln in output.splitlines() if ln.strip()][-1])
    assert code == 0 and "all steps passed" in output and "FAILED" not in output
    json.dumps(output)  # output is plain text


JOB_121 = LATER / "121_monitor_attribution.ps1"
CASES_121 = {
    "attribution validation failed": dict(FAKE_STATUS="done", FAKE_FAIL_SCRIPT="check_monitor_attribution.py"),
    "overhead probe failed": dict(FAKE_STATUS="done", FAKE_FAIL_SCRIPT="check_verify_cost.py"),
    "results files of another run": dict(FAKE_STATUS="done", FAKE_RUN_ID="old-run"),
    "no results files": dict(FAKE_PROBE_EXIT="0"),
}


JOB_122 = LATER / "122_boundary_and_model_paging.ps1"


@needs_shell
@pytest.mark.parametrize("job", [JOB_121, JOB_122], ids=["121", "122"])
@pytest.mark.parametrize("case", list(CASES_121), ids=list(CASES_121))
def test_job_121_never_reports_success_when_a_step_failed(case: str, job: Path, tmp_path: Path) -> None:
    code, output = _run_wrapper(tmp_path, job=job, **CASES_121[case])  # type: ignore[arg-type]
    print(f"\n{job.name[:3]} {case}: exit {code}; {[ln for ln in output.splitlines() if ln.strip()][-1]}")
    assert code == 2 and "all steps passed" not in output and "FAILED:" in output


@needs_shell
@pytest.mark.parametrize("job", [JOB_121, JOB_122], ids=["121", "122"])
def test_job_121_reports_success_only_when_both_steps_passed(job: Path, tmp_path: Path) -> None:
    code, output = _run_wrapper(tmp_path, job=job, FAKE_STATUS="done")
    assert code == 0 and "all steps passed" in output


def test_job_122_runs_the_bounded_evidence_checks() -> None:
    """Job 122's model step uses every new probe option, the processor-only setup and
    an uncached model copy; its attribution step is the full plan (no --quick)."""
    text = JOB_122.read_text(encoding="ascii")
    for flag in ("--layers 0", "--monitors measurement", "--finalize-warmups", "--cold-model-copy",
                 "--consistency-sampler", "--skip-part-a0", "--skip-part-b", "--interrupt-rounds 0"):
        assert flag in text, flag
    assert "check_monitor_attribution.py --out $attrOut --run-id $runId" in text and "--quick" not in text


# -------------------- reviewer, after the bounds patch: job 122 stops after step 1
CASES_122_STOP = {
    "unresolved sampler cleanup": dict(
        FAKE_STATUS="done", FAKE_SCRIPT_STATUS="check_monitor_attribution.py=failed: sampler cleanup unresolved"),
    "step 1 inconclusive": dict(
        FAKE_STATUS="done", FAKE_SCRIPT_STATUS="check_monitor_attribution.py=inconclusive: quiet: 1 usable window(s)"),
    "step 1 error": dict(FAKE_STATUS="done", FAKE_SCRIPT_STATUS="check_monitor_attribution.py=error: OSError: boom",
                         FAKE_SCRIPT_EXIT="3"),
    "step 1 exit 0, no results": dict(FAKE_PROBE_EXIT="0"),
    "step 1 could not start": dict(FAKE_STATUS="done", missing_python="1"),
}


@needs_shell
@pytest.mark.parametrize("case", list(CASES_122_STOP), ids=list(CASES_122_STOP))
def test_job_122_never_starts_step_2_after_step_1_did_not_pass(case: str, tmp_path: Path) -> None:
    """The second command (a model and a second sampler) is never invoked -- checked from
    the fake interpreter's own call log, not only from the exit code."""
    env = dict(CASES_122_STOP[case])
    calls = tmp_path / "calls.log"
    python = str(tmp_path / "no_such_python.exe") if env.pop("missing_python", "") else None
    code, output = _run_wrapper(tmp_path, job=JOB_122, python=python, FAKE_CALL_LOG=str(calls),
                                **env)  # type: ignore[arg-type]
    invoked = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
    lines = [ln for ln in output.splitlines() if ln.strip()]
    print(f"\n122 {case}: exit {code}; invoked {invoked}; {lines[-1]}")
    assert not any("check_verify_cost.py" in line for line in invoked)  # step 2 never started
    assert "step 2: not run" in output and "all steps passed" not in output
    if case == "step 1 inconclusive":
        assert code == 5 and "INCONCLUSIVE (not a success)" in output and "FAILED" not in output
    else:
        assert code == 2 and "FAILED:" in output


@needs_shell
def test_job_122_step_2_inconclusive_is_not_a_success(tmp_path: Path) -> None:
    calls = tmp_path / "calls.log"
    code, output = _run_wrapper(tmp_path, job=JOB_122, FAKE_STATUS="done", FAKE_CALL_LOG=str(calls),
                                FAKE_SCRIPT_STATUS="check_verify_cost.py=inconclusive: R inconclusive: consistent, "
                                                   "but paging was not exercised")
    invoked = calls.read_text(encoding="utf-8").splitlines()
    print(f"\n122 step 2 inconclusive: exit {code}; invoked {invoked}")
    assert any("check_verify_cost.py" in line for line in invoked)
    assert code == 5 and "INCONCLUSIVE (not a success)" in output and "all steps passed" not in output


# ------------------------------------------- job 123: first practical-mode evaluation
def _evaluation_job(name: str) -> Path:
    # The runner moves a completed job to ran/. Keep exercising the same script
    # after the authorized hardware run, rather than failing because it moved.
    prepared = LATER / name
    return prepared if prepared.exists() else ROOT / "_jobs" / "ran" / name


JOB_123 = _evaluation_job("123_practical_llama_15min.ps1")
FAKE_123 = r"""
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_CALL_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(args) + "\n")
def arg(name):
    return args[args.index(name) + 1].replace("\\", os.sep)
touch = os.environ.get("FAKE123_TOUCH")
if touch:
    with open(touch, "a", encoding="utf-8") as handle:
        handle.write("changed by the run\n")
if not os.environ.get("FAKE123_NO_SUMMARY"):
    summary = arg("--summary")
    os.makedirs(os.path.dirname(summary), exist_ok=True)
    run_id = "another-run" if os.environ.get("FAKE123_STALE") else arg("--run-id")
    unchanged = not os.environ.get("FAKE123_RUNNER_BREACH")
    report, stem = arg("--out"), arg("--explain")
    if not os.environ.get("FAKE123_NO_REPORT"):
        with open(report, "w", encoding="utf-8") as handle:
            handle.write("{}")
    if not os.environ.get("FAKE123_NO_EXPLAIN"):
        for ext in (".json", ".md"):
            with open(stem + ext, "w", encoding="utf-8") as handle:
                handle.write("{}" if ext == ".json" else "# explanation")
    complete = not os.environ.get("FAKE123_INCOMPLETE")
    with open(summary, "w", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "status": os.environ.get("FAKE123_STATUS", "done"),
                   "outcome": os.environ.get("FAKE123_OUTCOME", "no change"),
                   "completion": {"complete": complete,
                                  "problems": [] if complete else ["the explanation could not be written"]},
                   "isolation": {"real_data_unchanged": unchanged,
                                 "problems": [] if unchanged else ["saved_setups.json: changed"]}}, handle)
sys.exit(int(os.environ.get("FAKE123_EXIT", "0")))
"""


def _run_123(tmp_path: Path, **env: str) -> tuple[int, str, list[list[str]]]:
    root = tmp_path / "repo"
    data = root / "data"
    data.mkdir(parents=True)
    (data / "saved_setups.json").write_text('{"setups": []}', encoding="utf-8")
    (data / "load_spread.json").write_text("{}", encoding="utf-8")
    script = tmp_path / "fake_123.py"
    script.write_text(FAKE_123, encoding="utf-8")
    if sys.platform == "win32":
        launcher = tmp_path / "fake_123.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n@exit /b %ERRORLEVEL%\r\n', encoding="ascii")
    else:
        launcher = tmp_path / "fake_123.sh"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="ascii")
        launcher.chmod(0o755)
    log = tmp_path / "calls.log"
    env = {**env, "FAKE_CALL_LOG": str(log)}
    if env.get("FAKE123_TOUCH") == "real":
        env["FAKE123_TOUCH"] = str(data / "saved_setups.json")
    cmd = [str(SHELL), "-NoProfile", "-NonInteractive"] + (["-ExecutionPolicy", "Bypass"] if sys.platform == "win32" else [])
    done = subprocess.run(cmd + ["-File", str(JOB_123), "-Root", str(root), "-Python", str(launcher), "-SkipPowerWait"],
                          capture_output=True, text=True, timeout=180, env={**os.environ, **env})
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return done.returncode, done.stdout + done.stderr, calls


@needs_shell
def test_job_123_runs_the_isolated_evaluation_with_job_101s_start(tmp_path: Path) -> None:
    code, out, calls = _run_123(tmp_path)
    print("\n" + out[-600:])
    assert code == 0 and "EVALUATION COMPLETED: no change" in out and "data folder unchanged" in out
    assert len(calls) == 1
    args = calls[0]
    joined = " ".join(args)
    assert args[0] == "-u" and args[1].replace("\\", "/").endswith("scripts/run_practical_eval.py")
    for expected in ("--budget-s 900", "--start-config gpu_offload_layers=23", "--context 4096",
                     "--effort performance_first", "--goal balanced",
                     "--expect-model-sha256 6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
                     "Llama-3.2-3B-Instruct-Q4_K_M.gguf", "101_llama3b_report.json"):
        assert expected in joined, expected
    scratch = args[args.index("--scratch") + 1]
    assert "123_scratch_" in scratch and "data" not in scratch.split("123_scratch_")[0].replace("_jobs", "")


CASES_123 = {
    "the run changed the owner's data": ({"FAKE123_TOUCH": "real"}, 6, "ISOLATION BREACH"),
    "the runner reports a breach": ({"FAKE123_EXIT": "6", "FAKE123_RUNNER_BREACH": "1"}, 6, "ISOLATION BREACH"),
    "refused before starting": ({"FAKE123_EXIT": "7", "FAKE123_STATUS": "refused: x"}, 7, "REFUSED"),
    "the tune failed": ({"FAKE123_EXIT": "1", "FAKE123_STATUS": "failed", "FAKE123_OUTCOME": "failed: x"}, 2,
                        "TUNE FAILED"),
    # Reviewer's reproduction: exit 0 with a summary saying error / failed: missing report.
    "summary says error, runner exit 0": ({"FAKE123_STATUS": "error", "FAKE123_OUTCOME": "failed: missing report"},
                                          8, "REPORTING ERROR"),
    "evidence incomplete": ({"FAKE123_INCOMPLETE": "1"}, 8, "REPORTING ERROR"),
    "report file missing": ({"FAKE123_NO_REPORT": "1"}, 8, "the report is missing"),
    "explanation missing": ({"FAKE123_NO_EXPLAIN": "1"}, 8, "the explanation is missing"),
    "failed outcome, runner exit 0": ({"FAKE123_OUTCOME": "failed: x"}, 8, "REPORTING ERROR"),
    "tune failed, summary says done": ({"FAKE123_EXIT": "1", "FAKE123_OUTCOME": "failed: x"}, 8, "REPORTING ERROR"),
    "runner reporting error": ({"FAKE123_EXIT": "8", "FAKE123_STATUS": "reporting error: the tune report is missing",
                                "FAKE123_OUTCOME": ""}, 8, "REPORTING ERROR"),
    "no summary": ({"FAKE123_NO_SUMMARY": "1"}, 3, "no summary for this run"),
    "another run's summary": ({"FAKE123_STALE": "1"}, 3, "no summary for this run"),
    "runner error": ({"FAKE123_EXIT": "3", "FAKE123_STATUS": "error"}, 3, "RUNNER ERROR"),
}


@needs_shell
@pytest.mark.parametrize("case", sorted(CASES_123))
def test_job_123_never_reports_completion_when_something_went_wrong(case: str, tmp_path: Path) -> None:
    env, expected_code, words = CASES_123[case]
    code, out, _ = _run_123(tmp_path, **env)
    print(f"\n{case}: exit {code} | {out.strip().splitlines()[-1] if out.strip() else ''}")
    assert code == expected_code and words in out and "EVALUATION COMPLETED" not in out


@needs_shell
def test_job_124_uses_granite_historical_baseline_and_isolated_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _evaluation_job("124_practical_granite_moe_15min.ps1")
    code, errors = _parse_errors(job, tmp_path)
    assert code == 0, errors
    monkeypatch.setattr(sys.modules[__name__], "JOB_123", job)
    code, output, calls = _run_123(tmp_path)
    assert code == 0 and "EVALUATION COMPLETED: no change" in output
    assert len(calls) == 1
    args = calls[0]
    expected = {
        "--model": "models/study/granite-3.1-3b-a800m-instruct-Q4_K_M.gguf",
        "--context": "4096", "--budget-s": "900",
        "--start-config": "gpu_offload_layers=33",
        "--effort": "performance_first", "--goal": "balanced",
        "--expect-model-sha256": "48e0edcd578fd4462f26127f04c651d0e650741110185297741089aea01a82b3",
        "--historical": "_jobs/out/102_granite31moe_report.json",
    }
    for flag, value in expected.items():
        assert args[args.index(flag) + 1].replace("\\", "/") == value
    assert "124_scratch_" in args[args.index("--scratch") + 1]


@needs_shell
def test_job_125_repeats_job_123s_isolated_llama_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prepared confirmation-planning check: the same model, hash, start, modes,
    target, historical baseline and isolation as job 123; only its own artifact names
    differ."""
    job = _evaluation_job("125_practical_llama_planned_confirmation.ps1")
    code, errors = _parse_errors(job, tmp_path)
    assert code == 0, errors
    monkeypatch.setattr(sys.modules[__name__], "JOB_123", job)
    code, output, calls = _run_123(tmp_path)
    assert code == 0 and "EVALUATION COMPLETED: no change" in output
    assert len(calls) == 1
    args = calls[0]
    expected = {
        "--model": "models/study/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "--context": "4096", "--budget-s": "900",
        "--start-config": "gpu_offload_layers=23",
        "--effort": "performance_first", "--goal": "balanced",
        "--expect-model-sha256": "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
        "--historical": "_jobs/out/101_llama3b_report.json",
    }
    for flag, value in expected.items():
        assert args[args.index(flag) + 1].replace("\\", "/") == value
    assert "125_scratch_" in args[args.index("--scratch") + 1]
    assert "125_practical_llama_" in args[args.index("--out") + 1]
    assert "ASCII" in job.read_text(encoding="ascii")


JOB_127 = LATER / "127_prompt_reuse_study.ps1"
needs_127 = pytest.mark.skipif(not JOB_127.exists(), reason="job 127 is not in this checkout")


@needs_127
def test_job_127_runs_only_the_isolated_prompt_reuse_study() -> None:
    """Prepared 16 Oct 2026: the prompt-reuse study reads saved setups and never tunes,
    saves or changes settings."""
    text = JOB_127.read_text(encoding="ascii")
    assert "scripts\\evaluate_prompt_reuse.py" in text and "--context 4096" in text
    for forbidden in ("inferenceos tune", "-m inferenceos", "run_tune", "run_start", "--simulate", "Set-ItemProperty",
                      "powercfg", "Invoke-WebRequest"):
        assert forbidden not in text, forbidden


FAKE_127 = r"""
import json, os, sys
args = sys.argv[1:]
log = os.environ["FAKE127_LOG"]
with open(log, "a", encoding="utf-8") as handle:
    handle.write(args[args.index("--model") + 1] + " " + " ".join(a for a in args if a == "--no-power-check") + "\n")
calls = sum(1 for _ in open(log, encoding="utf-8"))
exits = [int(x) for x in os.environ.get("FAKE127_EXITS", "0").split(",")]
code = exits[min(calls - 1, len(exits) - 1)]
out = args[args.index("--out") + 1].replace("\\", os.sep)
os.makedirs(os.path.dirname(out), exist_ok=True)
status = {0: "done", 2: "incomplete", 5: "stopped: cleanup incomplete", 7: "stopped: validity gate"}.get(code, "error")
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"run_id": args[args.index("--run-id") + 1], "status": status,
               "data_folder_unchanged": code != 6}, handle)
sys.exit(code)
"""


def _launcher(tmp_path: Path, name: str, body: str) -> Path:
    script = tmp_path / f"{name}.py"
    script.write_text(body, encoding="utf-8")
    if sys.platform == "win32":
        launcher = tmp_path / f"{name}.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n@exit /b %ERRORLEVEL%\r\n', encoding="ascii")
    else:
        launcher = tmp_path / f"{name}.sh"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="ascii")
        launcher.chmod(0o755)
    return launcher


POWER = r"""
import os
seq = os.environ["FAKE127_POWER"].split(",")
counter = os.environ["FAKE127_POWER_COUNT"]
n = int(open(counter).read()) if os.path.exists(counter) else 0
open(counter, "w").write(str(n + 1))
state = seq[min(n, len(seq) - 1)]
if state == "raise":
    raise SystemExit(3)
print(state)
"""


def _run_127(tmp_path: Path, *, exits: str = "0", power: str | None = "ac", skip_power: bool = False,
             power_wait: int = 0) -> tuple[int, str, list[str], int]:
    root = tmp_path / "repo"
    root.mkdir()
    log = tmp_path / "calls.txt"
    log.write_text("", encoding="utf-8")
    launcher = _launcher(tmp_path, "fake_python", FAKE_127)
    command = [str(SHELL), "-NoProfile", "-NonInteractive"]
    if sys.platform == "win32":
        command += ["-ExecutionPolicy", "Bypass"]
    command += ["-File", str(JOB_127), "-Root", str(root), "-Python", str(launcher), "-RestSeconds", "0",
                "-PowerWaitSeconds", str(power_wait)]
    env = {**os.environ, "FAKE127_LOG": str(log), "FAKE127_EXITS": exits,
           "FAKE127_POWER_COUNT": str(tmp_path / "power_count.txt")}
    if power is not None:
        command += ["-PowerStateScript", str(_launcher(tmp_path, "fake_power", POWER))]
        env["FAKE127_POWER"] = power
    if skip_power:
        command += ["-SkipPowerWait"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=180, env=env)
    calls = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = int((tmp_path / "power_count.txt").read_text()) if (tmp_path / "power_count.txt").exists() else 0
    return done.returncode, done.stdout + done.stderr, calls, queries


@needs_shell
@needs_127
def test_job_127_reports_success_only_when_every_model_is_done(tmp_path: Path) -> None:
    code, output, calls, queries = _run_127(tmp_path)
    assert code == 0 and "all steps passed" in output and len(calls) == 4
    assert queries >= 5  # before the job and before every model


@needs_shell
@needs_127
@pytest.mark.parametrize("stop_code", ["3", "5", "6", "7", "9", "1"])
def test_job_127_stops_after_a_cleanup_isolation_or_validity_stop(stop_code: str, tmp_path: Path) -> None:
    code, output, calls, _ = _run_127(tmp_path, exits=f"{stop_code},0")
    assert len(calls) == 1, calls  # no further model is loaded
    assert code == 2 and "STOPPED" in output and "all steps passed" not in output


@needs_shell
@needs_127
def test_job_127_continues_after_an_incomplete_or_refused_model(tmp_path: Path) -> None:
    code, output, calls, _ = _run_127(tmp_path, exits="2,4,0,0")
    assert len(calls) == 4 and code == 2 and "all steps passed" not in output


@needs_shell
@needs_127
@pytest.mark.parametrize("power", ["raise", "unknown", "battery"])
def test_job_127_never_starts_without_verified_mains_power(power: str, tmp_path: Path) -> None:
    code, output, calls, queries = _run_127(tmp_path, power=power)
    assert calls == [] and code == 4 and queries >= 1


@needs_shell
@needs_127
def test_job_127_rechecks_power_before_every_model(tmp_path: Path) -> None:
    code, output, calls, queries = _run_127(tmp_path, power="ac,ac,battery")
    assert len(calls) == 1 and code == 2 and "not on mains power" in output and queries == 3


@needs_shell
@needs_127
def test_job_127_test_run_is_labelled_and_never_claims_success(tmp_path: Path) -> None:
    code, output, calls, _ = _run_127(tmp_path, power=None, skip_power=True)
    assert len(calls) == 4 and all("--no-power-check" in c for c in calls)
    assert "TEST RUN" in output and "all steps passed" not in output


JOB_128 = LATER / "128_reuse_correctness_study.ps1"
needs_128 = pytest.mark.skipif(not JOB_128.exists(), reason="job 128 is not in this checkout")


@needs_128
def test_job_128_runs_only_the_isolated_correctness_study() -> None:
    """Prepared 17 Oct 2026: the reuse-correctness study reads saved setups and never
    tunes, saves or changes settings; --swa-full only for the sliding-window model."""
    text = JOB_128.read_text(encoding="ascii")
    assert "scripts\\evaluate_prompt_reuse.py" in text and "--context 4096" in text
    assert "'--study', 'correctness', '--chat-exposure'" in text
    assert text.count("--swa-full-variant") == 1 and "gemma1b" in text and "llama3b" in text
    for forbidden in ("inferenceos tune", "-m inferenceos", "run_tune", "run_start", "--simulate", "Set-ItemProperty",
                      "powercfg", "Invoke-WebRequest"):
        assert forbidden not in text, forbidden


FAKE_128 = FAKE_127.replace(
    'handle.write(args[args.index("--model") + 1] + " " + " ".join(a for a in args if a == "--no-power-check") + "\\n")',
    'handle.write(" ".join(args) + "\\n")').replace(
    '"data_folder_unchanged": code != 6',
    '"data_folder_unchanged": code != 6, "simulated": os.environ.get("FAKE128_PROBLEM") == "simulation", '
    '"study": {"version": "wrong" if os.environ.get("FAKE128_PROBLEM") == "version" else "reuse-correctness-8"}, '
    '"preregistration": {"sha256": "wrong" if os.environ.get("FAKE128_PROBLEM") == "rules" '
    'else args[args.index("--preregistration-sha256") + 1]}')


def _run_128(tmp_path: Path, *, exits: str = "0", power: str | None = "ac",
             skip_power: bool = False, rules: str = "valid", report_problem: str = "") -> tuple[int, str, list[str], int]:
    root = tmp_path / "repo"
    root.mkdir()
    rule_path = root / "_jobs" / "later" / "128_rules_reuse-correctness-8.txt"
    rule_path.parent.mkdir(parents=True)
    if rules != "missing":
        rule_path.write_bytes((LATER / rule_path.name).read_bytes() + (b"tamper" if rules == "changed" else b""))
    log = tmp_path / "calls.txt"
    log.write_text("", encoding="utf-8")
    launcher = _launcher(tmp_path, "fake_python", FAKE_128)
    command = [str(SHELL), "-NoProfile", "-NonInteractive"]
    if sys.platform == "win32":
        command += ["-ExecutionPolicy", "Bypass"]
    command += ["-File", str(JOB_128), "-Root", str(root), "-Python", str(launcher), "-RestSeconds", "0",
                "-PowerWaitSeconds", "0"]
    env = {**os.environ, "FAKE127_LOG": str(log), "FAKE127_EXITS": exits,
           "FAKE127_POWER_COUNT": str(tmp_path / "power_count.txt"), "FAKE128_PROBLEM": report_problem}
    if power is not None:
        command += ["-PowerStateScript", str(_launcher(tmp_path, "fake_power", POWER))]
        env["FAKE127_POWER"] = power
    if skip_power:
        command += ["-SkipPowerWait"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=180, env=env)
    calls = [line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = int((tmp_path / "power_count.txt").read_text()) if (tmp_path / "power_count.txt").exists() else 0
    return done.returncode, done.stdout + done.stderr, calls, queries


@needs_shell
@needs_128
def test_job_128_runs_gemma_with_both_windows_then_the_control(tmp_path: Path) -> None:
    code, output, calls, queries = _run_128(tmp_path)
    assert code == 0 and "all steps passed" in output and len(calls) == 2 and queries >= 3
    assert "governing rules: reuse-correctness-8 SHA-256" in output
    gemma, llama = calls
    assert "gemma-3-1b" in gemma and "--swa-full-variant" in gemma
    assert "Llama-3.2-3B" in llama and "--swa-full-variant" not in llama
    for call in calls:
        assert "--study correctness" in call and "--chat-exposure" in call and "--blocks 3" in call
        assert "128_correctness_" in call and "--no-power-check" not in call


@needs_shell
@needs_128
@pytest.mark.parametrize("rules", ["missing", "changed"])
@pytest.mark.parametrize("skip_power", [False, True])
def test_job_128_refuses_missing_or_changed_frozen_rules_before_dispatch(
    tmp_path: Path, rules: str, skip_power: bool,
) -> None:
    code, output, calls, queries = _run_128(tmp_path, rules=rules, skip_power=skip_power)
    assert code == 4 and "REFUSED: cannot verify frozen job 128 rules" in output
    assert calls == [] and queries == 0 and "all steps passed" not in output


@needs_shell
@needs_128
@pytest.mark.parametrize("problem", ["version", "rules", "simulation"])
def test_job_128_never_continues_after_contradictory_success_evidence(tmp_path: Path, problem: str) -> None:
    code, output, calls, _ = _run_128(tmp_path, report_problem=problem)
    assert code == 2 and len(calls) == 1 and "STOPPED" in output and "all steps passed" not in output


@needs_shell
@needs_128
@pytest.mark.parametrize("stop_code", ["3", "5", "6", "7", "9", "1"])
def test_job_128_stops_after_a_cleanup_isolation_or_validity_stop(stop_code: str, tmp_path: Path) -> None:
    code, output, calls, _ = _run_128(tmp_path, exits=f"{stop_code},0")
    assert len(calls) == 1 and code == 2 and "STOPPED" in output and "all steps passed" not in output


@needs_shell
@needs_128
def test_job_128_continues_after_an_incomplete_or_refused_model(tmp_path: Path) -> None:
    for exits in ("2,0", "4,0"):
        folder = tmp_path / exits.replace(",", "_")
        folder.mkdir()
        code, output, calls, _ = _run_128(folder, exits=exits)
        assert len(calls) == 2 and code == 2 and "all steps passed" not in output


@needs_shell
@needs_128
@pytest.mark.parametrize("power", ["battery", "unknown", "raise"])
def test_job_128_never_starts_without_verified_mains_power(power: str, tmp_path: Path) -> None:
    code, output, calls, _ = _run_128(tmp_path, power=power)
    assert calls == [] and code != 0 and "all steps passed" not in output


@needs_shell
@needs_128
def test_job_128_test_run_is_labelled_and_never_claims_success(tmp_path: Path) -> None:
    code, output, calls, _ = _run_128(tmp_path, power=None, skip_power=True)
    assert "TEST RUN" in output and "all steps passed" not in output
    assert all("--no-power-check" in call for call in calls) and len(calls) == 2
