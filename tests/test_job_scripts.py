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
if "pytest" in args:
    sys.exit(int(os.environ.get("FAKE_LOCK_EXIT", "0")))
if any(a.endswith("time_model_hash.py") for a in args):
    if os.environ.get("FAKE_REMOVE_LAUNCHER"):
        os.remove(os.environ["FAKE_REMOVE_LAUNCHER"])  # the next launch fails to start
    sys.exit(int(os.environ.get("FAKE_HASH_EXIT", "0")))
out = args[args.index("--out") + 1].replace("\\", os.sep)
run_id = args[args.index("--run-id") + 1] if "--run-id" in args else None
status = os.environ.get("FAKE_STATUS")
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
                 job: Path = JOB_120, **env: str) -> tuple[int, str]:
    root = tmp_path / "repo"
    out = root / "_jobs" / "out"
    out.mkdir(parents=True)
    if stale:
        # Successful results left by earlier runs, under the old fixed name and others.
        for name in ("120_verify_cost_llama.json", "120_verify_cost_llama_20261001_000000_deadbeef.json"):
            (out / name).write_text(json.dumps({"status": "done", "run_id": "old-run"}), encoding="utf-8")
    launcher = _fake_python(tmp_path)
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


@needs_shell
@pytest.mark.parametrize("case", list(CASES_121), ids=list(CASES_121))
def test_job_121_never_reports_success_when_a_step_failed(case: str, tmp_path: Path) -> None:
    code, output = _run_wrapper(tmp_path, job=JOB_121, **CASES_121[case])  # type: ignore[arg-type]
    print(f"\n121 {case}: exit {code}; {[ln for ln in output.splitlines() if ln.strip()][-1]}")
    assert code == 2 and "all steps passed" not in output and "FAILED:" in output


@needs_shell
def test_job_121_reports_success_only_when_both_steps_passed(tmp_path: Path) -> None:
    code, output = _run_wrapper(tmp_path, job=JOB_121, FAKE_STATUS="done")
    assert code == 0 and "all steps passed" in output
