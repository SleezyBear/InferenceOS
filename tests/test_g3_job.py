"""Execute the prepared wrapper with fake programs; never load a model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import pytest

from tests.test_job_scripts import LATER, SHELL, _launcher

JOB = LATER / "129_g3_startup_evidence_study.ps1"
RULES = LATER / "129_rules_g3-evidence-2.json"
pytestmark = pytest.mark.skipif(not JOB.is_file() or not RULES.is_file() or SHELL is None,
                                reason="prepared job129 and PowerShell required")

FAKE = r'''
import hashlib,json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['G3_CALLS'],'a') as h:h.write(' '.join(args)+'\n')
fault=os.environ.get('G3_FAULT','')
out=Path(args[args.index('--out')+1].replace('\\',os.sep))
out.parent.mkdir(parents=True,exist_ok=True)
if any(a.endswith('evaluate_prompt_reuse.py') for a in args):
    code=int(os.environ.get('G3_EXIT','0'))
    d={'run_id':args[args.index('--run-id')+1], 'status':'done' if code==0 else 'error',
       'study':{'version':'reuse-correctness-8'},'simulated':False,
       'preregistration':{'sha256':args[args.index('--preregistration-sha256')+1]}}
    if fault=='id':d['run_id']='older'
    if fault=='version':d['study']['version']='other'
    if fault=='rules':d['preregistration']['sha256']='other'
    if fault=='simulation':d['simulated']=True
    if fault=='status':d['status']='running'
    if fault=='changed_input':Path('source.py').write_text('changed')
else:
    source=next(Path(a.replace('\\',os.sep)) for a in args if a.endswith('.json') and '--' not in a)
    code=2 if fault=='audit_exit' else 0
    d={'protocol':'g3-evidence-2','observations_complete':fault!='incomplete',
       'g3_interpretable':False,'rules_sha256':args[args.index('--rules-sha256')+1],
       'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    if fault=='audit_id':d['source_sha256']='wrong'
if fault!='missing':out.write_text(json.dumps(d))
sys.exit(code)
'''


def run_job(tmp_path: Path, *, fault: str = "", code: int = 0, power: str = "ac",
            bad_rules: bool = False, model_problem: str = "") -> tuple[int, str, list[str]]:
    root = tmp_path / "repo"
    later = root / "_jobs" / "later"
    later.mkdir(parents=True)
    source = root / "source.py"
    source.write_bytes(b"known source")
    model = root / "fixture.gguf"
    model.write_bytes(b"fake model, never loaded")
    rules = {"python_files": {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}, "engine_files": {},
             "model_file": "fixture.gguf", "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    local_rules = later / RULES.name
    local_rules.write_text(json.dumps(rules), encoding="utf-8")
    digest = hashlib.sha256(local_rules.read_bytes()).hexdigest()
    text = re.sub(r"\$rulesSha = '[^']+'", f"$rulesSha = '{digest}'", JOB.read_text(encoding="ascii"))
    local_job = later / JOB.name
    local_job.write_text(text, encoding="ascii")
    if bad_rules:
        local_rules.write_text("tampered", encoding="ascii")
    if model_problem == "changed":
        model.write_bytes(b"replacement")
    elif model_problem == "missing":
        model.unlink()
    launcher = _launcher(tmp_path, "fake_python", FAKE)
    query = _launcher(tmp_path, "fake_power", "import os\nprint(os.environ['G3_POWER'])\n")
    log = tmp_path / "calls.txt"
    command = [str(SHELL), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(local_job),
               "-Root", str(root), "-Python", str(launcher), "-PowerStateScript", str(query), "-PowerWaitSeconds", "0"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=60,
                          env={**os.environ, "G3_CALLS": str(log), "G3_POWER": power, "G3_FAULT": fault,
                               "G3_EXIT": str(code)})
    calls = log.read_text().splitlines() if log.exists() else []
    return done.returncode, done.stdout + done.stderr, calls


def test_frozen_wrapper_inputs_match_and_original_judge_remains_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    rules = json.loads(RULES.read_bytes())
    digest = hashlib.sha256(RULES.read_bytes()).hexdigest()
    assert f"$rulesSha = '{digest}'" in JOB.read_text(encoding="ascii")
    assert hashlib.sha256(rules["original_judge_rules_text"].encode()).hexdigest() == rules["original_judge_rules_sha256"]
    for group in ("python_files", "engine_files"):
        for path, expected in rules[group].items():
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path


def test_job_success_only_means_operational_completion(tmp_path: Path) -> None:
    code, output, calls = run_job(tmp_path)
    assert code == 0 and len(calls) == 2, output
    assert "G3 interpretable: False" in output and "Independent review required" in output
    evaluation, audit = calls
    assert "--capture-load-evidence" in evaluation and "--blocks 3" in evaluation
    assert "--cases correctness/regression-pump-b-unit-7" in evaluation
    assert "--chat-exposure" not in evaluation and "--no-power-check" not in evaluation
    assert "audit_prompt_reuse_evidence.py" in audit


@pytest.mark.parametrize("power", ["battery", "unknown", "", "ac\nnoise"])
def test_power_is_fail_closed_before_dispatch(tmp_path: Path, power: str) -> None:
    code, _, calls = run_job(tmp_path, power=power)
    assert code == 4 and not calls


def test_changed_rules_refuse_before_any_dispatch(tmp_path: Path) -> None:
    code, _, calls = run_job(tmp_path, bad_rules=True)
    assert code == 4 and not calls


@pytest.mark.parametrize("problem", ["changed", "missing"])
def test_model_identity_refuses_before_python_dispatch(tmp_path: Path, problem: str) -> None:
    code, output, calls = run_job(tmp_path, model_problem=problem)
    assert code == 4 and not calls and "REFUSED" in output


@pytest.mark.parametrize("fault", ["id", "version", "rules", "simulation", "changed_input", "missing"])
def test_contradictory_or_absent_results_do_not_reach_analysis(tmp_path: Path, fault: str) -> None:
    code, _, calls = run_job(tmp_path, fault=fault)
    assert code == 2 and len(calls) == 1


@pytest.mark.parametrize("fault", ["incomplete", "audit_id", "audit_exit", "status"])
def test_no_success_on_incomplete_or_unbound_audit(tmp_path: Path, fault: str) -> None:
    code, output, calls = run_job(tmp_path, fault=fault)
    assert code == 2 and len(calls) == 2 and "Operational observations complete" not in output


@pytest.mark.parametrize("exit_code", [2, 3, 4, 5, 6, 7, 9])
def test_stopped_work_is_only_audited_never_followed_by_more_engine_work(tmp_path: Path, exit_code: int) -> None:
    code, _, calls = run_job(tmp_path, code=exit_code)
    assert code == 2 and len(calls) == 2
    assert sum("evaluate_prompt_reuse.py" in call for call in calls) == 1
