"""Execute prospective job130 using fake programs; never dispatch a model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import pytest

from tests.test_job_scripts import SHELL, LATER, _launcher

JOB = LATER / "130_lookup_decoding_study.ps1"
RULES = LATER / "130_rules_lookup-study-1-r5.json"
pytestmark = pytest.mark.skipif(not JOB.is_file() or SHELL is None, reason="prepared job130/PowerShell required")

FAKE = r'''
import hashlib,json,os,sys
from pathlib import Path
a=sys.argv[1:]; fault=os.environ.get('LOOKUP_FAULT','')
with open(os.environ['LOOKUP_CALLS'],'a') as h:h.write(' '.join(a)+'\n')
out=Path(a[a.index('--out')+1].replace('\\',os.sep));out.parent.mkdir(parents=True,exist_ok=True)
if any(x.endswith('evaluate_lookup_decoding.py') for x in a):
 d={'run_id':a[a.index('--run-id')+1],'preregistration':{'sha256':a[a.index('--preregistration-sha256')+1]},
    'protocol':'lookup-study-1','version':'lookup-decoding-1','simulated':False,'status':'done',
    'model':str(Path(a[a.index('--model')+1]).resolve()),
    'model_file_identity':{'sha256':hashlib.sha256(Path(a[a.index('--model')+1]).read_bytes()).hexdigest()},
    'configuration_sha256':'f'*64}
 if fault=='run':d['run_id']='older'
 if fault=='version':d['version']='older'
 if fault=='rules':d['preregistration']['sha256']='older'
 if fault=='simulated':d['simulated']=True
 if fault=='other-model':d['model']=str(Path('models/b.gguf').resolve())
 if fault=='other-config':d['configuration_sha256']='0'*64
 if fault=='status':d['status']='running'
 if fault=='changed-source':Path('source.py').write_bytes(b'changed')
 if fault=='changed-model':Path('models/a.gguf').write_bytes(b'changed')
 code=2 if fault=='exit' else 0
else:
 source=next(Path(x.replace('\\',os.sep)) for x in a if x.endswith('.json'))
 d={'protocol':'lookup-study-1','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'rules_sha256':a[a.index('--rules-sha256')+1], 'observations_complete':True,
    'performance_interpretable':True,'automatic_admission':False}
 if fault=='audit-source':d['source_sha256']='old'
 if fault=='audit-rules':d['rules_sha256']='old'
 if fault=='audit-incomplete':d['observations_complete']=False
 if fault=='audit-unknown':d['performance_interpretable']=False
 if fault=='admit':d['automatic_admission']=True
 code=2 if fault=='audit-exit' else 0
if fault!='missing':out.write_text(json.dumps(d))
sys.exit(code)
'''


def run_job(tmp_path: Path, *, fault: str = "", power: str = "ac", changed: bool = False) -> tuple[int, str, list[str]]:
    root = tmp_path / "repo"
    later = root / "_jobs/later"
    later.mkdir(parents=True)
    (root / "models").mkdir()
    for name in ("models/a.gguf", "models/b.gguf", "source.py"):
        (root / name).write_bytes(b"fixture, never executed or loaded")
    sha = hashlib.sha256(b"fixture, never executed or loaded").hexdigest()
    rules = {"protocol": "lookup-study-1", "version": "lookup-decoding-1", "model_order": ["models/a.gguf", "models/b.gguf"],
             "models": {"models/a.gguf": {"label": "a", "sha256": sha, "configuration_sha256": "f"*64},
                        "models/b.gguf": {"label": "b", "sha256": sha, "configuration_sha256": "f"*64}},
             "python_files": {"source.py": sha}, "engine_files": {}}
    local_rules = later / RULES.name
    local_rules.write_text(json.dumps(rules))
    digest = hashlib.sha256(local_rules.read_bytes()).hexdigest()
    local_job = later / JOB.name
    local_job.write_text(re.sub(r"\$rulesSha = '[^']+'", f"$rulesSha = '{digest}'", JOB.read_text()), encoding="ascii")
    if changed: (root / "models/b.gguf").write_bytes(b"replacement")
    python = _launcher(tmp_path, "lookup_fake", FAKE)
    query = _launcher(tmp_path, "power_fake", "import os\nprint(os.environ['LOOKUP_POWER'])\n")
    calls = tmp_path / "calls.txt"
    result = subprocess.run([str(SHELL), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(local_job),
                             "-Root", str(root), "-Python", str(python), "-PowerStateScript", str(query),
                             "-PowerWaitSeconds", "0", "-RestSeconds", "0"], capture_output=True, text=True, timeout=60,
                            env={**os.environ, "LOOKUP_CALLS": str(calls), "LOOKUP_FAULT": fault, "LOOKUP_POWER": power})
    return result.returncode, result.stdout + result.stderr, calls.read_text().splitlines() if calls.exists() else []


def test_success_audits_both_models_without_tuning_or_saves(tmp_path: Path) -> None:
    code, text, calls = run_job(tmp_path)
    assert code == 0 and len(calls) == 4, text
    assert sum('evaluate_lookup_decoding.py' in line for line in calls) == 2
    assert sum('audit_lookup_decoding.py' in line for line in calls) == 2
    assert "Independent results review required" in text
    assert not any('--layers' in line or '--no-power-check' in line or '--simulate' in line for line in calls)


@pytest.mark.parametrize("power", ["battery", "unknown", "", "ac\nnoise"])
def test_bad_power_never_dispatches_python(tmp_path: Path, power: str) -> None:
    code, _, calls = run_job(tmp_path, power=power)
    assert code == 4 and not calls


def test_changed_model_refuses_before_any_python(tmp_path: Path) -> None:
    code, _, calls = run_job(tmp_path, changed=True)
    assert code == 4 and not calls


@pytest.mark.parametrize("fault", ["run", "version", "rules", "simulated", "other-model", "other-config", "changed-source", "changed-model", "missing"])
def test_bad_identity_or_changed_input_never_reaches_next_model(tmp_path: Path, fault: str) -> None:
    code, _, calls = run_job(tmp_path, fault=fault)
    assert code == 2 and len(calls) == 1


@pytest.mark.parametrize("fault", ["status", "exit", "audit-source", "audit-rules", "audit-incomplete", "audit-unknown", "admit", "audit-exit"])
def test_bad_completion_or_audit_stops_after_first_model(tmp_path: Path, fault: str) -> None:
    code, _, calls = run_job(tmp_path, fault=fault)
    assert code == 2 and len(calls) == 2


def test_real_rules_bind_current_sources_and_model_population() -> None:
    if not RULES.is_file(): pytest.skip("rules will be frozen after implementation checks")
    root = Path(__file__).resolve().parents[1]
    rules = json.loads(RULES.read_bytes())
    sha = hashlib.sha256(RULES.read_bytes()).hexdigest()
    assert f"$rulesSha = '{sha}'" in JOB.read_text()
    assert len(rules["models"]) == 2 and rules["parameters"]["tokens"] == 128
    for group in ("python_files", "engine_files"):
        for name, expected in rules[group].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
