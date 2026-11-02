"""End-to-end native callers with small fake assets and no process creation."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import threading
import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock,patch
import pytest

from inferenceos.tune import lookup_answers_native as native
from inferenceos.tune import lookup_answers_runner as runner
from inferenceos.tune import lookup_answers_study as study
from inferenceos.tune.lookup_answer_publication import encoded,sha
from inferenceos.tune.lookup_answers_manifest import parse_answer_manifest
from inferenceos.tune.lookup_answers_sim import AnswerSimulator,FakeClock
from tests.test_lookup_answers_manifest import bundle,parsed  # noqa: F401; shared tiny-asset fixture


class FakeFitOwner:
    def __init__(self, preflight: Any, clock: FakeClock, data: dict[str,Any], *, wrong: bool) -> None:
        self.preflight,self.clock,self.data=preflight,clock,data
        self.engine=AnswerSimulator(clock);self.cancelled=threading.Event();self.owed=False
        self.model='';self.arm='';self.wrong=wrong;self.retry_calls=0
        self.backend=SimpleNamespace(is_active=lambda:self.owed,unconfirmed_pid=None,
                                     api_key_file=None,undeleted_key_files=[],base_url='http://127.0.0.1:17001')
    def start(self, model: Any,arm: str,*,deadline: float | None = None) -> None:
        assert deadline is not None
        self.model,self.arm=model.label,arm
        self.engine.start(model.label,arm);self.owed=True
        key=self.data['root']/'fake-study-password.txt';key.write_text('fixture-key',encoding='ascii')
        self.backend.api_key_file=key
    def observe(self) -> dict[str,Any]:
        source=next(r for r in self.data['record']['loads'] if r['model']==self.model and r['arm']==self.arm)
        snapshot: dict[str,Any]=deepcopy(source['snapshots'][0]);snapshot.pop('phase',None)
        snapshot['image'].update(pid=100000+self.engine.sim.starts,create_time_s=200000+self.engine.sim.starts)
        return snapshot
    def tokenize(self,text: str,*,add_special: bool) -> list[int]:
        return self.engine.tokenize(text,add_special=add_special)
    def send(self,body: dict[str,Any],receive: Any) -> None:
        def chunk(data: Any) -> None:
            receive(dict(data,content='wrong') if self.wrong and data.get('content') else data)
        self.engine.answer(body,chunk)
    def release(self,*,abort: bool) -> dict[str,Any]:
        receipt=self.engine.release(abort=abort)
        if self.backend.api_key_file:self.backend.api_key_file.unlink()
        self.backend.api_key_file=None;self.owed=False
        return receipt
    def retry_held(self) -> dict[str,Any] | None:
        self.retry_calls+=1
        return self.release(abort=True) if self.owed else None


def invoke(data: dict[str,Any],monkeypatch: pytest.MonkeyPatch,*,wrong: bool=True,
           power: Any=lambda:'ac',fault: str | None=None) -> tuple[int,Path,list[FakeFitOwner]]:
    clock=FakeClock();owners=[];out=data['root']/'answers.json'
    def factory(preflight: Any) -> FakeFitOwner:
        owner=FakeFitOwner(preflight,clock,data,wrong=wrong);owners.append(owner)
        if fault=='answer':
            def fail(*args: Any,**kwargs: Any) -> Any:raise RuntimeError('request defect')
            owner.send=fail  # type: ignore[method-assign]
        return owner
    def stream(url: str,key: str,body: dict[str,Any],chunk: Any,**kwargs: Any) -> None:
        assert key=='fixture-key';owners[-1].send(body,chunk)
    monkeypatch.setattr(native,'stream_completion',stream)
    with patch('subprocess.Popen',side_effect=AssertionError('no native process')):
        code=runner.run_native(root=data['root'],settings=data['settings'],rules=data['rules'],rules_sha=sha(data['rules'].read_bytes()),
                               out=out,owner_authorized=True,preflight_factory=data['runner'].Preflight,
                               fit_engine_factory=factory,power=power,clock=clock,rest=lambda n:clock.advance(n*1_000_000_000),execution_id='RUNNER_TEST')
    return code,out,owners


def audit(data: dict[str,Any],out: Path) -> dict[str,Any]:
    return study.audit_native_study(out.read_bytes(),out.with_name(out.name+'.closure.json').read_bytes(),
                                   out.with_name(out.name+'.journal').read_bytes(),parsed(data))


def test_real_preflight_native_adapter_journal_and_offline_audit_chain(bundle: Any,monkeypatch: pytest.MonkeyPatch) -> None:
    source=next(iter(bundle['original'].python_files))
    assert bundle['manifest']['python_files'][source]!=bundle['original'].python_files[source]
    code,out,owners=invoke(bundle,monkeypatch)
    assert code==0
    result=audit(bundle,out)
    assert result['evidence_valid'] and result['study_complete'],result
    assert all(not v['development_criteria_pass'] for v in result['analysis']['cells'].values())
    assert owners[0].engine.answers==owners[0].engine.sim.starts==owners[0].engine.sim.releases==6
    assert owners[0].retry_calls==1 and not owners[0].owed and not (bundle['root']/'fake-study-password.txt').exists()
    assert not result['actual_qualification'] and not result['forecast_available']


@pytest.mark.parametrize('fault',('historical_pin','false_pin','missing_pin','bytes','deleted','extra'))
def test_changed_historical_source_still_requires_exact_current_pins_before_owner(bundle: Any,monkeypatch: pytest.MonkeyPatch,fault: str) -> None:
    from tests.test_lookup_answers_manifest import encoded
    path=next(iter(bundle['original'].python_files))
    if fault=='historical_pin':bundle['manifest']['python_files'][path]=bundle['original'].python_files[path]
    elif fault=='false_pin':bundle['manifest']['python_files'][path]='0'*64
    elif fault=='missing_pin':bundle['manifest']['python_files'].pop(path)
    elif fault=='bytes':(bundle['root']/path).write_bytes(b'unreviewed execution source')
    elif fault=='deleted':(bundle['root']/path).unlink()
    elif fault=='extra':(bundle['root']/'src/unlisted.py').write_bytes(b'unlisted')
    bundle['rules'].write_bytes(encoded(bundle['manifest']))
    factory=Mock(side_effect=AssertionError('zero native owner constructions'))
    out=bundle['root']/'source_refused.json'
    with patch('subprocess.Popen',side_effect=AssertionError('no process')):
        code=runner.run_native(root=bundle['root'],settings=bundle['settings'],rules=bundle['rules'],
                               rules_sha=sha(bundle['rules'].read_bytes()),out=out,owner_authorized=True,
                               preflight_factory=bundle['runner'].Preflight,fit_engine_factory=factory,power=lambda:'ac')
    assert code==4
    factory.assert_not_called()
    record=json.loads(out.read_bytes())
    assert record['status']=='refused' and record['core'] is None and not record['actual_qualification']


def test_complete_1344_fake_native_requests_use_same_scoped_executor_and_public_audit(bundle: Any,monkeypatch: pytest.MonkeyPatch) -> None:
    code,out,owners=invoke(bundle,monkeypatch,wrong=False)
    assert code==0
    result=audit(bundle,out)
    assert result['evidence_valid'] and result['study_complete'],result
    assert owners[0].engine.sim.starts==owners[0].engine.sim.releases==owners[0].engine.answers==1344
    assert all(v['development_criteria_pass'] for v in result['analysis']['cells'].values())
    assert not result['actual_qualification'] and not result['automatic_admission'] and not result['deployment_allowed']
    cli=importlib.import_module('audit_lookup_answers')
    monkeypatch.setattr(cli,'ROOT',bundle['root'])
    report=bundle['root']/'public-audit.json'
    assert cli.main([str(out),'--rules',str(bundle['rules']),'--rules-sha256',sha(bundle['rules'].read_bytes()),'--out',str(report)])==0
    assert json.loads(report.read_bytes())['evidence_valid']


@pytest.mark.parametrize('fault',('rules_hash','model_bytes','model_deleted','saved_setup','source_deleted','artifact','battery'))
def test_actual_preflight_refusals_create_no_backend(bundle: Any,monkeypatch: pytest.MonkeyPatch,fault: str) -> None:
    factory=Mock(side_effect=AssertionError('must not construct'))
    inputs=parsed(bundle);rules_sha=sha(bundle['rules'].read_bytes())
    if fault=='rules_hash':rules_sha='0'*64
    elif fault in ('model_bytes','model_deleted'):
        path=bundle['root']/inputs.manifest.models[0].path
        if fault=='model_deleted':path.unlink()
        else:path.write_bytes(path.read_bytes()+b'changed')
    elif fault=='saved_setup':(bundle['root']/'data/saved_setups.json').write_bytes(b'{}')
    elif fault=='source_deleted':(bundle['root']/'scripts/current.py').unlink()
    elif fault=='artifact':(bundle['root']/inputs.manifest.stage_a_review.path).write_bytes(b'{}')
    with patch('subprocess.Popen',side_effect=AssertionError('no process')):
        code=runner.run_native(root=bundle['root'],settings=bundle['settings'],rules=bundle['rules'],rules_sha=rules_sha,
                               out=bundle['root']/'refused.json',owner_authorized=True,preflight_factory=bundle['runner'].Preflight,
                               fit_engine_factory=factory,power=lambda:'battery' if fault=='battery' else 'ac')
    assert code==4
    factory.assert_not_called()
    record=json.loads((bundle['root']/'refused.json').read_bytes())
    assert record['status']=='refused' and not record['actual_qualification']


def test_request_failure_releases_once_keeps_partial_journal_and_has_no_later_load(bundle: Any,monkeypatch: pytest.MonkeyPatch) -> None:
    code,out,owners=invoke(bundle,monkeypatch,fault='answer')
    assert code==2 and owners[0].engine.sim.starts==owners[0].engine.sim.releases==1
    assert len(json.loads(out.read_bytes())['core']['rows'])==1
    result=audit(bundle,out)
    assert not result['evidence_valid'] and not result['study_complete'] and not result['analysis']['cells']


@pytest.mark.parametrize('args',([],['--rules-sha256','0'*64],['--authorized-stage-b']))
def test_public_native_cli_refuses_without_explicit_frozen_inputs_and_authorization(tmp_path: Path,args: list[str]) -> None:
    cli=importlib.import_module('evaluate_lookup_answers')
    with patch.object(runner,'run_native',side_effect=AssertionError('no native caller')):
        assert cli.main(['--rules',str(tmp_path/'missing-rules.json'),'--out',str(tmp_path/'new.json'),*args])==4
    assert not (tmp_path/'new.json').exists()


def test_public_native_cli_passes_only_explicit_authorized_rules_to_owned_runner(tmp_path: Path) -> None:
    cli=importlib.import_module('evaluate_lookup_answers')
    with patch.object(runner,'run_native',return_value=2) as call:
        assert cli.main(['--rules',str(tmp_path/'rules.json'),'--rules-sha256','1'*64,
                         '--authorized-stage-b','--out',str(tmp_path/'new.json')])==2
    assert call.call_args.kwargs['owner_authorized'] is True and call.call_args.kwargs['rules_sha']=='1'*64
    assert not (tmp_path/'new.json').exists()


@pytest.mark.parametrize('fault',('error','prefix_missing','prefix_other_running','receipt_run','journal','caller_fail','origin','permission','provenance','missing_cleanup','core_stop','core_analysis'))
def test_native_envelope_cannot_hide_failures_or_mix_closure(bundle: Any,monkeypatch: pytest.MonkeyPatch,fault: str) -> None:
    _,out,_=invoke(bundle,monkeypatch);record=json.loads(out.read_bytes())
    closure=json.loads(out.with_name(out.name+'.closure.json').read_bytes());journal=out.with_name(out.name+'.journal').read_bytes()
    if fault=='error':record['error']=dict(type='OSError',message='failed')
    elif fault=='prefix_missing':record['caller_intervals'][0]={'end_ns':None,'outcome':'running'}
    elif fault=='prefix_other_running':record['caller_intervals'][2].update(end_ns=None,outcome='running')
    elif fault=='receipt_run':closure['run_id']='another'
    elif fault=='journal':journal+=b'{}\n'
    elif fault=='caller_fail':closure['caller_intervals'][-1]['outcome']='failed'
    elif fault=='origin':record['core']['target_origin_ns']+=1
    elif fault=='permission':closure['actual_qualification']=True
    elif fault=='provenance':record['provenance']['current_python_files_sha256']='0'*64
    elif fault=='missing_cleanup':record.pop('final_cleanup')
    elif fault in ('core_stop','core_analysis'):
        # Keep the marker in every checkpoint so no journal replacement is
        # involved. Even consistent retained evidence of failure must refuse.
        field='stop' if fault=='core_stop' else 'analysis_problem'
        record['core'][field]=dict(type='RuntimeError',message='retained failure')
        packets=[json.loads(line) for line in journal.splitlines()]
        previous='0'*64;lines=[]
        for packet in packets:
            packet['event']['meta'][field]=dict(record['core'][field])
            packet['previous_sha256']=previous
            line=encoded(packet);previous=sha(line);lines.append(line)
        journal=b''.join(lines)
        closure.update(journal_sha256=sha(journal),journal_bytes=len(journal))
    record_raw=encoded(record);closure['record_sha256']=sha(record_raw)
    result=study.audit_native_study(record_raw,encoded(closure),journal,parsed(bundle))
    assert not result['evidence_valid'] and not result['study_complete'] and not result['analysis']['cells']
