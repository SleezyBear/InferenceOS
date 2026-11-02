"""Adversarial software-only Stage-B scheduler, raw evidence and accounting checks."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import pytest

from inferenceos.tune.lookup_answers import (VERSION,SOFT_TARGET_NS,plan,suites,request_body,native_answer,analyze,scope_problems)
from inferenceos.tune.lookup_answers_collect import collect_answers,NO_CHECKPOINT
from inferenceos.tune.lookup_answers_sim import AnswerSimulator,FakeClock,fixture_scope,simulate_answers
from inferenceos.tune.lookup_answers_audit import audit_answers
from inferenceos.tune.lookup_corpus import build_corpus,digest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def run(engine: Any=None,*,clock: Any=None,power: Any=lambda:'ac',guard: Any=lambda:None,
        checkpoint: Any=NO_CHECKPOINT) -> tuple[dict[str,Any],AnswerSimulator]:
    clock=clock or FakeClock();engine=engine or AnswerSimulator(clock)
    raw=collect_answers(engine,fixture_scope(),run_id='test',clock=clock,power=power,guard=guard,
                        rest=lambda seconds:clock.advance(seconds*1_000_000_000),checkpoint=checkpoint,simulated=True)
    return raw,engine


def wrong_engine(clock: FakeClock) -> AnswerSimulator:
    engine=AnswerSimulator(clock);original=engine.answer
    def answer(body: Any,chunk: Any) -> None:
        def receive(c: Any) -> None:
            if c.get('content'):c=dict(c,content='incorrect')
            chunk(c)
        original(body,receive)
    engine.answer=answer  # type: ignore[method-assign]
    return engine


@pytest.fixture(scope='module')
def full() -> dict[str,Any]:
    with patch('subprocess.Popen',side_effect=AssertionError('no native process')):
        return simulate_answers()


@pytest.fixture(scope='module')
def futile() -> dict[str,Any]:
    clock=FakeClock();raw,_=run(wrong_engine(clock),clock=clock)
    return raw


def test_exact_registered_population_orders_suites_and_balancing() -> None:
    schedule=plan();definitions=suites()
    # Golden digests from the accepted protocol-2 proposal; no ignored artifact is
    # required to run the product suite on another checkout.
    assert digest(schedule)=='920c394cb9050a7c5cd8dc0b73ba95c4cae64f962615e486aea06b5d81d66543'
    assert digest(definitions)=='8eb43663f6290ef7c40d3c21b9cb574f7c00310f0a7877a2e16bf38219bbb82b'
    assert len(schedule)==1344 and len(definitions)==6
    counts=Counter((r['model'],r['case_id'],r['arm']) for r in schedule)
    assert len(counts)==168 and set(counts.values())=={8}
    first=Counter((schedule[i]['cell'],schedule[i]['role'],schedule[i]['arm']) for i in range(0,1344,14))
    assert len(first)==24 and set(first.values())=={4}


def test_full_software_execution_and_audit_never_grant_real_permission(full: dict[str,Any]) -> None:
    assert full['status']=='done' and len(full['rows'])==1344 and not full['not_run']
    assert len(full['rests'])==48 and len(full['pair_admissions'])==96
    assert full['simulated'] and not full['hardware_or_native_meaning'] and full['native_engine_starts']==0
    audit=audit_answers(full)
    assert audit['evidence_valid'],audit
    assert all(c['development_criteria_pass'] for c in audit['analysis']['cells'].values())
    assert not audit['actual_qualification'] and not audit['automatic_admission'] and not audit['deployment_allowed']
    assert audit['costs']['distributions']['suite-arm']['count']==192
    assert audit['costs']['distributions']['cycle']['count']==1344
    assert all(len(row['events'])==2 and row['complete'] for row in full['rows'])


def test_live_dispatch_is_unconditionally_refused_before_start() -> None:
    clock=FakeClock();engine=AnswerSimulator(clock)
    with pytest.raises(ValueError,match='live'):
        collect_answers(engine,fixture_scope(),run_id='x',clock=clock,power=lambda:'ac',guard=lambda:None,
                        rest=lambda _:None,checkpoint=NO_CHECKPOINT,simulated=False)
    assert engine.sim.starts==0


@pytest.mark.parametrize('field',tuple(fixture_scope()['models']['llama3b']['requested']))
def test_requested_and_observed_field_omissions_refuse_before_start(field: str) -> None:
    for side in ('requested','effective'):
        scope=fixture_scope();scope['models']['llama3b'][side].pop(field)
        assert scope_problems(scope)


def test_cell_futility_is_not_whole_study_failure_and_every_slot_remains(futile: dict[str,Any]) -> None:
    assert futile['status']=='done' and len(futile['futile'])==6
    assert len(futile['rows'])==6 and len(futile['not_run'])==1338 and len(futile['rests'])==48
    assert set(futile['futile'][k]['cause'] for k in futile['futile'])=={'task-unqualified control','lookup wrong; paired control unobserved'}
    audited=audit_answers(futile);assert audited['evidence_valid'],audited
    assert not any(c['development_criteria_pass'] for c in audited['analysis']['cells'].values())


def native_fixture() -> tuple[Any,Any,Any,Any]:
    clock=FakeClock();engine=AnswerSimulator(clock);case=build_corpus().cases[0]
    engine.start('llama3b','lookup');ids=engine.tokenize(case.prompt,add_special=True);chunks: list[dict[str,Any]]=[]
    engine.answer(request_body(case),chunks.append)
    events=[dict(received_ns=10+i,data=c) for i,c in enumerate(chunks)]
    return case,ids,events,request_body(case)


@pytest.mark.parametrize('fault',('tokens_evaluated','prompt_n','predicted_n','tokens_predicted','cache_n','stop_type','truncated',
                                'missing_terminal','duplicate_terminal','lost_incremental_ids','bool_id','float_id','id_slot','unknown_slot',
                                'missing_draft','null_drafts','bad_drafts','negative_drafts','settings','seed','temperature','n_predict',
                                'missing_timings','zero_time','nan_time','wrong_prompt','reuse_request','late_chunk','reordered_chunk','body_extra',
                                'early_counter','early_reuse'))
def test_native_raw_work_and_effective_settings_fail_closed(fault: str) -> None:
    case,ids,events,body=native_fixture();terminal=events[-1]['data']
    assert native_answer(events,body,case,'lookup',ids,0,100)['valid']
    if fault in ('tokens_evaluated','tokens_predicted'):terminal[fault]+=1
    elif fault in ('prompt_n','predicted_n','cache_n'):terminal['timings'][fault]+=1
    elif fault=='stop_type':terminal['stop_type']='limit'
    elif fault=='truncated':terminal['truncated']=True
    elif fault=='missing_terminal':events.pop()
    elif fault=='duplicate_terminal':events.append(deepcopy(events[-1]))
    elif fault=='lost_incremental_ids':events[0]['data']['tokens']=[]
    elif fault=='bool_id':events[0]['data']['tokens'][0]=True
    elif fault=='float_id':events[0]['data']['tokens'][0]=1.0
    elif fault=='id_slot':terminal['id_slot']=True
    elif fault=='unknown_slot':terminal['id_slot']=4
    elif fault=='missing_draft':terminal['timings'].pop('draft_n')
    elif fault=='null_drafts':terminal['timings'].update(draft_n=None,draft_n_accepted=None)
    elif fault=='bad_drafts':terminal['timings']['draft_n_accepted']=999
    elif fault=='negative_drafts':terminal['timings']['draft_n']=-1
    elif fault=='settings':terminal['generation_settings']['speculative.types']='other'
    elif fault in ('seed','temperature','n_predict'):terminal['generation_settings'][fault]=1
    elif fault=='missing_timings':terminal.pop('timings')
    elif fault=='zero_time':terminal['timings']['predicted_ms']=0
    elif fault=='nan_time':terminal['timings']['prompt_ms']=float('nan')
    elif fault=='wrong_prompt':body['prompt']='changed'
    elif fault=='reuse_request':body['cache_prompt']=True
    elif fault=='late_chunk':events[-1]['received_ns']=101
    elif fault=='reordered_chunk':events[-1]['received_ns']=9
    elif fault=='body_extra':body['grammar']='unregistered'
    elif fault=='early_counter':events[0]['data']['tokens_predicted']=True
    elif fault=='early_reuse':events[0]['data']['timings']={'cache_n':1}
    assert not native_answer(events,body,case,'lookup',ids,0,100)['valid']


def test_terminal_empty_id_array_is_not_empty_total_and_missing_drafts_are_not_gains() -> None:
    case,ids,events,body=native_fixture()
    assert events[-1]['data']['tokens']==[]
    result=native_answer(events,body,case,'lookup',ids,0,100)
    assert result['valid'] and result['token_ids']==events[0]['data']['tokens']
    for k in ('draft_n','draft_n_accepted'):events[-1]['data']['timings'].pop(k)
    result=native_answer(events,body,case,'lookup',ids,0,100)
    assert result['valid'] and result['accepted'] is None


def test_empty_but_counted_natural_completion_remains_wrong_task_not_invented_measurement() -> None:
    case,ids,events,body=native_fixture()
    events[0]['data'].update(content='',tokens=[])
    events[-1]['data']['tokens_predicted']=0
    events[-1]['data']['timings'].update(predicted_n=0,draft_n=0,draft_n_accepted=0)
    result=native_answer(events,body,case,'lookup',ids,0,100)
    assert result['valid'] and result['text']=='' and result['token_ids']==[]


@pytest.mark.parametrize('stage',('start','ready','tokenize','answer','before_stop','release','post_power','guard'))
@pytest.mark.parametrize('interrupt',(False,True))
def test_operational_failures_keep_original_interrupt_and_never_dispatch_later(stage: str,interrupt: bool) -> None:
    clock=FakeClock();engine=AnswerSimulator(clock);boom=KeyboardInterrupt('original') if interrupt else RuntimeError('original');snapshots: list[dict[str,Any]]=[]
    def fail(*a: Any,**k: Any) -> Any:raise boom
    if stage in ('start','tokenize','answer','release'):setattr(engine,stage,fail)
    if stage in ('ready','before_stop'):
        observe=engine.observe;calls=0
        def observation() -> Any:
            nonlocal calls
            calls+=1
            if calls==(1 if stage=='ready' else 2):raise boom
            return observe()
        engine.observe=observation  # type: ignore[method-assign]
    pc=0
    def power() -> str:
        nonlocal pc
        pc+=1
        if stage=='post_power' and pc==2:raise boom
        return 'ac'
    if interrupt:
        with pytest.raises(KeyboardInterrupt) as caught:run(engine,clock=clock,power=power,guard=fail if stage=='guard' else lambda:None,checkpoint=snapshots.append)
        assert caught.value is boom
    else:
        raw,_=run(engine,clock=clock,power=power,guard=fail if stage=='guard' else lambda:None,checkpoint=snapshots.append)
        assert raw['status']=='stopped'
    last=snapshots[-1]
    assert last['status']=='stopped' and len(last['rows'])==1 and len(last['not_run'])==1343
    assert engine.sim.starts<=1 and engine.answers<=1
    assert last['analysis']['analysis_looks']==0


@pytest.mark.parametrize('fault',('duplicate','reorder','missing','native_flag','verdict','event','raw_ids','cycle_ns','scope',
                                'release','power','not_run','futility','rest','admission','unknown_slot','cost_parent','run_id',
                                'failed_job','failed_gate','failed_pair','publication','future_admission','model_scope','engine_scope'))
def test_saved_record_auditor_reconstructs_and_refuses_corruption(futile: dict[str,Any],fault: str) -> None:
    raw=deepcopy(futile);row=raw['rows'][0]
    if fault=='duplicate':raw['rows'].append(deepcopy(row))
    elif fault=='reorder':raw['rows'][0],raw['rows'][1]=raw['rows'][1],raw['rows'][0]
    elif fault=='missing':raw['rows'].pop()
    elif fault=='native_flag':raw['simulated']=False
    elif fault=='verdict':row['verdict']='right'
    elif fault=='event':row['events'][-1]['data']['timings']['cache_n']=1
    elif fault=='raw_ids':row['events'][0]['data']['tokens'][0]+=1
    elif fault=='cycle_ns':row['cycle_ns']+=1
    elif fault=='scope':raw['scope']['models']['llama3b']['requested']['threads']=3
    elif fault=='release':row['cleanup']['credential_removed']=False
    elif fault=='power':row['power'][-1]='unknown'
    elif fault=='not_run':raw['not_run'].pop()
    elif fault=='futility':raw['futile'].clear()
    elif fault=='rest':raw['rests'][0]['seconds']=0
    elif fault=='admission':raw['pair_admissions'][0]['permitted']=False
    elif fault=='unknown_slot':row['slot']=9999
    elif fault=='cost_parent':raw['cost_intervals'][2]['parent_id']=None
    elif fault=='run_id':raw['run_id']='other'
    elif fault=='failed_job':raw['cost_intervals'][0]['outcome']='failed'
    elif fault=='failed_gate':next(c for c in raw['cost_intervals'] if c['kind']=='gate')['outcome']='failed'
    elif fault=='failed_pair':next(c for c in raw['cost_intervals'] if c['kind']=='pair')['outcome']='failed'
    elif fault=='publication':raw['publication_problem']=dict(type='OSError',message='checkpoint unavailable')
    elif fault=='future_admission':raw['pair_admissions'][-1]['at_ns']=raw['cost_intervals'][0]['end_ns']+1
    elif fault=='model_scope':next(c for c in raw['cost_intervals'] if c['kind']=='cycle')['scope']['model_sha256']='0'*64
    elif fault=='engine_scope':next(c for c in raw['cost_intervals'] if c['kind']=='cycle')['scope']['engine_files_sha256']='0'*64
    assert not audit_answers(raw)['evidence_valid']


@pytest.mark.parametrize('newline',(False,True))
def test_lookup_failure_uses_only_its_current_paired_control(newline: bool) -> None:
    clock=FakeClock();engine=AnswerSimulator(clock);original=engine.answer
    def answer(body: Any,chunk: Any) -> None:
        def receive(c: Any) -> None:
            if engine.sim.starts==8 and c.get('content'):
                c=dict(c,content=c['content']+'\n' if newline else 'incorrect')
            chunk(c)
        original(body,receive)
    engine.answer=answer  # type: ignore[method-assign]
    def guard() -> None:
        if engine.sim.starts>=8:raise RuntimeError('test ends after observed futility')
    raw,_=run(engine,clock=clock,guard=guard)
    reason=raw['futile']['llama3b/copy-prose']['cause']
    assert reason==('paired exact output preservation fails' if newline else 'candidate regression')
    assert raw['futile']['llama3b/copy-prose']['at_slot']==7 and engine.answers==8


def test_target_before_first_pair_starts_no_engine_and_takes_no_statistical_look() -> None:
    clock=FakeClock();calls=0
    def now() -> int:
        nonlocal calls
        calls+=1
        if calls==2:clock.advance(SOFT_TARGET_NS)
        return clock()
    engine=AnswerSimulator(clock)
    raw=collect_answers(engine,fixture_scope(),run_id='target',clock=now,power=lambda:'ac',guard=lambda:None,
                        rest=lambda _:None,checkpoint=NO_CHECKPOINT,simulated=True)
    assert raw['status']=='stopped' and not raw['rows'] and len(raw['not_run'])==1344
    assert engine.sim.starts==0 and raw['analysis']['analysis_looks']==0


def test_started_pair_can_finish_after_soft_target_and_next_pair_cannot_start() -> None:
    clock=FakeClock();engine=AnswerSimulator(clock);calls=0
    def now() -> int:
        nonlocal calls
        calls+=1
        if calls==2:clock.advance(SOFT_TARGET_NS-1_000_000_000)
        return clock()
    raw=collect_answers(engine,fixture_scope(),run_id='late_pair',clock=now,power=lambda:'ac',guard=lambda:None,
                        rest=lambda seconds:clock.advance(seconds*1_000_000_000),checkpoint=NO_CHECKPOINT,simulated=True)
    assert raw['status']=='stopped' and len(raw['rows'])==14 and engine.answers==14
    assert len(raw['pair_admissions'])==2 and raw['pair_admissions'][-1]['permitted'] is False


@pytest.mark.parametrize('stage',('start','ready','tokenize','answer','before_stop'))
def test_late_successful_operation_stops_before_any_subsequent_measurement(stage: str) -> None:
    clock=FakeClock();engine=AnswerSimulator(clock)
    if stage in ('start','tokenize','answer'):
        original=getattr(engine,stage)
        def delayed(*a: Any,**kw: Any) -> Any:
            result=original(*a,**kw)
            clock.advance(120_000_000_000 if stage in ('start','answer') else 30_000_000_000)
            return result
        setattr(engine,stage,delayed)
    else:
        observe=engine.observe;calls=0
        def observation() -> Any:
            nonlocal calls
            calls+=1
            result=observe()
            if calls==(1 if stage=='ready' else 2):clock.advance(30_000_000_000)
            return result
        engine.observe=observation  # type: ignore[method-assign]
    raw,_=run(engine,clock=clock)
    assert raw['status']=='stopped' and len(raw['rows'])==1 and len(raw['not_run'])==1343
    assert engine.sim.releases==1 and not engine.sim.active
    assert engine.answers==(1 if stage in ('answer','before_stop') else 0)


@pytest.mark.parametrize('screen',('zero_variance','calibration_harm','case_harm','missing_draft','incomplete'))
def test_registered_conservative_screens_cannot_be_rescued_by_positive_average(full: dict[str,Any],screen: str) -> None:
    raw=dict(full,rows=[dict(r,answer=dict(r['answer'])) for r in full['rows']])
    first_case=plan()[0]['case_id']
    for row in raw['rows']:
        effect=.2+.001*row['block']
        if screen=='zero_variance':effect=.2
        if screen=='calibration_harm' and row['role']=='calibration':effect=-.03+.001*row['block']
        if screen=='case_harm' and row['case_id']==first_case:effect=-.04+.001*row['block']
        row['cycle_ns']=1_000_000_000 if row['arm']=='control' else round(1_000_000_000*(1-effect))
        if screen=='missing_draft':row['answer']['accepted']=None
    if screen=='incomplete':raw['status']='stopped'
    cells=analyze(raw)['cells']
    if screen=='case_harm':assert not cells['llama3b/copy-prose']['development_criteria_pass']
    else:assert not any(c['development_criteria_pass'] for c in cells.values())


@pytest.mark.parametrize('effect',(-.05,0,.03,.20))
def test_single_final_analysis_rejects_flat_negative_and_below_minimum(full: dict[str,Any],effect: float) -> None:
    raw=dict(full,rows=[dict(r) for r in full['rows']]);offsets=(-1.1,.7,-.4,1.2,-.8,.5,-.2,.1)
    for row in raw['rows']:
        row['cycle_ns']=1_000_000_000 if row['arm']=='control' else round(1_000_000_000*(1-effect-.002*offsets[row['block']]))
    result=analyze(raw)
    assert all(c['development_criteria_pass']==(effect==.20) for c in result['cells'].values())
    assert not result['actual_qualification']


def test_simulation_cli_and_public_auditor_preserve_input_and_refuse_existing_output(tmp_path: Path,futile: dict[str,Any]) -> None:
    runner=importlib.import_module('evaluate_lookup_answers');auditor=importlib.import_module('audit_lookup_answers')
    record=tmp_path/'r.json';audit=tmp_path/'audit.json'
    with patch.object(runner,'simulate_answers',return_value=deepcopy(futile)),patch('subprocess.Popen',side_effect=AssertionError('no process')):
        assert runner.main(['--simulate','--out',str(record)])==0
        before=record.read_bytes()
        assert runner.main(['--simulate','--out',str(record)])==4
        assert auditor.main([str(record),'--out',str(audit)])==0
        assert auditor.main([str(record),'--out',str(audit)])==4
        assert record.read_bytes()==before
    with pytest.raises(SystemExit):runner.main(['--out',str(tmp_path/'no.json')])


@pytest.mark.parametrize('interrupt',(False,True))
def test_final_analysis_failure_still_checkpoints_and_preserves_original_interrupt(interrupt: bool) -> None:
    clock=FakeClock();engine=AnswerSimulator(clock);original=engine.answer;checkpoints: list[dict[str,Any]]=[]
    boom=KeyboardInterrupt('final analysis interrupted') if interrupt else RuntimeError('final analysis failed')
    def wrong(body: Any,chunk: Any) -> None:
        def receive(data: Any) -> None:
            chunk(dict(data,content='wrong') if data.get('content') else data)
        original(body,receive)
    engine.answer=wrong  # type: ignore[method-assign]
    with patch('inferenceos.tune.lookup_answers_collect.analyze',side_effect=boom):
        if interrupt:
            with pytest.raises(KeyboardInterrupt) as caught:run(engine,clock=clock,checkpoint=checkpoints.append)
            assert caught.value is boom
        else:
            result,_=run(engine,clock=clock,checkpoint=checkpoints.append)
            assert result['status']=='stopped'
    last=checkpoints[-1]
    assert last['status']=='stopped' and last['stop']['stage']=='final-analysis'
    assert last['analysis']['analysis_looks']==0 and last['analysis']['analysis_complete'] is False
    assert len(last['rows'])==6 and engine.sim.starts==engine.sim.releases==6 and not engine.sim.active
    assert not audit_answers(last)['evidence_valid']
