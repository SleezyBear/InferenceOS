"""Durable delta checkpoints and truthful caller costs, without model IO."""
from __future__ import annotations
from copy import deepcopy
import json
import importlib
from pathlib import Path
from typing import Any
from unittest.mock import patch
import pytest

from inferenceos.tune import lookup_answer_publication as pub
from inferenceos.tune.lookup_answers_collect import collect_answers
from inferenceos.tune.lookup_answers_sim import AnswerSimulator,FakeClock,fixture_scope
from inferenceos.tune.lookup_corpus import digest
from tests.test_lookup_answers_manifest import bundle,parsed  # noqa: F401; shared synthetic assets


def row_fixture(*,complete: bool=False,slot: int=0) -> dict[str,Any]:
    return dict(slot=slot,load_id=f'changes/B/{slot}',power=['ac'],snapshots=[dict(phase='ready',identity=1)],
                prompt_ids=[11,12],events=[dict(received_ns=10,data=dict(content='a',generation_settings=dict(seed=None)))],
                request=dict(seed=42),errors=[],cleanup=None,verdict='incomplete',complete=complete,cycle_ns=None)


def checkpoint_fixture(rows: list[dict[str,Any]]) -> dict[str,Any]:
    return dict(run_id='changes',scope={},rows=rows,cost_intervals=[],status='running',rests=[],pair_admissions=[],
                futile={},not_run=[])


def rechain(packets: list[dict[str,Any]]) -> bytes:
    previous='0'*64;lines=[]
    for index,packet in enumerate(packets):
        packet.update(sequence=index,previous_sha256=previous)
        raw=pub.encoded(packet);lines.append(raw);previous=pub.sha(raw)
    return b''.join(lines)


def transition(tmp_path: Path,before: dict[str,Any],after: dict[str,Any],*,replay: bool) -> None:
    owned=pub.OwnedAnswerOutput(tmp_path/'changes.json','changes')
    try:
        owned(before)
        if not replay:
            owned(after)
            return
        first=json.loads(owned.journal_path.read_bytes())
        # Rehash the forged packet: refusal must depend on evidence progression,
        # not merely the journal chain or final-envelope hash.
        second=dict(event=dict(version=pub.JOURNAL_VERSION,run_id='changes',row_start=max(0,len(before['rows'])-1),
                               rows=deepcopy(after['rows'][max(0,len(before['rows'])-1):]),costs=[],
                               meta={k:v for k,v in after.items() if k not in ('scope','rows','cost_intervals')}))
        pub.replay_journal(rechain([first,second]),'changes')
    finally:owned.close()


def small_record(out: pub.OwnedAnswerOutput,clock: FakeClock,boundary: Any=None) -> dict[str,Any]:
    engine=AnswerSimulator(clock);answer=engine.answer
    def wrong(body: Any,chunk: Any) -> None:
        answer(body,lambda data:chunk(dict(data,content='wrong') if data.get('content') else data))
    engine.answer=wrong  # type: ignore[method-assign]
    return collect_answers(engine,fixture_scope(),run_id=out.run_id,clock=clock,power=lambda:'ac',guard=out.verify,
                           rest=lambda n:clock.advance(n*1_000_000_000),checkpoint=out,simulated=True,boundary=boundary)


@pytest.fixture
def published(tmp_path: Path) -> tuple[dict[str,Any],dict[str,Any],Path]:
    clock=FakeClock();ledger=pub.CallerLedger('journal',clock);path=tmp_path/'study.json'
    with ledger.measure('whole-job'):
        with ledger.measure('initial-output'):owned=pub.OwnedAnswerOutput(path,'journal')
        with ledger.measure('initial-validation'):pass
        with ledger.measure('engine-construction'):pass
        with ledger.measure('execution'):core=small_record(owned,clock,ledger.measure)
        with ledger.measure('final-cleanup'):pass
        with ledger.measure('preservation'):pass
        with ledger.measure('final-publication'):
            envelope=dict(version=pub.VERSION,run_id='journal',core=core,caller_intervals=ledger.snapshot())
            owned.publish(envelope)
        with ledger.measure('closure-validation'):owned.prepare_closure()
    receipt=owned.seal(status='done',caller_intervals=ledger.snapshot(),manifest_sha256='1'*64)
    return envelope,receipt,path


def test_delta_journal_reconstructs_complete_raw_work_and_separately_closes_publication(published: Any) -> None:
    envelope,receipt,path=published
    journal=path.with_name(path.name+'.journal').read_bytes()
    rebuilt=pub.replay_journal(journal,'journal')
    assert digest(rebuilt)==digest(envelope['core'])
    assert len(rebuilt['rows'])==6 and len(rebuilt['rests'])==48
    assert receipt['record_sha256']==pub.sha(path.read_bytes()) and receipt['journal_sha256']==pub.sha(journal)
    assert not pub.caller_cost_problems(receipt['caller_intervals'],'journal',complete=True)
    assert receipt['receipt_publication']==dict(outcome='right-censored',included_in_observed_job=False)
    packets=[json.loads(line) for line in journal.splitlines()]
    assert sum(len(packet['event']['rows']) for packet in packets)==6  # no repeated whole histories
    assert not any(receipt[k] for k in ('actual_qualification','automatic_admission','deployment_allowed'))


def test_replay_does_not_revalidate_history_outside_the_incoming_delta(published: Any) -> None:
    envelope,_,path=published
    raw=path.with_name(path.name+'.journal').read_bytes()
    # This normal completed-row journal only appends rows. Its prior prefix is
    # already decoded private state, so repeatedly traversing those observations
    # would add quadratic replay work without checking any supplied new evidence.
    with patch.object(pub,'_row_problem',side_effect=AssertionError('old private prefix traversed')):
        replayed=pub.replay_journal(raw,'journal')
    assert digest(replayed)==digest(envelope['core'])


@pytest.mark.parametrize('fault',('chain','run','sequence','cost_rewrite','row_rewrite','truncated','scope','duplicate_cost'))
def test_journal_refuses_mixing_rewrites_and_partial_tails(published: Any,fault: str) -> None:
    _,_,path=published;raw=path.with_name(path.name+'.journal').read_bytes()
    if fault=='truncated':bad=raw[:-3]
    else:
        packets=[json.loads(line) for line in raw.splitlines()]
        if fault=='chain':packets[1]['previous_sha256']='2'*64
        elif fault=='run':packets[1]['event']['run_id']='other'
        elif fault=='sequence':packets[1]['sequence']=0
        elif fault=='cost_rewrite':packets[1]['event']['costs'].append(dict(index=2,value=packets[0]['event']['costs'][2]['value']))
        elif fault=='row_rewrite':packets[1]['event']['row_start']=0
        elif fault=='scope':packets[1]['event']['scope']=packets[0]['event']['scope']
        elif fault=='duplicate_cost':packets[1]['event']['costs'].append(deepcopy(packets[1]['event']['costs'][0]))
        # Rechain to distinguish structural integrity from merely a checksum.
        previous='0'*64;lines=[]
        for p in packets:
            if fault!='chain':p['previous_sha256']=previous
            line=pub.encoded(p);lines.append(line);previous=pub.sha(line)
        bad=b''.join(lines)
    with pytest.raises(ValueError):pub.replay_journal(bad,'journal')


@pytest.mark.parametrize('fault',('failed_oracle','failed_publication','nested','duplicate','bool_clock','overlap','missing','root_outside'))
def test_caller_costs_reject_hidden_failed_or_unowned_work(published: Any,fault: str) -> None:
    _,receipt,_=published;rows=deepcopy(receipt['caller_intervals'])
    if fault=='failed_oracle':next(r for r in rows if r['kind']=='final-analysis')['outcome']='failed'
    elif fault=='failed_publication':next(r for r in rows if r['kind']=='final-publication')['outcome']='failed'
    elif fault=='nested':
        parent=rows[2];rows.append(dict(sample_id='journal/caller/'+str(len(rows)),parent_id=parent['sample_id'],kind='final-analysis',
                                       start_ns=parent['start_ns']+1,end_ns=parent['end_ns']-1,outcome='complete'))
    elif fault=='duplicate':rows[-1]['kind']='final-publication'
    elif fault=='bool_clock':rows[1]['start_ns']=True
    elif fault=='overlap':rows[-1]['start_ns']=rows[-2]['start_ns']
    elif fault=='missing':rows.pop()
    elif fault=='root_outside':rows[0]['end_ns']=rows[-1]['start_ns']
    assert pub.caller_cost_problems(rows,'journal',complete=True)


@pytest.mark.parametrize('which',('record','journal','closure'))
def test_existing_paths_are_never_resumed_or_overwritten(tmp_path: Path,which: str) -> None:
    path=tmp_path/'r.json';other=path if which=='record' else path.with_name(path.name+('.journal' if which=='journal' else '.closure.json'))
    other.write_bytes(b'keep')
    with pytest.raises(ValueError):pub.OwnedAnswerOutput(path,'x')
    assert other.read_bytes()==b'keep'


@pytest.mark.parametrize('which',('record','journal','closure'))
def test_external_output_changes_close_admission(tmp_path: Path,which: str) -> None:
    out=pub.OwnedAnswerOutput(tmp_path/'r.json','x')
    try:
        if which=='record':out.path.write_bytes(b'changed')
        elif which=='journal':
            with out.journal_path.open('ab') as stream:stream.write(b'changed')
        else:out.closure_path.write_bytes(b'foreign')
        with pytest.raises(OSError):out.verify()
    finally:out.close()


def test_bounded_native_reader_accepts_realistic_size_beyond_old_fit_limit(tmp_path: Path) -> None:
    path=tmp_path/'65MiB.bin'
    with path.open('wb') as stream:stream.truncate(65*1024*1024)
    assert len(pub.bounded_study_bytes(path))==65*1024*1024
    with pytest.raises(ValueError):pub.bounded_study_bytes(path,maximum_bytes=64*1024*1024)
    with pytest.raises(ValueError):pub.bounded_study_bytes(path,maximum_bytes=True)


def test_quota_and_disk_refusals_preserve_initial_owned_output(tmp_path: Path) -> None:
    with patch.object(pub.shutil,'disk_usage',return_value=type('Disk',(),{'free':0})()):
        with pytest.raises(ValueError):pub.OwnedAnswerOutput(tmp_path/'low.json','x')
    assert not (tmp_path/'low.json').exists()
    out=pub.OwnedAnswerOutput(tmp_path/'quota.json','quota')
    try:
        with patch.object(pub,'MAX_JOURNAL_BYTES',4096):
            record=small_record(out,FakeClock())
            assert record['status']=='stopped' and len(record['rows'])==1
            assert record['publication_problem']
        assert out.path.read_bytes()==out.expected
    finally:out.close()


@pytest.mark.parametrize('replay',(False,True),ids=('writer','rehashed-replay'))
@pytest.mark.parametrize('fault',('settings','event_removed','event_added','error_removed','cleanup','verdict','complete',
                                  'cycle','field_removed','unknown_added','numeric_type','signed_zero'))
def test_completed_answer_observations_are_immutable(tmp_path: Path,replay: bool,fault: str) -> None:
    row=row_fixture(complete=True);row['errors']=[dict(type='Earlier',message='keep')]
    if fault=='signed_zero':row['snapshots'][0]['identity']=0.0
    before=checkpoint_fixture([row]);after=deepcopy(before);new=after['rows'][0]
    if fault=='settings':new['events'][0]['data']['generation_settings']['seed']=999
    elif fault=='event_removed':new['events']=[]
    elif fault=='event_added':new['events'].append(dict(received_ns=20,data={}))
    elif fault=='error_removed':new['errors']=[]
    elif fault=='cleanup':new['cleanup']=dict(clean=True)
    elif fault=='verdict':new['verdict']='right'
    elif fault=='complete':new['complete']=False
    elif fault=='cycle':new['cycle_ns']=100
    elif fault=='field_removed':new.pop('request')
    elif fault=='unknown_added':new['suppressed']=True
    elif fault=='numeric_type':new['snapshots'][0]['identity']=True
    elif fault=='signed_zero':new['snapshots'][0]['identity']=-0.0
    with pytest.raises((OSError,ValueError),match='completed checkpoint row changed'):
        transition(tmp_path,before,after,replay=replay)


@pytest.mark.parametrize('fault',('first','middle'))
def test_writer_checks_all_earlier_completed_rows_even_when_tail_matches(tmp_path: Path,fault: str) -> None:
    record=checkpoint_fixture([row_fixture(complete=True,slot=i) for i in range(3)])
    owned=pub.OwnedAnswerOutput(tmp_path/'changes.json','changes')
    try:
        owned(record);size=owned.journal_path.stat().st_size
        # Mutate the original dict, not a separate replacement object.
        record['rows'][0 if fault=='first' else 1]['events'][0]['data']['generation_settings']['seed']=999
        with pytest.raises(OSError,match='completed checkpoint row changed'):owned(record)
        assert owned.journal_path.stat().st_size==size and owned.sequence==1
        assert owned.memory_diagnostic()==record
    finally:owned.close()


@pytest.mark.parametrize('replay',(False,True),ids=('writer','rehashed-replay'))
@pytest.mark.parametrize('fault',('event_change','event_truncate','nested_null_fill','snapshot_change','power_change','error_erase',
                                  'request_change','prompt_change','prompt_append','field_drop','unknown_add','verdict_rewrite','cleanup_rewrite'))
def test_incomplete_row_progress_never_erases_previous_observations(tmp_path: Path,replay: bool,fault: str) -> None:
    row=row_fixture();row['errors']=[dict(type='Earlier',message='keep')]
    if fault=='cleanup_rewrite':row['cleanup']=dict(clean=False)
    if fault=='verdict_rewrite':row['verdict']='wrong'
    before=checkpoint_fixture([row]);after=deepcopy(before);new=after['rows'][0]
    if fault=='event_change':new['events'][0]['data']['content']='clean'
    elif fault=='event_truncate':new['events']=[]
    elif fault=='nested_null_fill':new['events'][0]['data']['generation_settings']['seed']=42
    elif fault=='snapshot_change':new['snapshots'][0]['identity']=2
    elif fault=='power_change':new['power'][0]='battery'
    elif fault=='error_erase':new['errors']=[]
    elif fault=='request_change':new['request']['seed']=999
    elif fault=='prompt_change':new['prompt_ids'][0]=99
    elif fault=='prompt_append':new['prompt_ids'].append(13)
    elif fault=='field_drop':new.pop('events')
    elif fault=='unknown_add':new['replacement_allowed']=True
    elif fault=='verdict_rewrite':new['verdict']='right'
    elif fault=='cleanup_rewrite':new['cleanup']['clean']=True
    with pytest.raises((OSError,ValueError),match='checkpoint row'):
        transition(tmp_path,before,after,replay=replay)


def test_incomplete_progress_retains_observations_and_errors_and_can_finish(tmp_path: Path) -> None:
    row=row_fixture();row['prompt_ids']=[];row['errors']=[dict(type='Earlier',message='still recorded')]
    record=checkpoint_fixture([row]);owned=pub.OwnedAnswerOutput(tmp_path/'changes.json','changes')
    try:
        owned(record)
        row['power'].append('ac');row['snapshots'].append(dict(phase='before_stop',identity=1))
        row['events'].append(dict(received_ns=20,data=dict(stop=True)))
        row['errors'].append(dict(type='Later',message='also recorded'));row['prompt_ids']=[11,12]
        row.update(cleanup=dict(clean=True),cycle_ns=100,answer=dict(valid=False),judge=dict(passed=False),verdict='wrong',complete=True)
        owned(record);owned(record)
        assert pub.replay_journal(owned.journal_path.read_bytes(),'changes')==record
        assert len(pub.replay_journal(owned.journal_path.read_bytes(),'changes')['rows'][0]['errors'])==2
    finally:owned.close()


def test_incomplete_row_must_finish_before_a_later_load_is_appended(tmp_path: Path) -> None:
    before=checkpoint_fixture([row_fixture()]);after=deepcopy(before);after['rows'].append(row_fixture(slot=1))
    with pytest.raises(OSError,match='unfinished row'):transition(tmp_path,before,after,replay=False)


@pytest.mark.parametrize('replay',(False,True),ids=('writer','rehashed-replay'))
@pytest.mark.parametrize('fault',('stop_erased','analysis_problem','publication_problem','rest_duration','rest_healed',
                                  'admission','futile','not_run','status','late_unknown'))
def test_checkpoint_metadata_cannot_erase_operational_history(tmp_path: Path,replay: bool,fault: str) -> None:
    before=checkpoint_fixture([row_fixture(complete=True)])
    if fault=='stop_erased':before.update(status='stopped',stop=dict(type='Earlier',message='failure'))
    elif fault in ('analysis_problem','publication_problem'):before[fault]=dict(type='Earlier',message='failure')
    elif fault in ('rest_duration','rest_healed'):before['rests']=[dict(sample_id='rest',seconds=999,complete=fault!='rest_healed')]
    elif fault=='admission':before['pair_admissions']=[dict(permitted=False,at_ns=10)]
    elif fault=='futile':before['futile']={'cell':dict(cause='operational failure')}
    elif fault=='not_run':before['not_run']=[dict(slot=7,cause='operational stop')]
    elif fault=='status':before['status']='done'
    after=deepcopy(before)
    if fault=='stop_erased':after.update(status='done');after.pop('stop')
    elif fault in ('analysis_problem','publication_problem'):after.pop(fault)
    elif fault=='rest_duration':after['rests'][0]['seconds']=30
    elif fault=='rest_healed':after['rests'][0]['complete']=True
    elif fault=='admission':after['pair_admissions'][0]['permitted']=True
    elif fault=='futile':after['futile']['cell']['cause']='task futility'
    elif fault=='not_run':after['not_run'][0]['cause']='task futility'
    elif fault=='status':after['status']='running'
    elif fault=='late_unknown':after['quietly_ignored_error']=False
    with pytest.raises((OSError,ValueError),match='checkpoint'):
        transition(tmp_path,before,after,replay=replay)


def test_open_rest_progress_requires_running_metadata_and_keeps_its_identity(tmp_path: Path) -> None:
    record=checkpoint_fixture([])
    record['rests']=[dict(sample_id='rest',after_slot=27,seconds=30,complete=False)]
    record['cost_intervals']=[dict(sample_id='rest',kind='rest',start_ns=1,end_ns=None,outcome='running')]
    owned=pub.OwnedAnswerOutput(tmp_path/'changes.json','changes')
    try:
        owned(record);record['rests'][0]['complete']=True
        record['cost_intervals'][0].update(end_ns=30_000_000_001,outcome='complete')
        record['pair_admissions'].append(dict(first_slot=28,at_ns=30_000_000_002,permitted=True))
        record['futile']['cell']=dict(cause='task futility');record['status']='done';record['not_run']=[dict(slot=28)]
        record['analysis']=dict(analysis_looks=0)
        owned(record)
        assert pub.replay_journal(owned.journal_path.read_bytes(),'changes')==record
    finally:owned.close()


@pytest.mark.parametrize('earlier_complete',(False,True),ids=('incomplete-observation','completed-observation'))
def test_public_native_auditor_rejects_rehashed_erasure_of_earlier_answer_evidence(bundle: Any,monkeypatch: pytest.MonkeyPatch,earlier_complete: bool) -> None:
    from tests.test_lookup_answers_runner import invoke
    code,out,_=invoke(bundle,monkeypatch);assert code==0
    raw=out.read_bytes();journal=out.with_name(out.name+'.journal')
    packets=[json.loads(line) for line in journal.read_bytes().splitlines()]
    clean=deepcopy(packets[0]['event']['rows'][0]);assert clean['complete'] is True
    packets[0]['event']['rows'][0]['complete']=earlier_complete
    packets[0]['event']['rows'][0]['events'][0]['data']['generation_settings']={'seed':999}
    packets[1]['event']['row_start']=0;packets[1]['event']['rows'].insert(0,clean)
    forged=rechain(packets);journal.write_bytes(forged)
    receipt_path=out.with_name(out.name+'.closure.json');receipt=json.loads(receipt_path.read_bytes())
    receipt.update(journal_sha256=pub.sha(forged),journal_bytes=len(forged));receipt_path.write_bytes(pub.encoded(receipt))
    cli=importlib.import_module('audit_lookup_answers');monkeypatch.setattr(cli,'ROOT',bundle['root'])
    report=bundle['root']/'history-audit.json'
    with patch('subprocess.Popen',side_effect=AssertionError('no native process')):
        assert cli.main([str(out),'--rules',str(bundle['rules']),'--rules-sha256',pub.sha(bundle['rules'].read_bytes()),'--out',str(report)])==2
    result=json.loads(report.read_bytes())
    assert not result['evidence_valid'] and not result['study_complete'] and result['analysis']['analysis_looks']==0
    assert out.read_bytes()==raw


def test_production_writer_refuses_clean_replacement_of_contradictory_completed_answer(bundle: Any,monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_lookup_answers_runner import invoke
    code,out,_=invoke(bundle,monkeypatch);assert code==0
    core=json.loads(out.read_bytes())['core'];first=pub.replay_journal(out.with_name(out.name+'.journal').read_bytes().splitlines(keepends=True)[0],core['run_id'])
    first['rows'][0]['events'][0]['data']['generation_settings']={'seed':999}
    from inferenceos.tune.lookup_answers import native_answer
    from inferenceos.tune.lookup_corpus import build_corpus
    row=first['rows'][0];case=next(c for c in build_corpus().cases if c.case_id==row['case_id'])
    span=next(c for c in first['cost_intervals'] if c['sample_id']==row['request_sample_id'])
    earlier=native_answer(row['events'],row['request'],case,row['arm'],row['prompt_ids'],span['start_ns'],span['end_ns'])
    assert not earlier['valid'] and earlier['problems']
    owned=pub.OwnedAnswerOutput(bundle['root']/'replacement.json',core['run_id'])
    try:
        owned(first)
        with pytest.raises(OSError,match='completed checkpoint row changed'):owned(core)
        assert owned.sequence==1
        assert pub.replay_journal(owned.journal_path.read_bytes(),core['run_id'])['rows'][0]['events'][0]['data']['generation_settings']=={'seed':999}
    finally:owned.close()


@pytest.mark.parametrize('replay',(False,True),ids=('writer','rehashed-replay'))
@pytest.mark.parametrize('fault',('failed_healed','identity_type','closed_changed','closed_failed_healed'))
def test_checkpoint_costs_cannot_discard_failures_or_rewrite_closed_work(tmp_path: Path,replay: bool,fault: str) -> None:
    before=checkpoint_fixture([])
    interval=dict(sample_id='work',parent_id=None,kind='oracle',start_ns=1,end_ns=None,outcome='failed')
    if fault in ('closed_changed','closed_failed_healed'):interval['end_ns']=2
    before['cost_intervals']=[interval];after=deepcopy(before);new=after['cost_intervals'][0]
    if fault=='failed_healed':new.update(end_ns=3,outcome='complete')
    elif fault=='identity_type':new['start_ns']=True
    elif fault=='closed_changed':new['end_ns']=3
    elif fault=='closed_failed_healed':new['outcome']='complete'
    owned=pub.OwnedAnswerOutput(tmp_path/'changes.json','changes')
    try:
        owned(before)
        if not replay:
            with pytest.raises(OSError,match='cost'):owned(after)
        else:
            first=json.loads(owned.journal_path.read_bytes())
            second=dict(event=dict(version=pub.JOURNAL_VERSION,run_id='changes',row_start=0,rows=[],
                                   costs=[dict(index=0,value=new)],meta=first['event']['meta']))
            with pytest.raises(ValueError,match='cost'):pub.replay_journal(rechain([first,second]),'changes')
    finally:owned.close()


@pytest.mark.parametrize('outcome',('running','failed'))
def test_open_cost_can_close_without_replacing_an_observed_failure(tmp_path: Path,outcome: str) -> None:
    record=checkpoint_fixture([])
    record['cost_intervals']=[dict(sample_id='work',kind='oracle',start_ns=1,end_ns=None,outcome=outcome)]
    owned=pub.OwnedAnswerOutput(tmp_path/'changes.json','changes')
    try:
        owned(record);record['cost_intervals'][0].update(end_ns=2,outcome='complete' if outcome=='running' else 'failed')
        owned(record)
        assert pub.replay_journal(owned.journal_path.read_bytes(),'changes')==record
    finally:owned.close()


def test_writer_compares_against_written_value_when_producer_mutates_during_fsync(tmp_path: Path,monkeypatch: pytest.MonkeyPatch) -> None:
    record=checkpoint_fixture([row_fixture(complete=True)])
    owned=pub.OwnedAnswerOutput(tmp_path/'changes.json','changes');fsync=pub.os.fsync
    def mutate_after_write(fd: int) -> None:
        fsync(fd);record['rows'][0]['events'][0]['data']['generation_settings']['seed']=999
    try:
        with monkeypatch.context() as mp:
            mp.setattr(pub.os,'fsync',mutate_after_write);owned(record)
        with pytest.raises(OSError,match='completed checkpoint row changed'):owned(record)
        assert pub.replay_journal(owned.journal_path.read_bytes(),'changes')['rows'][0]['events'][0]['data']['generation_settings']['seed'] is None
    finally:owned.close()
