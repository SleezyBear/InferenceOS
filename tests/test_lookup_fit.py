"""Fit-only collector adversarial checks. All models/observations are fabricated."""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from contextlib import ExitStack

import pytest

from inferenceos.contracts.enums import PowerState
from inferenceos.contracts.tune import SavedSetup, MachineCheck, ModelFileIdentity
from inferenceos.tune.lookup_corpus import digest
from inferenceos.tune.lookup_fit import (FitManifest, FitRefused, STAGE_A, REQUIRED_EFFECTS, INITIALIZATION_POLICY, audit_fit,
                                       collect_fit, observation_problems, parse_manifest, strict_json,
                                       token_inputs, token_problems)
from inferenceos.tune.lookup_fit_io import bounded_read, bounded_file_bytes, InspectionTimeout, inspect_json
from inferenceos.tune.lookup_fit_observers import observe_effects, machine_from_raw
from inferenceos.tune.lookup_fit_sim import FitSimulator, fixture_log, fixture_machine, fixture_manifest, simulate_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'scripts'))


def runner() -> Any:
    return importlib.import_module('collect_lookup_fit')


def collect(engine: Any = None, *, power: Any = lambda:'ac', guard: Any = lambda:None,
            checkpoint: Any = lambda _:None) -> tuple[dict[str, Any], Any]:
    manifest = fixture_manifest()
    engine = engine or FitSimulator(manifest)
    result = collect_fit(engine,manifest,run_id='test_only',power=power,guard=guard,checkpoint=checkpoint)
    return result,engine


def proposal(manifest: FitManifest) -> dict[str, Any]:
    return dict(protocol=manifest.protocol,executable=False,hardware_authorized=False,
                protocol_document_sha256=manifest.protocol_document_sha256,corpus_sha256=manifest.corpus_sha256,
                proposed_engine_files=manifest.engine_files,initialization=deepcopy(INITIALIZATION_POLICY),
                models=[dict(label=m.label,path=m.path,sha256=m.sha256,configuration=m.configuration.model_dump(mode='json'),
                             configuration_sha256=m.configuration_sha256,source=m.saved_setup_source,current_bytes_verified=False)
                        for m in manifest.models])


def complete_record(tmp_path: Path) -> tuple[dict[str, Any], FitManifest]:
    record,_ = collect()
    manifest = fixture_manifest()
    selected = str(tmp_path / next(n for n in manifest.engine_files if n.endswith('/llama-server.exe')))
    for row in record['loads']:
        for snap in row['snapshots']:
            for f in snap['image']['files']:
                f['path'] = str(tmp_path / f['path'].removeprefix('C:/D2-FIT-FIXTURE/'))
            snap['image']['executable'] = selected
            snap['selected_executable'] = selected
            snap['command'][0] = selected
    stored, receipts = [],[]
    for pin in manifest.models:
        identity = ModelFileIdentity(path=str(tmp_path / pin.path),size_bytes=17,modified_ns=1,sha256=pin.sha256)
        stored.append(SavedSetup(setup_id='fixture-'+pin.label,model=identity,model_name=pin.label,context_length=4096,
                                 power_source=PowerState.AC,machine=MachineCheck(),configuration=pin.configuration,
                                 starting_configuration=pin.configuration).model_dump(mode='json'))
        receipts.append(dict(label=pin.label,model_file=dict(path=identity.path,size_bytes=17,modified_ns=1,sha256=pin.sha256),
                             source=pin.saved_setup_source,configuration=pin.configuration.model_dump(mode='json')))
    raw = json.dumps(dict(setups=stored)).encode()
    data_map = {'saved_setups.json':dict(size=len(raw),mtime_ns=1,sha256=hashlib.sha256(raw).hexdigest())}
    record['preservation'] = dict(root=str(tmp_path),data_before=data_map,data_after=deepcopy(data_map),
        saved_file=dict(bytes_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),path=str(tmp_path/'data/saved_setups.json')),
        selected_executable=selected,models=receipts,frozen_hashes={**manifest.python_files,**manifest.engine_files},
        environment_sha256=digest('fixture environment, not host data'),
        ambient=dict(platform='win32',checked=True,environment=dict(APPDATA='',PROGRAMDATA=''),absent_paths=[]))
    return record,manifest


def test_full_fake_caller_population_has_no_generation_operation() -> None:
    with patch('subprocess.Popen',side_effect=AssertionError('no process')):
        result,engine = collect()
    assert result['status'] == 'done' and engine.starts == engine.releases == 4
    assert [(r['model'],r['arm']) for r in result['loads']] == list(STAGE_A)
    assert all(len(r['tokens']) == 200 for r in result['loads'])
    assert not any(name.startswith(('generate','request','chat')) for name in dir(engine))


def test_complete_raw_fixture_audits_without_granting_any_permission(tmp_path: Path) -> None:
    raw,manifest = complete_record(tmp_path)
    before = deepcopy(raw)
    audit = audit_fit(raw,manifest)
    assert audit['observations_complete'],audit
    assert audit['load_boundary_power_readings']==8 and audit['inventory_power_readings']==8
    assert audit['admission_power_readings']==0  # direct fixture collector, no CLI preflight
    with_admission=deepcopy(raw);with_admission['initial_power']='ac'
    assert audit_fit(with_admission,manifest)['admission_power_readings']==1
    assert raw == before
    assert all(audit[k] is False for k in ('generation_allowed','stage_B_authorized','automatic_admission','deployment_allowed','performance_measured'))


@pytest.mark.parametrize('field',REQUIRED_EFFECTS)
def test_every_required_effect_is_independently_required(field: str) -> None:
    engine = FitSimulator(fixture_manifest())
    engine.start(fixture_manifest().models[0],'control')
    raw = engine.observe()
    source = raw['effects']['sources'][field][0]
    raw['startup']['text'] = raw['startup']['text'].replace(source,'removed source')
    raw['startup']['text_sha256'] = hashlib.sha256(raw['startup']['text'].encode()).hexdigest()
    raw['effects'] = observe_effects(raw['startup']['text'],raw['slots'])
    # threads has two independent sources; remove both.
    if field == 'threads':
        raw['startup']['text'] = '\n'.join(line for line in raw['startup']['text'].splitlines() if 'threadpool init' not in line)
        raw['startup']['text_sha256'] = hashlib.sha256(raw['startup']['text'].encode()).hexdigest()
        raw['effects'] = observe_effects(raw['startup']['text'],raw['slots'])
    assert observation_problems(raw,fixture_manifest().models[0].configuration,'control',fixture_manifest().engine_files)


@pytest.mark.parametrize('fault',('missing_slot','duplicate_slot','unknown_slot','busy','missing_busy','numeric_busy','ctx','auto_flash',
                                  'contradictory_threads','contradictory_batch','mixed_k','truncated_log','reader_error','stale_log',
                                  'wrong_library','missing_library','wrong_executable','wrong_config','forged_effect','missing_machine',
                                  'unknown_driver','ambiguous_gpu','wrong_gpu','unknown_cpu','invented_isa','invalid_pid','missing_birth','nonfinite_birth'))
def test_live_observation_faults_stop_before_any_tokenizer_request(fault: str) -> None:
    engine = FitSimulator(fixture_manifest())
    original = engine.observe
    def observe() -> Any:
        raw = original()
        if fault == 'missing_slot':raw['slots'].pop()
        elif fault == 'duplicate_slot':raw['slots'][1]['id']=0
        elif fault == 'unknown_slot':raw['slots'][0]['id']=7
        elif fault == 'busy':raw['slots'][0]['is_processing']=True
        elif fault == 'missing_busy':raw['slots'][0].pop('is_processing')
        elif fault == 'numeric_busy':raw['slots'][0]['is_processing']=0
        elif fault == 'ctx':raw['slots'][0]['n_ctx']=8192
        elif fault == 'auto_flash':raw['startup']['text']=raw['startup']['text'].replace('resolve_fused_ops: Flash Attention enabled','llama_context: flash_attn = auto')
        elif fault == 'contradictory_threads':raw['startup']['text']+='\nllama threadpool init, n_threads = 7'
        elif fault == 'contradictory_batch':raw['startup']['text']+='\nllama_context: n_batch = 7'
        elif fault == 'mixed_k':raw['startup']['text']+='\nllama_kv_cache: K (q8_0): 1 MiB'
        elif fault == 'truncated_log':raw['startup']['truncated']=True
        elif fault == 'reader_error':raw['startup']['reader_errors']=['bad read']
        elif fault == 'stale_log':raw['startup']['generation_current']=False
        elif fault == 'wrong_library':raw['image']['files'][0]['sha256']='0'*64
        elif fault == 'missing_library':raw['image']['files'].pop()
        elif fault == 'wrong_executable':raw['image']['executable']='C:/different/llama-server.exe'
        elif fault == 'wrong_config':raw['startup']['text']=raw['startup']['text'].replace('n_ubatch = 256','n_ubatch = 512')
        elif fault == 'forged_effect':raw['effects']['configuration']['threads']=99
        elif fault == 'missing_machine':raw.pop('machine')
        elif fault == 'unknown_driver':raw['machine']['cim']['video_controllers'][0].pop('DriverVersion')
        elif fault == 'ambiguous_gpu':raw['machine']['vulkan']*=2
        elif fault == 'wrong_gpu':raw['machine']['vulkan'][0]['index']=1
        elif fault == 'unknown_cpu':raw['machine']['cim']['processors'][0].pop('NumberOfCores')
        elif fault == 'invented_isa':raw['machine']['cpuid']['avx']='false'
        elif fault == 'invalid_pid':raw['image']['pid']=True
        elif fault == 'missing_birth':raw['image'].pop('create_time_s')
        elif fault == 'nonfinite_birth':raw['image']['create_time_s']=float('inf')
        raw['startup']['text_sha256']=hashlib.sha256(raw['startup']['text'].encode()).hexdigest()
        if fault != 'forged_effect':raw['effects']=observe_effects(raw['startup']['text'],raw['slots'])
        return raw
    engine.observe=observe  # type: ignore[method-assign]
    result,_=collect(engine)
    assert result['status']=='stopped' and engine.starts==engine.releases==1
    assert not any(c[0]=='tokenize' for c in engine.calls)


@pytest.mark.parametrize('fault',('empty','bool','negative','float','missing','long_prompt','long_output','changed_second_arm'))
def test_bad_token_evidence_stops_at_its_first_boundary(fault: str) -> None:
    engine=FitSimulator(fixture_manifest())
    original=engine.tokenize
    def tokenize(text: str, *, add_special: bool) -> Any:
        ids=original(text,add_special=add_special)
        if fault=='empty':return []
        if fault=='bool':return [True]
        if fault=='negative':return [-1]
        if fault=='float':return [1.0]
        if fault=='missing':return None
        if fault=='long_prompt' and add_special:return [1]*3505
        if fault=='long_output' and not add_special:return [1]*481
        if fault=='changed_second_arm' and engine.starts==2:return ids+[9]
        return ids
    engine.tokenize=tokenize  # type: ignore[method-assign]
    result,_=collect(engine)
    assert result['status']=='stopped' and engine.starts==engine.releases
    assert engine.starts==(2 if fault=='changed_second_arm' else 1)


@pytest.mark.parametrize('stage',('start','observe','tokenize','release','post_power','guard'))
@pytest.mark.parametrize('interrupt',(False,True))
def test_same_interrupt_is_checkpointed_after_release_and_never_dispatches_later(stage: str,interrupt: bool) -> None:
    engine=FitSimulator(fixture_manifest())
    boom=KeyboardInterrupt('original') if interrupt else RuntimeError('original')
    checkpoints=[]
    def fail(*args: Any,**kwargs: Any) -> Any:raise boom
    if stage in ('start','observe','tokenize','release'):
        setattr(engine,stage,fail)
    powers=iter(('ac',boom))
    def power() -> str:
        value=next(powers)
        if isinstance(value,BaseException):raise value
        return value
    guards=0
    def guard() -> None:
        nonlocal guards
        guards+=1
        if guards==2:raise boom
    if interrupt:
        with pytest.raises(KeyboardInterrupt) as caught:
            collect(engine,power=power if stage=='post_power' else lambda:'ac',guard=guard if stage=='guard' else lambda:None,
                    checkpoint=lambda r:checkpoints.append(deepcopy(r)))
        assert caught.value is boom
    else:
        result,_=collect(engine,power=power if stage=='post_power' else lambda:'ac',guard=guard if stage=='guard' else lambda:None,
                         checkpoint=lambda r:checkpoints.append(deepcopy(r)))
        assert result['status']=='stopped'
    assert len(checkpoints[-1]['loads'])==1
    if stage not in ('release',):assert engine.releases==1
    assert any(e['type']==type(boom).__name__ for e in checkpoints[-1]['loads'][0]['errors'])


@pytest.mark.parametrize('power',('battery','unknown','no battery','',None))
def test_no_initial_ac_means_zero_starts(power: Any) -> None:
    result,engine=collect(power=lambda:power)
    assert result['status']=='stopped' and engine.starts==engine.releases==0


@pytest.mark.parametrize('stage',('release','post_power'))
@pytest.mark.parametrize('exception',(KeyboardInterrupt,SystemExit))
def test_secondary_unwind_errors_do_not_swallow_a_pending_exit(stage: str,exception: Any) -> None:
    engine=FitSimulator(fixture_manifest())
    boom=exception('original')
    checkpoints=[]
    def exit_now(*args: Any,**kwargs: Any) -> Any:raise boom
    def secondary(*args: Any,**kwargs: Any) -> Any:raise RuntimeError('secondary unwind error')
    engine.tokenize=exit_now  # type: ignore[method-assign]
    if stage=='release':engine.release=secondary  # type: ignore[method-assign]
    powers=0
    def power() -> str:
        nonlocal powers
        powers+=1
        if stage=='post_power' and powers==2:secondary()
        return 'ac'
    with pytest.raises(exception) as caught:
        collect(engine,power=power,checkpoint=lambda r:checkpoints.append(deepcopy(r)))
    assert caught.value is boom
    assert engine.starts==1 and len(checkpoints[-1]['not_run'])==3
    assert checkpoints[-1]['status']=='interrupted'
    assert [e['type'] for e in checkpoints[-1]['loads'][0]['errors']]==[exception.__name__,'RuntimeError']


@pytest.mark.parametrize('field',('process_gone','credential_removed'))
@pytest.mark.parametrize('value',(1,1.0))
@pytest.mark.parametrize('route',('collector','audit'))
def test_release_evidence_requires_literal_booleans(tmp_path: Path,field: str,value: Any,route: str) -> None:
    if route=='audit':
        raw,m=complete_record(tmp_path)
        raw['loads'][0]['cleanup'][field]=value
        checked=audit_fit(raw,m)
        assert not checked['observations_complete'] and any('release unconfirmed' in p for p in checked['problems'])
    else:
        engine=FitSimulator(fixture_manifest())
        original=engine.release
        def release(*,abort: bool) -> Any:
            result=original(abort=abort)
            result[field]=value
            return result
        engine.release=release  # type: ignore[method-assign]
        result,_=collect(engine)
        assert result['status']=='stopped' and engine.starts==engine.releases==1
        assert len(result['not_run'])==3


@pytest.mark.parametrize('field,value',(('size',False),('size',0.0),('mtime_ns',True),('mtime_ns',1.0)))
def test_preserved_data_maps_do_not_accept_numeric_type_aliases(tmp_path: Path,field: str,value: Any) -> None:
    raw,m=complete_record(tmp_path)
    item=dict(size=0,mtime_ns=1,sha256=hashlib.sha256(b'').hexdigest())
    raw['preservation']['data_before']['empty.txt']=deepcopy(item)
    raw['preservation']['data_after']['empty.txt']=deepcopy(item)
    assert audit_fit(raw,m)['observations_complete']
    raw['preservation']['data_after']['empty.txt'][field]=value
    checked=audit_fit(raw,m)
    assert not checked['observations_complete'] and any('equal data maps' in p for p in checked['problems'])


@pytest.mark.parametrize('part',('process','password','problem'))
def test_unclean_release_prohibits_next_load(part: str) -> None:
    engine=FitSimulator(fixture_manifest())
    def release(*,abort: bool) -> Any:
        engine.releases+=1
        return dict(process_gone=part!='process',credential_removed=part!='password',problem='held' if part=='problem' else None)
    engine.release=release  # type: ignore[method-assign]
    result,_=collect(engine)
    assert result['status']=='stopped' and engine.starts==engine.releases==1


@pytest.mark.parametrize('fault',('missing','duplicate','reordered','model','run','tokens','token_hash','saved_bytes','saved_hash','map',
                                  'source','executable','model_sha','status','simulated','stored_pass','same_process','missing_preservation'))
def test_auditor_recomputes_population_binding_and_preservation(tmp_path: Path,fault: str) -> None:
    raw,manifest=complete_record(tmp_path)
    if fault=='missing':raw['loads'].pop()
    elif fault=='duplicate':raw['loads'].append(deepcopy(raw['loads'][-1]))
    elif fault=='reordered':raw['loads'].reverse()
    elif fault=='model':raw['loads'][0]['model']='other'
    elif fault=='run':raw['run_id']='different'
    elif fault=='tokens':raw['loads'][0]['tokens'][0]['ids'][0]+=1
    elif fault=='token_hash':raw['loads'][0]['tokens'][0]['ids_sha256']='f'*64
    elif fault=='saved_bytes':raw['preservation']['saved_file']['bytes_base64']='bWFsZm9ybWVk'
    elif fault=='saved_hash':raw['preservation']['saved_file']['sha256']='f'*64
    elif fault=='map':raw['preservation']['data_after']={}
    elif fault=='source':raw['preservation']['frozen_hashes']={}
    elif fault=='executable':raw['preservation']['selected_executable']=str(tmp_path/'copy/llama-server.exe')
    elif fault=='model_sha':raw['preservation']['models'][0]['model_file']['sha256']='f'*64
    elif fault=='status':raw['status']='running'
    elif fault=='simulated':raw['simulated']=True
    elif fault=='stored_pass':raw['loads'][0]['tokens']=[];raw['fit_passed']=True;raw['observations_complete']=True
    elif fault=='same_process':
        for snap in raw['loads'][1]['snapshots']:snap['image']['pid']=10001;snap['image']['create_time_s']=1001
    elif fault=='missing_preservation':raw.pop('preservation')
    checked=audit_fit(raw,manifest)
    assert not checked['observations_complete'],fault


def test_manifest_binds_the_entire_proposal_not_a_model_name() -> None:
    m=fixture_manifest()
    p=proposal(m)
    assert parse_manifest(m.model_dump_json(),p,m.proposal_sha256)==m
    for field in ('path','sha256','configuration_sha256','source'):
        changed=deepcopy(p);changed['models'][0][field]='different'
        with pytest.raises(ValueError):parse_manifest(m.model_dump_json(),changed,m.proposal_sha256)


@pytest.mark.parametrize('field',tuple(type(fixture_manifest().models[0].configuration).model_fields))
def test_manifest_omitted_configuration_fields_never_get_filled(field: str) -> None:
    m=fixture_manifest();raw=m.model_dump(mode='json');raw['models'][0]['configuration'].pop(field)
    with pytest.raises(ValueError):parse_manifest(json.dumps(raw),proposal(m),m.proposal_sha256)


@pytest.mark.parametrize('text',('{"a":1,"a":2}','{"a":1,"A":2}','{"a":NaN}','{"a":Infinity}',
                                  '{"a":1e999}','{"a":0.123456789012345678901}','{"a":9007199254740993.0}'))
def test_ambiguous_json_is_refused(text: str) -> None:
    with pytest.raises(ValueError):strict_json(text)


def test_late_read_never_publishes_or_blocks_caller_forever() -> None:
    release=threading.Event();finished=threading.Event()
    def read() -> str:
        release.wait(2);finished.set();return 'late'
    try:
        with pytest.raises(InspectionTimeout):bounded_read(read,timeout_s=.03)
        assert not finished.is_set()
    finally:release.set()
    assert finished.wait(1)


@pytest.mark.parametrize('path',('/completion','/v1/chat/completions','/props','/slots?x=1','http://other/slots'))
def test_inspection_client_cannot_dispatch_answers_or_unknown_endpoints(path: str) -> None:
    with patch('http.client.HTTPConnection',side_effect=AssertionError('zero connections')):
        with pytest.raises(FitRefused):inspect_json('http://127.0.0.1:9000','fixture',path)


def test_simulation_and_cli_are_explicitly_not_hardware_qualification(tmp_path: Path) -> None:
    with patch('subprocess.Popen',side_effect=AssertionError('zero processes')):
        result=simulate_fit()
        assert result['status']=='done' and not result['audit']['observations_complete']
        output=tmp_path/'sim.json'
        assert runner().main(['--simulate','--out',str(output)])==0
        before=output.read_bytes()
        assert runner().main(['--simulate','--out',str(output)])==4 and output.read_bytes()==before
        assert runner().main(['--out',str(tmp_path/'real.json')])==4


def test_protected_output_never_changes_saved_data() -> None:
    output=ROOT/'data/NO_SUCH_FIT_OUTPUT.json'
    assert not output.exists()
    assert runner().main(['--simulate','--out',str(output)])==4
    assert not output.exists()


def native_fixture(tmp_path: Path) -> tuple[Any, FitManifest, Path, Path, Any]:
    from inferenceos.config import Settings
    from tests.test_gguf import _write_gguf, _kv_string, _kv_u32
    r=runner();data=fixture_manifest().model_dump(mode='json')
    stored=[]
    for model in data['models']:
        path=tmp_path/model['path'];path.parent.mkdir(exist_ok=True)
        _write_gguf(path,3,[_kv_string('general.architecture','qwen2'),_kv_u32('qwen2.block_count',2),
                          _kv_u32('qwen2.context_length',4096),_kv_u32('qwen2.embedding_length',16),
                          _kv_u32('qwen2.attention.head_count',2),_kv_u32('qwen2.attention.head_count_kv',1)],[])
        model['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        st=path.stat()
        from inferenceos.tune.lookup_qualification import ConfigurationSnapshot
        config=ConfigurationSnapshot(**model['configuration'])
        setup=SavedSetup(setup_id='fixture-'+model['label'],model_name='fixture',context_length=4096,
                         power_source=PowerState.AC,machine=MachineCheck(),configuration=config,starting_configuration=config,
                         model=ModelFileIdentity(path=str(path),size_bytes=st.st_size,modified_ns=st.st_mtime_ns,sha256=model['sha256']))
        stored.append(setup.model_dump(mode='json'))
    for relative in (*data['python_files'],*data['engine_files']):
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture, never executed')
        group='python_files' if relative in data['python_files'] else 'engine_files'
        data[group][relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    doc=tmp_path/'docs/lookup-qualification-protocol.md';doc.parent.mkdir();doc.write_bytes(b'fixture protocol')
    data['protocol_document_sha256']=hashlib.sha256(doc.read_bytes()).hexdigest()
    store=tmp_path/'data/saved_setups.json';store.parent.mkdir();store.write_text(json.dumps(dict(setups=stored)))
    m=FitManifest.model_validate_json(json.dumps(data));p=proposal(m)
    proposal_path=tmp_path/'proposal.json';proposal_path.write_text(json.dumps(p))
    data['proposal_sha256']=hashlib.sha256(proposal_path.read_bytes()).hexdigest()
    m=FitManifest.model_validate_json(json.dumps(data));rules=tmp_path/'rules.json';rules.write_text(m.model_dump_json())
    settings=Settings().model_copy(update=dict(project_root=tmp_path,data_dir=tmp_path/'data',llama_cpp_dir=tmp_path/'tools/llama.cpp'))
    return r,m,rules,proposal_path,settings


@pytest.mark.parametrize('fault',('model','saved_model','saved_config','missing_setup','duplicate_setup','source','engine',
                                  'alternate_executable','identical_executable_copy','alternate_data','rules','proposal','doc','extra_source'))
def test_production_preflight_refuses_before_backend_construction(tmp_path: Path,fault: str) -> None:
    r,m,rules,p,settings=native_fixture(tmp_path)
    original_rules_sha=hashlib.sha256(rules.read_bytes()).hexdigest()
    original_proposal_sha=hashlib.sha256(p.read_bytes()).hexdigest()
    store=tmp_path/'data/saved_setups.json'
    if fault=='model':(tmp_path/m.models[0].path).write_bytes(b'replaced model')
    elif fault in ('saved_model','saved_config','missing_setup','duplicate_setup'):
        raw=json.loads(store.read_text())
        if fault=='saved_model':raw['setups'][0]['model']['sha256']='0'*64
        if fault=='saved_config':raw['setups'][0]['configuration']['threads']=99
        if fault=='missing_setup':raw['setups'].pop(0)
        if fault=='duplicate_setup':raw['setups'].append(deepcopy(raw['setups'][0]))
        store.write_text(json.dumps(raw))
    elif fault=='source':(tmp_path/next(iter(m.python_files))).write_bytes(b'changed source')
    elif fault=='engine':(tmp_path/next(iter(m.engine_files))).write_bytes(b'changed library')
    elif fault in ('alternate_executable','identical_executable_copy'):
        directory=tmp_path/'alternate';directory.mkdir();(directory/'llama-server.exe').write_bytes(b'fixture, never executed' if fault=='identical_executable_copy' else b'different')
        settings=settings.model_copy(update=dict(llama_cpp_dir=directory))
    elif fault=='alternate_data':settings=settings.model_copy(update=dict(data_dir=tmp_path/'elsewhere'))
    elif fault=='rules':rules.write_bytes(b'changed rules')
    elif fault=='proposal':p.write_bytes(b'changed proposal')
    elif fault=='doc':(tmp_path/'docs/lookup-qualification-protocol.md').write_bytes(b'changed protocol')
    elif fault=='extra_source':(tmp_path/'src/added.py').write_bytes(b'new module')
    frozen=r.FrozenInputs
    def windows(*args: Any,**kwargs: Any) -> Any:
        return frozen(*args,**kwargs,platform='win32',environment=lambda:dict(APPDATA='',PROGRAMDATA=''))
    with patch.object(r,'FrozenInputs',side_effect=windows),patch('subprocess.Popen',side_effect=AssertionError('zero process launch')):
        with pytest.raises((ValueError,LookupError,OSError)):
            r.Preflight(tmp_path,settings,m,rules,original_rules_sha,p,original_proposal_sha)
        with patch.object(r,'ROOT',tmp_path),patch.object(r,'Settings',return_value=settings),patch.object(r,'power_state',return_value='ac'),\
             patch.object(r,'LiveFitEngine',side_effect=AssertionError('zero backend constructions')) as made:
            code=r.main(['--authorized-stage-a','--rules',str(rules),'--rules-sha256',original_rules_sha,
                         '--proposal',str(p),'--proposal-sha256',original_proposal_sha,'--out',str(tmp_path/'record.json')])
            assert code==4
            made.assert_not_called()


def test_production_preflight_accepts_real_gguf_metadata_and_raw_saved_binding(tmp_path: Path) -> None:
    r,m,rules,p,settings=native_fixture(tmp_path)
    frozen=r.FrozenInputs
    def windows(*args: Any,**kwargs: Any) -> Any:
        return frozen(*args,**kwargs,platform='win32',environment=lambda:dict(APPDATA='',PROGRAMDATA=''))
    with patch.object(r,'FrozenInputs',side_effect=windows),patch('subprocess.Popen',side_effect=AssertionError('zero process launch')):
        gate=r.Preflight(tmp_path,settings,m,rules,hashlib.sha256(rules.read_bytes()).hexdigest(),p,m.proposal_sha256)
        assert len(gate.models)==2
        gate.check(full=True)
        from inferenceos.tune.lookup_fit import preservation_problems
        assert preservation_problems(gate.preservation(),m)==[]
        (tmp_path/m.models[0].path).write_bytes(b'changed between loads')
        with pytest.raises(ValueError):gate.check()


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str]=[]
        self.active=False
        self.unconfirmed_pid: int | None=None
        self.api_key_file: Path | None=None
        self.undeleted_key_files: list[Path]=[]
        self.delay: threading.Event | None=None
        self.entered=threading.Event()
        self.halted=False
        self.measurement=SimpleNamespace(cleanup_complete=True,credential_removed=True)

    def start_if_current(self,model: Any,params: Any,*,current: Any) -> None:
        self.calls.append('attempt');self.entered.set()
        if self.delay:self.delay.wait(2)
        if not current() or self.halted:raise RuntimeError('late admission refused')
        self.calls.append('process-start');self.active=True

    def stop(self) -> Any:
        self.calls.append('stop');self.active=False;return self.measurement

    def halt(self) -> Any:
        self.halted=True;return self.stop()

    def cleanup(self) -> None:self.calls.append('cleanup');self.active=False
    def confirm_previous_gone(self) -> bool:return self.unconfirmed_pid is None
    def is_active(self) -> bool:return self.active


def fake_live(backend: FakeBackend) -> Any:
    preflight=SimpleNamespace(selected=Path('fixture/llama-server.exe'),settings=SimpleNamespace(llama_cpp_dir=Path('fixture')),
                              inputs=SimpleNamespace(executable=lambda _:Path('fixture/llama-server.exe')),
                              check=lambda:None,models=dict(llama3b=SimpleNamespace(),gemma1b=SimpleNamespace()))
    return runner().LiveFitEngine(preflight,backend_factory=lambda:backend)


def test_late_start_is_revoked_before_native_admission_and_no_restart_occurs() -> None:
    r=runner();backend=FakeBackend();backend.delay=threading.Event();engine=fake_live(backend)
    original=r.bounded_read
    def short(call: Any,*,timeout_s: float) -> Any:return original(call,timeout_s=.04)
    try:
        with patch.object(r,'bounded_read',side_effect=short):
            with pytest.raises(InspectionTimeout):engine.start(fixture_manifest().models[0],'control')
        assert backend.entered.is_set()
        result=engine.release(abort=True)
        assert result['process_gone'] and result['credential_removed']
        backend.delay.set()
        with pytest.raises(ValueError):engine.start(fixture_manifest().models[0],'control')
        assert 'process-start' not in backend.calls
    finally:backend.delay.set()


@pytest.mark.parametrize('fault',('false_measurement','false_credential','unconfirmed_pid','held_password','active'))
def test_live_release_uses_all_owner_evidence_not_liveness_alone(fault: str) -> None:
    backend=FakeBackend();engine=fake_live(backend);engine.start(fixture_manifest().models[0],'control')
    if fault=='false_measurement':backend.measurement.cleanup_complete=False
    elif fault=='false_credential':backend.measurement.credential_removed=False
    elif fault=='unconfirmed_pid':backend.unconfirmed_pid=7001
    elif fault=='held_password':backend.api_key_file=Path('held-password.txt')
    elif fault=='active':
        backend.is_active=lambda:True  # type: ignore[method-assign]
    released=engine.release(abort=False)
    assert released!=dict(process_gone=True,credential_removed=True,problem=None)
    with pytest.raises(ValueError):engine.start(fixture_manifest().models[0],'control')


@pytest.mark.parametrize('mode',('dribble','late_headers','oversize','malformed','redirect'))
def test_bounded_http_reader_rejects_delayed_or_bad_replies(mode: str) -> None:
    import http.server
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args: Any) -> None:pass
        def do_GET(self) -> None:
            try:
                if mode=='late_headers':time.sleep(.15)
                self.send_response(302 if mode=='redirect' else 200);self.send_header('Connection','close');self.end_headers()
                if mode=='dribble':
                    for piece in b'[{"id":0}]':
                        self.wfile.write(bytes([piece]));self.wfile.flush();time.sleep(.02)
                else:self.wfile.write(b'x'*500 if mode=='oversize' else b'not-json' if mode=='malformed' else b'[]')
            except (OSError,BrokenPipeError):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with pytest.raises((ValueError,OSError,TimeoutError)):
            inspect_json(f'http://127.0.0.1:{server.server_address[1]}','fixture','/slots',timeout_s=.06,maximum_bytes=100)
    finally:server.shutdown();server.server_close();thread.join(timeout=1)


@pytest.mark.parametrize('fault',('none','missing','duplicate','wrong','simulation','lookup_disabled','control_enabled','startup_eight'))
def test_offline_cli_recomputes_and_preserves_input_and_existing_output(tmp_path: Path,fault: str) -> None:
    raw,m=complete_record(tmp_path)
    p=proposal(m);pbytes=json.dumps(p).encode();phash=hashlib.sha256(pbytes).hexdigest()
    manifest=m.model_copy(update=dict(proposal_sha256=phash))
    raw['manifest_sha256']=digest(manifest.model_dump(mode='json'))
    if fault=='missing':raw['loads'].pop()
    elif fault=='duplicate':raw['loads'].append(deepcopy(raw['loads'][0]))
    elif fault=='wrong':raw['loads'][0]['tokens'][0]['ids'][0]+=1
    elif fault=='simulation':raw['simulated']=True
    elif fault in ('lookup_disabled','control_enabled','startup_eight'):
        row=raw['loads'][1 if fault=='lookup_disabled' else 0]
        damage_slot_evidence(row['snapshots'][0],fault)
    paths=[tmp_path/name for name in ('record.json','rules.json','proposal.json','audit.json')]
    source,rules,proposal_file,out=paths
    source.write_text(json.dumps(raw));rules.write_text(manifest.model_dump_json());proposal_file.write_bytes(pbytes)
    original=source.read_bytes()
    auditor=importlib.import_module('audit_lookup_fit')
    args=[str(source),'--rules',str(rules),'--rules-sha256',hashlib.sha256(rules.read_bytes()).hexdigest(),
          '--proposal',str(proposal_file),'--proposal-sha256',phash,'--out',str(out)]
    with patch('subprocess.Popen',side_effect=AssertionError('no processes')):
        assert auditor.main(args)==(0 if fault=='none' else 2)
        before=out.read_bytes()
        assert auditor.main(args)==4 and out.read_bytes()==before and source.read_bytes()==original


@pytest.mark.parametrize('where',('loads','load','snapshot','image','tokens','preservation'))
def test_malformed_record_shapes_are_audited_as_incomplete_not_accepted(tmp_path: Path,where: str) -> None:
    raw,m=complete_record(tmp_path)
    if where=='loads':raw['loads']='not a list'
    elif where=='load':raw['loads'][0]=None
    elif where=='snapshot':raw['loads'][0]['snapshots'][0]=None
    elif where=='image':raw['loads'][0]['snapshots'][0]['image']=None
    elif where=='tokens':raw['loads'][0]['tokens']=[None]*200
    elif where=='preservation':raw['preservation']=None
    assert not audit_fit(raw,m)['observations_complete']


@pytest.mark.parametrize('field',tuple(type(fixture_manifest().models[0].configuration).model_fields))
@pytest.mark.parametrize('route',('collector','auditor'))
def test_constructed_omissions_are_checked_before_manifest_serialization(field: str,route: str,tmp_path: Path) -> None:
    from inferenceos.tune.lookup_qualification import ConfigurationSnapshot
    raw,m=complete_record(tmp_path)
    values=m.models[0].configuration.model_dump();values.pop(field)
    incomplete=ConfigurationSnapshot.model_construct(**values)
    bad=m.model_copy(update=dict(models=(m.models[0].model_copy(update=dict(configuration=incomplete)),m.models[1])))
    if route=='auditor':assert not audit_fit(raw,bad)['observations_complete']
    else:
        engine=FitSimulator(m)
        with pytest.raises(ValueError):collect_fit(engine,bad,run_id='test',power=lambda:'ac',guard=lambda:None,checkpoint=lambda _:None)
        assert engine.starts==0


def test_cli_checkpointed_interrupt_keeps_original_exception(tmp_path: Path) -> None:
    import inferenceos.tune.lookup_fit_sim as sim
    boom=KeyboardInterrupt('same owner interrupt')
    def interrupted(*,checkpoint: Any) -> Any:
        checkpoint(dict(status='running',loads=[dict(partial='kept')]))
        raise boom
    out=tmp_path/'interrupt.json'
    with patch.object(sim,'simulate_fit',side_effect=interrupted):
        with pytest.raises(KeyboardInterrupt) as caught:runner().main(['--simulate','--out',str(out)])
    assert caught.value is boom
    raw=json.loads(out.read_text());assert raw['loads']==[dict(partial='kept')] and raw['status']=='interrupted'


def test_changed_owned_output_is_never_overwritten_by_later_checkpoints(tmp_path: Path) -> None:
    import inferenceos.tune.lookup_fit_sim as sim
    out=tmp_path/'changed.json'
    def changed(*,checkpoint: Any) -> Any:
        out.write_bytes(b'ANOTHER WRITER')
        checkpoint(dict(status='running'))
    with patch.object(sim,'simulate_fit',side_effect=changed):
        assert runner().main(['--simulate','--out',str(out)])==7
    assert out.read_bytes()==b'ANOTHER WRITER'


def test_tokenizer_error_envelope_cannot_be_accepted_alongside_ids() -> None:
    engine=fake_live(FakeBackend())
    with patch.object(engine,'_inspect',return_value=dict(tokens=[1,2],error='request failed')):
        with pytest.raises(ValueError):engine.tokenize('fixture',add_special=True)


@pytest.mark.parametrize('changed',('machine','effects','process'))
def test_before_stop_drift_closes_live_admission_not_just_offline_verdict(changed: str) -> None:
    engine=FitSimulator(fixture_manifest());original=engine.observe;calls=0
    def drift() -> Any:
        nonlocal calls
        calls+=1;raw=original()
        if calls==2:
            if changed=='machine':raw['machine']['power_plan']='66666666-2222-3333-4444-555555555555'
            elif changed=='effects':
                raw['startup']['text']=raw['startup']['text'].replace('n_threads = 4','n_threads = 5')
                raw['startup']['text_sha256']=hashlib.sha256(raw['startup']['text'].encode()).hexdigest()
                raw['effects']=observe_effects(raw['startup']['text'],raw['slots'])
            elif changed=='process':raw['image']['pid']+=99;raw['image']['create_time_s']+=99
        return raw
    engine.observe=drift  # type: ignore[method-assign]
    record,_=collect(engine)
    assert engine.starts==engine.releases==1 and record['status']=='stopped'
    assert len(record['not_run'])==3 and all('cause' in r for r in record['not_run'])


def test_owned_snapshots_do_not_alias_a_readers_mutable_cache() -> None:
    engine=FitSimulator(fixture_manifest());original=engine.observe;shared: dict[str, Any]={};calls=0
    def observe() -> Any:
        nonlocal calls
        calls+=1
        if not shared:shared.update(original())
        else:shared['machine']['power_plan']='66666666-2222-3333-4444-555555555555'
        return shared
    engine.observe=observe  # type: ignore[method-assign]
    raw,_=collect(engine)
    assert raw['status']=='stopped' and engine.starts==1
    snaps=raw['loads'][0]['snapshots']
    assert snaps[0]['machine']['power_plan']!=snaps[1]['machine']['power_plan']


def test_credential_and_process_identity_reads_are_inside_deadline_workers() -> None:
    r=runner();local=threading.local();reads=[];native_bounded=r.bounded_read
    def scope(call: Any,*,timeout_s: float) -> Any:
        def checked() -> Any:
            local.inside=True
            return call()
        return native_bounded(checked,timeout_s=timeout_s)
    sim=FitSimulator(fixture_manifest());sim.start(fixture_manifest().models[0],'control');snap=sim.observe()
    class Password:
        def read_text(self,**kwargs: Any) -> str:
            assert getattr(local,'inside',False),'credential read escaped deadline'
            reads.append('password');return 'fake key\n'
    class Backend:
        pid=10001
        base_url='http://127.0.0.1:9000'
        api_key_file=Password()
        last_command=snap['command']
        @property
        def identity(self) -> Any:
            assert getattr(local,'inside',False),'identity read escaped deadline'
            reads.append('identity');return SimpleNamespace(pid=10001,create_time=1001)
        def startup_log_snapshot(self) -> Any:return deepcopy(snap['startup'])
        def engine_settings(self) -> Any:return {}
    preflight=SimpleNamespace(selected=Path(snap['image']['executable']))
    engine=r.LiveFitEngine(preflight,backend_factory=Backend,inventory=fixture_machine);engine.owed=True
    def http(url: Any,key: Any,path: str,**kwargs: Any) -> Any:
        assert getattr(local,'inside',False)
        return deepcopy(snap['slots']) if path=='/slots' else {'tokens':[1,2]}
    with patch.object(r,'bounded_read',side_effect=scope),patch.object(r,'process_image',return_value=deepcopy(snap['image'])),\
         patch.object(r,'inspect_json',side_effect=http):
        assert engine.observe()['process_current']
        assert engine.tokenize('fixture',add_special=True)==[1,2]
    assert reads.count('identity')==2 and reads.count('password')==2


@pytest.mark.parametrize('inline_key',(False,True))
@pytest.mark.parametrize('fault',('none','wrong_executable','empty_executable','missing_executable','empty_command','absent_command'))
def test_production_command_builder_observer_and_collector_agree(fault: str,inline_key: bool) -> None:
    """Keep real argv evidence and credential redaction through the full live adapter.

    Only external IO is fabricated: no patched observe, redactor, command builder or
    validator, and no real process, model or password file is created.
    """
    from inferenceos.backends.llama_server_toolchain import build_server_command
    from inferenceos.backends.resident_server import _options_from, redacted_command
    r=runner();manifest=fixture_manifest();sim=FitSimulator(manifest)
    selected=Path('C:/D2-FIT-FIXTURE/tools/llama.cpp/llama-server.exe')
    secret_path=Path('C:/D2-FIT-FIXTURE/never-created-password-file')
    secret='never-written-password-value'
    requests=[];commands=[];command_reads=[]
    class Password:
        def read_text(self,**kwargs: Any) -> str:return secret+'\n'
    class Backend:
        api_key_file: Any=None
        unconfirmed_pid=None
        undeleted_key_files: list[Path]=[]
        base_url='http://127.0.0.1:9000'
        command: list[str] | None=None
        @property
        def pid(self) -> int:return 10000+sim.starts
        @property
        def identity(self) -> Any:return SimpleNamespace(pid=self.pid,create_time=1000+sim.starts)
        @property
        def last_command(self) -> list[str] | None:
            command_reads.append(sim.starts)
            return self.command
        def start_if_current(self,model: Any,params: Any,*,current: Any) -> None:
            assert current()
            pin=next(m for m in manifest.models if m.label==model.label)
            sim.start(pin,'lookup' if params.lookup_decoding.algorithm=='ngram-simple' else 'control')
            self.api_key_file=Password()
            command=build_server_command(selected,_options_from(Path(model.path),params),
                                         host='127.0.0.1',port=9000,api_key_file=secret_path)
            if inline_key:command+=['--api-key',secret]
            assert command[0]==str(selected) and redacted_command(command)[0]=='-m'
            if fault=='wrong_executable':command[0]=str(selected.with_name('unselected-server.exe'))
            elif fault=='empty_executable':command[0]=''
            elif fault=='missing_executable':command=command[1:]
            elif fault in ('empty_command','absent_command'):command=[]
            self.command=None if fault=='absent_command' else command
            commands.append(deepcopy(self.command))
        def startup_log_snapshot(self) -> Any:return sim.observe()['startup']
        def engine_settings(self) -> Any:return {}
        def stop(self) -> Any:
            sim.release(abort=False);self.api_key_file=None
            return SimpleNamespace(cleanup_complete=True,credential_removed=True)
        def halt(self) -> Any:return self.stop()
        def is_active(self) -> bool:return sim.active
    backend=Backend()
    preflight=SimpleNamespace(selected=selected,settings=SimpleNamespace(llama_cpp_dir=selected.parent),
                              inputs=SimpleNamespace(executable=lambda _:selected),check=lambda:None,
                              models={m.label:SimpleNamespace(label=m.label,path='C:/D2-FIT-FIXTURE/'+m.path) for m in manifest.models})
    engine=r.LiveFitEngine(preflight,backend_factory=lambda:backend,inventory=fixture_machine)
    def image(pid: int) -> Any:
        raw=sim.observe()['image'];assert raw['pid']==pid
        raw['executable']=str(selected)
        return raw
    def http(url: str,key: str,path: str,*,body: Any,timeout_s: float) -> Any:
        assert url==backend.base_url and key==secret and timeout_s>0
        requests.append(path)
        if path=='/slots':return sim.observe()['slots']
        assert path=='/tokenize'
        return dict(tokens=sim.tokenize(body['content'],add_special=body['add_special']))
    with patch('subprocess.Popen',side_effect=AssertionError('no process creation')) as launches,\
         patch.object(r,'process_image',side_effect=image),patch.object(r,'inspect_json',side_effect=http):
        record=collect_fit(engine,manifest,run_id='production_command_test',power=lambda:'ac',guard=lambda:None,checkpoint=lambda _:None)
        launches.assert_not_called()
    assert secret not in json.dumps(record) and str(secret_path) not in json.dumps(record)
    assert sim.starts==sim.releases==(4 if fault=='none' else 1)
    assert not engine.owed and backend.api_key_file is None and not sim.active
    assert requests.count('/tokenize')==(800 if fault=='none' else 0)
    assert requests.count('/slots')==len(command_reads)==(8 if fault=='none' else 1)
    assert record['generation_requests']==0
    if fault=='none':
        assert record['status']=='done' and not record['not_run']
        assert len(record['loads'])==4 and all(len(row['tokens'])==200 for row in record['loads'])
        for row,command in zip(record['loads'],commands):
            for snap in row['snapshots']:
                assert command is not None and snap['command']==[command[0],*redacted_command(command)]
                assert snap['command'][0]==str(selected)
                assert observation_problems(snap,manifest.models[0].configuration,row['arm'],manifest.engine_files)==[]
    else:
        assert record['status']=='stopped' and len(record['not_run'])==3
        snap=record['loads'][0]['snapshots'][0]
        assert snap['command']==([] if not commands[0] else [commands[0][0],*redacted_command(commands[0])])
        assert observation_problems(snap,manifest.models[0].configuration,'control',manifest.engine_files)


@pytest.mark.parametrize('changed',('record','rules','proposal'))
def test_offline_audit_refuses_a_source_changed_during_analysis(tmp_path: Path,changed: str) -> None:
    raw,m=complete_record(tmp_path);pbytes=json.dumps(proposal(m)).encode();phash=hashlib.sha256(pbytes).hexdigest()
    manifest=m.model_copy(update=dict(proposal_sha256=phash));raw['manifest_sha256']=digest(manifest.model_dump(mode='json'))
    files={name:tmp_path/(name+'.json') for name in ('record','rules','proposal')}
    files['record'].write_text(json.dumps(raw));files['rules'].write_text(manifest.model_dump_json());files['proposal'].write_bytes(pbytes)
    auditor=importlib.import_module('audit_lookup_fit');original=auditor.audit_fit
    def mutate(*args: Any,**kwargs: Any) -> Any:
        result=original(*args,**kwargs);files[changed].write_bytes(b'replaced during audit');return result
    out=tmp_path/'audit.json'
    with patch.object(auditor,'audit_fit',side_effect=mutate):
        assert auditor.main([str(files['record']),'--rules',str(files['rules']),
                             '--rules-sha256',hashlib.sha256(files['rules'].read_bytes()).hexdigest(),
                             '--proposal',str(files['proposal']),'--proposal-sha256',phash,'--out',str(out)])==4
    assert not out.exists()


def test_bounded_file_reader_does_not_accept_or_allocate_an_oversize_input(tmp_path: Path) -> None:
    path=tmp_path/'small.json';path.write_bytes(b'a'*201)
    with pytest.raises(ValueError):bounded_file_bytes(path,maximum_bytes=200)
    assert bounded_file_bytes(path,maximum_bytes=201)==b'a'*201


def damage_slot_evidence(raw: dict[str,Any],fault: str) -> None:
    if fault=='lookup_disabled':
        for slot in raw['slots']:slot['speculative']=False
    elif fault=='control_enabled':
        for slot in raw['slots']:slot['speculative']=True
    elif fault=='mixed_slots':raw['slots'][0]['speculative']=not raw['slots'][0].get('speculative',False)
    elif fault=='missing_speculative':raw['slots'][0].pop('speculative',None)
    elif fault=='missing_last_speculative':raw['slots'][-1].pop('speculative',None)
    elif fault=='null_speculative':raw['slots'][0]['speculative']=None
    elif fault=='numeric_speculative':raw['slots'][0]['speculative']=int(raw['slots'][0].get('speculative',False))
    elif fault=='float_speculative':raw['slots'][0]['speculative']=float(raw['slots'][0].get('speculative',False))
    elif fault=='string_speculative':raw['slots'][0]['speculative']=str(raw['slots'][0].get('speculative',False)).lower()
    elif fault=='startup_eight':raw['startup']['text']=raw['startup']['text'].replace('n_slots = 4','n_slots = 8')
    elif fault=='startup_missing':
        raw['startup']['text']='\n'.join(line for line in raw['startup']['text'].splitlines() if 'n_slots =' not in line)
    elif fault=='startup_conflicting_repeated':raw['startup']['text']+='\nload_model: initializing, n_slots = 8, n_ctx_slot = 4096\n'
    else:raise AssertionError(fault)
    raw['startup']['text_sha256']=hashlib.sha256(raw['startup']['text'].encode()).hexdigest()
    raw['effects']=observe_effects(raw['startup']['text'],raw['slots'])


@pytest.mark.parametrize('fault',('lookup_disabled','control_enabled','mixed_slots','missing_speculative','missing_last_speculative',
                                  'null_speculative','numeric_speculative','float_speculative','string_speculative',
                                  'startup_eight','startup_missing','startup_conflicting_repeated'))
@pytest.mark.parametrize('phase',('ready','before_stop'))
@pytest.mark.parametrize('route',('collector','audit'))
def test_slot_contradictions_refuse_at_their_live_or_offline_boundary(tmp_path: Path,fault: str,phase: str,route: str) -> None:
    target=1 if fault=='lookup_disabled' else 0
    if route=='audit':
        raw,m=complete_record(tmp_path)
        damage_slot_evidence(raw['loads'][target]['snapshots'][int(phase=='before_stop')],fault)
        checked=audit_fit(raw,m)
        assert not checked['observations_complete'],checked
        return
    engine=FitSimulator(fixture_manifest());original=engine.observe
    observations: dict[int,int]={}
    def observe() -> Any:
        raw=original()
        index=observations.get(engine.starts,0);observations[engine.starts]=index+1
        if engine.starts==target+1 and index==int(phase=='before_stop'):damage_slot_evidence(raw,fault)
        return raw
    engine.observe=observe  # type: ignore[method-assign]
    result,_=collect(engine)
    assert result['status']=='stopped' and engine.starts==engine.releases==target+1
    assert len(result['not_run'])==3-target
    expected_tokens=200*(target+int(phase=='before_stop'))
    assert sum(c[0]=='tokenize' for c in engine.calls)==expected_tokens
    assert len(result['loads'][-1]['tokens'])==(200 if phase=='before_stop' else 0)


@pytest.mark.parametrize('reuse_at,reuse_from',((1,0),(2,0),(3,1)))
def test_reused_process_is_refused_live_before_tokenizing_that_load(reuse_at: int,reuse_from: int) -> None:
    engine=FitSimulator(fixture_manifest());original=engine.observe
    def observe() -> Any:
        raw=original()
        if engine.starts==reuse_at+1:
            raw['image'].update(pid=10001+reuse_from,create_time_s=1001+reuse_from)
        return raw
    engine.observe=observe  # type: ignore[method-assign]
    result,_=collect(engine)
    assert result['status']=='stopped' and engine.starts==engine.releases==reuse_at+1
    assert len(result['not_run'])==3-reuse_at
    assert sum(c[0]=='tokenize' for c in engine.calls)==reuse_at*200
    assert result['loads'][-1]['tokens']==[]
    assert any('process identity reused' in e['message'] for e in result['loads'][-1]['errors'])


@pytest.mark.parametrize('same_component',('pid','create_time_s'))
def test_only_the_same_process_pair_is_a_reuse_not_one_component(tmp_path: Path,same_component: str) -> None:
    engine=FitSimulator(fixture_manifest());original=engine.observe
    def observe() -> Any:
        raw=original();raw['image'][same_component]=10001 if same_component=='pid' else 1001
        return raw
    engine.observe=observe  # type: ignore[method-assign]
    result,_=collect(engine)
    assert result['status']=='done' and engine.starts==engine.releases==4
    raw,m=complete_record(tmp_path)
    for row in raw['loads']:
        for snap in row['snapshots']:snap['image'][same_component]=10001 if same_component=='pid' else 1001
    assert audit_fit(raw,m)['observations_complete']


@pytest.mark.parametrize('field',tuple(INITIALIZATION_POLICY))
@pytest.mark.parametrize('change',('missing','contradiction','numeric_alias'))
def test_prospective_initialization_contract_is_bound_before_live_inputs(field: str,change: str) -> None:
    m=fixture_manifest();p=proposal(m)
    if change=='missing':p['initialization'].pop(field)
    elif change=='contradiction':p['initialization'][field]=None
    else:
        old=p['initialization'][field]
        if type(old) is bool:p['initialization'][field]=int(old)
        elif type(old) is int:p['initialization'][field]=float(old)
        else:p['initialization'][field]='disabled'
    with pytest.raises(ValueError):parse_manifest(m.model_dump_json(),p,m.proposal_sha256)


@pytest.mark.parametrize('field,value',(('version','lookup-fit-manifest-1'),('protocol','lookup-qualification-protocol-1')))
def test_version_one_inputs_cannot_silently_enter_the_amended_protocol(field: str,value: str) -> None:
    m=fixture_manifest();raw=m.model_dump(mode='json');raw[field]=value
    with pytest.raises(ValueError):parse_manifest(json.dumps(raw),proposal(m),m.proposal_sha256)


def test_version_one_record_is_not_reinterpreted_as_new_evidence(tmp_path: Path) -> None:
    raw,m=complete_record(tmp_path);raw['version']='lookup-fit-1'
    assert not audit_fit(raw,m)['observations_complete']


@pytest.mark.parametrize('fault',('missing_log','disable','enable','equals_override'))
@pytest.mark.parametrize('phase',('ready','before_stop'))
@pytest.mark.parametrize('route',('collector','audit'))
def test_initialization_observations_keep_the_pinned_default(tmp_path: Path,fault: str,phase: str,route: str) -> None:
    def damage(raw: dict[str,Any]) -> None:
        if fault=='missing_log':
            raw['startup']['text']='\n'.join(line for line in raw['startup']['text'].splitlines()
                                             if 'warming up the model with an empty run' not in line)
            raw['startup']['text_sha256']=hashlib.sha256(raw['startup']['text'].encode()).hexdigest()
        else:raw['command'].append({'disable':'--no-warmup','enable':'--warmup','equals_override':'--no-warmup=true'}[fault])
    if route=='audit':
        raw,m=complete_record(tmp_path);damage(raw['loads'][0]['snapshots'][int(phase=='before_stop')])
        assert not audit_fit(raw,m)['observations_complete']
        return
    engine=FitSimulator(fixture_manifest());original=engine.observe;observations=0
    def observe() -> Any:
        nonlocal observations
        raw=original()
        if observations==int(phase=='before_stop'):damage(raw)
        observations+=1
        return raw
    engine.observe=observe  # type: ignore[method-assign]
    result,_=collect(engine)
    assert result['status']=='stopped' and engine.starts==engine.releases==1
    assert sum(c[0]=='tokenize' for c in engine.calls)==(200 if phase=='before_stop' else 0)
    assert len(result['not_run'])==3
