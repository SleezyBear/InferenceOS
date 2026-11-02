"""D2 Stage-B software evidence core. No live dispatcher or tuning permission."""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import math
import statistics
from typing import Any

from inferenceos.tune.lookup_corpus import build_corpus, digest, TaskCase
from inferenceos.tune.lookup_fit import observation_problems, token_problems, REQUIRED_EFFECTS,relative_path
from inferenceos.tune.lookup_fit_observers import machine_from_raw
from inferenceos.tune.lookup_role_cost import RoleScope
from inferenceos.tune.stats import block_bounds

VERSION = "lookup-answer-core-1"
MODELS = ("llama3b", "gemma1b")
FAMILIES = ("copy-prose", "copy-code", "extract-rows")
ROLES = ("primary", "calibration")
SOFT_TARGET_NS = 43_200_000_000_000


def suites() -> list[dict[str, Any]]:
    corpus=build_corpus();result=[]
    for family in FAMILIES:
        for role,split in (("primary","holdout"),("calibration","calibration")):
            cases=[c for c in corpus.cases if c.task.family==family and c.split==split]
            definition=dict(version="lookup-finite-suite-1",family=family,role=role,split=split,
                            corpus_sha256=corpus.sha256,ordered_cases=[(c.case_id,c.sha256) for c in cases],
                            fresh_load_per_case=True,quantity_per_case=1,request_ceiling=512,
                            context_per_slot=4096,server_slots=4,native=True)
            result.append(dict(definition,sha256=digest(definition)))
    return result


def plan() -> list[dict[str, Any]]:
    result: list[dict[str,Any]]=[];definitions={(s['family'],s['role']):s for s in suites()}
    for block in range(8):
        models=list(enumerate(MODELS))
        if block%2:models.reverse()
        families=FAMILIES[block%3:]+FAMILIES[:block%3]
        for m,model in models:
            for family in families:
                for r,role in enumerate(ROLES):
                    arms=("control","lookup") if (block+m+FAMILIES.index(family)+r)%2==0 else ("lookup","control")
                    suite=definitions[family,role]
                    for arm in arms:
                        for case_id,case_sha in suite['ordered_cases']:
                            result.append(dict(slot=len(result),block=block,model=model,family=family,
                                               cell=model+'/'+family,role=role,arm=arm,case_id=case_id,
                                               case_sha256=case_sha,suite_sha256=suite['sha256']))
    return result


def request_body(case: TaskCase) -> dict[str, Any]:
    return dict(prompt=case.prompt,n_predict=512,seed=42,temperature=0,stop=[],stream=True,
                cache_prompt=False,ignore_eos=False,return_tokens=True)


def scope_problems(scope: Any) -> list[str]:
    """Scope is evidence data, never a live manifest or an authorization object."""
    try:
        if not isinstance(scope,dict) or set(scope)!= {'models','machine','engine_files','executor_sha256'}:
            raise ValueError('complete software evidence scope required')
        if set(scope['models'])!=set(MODELS):raise ValueError('both fixed model scopes required')
        for model in MODELS:
            value=scope['models'][model]
            if set(value)!= {'requested','effective','fit_tokens','model_sha256'}:raise ValueError('complete model scope required')
            from inferenceos.tune.lookup_qualification import ConfigurationSnapshot
            import json
            requested=ConfigurationSnapshot.model_validate_json(json.dumps(value['requested']))
            effective=ConfigurationSnapshot.model_validate_json(json.dumps(value['effective']))
            if requested.context_length!=4096 or effective.context_length!=4096:raise ValueError('fixed context required')
            if any(getattr(effective,k) is None for k in REQUIRED_EFFECTS):raise ValueError('unknown required effective setting')
            if any(getattr(requested,k) is not None for k in ('engine_switches','placement_overrides','kv_cache_type')):
                raise ValueError('unregistered custom override/alias')
            if token_problems(value['fit_tokens']):raise ValueError('complete Stage-A fit arrays required')
            for name in ('model_sha256',):
                if not valid_digest(value[name]):raise ValueError('model content identity required')
        from inferenceos.tune.lookup_qualification import MachineBinding
        import json
        MachineBinding.model_validate_json(json.dumps(scope['machine']))
        if not valid_digest(scope['executor_sha256']):raise ValueError('executor identity required')
        from inferenceos.tune.lookup_audit import REQUIRED_ENGINE_FILES
        names=[]
        for path,sha in scope['engine_files'].items():
            relative_path(path)
            if not valid_digest(sha):raise ValueError('engine file hash required')
            names.append(path.rsplit('/',1)[-1].lower())
        if len(set(names))!=len(names) or not REQUIRED_ENGINE_FILES.issubset(names):raise ValueError('unambiguous required engine pins')
        return []
    except (ValueError,TypeError,KeyError,AttributeError) as exc:
        return [str(exc)]


def valid_digest(value: Any) -> bool:
    import re
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def role_scope(slot: dict[str,Any],scope: dict[str,Any]) -> RoleScope:
    suite=next(s for s in suites() if (s['family'],s['role'])==(slot['family'],slot['role']))
    model=scope['models'][slot['model']]
    return RoleScope.model_validate_json(__import__('json').dumps(dict(
        model=slot['model'],family=slot['family'],role=slot['role'],arm=slot['arm'],suite_sha256=suite['sha256'],
        model_sha256=model['model_sha256'],engine_files_sha256=digest(scope['engine_files']),
        requested_sha256=digest(model['requested']),effective_sha256=digest(model['effective']),machine_sha256=digest(scope['machine']),
        request_sha256=digest(dict(seed=42,temperature=0,n_predict=512,stop=[],stream=True,cache_prompt=False,ignore_eos=False,return_tokens=True)),
        executor_sha256=scope['executor_sha256'],ordered_cases=suite['ordered_cases'])))


def native_answer(events: Any,body: Any,case: TaskCase,arm: str,prompt_ids: list[int],
                  start_ns: int,end_ns: int) -> dict[str,Any]:
    """Reconstruct ALL incremental chunks, including terminal empty token arrays."""
    problems=[];text=[];ids=[];ids_seen=False;terminal=None;previous=start_ns
    if (arm not in ('control','lookup') or type(start_ns) is not int or type(end_ns) is not int or end_ns<=start_ns
            or not isinstance(prompt_ids,list) or not prompt_ids or any(type(t) is not int or t<0 for t in prompt_ids)):
        return dict(valid=False,problems=['invalid request interval, arm or complete prompt IDs'])
    if end_ns-start_ns>=120_000_000_000:
        return dict(valid=False,problems=['late completion exceeds the registered total request deadline'])
    if digest(body)!=digest(request_body(case)):problems.append('request differs from fixed fresh native scope')
    if not isinstance(events,list) or not events:return dict(valid=False,problems=['native stream events missing'])
    for event in events:
        if not isinstance(event,dict) or set(event)!= {'received_ns','data'}:
            problems.append('invalid native event shape');continue
        at=event['received_ns'];chunk=event['data']
        if type(at) is not int or not previous<=at<=end_ns:
            problems.append('chunk outside ordered request interval')
        else:previous=at
        if not isinstance(chunk,dict) or 'error' in chunk:
            problems.append('native error/non-object chunk');continue
        if terminal is not None:problems.append('work after terminal completion')
        if 'content' in chunk:
            if not isinstance(chunk['content'],str):problems.append('non-text content')
            else:text.append(chunk['content'])
        if 'tokens' in chunk:
            ids_seen=True
            tokens=chunk['tokens']
            if not isinstance(tokens,list) or any(type(t) is not int or t<0 for t in tokens):problems.append('malformed generated IDs')
            else:ids.extend(tokens)
        if chunk.get('stop') is True:terminal=chunk
        elif 'stop' in chunk and type(chunk['stop']) is not bool:problems.append('stop marker is not boolean')
    if terminal is None:return dict(valid=False,problems=problems+['terminal completion missing'])
    timings=terminal.get('timings');settings=terminal.get('generation_settings')
    if not isinstance(timings,dict):timings={};problems.append('native timings missing')
    if not isinstance(settings,dict):settings={};problems.append('effective settings missing')
    if not ids_seen:problems.append('returned ID array missing even if reported output count is zero')
    if len(ids)>512:problems.append('complete bounded output IDs required')
    if terminal.get('stop_type')!='eos' or terminal.get('truncated') is not False:
        problems.append('natural EOS and explicit no truncation required')
    for source,key,expected in ((terminal,'tokens_evaluated',len(prompt_ids)),(timings,'prompt_n',len(prompt_ids)),
                                (terminal,'tokens_predicted',len(ids)),(timings,'predicted_n',len(ids)),(timings,'cache_n',0)):
        if type(source.get(key)) is not int or source[key]!=expected:problems.append(key+' contradicts complete fresh work')
    for key in ('prompt_ms','predicted_ms'):
        if type(timings.get(key)) not in (int,float) or not math.isfinite(timings[key]) or timings[key]<=0:
            problems.append(key+' must be positive and finite')
    if type(terminal.get('id_slot')) is not int or not 0<=terminal['id_slot']<4:problems.append('valid slot identity required')
    allowed=('none','none,none') if arm=='control' else ('ngram-simple','none,ngram-simple')
    if settings.get('speculative.types') not in allowed:problems.append('effective speculative arm differs')
    for key,value in (('seed',42),('n_predict',512)):
        if type(settings.get(key)) is not int or settings[key]!=value:problems.append('effective '+key+' differs')
    if type(settings.get('temperature')) not in (int,float) or settings['temperature']!=0 or settings.get('ignore_eos') is not False:
        problems.append('effective sampling differs')
    drafted,accepted=timings.get('draft_n'),timings.get('draft_n_accepted')
    if 'draft_n' in timings or 'draft_n_accepted' in timings:
        if type(drafted) is not int or type(accepted) is not int or not 0<=accepted<=drafted:
            problems.append('draft counters contradictory or incomplete')
        elif arm=='control' and drafted:problems.append('control reports draft activity')
    observed_counts: dict[str,int]={}
    for event in events:
        chunk=event.get('data') if isinstance(event,dict) else None
        if not isinstance(chunk,dict):continue
        # b11139 leaves partial-result id_slot at the base class's documented
        # -1 sentinel. A concrete partial slot must agree with the final slot;
        # neither absence nor this sentinel is invented as a real observation.
        if 'id_slot' in chunk:
            value=chunk['id_slot']
            if (type(value) is not int or (value!=-1 and (not 0<=value<4 or value!=terminal.get('id_slot')))
                    or (chunk is terminal and value==-1)):
                problems.append('intermediate slot contradicts completed request')
        if 'truncated' in chunk and chunk['truncated'] is not False:
            problems.append('intermediate truncation contradicts completed work')
        if 'stop_type' in chunk and chunk['stop_type']!=('eos' if chunk.get('stop') is True else 'none'):
            problems.append('intermediate stopping reason contradicts completion')
        if 'generation_settings' in chunk:
            shown=chunk['generation_settings']
            if not isinstance(shown,dict):problems.append('malformed intermediate effective settings')
            else:
                for key,value in (('seed',42),('n_predict',512)):
                    if key in shown and (type(shown[key]) is not int or shown[key]!=value):
                        problems.append('intermediate effective '+key+' differs')
                if 'temperature' in shown and (type(shown['temperature']) not in (int,float) or shown['temperature']!=0):
                    problems.append('intermediate effective temperature differs')
                for key,value in (('ignore_eos',False),('cache_prompt',False),('stream',True),('return_tokens',True)):
                    if key in shown and shown[key] is not value:problems.append('intermediate effective '+key+' differs')
                if 'stop' in shown and shown['stop']!=[]:problems.append('intermediate effective stop differs')
                if 'speculative.types' in shown and shown['speculative.types'] not in allowed:
                    problems.append('intermediate effective speculative arm differs')
        for key,maximum in (('tokens_predicted',len(ids)),('tokens_evaluated',len(prompt_ids))):
            if key in chunk:
                value=chunk[key]
                if type(value) is not int or not observed_counts.get(key,0)<=value<=maximum:
                    problems.append('intermediate '+key+' contradicts work')
                else:observed_counts[key]=value
        if 'timings' in chunk:
            part=chunk['timings']
            if not isinstance(part,dict):problems.append('malformed intermediate timings');continue
            if 'cache_n' in part and (type(part['cache_n']) is not int or part['cache_n']!=0):problems.append('intermediate fresh-work contradiction')
            for key,maximum in (('predicted_n',len(ids)),('prompt_n',len(prompt_ids))):
                if key in part:
                    value=part[key];name='timings.'+key
                    if type(value) is not int or not observed_counts.get(name,0)<=value<=maximum:
                        problems.append('intermediate '+key+' contradicts work')
                    else:observed_counts[name]=value
            for key in ('draft_n','draft_n_accepted'):
                if key in part:
                    value=part[key];name='timings.'+key;final=timings.get(key)
                    if (type(value) is not int or value<observed_counts.get(name,0)
                            or type(final) is not int or value>final or (arm=='control' and value!=0)):
                        problems.append('intermediate '+key+' contradicts final draft accounting')
                    else:observed_counts[name]=value
            if ('draft_n' in part and 'draft_n_accepted' in part and type(part['draft_n']) is int
                    and type(part['draft_n_accepted']) is int and part['draft_n_accepted']>part['draft_n']):
                problems.append('intermediate accepted count exceeds drafted work')
            for key in ('prompt_ms','predicted_ms'):
                if key in part and (type(part[key]) not in (int,float) or not math.isfinite(part[key]) or part[key]<0):
                    problems.append('intermediate '+key+' must be finite and nonnegative')
    return dict(valid=not problems,problems=problems,text=''.join(text),token_ids=ids,
                drafted=drafted,accepted=accepted,terminal=terminal)


def observe_problems(raw: Any,slot: dict[str,Any],scope: dict[str,Any]) -> list[str]:
    try:
        from inferenceos.tune.lookup_qualification import ConfigurationSnapshot
        requested=ConfigurationSnapshot.model_validate_json(__import__('json').dumps(scope['models'][slot['model']]['requested']))
        problems=observation_problems(raw,requested,slot['arm'],scope['engine_files'])
        if problems:return problems
        if machine_from_raw(raw['machine'],raw['effects']['configuration']['device'])!=scope['machine']:
            problems.append('machine differs from frozen observed scope')
        if raw['effects']['configuration']!=scope['models'][slot['model']]['effective']:
            problems.append('effective base differs from frozen observed scope')
        return problems
    except (ValueError,TypeError,KeyError,AttributeError) as exc:return [str(exc)]


def analyze(record: dict[str,Any]) -> dict[str,Any]:
    """Final math on reconstructed rows; audit_answers is the evidence gate.

    No interim/favourable-prefix look. This result is never an admission certificate.
    """
    cells={};by_case={c.case_id:c for c in build_corpus().cases};planned=plan()
    for model in MODELS:
        for family in FAMILIES:
            cell=model+'/'+family
            rows=[r for r in record.get('rows',[]) if r.get('cell')==cell]
            eligible=record.get('status')=='done' and cell not in record.get('futile',{}) and len(rows)==224
            detail: dict[str,Any]=dict(complete=eligible,development_criteria_pass=False,problems=[],roles={})
            if not eligible:
                detail['problems']=['incomplete/operational stop or cell futility'];cells[cell]=detail;continue
            try:
                for role in ROLES:
                    effects=[];raw_pairs=[];draft_blocks=[]
                    ids=[c.case_id for c in by_case.values() if c.task.family==family and c.split==('holdout' if role=='primary' else 'calibration')]
                    for block in range(8):
                        selected=[r for r in rows if r['role']==role and r['block']==block]
                        ordered=[r for r in planned if r['cell']==cell and r['role']==role and r['block']==block]
                        if [r['slot'] for r in selected]!=[r['slot'] for r in ordered]:raise ValueError('missing/duplicate/reordered block')
                        pairs=[];accepted=0
                        for case_id in ids:
                            values=[next(r for r in selected if r['case_id']==case_id and r['arm']==arm) for arm in ('control','lookup')]
                            if any(r.get('verdict')!='right' for r in values):raise ValueError('whole-answer/evidence gate fails')
                            if (values[0]['answer']['text'],values[0]['answer']['token_ids'])!=(values[1]['answer']['text'],values[1]['answer']['token_ids']):raise ValueError('paired outputs differ')
                            times=[r['cycle_ns'] for r in values]
                            if any(type(v) is not int or v<=0 for v in times):raise ValueError('invalid inclusive cycle')
                            pairs.append(times);accepted+=values[1]['answer']['accepted'] or 0
                        effects.append(Fraction(sum(x-y for x,y in pairs),sum(x for x,y in pairs)))
                        raw_pairs.append(pairs);draft_blocks.append(accepted>0)
                    bounds=asdict(block_bounds([float(v) for v in effects],alpha=.05/12))
                    threshold=.05 if role=='primary' else -.02
                    passed=(statistics.stdev(float(v) for v in effects)>0 and bounds['lower'] is not None and bounds['lower']>threshold)
                    case_effects=[sum((Fraction(b[i][0]-b[i][1],b[i][0]) for b in raw_pairs),Fraction())/8 for i in range(7)]
                    passed=passed and all(v>=Fraction(-1,50) for v in case_effects)
                    if role=='primary':
                        passed=passed and all(draft_blocks)
                        passed=passed and all(sum(p[1] for b in raw_pairs[parity::2] for p in b)<sum(p[0] for b in raw_pairs[parity::2] for p in b) for parity in (0,1))
                    detail['roles'][role]=dict(passed=passed,bounds=bounds,rational_effects=[dict(numerator=v.numerator,denominator=v.denominator) for v in effects],
                                                case_means=[dict(numerator=v.numerator,denominator=v.denominator) for v in case_effects],accepted_draft_each_block=draft_blocks)
                detail['development_criteria_pass']=all(v['passed'] for v in detail['roles'].values())
            except (ValueError,KeyError,TypeError,StopIteration,ArithmeticError) as exc:
                detail['complete']=False;detail['problems']=[str(exc)]
            cells[cell]=detail
    return dict(cells=cells,analysis_looks=1 if record.get('status')=='done' else 0,
                actual_qualification=False,automatic_admission=False,deployment_allowed=False)
