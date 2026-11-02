"""Recompute software-only Stage-B evidence. No engine, forecast or permission."""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from inferenceos.tune.lookup_answers import (VERSION,SOFT_TARGET_NS,plan,scope_problems,role_scope,native_answer,observe_problems,analyze)
from inferenceos.tune.lookup_corpus import build_corpus,digest,judge
from inferenceos.tune.lookup_fit import release_complete
from inferenceos.tune.lookup_role_cost import audit_intervals


def audit_answers(raw: Any) -> dict[str,Any]:
    try:return _audit(raw)
    except (ValueError,TypeError,KeyError,AttributeError,IndexError,StopIteration,OverflowError) as exc:
        return dict(version=VERSION,evidence_valid=False,study_complete=False,
                    problems=['malformed evidence: '+str(exc)],
                    analysis=dict(analysis_complete=False,analysis_looks=0,cells={},actual_qualification=False,
                                  automatic_admission=False,deployment_allowed=False),
                    actual_qualification=False,automatic_admission=False,deployment_allowed=False)


def _audit_schedule(raw: dict[str,Any], schedule: list[dict[str,Any]], intervals: dict[str,dict[str,Any]],
                    children: dict[str,list[dict[str,Any]]], futile: dict[str,Any], root: dict[str,Any]) -> None:
    """Consume measured work once in fixed admission order, including skipped cells.

    Admissions carry no duration, so their position is proved by the preceding
    measured pair/rest and their own assigned work. A skipped futile pair still
    advances the decision clock and still owns its scheduled rest.
    """
    operational = [span for span in intervals.values()
                   if span['kind'] not in ('job','suite-arm') and span['outcome'] != 'complete']
    if operational:
        raise ValueError('unsuccessful required operation: '+operational[0]['kind'])
    admissions=raw['pair_admissions'];rests=raw['rests'];rows=raw['rows']
    if not isinstance(rests,list):raise ValueError('ordered rest receipts required')
    rest_ids=[item['sample_id'] for item in rests]
    measured_rests=[span['sample_id'] for span in intervals.values() if span['kind']=='rest']
    if len(set(rest_ids))!=len(rest_ids) or rest_ids!=measured_rests:
        raise ValueError('each measured rest needs one unique ordered schedule receipt')
    actual=children.get(root['sample_id'],[]);position=0;rest_position=0;cursor=root['start_ns']
    by_group: dict[int,list[dict[str,Any]]]={}
    for row in rows:by_group.setdefault(row['slot']//14,[]).append(row)
    for number,admission in enumerate(admissions):
        if admission['at_ns']<cursor:raise ValueError('pair admission predates preceding measured work')
        cursor=admission['at_ns']
        if not admission['permitted']:
            if raw['status']!='stopped' or number!=len(admissions)-1:
                raise ValueError('refused admission must terminate the study')
            break
        first=14*number;cell=schedule[first]['cell']
        skipped=cell in futile and futile[cell]['at_slot']<first
        members=by_group.get(number,[])
        if skipped:
            if members:raise ValueError('attempted work for a skipped futile pair')
        else:
            if not members or position>=len(actual) or actual[position]['kind']!='pair':
                raise ValueError('admitted active pair lacks its own measured work')
            pair=actual[position];position+=1
            if pair['start_ns']<cursor:raise ValueError('assigned pair predates its own admission')
            owners={intervals[intervals[row['cycle_sample_id']]['parent_id']]['parent_id'] for row in members}
            if owners!={pair['sample_id']}:
                raise ValueError('pair admission differs from actual case ownership')
            cursor=pair['end_ns']
        if number%2:
            if position>=len(actual) or actual[position]['kind']!='rest' or rest_position>=len(rests):
                raise ValueError('scheduled rest missing before the next admission')
            span=actual[position];item=rests[rest_position];position+=1;rest_position+=1
            if (type(item['after_slot']) is not int or item['after_slot']!=first+13
                    or type(item['seconds']) is not int or item['seconds']!=30 or item['complete'] is not True
                    or item['sample_id']!=span['sample_id'] or span['start_ns']<cursor
                    or span['outcome']!='complete' or span['end_ns']-span['start_ns']<30_000_000_000):
                raise ValueError('scheduled rest is incomplete, short, reordered or assigned to other work')
            cursor=span['end_ns']
    if position!=len(actual) or rest_position!=len(rests):
        raise ValueError('unowned work/rest outside the admitted schedule')
    if raw['status']=='stopped' and (not admissions or admissions[-1]['permitted']):
        raise ValueError('operational stop cannot certify completed evidence')


def _audit(raw: dict[str,Any], *, native: bool=False) -> dict[str,Any]:
    problems=[]
    if (raw.get('version')!=VERSION or raw.get('simulated') is not (not native)
            or raw.get('live_dispatch_implemented') is not native):
        raise ValueError('core execution domain differs from its audited envelope')
    if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',raw.get('run_id','')):raise ValueError('run identity missing')
    if raw.get('cost_clock_domain')!=('monotonic-ns' if native else 'simulated') or raw.get('final_publication_cost_right_censored') is not True:
        raise ValueError('honest simulated clock and final publication censoring required')
    if raw.get('unmeasured_operations')!=[dict(kind=k,state='right-censored') for k in ('initial-scope-validation','final-analysis','final-publication')]:
        raise ValueError('unimplemented caller cost boundaries must remain explicit, not zero')
    scope=raw['scope'];issues=scope_problems(scope)
    if issues:raise ValueError('; '.join(issues))
    schedule=plan();cases={c.case_id:c for c in build_corpus().cases}
    if raw['scope_sha256']!=digest(scope) or raw['plan_sha256']!=digest(schedule):raise ValueError('scope/plan digest differs')
    costs=audit_intervals(raw['cost_intervals'])
    if not costs['valid']:raise ValueError('; '.join(costs['problems']))
    intervals={c['sample_id']:c for c in raw['cost_intervals']}
    if list(intervals)!=[f"{raw['run_id']}/cost/{i}" for i in range(len(intervals))]:raise ValueError('cost IDs must belong to this run and order')
    children: dict[str,list[dict[str,Any]]]={}
    for span in intervals.values():children.setdefault(span['parent_id'],[]).append(span)
    root=raw['cost_intervals'][0]
    origin=raw.get('target_origin_ns',root['start_ns'])
    if type(origin) is not int or not 0<=origin<=root['start_ns']:
        raise ValueError('target origin must precede measured collection')
    if root['outcome']!=('complete' if raw['status']=='done' else 'failed') or root['end_ns'] is None:
        raise ValueError('job cost outcome differs from study termination')
    if any(key in raw for key in ('publication_problem','analysis_problem')):
        raise ValueError('observed publication/analysis failure cannot certify evidence')
    if raw['status']=='done' and 'stop' in raw:
        raise ValueError('observed stop cannot certify a completed record')
    if any(s['kind'] not in ('pair','rest') for s in children.get(root['sample_id'],[])):
        raise ValueError('unexpected work at job scope')
    admissions=raw['pair_admissions']
    if not isinstance(admissions,list) or len(admissions)>96:raise ValueError('fixed pair admissions required')
    for i,admission in enumerate(admissions):
        if (admission['first_slot']!=14*i or type(admission['first_slot']) is not int or type(admission['at_ns']) is not int
                or type(admission['permitted']) is not bool or not root['start_ns']<=admission['at_ns']<=root['end_ns']
                or (i and admission['at_ns']<admissions[i-1]['at_ns'])
                or admission['permitted']!=(admission['at_ns']-origin<SOFT_TARGET_NS)):
            raise ValueError('soft-target admission differs from measured clock')
        if not admission['permitted'] and i!=len(admissions)-1:raise ValueError('work admission after target stop')
    rows=raw['rows'];positions=[r['slot'] for r in rows]
    if any(type(i) is not int or not 0<=i<1344 for i in positions) or positions!=sorted(set(positions)):
        raise ValueError('duplicate/reordered/unknown answer position')
    done=deepcopy(raw);futile={};rebuilt: list[dict[str,Any]]=[];identities=set();pair_groups: dict[str,int]={}
    for row in rows:
        slot=schedule[row['slot']]
        admission=admissions[row['slot']//14]
        if not admission['permitted']:raise ValueError('answer after refused pair admission')
        if any(digest(row.get(k))!=digest(v) for k,v in slot.items()):raise ValueError('answer slot identity differs')
        if row['load_id']!=f"{raw['run_id']}/B/{row['slot']}":raise ValueError('mixed run/load identity')
        if row['cell'] in futile:raise ValueError('measurement after cell futility')
        if row['errors']:
            if row is not rows[-1] or raw['status']!='stopped':raise ValueError('work after operational error')
            problems.append('operationally interrupted row retained');continue
        if row['power']!=['ac','ac'] or not release_complete(row['cleanup']) or row['complete'] is not True:
            raise ValueError('incomplete power/release work')
        cycle=intervals[row['cycle_sample_id']];request=intervals[row['request_sample_id']]
        if (cycle['kind']!='cycle' or cycle['case_id']!=row['case_id'] or cycle['outcome']!='complete'
                or digest(cycle['scope'])!=digest(role_scope(slot,scope).model_dump(mode='json'))):
            raise ValueError('inclusive cycle scope differs')
        if request['kind']!='request' or request['parent_id']!=cycle['sample_id'] or request['outcome']!='complete':
            raise ValueError('request interval does not belong to cycle')
        suite=intervals[cycle['parent_id']]
        pair=intervals[suite['parent_id']]
        if (suite['kind']!='suite-arm' or pair['kind']!='pair' or pair['parent_id']!=root['sample_id']
                or digest(suite['scope'])!=digest(cycle['scope']) or cycle['start_ns']<admission['at_ns']):
            raise ValueError('case/suite/pair containment or scope differs')
        if [s['kind'] for s in children.get(cycle['sample_id'],[])]!=['power','gate','start','ready-identity','tokenize','request','oracle','release-identity','cleanup','power']:
            raise ValueError('cycle omits/reorders actual required work')
        if any(s['outcome']!='complete' for s in children[cycle['sample_id']]):
            raise ValueError('completed case cannot hide failed work')
        if pair_groups.setdefault(pair['sample_id'],row['slot']//14)!=row['slot']//14:
            raise ValueError('one measured pair cannot alias separate scheduled pairs')
        if pair['start_ns']<admission['at_ns']:
            raise ValueError('pair work predates admission')
        for child in children[cycle['sample_id']]:
            limit={'start':120_000_000_000,'request':120_000_000_000,'ready-identity':30_000_000_000,
                   'tokenize':30_000_000_000,'release-identity':30_000_000_000}.get(child['kind'])
            if limit is not None and child['end_ns']-child['start_ns']>=limit:raise ValueError('late operation cannot be a completed sample')
        if type(row['cycle_ns']) is not int or row['cycle_ns']!=cycle['end_ns']-cycle['start_ns']:
            raise ValueError('derived cycle duration differs')
        if [s['phase'] for s in row['snapshots']]!=['ready','before_stop']:raise ValueError('both observation phases required')
        images=[]
        for snap in row['snapshots']:
            issues=observe_problems(snap,slot,scope)
            if issues:raise ValueError('; '.join(issues))
            images.append((snap['image']['pid'],snap['image']['create_time_s']))
        if images[0]!=images[1] or images[0] in identities:raise ValueError('reused/replaced process')
        identities.add(images[0])
        expected=next(t['ids'] for t in scope['models'][slot['model']]['fit_tokens'] if t['case_id']==slot['case_id'] and t['kind']=='prompt')
        if digest(row['prompt_ids'])!=digest(expected):raise ValueError('prompt IDs differ from fit')
        answer=native_answer(row['events'],row['request'],cases[row['case_id']],row['arm'],row['prompt_ids'],request['start_ns'],request['end_ns'])
        if not answer['valid']:raise ValueError('; '.join(answer['problems']))
        verdict=judge(cases[row['case_id']],answer['text'],ending='eos',truncated=False)
        word='right' if verdict.passed else 'wrong'
        if digest(row['answer'])!=digest(answer) or row['verdict']!=word or digest(row['judge'])!=digest(verdict.model_dump(mode='json')):
            raise ValueError('stored answer/judge does not match raw reconstruction')
        derived=deepcopy(row);derived.update(answer=answer,verdict=word)
        paired=next((r for r in rebuilt if (r['block'],r['cell'],r['role'],r['case_id'])==(row['block'],row['cell'],row['role'],row['case_id']) and r['arm']!=row['arm']),None)
        cause=None
        if word=='wrong':
            cause=('task-unqualified control' if row['arm']=='control' else
                   'candidate regression' if paired and paired['verdict']=='right' else 'lookup wrong; paired control unobserved')
        elif paired and (answer['text'],answer['token_ids'])!=(paired['answer']['text'],paired['answer']['token_ids']):
            cause='paired exact output preservation fails'
        if cause:futile[row['cell']]=dict(cause=cause,at_slot=row['slot'])
        rebuilt.append(derived)
    cycle_ids={r['cycle_sample_id'] for r in rows}
    if {i for i,s in intervals.items() if s['kind']=='cycle'}!=cycle_ids:raise ValueError('cost cycles differ from actual attempted answers')
    if {i for i,s in intervals.items() if s['kind']=='request'}!={r['request_sample_id'] for r in rows if 'request_sample_id' in r}:
        raise ValueError('request costs differ from actual requests')
    row_by_cycle={r['cycle_sample_id']:r for r in rows}
    if raw['status']=='done':
        pairs=[s for s in intervals.values() if s['kind']=='pair']
        if [p['sample_id'] for p in pairs]!=list(pair_groups):raise ValueError('pair costs differ from actual ordered work')
        for pair in pairs:
            group=pair_groups[pair['sample_id']]
            members=[r for r in rows if r['slot']//14==group]
            actual_suites=[s for s in children.get(pair['sample_id'],[]) if s['kind']=='suite-arm']
            suite_ids=list(dict.fromkeys(intervals[r['cycle_sample_id']]['parent_id'] for r in members))
            if (pair['outcome']!='complete' or [s['sample_id'] for s in actual_suites]!=suite_ids
                    or len(children.get(pair['sample_id'],[]))!=len(actual_suites)):
                raise ValueError('pair completion/ordered suite ownership differs')
    for span in intervals.values():
        if span['kind']!='suite-arm':continue
        cycles=[c for c in children.get(span['sample_id'],[]) if c['kind']=='cycle']
        if not cycles:raise ValueError('role cost has no actual case work')
        members=[row_by_cycle[c['sample_id']] for c in cycles]
        if digest(span['scope'])!=digest(role_scope(schedule[members[0]['slot']],scope).model_dump(mode='json')):
            raise ValueError('role cost scope differs from its actual cases')
        failed=any(r['errors'] or (r['cell'] in futile and futile[r['cell']]['at_slot']==r['slot']) for r in members)
        if span['outcome']!=('failed' if failed else 'complete'):raise ValueError('role cost hides failure or invents completion')
        # The comparison/checkpoint after each case is required work in its own
        # right. Completing seven cycles alone cannot prove a completed role.
        direct=children[span['sample_id']];offset=0
        for member in members:
            expected=member['cycle_sample_id']
            if (offset>=len(direct) or direct[offset]['sample_id']!=expected
                    or offset+1>=len(direct) or direct[offset+1]['kind']!='oracle'
                    or direct[offset+1]['outcome']!='complete'):
                raise ValueError('case comparison missing, reordered or unsuccessful')
            offset+=2
            if offset<len(direct) and direct[offset]['kind']=='publication':
                if direct[offset]['outcome']!='complete':raise ValueError('checkpoint publication failed')
                offset+=1
        if offset!=len(direct):raise ValueError('unowned work in role cost')
    if digest(raw['futile'])!=digest(futile):raise ValueError('futility claims differ from observed task outcomes')
    missing=[s for s in schedule if s['slot'] not in set(positions)]
    expected_missing=[dict(s,cause=futile[s['cell']]['cause'] if s['cell'] in futile else 'operational stop') for s in missing]
    if digest(raw['not_run'])!=digest(expected_missing):raise ValueError('not-run accounting differs')
    if raw['status']=='done':
        if len(admissions)!=96 or not all(a['permitted'] for a in admissions):raise ValueError('complete study needs all pair decisions')
        if any(s['cell'] not in futile or s['slot']<=futile[s['cell']]['at_slot'] for s in missing):raise ValueError('missing work without prior futility')
        expected_rests=[dict(after_slot=i,seconds=30) for i in range(27,1344,28)]
        if [{k:r[k] for k in ('after_slot','seconds')} for r in raw['rests']]!=expected_rests or not all(r['complete'] is True for r in raw['rests']):
            raise ValueError('full fixed rest schedule required')
    elif raw['status']!='stopped':raise ValueError('unknown study terminal status')
    _audit_schedule(raw,schedule,intervals,children,futile,root)
    done['rows']=rebuilt;done['futile']=futile;summary=analyze(done)
    if digest(summary)!=digest(raw['analysis']):raise ValueError('analysis differs from complete raw evidence')
    if raw['generation_allowed'] is not native or any(raw[k] is not False for k in ('automatic_admission','deployment_allowed')):
        raise ValueError('permission flags must remain false')
    if problems:
        raise ValueError('; '.join(problems))
    return dict(version=VERSION,evidence_valid=True,problems=[],study_complete=raw['status']=='done',
                simulated=not native,analysis=summary,costs=costs,actual_qualification=False,
                automatic_admission=False,deployment_allowed=False)
