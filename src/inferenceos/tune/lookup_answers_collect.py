"""Fixed Stage-B scheduler over an explicit engine protocol. No native factory."""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass
import re
from typing import Any, Callable, Protocol

from inferenceos.tune.lookup_answers import (VERSION, SOFT_TARGET_NS, plan, scope_problems, role_scope,
                                           request_body, native_answer, observe_problems, analyze)
from inferenceos.tune.lookup_corpus import build_corpus, judge, digest
from inferenceos.tune.lookup_fit import release_complete
from inferenceos.tune.lookup_role_cost import Ledger


class AnswerEngine(Protocol):
    def start(self, model: str, arm: str) -> None: ...
    def observe(self) -> dict[str,Any]: ...
    def tokenize(self, text: str, *, add_special: bool) -> list[int]: ...
    def answer(self, body: dict[str,Any], chunk: Callable[[dict[str,Any]],None]) -> None: ...
    def release(self, *, abort: bool) -> dict[str,Any]: ...


class EvidenceFailure(ValueError):
    pass


@dataclass(frozen=True)
class NativeExecution:
    """Run-local acknowledgement after frozen inputs and owner scope were verified."""
    run_id: str
    manifest_sha256: str
    scope_sha256: str
    origin_ns: int
    owner_authorized: bool

    def validate(self, run_id: str, scope: dict[str,Any]) -> None:
        from inferenceos.tune.lookup_answers import valid_digest
        if (self.run_id!=run_id or not valid_digest(self.manifest_sha256)
                or self.scope_sha256!=digest(scope) or self.owner_authorized is not True
                or type(self.origin_ns) is not int or self.origin_ns<0):
            raise EvidenceFailure('verified run-local native execution acknowledgement required')


def NO_CHECKPOINT(record: dict[str,Any]) -> None:
    """Explicit software-fixture sink; no publication cost is fabricated."""


def collect_answers(engine: AnswerEngine, scope: dict[str,Any], *, run_id: str,
                    clock: Callable[[],int], power: Callable[[],str], guard: Callable[[],None],
                    rest: Callable[[int],None], checkpoint: Callable[[dict[str,Any]],None],
                    simulated: bool, execution: NativeExecution | None=None,
                    boundary: Callable[[str],Any] | None=None) -> dict[str,Any]:
    """Never retries a load; futility closes one cell, operational errors close all.

    The native caller must supply verified run-local scope, bounded transport and
    owned publication. This function alone never constructs a native backend.
    """
    if scope_problems(scope):raise EvidenceFailure('; '.join(scope_problems(scope)))
    if not re.fullmatch('[a-zA-Z0-9_-]{1,100}',run_id):raise ValueError('bounded explicit run identity required')
    if type(simulated) is not bool:raise EvidenceFailure('explicit execution domain required')
    if not simulated:
        if execution is None:raise EvidenceFailure('live Stage-B requires separately authorized frozen inputs')
        execution.validate(run_id,scope)
    elif execution is not None:raise EvidenceFailure('native authorization cannot label a simulation')
    scope=deepcopy(scope);schedule=plan();cases={c.case_id:c for c in build_corpus().cases}
    ledger=Ledger(run_id,clock)
    record: dict[str,Any]=dict(version=VERSION,run_id=run_id,simulated=simulated,scope=scope,scope_sha256=digest(scope),
         plan_sha256=digest(schedule),status='running',rows=[],not_run=[],futile={},rests=[],pair_admissions=[],cost_intervals=[],
         generation_allowed=not simulated,automatic_admission=False,deployment_allowed=False,
         live_dispatch_implemented=not simulated,cost_clock_domain='simulated' if simulated else 'monotonic-ns',final_publication_cost_right_censored=True)
    if execution is not None:
        record['target_origin_ns']=execution.origin_ns
        record['execution']=dict(run_id=run_id,manifest_sha256=execution.manifest_sha256,
                                 scope_sha256=execution.scope_sha256,owner_authorized=True)
    record['unmeasured_operations']=[dict(kind=kind,state='right-censored') for kind in
                                     ('initial-scope-validation','final-analysis','final-publication')]
    pending: BaseException | None=None

    def remember(row: dict[str,Any],exc: BaseException) -> None:
        nonlocal pending
        row['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
        if pending is None or isinstance(exc,(KeyboardInterrupt,SystemExit)):
            if pending is not None:exc.__context__=pending
            pending=exc

    def publish() -> None:
        if checkpoint is NO_CHECKPOINT:
            return
        with ledger.measure('publication'):
            incremental=getattr(checkpoint,'incremental',False) is True
            record['cost_intervals']=ledger.rows if incremental else ledger.snapshot()
            checkpoint(record if incremental else deepcopy(record))

    def attempt(slot: dict[str,Any]) -> dict[str,Any]:
        nonlocal pending
        case=cases[slot['case_id']];rs=role_scope(slot,scope)
        row=dict(slot,load_id=f"{run_id}/B/{slot['slot']}",power=[],snapshots=[],prompt_ids=[],events=[],
                 request=request_body(case),errors=[],cleanup=None,verdict='incomplete',complete=False,cycle_ns=None)
        record['rows'].append(row);owed=False
        with ledger.measure('cycle',scope=rs,case_id=case.case_id) as cycle:
            row['cycle_sample_id']=cycle['sample_id']
            try:
                with ledger.measure('power'):
                    row['power'].append(power())
                    if row['power']!=['ac']:raise EvidenceFailure('AC required before attempted load')
                with ledger.measure('gate'):guard()
                owed=True
                with ledger.measure('start') as startup:engine.start(slot['model'],slot['arm'])
                if startup['end_ns']-startup['start_ns']>=120_000_000_000:raise EvidenceFailure('late start refused')
                with ledger.measure('ready-identity') as ready_span:
                    ready=deepcopy(engine.observe());row['snapshots'].append(dict(ready,phase='ready'))
                    issues=observe_problems(ready,slot,scope)
                    if issues:raise EvidenceFailure('; '.join(issues))
                    identity=(ready['image']['pid'],ready['image']['create_time_s'])
                    if any((s['image']['pid'],s['image']['create_time_s'])==identity for old in record['rows'][:-1] for s in old['snapshots']):
                        raise EvidenceFailure('process identity reused; no tokenization or answer')
                if ready_span['end_ns']-ready_span['start_ns']>=30_000_000_000:raise EvidenceFailure('late ready observation refused')
                with ledger.measure('tokenize') as token_span:
                    row['prompt_ids']=deepcopy(engine.tokenize(case.prompt,add_special=True))
                    expected=next(t['ids'] for t in scope['models'][slot['model']]['fit_tokens'] if t['case_id']==case.case_id and t['kind']=='prompt')
                    if digest(row['prompt_ids'])!=digest(expected):raise EvidenceFailure('prompt tokenization differs from accepted fit')
                if token_span['end_ns']-token_span['start_ns']>=30_000_000_000:raise EvidenceFailure('late tokenization refused')
                with ledger.measure('request') as request:
                    row['request_sample_id']=request['sample_id']
                    def receive(chunk: dict[str,Any]) -> None:
                        row['events'].append(dict(received_ns=clock(),data=deepcopy(chunk)))
                        if not isinstance(chunk,dict) or 'error' in chunk:raise EvidenceFailure('native stream error')
                    engine.answer(deepcopy(row['request']),receive)
                row['answer']=native_answer(row['events'],row['request'],case,slot['arm'],row['prompt_ids'],request['start_ns'],request['end_ns'])
                if not row['answer']['valid']:raise EvidenceFailure('; '.join(row['answer']['problems']))
                with ledger.measure('oracle'):
                    verdict=judge(case,row['answer']['text'],ending='eos',truncated=False)
                    row['judge']=verdict.model_dump(mode='json')
                    row['verdict']='right' if verdict.passed else 'wrong'
                with ledger.measure('release-identity') as last_span:
                    last=deepcopy(engine.observe());row['snapshots'].append(dict(last,phase='before_stop'))
                    issues=observe_problems(last,slot,scope)
                    if issues:raise EvidenceFailure('; '.join(issues))
                    if (last['image']['pid'],last['image']['create_time_s'])!=identity:raise EvidenceFailure('process changed before release')
                if last_span['end_ns']-last_span['start_ns']>=30_000_000_000:raise EvidenceFailure('late before-stop observation refused')
            except BaseException as exc:
                remember(row,exc)
            finally:
                if owed:
                    try:
                        with ledger.measure('cleanup'):
                            row['cleanup']=deepcopy(engine.release(abort=pending is not None))
                            if not release_complete(row['cleanup']):raise EvidenceFailure('process/credential release unconfirmed')
                    except BaseException as exc:remember(row,exc)
                try:
                    with ledger.measure('power'):
                        row['power'].append(power())
                        if row['power']!=['ac','ac']:raise EvidenceFailure('post-load AC not confirmed')
                except BaseException as exc:remember(row,exc)
            if pending is not None:raise pending
            row['complete']=True
        row['cycle_ns']=cycle['end_ns']-cycle['start_ns']
        return row

    try:
        with ledger.measure('job') as job:
            origin=execution.origin_ns if execution is not None else job['start_ns']
            if origin>job['start_ns']:raise EvidenceFailure('whole-job origin cannot follow collector entry')
            for first in range(0,len(schedule),14):
                group=schedule[first:first+14];cell=group[0]['cell']
                admission=clock()
                record['pair_admissions'].append(dict(first_slot=first,at_ns=admission,permitted=admission-origin<SOFT_TARGET_NS))
                if admission-origin>=SOFT_TARGET_NS:raise EvidenceFailure('soft target reached before paired role block')
                if cell not in record['futile']:
                    with ledger.measure('pair'):
                        for arm_first in (0,7):
                            arm_rows=group[arm_first:arm_first+7]
                            with ledger.measure('suite-arm',scope=role_scope(arm_rows[0],scope)) as suite:
                                for slot in arm_rows:
                                    row=attempt(slot)
                                    with ledger.measure('oracle'):
                                        paired=next((r for r in record['rows'] if r['block']==slot['block'] and r['cell']==cell and r['role']==slot['role']
                                                     and r['case_id']==slot['case_id'] and r['arm']!=slot['arm']),None)
                                        cause=None
                                        if row['verdict']=='wrong':
                                            cause=('task-unqualified control' if slot['arm']=='control' else
                                                   'candidate regression' if paired and paired['verdict']=='right' else 'lookup wrong; paired control unobserved')
                                        elif paired and (row['answer']['text'],row['answer']['token_ids'])!=(paired['answer']['text'],paired['answer']['token_ids']):
                                            cause='paired exact output preservation fails'
                                        if cause:record['futile'][cell]=dict(cause=cause,at_slot=slot['slot'])
                                    publish()
                                    if cell in record['futile']:suite['outcome']='failed';break
                            if cell in record['futile']:break
                if first%28==14:
                    with ledger.measure('rest') as interval:
                        item=dict(after_slot=first+13,seconds=30,sample_id=interval['sample_id'],complete=False)
                        record['rests'].append(item);rest(30);item['complete']=True
            record['status']='done'
    except BaseException as exc:
        if pending is None:pending=exc
        record['status']='stopped';record['stop']=dict(type=type(exc).__name__,message=str(exc))
    finally:
        done={r['slot'] for r in record['rows']}
        record['not_run']=[dict(s,cause=record['futile'][s['cell']]['cause'] if s['cell'] in record['futile'] else 'operational stop') for s in schedule if s['slot'] not in done]
        record['cost_intervals']=ledger.snapshot()
        try:
            with boundary('final-analysis') if boundary else nullcontext():
                record['analysis']=analyze(record)
        except BaseException as exc:
            record['analysis_problem']=dict(type=type(exc).__name__,message=str(exc))
            record['status']='stopped'
            record['stop']=dict(type=type(exc).__name__,message=str(exc),stage='final-analysis')
            record['analysis']=dict(analysis_complete=False,analysis_looks=0,actual_qualification=False,
                                    automatic_admission=False,deployment_allowed=False)
            if pending is None or isinstance(exc,(KeyboardInterrupt,SystemExit)):
                if pending is not None:exc.__context__=pending
                pending=exc
        try:
            if checkpoint is not NO_CHECKPOINT:
                with boundary('final-core-checkpoint') if boundary else nullcontext():
                    checkpoint(record if getattr(checkpoint,'incremental',False) is True else deepcopy(record))
        except BaseException as exc:
            record['publication_problem']=dict(type=type(exc).__name__,message=str(exc))
            if pending is None or isinstance(exc,(KeyboardInterrupt,SystemExit)):pending=exc
    if isinstance(pending,(KeyboardInterrupt,SystemExit)):raise pending
    return record
