"""Evaluation-only owned journal and separately closed caller costs.

Checkpoints append deltas, not 1,344 copies of an ever-growing result. The final
record is immutable once published; a separate receipt measures that publication.
The receipt's own publication is explicitly outside the observed boundary, never
invented as a zero-cost operation or a forecasting allowance.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterator

from inferenceos.storage import atomic_write_text
from inferenceos.tune.lookup_corpus import digest
from inferenceos.tune.lookup_fit import FitRefused, strict_json

VERSION = 'lookup-answer-study-1'
JOURNAL_VERSION = 'lookup-answer-journal-1'
CALLER_VERSION = 'lookup-answer-caller-costs-1'
MAX_RECORD_BYTES=512*1024*1024
MAX_JOURNAL_BYTES=1024*1024*1024
MIN_FREE_BYTES=64*1024*1024
CALLER_KINDS = ('whole-job','initial-output','initial-validation','engine-construction',
                'execution','final-analysis','final-core-checkpoint','final-cleanup',
                'preservation','final-publication','closure-validation')


def encoded(value: Any) -> bytes:
    return (json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _same_json(before: Any, after: Any) -> bool:
    """Exact JSON-value equality with a fast immutable-leaf identity path.

    Writer snapshots are detached from the producer's mutable containers. Comparing
    their contents avoids serializing the complete answer history at every append;
    the type check still distinguishes observations such as 1, 1.0 and True.
    """
    if type(before) is not type(after):return False
    if isinstance(before,dict):
        return before.keys()==after.keys() and all(_same_json(v,after[k]) for k,v in before.items())
    if isinstance(before,list):
        return len(before)==len(after) and all(_same_json(a,b) for a,b in zip(before,after))
    if isinstance(before,float):return isinstance(after,float) and before.hex()==after.hex()
    return before is after or before==after


def _prefix(before: Any, after: Any) -> bool:
    return (isinstance(before,list) and isinstance(after,list) and len(after)>=len(before)
            and all(_same_json(a,b) for a,b in zip(before,after)))


def _row_problem(before: Any, after: Any) -> str | None:
    """Only an unfinished row may gain new observations, never replace them."""
    if not isinstance(before,dict) or not isinstance(after,dict):return 'checkpoint rows must be objects'
    if _same_json(before,after):return None
    if before.get('complete') is not False:return 'completed checkpoint row changed'
    if not before.keys()<=after.keys():return 'checkpoint row discarded existing evidence'
    for key,value in before.items():
        newer=after[key]
        if _same_json(value,newer):continue
        if key in ('power','snapshots','events','errors') and _prefix(value,newer):continue
        # Tokenization supplies one complete array, not an evolving stream.
        if key=='prompt_ids' and value==[] and isinstance(newer,list):continue
        if key in ('cleanup','cycle_ns') and value is None:continue
        if key=='complete' and value is False and newer is True:continue
        if key=='verdict' and value=='incomplete' and newer in ('right','wrong'):continue
        return f'checkpoint row changed existing {key} evidence'
    # These derived fields are supplied at distinct operation boundaries. Unknown
    # late fields cannot silently change the interpretation of recorded evidence.
    if not (after.keys()-before.keys())<={'cycle_sample_id','request_sample_id','answer','judge'}:
        return 'checkpoint row added an unknown late field'
    return None


def _rows_problem(before: list[Any], after: list[Any], *, unchanged_prefix: int=0) -> str | None:
    if len(after)<len(before):return 'checkpoint population regressed'
    # Replay constructs this prefix from its own private decoded history; an
    # incoming packet cannot supply or mutate it. The live writer uses zero and
    # still checks every detached observation against the producer's state.
    for index in range(unchanged_prefix,len(before)):
        old=before[index]
        issue=_row_problem(old,after[index])
        if issue:return issue
        if index<len(before)-1 and not _same_json(old,after[index]):
            return 'checkpoint rewrites an earlier unfinished row'
    if (before and len(after)>len(before) and before[-1].get('complete') is False
            and after[len(before)-1].get('complete') is not True):
        return 'checkpoint dispatch followed an unfinished row'
    return None


def _meta_problem(before: dict[str,Any], after: dict[str,Any], open_rest_ids: set[str]) -> str | None:
    if not before.keys()<=after.keys():return 'checkpoint discarded existing metadata evidence'
    if not (after.keys()-before.keys())<={'analysis','stop','analysis_problem','publication_problem'}:
        return 'checkpoint added an unknown late metadata field'
    for key,value in before.items():
        newer=after[key]
        if _same_json(value,newer):continue
        if before.get('status')!='running':return f'terminal checkpoint changed existing {key} metadata evidence'
        if key=='status' and value=='running' and newer in ('done','stopped'):continue
        if key=='pair_admissions' and _prefix(value,newer):continue
        if key=='not_run' and value==[] and isinstance(newer,list):continue
        if (key=='futile' and isinstance(value,dict) and isinstance(newer,dict)
                and value.keys()<=newer.keys() and all(_same_json(v,newer[k]) for k,v in value.items())):continue
        if key=='rests' and isinstance(value,list) and isinstance(newer,list) and len(newer)>=len(value):
            valid=True
            for old,item in zip(value,newer):
                if _same_json(old,item):continue
                if (not isinstance(old,dict) or not isinstance(item,dict) or old.keys()!=item.keys()
                        or old.get('complete') is not False or item.get('complete') is not True
                        or old.get('sample_id') not in open_rest_ids
                        or any(not _same_json(v,item[k]) for k,v in old.items() if k!='complete')):
                    valid=False;break
            if valid:continue
        return f'checkpoint changed existing {key} metadata evidence'
    return None


def _cost_problem(before: Any, after: Any) -> str | None:
    if not isinstance(before,dict) or not isinstance(after,dict):return 'cost intervals must be objects'
    if _same_json(before,after):return None
    if before.keys()!=after.keys() or before.get('end_ns') is not None:return 'closed or identified cost rewritten'
    if any(not _same_json(value,after[key]) for key,value in before.items() if key not in ('end_ns','outcome')):
        return 'identified cost rewritten'
    if after.get('end_ns') is not None and type(after['end_ns']) is not int:return 'integer cost end required'
    if before.get('outcome')!='running' and not _same_json(before.get('outcome'),after.get('outcome')):
        return 'observed cost outcome rewritten'
    if after.get('outcome') not in ('running','complete','failed'):return 'known cost outcome required'
    return None


def bounded_study_bytes(path: Path, *, maximum_bytes: int=MAX_RECORD_BYTES) -> bytes:
    """Large native evidence is bounded separately from small Stage-A JSON inputs."""
    if type(maximum_bytes) is not int or not 1<=maximum_bytes<=MAX_JOURNAL_BYTES:
        raise ValueError('bounded study evidence size required')
    if path.stat().st_size>maximum_bytes:raise FitRefused('study evidence exceeds size limit')
    with path.open('rb') as stream:raw=stream.read(maximum_bytes+1)
    if len(raw)>maximum_bytes:raise FitRefused('study evidence grew past size limit')
    return raw


class CallerLedger:
    def __init__(self, run_id: str, clock: Callable[[],int]) -> None:
        self.run_id,self.clock=run_id,clock
        self.rows: list[dict[str,Any]]=[]
        self.stack: list[str]=[]

    @contextmanager
    def measure(self, kind: str) -> Iterator[dict[str,Any]]:
        if kind not in CALLER_KINDS:raise ValueError('known caller boundary required')
        row: dict[str,Any]=dict(sample_id=f'{self.run_id}/caller/{len(self.rows)}',parent_id=self.stack[-1] if self.stack else None,
                 kind=kind,start_ns=self.clock(),end_ns=None,outcome='running')
        self.rows.append(row);self.stack.append(row['sample_id'])
        try:yield row
        except BaseException:
            row['outcome']='failed';raise
        else:
            if row['outcome']=='running':row['outcome']='complete'
        finally:
            row['end_ns']=self.clock();self.stack.pop()

    def snapshot(self) -> list[dict[str,Any]]:
        return deepcopy(self.rows)


class OwnedAnswerOutput:
    """One synchronous append owner. Existing outputs/receipts never resume."""
    incremental=True

    def __init__(self, path: Path, run_id: str) -> None:
        self.path=path
        self.journal_path=path.with_name(path.name+'.journal')
        self.closure_path=path.with_name(path.name+'.closure.json')
        self.run_id=run_id;self.closed=False;self.sequence=0;self.previous='0'*64
        self.row_count=0;self.cost_count=0;self.open_costs: set[int]=set()
        self._row_snapshots: list[Any]=[]
        self._cost_snapshots: list[Any]=[]
        self._meta_snapshot: dict[str,Any] | None=None
        self._open_rest_ids: set[str]=set()
        self.scope_sha: str | None=None
        self._last_core: dict[str,Any] | None=None
        self.prepared: dict[str,Any] | None=None
        self.expected=encoded(dict(version=VERSION,run_id=run_id,status='preparing',
                                   journal=self.journal_path.name,qualification=False))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if any(p.exists() or p.is_symlink() for p in (self.path,self.journal_path,self.closure_path)):
            raise FitRefused('existing record/journal/closure refuses a new run')
        if shutil.disk_usage(self.path.parent).free<MIN_FREE_BYTES:raise FitRefused('insufficient output-disk margin')
        # Do not remove an already created owned file if a later reservation fails.
        # A diagnostic reservation is not evidence and cannot be resumed.
        with self.path.open('xb') as stream:stream.write(self.expected)
        self.stream=self.journal_path.open('xb+')
        self._signature=self._journal_signature()

    def _journal_signature(self) -> tuple[int,int,int,int]:
        st=self.journal_path.stat();fd=os.fstat(self.stream.fileno())
        if (st.st_dev,st.st_ino)!=(fd.st_dev,fd.st_ino):raise OSError('owned journal replaced')
        return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns

    def verify(self) -> None:
        if (self.closed or self.path.is_symlink() or self.journal_path.is_symlink()
                or self.path.stat().st_size!=len(self.expected)
                or bounded_study_bytes(self.path,maximum_bytes=max(1,len(self.expected)))!=self.expected or self._journal_signature()!=self._signature
                or self.closure_path.exists() or self.closure_path.is_symlink()):
            raise OSError('owned result/journal changed; no subsequent dispatch')
        if shutil.disk_usage(self.path.parent).free<MIN_FREE_BYTES:raise OSError('output-disk margin exhausted; stop dispatch')

    def __call__(self, record: dict[str,Any]) -> None:
        if record['run_id']!=self.run_id:raise OSError('mixed checkpoint run')
        # This reference is diagnostics owned by the synchronous caller, not a
        # durable receipt. Retain it even if a write/interrupt prevents a journal
        # commit; later mutations in the stopped core complete its failure report.
        self._last_core=record
        self.verify()
        scope_sha=digest(record['scope'])
        if self.scope_sha is not None and self.scope_sha!=scope_sha:raise OSError('checkpoint scope changed')
        rows=record['rows'];costs=record['cost_intervals']
        if len(rows)<self.row_count or len(costs)<self.cost_count:raise OSError('checkpoint population regressed')
        issue=_rows_problem(self._row_snapshots,rows)
        if issue:raise OSError(issue)
        for index,old_cost in enumerate(self._cost_snapshots):
            issue=_cost_problem(old_cost,costs[index])
            if issue:raise OSError(issue)
        meta={k:v for k,v in record.items() if k not in ('rows','cost_intervals','scope')}
        if self._meta_snapshot is not None:
            issue=_meta_problem(self._meta_snapshot,meta,self._open_rest_ids)
            if issue:raise OSError(issue)
        indices=sorted(self.open_costs | set(range(self.cost_count,len(costs))))
        row_start=self.row_count
        if self.row_count and not _same_json(rows[self.row_count-1],self._row_snapshots[-1]):row_start=self.row_count-1
        # Commit detached observations, not producer references that can mutate
        # during/after IO and make the next comparison forget what was written.
        row_delta=deepcopy(rows[row_start:]);meta=deepcopy(meta)
        cost_delta=[dict(index=i,value=deepcopy(costs[i])) for i in indices]
        event=dict(version=JOURNAL_VERSION,run_id=self.run_id,
                   meta=meta,rows=row_delta,row_start=row_start,
                   costs=cost_delta)
        if self.scope_sha is None:event['scope']=record['scope']
        packet=dict(sequence=self.sequence,previous_sha256=self.previous,event=event)
        packet_raw=encoded(packet)
        if self._signature[2]+len(packet_raw)>MAX_JOURNAL_BYTES:raise OSError('checkpoint journal quota exhausted')
        self.stream.write(packet_raw);self.stream.flush();os.fsync(self.stream.fileno())
        self.previous=sha(packet_raw);self.sequence+=1;self.scope_sha=scope_sha
        self._row_snapshots[row_start:]=row_delta;self._meta_snapshot=meta
        for update in cost_delta:
            if update['index']==len(self._cost_snapshots):self._cost_snapshots.append(update['value'])
            else:self._cost_snapshots[update['index']]=update['value']
        self.row_count=len(self._row_snapshots);self.cost_count=len(self._cost_snapshots)
        self.open_costs={i for i in indices if self._cost_snapshots[i]['end_ns'] is None}
        self._open_rest_ids={self._cost_snapshots[i]['sample_id'] for i in self.open_costs if self._cost_snapshots[i].get('kind')=='rest'}
        self._signature=self._journal_signature()

    def publish(self, envelope: dict[str,Any]) -> None:
        self.verify()
        if envelope.get('run_id')!=self.run_id:raise OSError('foreign final record')
        raw=encoded(envelope)
        if len(raw)>MAX_RECORD_BYTES or shutil.disk_usage(self.path.parent).free<len(raw)+MIN_FREE_BYTES:
            raise OSError('final result exceeds output capacity/disk margin')
        atomic_write_text(self.path,raw.decode('utf-8'))
        # atomic_write_text uses the platform's text newline policy.
        actual=bounded_study_bytes(self.path,maximum_bytes=min(MAX_RECORD_BYTES,len(raw)+1))
        if actual.decode('utf-8').replace('\r\n','\n')!=raw.decode('utf-8'):
            raise OSError('owned result changed during publication')
        self.expected=actual
        self.verify()

    def prepare_closure(self) -> None:
        """Heavy replay/hashing is measured before closing the observed job span."""
        self.verify()
        self.stream.flush();os.fsync(self.stream.fileno())
        self.stream.seek(0);journal=self.stream.read()
        rebuilt=replay_journal(journal,self.run_id)
        raw=bounded_study_bytes(self.path);envelope=strict_json(raw)
        if 'core' in envelope and digest(rebuilt)!=digest(envelope['core']):
            raise OSError('checkpoint journal and final core differ')
        stat=self.path.stat()
        self.prepared=dict(record_sha256=sha(raw),journal_sha256=sha(journal),journal_bytes=len(journal),
                           record_signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns),
                           journal_signature=self._signature)

    def seal(self, *, status: str, caller_intervals: list[dict[str,Any]], manifest_sha256: str) -> dict[str,Any]:
        if self.prepared is None or self.closed:raise OSError('measured closure validation required')
        stat=self.path.stat()
        if ((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)!=self.prepared['record_signature']
                or self._journal_signature()!=self.prepared['journal_signature']
                or self.path.is_symlink() or self.journal_path.is_symlink()):
            raise OSError('owned output changed after closure validation')
        receipt=dict(version=VERSION,run_id=self.run_id,status=status,manifest_sha256=manifest_sha256,
                     record_sha256=self.prepared['record_sha256'],journal_sha256=self.prepared['journal_sha256'],journal_bytes=self.prepared['journal_bytes'],
                     caller_cost_version=CALLER_VERSION,caller_intervals=caller_intervals,
                     receipt_publication=dict(outcome='right-censored',included_in_observed_job=False),
                     actual_qualification=False,automatic_admission=False,deployment_allowed=False)
        with self.closure_path.open('xb') as stream:
            stream.write(encoded(receipt));stream.flush();os.fsync(stream.fileno())
        self.stream.close();self.closed=True
        return receipt

    def close(self) -> None:
        if not self.closed:
            self.stream.close();self.closed=True

    def latest(self) -> dict[str,Any] | None:
        self.verify()
        if not self.sequence:return None
        self.stream.flush();self.stream.seek(0)
        return replay_journal(self.stream.read(),self.run_id)

    def memory_diagnostic(self) -> dict[str,Any] | None:
        return deepcopy(self._last_core)


def replay_journal(raw: bytes, run_id: str) -> dict[str,Any]:
    """A partial journal can recover diagnostics, never resume or qualify a study."""
    previous='0'*64;result: dict[str,Any]={};rows: list[Any]=[];costs: list[Any]=[];scope=None
    lines=raw.splitlines(keepends=True)
    for number,line in enumerate(lines):
        if not line.endswith(b'\n'):raise FitRefused('incomplete journal tail')
        value=strict_json(line)
        if set(value)!= {'sequence','previous_sha256','event'} or type(value['sequence']) is not int or value['sequence']!=number or value['previous_sha256']!=previous:
            raise FitRefused('broken journal sequence/hash chain')
        event=value['event']
        if event['version']!=JOURNAL_VERSION or event['run_id']!=run_id:raise FitRefused('foreign checkpoint')
        if 'scope' in event:
            if scope is not None:raise FitRefused('duplicate journal scope')
            scope=event['scope']
        start=event['row_start']
        if type(start) is not int or start not in (len(rows),max(0,len(rows)-1)):raise FitRefused('checkpoint row continuity differs')
        replacement=event['rows']
        if not isinstance(replacement,list):raise FitRefused('invalid checkpoint rows')
        if start<len(rows) and not replacement:raise FitRefused('checkpoint discarded its last row')
        candidate=rows[:start]+replacement
        issue=_rows_problem(rows,candidate,unchanged_prefix=start)
        if issue:raise FitRefused(issue)
        meta=event['meta']
        if not isinstance(meta,dict):raise FitRefused('invalid checkpoint metadata')
        if result:
            issue=_meta_problem(result,meta,{r['sample_id'] for r in costs if r.get('kind')=='rest' and r.get('end_ns') is None})
            if issue:raise FitRefused(issue)
        rows=candidate
        seen=set()
        for update in event['costs']:
            index=update['index'];interval=update['value']
            if type(index) is not int or index<0 or index>len(costs) or index in seen:raise FitRefused('invalid cost delta')
            seen.add(index)
            if index==len(costs):costs.append(interval)
            else:
                old=costs[index]
                issue=_cost_problem(old,interval)
                if issue:raise FitRefused(issue)
                costs[index]=interval
        result=meta;previous=sha(line)
    if scope is None:raise FitRefused('journal has no complete checkpoint')
    return dict(result,scope=scope,rows=rows,cost_intervals=costs)


def caller_cost_problems(raw: Any, run_id: str, *, complete: bool) -> list[str]:
    try:
        if not isinstance(raw,list) or not raw:raise ValueError('caller cost population required')
        by_id={r['sample_id']:r for r in raw}
        if len(by_id)!=len(raw) or list(by_id)!=[f'{run_id}/caller/{i}' for i in range(len(raw))]:raise ValueError('caller IDs differ')
        root=raw[0]
        if root['kind']!='whole-job' or root['parent_id'] is not None:raise ValueError('one caller root required')
        children: dict[str,list[Any]]={}
        for row in raw:
            if (set(row)!= {'sample_id','parent_id','kind','start_ns','end_ns','outcome'} or row['kind'] not in CALLER_KINDS
                    or type(row['start_ns']) is not int or type(row['end_ns']) is not int
                    or not 0<=row['start_ns']<row['end_ns'] or row['outcome'] not in ('complete','failed')):
                raise ValueError('complete integer caller boundaries required')
            if complete and row['outcome']!='complete':raise ValueError('failed caller work cannot support completion')
            if row is root:continue
            parent=by_id.get(row['parent_id'])
            if (parent is None or parent['start_ns']>row['start_ns'] or parent['end_ns']<row['end_ns']
                    or list(by_id).index(parent['sample_id'])>=list(by_id).index(row['sample_id'])):
                raise ValueError('caller containment/order differs')
            children.setdefault(row['parent_id'],[]).append(row)
        for group in children.values():
            if any(a['end_ns']>b['start_ns'] for a,b in zip(group,group[1:])):raise ValueError('overlapping caller siblings')
        if any(row['kind'] not in ('whole-job','execution') and children.get(row['sample_id']) for row in raw):
            raise ValueError('caller operation leaves cannot hide nested work')
        if len({r['kind'] for r in raw})!=len(raw):raise ValueError('caller operation kinds cannot repeat')
        if complete:
            if [r['kind'] for r in children.get(root['sample_id'],[])]!=['initial-output','initial-validation','engine-construction','execution','final-cleanup','preservation','final-publication','closure-validation']:
                raise ValueError('missing/reordered caller work')
            execution=next(r for r in raw if r['kind']=='execution')
            if [r['kind'] for r in children.get(execution['sample_id'],[])]!=['final-analysis','final-core-checkpoint']:
                raise ValueError('missing analysis/publication boundary')
        return []
    except (ValueError,TypeError,KeyError,AttributeError) as exc:return [str(exc)]
