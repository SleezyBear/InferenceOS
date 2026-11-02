"""Offline native Stage-B envelope audit. Never constructs a backend."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from inferenceos.tune.lookup_answers_audit import _audit
from inferenceos.tune.lookup_answers_manifest import AnswerInputs
from inferenceos.tune.lookup_answer_publication import (VERSION,CALLER_VERSION,sha,replay_journal,caller_cost_problems)
from inferenceos.tune.lookup_corpus import digest
from inferenceos.tune.lookup_fit import preservation_problems,strict_json


def audit_native_study(record_bytes: bytes, closure_bytes: bytes, journal_bytes: bytes,
                       inputs: AnswerInputs) -> dict[str,Any]:
    """A complete native receipt is still not independent qualification/permission."""
    try:return _audit_native(record_bytes,closure_bytes,journal_bytes,inputs.revalidated())
    except (ValueError,TypeError,KeyError,AttributeError,IndexError,StopIteration,OverflowError) as exc:
        return dict(version=VERSION,evidence_valid=False,study_complete=False,simulated=False,
                    problems=[str(exc)],analysis=dict(cells={},analysis_looks=0,analysis_complete=False),
                    actual_qualification=False,automatic_admission=False,deployment_allowed=False,forecast_available=False)


def _audit_native(record_bytes: bytes,closure_bytes: bytes,journal_bytes: bytes,inputs: AnswerInputs) -> dict[str,Any]:
    record,receipt=strict_json(record_bytes),strict_json(closure_bytes)
    manifest_sha=sha(inputs._raw_manifest)
    if (record['version']!=VERSION or receipt['version']!=VERSION or record['status']!='collected-pending-closure'
            or receipt['status']!='done' or record['run_id']!=receipt['run_id']
            or record['manifest_sha256']!=manifest_sha or receipt['manifest_sha256']!=manifest_sha
            or receipt['record_sha256']!=sha(record_bytes) or receipt['journal_sha256']!=sha(journal_bytes)
            or type(receipt['journal_bytes']) is not int or receipt['journal_bytes']!=len(journal_bytes)):
        raise ValueError('record/closure/journal/run/frozen-input identity differs')
    if (record['simulated'] is not False or record['initial_power']!='ac'
            or record['final_cleanup'] is not None or record['error'] is not None):
        raise ValueError('native AC and clean final owner boundary required')
    if receipt.get('caller_cost_version')!=CALLER_VERSION or receipt.get('receipt_publication')!=dict(outcome='right-censored',included_in_observed_job=False):
        raise ValueError('observed caller boundary and censored receipt publication must remain explicit')
    if digest(record['provenance'])!=digest(inputs.provenance()):raise ValueError('provenance differs from original accepted A inputs')
    issues=preservation_problems(record['preservation'],inputs.derived_fit_manifest())
    if issues:raise ValueError('; '.join(issues))
    for obj in (record,receipt):
        if any(obj.get(k) is not False for k in ('actual_qualification','automatic_admission','deployment_allowed')):
            raise ValueError('development receipts cannot mint qualification or deployment permission')
    core=record['core']
    if core['run_id']!=record['run_id'] or digest(core['scope'])!=digest(inputs.derived_scope()):
        raise ValueError('core run or scope differs from frozen accepted observations')
    if core.get('execution')!=dict(run_id=core['run_id'],manifest_sha256=manifest_sha,
                                  scope_sha256=digest(core['scope']),owner_authorized=True):
        raise ValueError('run-local explicit owner acknowledgement is missing or mixed')
    rebuilt=replay_journal(journal_bytes,record['run_id'])
    if digest(rebuilt)!=digest(core):raise ValueError('final core differs from append-only checkpoints')
    caller=receipt['caller_intervals'];issues=caller_cost_problems(caller,record['run_id'],complete=True)
    if issues:raise ValueError('; '.join(issues))
    prefix=record['caller_intervals']
    if (len(caller)!=len(prefix)+1 or caller[-1]['kind']!='closure-validation'
            or prefix[-1]['kind']!='final-publication'):
        raise ValueError('caller closure changed the operation population')
    fields={'sample_id','parent_id','kind','start_ns','end_ns','outcome'}
    for index,(before,after) in enumerate(zip(prefix,caller)):
        if set(before)!=fields:raise ValueError('caller prefix lost explicit identity/boundaries')
        if before['end_ns'] is not None:
            if before!=after:raise ValueError('closed caller work changed after publication')
        elif (index not in (0,len(prefix)-1) or before['outcome']!='running'
                or any(before[k]!=after[k] for k in fields-{'end_ns','outcome'})):
            raise ValueError('caller identity/start changed after publication')
    if core['target_origin_ns']!=caller[0]['start_ns']:raise ValueError('soft target excludes initial caller work')
    execution=next(c for c in caller if c['kind']=='execution')
    analysis=next(c for c in caller if c['kind']=='final-analysis')
    checkpoint=next(c for c in caller if c['kind']=='final-core-checkpoint')
    body=core['cost_intervals'][0]
    if not execution['start_ns']<=body['start_ns']<body['end_ns']<=analysis['start_ns']<analysis['end_ns']<=checkpoint['start_ns']<checkpoint['end_ns']<=execution['end_ns']:
        raise ValueError('core/analysis/checkpoint chronology differs from measured caller execution')
    result=_audit(core,native=True)
    if not result['evidence_valid'] or not result['study_complete']:raise ValueError('native core incomplete')
    result=deepcopy(result)
    result.update(version=VERSION,manifest_sha256=manifest_sha,
                  caller_costs=dict(version=CALLER_VERSION,valid=True,observed_job_ns=caller[0]['end_ns']-caller[0]['start_ns'],
                                    receipt_publication='right-censored',forecast_available=False),
                  actual_qualification=False,automatic_admission=False,deployment_allowed=False,forecast_available=False)
    return result
