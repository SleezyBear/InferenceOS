"""Role samples are measured intervals, not seven aliases of a case reading."""
from copy import deepcopy
import json
from typing import Any
import pytest

from inferenceos.tune.lookup_answers import plan,role_scope
from inferenceos.tune.lookup_answers_sim import fixture_scope,FakeClock
from inferenceos.tune.lookup_role_cost import Ledger,Interval,RoleScope,audit_intervals


def intervals() -> list[dict[str,Any]]:
    clock=FakeClock();ledger=Ledger('test',clock);scope=role_scope(plan()[0],fixture_scope())
    with ledger.measure('job'):
        with ledger.measure('suite-arm',scope=scope):
            for case_id,_ in scope.ordered_cases:
                with ledger.measure('cycle',scope=scope,case_id=case_id):
                    with ledger.measure('request'):clock.advance(10)
    return ledger.snapshot()


def test_whole_suite_sample_contains_seven_case_measurements_and_reports_gaps() -> None:
    raw=intervals();result=audit_intervals(raw)
    assert result['valid'] and result['inclusive_do_not_sum'] and not result['forecast_available']
    assert result['distributions']['suite-arm']['count']==1 and result['distributions']['cycle']['count']==7
    assert all(v>=0 for v in result['unassigned_ns'].values())
    parent=raw[1]
    assert parent['end_ns']-parent['start_ns']>sum(r['end_ns']-r['start_ns'] for r in raw if r['kind']=='cycle')


@pytest.mark.parametrize('fault',('duplicate','orphan','cycle','overlap','outside','negative','bool','float',
                                'missing_case','reordered_cases','wrong_scope','fake_whole_role','unknown_kind','second_root','running_as_complete'))
def test_corrupt_interval_evidence_is_refused(fault: str) -> None:
    raw=intervals()
    if fault=='duplicate':raw.append(deepcopy(raw[-1]))
    elif fault=='orphan':raw[2]['parent_id']='missing'
    elif fault=='cycle':raw[1]['parent_id']=raw[2]['sample_id']
    elif fault=='overlap':raw[4]['start_ns']=raw[2]['start_ns']
    elif fault=='outside':raw[2]['end_ns']=raw[0]['end_ns']+1
    elif fault=='negative':raw[2]['start_ns']=-1
    elif fault=='bool':raw[2]['start_ns']=True
    elif fault=='float':raw[2]['end_ns']=float(raw[2]['end_ns'])
    elif fault=='missing_case':raw[2]['case_id']='not-in-suite'
    elif fault=='reordered_cases':raw[2]['case_id']=raw[4]['case_id']
    elif fault=='wrong_scope':raw[2]['scope']['arm']='lookup'
    elif fault=='fake_whole_role':raw[1]['scope']['ordered_cases'][1]=deepcopy(raw[1]['scope']['ordered_cases'][0])
    elif fault=='unknown_kind':raw[3]['kind']='guessed-cost'
    elif fault=='second_root':raw[2]['parent_id']=None
    elif fault=='running_as_complete':raw[3]['outcome']='running'
    assert not audit_intervals(raw)['valid']


def test_same_exception_and_failed_cost_not_fabricated_as_completed() -> None:
    ledger=Ledger('x',FakeClock());boom=KeyboardInterrupt('original')
    with pytest.raises(KeyboardInterrupt) as caught:
        with ledger.measure('job'):
            with ledger.measure('request'):raise boom
    assert caught.value is boom
    result=audit_intervals(ledger.snapshot())
    assert result['valid'] and result['outcomes']=={'failed':2}
    assert result['distributions']['request']['count']==0


def test_partial_scope_is_not_a_completed_role_and_numeric_median_is_exact() -> None:
    clock=FakeClock();ledger=Ledger('x',clock);scope=role_scope(plan()[0],fixture_scope())
    with ledger.measure('job'):
        with ledger.measure('suite-arm',scope=scope) as row:
            with ledger.measure('cycle',scope=scope,case_id=scope.ordered_cases[0][0]):pass
            row['outcome']='failed'
    result=audit_intervals(ledger.snapshot());assert result['valid']
    assert result['distributions']['suite-arm']['count']==0
    median=result['distributions']['cycle']['median_ns']
    assert type(median['numerator']) is int and type(median['denominator']) is int


def test_integer_nanoseconds_beyond_float_precision_remain_exact() -> None:
    huge=9_007_199_254_740_993
    rows=[dict(sample_id='root',parent_id=None,kind='job',start_ns=0,end_ns=3*huge,outcome='complete',scope=None,case_id=None),
          dict(sample_id='a',parent_id='root',kind='request',start_ns=1,end_ns=huge+1,outcome='complete',scope=None,case_id=None),
          dict(sample_id='b',parent_id='root',kind='request',start_ns=huge+2,end_ns=2*huge+4,outcome='complete',scope=None,case_id=None)]
    result=audit_intervals(rows);assert result['valid']
    assert result['distributions']['request']['median_ns']==dict(numerator=2*huge+2,denominator=2)
