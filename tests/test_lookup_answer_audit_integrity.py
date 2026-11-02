"""Operation outcomes and chronological ownership are evidence, not annotations."""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

import pytest

from inferenceos.tune.lookup_answers import SOFT_TARGET_NS, analyze, native_answer, request_body
from inferenceos.tune.lookup_answers_audit import audit_answers
from inferenceos.tune.lookup_answers_collect import NO_CHECKPOINT, collect_answers
from inferenceos.tune.lookup_answers_sim import AnswerSimulator, FakeClock, fixture_scope, simulate_answers
from inferenceos.tune.lookup_role_cost import audit_intervals
from inferenceos.tune.lookup_corpus import build_corpus

ROOT = Path(__file__).resolve().parents[1]


def futile_record(*, publication: bool = False, short_rests: bool = False) -> dict[str, Any]:
    clock = FakeClock()
    engine = AnswerSimulator(clock)
    answer = engine.answer

    def wrong(body: Any, chunk: Any) -> None:
        answer(body, lambda data: chunk(dict(data, content='wrong') if data.get('content') else data))

    engine.answer = wrong  # type: ignore[method-assign]
    calls = 0

    def rest(seconds: int) -> None:
        nonlocal calls
        calls += 1
        if not short_rests or calls == 1:
            clock.advance(seconds * 1_000_000_000)

    return collect_answers(engine, fixture_scope(), run_id='audit_integrity', clock=clock,
                           power=lambda: 'ac', guard=lambda: None, rest=rest,
                           checkpoint=(lambda _: None) if publication else NO_CHECKPOINT, simulated=True)


@pytest.fixture(scope='module')
def futile() -> dict[str, Any]:
    return futile_record()


@pytest.fixture(scope='module')
def published() -> dict[str, Any]:
    return futile_record(publication=True)


@pytest.fixture(scope='module')
def full() -> dict[str, Any]:
    with patch('subprocess.Popen', side_effect=AssertionError('no native process')):
        return simulate_answers()


def assert_refused(raw: dict[str, Any]) -> dict[str, Any]:
    result = audit_answers(raw)
    assert result['evidence_valid'] is False, result
    assert result['study_complete'] is False
    assert result['analysis']['analysis_looks'] == 0
    assert not any(c.get('development_criteria_pass') for c in result['analysis'].get('cells', {}).values())
    assert result['actual_qualification'] is result['automatic_admission'] is result['deployment_allowed'] is False
    return result


def test_genuine_futility_and_optional_measured_publication_remain_acceptable(
    futile: dict[str, Any], published: dict[str, Any]
) -> None:
    for raw in (futile, published):
        result = audit_answers(raw)
        assert result['evidence_valid'] and result['study_complete'], result
        assert len(raw['rows']) == 6 and len(raw['futile']) == 6
        assert len(raw['rests']) == 48 and len({r['sample_id'] for r in raw['rests']}) == 48
        assert all(s['outcome'] == 'failed' for s in raw['cost_intervals'] if s['kind'] == 'suite-arm')
        assert result['costs']['distributions']['suite-arm']['count'] == 0


@pytest.mark.parametrize('kind', ('job', 'pair', 'cycle', 'gate', 'start', 'ready-identity', 'tokenize',
                                 'request', 'release-identity', 'cleanup', 'power', 'rest', 'publication',
                                 'case-oracle', 'pair-oracle'))
def test_every_failed_required_operation_refuses_positive_completion(published: dict[str, Any], kind: str) -> None:
    raw = deepcopy(published)
    spans = {s['sample_id']: s for s in raw['cost_intervals']}
    if kind.endswith('-oracle'):
        parent_kind = 'cycle' if kind == 'case-oracle' else 'suite-arm'
        span = next(s for s in spans.values() if s['kind'] == 'oracle' and spans[s['parent_id']]['kind'] == parent_kind)
    else:
        span = next(s for s in spans.values() if s['kind'] == kind)
    span['outcome'] = 'failed'
    assert_refused(raw)


def test_full_positive_run_cannot_keep_any_pass_after_failed_pair_comparison(full: dict[str, Any]) -> None:
    baseline = audit_answers(full)
    assert baseline['evidence_valid'] and sum(c['development_criteria_pass'] for c in baseline['analysis']['cells'].values()) == 6
    raw = deepcopy(full)
    spans = {s['sample_id']: s for s in raw['cost_intervals']}
    next(s for s in spans.values() if s['kind'] == 'oracle' and spans[s['parent_id']]['kind'] == 'suite-arm')['outcome'] = 'failed'
    assert not audit_intervals(raw['cost_intervals'])['valid']
    assert_refused(raw)


@pytest.mark.parametrize('field', ('stop', 'analysis_problem', 'publication_problem'))
@pytest.mark.parametrize('problem', (dict(type='RuntimeError', message='observed failure'), {}, None, False))
def test_retained_failure_metadata_cannot_be_ignored_when_status_says_done(
        futile: dict[str, Any], field: str, problem: Any) -> None:
    raw=deepcopy(futile)
    raw[field]=deepcopy(problem)
    assert raw['status']=='done'
    assert_refused(raw)


@pytest.mark.parametrize('field', ('stop', 'analysis_problem'))
def test_retained_failure_metadata_suppresses_all_otherwise_positive_cells(
        full: dict[str, Any], field: str) -> None:
    raw=deepcopy(full)
    assert all(c['development_criteria_pass'] for c in raw['analysis']['cells'].values())
    raw[field]=dict(type='RuntimeError', message='observed failure remains in the record')
    assert_refused(raw)


@pytest.mark.parametrize('kind', ('rest', 'pair-oracle', 'publication'))
def test_operational_failure_cannot_be_relabelled_as_task_futility(published: dict[str, Any], kind: str) -> None:
    raw = deepcopy(published)
    spans = {s['sample_id']: s for s in raw['cost_intervals']}
    if kind == 'pair-oracle':
        span = next(s for s in spans.values() if s['kind'] == 'oracle' and spans[s['parent_id']]['kind'] == 'suite-arm')
    else:
        span = next(s for s in spans.values() if s['kind'] == kind)
    span['outcome'] = 'failed'
    raw['status'] = 'stopped'
    raw['cost_intervals'][0]['outcome'] = 'failed'
    raw['analysis'] = analyze(raw)
    assert_refused(raw)


@pytest.mark.parametrize('fault', ('all_alias', 'swap', 'missing', 'extra', 'wrong_position', 'bool_position',
                                  'bool_seconds', 'complete_integer', 'foreign_owner', 'wrong_kind'))
def test_rest_receipts_are_unique_complete_and_owned_in_measured_order(futile: dict[str, Any], fault: str) -> None:
    raw = deepcopy(futile)
    if fault == 'all_alias':
        for item in raw['rests']:
            item['sample_id'] = raw['rests'][0]['sample_id']
    elif fault == 'swap':
        raw['rests'][0]['sample_id'], raw['rests'][1]['sample_id'] = raw['rests'][1]['sample_id'], raw['rests'][0]['sample_id']
    elif fault == 'missing': raw['rests'].pop(5)
    elif fault == 'extra': raw['rests'].append(deepcopy(raw['rests'][-1]))
    elif fault == 'wrong_position': raw['rests'][0]['after_slot'] = 55
    elif fault == 'bool_position': raw['rests'][0]['after_slot'] = True
    elif fault == 'bool_seconds': raw['rests'][0]['seconds'] = True
    elif fault == 'complete_integer': raw['rests'][0]['complete'] = 1
    elif fault == 'foreign_owner':
        rest = next(s for s in raw['cost_intervals'] if s['kind'] == 'rest')
        rest['parent_id'] = raw['cost_intervals'][1]['sample_id']
    elif fault == 'wrong_kind': next(s for s in raw['cost_intervals'] if s['kind'] == 'rest')['kind'] = 'pair'
    assert_refused(raw)


def test_one_actual_rest_cannot_serve_48_schedule_entries() -> None:
    raw = futile_record(short_rests=True)
    assert raw['cost_intervals'][0]['end_ns'] - raw['cost_intervals'][0]['start_ns'] < 32_000_000_000
    assert_refused(raw)
    for item in raw['rests']:
        item['sample_id'] = raw['rests'][0]['sample_id']
    assert_refused(raw)


@pytest.mark.parametrize('where', ('after_pair', 'after_rest', 'skipped_cell', 'all_stale'))
def test_admissions_must_follow_the_preceding_work_even_when_cells_are_futile(futile: dict[str, Any], where: str) -> None:
    raw = deepcopy(futile)
    index = {'after_pair': 1, 'after_rest': 2, 'skipped_cell': 12, 'all_stale': 0}[where]
    if where == 'all_stale':
        at = raw['pair_admissions'][0]['at_ns']
        for item in raw['pair_admissions']: item['at_ns'] = at
    else:
        raw['pair_admissions'][index]['at_ns'] = raw['pair_admissions'][index - 1]['at_ns']
    assert_refused(raw)


def test_stale_admissions_cannot_authorize_work_after_12_hour_target(futile: dict[str, Any]) -> None:
    raw = deepcopy(futile)
    anchor = next(s['end_ns'] for s in raw['cost_intervals'] if s['kind'] == 'rest')
    offset = SOFT_TARGET_NS + 1_000_000_000
    for span in raw['cost_intervals']:
        for key in ('start_ns', 'end_ns'):
            if span[key] >= anchor: span[key] += offset
    for row in raw['rows']:
        for event in row['events']:
            if event['received_ns'] >= anchor: event['received_ns'] += offset
    assert raw['pair_admissions'][-1]['at_ns'] - raw['cost_intervals'][0]['start_ns'] < SOFT_TARGET_NS
    assert_refused(raw)


@pytest.mark.parametrize('fault', ('missing_comparison', 'failed_suite_as_complete', 'unowned_operation'))
def test_role_cost_requires_the_comparison_and_owns_each_operation(published: dict[str, Any], fault: str) -> None:
    raw = deepcopy(published)
    spans = {s['sample_id']: s for s in raw['cost_intervals']}
    if fault == 'missing_comparison':
        next(s for s in spans.values() if s['kind'] == 'oracle' and spans[s['parent_id']]['kind'] == 'suite-arm')['kind'] = 'publication'
    elif fault == 'failed_suite_as_complete':
        next(s for s in spans.values() if s['kind'] == 'suite-arm')['outcome'] = 'complete'
    else:
        next(s for s in spans.values() if s['kind'] == 'publication')['kind'] = 'power'
    assert_refused(raw)


@pytest.mark.parametrize('origin', (True, -1, 1.0, '0', None))
def test_target_origin_is_an_exact_nonnegative_clock_value(futile: dict[str, Any], origin: Any) -> None:
    raw = deepcopy(futile)
    raw['target_origin_ns'] = origin
    assert_refused(raw)


def test_valid_outer_target_origin_and_refused_next_pair_remain_auditable() -> None:
    clock = FakeClock()
    calls = 0

    def now() -> int:
        nonlocal calls
        calls += 1
        if calls == 2: clock.advance(SOFT_TARGET_NS - 1_000_000_000)
        return clock()

    engine = AnswerSimulator(clock)
    raw = collect_answers(engine, fixture_scope(), run_id='target_order', clock=now,
                          power=lambda: 'ac', guard=lambda: None,
                          rest=lambda s: clock.advance(s * 1_000_000_000),
                          checkpoint=NO_CHECKPOINT, simulated=True)
    assert raw['status'] == 'stopped' and len(raw['rows']) == 14
    assert len(raw['pair_admissions']) == 2 and raw['pair_admissions'][-1]['permitted'] is False
    raw['target_origin_ns'] = raw['cost_intervals'][0]['start_ns']
    result = audit_answers(raw)
    assert result['evidence_valid'] and result['study_complete'] is False, result
    assert result['analysis']['analysis_looks'] == 0


def test_operation_leaves_cannot_hide_nested_work(futile: dict[str, Any]) -> None:
    raw = deepcopy(futile)
    spans = raw['cost_intervals']
    parent = next(s for s in spans if s['kind'] == 'request')
    child = next(s for s in spans if s['kind'] == 'oracle')
    child['parent_id'] = parent['sample_id']
    child['start_ns'], child['end_ns'] = parent['start_ns'] + 1, parent['end_ns'] - 1
    assert not audit_intervals(spans)['valid']
    assert_refused(raw)


def test_public_cli_refuses_failed_rest_without_positive_analysis(tmp_path: Path, futile: dict[str, Any]) -> None:
    sys.path.insert(0, str(ROOT / 'scripts'))
    cli = importlib.import_module('audit_lookup_answers')
    raw = deepcopy(futile)
    next(s for s in raw['cost_intervals'] if s['kind'] == 'rest')['outcome'] = 'failed'
    path, out = tmp_path / 'input.json', tmp_path / 'audit.json'
    path.write_text(json.dumps(raw), encoding='utf-8')
    before = path.read_bytes()
    with patch('subprocess.Popen', side_effect=AssertionError('offline audit only')):
        assert cli.main([str(path), '--out', str(out)]) == 2
    assert path.read_bytes() == before
    result = json.loads(out.read_text(encoding='utf-8'))
    assert result['evidence_valid'] is result['study_complete'] is False
    assert result['analysis']['analysis_looks'] == 0


def native_stream() -> tuple[Any, list[int], list[dict[str, Any]], dict[str, Any]]:
    engine = AnswerSimulator(FakeClock())
    case = build_corpus().cases[0]
    engine.start('llama3b', 'lookup')
    ids = engine.tokenize(case.prompt, add_special=True)
    chunks: list[dict[str, Any]] = []
    engine.answer(request_body(case), chunks.append)
    return case, ids, [dict(received_ns=10+i, data=chunk) for i, chunk in enumerate(chunks)], request_body(case)


@pytest.mark.parametrize('field,value', (
    ('seed', 999), ('seed', True), ('n_predict', 1), ('n_predict', True), ('temperature', 9),
    ('temperature', True), ('temperature', float('nan')), ('ignore_eos', True), ('ignore_eos', 0),
    ('cache_prompt', True), ('stream', False), ('return_tokens', False), ('stop', ['early']),
    ('speculative.types', 'none'), ('speculative.types', 'ngram-simple,none,other'),
))
def test_present_intermediate_effective_settings_cannot_be_overwritten_by_terminal_defaults(field: str, value: Any) -> None:
    case, ids, events, body = native_stream()
    events[0]['data']['generation_settings'] = {field: value}
    assert not native_answer(events, body, case, 'lookup', ids, 0, 100)['valid']


@pytest.mark.parametrize('field,value', (
    ('generation_settings', None), ('generation_settings', []), ('id_slot', True), ('id_slot', 4),
    ('id_slot', -2), ('id_slot', 1), ('truncated', True), ('truncated', 0), ('stop', 0),
    ('stop_type', 'limit'), ('stop_type', 'eos'),
))
def test_intermediate_identity_and_ending_contradictions_fail_closed(field: str, value: Any) -> None:
    case, ids, events, body = native_stream()
    events[0]['data'][field] = value
    assert not native_answer(events, body, case, 'lookup', ids, 0, 100)['valid']


@pytest.mark.parametrize('timings', (
    {'draft_n': -1}, {'draft_n': True}, {'draft_n_accepted': True}, {'draft_n_accepted': -1},
    {'draft_n': 4, 'draft_n_accepted': 5}, {'draft_n': 999}, {'draft_n_accepted': 999},
    {'prompt_ms': True}, {'prompt_ms': -1}, {'predicted_ms': float('nan')},
))
def test_intermediate_draft_counters_and_times_must_remain_consistent(timings: dict[str, Any]) -> None:
    case, ids, events, body = native_stream()
    events[0]['data']['timings'] = timings
    assert not native_answer(events, body, case, 'lookup', ids, 0, 100)['valid']


@pytest.mark.parametrize('counter', ('tokens_predicted', 'tokens_evaluated', 'predicted_n', 'prompt_n', 'draft_n', 'draft_n_accepted'))
def test_cumulative_intermediate_counters_cannot_go_backwards(counter: str) -> None:
    case, ids, events, body = native_stream()
    middle: dict[str, Any] = dict(received_ns=11, data=dict(content='', tokens=[], stop=False))
    events[-1]['received_ns'] = 12
    events.insert(1, middle)
    if counter in ('tokens_predicted', 'tokens_evaluated'):
        events[0]['data'][counter] = 1
        middle['data'][counter] = 0
    else:
        events[0]['data']['timings'] = {counter: 1}
        middle['data']['timings'] = {counter: 0}
    assert not native_answer(events, body, case, 'lookup', ids, 0, 100)['valid']


@pytest.mark.parametrize('shown', ({}, {'seed': 42}, {'speculative.types': 'ngram-simple'},
                                   {'seed': 42, 'temperature': 0.0, 'ignore_eos': False}))
@pytest.mark.parametrize('slot', (-1, 0))
def test_documented_partial_unknown_slot_and_partial_settings_remain_valid(shown: dict[str, Any], slot: int) -> None:
    case, ids, events, body = native_stream()
    events[0]['data'].update(id_slot=slot, generation_settings=shown, stop_type='none', truncated=False,
                            timings=dict(prompt_ms=0, predicted_ms=0, draft_n=1, draft_n_accepted=0))
    result = native_answer(events, body, case, 'lookup', ids, 0, 100)
    assert result['valid'], result


def test_speculative_accepts_may_be_counted_before_output_loop_stops_at_eos() -> None:
    # b11139 increments accepted stats for a verified batch BEFORE process_token
    # walks it and can encounter EOS. Do not invent accepted <= emitted IDs.
    case, ids, events, body = native_stream()
    accepted = len(events[0]['data']['tokens']) + 2
    events[-1]['data']['timings'].update(draft_n=accepted+2, draft_n_accepted=accepted)
    events[0]['data']['timings'] = dict(draft_n=accepted+2, draft_n_accepted=accepted)
    assert native_answer(events, body, case, 'lookup', ids, 0, 100)['valid']


def test_observed_early_settings_contradiction_reaches_the_public_evidence_gate(futile: dict[str, Any]) -> None:
    raw = deepcopy(futile)
    raw['rows'][0]['events'][0]['data']['generation_settings'] = dict(seed=999, temperature=9, n_predict=1, ignore_eos=True)
    assert_refused(raw)
