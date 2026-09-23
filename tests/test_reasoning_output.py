"""Parsing, strict schema validation, grounding validation, prompts and gating."""

from __future__ import annotations

import json

import pytest

from inferenceos.contracts.enums import BottleneckKind, InferenceStage
from inferenceos.contracts.reasoning import ReasoningResult
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.reasoning.escalation import decide_escalation
from inferenceos.reasoning.parse import (
    ReasoningParseError,
    extract_json_object,
    parse_reasoning_result,
)
from inferenceos.reasoning.prompt import build_json_schema, render_user_prompt
from inferenceos.reasoning.validate import ReasoningValidationError, validate_reasoning_result
from tests.diagnosis_fixtures import make_report, make_run
from tests.reasoning_helpers import (
    healthy_bundle,
    insufficient_bundle,
    paging_bundle,
    paging_report,
    valid_result_json,
)


def _ids(context: object) -> list[str]:
    return [item.evidence_id for item in getattr(context, "evidence")]


def test_extract_json_object_from_noise() -> None:
    text = 'chatter {"a": 1} more chatter {"root_cause_class": "paging"} tail'
    assert extract_json_object(text) == {"root_cause_class": "paging"}
    assert extract_json_object("no json at all") is None


def test_parse_valid_result() -> None:
    context, _, _ = paging_bundle()
    result, stripped = parse_reasoning_result(
        valid_result_json(context), allowed_evidence_ids=_ids(context)
    )
    assert result.root_cause_class.value == "paging"
    assert result.affected_stage is InferenceStage.END_TO_END
    assert stripped == []


def test_parse_missing_key_fails() -> None:
    context, _, _ = paging_bundle()
    payload = json.loads(valid_result_json(context))
    del payload["reasoning"]
    with pytest.raises(ReasoningParseError):
        parse_reasoning_result(json.dumps(payload), allowed_evidence_ids=_ids(context))


def test_parse_unknown_key_fails() -> None:
    context, _, _ = paging_bundle()
    payload = json.loads(valid_result_json(context))
    payload["secret_extra"] = 1
    with pytest.raises(ReasoningParseError):
        parse_reasoning_result(json.dumps(payload), allowed_evidence_ids=_ids(context))


def test_parse_strips_recommendation_keys() -> None:
    context, _, _ = paging_bundle()
    payload = json.loads(valid_result_json(context))
    payload["recommendations"] = ["reduce the context size"]
    payload["actions"] = ["change threads"]
    result, stripped = parse_reasoning_result(
        json.dumps(payload), allowed_evidence_ids=_ids(context)
    )
    assert result.root_cause_class.value == "paging"
    assert sorted(stripped) == ["actions", "recommendations"]


def test_parse_invalid_enum_fails() -> None:
    context, _, _ = paging_bundle()
    payload = json.loads(valid_result_json(context))
    payload["root_cause_class"] = "made_up_cause"
    with pytest.raises(ReasoningParseError):
        parse_reasoning_result(json.dumps(payload), allowed_evidence_ids=_ids(context))


def test_parse_schema_version_mismatch_fails() -> None:
    context, _, _ = paging_bundle()
    payload = json.loads(valid_result_json(context))
    payload["schema_version"] = "9.9"
    with pytest.raises(ReasoningParseError):
        parse_reasoning_result(json.dumps(payload), allowed_evidence_ids=_ids(context))


def test_parse_unknown_evidence_id_fails() -> None:
    context, _, _ = paging_bundle()
    payload = json.loads(valid_result_json(context, supporting=["E999"]))
    with pytest.raises(ReasoningParseError):
        parse_reasoning_result(json.dumps(payload), allowed_evidence_ids=_ids(context))


def test_validate_unknown_evidence_id() -> None:
    context, _, _ = paging_bundle()
    result = ReasoningResult(
        root_cause_class=BottleneckKind.PAGING, supporting_evidence_ids=["E999"]
    )
    with pytest.raises(ReasoningValidationError):
        validate_reasoning_result(result, context)


def test_validate_rejects_ungrounded_number() -> None:
    context, _, _ = paging_bundle()
    result = ReasoningResult(
        root_cause_class=BottleneckKind.PAGING,
        supporting_evidence_ids=[context.evidence[0].evidence_id],
        summary="GPU utilization was 98% during the run.",
    )
    with pytest.raises(ReasoningValidationError):
        validate_reasoning_result(result, context)


def test_validate_accepts_grounded_number() -> None:
    context, _, _ = paging_bundle()
    value = context.evidence[0].observed_value
    result = ReasoningResult(
        root_cause_class=BottleneckKind.PAGING,
        supporting_evidence_ids=[context.evidence[0].evidence_id],
        summary=f"page faults reached {value} per second",
    )
    validate_reasoning_result(result, context)


def test_validate_rejects_optimization_recommendation() -> None:
    context, _, _ = paging_bundle()
    result = ReasoningResult(
        root_cause_class=BottleneckKind.PAGING,
        summary="You should reduce the context size to fix this.",
    )
    with pytest.raises(ReasoningValidationError):
        validate_reasoning_result(result, context)


def test_prompt_and_schema_enumerate_ids() -> None:
    context, _, _ = paging_bundle()
    prompt = render_user_prompt(context)
    assert "DETERMINISTIC DIAGNOSIS" in prompt
    assert context.evidence[0].evidence_id in prompt
    schema = build_json_schema(_ids(context))
    properties = schema["properties"]
    assert isinstance(properties, dict)
    supporting = properties["supporting_evidence_ids"]
    assert isinstance(supporting, dict)
    items = supporting["items"]
    assert isinstance(items, dict)
    assert items["enum"] == _ids(context)


def test_escalation_strong_deterministic_not_escalated() -> None:
    _, diagnosis, _ = paging_bundle()
    assert diagnosis.meaningful_opportunity
    decision = decide_escalation(diagnosis)
    assert decision.should_escalate is False
    assert decision.reason_code == "strong_deterministic"


def test_escalation_insufficient_evidence_escalates() -> None:
    _, diagnosis, _ = insufficient_bundle()
    decision = decide_escalation(diagnosis)
    assert decision.should_escalate is True
    assert decision.reason_code == "insufficient_evidence"


def test_escalation_no_signal_does_not_escalate() -> None:
    _, diagnosis, _ = healthy_bundle()
    decision = decide_escalation(diagnosis)
    assert decision.should_escalate is False
    assert decision.reason_code in {"no_signal", "deterministic_sufficient"}


def test_escalation_disabled() -> None:
    _, diagnosis, _ = paging_bundle()
    decision = decide_escalation(diagnosis, enabled=False)
    assert decision.should_escalate is False
    assert decision.reason_code == "disabled"


def test_escalation_competing_hypotheses() -> None:
    runs = [
        make_run(model_name="cpu", workload_id="short-short", condition="gpu", metrics=_healthy()),
        make_run(model_name="cpu", workload_id="short-short", condition="gpu", metrics=_healthy()),
        make_run(model_name="cpu", workload_id="short-short", condition="cpu", metrics=_healthy(gpu_utilization_percent=5.0, process_cpu_percent=520.0, decode_tokens_per_sec=6.0), gpu_offload_layers=0),
        make_run(model_name="cpu", workload_id="short-short", condition="cpu", metrics=_healthy(gpu_utilization_percent=5.0, process_cpu_percent=520.0, decode_tokens_per_sec=6.0), gpu_offload_layers=0),
        make_run(model_name="io", workload_id="short-short", condition="control", metrics=_healthy(hard_page_faults_per_sec=0.5), memory_pressure=0.6),
        make_run(model_name="io", workload_id="short-short", condition="control", metrics=_healthy(hard_page_faults_per_sec=0.6), memory_pressure=0.6),
        make_run(model_name="io", workload_id="short-short", condition="pressured", metrics=_healthy(hard_page_faults_per_sec=16000.0), memory_pressure=0.84),
        make_run(model_name="io", workload_id="short-short", condition="pressured", metrics=_healthy(hard_page_faults_per_sec=17000.0), memory_pressure=0.85),
    ]
    report = make_report(runs)
    diagnosis = BottleneckDetector().detect(report, source_report_path="<test>")
    decision = decide_escalation(diagnosis)
    assert decision.should_escalate is True
    assert decision.reason_code == "competing_hypotheses"
    assert decision.competing_families >= 2


def _healthy(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "decode_tokens_per_sec": 24.0,
        "total_latency_s": 1.4,
        "process_cpu_percent": 120.0,
        "gpu_utilization_percent": 70.0,
        "peak_rss_bytes": 1_300_000_000,
        "available_ram_bytes": 8 * 1024**3,
        "hard_page_faults_per_sec": 0.5,
        "generated_tokens": 24,
    }
    base.update(overrides)
    return base
