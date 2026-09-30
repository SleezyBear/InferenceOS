"""The LLM may propose candidates but can never bypass deterministic safety."""

from __future__ import annotations

import json

import pytest

from inferenceos.contracts.enums import BottleneckKind
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.optimization.reasoning import (
    ProposalParseError,
    parse_proposals,
)
from inferenceos.reasoning.providers.fake import FakeReasoningProvider
from tests.optimization_helpers import paging_bundle, synthetic_context, synthetic_diagnosis


def _payload(actions: list[dict[str, object]]) -> str:
    return json.dumps(
        {"schema_version": "1.0", "proposed_actions": actions, "summary": "s"}
    )


def _adversarial() -> list[dict[str, object]]:
    return [
        {
            "action_id": "shell.run_command",
            "parameter_names": ["cmd"],
            "parameter_values": ["Remove-Item -Recurse C:\\"],
            "evidence_ids": [],
            "rationale": "free memory",
        },
        {
            "action_id": "runtime.reduce_threads",
            "parameter_names": ["threads"],
            "parameter_values": [4096],
            "evidence_ids": ["E999"],
            "rationale": "hallucinated evidence",
        },
        {
            "action_id": "runtime.reduce_threads",
            "parameter_names": ["threads"],
            "parameter_values": [4096],
            "evidence_ids": [],
            "rationale": "reduce page faults by 99%",
        },
    ]


def _parse(text: str) -> tuple[list[dict[str, object]], list[str], list[str]]:
    ctx = synthetic_context(BottleneckKind.PAGING)
    return parse_proposals(
        text,
        registry=build_default_registry(),
        allowed_evidence_ids=list(ctx.evidence_ids),
        allowed_values=[0.7, 4096.0],
    )


def test_unregistered_action_rejected() -> None:
    accepted, rejections, proposed = _parse(
        _payload([_adversarial()[0]])
    )
    assert accepted == []
    assert any("unregistered action" in item for item in rejections)
    assert "shell.run_command" in proposed


def test_hallucinated_evidence_rejected() -> None:
    _, rejections, _ = _parse(_payload([_adversarial()[1]]))
    assert any("hallucinated evidence" in item for item in rejections)


def test_fabricated_measurement_rejected() -> None:
    _, rejections, _ = _parse(_payload([_adversarial()[2]]))
    assert any("fabricated measurement" in item for item in rejections)


def test_parameter_out_of_range_rejected() -> None:
    _, rejections, _ = _parse(
        _payload(
            [
                {
                    "action_id": "runtime.reduce_threads",
                    "parameter_names": ["threads"],
                    "parameter_values": [-5],
                    "evidence_ids": [],
                    "rationale": "negative threads",
                }
            ]
        )
    )
    assert any("below minimum" in item for item in rejections)


def test_malformed_output_raises() -> None:
    with pytest.raises(ProposalParseError):
        _parse("not json at all")


def test_generator_enforces_registry_and_grounding() -> None:
    diagnosis = synthetic_diagnosis(
        BottleneckKind.INSUFFICIENT_EVIDENCE, meaningful=False, insufficient=True
    )
    provider = FakeReasoningProvider(scripted=[_payload(_adversarial())])
    plan = generate_candidates(
        diagnosis,
        report=None,
        policy=None,
        provider=provider,
        enable_reasoning=True,
    )
    assert plan.reasoning is not None
    assert plan.reasoning.used is True
    assert any("unregistered" in item for item in plan.reasoning.rejections)
    assert any("hallucinated evidence" in item for item in plan.reasoning.rejections)
    assert any("fabricated measurement" in item for item in plan.reasoning.rejections)
    assert all("shell" not in c.action_id for c in plan.candidates)
    assert all(not c.executable for c in plan.candidates)


def test_generator_falls_back_when_provider_fails() -> None:
    diagnosis = synthetic_diagnosis(
        BottleneckKind.INSUFFICIENT_EVIDENCE, meaningful=False, insufficient=True
    )
    provider = FakeReasoningProvider(fail_with="model unavailable")
    plan = generate_candidates(
        diagnosis, report=None, provider=provider, enable_reasoning=True
    )
    assert plan.reasoning is not None and plan.reasoning.status == "failed"
    # Deterministic terminal outcome is preserved; nothing executable was invented.
    assert all(not c.executable for c in plan.candidates)


def test_reasoning_not_used_for_strong_deterministic_mapping() -> None:
    _, diagnosis, report = paging_bundle()
    provider = FakeReasoningProvider(scripted=[_payload(_adversarial())])
    plan = generate_candidates(
        diagnosis, report=report, provider=provider, enable_reasoning=True
    )
    # Strong paging mapping is deterministic; the model is not consulted.
    assert plan.reasoning is None
    assert any(c.action_id == "runtime.enable_mmap" and c.executable for c in plan.candidates)
    # Context right-sizing never produces a candidate at all.
    assert all(
        c.action_id != "runtime.right_size_context" for c in plan.candidates
    )


def test_provider_exception_is_contained() -> None:
    class ExplodingProvider:
        name = "exploding"
        model_id = "boom"

        def is_available(self) -> bool:
            return True

        def complete(self, **kwargs: object) -> object:
            raise RuntimeError("boom")

    diagnosis = synthetic_diagnosis(
        BottleneckKind.INSUFFICIENT_EVIDENCE, meaningful=False, insufficient=True
    )
    plan = generate_candidates(
        diagnosis,
        report=None,
        provider=ExplodingProvider(),  # type: ignore[arg-type]
        enable_reasoning=True,
    )
    assert plan.reasoning is not None and plan.reasoning.status == "failed"
    assert plan.outcome.value == "insufficient_evidence"
