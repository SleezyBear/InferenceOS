"""Local reasoning layer (Phase 5 deep-diagnosis mode).

Deterministic escalation -> local reasoning model -> strict structured output ->
evidence grounding -> validated root-cause hypothesis, with safe fallback to the
Phase 4 deterministic diagnosis. Diagnosis only — no optimization recommendations.
"""

from __future__ import annotations

from inferenceos.reasoning.context import DEFAULT_OBJECTIVE, build_reasoning_context
from inferenceos.reasoning.engine import LocalReasoningEngine
from inferenceos.reasoning.escalation import decide_escalation
from inferenceos.reasoning.factory import build_reasoning_engine
from inferenceos.reasoning.lifecycle import TARGET_STATE, run_deep_diagnosis
from inferenceos.reasoning.parse import (
    ReasoningParseError,
    extract_json_object,
    parse_reasoning_result,
)
from inferenceos.reasoning.prompt import (
    RECOMMENDATION_KEYS,
    REASONING_OUTPUT_KEYS,
    build_json_schema,
    render_system_prompt,
    render_user_prompt,
)
from inferenceos.reasoning.providers import (
    FakeReasoningProvider,
    LlamaCppReasoningProvider,
    ProviderResult,
    ReasoningProvider,
)
from inferenceos.reasoning.report import render_deep_diagnosis
from inferenceos.reasoning.validate import (
    ReasoningValidationError,
    validate_reasoning_result,
)

__all__ = [
    "DEFAULT_OBJECTIVE",
    "FakeReasoningProvider",
    "LlamaCppReasoningProvider",
    "LocalReasoningEngine",
    "ProviderResult",
    "RECOMMENDATION_KEYS",
    "REASONING_OUTPUT_KEYS",
    "ReasoningParseError",
    "ReasoningProvider",
    "ReasoningValidationError",
    "TARGET_STATE",
    "build_json_schema",
    "build_reasoning_context",
    "build_reasoning_engine",
    "decide_escalation",
    "extract_json_object",
    "parse_reasoning_result",
    "render_deep_diagnosis",
    "render_system_prompt",
    "render_user_prompt",
    "run_deep_diagnosis",
    "validate_reasoning_result",
]
