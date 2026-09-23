"""Strict parsing of the local model's JSON output.

Malformed, schema-invalid, or unexpected output is a reasoning failure — never
silently accepted.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from pydantic import ValidationError

from inferenceos.contracts.reasoning import REASONING_SCHEMA_VERSION, ReasoningResult
from inferenceos.reasoning.prompt import REASONING_OUTPUT_KEYS, RECOMMENDATION_KEYS


class ReasoningParseError(ValueError):
    """Raised when the model output is not a valid reasoning result."""


def _balanced_objects(text: str) -> Iterator[str]:
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : index + 1]
                    break


def extract_json_object(text: str) -> dict[str, object] | None:
    """Return the last balanced JSON object in the text, or ``None``."""
    best: dict[str, object] | None = None
    for candidate in _balanced_objects(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            best = parsed
    return best


def parse_reasoning_result(
    text: str,
    *,
    allowed_evidence_ids: list[str],
    expected_schema_version: str = REASONING_SCHEMA_VERSION,
) -> tuple[ReasoningResult, list[str]]:
    """Parse, strictly validate and return the result plus stripped keys."""
    obj = extract_json_object(text)
    if obj is None:
        raise ReasoningParseError("no JSON object found in model output")

    stripped = sorted(key for key in list(obj) if key in RECOMMENDATION_KEYS)
    for key in stripped:
        obj.pop(key, None)

    unknown = sorted(set(obj) - REASONING_OUTPUT_KEYS)
    if unknown:
        raise ReasoningParseError(f"unexpected output keys: {unknown}")
    missing = sorted(REASONING_OUTPUT_KEYS - set(obj))
    if missing:
        raise ReasoningParseError(f"missing required output keys: {missing}")
    if obj.get("schema_version") != expected_schema_version:
        raise ReasoningParseError(f"schema_version mismatch: {obj.get('schema_version')!r}")

    try:
        result = ReasoningResult.model_validate(obj)
    except ValidationError as exc:
        raise ReasoningParseError(f"schema validation failed: {exc.errors()[:3]}") from exc

    unknown_ids = (
        set(result.supporting_evidence_ids) | set(result.contradicting_evidence_ids)
    ) - set(allowed_evidence_ids)
    if unknown_ids:
        raise ReasoningParseError(f"unknown evidence ids: {sorted(unknown_ids)}")
    return result, stripped
