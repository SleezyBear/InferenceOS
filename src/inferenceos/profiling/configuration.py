"""Lossless requested-runtime snapshots inside the existing profiling extra map.

These are requests, separate from observed/effective settings. Older profiles keep
their legacy conversion; a present but damaged full snapshot cannot silently fall
back to an incomplete setup.
"""
from __future__ import annotations

import json

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.profiling import ProfilingConfig

RUNTIME_REQUEST_KEY = "runtime_request_v1"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    if len(pairs) != len({key for key, _ in pairs}):
        raise ValueError("repeated field in runtime request snapshot")
    return dict(pairs)


def encode_runtime_request(configuration: RuntimeConfiguration) -> str:
    return configuration.model_dump_json()


def requested_runtime(configuration: ProfilingConfig) -> RuntimeConfiguration | None:
    raw = configuration.extra.get(RUNTIME_REQUEST_KEY)
    if raw is None:
        return None
    data = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or set(data) != set(RuntimeConfiguration.model_fields):
        raise ValueError("incomplete runtime request snapshot; baseline configuration cannot be reconstructed")
    return RuntimeConfiguration.model_validate(data)
