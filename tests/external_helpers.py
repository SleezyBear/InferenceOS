"""Shared fixtures for Phase 11 external-recommendation tests.

The controlled doubles live in ``inferenceos.external.verification`` so
``verify-phase11`` and the tests share a single implementation.
"""

from __future__ import annotations

from inferenceos.external.controlled import (
    FakeClock,
    ScriptedProcessObserver,
    StubExternalActionExecutor,
    launch_cpu_spinner,
    launch_memory_holder,
    launch_sleeper,
    make_identity,
    make_process_observation,
    make_report,
)


def default_large_rss() -> int:
    return 2 * 1024**3


__all__ = [
    "FakeClock",
    "ScriptedProcessObserver",
    "StubExternalActionExecutor",
    "default_large_rss",
    "launch_cpu_spinner",
    "launch_memory_holder",
    "launch_sleeper",
    "make_identity",
    "make_process_observation",
    "make_report",
]
