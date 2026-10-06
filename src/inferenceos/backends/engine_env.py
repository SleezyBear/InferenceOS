"""Engine switches: environment variables the engine reads when it starts.

Stored in a setup as one canonical text, "NAME=VALUE;NAME=VALUE" sorted by name, so two
setups with the same switches compare equal and a saved setup round-trips unchanged.
Which switches may be tried at all is decided per engine build and hardware in
``tune/engine_switches.py``; this module only reads and writes the text.
"""

from __future__ import annotations

import re

_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def parse_switches(text: str | None) -> dict[str, str]:
    """``"A=1;B=2"`` -> ``{"A": "1", "B": "2"}`` (empty for None or "")."""
    result: dict[str, str] = {}
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        name, sep, value = part.partition("=")
        name = name.strip()
        if not sep or not _NAME_RE.match(name):
            raise ValueError(f"not an engine switch: {part!r} (expected NAME=VALUE)")
        result[name] = value.strip()
    return result


def format_switches(switches: dict[str, str]) -> str | None:
    """The canonical text for ``switches`` (None when there are none)."""
    if not switches:
        return None
    for name in switches:
        if not _NAME_RE.match(name):
            raise ValueError(f"not an engine switch name: {name!r}")
    return ";".join(f"{name}={switches[name]}" for name in sorted(switches))


def with_switch(text: str | None, name: str, value: str | None) -> str | None:
    """``text`` with one switch set (``value``) or removed (``None``)."""
    switches = parse_switches(text)
    if value is None:
        switches.pop(name, None)
    else:
        switches[name] = value
    return format_switches(switches)


#: Engine variables that only switch on diagnostics (timers, extra logging). They change
#: how the engine runs (the Vulkan timer adds a sync after every step), so they must
#: never reach a speed test, not even when inherited from the parent environment
#: (review 4, 4 Oct 2026). Most are presence-based: "=0" does not turn them off, so they
#: are removed, not reset. A diagnostic run asks for them explicitly.
PROFILING_VARIABLES: frozenset[str] = frozenset(
    {
        # Vulkan (llama.cpp ggml-vulkan.cpp, build 11139)
        "GGML_VK_PERF_LOGGER",
        "GGML_VK_PERF_LOGGER_CONCURRENT",
        "GGML_VK_PERF_LOGGER_FREQUENCY",
        "GGML_VK_SYNC_LOGGER",
        "GGML_VK_MEMORY_LOGGER",
        "GGML_VK_PIPELINE_STATS",
        "GGML_VK_DEBUG_MARKERS",
        # CUDA: makes every launch wait for the card
        "CUDA_LAUNCH_BLOCKING",
    }
)

#: Prefixes of variables that change how an engine runs (shown in reports when set).
ENGINE_PREFIXES: tuple[str, ...] = ("GGML_", "LLAMA_", "CUDA_", "HIP_", "ROCR_", "MTL_")


def process_environment(
    parent: dict[str, str], explicit: dict[str, str] | None
) -> dict[str, str]:
    """The environment for an engine process: the parent's, with every diagnostic
    variable removed unless ``explicit`` asks for it, plus ``explicit``."""
    wanted = dict(explicit or {})
    env = {name: value for name, value in parent.items() if name not in PROFILING_VARIABLES}
    env.update(wanted)
    return env


def engine_settings(env: dict[str, str]) -> dict[str, str]:
    """The engine-related variables in ``env`` (what the engine will actually read)."""
    return {
        name: value
        for name, value in sorted(env.items())
        if name.upper().startswith(ENGINE_PREFIXES)
    }


__all__ = [
    "ENGINE_PREFIXES",
    "PROFILING_VARIABLES",
    "engine_settings",
    "format_switches",
    "parse_switches",
    "process_environment",
    "with_switch",
]
