"""What the installed llama-server build says it accepts (its ``--help`` text).

This answers only the first of the three questions (``contracts/capability.py``):
whether the engine accepts an option and a value. It never says a setting applies to
a model or took effect. Which RuntimeConfiguration setting maps to which option lives
here, in the llama.cpp adapter, so another engine brings its own table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from inferenceos.contracts.capability import Support

#: RuntimeConfiguration setting -> the llama-server option that sets it.
SETTING_OPTIONS: dict[str, str] = {
    "threads": "--threads",
    "threads_batch": "--threads-batch",
    "batch_size": "--batch-size",
    "ubatch_size": "--ubatch-size",
    "flash_attention": "--flash-attn",
    "gpu_offload_layers": "--n-gpu-layers",
    "kv_cache_type": "--cache-type-k",
    "kv_cache_type_k": "--cache-type-k",
    "kv_cache_type_v": "--cache-type-v",
    "placement_overrides": "--override-tensor",
    "device": "--device",
}

_OPTION_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]+)")
_ALLOWED_RE = re.compile(r"allowed values:\s*(.+)", re.IGNORECASE)


@dataclass(frozen=True)
class ServerOptions:
    """The options one llama-server build lists, with allowed values where it says."""

    names: frozenset[str]
    #: Option -> the values its help text lists as allowed (only where it lists them).
    allowed: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def accepts(self, setting: str, value: object = None) -> tuple[Support, str]:
        """Whether this build accepts ``setting`` (and ``value``, when the help text
        lists allowed values). Unknown settings and unlisted values stay unknown or no;
        a yes here says nothing about the model or the effect."""
        option = SETTING_OPTIONS.get(setting)
        if option is None:
            return Support.UNKNOWN, f"no llama-server option is known for {setting}"
        if option not in self.names:
            return Support.NO, f"this llama-server does not list {option}"
        if value is None:
            return Support.YES, f"{option} is listed"
        allowed = self.allowed.get(option)
        if allowed is None:
            return Support.YES, f"{option} is listed (its values are not listed)"
        text = str(value)
        if text in allowed:
            return Support.YES, f"{option} lists {text}"
        return Support.NO, f"{option} does not list {text} (listed: {', '.join(allowed)})"


def parse_server_help(text: str) -> ServerOptions | None:
    """The options and allowed values in a ``--help`` text (None when none found)."""
    names = frozenset(_OPTION_RE.findall(text))
    if not names:
        return None
    allowed: dict[str, tuple[str, ...]] = {}
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            current = _OPTION_RE.findall(stripped.split("  ")[0]) or _OPTION_RE.findall(stripped)
            continue
        found = _ALLOWED_RE.search(stripped)
        if found and current:
            values = tuple(item.strip() for item in found.group(1).split(",") if item.strip())
            for option in current:
                allowed[option] = values
    return ServerOptions(names=names, allowed=allowed)


__all__ = ["SETTING_OPTIONS", "ServerOptions", "parse_server_help"]
