"""Engine switches: settings inside the engine, tried like any other setting.

An engine reads some environment variables when it starts that choose how it works
inside (which code multiplies matrices, whether steps are merged, where buffers live).
Owner's rules (4 Oct 2026, refined after review 4):

* only switches listed here for the installed engine build are ever tried (a list per
  engine type and build, written from that build's source; nothing is discovered
  automatically and nothing is assumed to carry over to another build or engine);
* before a switch is tried for speed, a timed run must show that it **took effect**:
  something the engine prints about how it ran must change (the steps of a written
  word, or the compute programs it built). A change in timing alone is not proof.
  A switch whose effect this engine cannot show is not tested ("activation
  unverified; not tested"), and that is not a claim that it does not help;
* the answers must stay correct: every timed run complete, and on a fixed set of
  questions (short, with numbers, long) every answer the same word for word, with the
  engine's own word probabilities within a tolerance (``answers_agree``);
* the speed test is the normal fair comparison, with every timer off.

The list for another engine (for example the NVIDIA one) is added the same way, from that
build's source, together with that engine's observer (``tune/engine_probe.py``); the
checks and the fair test stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferenceos.backends.engine_env import format_switches, parse_switches
from inferenceos.tune.engine_probe import ProbeResult

#: Results of the activation check.
SHOWN, NOT_SEEN, UNVERIFIED = "shown", "not seen", "unverified"

#: Words for each evidence channel (``ProbeResult.evidence``).
_CHANNEL_WORDS = {
    "steps": "the steps of a written word",
    "kernels": "the compute programs the engine built",
    "device": "the engine's device features",
}


@dataclass(frozen=True)
class SwitchSpec:
    name: str
    value: str
    #: Plain words for the report.
    label: str
    #: Evidence channels that show this switch took effect, from the engine's source
    #: (empty: this engine cannot show it, so the switch is never tested).
    evidence: tuple[str, ...] = ()
    #: Switches that cannot be set together with this one (removed when it is set).
    conflicts: tuple[str, ...] = ()
    #: Device features the switch needs, e.g. (("int dot", "1"),); checked only when
    #: the engine printed its device features.
    requires: tuple[tuple[str, str], ...] = ()

    @property
    def text(self) -> str:
        return f"{self.name}={self.value}"


#: Vulkan engine, llama.cpp build 11139 (commit 42916d83f): switches read in
#: ggml/src/ggml-vulkan/ggml-vulkan.cpp of that commit. Evidence from the same source:
#: * MMVQ choice: the integer dot-product programs (``*_q8_1_*``, ``quantize_q8_1*``)
#:   are built only when used ("kernels");
#: * fusion and graph reordering: the timer names merged steps ("steps");
#: * integer dot and half precision: the programs built change ("kernels"), and the
#:   device line (printed only at debug log level) shows "int dot" / "fp16";
#: * transfer queue, async copies, submit batch size, host-visible card memory: nothing
#:   the engine prints shows them, so they are not tested.
#: Laptop probe, 4 Oct 2026: none made the Radeon Pro 560X faster; fusion off and
#: graph reordering off made it slower.
VULKAN_B11139: tuple[SwitchSpec, ...] = (
    SwitchSpec(
        "GGML_VK_FORCE_MMVQ", "1", "always use the integer dot-product code for one-word steps",
        evidence=("kernels",),
        conflicts=("GGML_VK_DISABLE_MMVQ", "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT"),
        requires=(("int dot", "1"),),
    ),
    SwitchSpec(
        "GGML_VK_DISABLE_MMVQ", "1", "never use the integer dot-product code for one-word steps",
        evidence=("kernels",), conflicts=("GGML_VK_FORCE_MMVQ",),
    ),
    SwitchSpec(
        "GGML_VK_ASYNC_USE_TRANSFER_QUEUE", "1", "copy data on the separate transfer queue",
        conflicts=("GGML_VK_DISABLE_ASYNC",),
    ),
    SwitchSpec("GGML_VK_MAX_NODES_PER_SUBMIT", "400", "send larger batches of work to the card"),
    SwitchSpec("GGML_VK_MAX_NODES_PER_SUBMIT", "25", "send smaller batches of work to the card"),
    SwitchSpec(
        "GGML_VK_DISABLE_ASYNC", "1", "no overlapping of copies and work",
        conflicts=("GGML_VK_ASYNC_USE_TRANSFER_QUEUE",),
    ),
    SwitchSpec(
        "GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1", "no integer dot-product instructions",
        evidence=("kernels", "device"), conflicts=("GGML_VK_FORCE_MMVQ",),
    ),
    SwitchSpec(
        "GGML_VK_DISABLE_F16", "1", "no half-precision arithmetic", evidence=("kernels", "device"),
    ),
    SwitchSpec("GGML_VK_DISABLE_FUSION", "1", "do not merge neighbouring steps", evidence=("steps",)),
    SwitchSpec("GGML_VK_DISABLE_GRAPH_OPTIMIZE", "1", "do not reorder steps", evidence=("steps",)),
    SwitchSpec(
        "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "1", "no card memory the processor can see",
    ),
)

#: (engine, build marker in the full version line) -> switches for that build.
CAPABILITIES: dict[tuple[str, str], tuple[SwitchSpec, ...]] = {
    ("vulkan", "42916d83f"): VULKAN_B11139,
}


def switches_for(
    engine: str, version: str | None, features: dict[str, str] | None = None
) -> tuple[SwitchSpec, ...]:
    """The switches listed for this engine build, or none when the build is not listed.

    With the device's features known, switches whose needs are not met are left out
    (e.g. forcing integer dot-product code on a card without integer dot products).
    """
    text = version or ""
    for (name, marker), specs in CAPABILITIES.items():
        if name == engine and marker in text:
            return tuple(spec for spec in specs if _meets(spec, features))
    return ()


def _meets(spec: SwitchSpec, features: dict[str, str] | None) -> bool:
    if not features:
        return True
    return all(features.get(key, wanted) == wanted for key, wanted in spec.requires)


def spec_named(engine: str, version: str | None, name: str) -> SwitchSpec | None:
    for spec in switches_for(engine, version):
        if spec.name == name:
            return spec
    return None


def apply_switch(text: str | None, spec: SwitchSpec) -> str | None:
    """``text`` with ``spec`` set and every switch it conflicts with removed."""
    switches = parse_switches(text)
    for name in spec.conflicts:
        switches.pop(name, None)
    switches[spec.name] = spec.value
    return format_switches(switches)


def remove_switch(text: str | None, name: str) -> str | None:
    """``text`` with one switch back at the engine's default (removed, never "=0")."""
    switches = parse_switches(text)
    switches.pop(name, None)
    return format_switches(switches)


def activation(
    channels: tuple[str, ...], base: ProbeResult, base_again: ProbeResult, other: ProbeResult
) -> tuple[str, str]:
    """Did a switch take effect? Judged only on what the engine prints about how it ran.

    ``channels``: the evidence the switch's spec names. A channel counts only when both
    runs of the unchanged setup show it and agree with each other. Returns ``SHOWN``,
    ``NOT_SEEN`` (visible but unchanged) or ``UNVERIFIED`` (nothing visible), with words.
    A timing change is never evidence (review 4, 4 Oct 2026).
    """
    if not channels:
        return UNVERIFIED, "activation unverified; not tested (this engine shows nothing about it)"
    usable: list[str] = []
    for channel in channels:
        first, second, seen = (
            base.evidence.get(channel), base_again.evidence.get(channel), other.evidence.get(channel)
        )
        if not first or not second or not seen or first != second:
            continue
        usable.append(channel)
        if seen != first:
            changed = len(seen ^ first)
            return SHOWN, f"{_CHANNEL_WORDS.get(channel, channel)} changed ({changed} differ)"
    if usable:
        words = ", ".join(_CHANNEL_WORDS.get(channel, channel) for channel in usable)
        return NOT_SEEN, f"no effect seen ({words} unchanged); not tested"
    return UNVERIFIED, (
        "activation unverified; not tested (the engine did not print "
        + " or ".join(_CHANNEL_WORDS.get(channel, channel) for channel in channels)
        + " in this run)"
    )


def answers_agree(
    base: ProbeResult,
    other: ProbeResult,
    *,
    prob_tolerance: float = 0.03,
) -> tuple[bool, str]:
    """Are the answers still correct with the switch? (Before any speed test.)

    Both runs must be complete (every question answered, no error). Every answer must
    choose exactly the same words, and word by word the engine's probability for the
    chosen word may move by at most ``prob_tolerance``. Any different word fails, even
    where the unchanged run's two best words were close: uncertainty in the unchanged
    run does not show a changed answer is harmless (review 5, 5 Oct 2026). Without
    probabilities, the texts must be identical.
    """
    if not base.complete:
        return False, "the unchanged run was incomplete" + (f" ({base.error})" if base.error else "")
    if not other.complete:
        return False, "the run with the switch was incomplete" + (
            f" ({other.error})" if other.error else ""
        )
    if len(base.answers) != len(other.answers):
        return False, "a different number of answers"
    worst = 0.0
    for index, (text_a, text_b) in enumerate(zip(base.answers, other.answers)):
        if text_a != text_b:
            return False, f"answer {index + 1} changed"
        choices_a = base.token_choices[index] if index < len(base.token_choices) else None
        choices_b = other.token_choices[index] if index < len(other.token_choices) else None
        if not choices_a or not choices_b:
            continue
        if len(choices_a) != len(choices_b):
            return False, f"answer {index + 1} has a different length"
        for word_a, word_b in zip(choices_a, choices_b):
            if word_a.token != word_b.token:
                return False, f"answer {index + 1} chose a different word"
            moved = abs(word_a.prob - word_b.prob)
            worst = max(worst, moved)
            if moved > prob_tolerance:
                return False, (
                    f"answer {index + 1}: a word's probability moved by {moved:.3f} "
                    f"(limit {prob_tolerance:.3f})"
                )
    return True, (
        f"answers identical on {len(base.answers)} questions "
        f"(largest probability change {worst:.3f})"
    )


__all__ = [
    "CAPABILITIES",
    "NOT_SEEN",
    "SHOWN",
    "SwitchSpec",
    "UNVERIFIED",
    "VULKAN_B11139",
    "activation",
    "answers_agree",
    "apply_switch",
    "remove_switch",
    "spec_named",
    "switches_for",
]
