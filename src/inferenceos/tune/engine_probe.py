"""One diagnostic run of the engine with its own timers on (never used for speed).

Starts the model with a setup plus the engine's per-operation timer (for the Vulkan
engine ``GGML_VK_PERF_LOGGER``), asks a question a few times at temperature 0, and
returns:

* the engine's device lines (features such as fp16, int dot, matrix cores);
* the card time of every graph, split into reading graphs (the question) and writing
  graphs (one word at a time), and per operation;
* the answers and the engine's own reading and writing times.

The timer makes the engine slower, so nothing here is compared for speed; the fair
comparison always runs with it off. Used by the diagnosis (``tune/diagnose.py``) and
by the engine-switch checks (``tune/engine_switches.py``).
"""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from inferenceos.backends.engine_env import parse_switches, format_switches
from inferenceos.backends.llama_server_toolchain import resolve_server_executable
from inferenceos.backends.process_control import identity_matches
from inferenceos.contracts.lifecycle import ProcessIdentity
from inferenceos.backends.resident_server import LlamaServerResidentBackend
from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.experiment.restoration import runtime_params_from
from inferenceos.interfaces.backend_adapter import GenerationRequest, TokenChoice

#: The engine's own timer, by engine type. For Vulkan the per-operation mode is used: it
#: puts a barrier after every operation, which makes the card's time longer than in a
#: normal run (laptop, 4 Oct 2026: Gemma all on the card wrote a word in 49 ms normally,
#: 94 ms timed), so card time is only used as a *share* of the same timed run's word
#: time (``card_share``). The "concurrent" mode was tried and left out: it does not time
#: the operations after the last sync point, so it missed the output step (5-12 ms of
#: card time per word on Gemma and the 9B).
TIMER_SWITCH = {"vulkan": {"GGML_VK_PERF_LOGGER": "1"}}

_OP_RE = re.compile(r"^(?P<name>.+?): (?P<count>\d+) x (?P<avg>[\d.]+) us = (?P<total>[\d.]+) us")
_TOTAL_RE = re.compile(r"^Total time: (?P<total>[\d.]+) us\.")
_N_RE = re.compile(r" n=(\d+)")

PROBE_PROMPT = "Explain in two sentences what a key value cache does in a language model."

#: Fixed correctness set for engine switches (review 4, 4 Oct 2026): a short question, a
#: question with numbers, and a long input, so one matching answer is not the only
#: evidence. Each is asked once at temperature 0 with the engine's word probabilities.
NUMBER_PROMPT = (
    "A shop sells pens at 17 rupees each and notebooks at 46 rupees each. Priya buys 23 "
    "pens and 9 notebooks. Work out the total cost step by step and give the final number."
)
LONG_PROMPT = (
    "Read the notes below and then answer the question at the end.\n\n"
    + "\n".join(
        f"Note {i}: the warehouse in district {i % 7 + 1} received {12 * i + 5} crates of "
        f"supplies on day {i}, and shipped {7 * i + 3} crates to the {['north', 'south', 'east', 'west'][i % 4]} depot."
        for i in range(1, 41)
    )
    + "\n\nQuestion: how many crates did the warehouse receive on day 17, and where were "
    "the crates shipped that day? Answer in one sentence."
)
CHECK_PROMPTS: tuple[str, ...] = (PROBE_PROMPT, NUMBER_PROMPT, LONG_PROMPT)


@dataclass(frozen=True)
class OpTime:
    name: str
    count: int
    total_us: float


@dataclass(frozen=True)
class Graph:
    """One engine graph run on the card: reading the question, or writing one word."""

    kind: str
    total_us: float
    ops: tuple[OpTime, ...]


@dataclass
class ProbeResult:
    config: RuntimeConfiguration
    ready: bool
    device_lines: list[str] = field(default_factory=list)
    #: Engine type shown by this run: its device lines, or timing blocks from that
    #: engine's own timer (``None`` when neither was seen).
    engine_seen: str | None = None
    startup_lines: int = 0
    graphs: list[Graph] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    #: The engine's own times per answer: milliseconds per written word and to read.
    writing_ms_per_word: list[float] = field(default_factory=list)
    reading_ms: list[float] = field(default_factory=list)
    reading_pieces: list[int] = field(default_factory=list)
    #: The engine's word probabilities per answer (when they were asked for).
    token_choices: list[list[TokenChoice] | None] = field(default_factory=list)
    #: How many answers the probe asked for; a probe is complete only with all of them.
    expected_answers: int = 0
    #: What the engine's output shows about how it ran, by channel ("steps": the
    #: operations of a written word with their counts; "kernels": the compute programs
    #: the engine built; "device": features in the device line). Missing channel = not
    #: visible in this run.
    evidence: dict[str, frozenset[str]] = field(default_factory=dict)
    #: Engine variables the timed run's process really got (diagnostic ones included).
    engine_settings: dict[str, str] = field(default_factory=dict)
    error: str = ""
    #: The engine's last log lines when something went wrong.
    log_tail: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Started, no error, and every answer asked for came back."""
        return (
            self.ready
            and not self.error
            and self.expected_answers > 0
            and len(self.answers) == self.expected_answers
        )

    def card_us_per_word(self) -> float | None:
        """Median card time of one writing graph (one written word)."""
        values = [graph.total_us for graph in self.graphs if graph.kind == "writing"]
        return float(statistics.median(values)) if values else None

    def card_share(self) -> float | None:
        """Card time per written word as a share of this timed run's own word time.

        Both include the timer's slowdown, so the share is a fairer estimate than the
        card time compared with a normal run's word time.
        """
        card_us = self.card_us_per_word()
        if card_us is None or not self.writing_ms_per_word:
            return None
        word_us = float(statistics.median(self.writing_ms_per_word)) * 1000.0
        return min(card_us / word_us, 1.0) if word_us > 0 else None

    def card_us_reading(self) -> float | None:
        values = [graph.total_us for graph in self.graphs if graph.kind == "reading"]
        return float(statistics.median(values)) if values else None

    def op_names(self, kind: str) -> set[str]:
        return {op.name for graph in self.graphs if graph.kind == kind for op in graph.ops}

    def op_us_per_graph(self, kind: str) -> dict[str, float]:
        """Mean card time of each operation per graph of ``kind``."""
        selected = [graph for graph in self.graphs if graph.kind == kind]
        table: dict[str, float] = {}
        for graph in selected:
            for op in graph.ops:
                table[op.name] = table.get(op.name, 0.0) + op.total_us
        return {name: total / len(selected) for name, total in table.items()} if selected else {}

    def device_features(self) -> dict[str, str]:
        """``fp16: 1 | int dot: 0 | matrix cores: none`` style pairs from the device line."""
        features: dict[str, str] = {}
        for line in self.device_lines:
            for part in line.split("|"):
                name, sep, value = part.partition(":")
                if sep and name.strip() and " = " not in name:
                    features[name.strip().lower()] = value.strip()
        return features

    def record(self) -> dict[str, object]:
        return {
            "config": self.config.model_dump(exclude_none=True),
            "ready": self.ready,
            "error": self.error,
            "device_lines": self.device_lines[:20],
            "complete": self.complete,
            "evidence_channels": {name: len(items) for name, items in self.evidence.items()},
            "engine_seen": self.engine_seen,
            "engine_settings": dict(self.engine_settings),
            "log_tail": list(self.log_tail),
            "startup_lines": self.startup_lines,
            "graphs": {
                "reading": sum(1 for g in self.graphs if g.kind == "reading"),
                "writing": sum(1 for g in self.graphs if g.kind == "writing"),
            },
            "card_us_per_word": self.card_us_per_word(),
            "card_share": self.card_share(),
            "card_us_reading": self.card_us_reading(),
            "writing_ms_per_word": self.writing_ms_per_word,
            "reading_ms": self.reading_ms,
            "reading_pieces": self.reading_pieces,
            "answers": [answer[:200] for answer in self.answers],
            "writing_ops_us": dict(
                sorted(self.op_us_per_graph("writing").items(), key=lambda item: -item[1])[:25]
            ),
        }


def graph_kind(ops: list[OpTime]) -> str:
    """"writing" when every matrix product works on one word (n=1), else "reading"."""
    sizes = [int(found.group(1)) for op in ops if "MUL_MAT" in op.name for found in [_N_RE.search(op.name)] if found]
    if not sizes:
        return "unknown"
    return "writing" if max(sizes) == 1 else "reading"


def parse_timing_blocks(lines: list[str]) -> list[Graph]:
    """Every "Vulkan Timings:" block in the engine log."""
    graphs: list[Graph] = []
    ops: list[OpTime] | None = None
    for line in lines:
        text = line.strip()
        if text == "Vulkan Timings:":
            ops = []
            continue
        if ops is None:
            continue
        match = _OP_RE.match(text)
        if match:
            ops.append(OpTime(match["name"], int(match["count"]), float(match["total"])))
            continue
        total = _TOTAL_RE.match(text)
        if total:
            graphs.append(Graph(graph_kind(ops), float(total["total"]), tuple(ops)))
            ops = None
    return graphs


class ProbeNotSafe(RuntimeError):
    """A timed run could not be started or ended safely (an engine process did not
    exit); no further timed runs should be started in this tune."""


def wait_until_gone(identity: ProcessIdentity, *, timeout_s: float = 20.0, poll_s: float = 0.2) -> bool:
    """Whether the process ``identity`` names has exited within ``timeout_s``
    (safe against a reused process number)."""
    deadline = time.monotonic() + timeout_s
    while identity_matches(identity):
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_s)
    return True


_PIPELINE_RE = re.compile(r"ggml_vulkan: pipeline stats for (?P<name>\S+):")
_DEVICE_RE = re.compile(r"ggml_vulkan: \d+ = ")


def _vulkan_evidence(lines: list[str], graphs: list[Graph]) -> dict[str, frozenset[str]]:
    evidence: dict[str, frozenset[str]] = {}
    steps = {
        f"{op.name} x{op.count}" for graph in graphs if graph.kind == "writing" for op in graph.ops
    }
    if steps:
        evidence["steps"] = frozenset(steps)
    kernels = {found["name"] for line in lines for found in [_PIPELINE_RE.search(line)] if found}
    if kernels:
        evidence["kernels"] = frozenset(kernels)
    device = {
        part.strip().lower()
        for line in lines
        if _DEVICE_RE.search(line)
        for part in line.split("|")[1:]
        if ":" in part
    }
    if device:
        evidence["device"] = frozenset(device)
    return evidence


def _vulkan_recognises(lines: list[str], graphs: list[Graph]) -> bool:
    # Vulkan timing blocks only come from the Vulkan engine's own timer.
    return bool(graphs) or any("vulkan" in line.lower() for line in lines)


@dataclass(frozen=True)
class EngineObserver:
    """What one engine type shows about itself: its timer, the extra output that shows
    which code it chose, how to read both, and how to recognise it.

    Another engine (e.g. NVIDIA's) gets its own observer; the checks and the fair test
    that use it stay the same (review 4: the probing, timing and recognition parts are
    engine-specific, the orchestration is not).
    """

    name: str
    timer: dict[str, str]
    #: Extra variables that make the engine print what it chose (diagnostic only).
    evidence_switches: dict[str, str]
    parse_graphs: Callable[[list[str]], list[Graph]]
    evidence_from: Callable[[list[str], list[Graph]], dict[str, frozenset[str]]]
    recognises: Callable[[list[str], list[Graph]], bool]


#: Vulkan, llama.cpp build 11139. ``GGML_VK_PIPELINE_STATS`` prints a line for every
#: compute program the engine builds; programs are built when first used, so the list
#: shows which code really ran (e.g. the integer dot-product programs, ``*_q8_1_*``).
#: It needs the driver's ``VK_KHR_pipeline_executable_properties``; without it the
#: "kernels" channel is simply missing. The filter is a substring of the program name.
VULKAN = EngineObserver(
    name="vulkan",
    timer=dict(TIMER_SWITCH["vulkan"]),
    evidence_switches={"GGML_VK_PIPELINE_STATS": "_"},
    parse_graphs=parse_timing_blocks,
    evidence_from=_vulkan_evidence,
    recognises=_vulkan_recognises,
)

OBSERVERS: dict[str, EngineObserver] = {"vulkan": VULKAN}


def run_probe(
    *,
    settings: Settings,
    model: ModelProfile,
    config: RuntimeConfiguration,
    engine: str = "vulkan",
    repeats: int = 3,
    n_predict: int = 32,
    prompt: str = PROBE_PROMPT,
    prompts: Sequence[str] | None = None,
    n_probs: int = 0,
    exit_timeout_s: float = 20.0,
    evidence: bool = True,
    timer: bool = True,
    backend_factory: Callable[[], LlamaServerResidentBackend] | None = None,
    deadline: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> ProbeResult:
    """Run the setup once with the engine's timer (and evidence output) on.

    ``prompts`` (default: ``prompt`` ``repeats`` times) are asked once each at
    temperature 0; ``n_probs`` > 0 also asks for the engine's word probabilities. The
    model is unloaded after. The probe is complete only when every answer came back.
    Nothing new starts once ``deadline`` (on ``clock``) has passed: not the engine, and
    not another question (review 6); the probe is then reported incomplete.
    """
    observer = OBSERVERS.get(engine)
    switches = parse_switches(config.engine_switches)
    if observer is not None:
        if timer:
            switches.update(observer.timer)
        if evidence:
            switches.update(observer.evidence_switches)
    run_config = config.model_copy(update={"engine_switches": format_switches(switches)})
    backend = (
        backend_factory()
        if backend_factory is not None
        else LlamaServerResidentBackend(server=resolve_server_executable(settings.llama_cpp_dir))
    )
    asked = list(prompts) if prompts is not None else [prompt] * max(int(repeats), 1)
    result = ProbeResult(config=config, ready=False, expected_answers=len(asked))
    if deadline is not None and clock() >= deadline:
        result.error = "time limit reached before the engine was started"
        return result
    try:
        backend.start(model, runtime_params_from(run_config))
        result.ready = True
        settings_reader = getattr(backend, "engine_settings", None)
        if callable(settings_reader):
            result.engine_settings = dict(settings_reader())
        for number, text in enumerate(asked):
            if deadline is not None and clock() >= deadline:
                result.error = (
                    f"time limit reached after {number} of {len(asked)} question(s)"
                )
                break
            answer = backend.generate(
                GenerationRequest(
                    prompt=text, n_predict=n_predict, seed=1, temperature=0.0,
                    ignore_eos=True, n_probs=n_probs,
                )
            )
            result.answers.append(answer.text)
            result.token_choices.append(answer.token_choices)
            if answer.decode_tokens_per_sec:
                result.writing_ms_per_word.append(1000.0 / answer.decode_tokens_per_sec)
            if answer.prefill_s is not None:
                result.reading_ms.append(answer.prefill_s * 1000.0)
            if answer.prompt_tokens is not None:
                result.reading_pieces.append(int(answer.prompt_tokens))
    except Exception as exc:  # noqa: BLE001 - a failed probe is reported, not raised
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        lines = backend.server_log().splitlines()
        startup_reader = getattr(backend, "server_startup_log", None)
        startup = startup_reader().splitlines() if callable(startup_reader) else []
        identity_reader = getattr(backend, "identity", None)
        identity = identity_reader() if callable(identity_reader) else None
        try:
            backend.stop()
        except Exception as exc:  # noqa: BLE001 - checked below: no copy may stay behind
            result.error = result.error or f"the timed run did not stop cleanly: {exc}"
        if isinstance(identity, ProcessIdentity) and not wait_until_gone(
            identity, timeout_s=exit_timeout_s
        ):
            raise ProbeNotSafe(
                "the timed run's engine process did not exit; no further timed runs"
            )
    # Device lines come from the server's first lines: the timer's output can push them
    # out of the recent-lines buffer in a long timed run.
    result.device_lines = [
        line
        for line in (startup or lines)
        if "vulkan" in line.lower() and "Vulkan Timings" not in line
    ][:20]
    parse = observer.parse_graphs if observer is not None else parse_timing_blocks
    result.graphs = parse(lines)
    result.startup_lines = len(startup)
    if result.error:
        result.log_tail = [line for line in lines if "Vulkan Timings" not in line][-40:]
    if observer is not None:
        every_line = startup + lines
        result.evidence = observer.evidence_from(every_line, result.graphs)
        if observer.recognises(startup or lines, result.graphs):
            result.engine_seen = observer.name
    return result


def run_switch_probe(
    *,
    settings: Settings,
    model: ModelProfile,
    config: RuntimeConfiguration,
    engine: str = "vulkan",
    n_probs: int = 3,
    prompts: Sequence[str] = CHECK_PROMPTS,
    backend_factory: Callable[[], LlamaServerResidentBackend] | None = None,
    exit_timeout_s: float = 20.0,
    deadline: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> ProbeResult:
    """The two runs an engine-switch check needs, as one result.

    1. A timed run on the short question: the steps of a written word and card time.
    2. A run with the timer off: the correctness questions with word probabilities,
       and the compute programs the engine built.

    Laptop job 086 (4 Oct 2026): with the model split between processor and card, the
    Vulkan timer of build 11139 stops the engine (``GGML_ASSERT(ctx->compute_ctx.
    expired())``) on any question longer than the short one, so the correctness
    questions never run with the timer on.
    """
    timed = run_probe(
        settings=settings, model=model, config=config, engine=engine,
        prompts=(PROBE_PROMPT,), evidence=False, exit_timeout_s=exit_timeout_s,
        backend_factory=backend_factory, deadline=deadline, clock=clock,
    )
    if deadline is not None and clock() >= deadline:
        # No second run after the time limit (review 5): an incomplete result.
        timed.error = timed.error or "time limit reached before the questions were asked"
        timed.expected_answers = len(prompts)
        timed.answers = []
        timed.token_choices = []
        return timed
    answers = run_probe(
        settings=settings, model=model, config=config, engine=engine, prompts=prompts,
        n_probs=n_probs, timer=False, exit_timeout_s=exit_timeout_s,
        backend_factory=backend_factory, deadline=deadline, clock=clock,
    )
    result = answers
    result.graphs = timed.graphs
    result.writing_ms_per_word = timed.writing_ms_per_word or answers.writing_ms_per_word
    evidence = dict(answers.evidence)
    evidence.pop("steps", None)
    if "steps" in timed.evidence:
        evidence["steps"] = timed.evidence["steps"]
    result.evidence = evidence
    result.engine_seen = timed.engine_seen or answers.engine_seen
    if timed.error:
        result.error = "; ".join(item for item in (f"timed run: {timed.error}", result.error) if item)
        result.log_tail = timed.log_tail or result.log_tail
    if not timed.ready:
        result.ready = False
    return result


__all__ = [
    "CHECK_PROMPTS",
    "EngineObserver",
    "Graph",
    "OBSERVERS",
    "ProbeNotSafe",
    "VULKAN",
    "OpTime",
    "PROBE_PROMPT",
    "ProbeResult",
    "TIMER_SWITCH",
    "graph_kind",
    "parse_timing_blocks",
    "run_probe",
    "run_switch_probe",
    "wait_until_gone",
]
