"""Laptop smoke test of the engine-switch checks (review 4, 4 Oct 2026).

For one model and setup, runs the unchanged setup twice and every listed switch once,
each as a timed run with the correctness questions and word probabilities, exactly as
``tune`` does before any speed test. Reports per switch: whether its effect could be
seen (and how), whether the answers agree, and that every engine process exited.
Never measures speed.

    python scripts/check_switches.py --model models\\study\\google_gemma-3-1b-it-Q4_K_M.gguf \
        --gpu-layers 1 --threads 9 --ubatch 1024 --out _jobs\\out\\085_switches.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from inferenceos.backends.engine_env import parse_switches
from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.models.loader import load_model_profile
from inferenceos.tune.engine_probe import ProbeNotSafe, ProbeResult, run_switch_probe
from inferenceos.tune.engine_switches import (
    VULKAN_B11139,
    activation,
    answers_agree,
    apply_switch,
)


def _engine_processes() -> list[int]:
    try:
        import psutil
    except ImportError:
        return []
    found = []
    for process in psutil.process_iter(["name"]):
        name = str(process.info.get("name") or "").lower()
        if "llama-server" in name:
            found.append(int(process.pid))
    return found


def _summary(probe: ProbeResult) -> dict[str, object]:
    return {
        "complete": probe.complete,
        "error": probe.error,
        "engine_seen": probe.engine_seen,
        "startup_lines": probe.startup_lines,
        "engine_settings": probe.engine_settings,
        "evidence": {name: sorted(items)[:40] for name, items in probe.evidence.items()},
        "evidence_counts": {name: len(items) for name, items in probe.evidence.items()},
        "probabilities": [len(choices or []) for choices in probe.token_choices],
        "answers": [answer[:160] for answer in probe.answers],
        "card_ms_per_word": (probe.card_us_per_word() or 0.0) / 1000.0,
        "device_lines": probe.device_lines[:8],
        "log_tail": probe.log_tail[-15:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--ubatch", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    settings = Settings()
    model = load_model_profile(args.model)
    config = RuntimeConfiguration(
        context_length=args.context,
        gpu_offload_layers=args.gpu_layers,
        threads=args.threads,
        ubatch_size=args.ubatch,
    )
    report: dict[str, object] = {"config": config.model_dump(exclude_none=True)}
    before = _engine_processes()
    report["engine_processes_before"] = before

    def probe(setup: RuntimeConfiguration) -> ProbeResult:
        return run_switch_probe(settings=settings, model=model, config=setup)

    started = time.perf_counter()
    try:
        base, again = probe(config), probe(config)
    except ProbeNotSafe as exc:
        report["stopped"] = str(exc)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"stopped: {exc}")
        return 1
    report["base"] = _summary(base)
    report["base_again"] = _summary(again)
    print(
        f"unchanged runs: complete {base.complete}/{again.complete}; evidence channels "
        f"{sorted(base.evidence)}; word probabilities {[len(c or []) for c in base.token_choices]}"
    )
    ok, why = answers_agree(base, again)
    report["base_agree"] = why
    print(f"  the two unchanged runs: {why}")
    checks = []
    for spec in VULKAN_B11139:
        setup = config.model_copy(update={"engine_switches": apply_switch(config.engine_switches, spec)})
        try:
            result = probe(setup)
        except ProbeNotSafe as exc:
            checks.append({"switch": spec.text, "stopped": str(exc)})
            print(f"{spec.text}: stopped ({exc})")
            break
        status, effect = activation(spec.evidence, base, again, result)
        same, answers = answers_agree(base, result)
        print(f"{spec.text}: {status} ({effect}); {answers}")
        checks.append(
            {
                "switch": spec.text,
                "switches_sent": sorted(parse_switches(setup.engine_switches)),
                "activation": status,
                "effect": effect,
                "answers_agree": same,
                "answers": answers,
                "run": _summary(result),
            }
        )
    report["checks"] = checks
    report["minutes"] = round((time.perf_counter() - started) / 60.0, 1)
    after = _engine_processes()
    report["engine_processes_after"] = after
    leftover = sorted(set(after) - set(before))
    report["leftover_engine_processes"] = leftover
    print(f"engine processes left running: {leftover or 'none'}")
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if not leftover else 2


if __name__ == "__main__":
    sys.exit(main())
