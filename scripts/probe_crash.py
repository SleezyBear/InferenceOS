"""Find why a timed run with the correctness questions stopped answering (job 085).

Runs the same setup with one thing changed at a time and prints, for each, whether all
answers came back and the engine's last log lines when not. Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.models.loader import load_model_profile
from inferenceos.tune.engine_probe import (
    CHECK_PROMPTS,
    LONG_PROMPT,
    NUMBER_PROMPT,
    PROBE_PROMPT,
    run_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-layers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--ubatch", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    settings = Settings()
    model = load_model_profile(args.model)
    config = RuntimeConfiguration(
        context_length=4096, gpu_offload_layers=args.gpu_layers, threads=args.threads,
        ubatch_size=args.ubatch,
    )
    variants = {
        "all three, probabilities, programs listed": dict(prompts=CHECK_PROMPTS, n_probs=3),
        "all three, no probabilities": dict(prompts=CHECK_PROMPTS, n_probs=0),
        "all three, probabilities, programs not listed": dict(prompts=CHECK_PROMPTS, n_probs=3, evidence=False),
        "short x3, probabilities": dict(prompts=(PROBE_PROMPT,) * 3, n_probs=3),
        "numbers only, probabilities": dict(prompts=(NUMBER_PROMPT,), n_probs=3),
        "long only, probabilities": dict(prompts=(LONG_PROMPT,), n_probs=3),
        "long only, no timer": dict(prompts=(LONG_PROMPT,), n_probs=3, engine="none"),
    }
    report = {}
    for name, options in variants.items():
        result = run_probe(settings=settings, model=model, config=config, **options)  # type: ignore[arg-type]
        report[name] = {
            "complete": result.complete,
            "answers": len(result.answers),
            "expected": result.expected_answers,
            "error": result.error,
            "reading_pieces": result.reading_pieces,
            "log_tail": result.log_tail,
        }
        print(f"{name}: complete={result.complete} answers={len(result.answers)}/{result.expected_answers} {result.error}")
        for line in result.log_tail[-12:]:
            print("    | " + line[:200])
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
