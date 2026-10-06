"""Compare two long-question lengths for the paths that read the question.

Option 1: the long question fits ~20 s per answer (tune's Balanced rule).
Option 2: a fixed, longer question (default: half the context).

Runs ``tune`` twice from the same saved setup (the saved setups file is put back after
each run), with the layer path off and only the paths that use the long question, then
checks the two chosen setups against each other in turns on option 2's question and
the short question. Prints and saves: time taken by each option, what each chose, and
how much faster option 2's choice really is on the long question.

    .\\.venv\\Scripts\\python.exe scripts\\study_question_length.py --model M.gguf
        [--context 4096] [--long-tokens 2048] [--turns 4] --out-dir _jobs\\out\\study
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PATHS = "reading_threads,flash_attention,prompt_batch"


def _config_text(config: dict[str, Any]) -> str:
    parts = []
    for key, value in sorted(config.items()):
        if key == "context_length" or value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        parts.append(f"{key}={value}")
    return ",".join(parts)


def _final(report: dict[str, Any]) -> dict[str, Any]:
    saved = report.get("saved_setup")
    if isinstance(saved, dict) and isinstance(saved.get("configuration"), dict):
        return dict(saved["configuration"])
    return dict(report.get("final_configuration") or {})


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"outcome": report.get("outcome"), "final": _final(report)}
    for item in report.get("rounds", []):
        if item.get("round") == "long test question":
            out["long_question"] = {
                key: item.get(key)
                for key in ("pieces", "workload", "expected_seconds_per_answer", "rule", "reading_speed")
            }
        if item.get("path"):
            out[str(item["path"])] = {
                "applies": item.get("applies"),
                "reason": item.get("reason"),
                "selected": item.get("selected"),
                "gain": item.get("gain"),
                "first_look": {
                    str(entry.get("candidate")): entry.get("screen_gain")
                    for entry in item.get("screened", [])
                },
            }
    return out


def _run_tune(
    python: str, model: str, context: int, tokens: int, report: Path, log: Path
) -> float:
    env = dict(os.environ)
    env["INFERENCEOS_TUNE_LAYER_PATH"] = "false"
    env["INFERENCEOS_TUNE_PATHS"] = PATHS
    env["INFERENCEOS_TUNE_LONG_QUESTION_TOKENS"] = str(tokens)
    command = [
        python, "-u", "-m", "inferenceos", "tune", "--model", model, "--context", str(context),
        "--effort", "performance_first", "--yes", "--out", str(report),
    ]
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        process.wait()
    return time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--long-tokens", type=int, default=0, help="option 2 length (0 = half the context)")
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cool-down", type=int, default=180, help="seconds between runs")
    args = parser.parse_args(argv)

    from inferenceos.config import Settings

    settings = Settings()
    store = settings.data_dir / "saved_setups.json"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    backup = out / "saved_setups.before.json"
    if store.is_file():
        shutil.copy2(store, backup)
    python = sys.executable
    long_tokens = args.long_tokens or args.context // 2

    def restore() -> None:
        if backup.is_file():
            shutil.copy2(backup, store)

    results: dict[str, Any] = {"model": args.model, "context": args.context}
    reports: dict[str, Path] = {}
    for name, tokens in (("option1", 0), ("option2", long_tokens)):
        restore()
        print(f"=== {name}: long question {'fit to the time limit' if tokens == 0 else f'{tokens} pieces'}", flush=True)
        report = out / f"{name}_report.json"
        seconds = _run_tune(python, args.model, args.context, tokens, report, out / f"{name}.log")
        reports[name] = report
        entry: dict[str, Any] = {"minutes": round(seconds / 60.0, 1)}
        if report.is_file():
            entry.update(_summary(json.loads(report.read_text(encoding="utf-8"))))
        results[name] = entry
        print(f"=== {name} took {entry['minutes']} min", flush=True)
        time.sleep(max(args.cool_down, 0))
    restore()

    first, second = results.get("option1", {}), results.get("option2", {})
    final1, final2 = first.get("final"), second.get("final")
    long_id = (second.get("long_question") or {}).get("workload") or "long-prefill-steady"
    if final1 and final2 and _config_text(final1) != _config_text(final2):
        print("=== the two options chose different setups: checking them in turns", flush=True)
        check = out / "crosscheck.json"
        command = [
            python, "-u", "scripts/check_tune_result.py", "--model", args.model,
            "--context", str(args.context), "--start", _config_text(final1),
            "--final", _config_text(final2), "--turns", str(args.turns),
            "--workloads", f"{long_id},short-short", "--out", str(check),
        ]
        subprocess.run(command, check=False)
        if check.is_file():
            results["crosscheck"] = {
                "start_is_option1": _config_text(final1),
                "final_is_option2": _config_text(final2),
                "result": json.loads(check.read_text(encoding="utf-8")),
            }
    else:
        results["crosscheck"] = "both options chose the same setup"
    (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
