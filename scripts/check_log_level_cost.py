"""Does the engine's debug log level (-lv 5) cost writing speed? (no tuning, nothing saved)

llama.cpp b11139 prints which pieces a placement moved ("tensor X ... buffer type
overridden to Y") only at its debug level (5); tune measures at level 4. This alternates
loads at level 4 and level 5 with the same setup, writes a fixed number of words twice per
load, and reports writing speed, first-word time, how many log lines each level produced
and how many per-piece lines were seen. If level 5 costs nothing measurable, tune can
measure every setup at level 5, so every measured load shows its own placement.

    .\\.venv\\Scripts\\python.exe scripts\\check_log_level_cost.py --model models\\study\\granite-3.1-3b-a800m-instruct-Q4_K_M.gguf --layers 33 --overrides "ffn_(up|down|gate)_exps=CPU" --rounds 4 --out _jobs\\out\\112_granite31moe.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--overrides", default=None)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from inferenceos.backends.llama_server_toolchain import resolve_server_executable
    from inferenceos.backends.resident_server import LlamaServerResidentBackend
    from inferenceos.config import Settings
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.experiment.restoration import runtime_params_from
    from inferenceos.interfaces.backend_adapter import GenerationRequest
    from inferenceos.models.loader import load_model_profile

    settings = Settings()
    model = load_model_profile(Path(args.model))
    config = RuntimeConfiguration(
        context_length=args.context, gpu_offload_layers=args.layers, placement_overrides=args.overrides or None
    )
    prompt = "Write a short story about a lighthouse keeper who finds a message in a bottle."
    rows: list[dict[str, object]] = []
    for round_index in range(args.rounds):
        order = (4, 5) if round_index % 2 == 0 else (5, 4)
        for level in order:
            backend = LlamaServerResidentBackend(server=resolve_server_executable(settings.llama_cpp_dir))
            row: dict[str, object] = {"round": round_index + 1, "level": level}
            try:
                backend.start(model, runtime_params_from(config).model_copy(update={"log_verbosity": level}))
                speeds, firsts = [], []
                for _ in range(2):
                    answer = backend.generate(
                        GenerationRequest(prompt=prompt, n_predict=128, temperature=0.0, seed=1, ignore_eos=True)
                    )
                    if answer.decode_tokens_per_sec:
                        speeds.append(answer.decode_tokens_per_sec)
                    first = getattr(answer, "ttft_s", None)
                    if first:
                        firsts.append(first)
                row["writing_tokens_per_s"] = speeds
                row["first_word_s"] = firsts
                startup = backend.server_startup_log()
                row["startup_lines"] = startup.count("\n") + 1 if startup else 0
                row["overridden_lines"] = startup.count("buffer type overridden")
                row["log_lines_total"] = backend.server_log().count("\n") + 1
                row["observed_overridden"] = len((backend.telemetry().observed or {}).get("overridden_tensors") or {})
            except Exception as exc:  # noqa: BLE001 - reported
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    backend.stop()
                except Exception as exc:  # noqa: BLE001 - reported
                    row["stop_error"] = str(exc)
            rows.append(row)
            print(json.dumps(row, default=str)[:400])
            time.sleep(3)
    summary: dict[str, object] = {"rows": rows}
    for level in (4, 5):
        speeds = [s for r in rows if r.get("level") == level for s in r.get("writing_tokens_per_s", [])]  # type: ignore[union-attr]
        if speeds:
            summary[f"level_{level}_median_writing"] = statistics.median(speeds)
    if "level_4_median_writing" in summary and "level_5_median_writing" in summary:
        a = float(summary["level_4_median_writing"])  # type: ignore[arg-type]
        b = float(summary["level_5_median_writing"])  # type: ignore[arg-type]
        summary["level_5_vs_4"] = (b - a) / a
        print(f"level 5 vs level 4 writing speed: {(b - a) / a:+.2%}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
