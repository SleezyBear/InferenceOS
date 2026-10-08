"""Check, on the real engine, what the new placement and cache settings do (no tuning).

For one model, starts llama-server briefly with a few setups, asks one short question
and two known-answer questions, and reports for each: whether it started and answered,
whether the known answers were right (judged from the answer line, as tune does), what
the engine's own log showed (model buffers per device, the per-piece "overridden" lines,
conversation-memory formats, flash attention), and whether that matches what
InferenceOS expected (``verify_placement`` / ``verify_cache``). The raw startup log of
every setup is saved, so the log patterns can be checked and kept as test fixtures, and
the grouped placement of a hybrid model gets the hardware probe it needs before tune may
accept it (``tune_grouped_placement_probed``). Nothing is saved to the user's setups.

    .\\.venv\\Scripts\\python.exe scripts\\check_new_paths.py --model models\\ab\\granite-3.1-3b-a800m-instruct-Q4_K_M.gguf --layers 33 --out _jobs\\out\\check_new_paths
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", type=int, required=True, help="the model's current layer count")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from inferenceos.backends.llama_server_toolchain import resolve_server_executable
    from inferenceos.backends.resident_server import LlamaServerResidentBackend
    from inferenceos.config import Settings
    from inferenceos.contracts.experiment import RuntimeConfiguration
    from inferenceos.experiment.restoration import runtime_params_from
    from inferenceos.interfaces.backend_adapter import GenerationRequest
    from inferenceos.models.loader import load_model_profile
    from inferenceos.tune.known_answers import QUESTIONS, judge_answer, workload_for
    from inferenceos.tune.kv_cache import verify_cache
    from inferenceos.tune.model_facts import read_model_facts
    from inferenceos.tune.tensor_placement import placement_plans, verify_placement

    settings = Settings()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model)
    model = load_model_profile(model_path)
    facts = read_model_facts(model_path, model)
    base = RuntimeConfiguration(context_length=args.context, gpu_offload_layers=args.layers)
    setups: list[tuple[str, RuntimeConfiguration]] = [("current", base)]
    options = placement_plans(
        facts, gpu_layers=args.layers, current_overrides=None, card_budget_bytes=None, card=True
    )
    for plan in options.plans[:4]:
        setups.append(
            (plan.label, base.model_copy(update={"gpu_offload_layers": plan.gpu_layers, "placement_overrides": plan.overrides}))
        )
    setups += [
        ("cache keys q8_0", base.model_copy(update={"kv_cache_type_k": "q8_0"})),
        ("cache both q8_0, flash on", base.model_copy(update={"kv_cache_type_k": "q8_0", "kv_cache_type_v": "q8_0", "flash_attention": True})),
        ("cache values q8_0, flash off (expected to be refused)", base.model_copy(update={"kv_cache_type_v": "q8_0", "flash_attention": False})),
    ]
    rows = []
    print(f"model: {model_path.name}; {facts.design_words()}; tied output: {facts.tied_output}")
    for note in options.notes:
        print(f"  note: {note}")
    for index, (label, config) in enumerate(setups):
        backend = LlamaServerResidentBackend(server=resolve_server_executable(settings.llama_cpp_dir))
        row: dict[str, object] = {"setup": label, "config": config.model_dump(exclude_none=True)}
        started = time.perf_counter()
        try:
            # Placements start at the engine's debug level (5): b11139 lists the pieces it
            # moved only there (job 110); everything else at the normal level (4).
            level = 5 if config.placement_overrides else 4
            row["log_level"] = level
            backend.start(model, runtime_params_from(config).model_copy(update={"log_verbosity": level}))
            answer = backend.generate(GenerationRequest(prompt="Say hello in five words.", n_predict=16, temperature=0.0, seed=1))
            row["answered"] = bool(answer.text.strip())
            row["writing_tokens_per_s"] = answer.decode_tokens_per_sec
            known: dict[str, object] = {}
            for question in (QUESTIONS[0], QUESTIONS[2]):
                if question.context_tokens > args.context:
                    continue
                reply = backend.generate(
                    GenerationRequest(prompt=question.prompt, n_predict=workload_for(question).n_predict, temperature=0.0, seed=1234)
                )
                right, words = judge_answer(question, reply.text)
                known[question.question_id] = {"right": right, "words": words, "text": reply.text[-200:]}
            row["known_answers"] = known
        except Exception as exc:  # noqa: BLE001 - reported
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            log = backend.server_startup_log() or backend.server_log()
            observed = backend.telemetry().observed if backend.is_running() else {}
            try:
                backend.stop()
            except Exception as exc:  # noqa: BLE001 - reported
                row["stop_error"] = str(exc)
        row["seconds"] = round(time.perf_counter() - started, 1)
        row["observed"] = observed
        if config.placement_overrides or label == "current":
            row["placement_check"] = verify_placement(
                facts, config.gpu_offload_layers, config.placement_overrides, observed
            )
        if config.kv_cache_type_k or config.kv_cache_type_v:
            row["cache_check"] = verify_cache(config, observed)
        (out / f"log_{index:02d}.txt").write_text(log, encoding="utf-8")
        rows.append(row)
        print(f"- {label}: " + json.dumps({k: v for k, v in row.items() if k not in ("config",)}, default=str)[:600])
    (out / "summary.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
