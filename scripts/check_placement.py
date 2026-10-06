"""Check which pieces llama.cpp puts on the graphics card for a layer count.

Starts llama-server on a model with ``-ngl N`` for a few counts (default 0, 1, 2 and
blocks+1), reads the engine's own load log ("offloading output layer to GPU",
"offloaded X/Y layers", graphics-card buffer size) and compares the graphics-card
buffer with what each placement rule predicts from the model file's sizes. Prints
which rule this engine build follows. Nothing is changed or saved.

    .\\.venv\\Scripts\\python.exe scripts\\check_placement.py --model models\\study\\google_gemma-3-1b-it-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

_BUFFER_RE = re.compile(r"load_tensors:\s+(\S+)\s+model buffer size\s*=\s*([\d.]+)\s*MiB")
_OFFLOADED_RE = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers")


def run_once(server: Path, model: Path, count: int, port: int, timeout_s: float) -> dict[str, object]:
    command = [
        str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-ngl", str(count), "-c", "4096", "--no-webui", "-lv", "4",
    ]
    flags = 0x08000000 if sys.platform == "win32" else 0
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", creationflags=flags,
    )
    lines: list[str] = []

    def read() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip())

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    started = time.perf_counter()
    while time.perf_counter() - started < timeout_s:
        text = "\n".join(lines)
        if "server is listening" in text or "main: model loaded" in text or process.poll() is not None:
            break
        time.sleep(0.5)
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
    reader.join(timeout=5)
    text = "\n".join(lines)
    buffers: dict[str, float] = {}
    for name, size in _BUFFER_RE.findall(text):
        buffers[name] = buffers.get(name, 0.0) + float(size)
    offloaded = _OFFLOADED_RE.search(text)
    return {
        "count": count,
        "output_line": "offloading output layer to GPU" in text,
        "offloaded": f"{offloaded.group(1)}/{offloaded.group(2)}" if offloaded else None,
        "buffers_mib": buffers,
        "card_mib": sum(v for k, v in buffers.items() if not k.upper().startswith("CPU")),
        "log_tail": [line for line in lines if "load_tensors" in line or "offload" in line][-12:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--counts", default="")
    parser.add_argument("--port", type=int, default=8137)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from inferenceos.backends.llama_cpp_toolchain import cli_version
    from inferenceos.backends.llama_server_toolchain import resolve_server_executable
    from inferenceos.config import Settings
    from inferenceos.models.loader import load_model_profile
    from inferenceos.tune.model_facts import read_model_facts
    from inferenceos.tune.placement import BLOCKS_FIRST, OUTPUT_FIRST, placement

    settings = Settings()
    server = resolve_server_executable(settings.llama_cpp_dir)
    model_path = Path(args.model)
    model = load_model_profile(model_path)
    facts = read_model_facts(model_path, model)
    total = facts.total_layers
    counts = (
        [int(item) for item in args.counts.split(",") if item.strip()]
        if args.counts
        else [0, 1, 2, total, total + 1]
    )
    layer_mib = [b / 1024**2 for b in (facts.layer_bytes or ())]
    output_mib = (facts.output_bytes or 0) / 1024**2

    def predicted(count: int, rule: str) -> float:
        where = placement(count, total, rule)
        size = sum(layer_mib[i] for i in where.blocks_on_card if i < len(layer_mib))
        return float(size + (output_mib if where.output_on_card else 0.0))

    version = cli_version(Path(server))
    print(f"engine : {version}")
    print(f"model  : {model_path.name}, {total} blocks, output step {output_mib:.0f} MiB")
    rows = []
    votes = {OUTPUT_FIRST: 0.0, BLOCKS_FIRST: 0.0}
    for index, count in enumerate(counts):
        row = run_once(Path(server), model_path, count, args.port + index, args.timeout)
        row["predicted_mib"] = {
            OUTPUT_FIRST: round(predicted(count, OUTPUT_FIRST), 1),
            BLOCKS_FIRST: round(predicted(count, BLOCKS_FIRST), 1),
        }
        card = float(row["card_mib"])  # type: ignore[arg-type]
        for rule in votes:
            votes[rule] += abs(card - predicted(count, rule))
        rows.append(row)
        print(
            f"-ngl {count:>3}: card buffers {card:8.1f} MiB | output-first predicts "
            f"{row['predicted_mib'][OUTPUT_FIRST]:8.1f} | blocks-first predicts "  # type: ignore[index]
            f"{row['predicted_mib'][BLOCKS_FIRST]:8.1f} | 'offloading output layer' "  # type: ignore[index]
            f"{'yes' if row['output_line'] else 'no'} | offloaded {row['offloaded']}"
        )
    rule = min(votes, key=lambda key: votes[key])
    print(f"this engine follows: {rule} (total misfit {votes[rule]:.0f} MiB vs "
          f"{max(votes.values()):.0f} MiB for the other rule)")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"engine": version, "model": str(model_path), "rule": rule, "rows": rows}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
