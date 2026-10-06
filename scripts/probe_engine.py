"""Look inside one llama-server run: device features, per-operation card timings, answers.

Starts llama-server for one model and setup, optionally with engine switches (environment
variables), with the Vulkan per-operation timer on (``GGML_VK_PERF_LOGGER``), asks the
short question a few times at temperature 0, and saves:

* the engine's device lines (features such as fp16, int dot, matrix cores);
* every "Vulkan Timings" block, parsed into operations, and split into reading graphs
  (the question) and writing graphs (one word at a time);
* the answers and the engine's own timings.

Used to find out which engine switches change anything on this machine, and where the
time of one written word goes. The timer itself slows things down; these runs are never
used to compare speed. Nothing is changed or saved outside ``--out``.

    .\\.venv\\Scripts\\python.exe scripts\\probe_engine.py --model M.gguf --ngl 1
        [--switch GGML_VK_DISABLE_FUSION=1 ...] [--repeats 3] --out probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

_OP_RE = re.compile(r"^(?P<name>.+?): (?P<count>\d+) x (?P<avg>[\d.]+) us = (?P<total>[\d.]+) us")
_TOTAL_RE = re.compile(r"^Total time: (?P<total>[\d.]+) us\.")
_PROMPT = "Explain in two sentences what a key value cache does in a language model."


def parse_timing_blocks(lines: list[str]) -> list[dict[str, object]]:
    """Every "Vulkan Timings:" block as {ops: [...], total_us, kind}."""
    blocks: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in lines:
        text = line.strip()
        if text == "Vulkan Timings:":
            current = {"ops": []}
            continue
        if current is None:
            continue
        match = _OP_RE.match(text)
        if match:
            ops = current["ops"]
            assert isinstance(ops, list)
            ops.append(
                {
                    "name": match["name"],
                    "count": int(match["count"]),
                    "avg_us": float(match["avg"]),
                    "total_us": float(match["total"]),
                }
            )
            continue
        total = _TOTAL_RE.match(text)
        if total:
            current["total_us"] = float(total["total"])
            current["kind"] = graph_kind(current["ops"])  # type: ignore[arg-type]
            blocks.append(current)
            current = None
    return blocks


def graph_kind(ops: list[dict[str, object]]) -> str:
    """"writing" when every matrix product works on one word (n=1), else "reading"."""
    sizes = [
        int(found.group(1))
        for op in ops
        if str(op["name"]).split(" ")[-1] != "" and "MUL_MAT" in str(op["name"])
        for found in [re.search(r" n=(\d+)", str(op["name"]))]
        if found
    ]
    if not sizes:
        return "unknown"
    return "writing" if max(sizes) == 1 else "reading"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _post(port: int, body: dict[str, object], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback
        result = json.loads(response.read().decode("utf-8"))
    assert isinstance(result, dict)
    return result


def probe(
    server: Path,
    model: Path,
    *,
    ngl: int,
    threads: int | None,
    switches: dict[str, str],
    repeats: int,
    n_predict: int,
    timeout_s: float,
    profile: bool = True,
    concurrent: bool = True,
) -> dict[str, object]:
    port = _free_port()
    command = [
        str(server), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "-ngl", str(ngl), "-c", "4096", "--no-webui", "-lv", "4",
    ]
    if threads:
        command += ["-t", str(threads)]
    env = dict(os.environ)
    env.update(switches)
    if profile:
        env["GGML_VK_PERF_LOGGER"] = "1"
        if concurrent:
            # Stamps only at the engine's own sync points (close to a normal run).
            env["GGML_VK_PERF_LOGGER_CONCURRENT"] = "1"
    flags = 0x08000000 if sys.platform == "win32" else 0
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", creationflags=flags, env=env,
    )
    lines: list[str] = []

    def read() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip())

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    started = time.perf_counter()
    ready = False
    while time.perf_counter() - started < timeout_s:
        joined = "\n".join(lines[-50:])
        if (
            "server is listening" in joined
            or "main: model loaded" in joined
            or "llama_server: model loaded" in joined
            or "listening on http" in joined
        ):
            ready = True
            break
        if process.poll() is not None:
            break
        time.sleep(0.25)
    answers: list[dict[str, object]] = []
    marks: list[int] = []
    try:
        if ready:
            for _ in range(repeats):
                marks.append(len(lines))
                result = _post(
                    port,
                    {
                        "prompt": _PROMPT, "n_predict": n_predict, "temperature": 0.0,
                        "seed": 1, "cache_prompt": False, "ignore_eos": True,
                    },
                    timeout_s,
                )
                time.sleep(0.3)  # let the last timing block reach the log
                answers.append(
                    {"text": result.get("content"), "timings": result.get("timings")}
                )
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        reader.join(timeout=5)
    device_lines = [
        line for line in lines
        if "vulkan" in line.lower() and "Vulkan Timings" not in line
        or "fp16" in line or "int dot" in line or "matrix cores" in line
    ]
    blocks = parse_timing_blocks(lines)
    writing = [block for block in blocks if block.get("kind") == "writing"]
    reading = [block for block in blocks if block.get("kind") == "reading"]

    def per_op(selected: list[dict[str, object]]) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        for block in selected:
            for op in block["ops"]:  # type: ignore[attr-defined]
                row = table.setdefault(str(op["name"]), {"count": 0.0, "total_us": 0.0})
                row["count"] += float(op["count"])
                row["total_us"] += float(op["total_us"])
        for row in table.values():
            row["avg_us"] = row["total_us"] / row["count"] if row["count"] else 0.0
        return dict(sorted(table.items(), key=lambda item: -item[1]["total_us"]))

    return {
        "command": command,
        "switches": switches,
        "ready": ready,
        "returncode": process.returncode,
        "device_lines": device_lines[:40],
        "answers": answers,
        "graphs": {"reading": len(reading), "writing": len(writing), "all": len(blocks)},
        "writing_graph_us": [block.get("total_us") for block in writing],
        "reading_graph_us": [block.get("total_us") for block in reading],
        "writing_ops": per_op(writing),
        "reading_ops": per_op(reading),
        "log_tail": lines[-30:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ngl", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--switch", action="append", default=[], help="NAME=VALUE (repeatable)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--per-op", action="store_true", help="time every operation (adds barriers)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from inferenceos.backends.llama_server_toolchain import resolve_server_executable
    from inferenceos.config import Settings

    server = resolve_server_executable(Settings().llama_cpp_dir)
    switches = dict(item.split("=", 1) for item in args.switch)
    result = probe(
        Path(server), Path(args.model), ngl=args.ngl, threads=args.threads or None,
        switches=switches, repeats=args.repeats, n_predict=args.n_predict,
        timeout_s=args.timeout, profile=not args.no_profile, concurrent=not args.per_op,
    )
    print(f"switches: {switches or 'none'}; ready: {result['ready']}")
    for line in result["device_lines"][:6]:  # type: ignore[index]
        print("  " + str(line))
    print(f"  graphs: {result['graphs']}")
    writing = [v for v in result["writing_graph_us"] if v]  # type: ignore[attr-defined]
    if writing:
        writing_sorted = sorted(float(v) for v in writing)
        print(f"  card time per written word (median): {writing_sorted[len(writing_sorted) // 2]:.0f} us")
    for answer in result["answers"]:  # type: ignore[attr-defined]
        timings = answer.get("timings") or {}
        print(
            f"  answer: {str(answer.get('text'))[:60]!r} | writing "
            f"{timings.get('predicted_per_token_ms')} ms/word, reading {timings.get('prompt_ms')} ms"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
