"""Investigation-only: distinguish CPU utilization vs efficiency/contention for 12 vs 6
threads on the real resident llama-server (CPU-only). Does not modify production behavior.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\investigate_cpu_bottleneck_metric.py ^
        --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_ab_effectiveness as ab  # noqa: E402
import validate_cpu_thread_surface as v  # noqa: E402

import psutil  # noqa: E402

from inferenceos.config import Settings  # noqa: E402
from inferenceos.external.controlled import launch_cpu_spinner  # noqa: E402
from inferenceos.hardware.profiler import detect_hardware_profile  # noqa: E402
from inferenceos.models.loader import load_model_profile  # noqa: E402


def _measure(
    *, model: Any, server_path: Path, settings: Settings, threads: int,
    requests: list[ab.FixedRequest], spinners: int, log_path: Path,
) -> dict[str, Any]:
    launched = ab.start_unmanaged_server(
        server=server_path, model_path=Path(model.path), host=settings.resident_host,
        context_length=4096, gpu_layers=0, threads=threads, ignore_eos=False,
        log_path=log_path, readiness_timeout_s=settings.resident_readiness_timeout_s,
    )
    process = psutil.Process(launched.pid)
    procs = [launch_cpu_spinner(240.0) for _ in range(spinners)]
    for ctx in procs:
        ctx.__enter__()
    try:
        time.sleep(3.0)
        # Warm-up (excluded).
        ab.run_unmanaged_request(
            test="warmup", request=requests[0], host=launched.host, port=launched.port,
            server_pid=launched.pid, timeout_s=settings.resident_request_timeout_s,
        )
        times0 = process.cpu_times()
        cpu0 = float(times0.user) + float(times0.system)
        sw0 = process.num_ctx_switches()
        ctx0 = int(sw0.voluntary) + int(sw0.involuntary)
        threads_seen = process.num_threads()
        t0 = time.perf_counter()
        outcomes = [
            ab.run_unmanaged_request(
                test="measure", request=request, host=launched.host, port=launched.port,
                server_pid=launched.pid, timeout_s=settings.resident_request_timeout_s,
            )
            for request in requests
        ]
        t1 = time.perf_counter()
        times1 = process.cpu_times()
        cpu1 = float(times1.user) + float(times1.system)
        sw1 = process.num_ctx_switches()
        ctx1 = int(sw1.voluntary) + int(sw1.involuntary)
    finally:
        for ctx in reversed(procs):
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        ab.stop_unmanaged_server(launched)

    wall = max(t1 - t0, 1e-9)
    cpu_seconds = max(cpu1 - cpu0, 0.0)
    ctx_switches = max(ctx1 - ctx0, 0)
    decode_values = [o.decode_tokens_per_sec for o in outcomes if o.decode_tokens_per_sec]
    tokens = sum(o.generated_tokens or 0 for o in outcomes)
    decode_tps = statistics.fmean(decode_values) if decode_values else None
    return {
        "requested_threads": threads,
        "effective_threads_log": (v._server_log_threads(log_path) or [None])[-1],
        "os_threads_seen": threads_seen,
        "requests": len(outcomes),
        "generated_tokens": tokens,
        "decode_tokens_per_sec": decode_tps,
        "wall_s": wall,
        "cpu_seconds": cpu_seconds,
        "process_cpu_percent": cpu_seconds / wall * 100.0,
        "context_switches": ctx_switches,
        "context_switches_per_1k_tokens": (ctx_switches / tokens * 1000.0) if tokens else None,
        "cpu_seconds_per_token": (cpu_seconds / tokens) if tokens else None,
        "cpu_percent_per_tok_per_sec": (
            (cpu_seconds / wall * 100.0) / decode_tps if decode_tps else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--spinners", type=int, default=4)
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_directories()
    model = load_model_profile(Path(args.model))
    hardware = detect_hardware_profile(settings)
    server_path = settings.llama_cpp_dir / "llama-server.exe"
    requests = ab.build_fixed_requests(
        rounds=args.rounds, n_predict=args.n_predict, seed=1234, temperature=0.0, ignore_eos=False
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, Any] = {
        "schema": "cpu_bottleneck_metric_investigation.v1",
        "timestamp": timestamp,
        "physical_cores": hardware.cpu.physical_cores,
        "logical_cores": hardware.cpu.logical_cores,
        "spinners": args.spinners,
        "conditions": {},
    }
    for threads in (hardware.cpu.logical_cores, hardware.cpu.physical_cores):
        print(f"--- threads={threads} (spinners={args.spinners}) ---")
        result = _measure(
            model=model, server_path=server_path, settings=settings, threads=threads,
            requests=requests, spinners=args.spinners,
            log_path=Path(settings.logs_dir) / f"cpu_metric_investigation_t{threads}.log",
        )
        payload["conditions"][f"threads_{threads}"] = result
        print(json.dumps(result, indent=1, default=str))

    base = payload["conditions"][f"threads_{hardware.cpu.logical_cores}"]
    cand = payload["conditions"][f"threads_{hardware.cpu.physical_cores}"]
    if base.get("cpu_seconds_per_token") and cand.get("cpu_seconds_per_token"):
        payload["comparison"] = {
            "decode_change": (
                (cand["decode_tokens_per_sec"] - base["decode_tokens_per_sec"])
                / base["decode_tokens_per_sec"]
                if base.get("decode_tokens_per_sec") and cand.get("decode_tokens_per_sec")
                else None
            ),
            "process_cpu_percent_change": (
                (cand["process_cpu_percent"] - base["process_cpu_percent"])
                / base["process_cpu_percent"]
                if base.get("process_cpu_percent")
                else None
            ),
            "cpu_seconds_per_token_change": (
                (cand["cpu_seconds_per_token"] - base["cpu_seconds_per_token"])
                / base["cpu_seconds_per_token"]
            ),
            "context_switches_per_1k_tokens_change": (
                (cand["context_switches_per_1k_tokens"] - base["context_switches_per_1k_tokens"])
                / base["context_switches_per_1k_tokens"]
                if base.get("context_switches_per_1k_tokens")
                else None
            ),
        }
        print("comparison:", json.dumps(payload["comparison"], indent=1))

    out = ROOT / "artifacts" / "ab_effectiveness" / f"cpu_bottleneck_metric_{timestamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
