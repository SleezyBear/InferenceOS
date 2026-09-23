"""InferenceOS command-line interface.

Phase 0's human-verifiable evidence command is ``inferenceos verify-phase0``: it
loads a real model, runs one prompt through the real Vulkan-enabled llama.cpp
binary, and prints (a) the generated text, (b) the backend/device that executed it,
and (c) GPU utilization observed via Windows GPU Engine counters while the call ran.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from inferenceos import __version__
from inferenceos.backends.llama_cpp_toolchain import (
    backend_type_from_name,
    find_cli,
    parse_device_info,
    parse_generated_text,
    parse_timings,
    run_cli,
)
from inferenceos.config import Settings
from inferenceos.contracts.enums import BackendType, MeasurementMode, ObservationSource
from inferenceos.contracts.observation import Observation, RunMetrics
from inferenceos.logging_config import get_logger, setup_logging
from inferenceos.selftest import run_reasoning_selftest, verify_contract_roundtrip
from inferenceos.telemetry.gpu import GpuUtilizationSampler, GpuUtilizationSummary

logger = get_logger("cli")

DEFAULT_PHASE0_PROMPT = "Explain in one sentence why the sky appears blue."


def _find_model(models_dir: Path, override: str | None) -> Path:
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(f"model not found: {path}")
        return path
    candidates = sorted(models_dir.glob("*.gguf"))
    if not candidates:
        raise FileNotFoundError(
            f"no .gguf model found in {models_dir}; pass --model"
        )
    return candidates[0]


def _cmd_version(args: argparse.Namespace) -> int:
    print(f"InferenceOS {__version__}")
    return 0


def _cmd_contracts_check(args: argparse.Namespace) -> int:
    passed = verify_contract_roundtrip()
    print(f"Contract round-trip: PASS ({len(passed)} contracts)")
    for name in passed:
        print(f"  - {name}")
    return 0


def _cmd_reasoning_selftest(args: argparse.Namespace) -> int:
    response = run_reasoning_selftest()
    print(
        "ReasoningEngine round-trip: PASS "
        f"(engine={response.engine.value}, status={response.status.value})"
    )
    print(f"  conclusion: {response.conclusion}")
    if response.bottleneck is not None:
        print(
            "  bottleneck: "
            f"{response.bottleneck.kind.value} "
            f"(meaningful={response.bottleneck.is_meaningful})"
        )
    return 0


def _cmd_verify_phase0(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings.log_level, log_file=settings.logs_dir / "phase0.log")

    cli = find_cli(settings.llama_cpp_dir)
    model = _find_model(settings.models_dir, args.model)
    prompt = args.prompt

    print("=== InferenceOS Phase 0 Verification ===")
    print(f"timestamp : {datetime.now(timezone.utc).isoformat()}")
    print(f"host      : {socket.gethostname()}")
    print(f"llama-cli : {cli}")
    print(f"model     : {model} ({model.stat().st_size / 1024**2:.1f} MiB)")
    print(f"request   : -ngl {args.gpu_layers} -n {args.n_predict} -s {args.seed} --temp 0 -st")
    print()

    passed = verify_contract_roundtrip()
    reasoning = run_reasoning_selftest()
    print(f"[contracts] {len(passed)} contract types round-trip JSON: PASS")
    print(f"[contracts] stub ReasoningEngine round-trip: PASS (status={reasoning.status.value})")
    print()

    sampler = GpuUtilizationSampler(interval_s=settings.gpu_sample_interval_s)
    sampler.start()
    result = None
    summary: GpuUtilizationSummary | None = None
    try:
        result = run_cli(
            cli,
            model,
            prompt,
            n_predict=args.n_predict,
            seed=args.seed,
            temperature=0.0,
            n_gpu_layers=args.gpu_layers,
            threads=settings.default_threads,
            verbose=True,
            timeout_s=args.timeout,
        )
    finally:
        pid = result.pid if result is not None else None
        summary = sampler.stop(target_pid=pid)

    assert result is not None and summary is not None  # for type narrowing

    device = parse_device_info(result.stderr)
    text = parse_generated_text(result.stdout)
    prompt_tps, gen_tps = parse_timings(result.stdout)
    backend = backend_type_from_name(device.backend) if device else None

    print("--- Generated text (real llama.cpp output) ---")
    print(text if text else "<none>")
    print()
    print("--- Backend / device ---")
    if device is not None:
        print(f"backend          : {device.backend}")
        print(f"device           : {device.device}")
        if device.layers_offloaded is not None and device.layers_total is not None:
            print(f"layers offloaded : {device.layers_offloaded}/{device.layers_total} to GPU")
        if device.vram_total_bytes is not None:
            print(
                "VRAM             : "
                f"{device.vram_total_bytes / 1024**2:.0f} MiB total, "
                f"{(device.vram_free_bytes or 0) / 1024**2:.0f} MiB free"
            )
    else:
        print("no device information parsed from llama.cpp logs")

    print()
    print("--- Performance (real) ---")
    print(f"prompt eval     : {prompt_tps} tokens/s")
    print(f"generation      : {gen_tps} tokens/s")
    print(f"wall clock      : {result.duration_s:.2f} s")
    print()
    print("--- GPU utilization (observed via Windows GPU Engine counters) ---")
    print(f"samples captured: {len(summary.samples)} over {summary.duration_s:.2f} s")
    print(f"max overall     : {summary.max_overall_percent:.1f}% (engine: {summary.max_engine})")
    print(f"max for inference pid {summary.target_pid}: {summary.max_target_percent:.1f}%")
    if summary.engine_max_percent:
        engines = ", ".join(
            f"{name}={value:.1f}%" for name, value in sorted(summary.engine_max_percent.items())
        )
        print(f"per-engine max  : {engines}")

    offload_ok = (
        device is not None
        and device.layers_offloaded is not None
        and device.layers_total is not None
        and device.layers_offloaded == device.layers_total
        and device.layers_offloaded > 0
    )
    checks = [
        ("real llama.cpp inference produced non-empty text", bool(text)),
        ("Vulkan GPU offload confirmed (all layers)", offload_ok),
        ("backend is Vulkan", backend == BackendType.VULKAN),
        ("device is AMD Radeon", bool(device and "radeon" in device.device.lower())),
        ("GPU utilization > 0 observed during call", summary.max_overall_percent > 0),
    ]

    print()
    print("--- Acceptance ---")
    for name, flag in checks:
        print(f"[{'PASS' if flag else 'FAIL'}] {name}")
    ok = all(flag for _, flag in checks)

    observation = Observation(
        observation_id=f"phase0-{int(time.time())}",
        captured_at=datetime.now(timezone.utc),
        source=ObservationSource.DIAGNOSTIC,
        mode=MeasurementMode.COLD,
        backend=backend,
        device=device.device if device else None,
        model_ref=model.name,
        workload_ref="phase0-smoke",
        metrics=RunMetrics(
            prompt_tokens_per_sec=prompt_tps,
            decode_tokens_per_sec=gen_tps,
            total_latency_s=result.duration_s,
            generated_tokens=args.n_predict,
            gpu_utilization_percent=summary.max_overall_percent,
        ),
        generated_text=text,
    )

    report = {
        "phase": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "llama_cli": str(cli),
        "model": str(model),
        "command": result.command,
        "device": asdict(device) if device is not None else None,
        "generated_text": text,
        "timings": {
            "prompt_tokens_per_sec": prompt_tps,
            "decode_tokens_per_sec": gen_tps,
            "wall_seconds": result.duration_s,
        },
        "gpu": {
            "sample_count": len(summary.samples),
            "duration_s": summary.duration_s,
            "max_overall_percent": summary.max_overall_percent,
            "max_engine": summary.max_engine,
            "max_target_percent": summary.max_target_percent,
            "target_pid": summary.target_pid,
            "engine_max_percent": summary.engine_max_percent,
            "samples": [asdict(sample) for sample in summary.samples],
        },
        "observation": json.loads(observation.model_dump_json()),
        "contracts": passed,
        "acceptance": {name: flag for name, flag in checks},
        "result": "PASS" if ok else "FAIL",
    }
    report_path = settings.reports_dir / (
        f"phase0_verification_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"=== RESULT: PHASE 0 ACCEPTANCE {'PASS' if ok else 'FAIL'} ===")
    print(f"report: {report_path}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferenceos",
        description="Autonomous adaptive local LLM optimization system",
    )
    parser.add_argument("--version", action="version", version=f"InferenceOS {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version_parser = subparsers.add_parser("version", help="print the version")
    version_parser.set_defaults(func=_cmd_version)

    contracts_parser = subparsers.add_parser(
        "contracts-check", help="instantiate and JSON round-trip every contract"
    )
    contracts_parser.set_defaults(func=_cmd_contracts_check)

    reasoning_parser = subparsers.add_parser(
        "reasoning-selftest", help="round-trip the stub ReasoningEngine interface"
    )
    reasoning_parser.set_defaults(func=_cmd_reasoning_selftest)

    verify_parser = subparsers.add_parser(
        "verify-phase0",
        help="run the real Phase 0 acceptance (llama.cpp + Vulkan + GPU utilization)",
    )
    verify_parser.add_argument("--model", default=None, help="path to a .gguf model")
    verify_parser.add_argument("--prompt", default=DEFAULT_PHASE0_PROMPT)
    verify_parser.add_argument("--n-predict", type=int, default=64)
    verify_parser.add_argument("--gpu-layers", type=int, default=99)
    verify_parser.add_argument("--seed", type=int, default=1234)
    verify_parser.add_argument("--timeout", type=float, default=900.0)
    verify_parser.set_defaults(func=_cmd_verify_phase0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result: int = args.func(args)
    except (FileNotFoundError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return result


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
