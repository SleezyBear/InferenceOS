"""Phase 12A — CPU thread surface controlled investigation (investigation only).

Answers whether the existing ``runtime.reduce_threads`` intervention can pass the
Phase 12 readiness gate. It does NOT implement a thread search, does not broaden the
intervention, and does not modify production architecture.

What it does:
  1. probes whether the *effective* runtime thread count is authoritatively observable
     (server log / props / OS thread count);
  2. runs a bounded, reversible controlled CPU-contention experiment against the real
     resident llama-server workload and records the production diagnosis under both
     CPU-only (offload not configured) and GPU-offload-configured conditions;
  3. prepares the existing ``runtime.reduce_threads`` candidate against the CPU
     diagnosis (no new search);
  4. experimentally exercises the existing control seam (apply -> measure -> restore)
     and verifies exact restoration.

Run:
    .\\.venv\\Scripts\\python.exe scripts\\validate_cpu_thread_surface.py \\
        --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_ab_effectiveness as ab  # noqa: E402

from inferenceos.config import Settings  # noqa: E402
from inferenceos.contracts.enums import (  # noqa: E402
    BackendType,
    ObservationSource,
    MeasurementMode,
)
from inferenceos.contracts.experiment import RuntimeConfiguration  # noqa: E402
from inferenceos.contracts.observation import Observation, RunMetrics  # noqa: E402
from inferenceos.contracts.profiling import (  # noqa: E402
    BaselineReport,
    ProfilingConfig,
    ProfilingRun,
    RunContext,
    RunIdentity,
)
from inferenceos.diagnosis.detector import BottleneckDetector  # noqa: E402
from inferenceos.external.controlled import launch_cpu_spinner  # noqa: E402
from inferenceos.hardware.profiler import detect_hardware_profile  # noqa: E402
from inferenceos.interfaces.backend_adapter import (  # noqa: E402
    GenerationRequest,
    RuntimeParams,
)
from inferenceos.models.loader import load_model_profile  # noqa: E402
from inferenceos.optimization.actions import build_default_registry  # noqa: E402
from inferenceos.optimization.context import build_optimization_context  # noqa: E402
from inferenceos.optimization.generator import generate_candidates  # noqa: E402
from inferenceos.optimization.policy import build_policy  # noqa: E402
from inferenceos.backends.llama_cpp_resident_adapter import (  # noqa: E402
    LlamaCppResidentAdapter,
)
from inferenceos.runtime.resident_workload import ResidentActiveWorkload  # noqa: E402
from inferenceos.runtime.telemetry import SystemTelemetrySampler  # noqa: E402
from inferenceos.runtime.workload import (  # noqa: E402
    ActiveWorkload,
    RuntimeWorkloadExecutor,
)
from inferenceos.contracts.runtime import WorkloadDescriptor, WorkloadSnapshot  # noqa: E402

_NOW = datetime.now(timezone.utc)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _parse_telemetry(path: Path) -> dict[str, Any]:
    cpu: list[float] = []
    gpu: list[float] = []
    syscpu: list[float] = []
    faults: list[float] = []
    if not path.is_file():
        return {"target_cpu_mean": None, "gpu_engine_mean": None, "system_cpu_mean": None,
                "hard_faults_mean": None, "samples": 0}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload.get("target_cpu_percent"), (int, float)):
            cpu.append(float(payload["target_cpu_percent"]))
        if isinstance(payload.get("gpu_engine_max_percent"), (int, float)):
            gpu.append(float(payload["gpu_engine_max_percent"]))
        if isinstance(payload.get("cpu_percent_total"), (int, float)):
            syscpu.append(float(payload["cpu_percent_total"]))
        if isinstance(payload.get("hard_page_faults_per_sec"), (int, float)):
            faults.append(float(payload["hard_page_faults_per_sec"]))
    return {
        "target_cpu_mean": _mean(cpu),
        "gpu_engine_mean": _mean(gpu),
        "gpu_engine_max": max(gpu) if gpu else None,
        "system_cpu_mean": _mean(syscpu),
        "hard_faults_mean": _mean(faults),
        "samples": len(cpu),
    }


def _server_log_threads(path: Path) -> list[int]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [int(value) for value in re.findall(r"n_threads\s*=\s*([0-9]+)", text)]


def _build_run(
    *,
    model: Any,
    workload: Any,
    hardware: Any,
    outcome: ab.RequestOutcome,
    condition: str,
    gpu_layers: int,
    total_layers: int,
    threads: int,
    process_cpu: float | None,
    system_cpu: float | None,
    gpu_util: float | None,
    faults: float | None,
    index: int,
) -> ProfilingRun:
    backend_type = BackendType.VULKAN if gpu_layers > 0 else BackendType.CPU
    config = ProfilingConfig(
        backend_name="llama.cpp (resident)",
        backend_type=backend_type,
        gpu_layers_offloaded=gpu_layers,
        gpu_layers_total=total_layers,
        threads=threads,
        context_length=4096,
        extra={"condition": condition},
    )
    metrics = RunMetrics(
        decode_tokens_per_sec=outcome.decode_tokens_per_sec,
        total_latency_s=outcome.total_latency_s,
        ttft_s=outcome.ttft_s,
        prompt_tokens_per_sec=None,
        process_cpu_percent=process_cpu,
        system_cpu_percent=system_cpu,
        gpu_utilization_percent=gpu_util,
        hard_page_faults_per_sec=faults,
        peak_rss_bytes=outcome.peak_rss_bytes,
        generated_tokens=outcome.generated_tokens,
    )
    run_id = f"cpu-surface-{condition}-{index:03d}-{uuid.uuid4().hex[:6]}"
    return ProfilingRun(
        identity=RunIdentity(
            run_id=run_id,
            profiler_version="phase12a.cpu-surface",
            sequence=index,
            repetition=0,
            mode=MeasurementMode.WARM,
            warmup=False,
            started_at=_NOW,
            finished_at=_NOW,
        ),
        context=RunContext(hardware=hardware, model=model, workload=workload, config=config),
        observation=Observation(
            observation_id=run_id,
            captured_at=_NOW,
            source=ObservationSource.HARNESS,
            mode=MeasurementMode.WARM,
            backend=backend_type,
            model_ref=model.name,
            workload_ref=workload.workload_id,
            metrics=metrics,
        ),
        metric_quality={},
    )


class _ProbeWorkload(ActiveWorkload):
    """Minimal ActiveWorkload stub so the production telemetry sampler can observe a
    directly-started resident llama-server (target pid + last request metrics)."""

    def __init__(self, *, pid: int, model: Any, workload: Any, configuration: Any) -> None:
        self._pid = pid
        self._model = model
        self._workload = workload
        self._configuration = configuration
        self._metrics: dict[str, float] = {}
        self._requests = 0

    def descriptor(self) -> Any:
        return WorkloadDescriptor(
            workload_id=self._workload.workload_id,
            model_ref=self._model.name,
            model_path=self._model.path,
            backend_name="llama.cpp (resident-probe)",
            configuration=self._configuration,
            context_length=self._configuration.context_length,
        )

    def snapshot(self) -> Any:
        return WorkloadSnapshot(running=True, requests_served=self._requests)

    def configuration(self) -> Any:
        return self._configuration

    def set_configuration(self, configuration: Any) -> None:
        self._configuration = configuration

    def pause(self) -> bool:
        return True

    def resume(self) -> bool:
        return True

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def model_profile(self) -> Any:
        return self._model

    def workload_profile(self) -> Any:
        return self._workload

    def last_metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def target_pid(self) -> int:
        return self._pid

    def set_last_metrics(self, metrics: dict[str, float], *, requests: int) -> None:
        self._metrics = dict(metrics)
        self._requests = requests


def _mean_metric(samples: list[dict[str, float]], key: str) -> float | None:
    values = [s[key] for s in samples if isinstance(s.get(key), (int, float))]
    return statistics.fmean(values) if values else None


def measure_condition(
    *,
    requests: list[ab.FixedRequest],
    model: Any,
    workload: Any,
    hardware: Any,
    server_path: Path,
    settings: Settings,
    scenario: str,
    condition: str,
    gpu_layers: int,
    threads: int,
    total_layers: int,
    contention_procs: int,
    spinner_seconds: float,
) -> tuple[list[ProfilingRun], dict[str, Any], dict[str, Any]]:
    label = f"{scenario}_{condition}"
    log_path = Path(settings.logs_dir) / f"cpu_surface_{label}_server.log"
    launched = ab.start_unmanaged_server(
        server=server_path,
        model_path=Path(model.path),
        host=settings.resident_host,
        context_length=4096,
        gpu_layers=gpu_layers,
        threads=threads if threads else None,
        ignore_eos=False,
        log_path=log_path,
        readiness_timeout_s=settings.resident_readiness_timeout_s,
    )
    probe = _ProbeWorkload(
        pid=launched.pid,
        model=model,
        workload=workload,
        configuration=RuntimeConfiguration(context_length=4096, gpu_offload_layers=gpu_layers, threads=threads),
    )
    sampler = SystemTelemetrySampler(
        settings=settings, include_gpu=True, gpu_scan_interval_s=1.0
    )
    samples: list[dict[str, float]] = []
    stop_flag = threading.Event()

    def _sample_loop() -> None:
        sequence = 0
        while not stop_flag.is_set():
            try:
                observation = sampler.sample(probe, sequence=sequence, condition=condition)
                samples.append(dict(observation.metrics))
            except Exception:  # noqa: BLE001 - sampling must not crash the experiment
                pass
            sequence += 1
            stop_flag.wait(0.4)

    spinners: list[Any] = []
    outcomes: list[ab.RequestOutcome] = []
    sample_thread = threading.Thread(target=_sample_loop, daemon=True)
    try:
        time.sleep(4.0)  # server warm-up / settle before any sampling
        sample_thread.start()
        if contention_procs > 0:
            spinners = [launch_cpu_spinner(spinner_seconds) for _ in range(contention_procs)]
            for ctx in spinners:
                ctx.__enter__()
            time.sleep(4.0)  # contention established
        for request in requests:
            outcome = ab.run_unmanaged_request(
                test=condition,
                request=request,
                host=launched.host,
                port=launched.port,
                server_pid=launched.pid,
                timeout_s=settings.resident_request_timeout_s,
            )
            outcomes.append(outcome)
            probe.set_last_metrics(
                {
                    key: float(value)
                    for key, value in (
                        ("decode_tokens_per_sec", outcome.decode_tokens_per_sec),
                        ("total_latency_s", outcome.total_latency_s),
                        ("ttft_s", outcome.ttft_s),
                    )
                    if value is not None
                },
                requests=len(outcomes),
            )
    finally:
        for ctx in reversed(spinners):
            try:
                ctx.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        stop_flag.set()
        sample_thread.join(timeout=2.0)
        try:
            sampler.close()
        except Exception:  # noqa: BLE001
            pass
        stop_report = ab.stop_unmanaged_server(launched)
    telemetry_stats = {
        "process_cpu_mean": _mean_metric(samples, "process_cpu_percent"),
        "gpu_util_mean": _mean_metric(samples, "gpu_utilization_percent"),
        "system_cpu_mean": _mean_metric(samples, "system_cpu_percent"),
        "hard_faults_mean": _mean_metric(samples, "hard_page_faults_per_sec"),
        "samples": len(samples),
        "gpu_samples": sum(1 for s in samples if s.get("gpu_utilization_percent") is not None),
    }
    log_threads = _server_log_threads(log_path)
    runs = [
        _build_run(
            model=model, workload=workload, hardware=hardware, outcome=outcome,
            condition=condition, gpu_layers=gpu_layers, total_layers=total_layers,
            threads=threads, process_cpu=telemetry_stats["process_cpu_mean"],
            system_cpu=telemetry_stats["system_cpu_mean"],
            gpu_util=telemetry_stats["gpu_util_mean"], faults=telemetry_stats["hard_faults_mean"],
            index=idx,
        )
        for idx, outcome in enumerate(outcomes)
    ]
    meta = {
        "scenario": scenario,
        "condition": condition,
        "gpu_layers": gpu_layers,
        "requested_threads": threads or None,
        "server_log_n_threads": log_threads,
        "contention_processes": contention_procs,
        "telemetry": telemetry_stats,
        "stop": stop_report,
        "decode_mean": _mean([r.observation.metrics.decode_tokens_per_sec for r in runs if r.observation.metrics.decode_tokens_per_sec]),
    }
    return runs, meta, telemetry_stats


def _diagnosis_summary(diagnosis: Any) -> dict[str, Any]:
    return {
        "meaningful_opportunity": diagnosis.meaningful_opportunity,
        "insufficient_evidence": diagnosis.insufficient_evidence,
        "escalation_required": diagnosis.escalation_required,
        "primary": (
            diagnosis.primary.kind.value
            if diagnosis.primary is not None
            else None
        ),
        "primary_confidence": diagnosis.primary.confidence if diagnosis.primary is not None else None,
        "findings": sorted({f.kind.value for f in getattr(diagnosis, "candidates", [])}),
        "observations": sorted({o.kind.value for o in diagnosis.observations}),
        "no_opportunity_reason": diagnosis.no_opportunity_reason,
        "summary": diagnosis.summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_directories()
    model = load_model_profile(Path(args.model))
    hardware = detect_hardware_profile(settings)
    physical = hardware.cpu.physical_cores
    logical = hardware.cpu.logical_cores
    server_path = settings.llama_cpp_dir / "llama-server.exe"
    total_layers = model.n_layers
    requests = ab.build_fixed_requests(
        rounds=args.rounds, n_predict=args.n_predict, seed=args.seed,
        temperature=0.0, ignore_eos=False,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload: dict[str, Any] = {
        "schema": "cpu_thread_surface_investigation.v1",
        "timestamp": timestamp,
        "hardware": {
            "physical_cores": physical,
            "logical_cores": logical,
            "cpu": hardware.cpu.name,
        },
        "model": {"name": model.name, "path": model.path, "n_layers": model.n_layers},
        "request_count": len(requests),
        "requested_oversubscribed_threads": logical,
    }
    _workload_profile = requests[0].to_benchmark_workload().to_workload_profile()
    print(f"physical={physical} logical={logical} layers={total_layers} requests={len(requests)}")

    # ------------------------------------------------------------------ #
    # 1. Effective-thread observability probe                             #
    # ------------------------------------------------------------------ #
    print("--- effective-thread observability ---")
    observability: list[dict[str, Any]] = []
    for probe_threads in (logical, None):
        log_path = Path(settings.logs_dir) / f"cpu_surface_probe_{probe_threads}.log"
        launched = ab.start_unmanaged_server(
            server=server_path, model_path=Path(model.path), host=settings.resident_host,
            context_length=4096, gpu_layers=8, threads=probe_threads, ignore_eos=False,
            log_path=log_path, readiness_timeout_s=settings.resident_readiness_timeout_s,
        )
        try:
            import urllib.request

            props: dict[str, Any] = {}
            try:
                with urllib.request.urlopen(
                    f"http://{launched.host}:{launched.port}/props", timeout=10
                ) as response:
                    props = json.loads(response.read().decode("utf-8", "replace"))
            except Exception as exc:  # noqa: BLE001
                props = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                import psutil

                os_threads = psutil.Process(launched.pid).num_threads()
            except Exception:  # noqa: BLE001
                os_threads = None
            log_threads = _server_log_threads(log_path)
            props_threads = None
            if isinstance(props, dict):
                default = props.get("default_generation_settings")
                if isinstance(default, dict):
                    props_threads = default.get("n_threads")
            observability.append(
                {
                    "requested_threads": probe_threads,
                    "server_log_n_threads": log_threads,
                    "props_n_threads": props_threads,
                    "os_process_num_threads": os_threads,
                    "props_keys_sample": sorted(list(props.keys()))[:12] if isinstance(props, dict) else [],
                }
            )
            print(
                f"requested={probe_threads} log_n_threads={log_threads} "
                f"props_n_threads={props_threads} os_threads={os_threads}"
            )
        finally:
            ab.stop_unmanaged_server(launched)
    payload["effective_thread_observability"] = observability

    # ------------------------------------------------------------------ #
    # 2. Controlled CPU contention diagnosis (CPU-only)                   #
    # ------------------------------------------------------------------ #
    print("--- controlled CPU contention: CPU-only (offload not configured) ---")
    cpu_control, cpu_control_meta, _ = measure_condition(
        requests=requests, model=model, workload=_workload_profile,
        hardware=hardware, server_path=server_path, settings=settings, scenario="cpu_only", condition="control",
        gpu_layers=0, threads=logical, total_layers=total_layers, contention_procs=0,
        spinner_seconds=60.0,
    )
    cpu_pressured, cpu_pressured_meta, _ = measure_condition(
        requests=requests, model=model, workload=_workload_profile,
        hardware=hardware, server_path=server_path, settings=settings, scenario="cpu_only", condition="contended",
        gpu_layers=0, threads=logical, total_layers=total_layers,
        contention_procs=max(physical, 4), spinner_seconds=180.0,
    )
    cpu_report = BaselineReport(
        profiler_version="phase12a.cpu-surface",
        suite_version="phase12a",
        generated_at=_NOW,
        hardware=hardware,
        backend=cpu_control[0].context.config if cpu_control else ProfilingConfig(),
        workloads=[requests[0].to_benchmark_workload().to_workload_profile()],
        runs=cpu_control + cpu_pressured,
        groups=[],
        cold_warm_semantics="controlled CPU contention (CPU-only backend)",
        resident_warm_supported=True,
        limitations=["investigation-only controlled condition"],
        notes="phase12a cpu-only contention",
    )
    cpu_diagnosis = BottleneckDetector().detect(cpu_report, source_report_path="<phase12a-cpu-only>")
    payload["cpu_only"] = {
        "control": cpu_control_meta,
        "contended": cpu_pressured_meta,
        "diagnosis": _diagnosis_summary(cpu_diagnosis),
    }
    print("  cpu-only contended diagnosis:", _diagnosis_summary(cpu_diagnosis)["primary"],
          "meaningful=", cpu_diagnosis.meaningful_opportunity)

    # ------------------------------------------------------------------ #
    # 3. Controlled CPU contention diagnosis (GPU offload configured)     #
    # ------------------------------------------------------------------ #
    print("--- controlled CPU contention: GPU offload configured (-ngl 8) ---")
    gpu_control, gpu_control_meta, _ = measure_condition(
        requests=requests, model=model, workload=_workload_profile,
        hardware=hardware, server_path=server_path, settings=settings, scenario="offload", condition="control",
        gpu_layers=8, threads=logical, total_layers=total_layers, contention_procs=0,
        spinner_seconds=60.0,
    )
    gpu_pressured, gpu_pressured_meta, _ = measure_condition(
        requests=requests, model=model, workload=_workload_profile,
        hardware=hardware, server_path=server_path, settings=settings, scenario="offload", condition="contended",
        gpu_layers=8, threads=logical, total_layers=total_layers,
        contention_procs=max(physical, 4), spinner_seconds=180.0,
    )
    gpu_report = BaselineReport(
        profiler_version="phase12a.cpu-surface",
        suite_version="phase12a",
        generated_at=_NOW,
        hardware=hardware,
        backend=gpu_control[0].context.config if gpu_control else ProfilingConfig(),
        workloads=[requests[0].to_benchmark_workload().to_workload_profile()],
        runs=gpu_control + gpu_pressured,
        groups=[],
        cold_warm_semantics="controlled CPU contention (offload configured)",
        resident_warm_supported=True,
        limitations=["investigation-only controlled condition"],
        notes="phase12a offload-configured contention",
    )
    gpu_diagnosis = BottleneckDetector().detect(gpu_report, source_report_path="<phase12a-offload>")
    payload["offload_configured"] = {
        "control": gpu_control_meta,
        "contended": gpu_pressured_meta,
        "diagnosis": _diagnosis_summary(gpu_diagnosis),
    }
    print("  offload-configured contended diagnosis:", _diagnosis_summary(gpu_diagnosis)["primary"],
          "meaningful=", gpu_diagnosis.meaningful_opportunity)

    # ------------------------------------------------------------------ #
    # 4. Reduce_threads candidate preparation (existing strategy)         #
    # ------------------------------------------------------------------ #
    print("--- existing runtime.reduce_threads candidate preparation ---")
    policy = build_policy(settings)
    registry = build_default_registry()
    candidate_info: dict[str, Any] = {}
    plan = generate_candidates(cpu_diagnosis, report=cpu_report, registry=registry, policy=policy)
    reduce_candidate = next(
        (c for c in plan.candidates if c.action_id == "runtime.reduce_threads"), None
    )
    if reduce_candidate is not None:
        candidate_info = {
            "candidate_id": reduce_candidate.candidate_id,
            "parameters": dict(reduce_candidate.parameters),
            "eligibility": reduce_candidate.eligibility.value,
            "executable": reduce_candidate.executable,
            "preconditions": [
                {"check_id": p.check_id, "satisfied": p.satisfied, "description": p.description}
                for p in reduce_candidate.preconditions
            ],
        }
    payload["reduce_threads_candidate"] = candidate_info
    print("  reduce_threads candidate:", candidate_info if candidate_info else "not generated")

    # ------------------------------------------------------------------ #
    # 5. Exercise the existing control seam (apply -> measure -> restore) #
    # ------------------------------------------------------------------ #
    print("--- exercise existing runtime.reduce_threads control seam ---")
    exercise: dict[str, Any] = {}
    workload = ResidentActiveWorkload(
        settings=settings,
        model=model,
        adapter=LlamaCppResidentAdapter(settings=settings, server=server_path),
        workload_id="short-short",
        configuration=RuntimeConfiguration(context_length=4096, gpu_offload_layers=8, threads=logical),
    )
    try:
        workload.start()
        adapter = workload.adapter
        initial = workload.configuration()
        before = _generate_once(adapter)
        executor = RuntimeWorkloadExecutor(workload, backend_name="llama.cpp (resident)")
        action = registry.get("runtime.reduce_threads")
        # Production semantics: drain/pause before mutating the load-time target config.
        workload.pause()
        application = executor.apply(action, {"threads": physical})
        param_application = getattr(adapter, "last_param_application", None)
        applied_threads = getattr(getattr(adapter, "params", None), "threads", None)
        workload.resume()
        after = _generate_once(adapter)
        # Restore the exact original configuration and verify by re-measurement.
        workload.pause()
        restoration = executor.restore(initial)
        restored = workload.configuration() == initial
        restored_threads = getattr(getattr(adapter, "params", None), "threads", None)
        workload.resume()
        exercise = {
            "requested_threads": logical,
            "target_threads": physical,
            "application": application.model_dump(mode="json"),
            "param_application": (
                param_application.model_dump(mode="json") if param_application is not None else None
            ),
            "applied_effective_adapter_threads": applied_threads,
            "before_decode": before,
            "after_decode": after,
            "restoration": restoration.model_dump(mode="json"),
            "restored_configuration_equals_initial": restored,
            "restored_effective_adapter_threads": restored_threads,
        }
        print("  application:", application.applied, "after decode:", after, "restored:", restored)
    except Exception as exc:  # noqa: BLE001
        exercise = {"error": f"{type(exc).__name__}: {exc}"}
        print("  control-seam exercise error:", exercise["error"])
    finally:
        try:
            workload.stop()
        except Exception:  # noqa: BLE001
            pass
    payload["control_seam_exercise"] = exercise

    # ------------------------------------------------------------------ #
    # 6. System recovery verification                                     #
    # ------------------------------------------------------------------ #
    import psutil

    time.sleep(2.0)
    payload["recovery"] = {
        "llama_server_processes": [p.pid for p in psutil.process_iter(["name"]) if (p.info.get("name") or "").lower() == "llama-server.exe"],
        "free_ram_bytes": int(psutil.virtual_memory().available),
    }
    print("  recovery:", payload["recovery"])

    out = Path(args.out) if args.out else ROOT / "artifacts" / "ab_effectiveness" / f"cpu_thread_surface_{timestamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")
    return 0


def _generate_once(adapter: Any) -> dict[str, Any]:
    result = adapter.generate(
        GenerationRequest(
            prompt="In one sentence, explain what a key value cache is.",
            n_predict=64,
            seed=1234,
            temperature=0.0,
        )
    )
    return {
        "decode_tokens_per_sec": result.decode_tokens_per_sec,
        "total_latency_s": result.total_s,
        "ttft_s": result.ttft_s,
        "generated_tokens": result.generated_tokens,
    }


if __name__ == "__main__":
    raise SystemExit(main())


