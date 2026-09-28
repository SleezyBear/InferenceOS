"""Phase 12B — real-machine control-path validation for `runtime.reduce_threads`.

Proves, on the real resident llama-server, the chain:

    observed effective threads
        -> existing candidate preparation (real effective value feeds the action)
        -> existing runtime.reduce_threads intervention
        -> controlled resident reload
        -> effective target verified
        -> exact restoration

This does NOT claim any performance improvement and does NOT retain a production
optimization. It only exercises the existing control path with authoritative runtime
state. No thread search exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inferenceos.config import Settings  # noqa: E402
from inferenceos.contracts.enums import BackendType  # noqa: E402
from inferenceos.contracts.experiment import RuntimeConfiguration  # noqa: E402
from inferenceos.diagnosis.detector import BottleneckDetector  # noqa: E402
from inferenceos.hardware.profiler import detect_hardware_profile  # noqa: E402
from inferenceos.models.loader import load_model_profile  # noqa: E402
from inferenceos.optimization.actions import build_default_registry  # noqa: E402
from inferenceos.optimization.generator import generate_candidates  # noqa: E402
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter  # noqa: E402
from inferenceos.runtime.resident_workload import ResidentActiveWorkload  # noqa: E402
from inferenceos.runtime.workload import RuntimeWorkloadExecutor  # noqa: E402
from tests.diagnosis_fixtures import make_report, make_run  # noqa: E402

THREAD_ACTION = "runtime.reduce_threads"


def _cpu_report(*, requested: int | None, effective: int | None, hardware: Any):
    """A controlled CPU-diagnosing report carrying the observed effective threads."""
    runs = []
    for _ in range(2):
        for condition, decode, cpu in (("control", 20.0, 600.0), ("contended", 10.0, 500.0)):
            run = make_run(
                condition=condition,
                backend=BackendType.CPU,
                gpu_offload_layers=0,
                gpu_offload_total=28,
                metrics={
                    "decode_tokens_per_sec": decode,
                    "total_latency_s": 2.0,
                    "process_cpu_percent": cpu,
                    "gpu_utilization_percent": 5.0,
                    "peak_rss_bytes": 1_300_000_000,
                    "available_ram_bytes": float(8 * 1024**3),
                    "total_ram_bytes": float(16 * 1024**3),
                },
            )
            config = run.context.config.model_copy(
                update={"requested_threads": requested, "effective_threads": effective}
            )
            runs.append(
                run.model_copy(
                    update={"context": run.context.model_copy(update={"config": config})}
                )
            )
    report = make_report(runs)
    return report.model_copy(update={"hardware": hardware})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    settings = Settings()
    settings.ensure_directories()
    model = load_model_profile(Path(args.model))
    hardware = detect_hardware_profile(settings)
    physical = hardware.cpu.physical_cores
    registry = build_default_registry()

    requested = 12
    configuration = RuntimeConfiguration(
        context_length=4096, gpu_offload_layers=8, threads=requested
    )
    workload = ResidentActiveWorkload(
        settings=settings,
        model=model,
        adapter=LlamaCppResidentAdapter(settings=settings),
        workload_id="short-short",
        configuration=configuration,
    )
    record: dict[str, Any] = {
        "schema": "reduce_threads_control_path.v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "physical_cores": physical,
        "requested_threads": requested,
    }
    try:
        workload.start()
        effective_before = workload.effective_threads()
        record["effective_threads_before"] = effective_before

        # Existing candidate preparation using the *observed* effective value.
        report = _cpu_report(requested=requested, effective=effective_before, hardware=hardware)
        diagnosis = BottleneckDetector().detect(report, source_report_path="<phase12b-real>")
        plan = generate_candidates(diagnosis, report=report)
        thread_candidates = [c for c in plan.candidates if c.action_id == THREAD_ACTION]
        candidate = thread_candidates[0] if thread_candidates else None
        target = candidate.parameters.get("threads") if candidate is not None else None
        record["diagnosis"] = {
            "primary": diagnosis.primary.kind.value if diagnosis.primary is not None else None,
            "meaningful": diagnosis.meaningful_opportunity,
        }
        record["prepared_candidate"] = (
            {
                "eligibility": candidate.eligibility.value,
                "executable": candidate.executable,
                "parameters": dict(candidate.parameters),
                "no_change_reason": candidate.rejection_reason,
            }
            if candidate is not None
            else None
        )
        record["prepared_target_threads"] = target
        print(
            f"physical={physical} requested={requested} effective_before={effective_before} "
            f"prepared_target={target}"
        )

        executor = RuntimeWorkloadExecutor(workload, backend_name=workload.descriptor().backend_name)
        if target is None:
            record["applied"] = False
            record["reason"] = "no target prepared (NO_CHANGE or not eligible)"
            print("no target prepared; not applying")
        else:
            workload.pause()
            application = executor.apply(registry.get(THREAD_ACTION), {"threads": int(target)})
            workload.resume()
            effective_after = workload.effective_threads()
            record["application"] = application.model_dump(mode="json")
            record["effective_threads_after_apply"] = effective_after
            print(f"applied={application.applied} effective_after={effective_after}")

            # Exact restoration.
            workload.pause()
            restoration = executor.restore(configuration)
            workload.resume()
            effective_restored = workload.effective_threads()
            record["restoration"] = restoration.model_dump(mode="json")
            record["restoration_target_threads"] = configuration.threads
            record["effective_threads_after_restoration"] = effective_restored
            exact_restore = workload.configuration() == configuration
            record["restored_configuration_equals_initial"] = exact_restore
            print(
                f"restored={restoration.status.value} effective_after_restore={effective_restored} "
                f"exact={exact_restore}"
            )
    finally:
        try:
            workload.stop()
        except Exception:  # noqa: BLE001
            pass

    import psutil

    time.sleep(2.0)
    record["orphan_check"] = {
        "llama_server_processes": [
            p.pid
            for p in psutil.process_iter(["name"])
            if (p.info.get("name") or "").lower() == "llama-server.exe"
        ]
    }
    print("orphan_check:", record["orphan_check"])

    out = (
        Path(args.out)
        if args.out
        else ROOT / "artifacts" / "ab_effectiveness" / f"reduce_threads_control_path_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
