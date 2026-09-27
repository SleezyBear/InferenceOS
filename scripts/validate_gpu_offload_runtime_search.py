"""Real Windows validation: resource-aware GPU-offload search through the runtime.

This is a *bounded validation*, not the final A/B effectiveness experiment. It drives
the real ``RuntimeMonitor`` over a real resident ``llama-server``:

    real workload (with a genuine workload shift)
      -> real per-process/GPU telemetry
      -> GPU-offload opportunity (real metrics; bottleneck kind forced by a
         verification-only detector so the path is exercised on demand)
      -> resource-aware multi-point search against the live resident server
      -> measured selection or NO_CHANGE
      -> (if selected) existing approval -> checkpoint -> apply -> validate -> rollback

Run:
    .\\.venv\\Scripts\\python.exe scripts\\validate_gpu_offload_runtime_search.py \
        --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf \
        --context 4096 --gpu-layers 8 --duration 900 \
        --out reports\\gpu_offload_runtime_validation.json
"""

from __future__ import annotations

import argparse
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.models.loader import load_model_profile
from inferenceos.profiling.workloads import get_workload
from inferenceos.runtime.approval import (
    ApprovalProvider,
    AutoApproveApprovalProvider,
    AutoRejectApprovalProvider,
)
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.notification import CollectingNotificationSink
from inferenceos.runtime.policy import build_runtime_budget, build_runtime_policy
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.runtime.telemetry import SystemTelemetrySampler
from inferenceos.runtime.verification import ControlledGpuUnderutilizationDetector


class ShiftingResidentWorkload(ResidentActiveWorkload):
    """Resident workload that performs one genuine workload shift after N requests.

    This creates a real change in decode/latency/GPU behaviour (a longer prompt), which
    is what a conservative runtime monitor is designed to react to. It adds no fake
    telemetry: the shift is real serving work.
    """

    def __init__(
        self,
        *,
        shift_after_requests: int,
        shifted_workload_id: str,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._shift_after = max(int(shift_after_requests), 1)
        self._shifted = get_workload(shifted_workload_id)
        self._shifted_done = False
        self._shift_lock = threading.Lock()

    def _serve_one(self) -> None:
        super()._serve_one()
        with self._shift_lock:
            if not self._shifted_done and self._completed >= self._shift_after:
                self._shifted_done = True
                with self._lock:
                    self._workload = self._shifted


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model path")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=8)
    parser.add_argument("--workload", default="short-short")
    parser.add_argument("--shift-workload", default="long-prefill")
    parser.add_argument("--shift-after-requests", type=int, default=6)
    parser.add_argument("--duration", type=float, default=900.0)
    parser.add_argument("--approve", action="store_true", help="auto-approve a selection")
    parser.add_argument("--out", default="reports/gpu_offload_runtime_validation.json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = Settings().model_copy(
        update={
            # Allow the (bounded) multi-point search to consume the intervention budget.
            "phase10_max_cumulative_intervention_s": 1800.0,
            "phase10_gpu_search_time_limit_s": 360.0,
            "phase10_gpu_search_enabled": True,
            # A single bounded validation: avoid repeated searches in the same session.
            "phase10_intervention_cooldown_s": 180.0,
            # Normalize the (real, measured) benefit against a longer scale so a measured
            # ~40% throughput improvement can justify the interruption (validation only).
            "phase10_benefit_scale_s": 300.0,
        }
    )
    model = load_model_profile(Path(args.model))
    settings.ensure_directories()

    workload = ShiftingResidentWorkload(
        settings=settings,
        model=model,
        workload_id=args.workload,
        shift_after_requests=args.shift_after_requests,
        shifted_workload_id=args.shift_workload,
        configuration=RuntimeConfiguration(
            context_length=args.context, gpu_offload_layers=args.gpu_layers
        ),
    )
    budget = build_runtime_budget(settings, max_duration_s=args.duration)
    policy = build_runtime_policy(
        settings, budget=budget, require_approval_for_disruptive=True
    )
    approvals: ApprovalProvider = (
        AutoApproveApprovalProvider() if args.approve else AutoRejectApprovalProvider()
    )
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=SystemTelemetrySampler(settings=settings, include_gpu=True),
        harness=None,  # the GPU search uses the live resident adapter
        policy=policy,
        approval_provider=approvals,
        notification_sink=CollectingNotificationSink(),
        detector=ControlledGpuUnderutilizationDetector(),
        settings=settings,
        session_notes=["real runtime GPU-offload search validation"],
    )

    print("=== Real runtime GPU-offload search validation ===")
    print(f"model      : {args.model} ({model.n_layers} layers)")
    print(f"context    : {args.context}")
    print(f"gpu layers : {args.gpu_layers}")
    print(f"duration   : {args.duration:.0f}s")
    print(f"approval   : {'auto-approve' if args.approve else 'auto-reject (search only)'}")
    print()

    result = monitor.run(duration_s=args.duration)
    workload.stop()
    run = result.run

    print(f"termination       : {result.termination.reason.value}")
    print(f"observations      : {run.observations_collected}")
    print(f"diagnosis events  : {run.diagnosis_events}")
    print(f"opportunities     : {len(run.opportunities)}")
    print(f"recommendations   : {len(run.recommendations)}")
    print(f"approvals         : {[(a.outcome.value) for a in run.approvals]}")
    print(f"interventions     : {len(run.interventions)}")
    print(f"gpu searches      : {len(run.gpu_searches)}")
    print(f"gpu search time s : {run.overhead.gpu_search_s:.1f}")
    print(f"final config      : {run.workload.configuration.as_dict()}")
    print(f"resident recoveries: {len(run.target_recoveries)}")
    print()

    for search in run.gpu_searches:
        envelope = search.resource
        print("--- search ---")
        print(f"  sufficient        : {envelope.sufficient} {envelope.reasons}")
        print(f"  total/used/free   : {envelope.total_vram_bytes}/{envelope.used_vram_bytes}/{envelope.free_vram_bytes}")
        print(f"  other consumers   : {envelope.other_consumers_bytes}")
        print(f"  safety headroom   : {envelope.safety_headroom_bytes}")
        print(f"  kv cache          : {envelope.kv_cache_bytes}")
        print(f"  per-layer weight  : {envelope.per_layer_weight_bytes}")
        print(f"  context           : {envelope.context_length}")
        print(f"  feasible range    : {envelope.feasible_min}-{envelope.feasible_max}")
        print(f"  sampled layers    : {search.sampled_layers}")
        print(f"  selected          : {search.selection.kind.value} {search.selection.selected_layers}")
        print(f"  reason            : {search.selection.reason}")
        for point in search.points:
            decode = point.metrics.get("decode_tokens_per_sec")
            latency = point.metrics.get("total_latency_s")
            vram = point.metrics.get("gpu_memory_used_bytes")
            print(
                f"    layers={point.layers:>3} status={point.status.value:<14} "
                f"valid={point.valid} safe={point.safe} decision={point.decision} "
                f"decode={decode.mean if decode else None} "
                f"latency={latency.mean if latency else None} "
                f"vram={vram.mean if vram else None}"
            )
        print()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.to_json(), encoding="utf-8")
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
