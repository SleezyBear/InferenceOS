"""Step 4 "Done when" check on the real laptop.

Done when: in balanced effort with default settings, the layer gain on the laptop is
found and kept.

This drives the real watcher (``RuntimeMonitor``) over a real resident llama-server,
exactly like ``scripts/validate_gpu_offload_runtime_search.py`` did, but with **your
default settings** (no ``phase10_benefit_scale_s`` workaround). Two things are forced
only so the path runs on demand, as before:

* a one-time real workload shift (a longer prompt) so the watcher has something to
  react to;
* a verification detector that labels the opportunity "graphics card underused", so
  the layer search is the fix it tries.

Kept changes normally ask you first; this check approves automatically unless you
pass ``--ask``.

Run (from the repo folder, one line in PowerShell):
    .\\.venv\\Scripts\\python.exe scripts\\check_step4_layer_gain.py --model models\\qwen2.5-1.5b-instruct-q4_k_m.gguf --gpu-layers 8
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Any, cast

from inferenceos.config import Settings
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.diagnosis.detector import BottleneckDetector
from inferenceos.models.loader import load_model_profile
from inferenceos.optimization.goals import build_goal_preference
from inferenceos.optimization.preference import build_optimization_preference
from inferenceos.profiling.workloads import get_workload
from inferenceos.runtime.approval import (
    ApprovalProvider,
    AutoApproveApprovalProvider,
    CliApprovalProvider,
)
from inferenceos.runtime.monitor import RuntimeMonitor
from inferenceos.runtime.notification import CollectingNotificationSink
from inferenceos.runtime.policy import build_runtime_budget, build_runtime_policy
from inferenceos.runtime.resident_workload import ResidentActiveWorkload
from inferenceos.runtime.telemetry import SystemTelemetrySampler
from inferenceos.runtime.verification import ControlledGpuUnderutilizationDetector


class ShiftingResidentWorkload(ResidentActiveWorkload):
    """Resident workload that makes one real workload shift after N requests."""

    def __init__(
        self,
        *,
        shift_after_requests: int,
        shifted_workload_id: str,
        **kwargs: Any,
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
    parser.add_argument("--ask", action="store_true", help="ask before keeping (y/n)")
    parser.add_argument("--out", default="reports/step4_layer_gain_check.json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = Settings()  # your saved settings, unchanged
    effort = build_optimization_preference(settings)
    goal = build_goal_preference(settings)
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
    policy = build_runtime_policy(settings, budget=budget, require_approval_for_disruptive=True)
    approvals: ApprovalProvider = (
        CliApprovalProvider() if args.ask else AutoApproveApprovalProvider()
    )
    monitor = RuntimeMonitor(
        workload=workload,
        sampler=SystemTelemetrySampler(settings=settings, include_gpu=True),
        harness=None,  # the layer search uses the live resident model
        policy=policy,
        approval_provider=approvals,
        notification_sink=CollectingNotificationSink(),
        # Same stand-in detector the older layer-search check used.
        detector=cast(BottleneckDetector, ControlledGpuUnderutilizationDetector()),
        settings=settings,
        session_notes=["step 4 done-when check: default settings, balanced effort"],
    )

    print("=== Step 4 check: layer gain found and kept with default settings ===")
    print(f"model      : {args.model} ({model.n_layers} layers)")
    print(f"start      : {args.gpu_layers} graphics card layers, context {args.context}")
    print(
        f"effort     : {effort.mode.value} (smallest gain {effort.minimum_improvement:.1%}, "
        f"search limit {effort.maximum_search_time_s:.0f}s)"
    )
    print(f"goal       : {goal.mode.value}")
    print(
        "payback    : "
        f"{settings.payback_answering_hours_while_running:g} h of answering while running"
    )
    print(f"duration   : up to {args.duration:.0f}s (please leave the computer alone)")
    print()

    result = monitor.run(duration_s=args.duration)
    workload.stop()
    run = result.run

    searches = run.gpu_searches
    selected = [s for s in searches if s.selection.selected_layers is not None]
    for search in searches:
        print("--- layer search ---")
        print(f"  tried layers : {search.sampled_layers}")
        print(f"  selected     : {search.selection.kind.value} {search.selection.selected_layers}")
        print(f"  reason       : {search.selection.reason}")
        for point in search.points:
            decode = point.metrics.get("decode_tokens_per_sec")
            print(
                f"    layers={point.layers:>3} valid={point.valid} safe={point.safe} "
                f"writing speed={decode.mean if decode else None}"
            )
    economics_stops = [c for c in run.cooldowns if c.trigger == "economics"]
    for cooldown in economics_stops:
        print(f"--- not worth testing: {cooldown.reason}")
    kept = [i for i in run.interventions if i.retained]
    for intervention in run.interventions:
        economics = intervention.retention_economics
        print("--- keep decision ---")
        print(f"  experiment   : {intervention.decision.value if intervention.decision else None}")
        print(f"  kept         : {intervention.retained}")
        if economics is not None:
            print(f"  verdict      : {economics.keep_verdict}")
            for line in economics.rationale:
                print(f"    {line}")
    print()
    print(f"final setup: {run.workload.configuration.as_dict()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.to_json(), encoding="utf-8")
    print(f"report: {out_path}")
    print()
    if selected and kept:
        print("RESULT: PASS - the layer gain was found and kept.")
        return 0
    if not searches:
        print("RESULT: NOT RUN - no layer search happened (see the report).")
    elif not selected:
        print("RESULT: NO GAIN - the search found no better layer count this time.")
    else:
        print("RESULT: FAIL - a better layer count was found but not kept.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
