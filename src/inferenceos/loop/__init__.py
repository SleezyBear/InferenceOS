"""Phase 9 pre-deployment autonomous optimization loop.

Wires the existing phases into one unattended flow: observe -> diagnose ->
deep-diagnose when warranted -> generate candidates -> query durable knowledge ->
select -> checkpoint -> experiment -> validate -> keep/revert -> persist -> continue
or terminate. A clean "no meaningful optimization opportunity" result is a success.
"""

from __future__ import annotations

from inferenceos.loop.checkpoint import (
    capture_checkpoint,
    restore_checkpoint,
    verify_checkpoint,
)
from inferenceos.loop.engine import OptimizationLoop, run_optimization_loop
from inferenceos.loop.knowledge import (
    gather_knowledge_evidence,
    selection_adjustment,
)
from inferenceos.loop.policy import build_budget, build_loop_policy
from inferenceos.loop.report import render_run, render_termination

__all__ = [
    "OptimizationLoop",
    "build_budget",
    "build_loop_policy",
    "capture_checkpoint",
    "gather_knowledge_evidence",
    "render_run",
    "render_termination",
    "restore_checkpoint",
    "run_optimization_loop",
    "selection_adjustment",
    "verify_checkpoint",
]
