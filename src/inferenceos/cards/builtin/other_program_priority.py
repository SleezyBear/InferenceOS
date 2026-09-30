"""Card (suggest only): lower another program's share of the processor."""

from __future__ import annotations

from inferenceos.cards.base import ActsOn, ApprovalRule, Card, When
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    OptimizationScope,
    OptimizationSurface,
    ParameterType,
)
from inferenceos.contracts.optimization import ParameterSpec

CARD = Card(
    card_id="external.reduce_priority",
    name="Recommend reducing an external process's CPU priority",
    what_it_changes=(
        "Suggests lowering the priority of another program that is using a lot of "
        "processor time. InferenceOS never changes it itself."
    ),
    acts_on=ActsOn.OTHER_PROGRAMS,
    programs=(),
    when=When.WHILE_RUNNING,
    needs_restart=False,
    applies_when=(
        "an external process is a plausible dominant CPU contributor",
        "human decision is required; InferenceOS performs no change",
    ),
    values_to_try="Below normal, or idle",
    should_improve=("process_cpu_percent", "decode_tokens_per_sec"),
    might_worsen=("the other program's speed",),
    undo="The user sets the program's priority back",
    approval=ApprovalRule.SUGGEST_ONLY,
    admin_rights=False,
    test_cost_estimate_s=0.0,
    answers=(BottleneckKind.CPU, BottleneckKind.EXTERNAL_PROCESS),
    parameters=(
        ParameterSpec(
            name="priority_class",
            type=ParameterType.STRING,
            description="Suggested priority class (lower than current), user-applied.",
            choices=["below_normal", "idle"],
        ),
    ),
    surface=OptimizationSurface.EXTERNAL_PROCESS,
    subsystem="cpu_contention",
    causal_rationale=(
        "When an external process contends for CPU and the target workload is "
        "degraded, lowering the external process's priority could increase the "
        "target's share without terminating the process; only the user can decide."
    ),
    expected_direction="decrease",
    scope=OptimizationScope.EXTERNAL_PROCESS,
    external_recommendation=True,
)
