"""Card (suggest only): ask the user to close a competing program."""

from __future__ import annotations

from inferenceos.cards.base import ActsOn, ApprovalRule, Card, When
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    OptimizationScope,
    OptimizationSurface,
    ParameterType,
    Reversibility,
    RiskLevel,
)
from inferenceos.contracts.optimization import ParameterSpec

CARD = Card(
    card_id="external.request_close",
    name="Recommend closing a competing process",
    what_it_changes=(
        "Suggests the user closes another program that is likely competing with the "
        "model. InferenceOS never closes anything itself."
    ),
    acts_on=ActsOn.OTHER_PROGRAMS,
    programs=(),
    when=When.WHILE_RUNNING,
    needs_restart=False,
    applies_when=(
        "an external process is a plausible dominant contributor under contention",
        "human approval/decision is required",
    ),
    values_to_try="The program most likely to be competing",
    should_improve=("available_ram_bytes",),
    might_worsen=("the other program's work",),
    undo="The user reopens the program (cannot be undone automatically)",
    approval=ApprovalRule.SUGGEST_ONLY,
    admin_rights=False,
    test_cost_estimate_s=0.0,
    answers=(
        BottleneckKind.EXTERNAL_PROCESS,
        BottleneckKind.MEMORY_PRESSURE,
        BottleneckKind.CPU,
    ),
    parameters=(
        ParameterSpec(
            name="process_name",
            type=ParameterType.STRING,
            description="Observed process name the user may consider closing.",
            choices=["memory_holder.exe", "python.exe", "pythonw.exe"],
        ),
    ),
    surface=OptimizationSurface.EXTERNAL_PROCESS,
    subsystem="user_action",
    causal_rationale=(
        "A process that is a plausible dominant contributor to a system resource "
        "under contention may be affecting the target workload if the user does not "
        "need it running; only the user can decide that."
    ),
    expected_direction="increase",
    reversibility=Reversibility.NON_REVERSIBLE,
    risk_level=RiskLevel.MEDIUM,
    scope=OptimizationScope.EXTERNAL_PROCESS,
    external_recommendation=True,
)
