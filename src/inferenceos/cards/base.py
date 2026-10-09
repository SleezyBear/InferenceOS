"""The optimization card: one description per kind of optimization.

Every kind of optimization InferenceOS can test or suggest is described by exactly one
:class:`Card`. The rest of the system reads cards only: the candidate generator asks a
card whether it applies and which value to try, the executors apply and undo a card's
change through the card's per-program setting, and the audit, quality and safety code
read the card's fields. Adding an optimization means adding one card file (see
:mod:`inferenceos.cards.builtin`); no other code has to change.

A card has two parts:

* the plain fields from the build plan ("what it changes", "where it acts", "values
  to try", "should improve", "undo", ...), which are what a person reads; and
* a few engine fields (parameter bounds, the diagnoses it answers, blast radius,
  risk) plus an optional ``prepare`` rule that decides, from the real measured
  context, whether the card applies right now and which value to try.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    CandidateKind,
    EvidenceStrength,
    ExternalActionSafetyClass,
    OptimizationScope,
    OptimizationSurface,
    PrivilegeRequirement,
    Reversibility,
    RiskLevel,
)
from inferenceos.contracts.optimization import CandidateAction, ParameterSpec
from inferenceos.contracts.path_declaration import PathDeclaration

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids an import cycle)
    from inferenceos.optimization.context import OptimizationContext
    from inferenceos.optimization.policy import OptimizationPolicy
    from inferenceos.optimization.strategies import PreparedAction

    PrepareFn = Callable[
        [CandidateAction, OptimizationContext, OptimizationPolicy], PreparedAction
    ]

#: The name of the only program connector that exists today.
LLAMA_CPP = "llama.cpp"


class ActsOn(StrEnum):
    """Where a card's change acts."""

    MODEL_SETTINGS = "model_settings"
    PROGRAM = "program"
    OPERATING_SYSTEM = "operating_system"
    OTHER_PROGRAMS = "other_programs"


class When(StrEnum):
    """When a card may be tested."""

    BEFORE_LOADING = "before_loading"
    WHILE_RUNNING = "while_running"
    BOTH = "both"


class ApprovalRule(StrEnum):
    """What needs the user's yes.

    Every kept change always needs approval. ``BEFORE_TESTING`` cards also ask before
    they are even tested (for example anything that pauses another program).
    ``SUGGEST_ONLY`` cards are never applied by InferenceOS; they are only suggested.
    """

    KEPT_CHANGE = "kept_change"
    BEFORE_TESTING = "before_testing"
    SUGGEST_ONLY = "suggest_only"


@dataclass(frozen=True)
class ProgramSetting:
    """How one program sets a card's value.

    ``setting`` is the runtime-configuration field that InferenceOS changes for that
    program (the apply step writes it, the undo step writes the old value back).
    ``how_set`` is the plain description of the program's own option.
    """

    program: str
    setting: str
    how_set: str


@dataclass(frozen=True)
class Card:
    """One kind of optimization. See the module docstring."""

    # --- plain fields (the build plan's card table) ------------------------------
    card_id: str
    name: str
    what_it_changes: str
    acts_on: ActsOn
    programs: tuple[ProgramSetting, ...]
    when: When
    needs_restart: bool
    applies_when: tuple[str, ...]
    values_to_try: str
    should_improve: tuple[str, ...]
    might_worsen: tuple[str, ...]
    undo: str
    approval: ApprovalRule
    admin_rights: bool
    works_with: tuple[str, ...] = ()
    clashes_with: tuple[str, ...] = ()
    quality_risk: bool = False
    #: Every card must state that it never changes context size. Checked on
    #: registration and before/after every applied change.
    context_safe: bool = True
    #: Rough test cost in seconds until a measured cost exists for this machine.
    test_cost_estimate_s: float = 1.0

    # --- engine fields -----------------------------------------------------------
    answers: tuple[BottleneckKind, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    kind: CandidateKind = CandidateKind.OTHER
    surface: OptimizationSurface = OptimizationSurface.RUNTIME_PARAMS
    subsystem: str = "runtime"
    causal_rationale: str = ""
    expected_direction: str = "increase"
    reversibility: Reversibility = Reversibility.REVERSIBLE
    risk_level: RiskLevel = RiskLevel.LOW
    scope: OptimizationScope = OptimizationScope.TARGET_MODEL_RUNTIME
    min_evidence_strength: EvidenceStrength = EvidenceStrength.MEDIUM
    #: May the external-process recommender (suggest-only) use this card?
    external_recommendation: bool = False
    #: Decides, from the real measured context, whether the card applies now and
    #: which value to try. ``None`` means the card never produces a candidate by
    #: itself (for example a suggest-only card used by the external recommender).
    prepare: PrepareFn | None = field(default=None, compare=False, repr=False)
    version: str = "1.0"
    #: Tune's declaration of the same optimization path (C3, 9 Oct 2026), for a card tune
    #: also records changes under. Absent by default. The older optimizer never reads it:
    #: it does not take part in ``to_action``, ``prepare``, card validation rules, card
    #: equality or catalog membership, and a declaration never makes a card (or a path
    #: without a card file) a candidate. The value is the canonical declaration object
    #: from ``tune/declarations.py``, not a copy.
    tune: PathDeclaration | None = field(default=None, compare=False, repr=False)

    # ------------------------------------------------------------------ #
    @property
    def suggest_only(self) -> bool:
        return self.approval is ApprovalRule.SUGGEST_ONLY

    def setting_for(self, program: str = LLAMA_CPP) -> str | None:
        """The runtime-configuration field this card changes for ``program``."""
        for binding in self.programs:
            if binding.program == program:
                return binding.setting
        return None

    def settings(self) -> tuple[str, ...]:
        """Every runtime-configuration field this card can change, across programs."""
        return tuple(dict.fromkeys(binding.setting for binding in self.programs))

    def to_action(self) -> CandidateAction:
        """The engine's view of this card (derived; the card is the source of truth)."""
        return CandidateAction(
            action_id=self.card_id,
            name=self.name,
            description=self.what_it_changes,
            surface=self.surface,
            subsystem=self.subsystem,
            supported_bottlenecks=list(self.answers),
            causal_rationale=self.causal_rationale or self.what_it_changes,
            parameters=list(self.parameters),
            preconditions=list(self.applies_when),
            expected_metrics=list(self.should_improve),
            expected_direction=self.expected_direction,
            reversibility=self.reversibility,
            risk_level=self.risk_level,
            scope=self.scope,
            privilege=(
                PrivilegeRequirement.ELEVATION_REQUIRED
                if self.admin_rights
                else PrivilegeRequirement.STANDARD_USER
            ),
            requires_human_approval=self.approval is not ApprovalRule.KEPT_CHANGE,
            execution_cost_estimate=self.test_cost_estimate_s,
            optimizer_overhead_estimate=0.02 if self.test_cost_estimate_s > 0 else 0.0,
            min_evidence_strength=self.min_evidence_strength,
            external_safety_class=ExternalActionSafetyClass.RECOMMENDATION_ONLY,
            # The other-program layer never executes anything itself (recommendation
            # only), so no card is ever marked as externally executable.
            implemented=False,
            phase11_allowed=self.external_recommendation,
            verification_supported=False,
            version=self.version,
        )
