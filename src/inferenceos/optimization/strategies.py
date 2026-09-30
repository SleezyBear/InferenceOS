"""Per-card candidate preparation.

Each card's ``prepare`` rule (see :mod:`inferenceos.cards`) binds concrete parameters
from the real context, evaluates its explicit preconditions, and produces bounded
benefit/cost estimates. Preparation does not decide eligibility or ranking (the
safety policy and ranker do), and it never executes anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from inferenceos.contracts.optimization import (
    BenefitEstimate,
    CandidateAction,
    CandidatePrecondition,
    CostEstimate,
)
from inferenceos.optimization.context import OptimizationContext
from inferenceos.optimization.policy import OptimizationPolicy


@dataclass(frozen=True)
class PreparedAction:
    parameters: dict[str, object]
    preconditions: list[CandidatePrecondition]
    benefit: BenefitEstimate
    cost: CostEstimate
    causal_rationale: str
    failure_consequences: str
    rollback_available: bool
    # A distinct, machine-readable reason when the action is a valid no-op rather than a
    # failure (e.g. the runtime is already at the desired state). ``None`` otherwise.
    no_change_reason: str | None = None


#: Distinct NO_CHANGE reason: the observed runtime is already at (or below) the desired
#: state, so there is no oversubscription remediation to perform.
ALREADY_AT_DESIRED_STATE = "ALREADY_AT_DESIRED_STATE"


def prepare(
    action: CandidateAction, ctx: OptimizationContext, policy: OptimizationPolicy
) -> PreparedAction | None:
    """Ask the action's card whether it applies now and which value to try.

    Returns ``None`` when there is no card for the action, or the card has no
    ``prepare`` rule (for example a suggest-only card used by the external layer).
    """
    from inferenceos.cards.catalog import get_card

    card = get_card(action.action_id)
    if card is None or card.prepare is None:
        return None
    return card.prepare(action, ctx, policy)
