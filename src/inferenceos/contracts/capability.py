"""Three separate questions about any setting InferenceOS may change.

1. Does the engine **accept** the option and this value (its own help text)?
2. Does it **apply** to this model on this device (the model's pieces, its layer
   kinds, the card, other settings it depends on)?
3. Did the requested setting **take effect** (what the engine's own output showed
   when it ran)?

Each answer is yes, no or unknown, and unknown stays unknown: help text alone never
makes a candidate verified, and a candidate is only accepted with three yeses.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from inferenceos.contracts.base import ContractBase


class Support(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class CapabilityCheck(ContractBase):
    """The three answers for one setting and value, with the reasons."""

    setting: str
    value: str | None = None
    engine_accepts: Support = Support.UNKNOWN
    applies_here: Support = Support.UNKNOWN
    took_effect: Support = Support.UNKNOWN
    reasons: list[str] = Field(default_factory=list)

    @property
    def verified(self) -> bool:
        return (
            self.engine_accepts is Support.YES
            and self.applies_here is Support.YES
            and self.took_effect is Support.YES
        )

    def record(self) -> dict[str, object]:
        return {
            "setting": self.setting,
            "value": self.value,
            "engine_accepts": self.engine_accepts.value,
            "applies_here": self.applies_here.value,
            "took_effect": self.took_effect.value,
            "reasons": list(self.reasons),
        }


__all__ = ["CapabilityCheck", "Support"]
