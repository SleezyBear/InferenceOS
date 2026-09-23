"""Shared base class for all InferenceOS data contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ContractBase(BaseModel):
    """Immutable, strictly-validated base for every contract.

    - ``extra="forbid"``: unknown fields are rejected rather than silently ignored.
    - ``frozen=True``: records are immutable and hashable, safe to persist/share.
    - ``validate_assignment`` is irrelevant while frozen; use ``model_copy``.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        validate_default=True,
    )
