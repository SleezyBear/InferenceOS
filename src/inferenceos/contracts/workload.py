"""Workload contracts (Phase 3 harness: prompt sets, seeds, measurement modes)."""

from __future__ import annotations

from pydantic import Field, model_validator

from inferenceos.contracts.base import ContractBase
from inferenceos.contracts.enums import MeasurementMode, WorkloadKind


class ContextRequirement(ContractBase):
    """A workload's context requirement, with observed/expected/hard bounds kept distinct.

    ``hard_required_context_tokens`` is the maximum prompt + generation context the
    workload/request is **guaranteed** to require. It is a hard lower bound on the
    configured ``n_ctx``: a smaller context can reject or silently truncate a valid
    request, so a context optimizer must never propose a target below it.

    The other fields are deliberately *not* interchangeable:

    * ``expected_context_tokens`` is the workload's declared (minimum) expectation — a
      lower bound, never a maximum and never a substitute for the hard requirement;
    * ``observed_context_tokens`` is the largest context actually seen — a lower bound,
      never a guarantee about future requests;
    * ``max_prompt_tokens`` + ``required_generation_tokens`` may decompose the hard
      requirement when the contract provides them.

    ``required_context_length()`` returning ``None`` means the hard requirement is
    **unknown**; it must never be guessed from the other fields.
    """

    #: Maximum prompt + generation context the workload/request is guaranteed to require.
    hard_required_context_tokens: int | None = Field(default=None, ge=1)
    #: Optional decomposition of the hard requirement (when the contract provides it).
    max_prompt_tokens: int | None = Field(default=None, ge=1)
    required_generation_tokens: int | None = Field(default=None, ge=0)
    #: Declared (minimum) expectation — NOT a maximum and NOT the hard requirement.
    expected_context_tokens: int | None = Field(default=None, ge=1)
    #: Largest context actually observed — a lower bound, never a future guarantee.
    observed_context_tokens: int | None = Field(default=None, ge=0)
    source: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _reconcile_hard_requirement(self) -> "ContextRequirement":
        # A decomposed prompt + generation bound is authoritative only when no explicit
        # hard bound was supplied. It augments (never overrides) an explicit hard value.
        if self.hard_required_context_tokens is None and self.max_prompt_tokens is not None:
            object.__setattr__(
                self,
                "hard_required_context_tokens",
                self.max_prompt_tokens + (self.required_generation_tokens or 0),
            )
        return self

    @property
    def known(self) -> bool:
        """True only when a hard requirement is actually established."""
        return self.required_context_length() is not None

    def required_context_length(self) -> int | None:
        """The hard requirement, or ``None`` when unknown (never guessed)."""
        if self.hard_required_context_tokens is not None:
            return self.hard_required_context_tokens
        if self.max_prompt_tokens is not None:
            return self.max_prompt_tokens + (self.required_generation_tokens or 0)
        return None


class WorkloadProfile(ContractBase):
    workload_id: str
    kind: WorkloadKind = WorkloadKind.UNKNOWN
    description: str = ""
    prompt_set_ref: str | None = None
    prompts: list[str] = Field(default_factory=list)
    seed: int = 0
    expected_context_length: int | None = None
    #: The workload's hard context requirement (authoritative safety bound). ``None``
    #: means unknown; a context optimizer must not reduce context autonomously.
    context_requirement: ContextRequirement | None = None
    concurrency: int = Field(default=1, ge=1)
    modes: list[MeasurementMode] = Field(
        default_factory=lambda: [MeasurementMode.COLD, MeasurementMode.WARM]
    )
