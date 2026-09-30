"""Step 3: the card format.

* The built-in cards are found automatically and all pass the card checks.
* The automatic context-safety check refuses any card, or any change, that would
  touch context size.
* **Done when:** a brand-new card works end to end without editing any other code.
  The new card below lives only in a file written by this test; nothing in
  ``src/`` knows about it.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from inferenceos.cards import (
    LLAMA_CPP,
    ApprovalRule,
    Card,
    CardCatalog,
    CardError,
    ContextSafetyError,
    ProgramSetting,
    check_context_unchanged,
    default_catalog,
    discover_builtin_cards,
    register_card,
    unregister_card,
)
from inferenceos.contracts.bottleneck import BottleneckKind
from inferenceos.contracts.enums import (
    CandidateEligibility,
    CandidateKind,
    ExperimentDecisionKind,
    ParameterType,
    RestorationStatus,
)
from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.optimization import ParameterSpec
from inferenceos.experiment import audit as audit_module
from inferenceos.experiment import executor as executor_module
from inferenceos.experiment.engine import run_experiment
from inferenceos.experiment.executor import (
    RuntimeConfigurationExecutor,
    apply_to_configuration,
    supported_actions,
)
from inferenceos.experiment.quality_policy import build_adaptive_quality_policy
from inferenceos.config import Settings
from inferenceos.optimization.actions import build_default_registry
from inferenceos.optimization.generator import generate_candidates
from inferenceos.profiling.runner import ProfilerOptions, ProfileTask
from inferenceos.profiling.workloads import get_workload
from inferenceos.runtime.verification import ScriptedWorkload
from inferenceos.runtime.workload import RuntimeWorkloadExecutor
from tests.experiment_helpers import SampleResult, ScriptedHarness, make_request
from tests.knowledge_helpers import make_service
from tests.optimization_helpers import synthetic_diagnosis

BUILTIN_IDS = {
    "runtime.enable_mmap",
    "runtime.increase_gpu_offload_layers",
    "runtime.reduce_gpu_offload_layers",
    "runtime.reduce_threads",
    "external.suspend_process",
    "external.request_close",
    "external.reduce_priority",
    "system.increase_pagefile",
}


# --------------------------------------------------------------------------- #
# Built-in cards                                                               #
# --------------------------------------------------------------------------- #
def test_builtin_cards_are_found_automatically() -> None:
    found = {card.card_id for card in discover_builtin_cards()}
    assert found == BUILTIN_IDS
    assert set(default_catalog().ids()) >= BUILTIN_IDS


def test_the_four_real_changes_are_applied_cards_on_llama_cpp() -> None:
    catalog = default_catalog()
    expected = {
        "runtime.enable_mmap": "use_mmap",
        "runtime.increase_gpu_offload_layers": "gpu_offload_layers",
        "runtime.reduce_gpu_offload_layers": "gpu_offload_layers",
        "runtime.reduce_threads": "threads",
    }
    for card_id, setting in expected.items():
        card = catalog.get(card_id)
        assert card.suggest_only is False
        assert card.setting_for(LLAMA_CPP) == setting
        assert card.approval is ApprovalRule.KEPT_CHANGE
    assert supported_actions() >= set(expected)


def test_other_program_and_page_file_cards_are_suggest_only() -> None:
    catalog = default_catalog()
    for card_id in (
        "external.suspend_process",
        "external.request_close",
        "external.reduce_priority",
        "system.increase_pagefile",
    ):
        card = catalog.get(card_id)
        assert card.suggest_only is True
        assert card.programs == ()
        assert card_id not in supported_actions()
    assert catalog.get("system.increase_pagefile").admin_rights is True


def test_every_card_fills_in_the_plain_fields() -> None:
    for card in default_catalog().all():
        assert card.what_it_changes
        assert card.values_to_try
        assert card.should_improve
        assert card.might_worsen
        assert card.undo
        assert card.applies_when
        assert card.context_safe is True


def test_registry_is_built_from_cards() -> None:
    registry = build_default_registry()
    for card in default_catalog().all():
        action = registry.get(card.card_id)
        assert action.description == card.what_it_changes
        assert list(action.supported_bottlenecks) == list(card.answers)


# --------------------------------------------------------------------------- #
# Automatic context-safety check                                               #
# --------------------------------------------------------------------------- #
def _mmap_card() -> Card:
    return default_catalog().get("runtime.enable_mmap")


def test_card_that_is_not_context_safe_is_refused() -> None:
    card = replace(_mmap_card(), card_id="test.not_context_safe", context_safe=False)
    with pytest.raises(ContextSafetyError):
        CardCatalog([card])


def test_card_whose_setting_is_context_size_is_refused() -> None:
    card = replace(
        _mmap_card(),
        card_id="test.context_setting",
        programs=(ProgramSetting(LLAMA_CPP, "context_length", "--ctx-size N"),),
    )
    with pytest.raises(ContextSafetyError):
        CardCatalog([card])


def test_card_whose_parameter_is_context_size_is_refused() -> None:
    card = replace(
        _mmap_card(),
        card_id="test.context_parameter",
        parameters=(ParameterSpec(name="context_length", type=ParameterType.INTEGER),),
    )
    with pytest.raises(ContextSafetyError):
        CardCatalog([card])


def test_context_check_on_every_change() -> None:
    before = RuntimeConfiguration(context_length=4096, threads=8)
    check_context_unchanged(before, before.model_copy(update={"threads": 4}), card_id="x")
    with pytest.raises(ContextSafetyError):
        check_context_unchanged(
            before, before.model_copy(update={"context_length": 2048}), card_id="x"
        )


def test_executors_refuse_a_change_that_would_alter_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate a faulty card whose apply step writes to context size: both executors
    # must refuse it and leave the configuration untouched.
    monkeypatch.setattr(
        executor_module,
        "_card_setting",
        lambda action_id, program=LLAMA_CPP: "context_length",
    )
    action = build_default_registry().get("runtime.reduce_threads")
    with pytest.raises(ContextSafetyError):
        apply_to_configuration(
            action.action_id, {"threads": 2}, RuntimeConfiguration(context_length=4096)
        )

    executor = RuntimeConfigurationExecutor()
    executor.set_configuration(RuntimeConfiguration(context_length=4096, threads=8))
    result = executor.apply(action, {"threads": 2})
    assert result.applied is False
    assert "context size" in result.reason
    assert executor.current().context_length == 4096

    workload = ScriptedWorkload()
    before = workload.configuration()
    result = RuntimeWorkloadExecutor(workload).apply(action, {"threads": 2})
    assert result.applied is False
    assert workload.configuration() == before


# --------------------------------------------------------------------------- #
# Other card checks                                                            #
# --------------------------------------------------------------------------- #
def test_card_with_unknown_setting_is_refused() -> None:
    card = replace(
        _mmap_card(),
        card_id="test.unknown_setting",
        programs=(ProgramSetting(LLAMA_CPP, "not_a_setting", "--nope"),),
    )
    with pytest.raises(CardError):
        CardCatalog([card])


def test_applied_card_without_a_program_setting_is_refused() -> None:
    card = replace(_mmap_card(), card_id="test.no_program", programs=())
    with pytest.raises(CardError):
        CardCatalog([card])


def test_duplicate_card_is_refused() -> None:
    with pytest.raises(CardError):
        CardCatalog([_mmap_card(), _mmap_card()])


# --------------------------------------------------------------------------- #
# Done when: a new card works without editing any other code                   #
# --------------------------------------------------------------------------- #
NEW_CARD_ID = "runtime.test_batch_size"

NEW_CARD_FILE = textwrap.dedent(
    '''
    """A brand-new card, written only by the test: try a bigger batch size."""

    from inferenceos.cards import LLAMA_CPP, ActsOn, ApprovalRule, Card, ProgramSetting, When
    from inferenceos.contracts.bottleneck import BottleneckKind
    from inferenceos.contracts.enums import ParameterType, PreconditionKind
    from inferenceos.contracts.optimization import ParameterSpec
    from inferenceos.optimization.cost import estimate_cost
    from inferenceos.optimization.estimate import unknown_benefit
    from inferenceos.optimization.preconditions import check_evidence_support, make
    from inferenceos.optimization.strategies import PreparedAction


    def prepare(action, ctx, policy):
        return PreparedAction(
            parameters={"batch_size": 256},
            preconditions=[
                check_evidence_support(ctx, action, policy),
                make("always", PreconditionKind.APPLICABILITY, "test card applies", True),
            ],
            benefit=unknown_benefit(
                "decode_tokens_per_sec",
                "increase",
                confidence=ctx.effective_confidence,
                evidence_basis=list(ctx.evidence_ids),
                notes="measured, not estimated",
            ),
            cost=estimate_cost(ctx, action, policy),
            causal_rationale=action.causal_rationale,
            failure_consequences="Reversible by restoring the old batch size.",
            rollback_available=True,
        )


    CARD = Card(
        card_id="runtime.test_batch_size",
        name="Test: bigger batch size",
        what_it_changes="Reads the prompt in bigger pieces.",
        acts_on=ActsOn.MODEL_SETTINGS,
        programs=(ProgramSetting(LLAMA_CPP, "batch_size", "--batch-size N"),),
        when=When.BOTH,
        needs_restart=True,
        applies_when=("always, for this test",),
        values_to_try="256",
        should_improve=("decode_tokens_per_sec",),
        might_worsen=("peak_rss_bytes",),
        undo="Restart with the old batch size",
        approval=ApprovalRule.KEPT_CHANGE,
        admin_rights=False,
        answers=(BottleneckKind.CPU,),
        parameters=(
            ParameterSpec(name="batch_size", type=ParameterType.INTEGER, minimum=1, maximum=8192),
        ),
        prepare=prepare,
    )
    '''
)


@pytest.fixture()
def new_card(tmp_path: Path) -> Iterator[Card]:
    # Drop the card file into a fresh folder, as a person adding a card would.
    package = tmp_path / "extra_cards"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "batch_size.py").write_text(NEW_CARD_FILE, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        cards = discover_builtin_cards("extra_cards")
        assert [card.card_id for card in cards] == [NEW_CARD_ID]
        card = register_card(cards[0])
        yield card
    finally:
        unregister_card(NEW_CARD_ID)
        sys.path.remove(str(tmp_path))
        for name in [name for name in sys.modules if name.startswith("extra_cards")]:
            del sys.modules[name]


def _bigger_batch_is_faster(
    task: ProfileTask, options: ProfilerOptions, call: int
) -> SampleResult:
    if options.batch_size == 256:
        return SampleResult(decode_tps=30.0)
    return SampleResult(decode_tps=20.0)


def test_new_card_works_end_to_end_without_editing_other_code(new_card: Card) -> None:
    # 1. It is in the registry the whole system uses.
    registry = build_default_registry()
    assert NEW_CARD_ID in registry

    # 2. The generator asks the card and produces a runnable candidate.
    plan = generate_candidates(synthetic_diagnosis(BottleneckKind.CPU), report=None)
    candidate = next(c for c in plan.candidates if c.action_id == NEW_CARD_ID)
    assert candidate.eligibility is CandidateEligibility.ELIGIBLE
    assert candidate.executable is True
    assert candidate.parameters == {"batch_size": 256}

    # 3. Both executors apply and undo it through the card's program setting.
    assert NEW_CARD_ID in supported_actions()
    executor = RuntimeConfigurationExecutor()
    baseline = RuntimeConfiguration(context_length=2048, batch_size=512)
    executor.set_configuration(baseline)
    applied = executor.apply(registry.get(NEW_CARD_ID), {"batch_size": 256})
    assert applied.applied is True
    assert applied.changes[0].prior_value == 512
    assert executor.current().batch_size == 256
    assert executor.current().context_length == 2048
    assert executor.restore(baseline).verified is True

    workload = ScriptedWorkload()
    result = RuntimeWorkloadExecutor(workload).apply(
        registry.get(NEW_CARD_ID), {"batch_size": 256}
    )
    assert result.applied is True
    assert workload.configuration().batch_size == 256

    # 4. A real (scripted) experiment measures it, keeps it, and restores the baseline.
    request = make_request(
        candidate=candidate,
        experiment=make_request().baseline_configuration.model_copy(update={"batch_size": 256}),
    ).model_copy(
        update={
            "diagnosed_bottleneck": "cpu",
            "bottleneck_metric": "decode_tokens_per_sec",
            "bottleneck_direction": "increase",
        }
    )
    outcome = run_experiment(
        request,
        executor=RuntimeConfigurationExecutor(),
        profiler=ScriptedHarness(_bigger_batch_is_faster),
        workload=get_workload("short-short"),
        action=registry.get(NEW_CARD_ID),
    )
    assert outcome.status is ExperimentDecisionKind.IMPROVED
    assert outcome.restoration is not None
    assert outcome.restoration.status is RestorationStatus.RESTORED

    # 5. Knowledge, audit and quality code all read the card.
    service = make_service()
    record = service.build_record(outcome)
    assert record.action_id == NEW_CARD_ID
    assert record.action_registered is True
    assert audit_module._surface_to_kind(NEW_CARD_ID) is CandidateKind.OTHER
    policy = build_adaptive_quality_policy(Settings(), action_id=NEW_CARD_ID)
    assert policy.affects_generation_semantics is False


def test_new_card_is_gone_after_unregistering(new_card: Card) -> None:
    unregister_card(NEW_CARD_ID)
    assert NEW_CARD_ID not in build_default_registry()
    assert NEW_CARD_ID not in supported_actions()
