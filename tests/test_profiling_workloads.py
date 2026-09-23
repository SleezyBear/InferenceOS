"""Deterministic benchmark workload registry and context resolution."""

from __future__ import annotations

import pytest

from inferenceos.profiling.workloads import (
    BENCHMARK_SUITE,
    BENCHMARK_SUITE_VERSION,
    choose_context_length,
    get_workload,
    select_workloads,
    workload_profiles,
)
from inferenceos.selftest import sample_model


def test_suite_is_versioned_and_stable() -> None:
    assert BENCHMARK_SUITE_VERSION == "1.0"
    ids = [workload.workload_id for workload in BENCHMARK_SUITE]
    assert ids == ["short-short", "long-prefill", "short-long-decode"]
    assert len(set(ids)) == len(ids)


def test_prompt_hashes_are_deterministic() -> None:
    first = {workload.workload_id: workload.prompt_hash for workload in BENCHMARK_SUITE}
    second = {workload.workload_id: workload.prompt_hash for workload in BENCHMARK_SUITE}
    assert first == second
    assert all(len(digest) == 64 for digest in first.values())


def test_long_prefill_prompt_is_larger_than_short() -> None:
    short = get_workload("short-short")
    long = get_workload("long-prefill")
    assert len(long.prompt) > len(short.prompt)
    assert long.min_context_tokens > short.min_context_tokens


def test_long_decode_workload_ignores_eos() -> None:
    short = get_workload("short-short")
    long = get_workload("short-long-decode")
    assert long.n_predict > short.n_predict
    assert long.ignore_eos is True
    assert short.ignore_eos is False


def test_get_workload_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_workload("does-not-exist")


def test_select_workloads_filters_by_model_context() -> None:
    tiny = sample_model().model_copy(update={"trained_context_length": 128})
    big = sample_model().model_copy(update={"trained_context_length": 32768})
    assert [w.workload_id for w in select_workloads(None, tiny)] == ["short-short"]
    assert len(select_workloads(None, big)) == len(BENCHMARK_SUITE)


def test_select_workloads_explicit_ids() -> None:
    big = sample_model().model_copy(update={"trained_context_length": 32768})
    selected = select_workloads(["long-prefill"], big)
    assert [w.workload_id for w in selected] == ["long-prefill"]


def test_choose_context_length_respects_trained_context() -> None:
    tiny = sample_model().model_copy(update={"trained_context_length": 128})
    big = sample_model().model_copy(update={"trained_context_length": 32768})
    assert choose_context_length(tiny) == 128
    assert choose_context_length(big) == 2048
    assert choose_context_length(big, 4096) == 4096
    assert choose_context_length(big, 100000) == 100000


def test_workload_profile_projection() -> None:
    profiles = workload_profiles(list(BENCHMARK_SUITE))
    assert len(profiles) == len(BENCHMARK_SUITE)
    for workload, profile in zip(BENCHMARK_SUITE, profiles):
        assert profile.workload_id == workload.workload_id
        assert profile.prompt_set_ref and profile.prompt_set_ref.startswith(workload.workload_id)
        assert profile.prompts == [workload.prompt]
        assert profile.seed == workload.seed
