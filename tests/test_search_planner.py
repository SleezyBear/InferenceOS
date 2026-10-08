"""Candidate planner rules (``tune/planner.py``), handoff of 6 Oct 2026."""

from __future__ import annotations

from inferenceos.tune.planner import BATCH, PLACEMENT, THREADS, Planner, Structure


def _structure(**kw: object) -> Structure:
    base: dict[str, object] = dict(
        count_max=33, top=33, uses_experts=False, mixed_kinds=False,
        physical_cores=6, logical_cores=12, long_question_tokens=3000,
    )
    base.update(kw)
    return Structure(**base)  # type: ignore[arg-type]


def _drive(planner: Planner, truth: dict[str, dict[int, float]], noise: float = 0.01) -> list[tuple[str, tuple[int, ...]]]:
    """Run the planner to the end with exact (noise-free) gains from ``truth``."""
    steps = []
    for _ in range(30):
        step = planner.next_step()
        if step is None:
            break
        steps.append((step.category, step.values))
        for value in step.values:
            planner.record(step.category, value, truth[step.category].get(value, 0.0), noise)
    return steps


def test_coarse_placements_first_and_without_duplicates() -> None:
    planner = Planner(_structure(), start_layers=33, start_threads=None, start_batch=None,
                      processor_share=lambda layers: 0.1)
    step = planner.next_step()
    assert step is not None and step.category == PLACEMENT
    # The current count (33 = top) is the reference, not measured again.
    assert set(step.values) == {0, 1}


def test_fully_fitting_clear_winner_needs_no_intermediate_counts() -> None:
    planner = Planner(_structure(), start_layers=20, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.05 if layers >= 33 else 0.6)
    truth = {PLACEMENT: {33: 0.60, 0: -0.5, 1: -0.45}, THREADS: {}, BATCH: {1024: 0.12}}
    steps = _drive(planner, truth)
    placement_steps = [values for category, values in steps if category == PLACEMENT]
    assert placement_steps == [(33, 0, 1)]
    assert planner.placement == 33
    # Card carries 95%: batch first, threads only a deferred option.
    assert steps[1][0] == BATCH
    assert all(category != THREADS for category, _ in steps)


def test_partial_offload_probes_intermediate_counts_even_when_an_extreme_wins() -> None:
    # Model does not fully fit (top 20 of 41): the top wins, but intermediates are probed.
    planner = Planner(_structure(count_max=41, top=20), start_layers=20, start_threads=6,
                      start_batch=512, processor_share=lambda layers: 0.7)
    truth = {PLACEMENT: {0: -0.3, 1: -0.28, 17: 0.03, 14: 0.01}, THREADS: {}, BATCH: {}}
    steps = _drive(planner, truth)
    placement_steps = [values for category, values in steps if category == PLACEMENT]
    assert placement_steps[0] == (0, 1)
    assert placement_steps[1] == (17, 14)  # intermediate counts below the top
    assert any("partial offload" in line for line in planner.log)


def test_ambiguous_coarse_result_probes_intermediates() -> None:
    # Fully fits, but top and CPU-only are within two noise widths of each other.
    planner = Planner(_structure(count_max=29, top=29), start_layers=29, start_threads=6,
                      start_batch=512, processor_share=lambda layers: 0.4)
    truth = {PLACEMENT: {0: 0.01, 1: -0.01, 25: 0.06, 21: 0.02}, THREADS: {}, BATCH: {}}
    steps = _drive(planner, truth)
    placement_steps = [values for category, values in steps if category == PLACEMENT]
    assert len(placement_steps) >= 2
    assert any("ambiguous" in line for line in planner.log)


def test_threads_first_when_processor_carries_most_work() -> None:
    planner = Planner(_structure(count_max=25, top=25, uses_experts=True), start_layers=25,
                      start_threads=6, start_batch=512, processor_share=lambda layers: 0.8)
    truth = {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {9: 0.08, 8: 0.05, 10: 0.09}, BATCH: {}}
    steps = _drive(planner, truth)
    after_placement = [category for category, _ in steps if category != PLACEMENT]
    assert after_placement[0] == THREADS
    # Clear thread winner refined one step either side (finds 10).
    assert planner.combined().get("threads") == 10


def test_batch_first_when_card_heavy_and_threads_skipped_below_a_fifth() -> None:
    planner = Planner(_structure(), start_layers=33, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.1)
    order = planner.category_order(33)
    assert order == [BATCH]
    assert "threads deferred" in planner.log[-1]


def test_question_length_limits_batch_but_keeps_current_and_fallback() -> None:
    # Long question of 400 pieces: anything above 512 collapses to 512, so 1024/2048 are
    # not distinct; 512 (current) is the reference; preferred list is empty above it.
    planner = Planner(_structure(long_question_tokens=400), start_layers=33, start_threads=6,
                      start_batch=256, processor_share=lambda layers: 0.1)
    planner.placement = 33
    assert planner.batch_candidates() == [512]
    assert planner.batch_fallback() == [128]


def test_memory_filter_is_respected_and_small_batches_only_deprioritised() -> None:
    planner = Planner(_structure(), start_layers=33, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.1,
                      batch_fits=lambda layers, size: size <= 1024)
    planner.placement = 33
    assert planner.batch_candidates() == [1024]
    assert planner.batch_fallback() == [256]


def test_fallback_after_preferred_candidates_disappoint() -> None:
    planner = Planner(_structure(), start_layers=33, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.1)
    truth = {PLACEMENT: {0: -0.5, 1: -0.5}, THREADS: {}, BATCH: {2048: -0.05, 1024: -0.02, 256: 0.04}}
    steps = _drive(planner, truth)
    batch_steps = [values for category, values in steps if category == BATCH]
    assert batch_steps[0] == (2048, 1024)
    assert (256,) in batch_steps
    assert planner.combined().get("ubatch_size") == 256


def test_weak_readings_are_deferred_not_discarded() -> None:
    planner = Planner(_structure(count_max=25, top=25), start_layers=25, start_threads=6,
                      start_batch=512, processor_share=lambda layers: 0.8)
    truth = {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {8: 0.012, 9: 0.01}, BATCH: {1024: 0.10}}
    _drive(planner, truth, noise=0.01)
    assert "threads" not in planner.combined()
    deferred = {(r.category, r.value) for r in planner.deferred()}
    assert (THREADS, 8) in deferred


def test_strongest_follow_up_first() -> None:
    planner = Planner(_structure(count_max=25, top=25), start_layers=25, start_threads=6,
                      start_batch=256, processor_share=lambda layers: 0.8)
    truth = {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {9: 0.03}, BATCH: {1024: 0.20, 2048: 0.15}}
    steps = _drive(planner, truth)
    follow = [category for category, values in steps[3:]]
    # Batch's refinement (stronger) comes before threads' refinement.
    assert follow and follow[0] == BATCH


def test_placement_change_rechecks_only_affected_settings() -> None:
    planner = Planner(_structure(count_max=25, top=25), start_layers=25, start_threads=6,
                      start_batch=512, processor_share=lambda layers: 0.8)
    truth = {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {9: 0.05, 8: 0.04}, BATCH: {1024: 0.10}}
    _drive(planner, truth)
    steps = planner.placement_changed(22)
    categories = [step.category for step in steps]
    assert set(categories) <= {THREADS, BATCH}
    nxt = planner.next_step()
    assert nxt is not None and nxt.category in (THREADS, BATCH)
    assert "recheck" in nxt.reason


def test_missing_reading_is_never_treated_as_a_gain() -> None:
    planner = Planner(_structure(), start_layers=33, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.1)
    planner.next_step()
    planner.record(PLACEMENT, 0, None, 0.01)
    planner.record(PLACEMENT, 1, None, 0.01)
    assert planner.kind_of(PLACEMENT, 0) == "missing"
    assert planner.best(PLACEMENT)[0] == 33


def test_adjacent_extremes_within_noise_are_not_ambiguous() -> None:
    # CPU-only and output-only are both far better than the top and within noise of each
    # other: nothing lies between them, so no interior probes for a fully fitting model.
    planner = Planner(_structure(), start_layers=33, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.9)
    truth = {PLACEMENT: {0: 0.73, 1: 0.74}, THREADS: {}, BATCH: {}}
    steps = _drive(planner, truth)
    placement_steps = [values for category, values in steps if category == PLACEMENT]
    assert placement_steps == [(0, 1)]


# --------------------------------------------------------------------------- #
# Recorded first readings (Unbounded campaign, Oct 2026) as fixtures. Historical      #
# readings only show which candidates the planner would ask for and how it would       #
# classify them; they cannot show what a different order would have measured.         #
# --------------------------------------------------------------------------- #
def _replay(planner: Planner, recorded: dict[str, dict[int, float]], noise: dict[str, float]) -> list[tuple[str, int, float | None]]:
    asked = []
    for _ in range(30):
        step = planner.next_step()
        if step is None:
            break
        for value in step.values:
            gain = recorded[step.category].get(value)
            asked.append((step.category, value, gain))
            planner.record(step.category, value, gain, noise[step.question])
    return asked


def test_replay_llama_101_top_wins_batch_1024_is_deferred() -> None:
    planner = Planner(_structure(count_max=29, top=29, long_question_tokens=1850), start_layers=23,
                      start_threads=None, start_batch=None,
                      processor_share=lambda layers: 0.07 if layers >= 29 else 0.3)
    recorded = {PLACEMENT: {29: 0.171, 0: -0.80, 1: -0.78}, THREADS: {}, BATCH: {1024: 0.019, 2048: 0.005}}
    asked = _replay(planner, recorded, {"short": 0.017, "long": 0.013})
    assert planner.combined() == {"gpu_offload_layers": 29}
    assert (BATCH, 1024, 0.019) in asked
    assert any(r.category == BATCH and r.value == 1024 for r in planner.deferred())


def test_replay_granite_moe_102_cpu_then_threads_then_batch() -> None:
    planner = Planner(_structure(count_max=33, top=33, uses_experts=True, long_question_tokens=1700),
                      start_layers=33, start_threads=None, start_batch=None,
                      processor_share=lambda layers: 1.0 if layers == 0 else 0.3)
    recorded = {PLACEMENT: {0: 0.97, 1: 0.90}, THREADS: {8: 0.01, 10: 0.096, 9: 0.036, 11: 0.0},
                BATCH: {1024: 0.142, 2048: 0.10}}
    asked = _replay(planner, recorded, {"short": 0.02, "long": 0.013})
    categories = [category for category, _, _ in asked]
    assert categories.index(THREADS) < categories.index(BATCH)
    assert planner.combined() == {"gpu_offload_layers": 0, "threads": 10, "ubatch_size": 1024}


def test_replay_qwen_105_interior_count_is_probed_but_weak() -> None:
    planner = Planner(_structure(count_max=33, top=14, long_question_tokens=420), start_layers=8,
                      start_threads=12, start_batch=256, processor_share=lambda layers: 0.6)
    recorded = {PLACEMENT: {14: 0.009, 0: -0.45, 1: -0.42, 12: -0.001, 10: 0.021, 9: -0.01, 11: -0.042},
                THREADS: {8: 0.026, 10: -0.01}, BATCH: {512: -0.34, 128: -0.08}}
    asked = _replay(planner, recorded, {"short": 0.013, "long": 0.02})
    placement_values = [value for category, value, _ in asked if category == PLACEMENT]
    assert 10 in placement_values  # reached by the partial-offload probes
    # +2.1% at a 1.3% reading noise is within two widths: deferred, not in the candidate.
    assert "gpu_offload_layers" not in planner.combined()
    assert any(r.category == PLACEMENT and r.value == 10 for r in planner.deferred())
    # The long question (~420 pieces) is shorter than 512: 1024 and 2048 are not distinct
    # from 512, so only 512 is preferred; 64/128 stay available as the fallback.
    batch_values = [value for category, value, _ in asked if category == BATCH]
    assert batch_values[0] == 512 and 1024 not in batch_values


# --------------------------------------------------------------------------- #
# Review 8: evidence scoped to its setup; best so far; weak readings eligible       #
# --------------------------------------------------------------------------- #
def test_review8_old_third_place_reading_is_never_chosen_after_a_placement_change() -> None:
    planner = Planner(_structure(count_max=25, top=25), start_layers=25, start_threads=6,
                      start_batch=512, processor_share=lambda layers: 0.8)
    truth = {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {8: 0.04, 10: 0.08, 9: 0.06, 11: 0.05}, BATCH: {}}
    _drive(planner, truth)
    assert planner.chosen[THREADS] == 10
    steps = planner.placement_changed(22)
    assert [s.category for s in steps] == [THREADS]
    recheck = planner.next_step()
    assert recheck is not None and set(recheck.values) == {10, 9} and recheck.base[0] == 22
    for value in recheck.values:  # both lose at the new placement
        planner.record(THREADS, value, -0.03, 0.01)
    # 8 and 11 were never measured at 22 layers: they are not chosen.
    assert planner.chosen[THREADS] == 6
    assert "threads" not in planner.combined()


def test_threads_are_rechecked_after_batch_changes_their_setup() -> None:
    planner = Planner(_structure(count_max=25, top=25), start_layers=25, start_threads=6,
                      start_batch=512, processor_share=lambda layers: 0.8)
    truth = {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {10: 0.08, 8: 0.03, 9: 0.05, 11: 0.04},
             BATCH: {1024: 0.12, 2048: 0.05}}
    _drive(planner, truth)
    assert planner.chosen[BATCH] == 1024
    # Threads were first chosen at batch 512; after batch 1024 was chosen they were
    # measured again at batch 1024 (bounded: once).
    rechecked = [r for r in planner.readings[THREADS] if r.setup[2] == 1024]
    assert rechecked
    assert any("recheck" in line for line in planner.log)


def test_review8_best_so_far_includes_a_clear_placement_before_it_is_settled() -> None:
    planner = Planner(_structure(), start_layers=8, start_threads=6, start_batch=512,
                      processor_share=lambda layers: 0.1)
    step = planner.next_step()
    assert step is not None
    planner.record(PLACEMENT, 33, 0.175, 0.017)
    # Exploration may stop here: the winner is already the best-so-far setup.
    assert planner.combined() == {"gpu_offload_layers": 33}


def test_review8_repeated_weak_reading_can_become_clear() -> None:
    planner = Planner(_structure(count_max=25, top=25), start_layers=25, start_threads=6,
                      start_batch=512, processor_share=lambda layers: 0.8, smallest_gain=0.015)
    truth = {PLACEMENT: {0: -0.2, 1: -0.2}, THREADS: {10: 0.03}, BATCH: {}}
    _drive(planner, truth, noise=0.017)
    weak = planner.weak_candidates()
    assert weak and weak[0].value == 10
    planner.record(THREADS, 10, 0.03, 0.017, weak[0].setup)
    reading = planner.reading(THREADS, 10)
    assert reading is not None and reading.count == 2 and reading.kind(0.015) == "clear"
    assert planner.chosen[THREADS] == 10
