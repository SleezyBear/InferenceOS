"""Finding the real layer top by load-only tries (prompt B)."""

from __future__ import annotations

from collections.abc import Callable

from inferenceos.tune.layer_top import LoadAttempt, find_layer_top


def _quiet(_message: str) -> None:
    return None


def _fits_through(top: int) -> Callable[[int], LoadAttempt]:
    def try_load(layers: int) -> LoadAttempt:
        return LoadAttempt(
            loaded=True,
            fits=layers <= top,
            used_bytes=1_500_000_000,
            free_bytes=2_500_000_000,
        )

    return try_load


def test_all_layers_load() -> None:
    result = find_layer_top(
        n_layers=24,
        estimate=8,
        try_load=lambda layers: LoadAttempt(loaded=True, fits=True),
        say=_quiet,
    )
    assert result.top == 24
    assert result.used_estimate is False


def test_halving_finds_the_exact_top() -> None:
    result = find_layer_top(
        n_layers=32,
        estimate=16,
        try_load=_fits_through(20),
        say=_quiet,
    )
    assert result.top == 20
    assert any(entry["layers"] == 20 for entry in result.tries)


def test_nothing_above_zero_loads() -> None:
    result = find_layer_top(
        n_layers=32,
        estimate=16,
        try_load=lambda layers: LoadAttempt(loaded=False, fits=False),
        say=_quiet,
    )
    assert result.top == 0


def test_out_of_memory_counts_as_not_fitting() -> None:
    # The real try_load turns an out-of-memory failure into fits=False; the top is below.
    result = find_layer_top(
        n_layers=32,
        estimate=8,
        try_load=_fits_through(12),
        say=_quiet,
    )
    assert result.top == 12


def test_failed_load_records_the_full_error() -> None:
    message = "ResidentBackendError: a resident server is already running"
    result = find_layer_top(
        n_layers=16,
        estimate=8,
        try_load=lambda layers: LoadAttempt(
            loaded=False, fits=False, error=message
        ),
        say=_quiet,
    )
    assert result.top == 0
    assert result.tries
    assert all(entry["loaded"] is False for entry in result.tries)
    assert all(entry["error"] == message for entry in result.tries)


def test_records_used_and_free_memory_correctly() -> None:
    def try_load(layers: int) -> LoadAttempt:
        return LoadAttempt(
            loaded=True,
            fits=layers <= 20,
            used_bytes=3_000_000_000,
            free_bytes=1_000_000_000,
        )

    result = find_layer_top(
        n_layers=32, estimate=16, try_load=try_load, say=_quiet
    )
    assert result.tries
    assert all(entry["used_bytes"] == 3_000_000_000 for entry in result.tries)
    assert all(entry["free_bytes"] == 1_000_000_000 for entry in result.tries)
    # free = total - used, and the two fields are no longer swapped.
    assert all(entry["used_bytes"] != entry["free_bytes"] for entry in result.tries)


def test_unreadable_memory_falls_back_to_the_estimate() -> None:
    called: list[int] = []

    def try_load(layers: int) -> LoadAttempt:
        called.append(layers)
        return LoadAttempt(loaded=True, fits=True)

    result = find_layer_top(
        n_layers=32,
        estimate=10,
        try_load=try_load,
        memory_readable=False,
        say=_quiet,
    )
    assert result.top == 10
    assert result.used_estimate is True
    assert called == []


def test_missing_try_load_falls_back_to_the_estimate() -> None:
    result = find_layer_top(n_layers=32, estimate=9, try_load=None, say=_quiet)
    assert result.top == 9
    assert result.used_estimate is True
