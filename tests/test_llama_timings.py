"""Parsing of llama.cpp's authoritative per-phase timings and log stamps."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferenceos.backends.llama_cpp_toolchain import (
    build_run_command,
    parse_llama_timings,
    parse_log_seconds,
)

STDERR = (
    "0.00.070.817 I cmn  common_param: common_params_print_info: build 11139\n"
    "0.00.081.894 I srv    load_model: loading model 'm.gguf'\n"
    "0.01.988.928 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 512, kv_unified = 'false'\n"
    "0.02.906.004 I slot print_timing: id  0 | task 0 | prompt eval time =     439.01 ms /    35 tokens"
    " (   12.54 ms per token,    86.07 tokens per second)\n"
    "0.02.906.019 I slot print_timing: id  0 | task 0 |        eval time =     289.88 ms /     8 tokens"
    " (   36.23 ms per token,    27.60 tokens per second)\n"
    "0.02.906.024 I slot print_timing: id  0 | task 0 |       total time =     728.89 ms /    43 tokens\n"
)


def test_parse_log_seconds() -> None:
    assert parse_log_seconds("0.00.081.894") == pytest.approx(0.081894)
    assert parse_log_seconds("0.01.988.928") == pytest.approx(1.988928)
    assert parse_log_seconds("bad") is None


def test_parse_llama_timings_extracts_phases() -> None:
    timings = parse_llama_timings(STDERR)
    assert timings is not None
    assert timings.prompt_eval_s == pytest.approx(0.43901)
    assert timings.prompt_tokens == 35
    assert timings.prefill_tokens_per_sec == pytest.approx(86.07)
    assert timings.eval_s == pytest.approx(0.28988)
    assert timings.generated_tokens == 8
    assert timings.decode_tokens_per_sec == pytest.approx(27.60)
    assert timings.total_s == pytest.approx(0.72889)
    assert timings.total_tokens == 43
    assert timings.model_load_s == pytest.approx(1.988928 - 0.081894)
    assert timings.ttft_s == pytest.approx(0.43901 + 0.28988 / 8)


def test_parse_llama_timings_absent() -> None:
    assert parse_llama_timings("no timings here") is None


def test_build_run_command_log_verbosity(tmp_path: Path) -> None:
    command = build_run_command(
        Path("llama-cli"),
        tmp_path / "m.gguf",
        "hi",
        log_verbosity=4,
    )
    joined = " ".join(command)
    assert "-lv 4" in joined
    assert joined.split().count("-v") == 0
    verbose = build_run_command(Path("llama-cli"), tmp_path / "m.gguf", "hi", verbose=True)
    assert "-v" in verbose


def test_build_run_command_ignore_eos(tmp_path: Path) -> None:
    command = build_run_command(
        Path("llama-cli"), tmp_path / "m.gguf", "hi", ignore_eos=True
    )
    assert "--ignore-eos" in command
    default = build_run_command(Path("llama-cli"), tmp_path / "m.gguf", "hi")
    assert "--ignore-eos" not in default
