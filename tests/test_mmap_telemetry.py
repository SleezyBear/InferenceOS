"""Phase 17 — effective mmap-state observability.

The effective weight-load mode is observed only from llama.cpp's own startup log
evidence (``load_tensors: ... (load_mode = mmap|none)``). It is deliberately never
inferred from the configured ``RuntimeParams.use_mmap``, launch arguments, or OS
memory statistics; missing/unrecognized evidence stays ``None`` (unknown), never
"disabled". No mmap behavior or optimization eligibility is changed here.
"""

from __future__ import annotations

import os
import types
from pathlib import Path
from typing import Any, cast

from inferenceos.backends.llama_cpp_adapter import LlamaCppAdapter
from inferenceos.backends.llama_cpp_toolchain import (
    parse_effective_threads,
    parse_load_mode,
)
from inferenceos.backends.resident_server import LlamaServerResidentBackend
from inferenceos.interfaces.backend_adapter import BackendTelemetry, RuntimeParams
from tests.resident_helpers import synthetic_model

MMAP_LINE = (
    "0.00.854.975 I load_tensors: loading model tensors, this can take a while... "
    "(load_mode = mmap)"
)
NONE_LINE = (
    "0.00.844.153 I load_tensors: loading model tensors, this can take a while... "
    "(load_mode = none)"
)
# Default server verbosity: no load-mode evidence line at all.
DEFAULT_LOG = (
    "0.00.077.463 I srv    load_model: loading model 'm.gguf'\n"
    "0.01.264.551 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 512\n"
)
THREADS_LINE = "0.00.9 I llama threadpool init, n_threads = 6"


# --------------------------------------------------------------------------- #
# 1 / 2 / 3 / 4. Log parsing (fixtures; no live model)                          #
# --------------------------------------------------------------------------- #
def test_explicit_mmap_evidence_is_true() -> None:
    assert parse_load_mode(MMAP_LINE) is True


def test_explicit_non_mmap_evidence_is_false() -> None:
    assert parse_load_mode(NONE_LINE) is False


def test_missing_evidence_is_unknown_not_disabled() -> None:
    assert parse_load_mode("") is None
    assert parse_load_mode(DEFAULT_LOG) is None


def test_malformed_or_unexpected_evidence_is_unknown() -> None:
    assert parse_load_mode("load_tensors: (load_mode = )") is None
    assert parse_load_mode("load_tensors: (load_mode = weird)") is None
    assert parse_load_mode("load_tensors: load_mode = mmap") is None  # no parentheses
    assert parse_load_mode("load_mode: mmap") is None
    assert parse_load_mode("(load_mode = mmap") is None  # truncated


def test_evidence_is_case_insensitive_and_last_load_wins() -> None:
    assert parse_load_mode("(LOAD_MODE = MMAP)") is True
    assert parse_load_mode("(Load_Mode = None)") is False
    assert parse_load_mode(f"{NONE_LINE}\nreload\n{MMAP_LINE}") is True
    assert parse_load_mode(f"{MMAP_LINE}\nreload\n{NONE_LINE}") is False


def test_parse_load_mode_coexists_with_thread_parsing() -> None:
    combined = f"{DEFAULT_LOG}\n{THREADS_LINE}\n{MMAP_LINE}"
    assert parse_effective_threads(combined) == 6
    assert parse_load_mode(combined) is True


# --------------------------------------------------------------------------- #
# 5. Configuration alone does not establish effective state                     #
# --------------------------------------------------------------------------- #
def test_telemetry_defaults_to_unknown() -> None:
    assert BackendTelemetry().mmap_effective is None


def test_configured_use_mmap_does_not_populate_effective_state() -> None:
    # A backend that logged no load-mode evidence must report unknown, regardless of
    # the requested (configured) use_mmap value.
    for configured in (True, False, None):
        telemetry = _resident_telemetry(DEFAULT_LOG, RuntimeParams(use_mmap=configured))
        assert telemetry.mmap_effective is None


# --------------------------------------------------------------------------- #
# 6. Propagation to the consumer (resident backend + CLI adapter)               #
# --------------------------------------------------------------------------- #
def test_resident_telemetry_reports_effective_mmap() -> None:
    assert _resident_telemetry(MMAP_LINE, RuntimeParams()).mmap_effective is True
    assert _resident_telemetry(NONE_LINE, RuntimeParams()).mmap_effective is False
    assert _resident_telemetry(DEFAULT_LOG, RuntimeParams()).mmap_effective is None


def test_effective_state_is_distinct_from_configured_state() -> None:
    # Requesting mmap off while the backend logged mmap on -> effective is True.
    telemetry = _resident_telemetry(MMAP_LINE, RuntimeParams(use_mmap=False))
    assert telemetry.mmap_effective is True
    # Requesting mmap on while the backend logged non-mmap -> effective is False.
    telemetry = _resident_telemetry(NONE_LINE, RuntimeParams(use_mmap=True))
    assert telemetry.mmap_effective is False


def test_cli_adapter_telemetry_reports_effective_mmap() -> None:
    adapter = LlamaCppAdapter.__new__(LlamaCppAdapter)
    adapter._params = RuntimeParams(use_mmap=True)
    adapter._last_device = None
    adapter._last_run = cast(
        Any, types.SimpleNamespace(stderr=NONE_LINE, memory=None)
    )
    telemetry = adapter.telemetry()
    assert telemetry.mmap_effective is False

    adapter._last_run = cast(
        Any, types.SimpleNamespace(stderr=DEFAULT_LOG, memory=None)
    )
    assert adapter.telemetry().mmap_effective is None


# --------------------------------------------------------------------------- #
# Helpers / fixtures                                                           #
# --------------------------------------------------------------------------- #
class _FakeServerProcess:
    """Process factory double exposing a scripted startup log (no real server)."""

    log = ""
    pid_value = os.getpid()

    def __init__(self, command: list[str]) -> None:
        self.command = command

    @property
    def pid(self) -> int:
        return _FakeServerProcess.pid_value

    def start(self) -> int:
        return _FakeServerProcess.pid_value

    def wait_ready(self, client: Any, *, timeout_s: float, poll_s: float = 0.25) -> float:
        return 0.0

    def log_text(self) -> str:
        return _FakeServerProcess.log

    def identity(self) -> None:
        return None

    def is_alive(self) -> bool:
        return True

    def stop(self, *, timeout_s: float = 10.0) -> tuple[int | None, str, bool]:
        return 0, "process_tree", True


def _resident_telemetry(log: str, params: RuntimeParams) -> BackendTelemetry:
    import tempfile

    _FakeServerProcess.log = log
    path = Path(tempfile.gettempdir()) / "inferenceos-mmap-telemetry-probe.gguf"
    path.write_bytes(b"\x00")
    backend = LlamaServerResidentBackend(
        server=Path("unused"),
        process_factory=_FakeServerProcess,  # type: ignore[arg-type]
    )
    try:
        backend.start(synthetic_model(str(path)), params)
        return backend.telemetry()
    finally:
        backend._cleanup_key_file()  # noqa: SLF001 - remove the temp API-key file
        try:
            path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - defensive
            pass
