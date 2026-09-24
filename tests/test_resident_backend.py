"""Resident server toolchain + platform process-control tests (no real server)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from inferenceos.backends.llama_server_toolchain import (
    ServerOptions,
    build_server_command,
    generate_api_key,
    pick_free_port,
)
from inferenceos.backends.process_control import (
    create_process_guard,
    identity_matches,
    process_identity,
    resource_snapshot,
    terminate_process_tree,
    wait_process_exit,
)


def test_build_server_command_hides_key_and_never_uses_prompt(tmp_path: Path) -> None:
    key_file = tmp_path / "key.txt"
    command = build_server_command(
        Path("llama-server"),
        ServerOptions(model_path=tmp_path / "m.gguf", n_gpu_layers=4, context_length=2048),
        host="127.0.0.1",
        port=18080,
        api_key_file=key_file,
    )
    joined = " ".join(command)
    assert "--api-key-file" in command
    assert "--api-key" not in command
    assert "--host 127.0.0.1" in joined
    assert "--port 18080" in joined
    assert "-p" not in command
    assert "secret" not in joined


def test_generate_api_key_is_random_and_long() -> None:
    first = generate_api_key()
    second = generate_api_key()
    assert first != second
    assert len(first) >= 32


def test_pick_free_port_is_valid() -> None:
    port = pick_free_port("127.0.0.1")
    assert 0 < port < 65536


def test_process_identity_and_resource_snapshot() -> None:
    pid = os.getpid()
    identity = process_identity(pid)
    assert identity is not None
    assert identity.pid == pid
    assert identity.create_time > 0
    assert identity_matches(identity)
    snapshot = resource_snapshot(pid)
    assert snapshot.exists is True
    assert (snapshot.rss_bytes or 0) > 0


def test_process_guard_is_available() -> None:
    guard = create_process_guard()
    assert guard.kind
    guard.close()


def test_terminate_process_tree_kills_a_real_child() -> None:
    code = "import time; time.sleep(60)"
    proc = subprocess.Popen([sys.executable, "-c", code])
    try:
        assert terminate_process_tree(proc.pid, timeout_s=5.0)
        assert wait_process_exit(proc.pid, timeout_s=5.0)
    finally:
        if proc.poll() is None:  # pragma: no cover - cleanup
            proc.kill()
