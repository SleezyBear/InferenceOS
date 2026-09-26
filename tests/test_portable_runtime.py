"""Portable runtime/toolchain fixes: executable names, process groups, key perms."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inferenceos.backends import resident_server
from inferenceos.backends.llama_cpp_resident_adapter import LlamaCppResidentAdapter
from inferenceos.backends.llama_cpp_toolchain import (
    default_cli_name,
    resolve_cli_executable,
)
from inferenceos.backends.llama_server_toolchain import (
    default_server_name,
    resolve_server_executable,
    server_process_names,
)
from inferenceos.backends.process_control import (
    subprocess_session_kwargs,
    terminate_process_group,
)
from inferenceos.config import Settings
from inferenceos.native import backend_library_names
from tests.resident_helpers import FakeResidentBackend


def test_executable_name_selection_windows_vs_unix() -> None:
    assert default_server_name(platform="win32") == "llama-server.exe"
    assert default_server_name(platform="linux") == "llama-server"
    assert default_server_name(platform="darwin") == "llama-server"
    assert default_cli_name(platform="win32") == "llama-cli.exe"
    assert default_cli_name(platform="linux") == "llama-cli"
    assert default_cli_name(platform="darwin") == "llama-cli"


def test_resolve_server_executable_prefers_existing_then_platform_default(
    tmp_path: Path,
) -> None:
    # Nothing present -> platform-appropriate default (never a hard-coded .exe).
    assert resolve_server_executable(tmp_path, platform="linux") == tmp_path / "llama-server"
    assert resolve_server_executable(tmp_path, platform="win32") == tmp_path / "llama-server.exe"
    # Existing unix binary wins (SERVER_NAMES behavior preserved).
    unix = tmp_path / "llama-server"
    unix.write_text("", encoding="utf-8")
    assert resolve_server_executable(tmp_path, platform="linux") == unix
    # Existing Windows binary wins.
    win = tmp_path / "llama-server.exe"
    win.write_text("", encoding="utf-8")
    assert resolve_server_executable(tmp_path, platform="win32") == win


def test_resolve_cli_executable(tmp_path: Path) -> None:
    assert resolve_cli_executable(tmp_path, platform="linux") == tmp_path / "llama-cli"
    assert resolve_cli_executable(tmp_path, platform="win32") == tmp_path / "llama-cli.exe"


def test_server_process_names_are_narrow_and_platform_neutral() -> None:
    assert server_process_names(platform="win32") == ("llama-server.exe",)
    assert server_process_names(platform="linux") == ("llama-server",)
    assert server_process_names(platform="darwin") == ("llama-server",)


def test_posix_process_group_setup_kwargs() -> None:
    assert subprocess_session_kwargs(platform="linux") == {"start_new_session": True}
    assert subprocess_session_kwargs(platform="darwin") == {"start_new_session": True}
    assert subprocess_session_kwargs(platform="win32") == {}
    # Windows always uses Job Objects; group termination is a no-op there.
    assert subprocess_session_kwargs() == {}
    assert terminate_process_group(os.getpid()) is False


def test_unix_api_key_file_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_chmod(path: object, mode: int) -> None:
        calls.append((str(path), mode))

    monkeypatch.setattr(os, "chmod", fake_chmod)
    key = tmp_path / "inferenceos-key-test.txt"
    key.write_text("secret", encoding="utf-8")

    resident_server.restrict_api_key_file(key, platform="linux")
    assert calls == [(str(key), 0o600)]

    calls.clear()
    resident_server.restrict_api_key_file(key, platform="darwin")
    assert calls == [(str(key), 0o600)]

    calls.clear()
    resident_server.restrict_api_key_file(key, platform="win32")
    assert calls == []  # Windows behavior unchanged


def test_backend_library_names_are_platform_specific() -> None:
    assert backend_library_names("vulkan", platform="win32") == ("vulkan-1.dll",)
    assert "libvulkan.so.1" in backend_library_names("vulkan", platform="linux")
    assert "libMoltenVK.dylib" in backend_library_names("vulkan", platform="darwin")
    assert backend_library_names("cuda", platform="win32") == ("nvcuda.dll",)
    assert "libcuda.so.1" in backend_library_names("cuda", platform="linux")
    assert backend_library_names("rocm", platform="win32") == ("amdhip64.dll", "hiprtc.dll")
    assert "libamdhip64.so" in backend_library_names("rocm", platform="linux")
    assert backend_library_names("unknown", platform="linux") == ()


def test_windows_resident_adapter_default_executable(tmp_path: Path) -> None:
    # On this Windows host, an existing llama-server.exe resolves without the
    # adapter hard-coding the name itself.
    exe = tmp_path / "llama-server.exe"
    exe.write_text("", encoding="utf-8")
    adapter = LlamaCppResidentAdapter(
        settings=Settings(llama_cpp_dir=tmp_path), backend=FakeResidentBackend()
    )
    assert adapter.server_path == exe


def test_windows_characterize_gpus_uses_windows_library_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inferenceos.hardware.gpu as gpu_module
    from inferenceos.hardware.cpu import build_cpu_profile

    cpu = build_cpu_profile(
        [{"Name": "Test CPU", "NumberOfCores": 4, "NumberOfLogicalProcessors": 8}],
        cpuid={"sse2": True},
        os_flags={"sse2": True},
    )
    probed: list[str] = []

    class _FakeNative:
        def library_present(self, name: str) -> bool:
            probed.append(name)
            return False

    monkeypatch.setattr(gpu_module, "get_platform", lambda: _FakeNative())
    gpu_module.characterize_gpus(cpu, vulkan_devices=[], video_controllers=[])
    # Windows naming is preserved (unchanged behavior on this platform).
    assert "vulkan-1.dll" in probed
    assert "nvcuda.dll" in probed
    assert "amdhip64.dll" in probed
