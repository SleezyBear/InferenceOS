"""Read the dependencies/device of an already loaded, owned Windows engine.

No process is started here. Unknown or inaccessible mappings disable cost reuse.
Hashes are cached only within this tune and invalidated by file-stat changes.
This describes the observed loaded modules, not every possible future DLL/plugin.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time
from typing import Any, Callable

import psutil

from inferenceos.backends.process_control import identity_matches


class RuntimeIdentityReader:
    def __init__(self, target: Callable[[], Any], device: Callable[[], Any], *, gpu_count: int) -> None:
        self.target, self.device, self.gpu_count = target, device, gpu_count
        self.cache: dict[str, tuple[tuple[int, int], str]] = {}
        self.bundles: dict[Path, set[str]] = {}
        self.disabled = False

    def unchanged(self, runtime: dict[str, Any] | None = None) -> bool:
        try:
            expected = {} if runtime is None else {**runtime["modules"], **runtime["bundle"]}
            return (all(p in self.cache and self.cache[p][1] == h for p, h in expected.items())
                    and not self.disabled and all(self._bundle_paths(root) == paths for root, paths in self.bundles.items())
                    and all((Path(p).stat().st_size, Path(p).stat().st_mtime_ns) == stamp
                            for p, (stamp, _) in self.cache.items()))
        except (OSError, KeyError, TypeError):
            return False

    def __call__(self) -> dict[str, Any] | None:
        if self.disabled or sys.platform != "win32" or self.gpu_count != 1:
            return None  # multi-device placement and non-Windows dependencies unproven
        target, device = self.target(), self.device()
        if target is None or device is None or any(
                not isinstance(value, str) or not value.strip() or value.strip().lower() == "unknown"
                for value in (device.backend, device.device)):
            return None
        if not identity_matches(target):
            return None
        process = psutil.Process(target.pid)
        paths = self._paths(process)
        executable = str(Path(process.exe()).resolve())
        if executable not in paths or len(paths) < 2:
            return None
        bundle_paths = self._bundle_paths(Path(executable).parent)
        self.bundles[Path(executable).parent] = bundle_paths
        modules: dict[str, str] = {}
        deadline = time.perf_counter() + 2.0
        for name in paths | bundle_paths:
            path = Path(name)
            before = path.stat()
            stamp = (before.st_size, before.st_mtime_ns)
            old = self.cache.get(name)
            if old is None or old[0] != stamp:
                hasher = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        if time.perf_counter() > deadline:
                            self.disabled = True
                            raise OSError("runtime identity exceeded its two-second hashing allowance; disabled for this tune")
                        hasher.update(chunk)
                digest = hasher.hexdigest()
                after = path.stat()
                if stamp != (after.st_size, after.st_mtime_ns):
                    return None
                self.cache[name] = (stamp, digest)
            modules[name] = self.cache[name][1]
        if (not identity_matches(target) or self._paths(process) != paths or not self.unchanged()
                or self._bundle_paths(Path(executable).parent) != bundle_paths):
            return None
        return {"version": "loaded-runtime-1", "modules": {p: modules[p] for p in paths},
                "bundle": {p: modules[p] for p in bundle_paths}, "executable": executable,
                "backend": device.backend, "device": device.device}

    @staticmethod
    def _bundle_paths(root: Path) -> set[str]:
        return {str(p.resolve()) for p in root.rglob("*") if p.suffix.lower() in (".dll", ".exe") and p.is_file()}

    @staticmethod
    def _paths(process: Any) -> set[str]:
        return {str(Path(m.path).resolve()) for m in process.memory_maps()
                if Path(m.path).suffix.lower() in (".dll", ".exe")}
