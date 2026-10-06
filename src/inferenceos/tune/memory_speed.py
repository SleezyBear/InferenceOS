"""How fast this computer's processor can move memory (no extra packages needed).

Copies a large block of memory with the C library's ``memmove`` (through ``ctypes``,
which lets other Python threads run meanwhile) from 1 up to N threads at once and
reports the best rate. A copy reads and writes every byte, so the traffic is twice the
block size. This is a ceiling for how fast the processor part of a model can write
words: each written word reads all the model weights that run on the processor once.

Used by the diagnosis (``tune/diagnose.py``). It is a rough, practical ceiling, not a
datasheet figure; the diagnosis treats it as such.
"""

from __future__ import annotations

import ctypes
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(frozen=True)
class MemorySpeed:
    #: Best total traffic, bytes per second (reads + writes).
    best_bytes_per_s: float
    #: Threads that gave the best rate.
    best_threads: int
    #: Rate per thread count tried, bytes per second.
    by_threads: dict[int, float] = field(default_factory=dict)
    block_bytes: int = 0

    def words(self) -> str:
        return (
            f"memory moves about {self.best_bytes_per_s / 1e9:.1f} GB/s at best "
            f"({self.best_threads} threads copying at once)"
        )

    def record(self) -> dict[str, object]:
        return {
            "best_gb_per_s": round(self.best_bytes_per_s / 1e9, 2),
            "best_threads": self.best_threads,
            "gb_per_s_by_threads": {k: round(v / 1e9, 2) for k, v in self.by_threads.items()},
            "block_mib": self.block_bytes // (1024 * 1024),
        }


def _copy_rate(threads: int, block_bytes: int, repeats: int) -> float:
    per_thread = max(block_bytes // threads, 1 << 20)
    sources = [ctypes.create_string_buffer(per_thread) for _ in range(threads)]
    targets = [ctypes.create_string_buffer(per_thread) for _ in range(threads)]
    for buffer in (*sources, *targets):  # touch every page once
        ctypes.memset(buffer, 1, per_thread)
    best = 0.0
    for _ in range(repeats):
        barrier = threading.Barrier(threads + 1)

        def work(index: int) -> None:
            barrier.wait()
            ctypes.memmove(targets[index], sources[index], per_thread)

        workers = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
        for worker in workers:
            worker.start()
        started = time.perf_counter()
        barrier.wait()
        for worker in workers:
            worker.join()
        elapsed = time.perf_counter() - started
        if elapsed > 0:
            best = max(best, 2.0 * per_thread * threads / elapsed)
    return best


def measure_memory_speed(
    *, max_threads: int, block_bytes: int = 512 * 1024 * 1024, repeats: int = 5
) -> MemorySpeed:
    """The best copy rate from 1 to ``max_threads`` threads (powers of two and the max)."""
    counts = sorted({1, 2, 4, 8, max_threads} & set(range(1, max(max_threads, 1) + 1)))
    rates = {count: _copy_rate(count, block_bytes, repeats) for count in counts}
    best_threads = max(rates, key=lambda count: rates[count])
    return MemorySpeed(
        best_bytes_per_s=rates[best_threads],
        best_threads=best_threads,
        by_threads=rates,
        block_bytes=block_bytes,
    )


def cached_memory_speed(
    path: Path, machine_key: str, *, max_threads: int, max_age_days: int = 30
) -> MemorySpeed:
    """The memory speed for this machine, measured once and kept in ``path``.

    Measured again when the machine changed (``machine_key``: processor, memory size
    and modules) or the reading is older than ``max_age_days``.
    """
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        when = datetime.fromisoformat(str(saved["measured_at"]))
        if (
            saved.get("machine") == machine_key
            and datetime.now(timezone.utc) - when < timedelta(days=max_age_days)
        ):
            return MemorySpeed(
                best_bytes_per_s=float(saved["best_bytes_per_s"]),
                best_threads=int(saved["best_threads"]),
                by_threads={int(k): float(v) for k, v in saved["by_threads"].items()},
                block_bytes=int(saved.get("block_bytes", 0)),
            )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    speed = measure_memory_speed(max_threads=max_threads)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "machine": machine_key,
                "measured_at": datetime.now(timezone.utc).isoformat(),
                "best_bytes_per_s": speed.best_bytes_per_s,
                "best_threads": speed.best_threads,
                "by_threads": {str(k): v for k, v in speed.by_threads.items()},
                "block_bytes": speed.block_bytes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return speed


__all__ = ["MemorySpeed", "cached_memory_speed", "measure_memory_speed"]
