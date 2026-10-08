"""Load-to-load spread per model, setup family and question, kept between tunes.

The number of runs per measurement depends on how much a reading moves from one fresh
load to the next. Review 4 (4 Oct 2026): one laptop-wide constant (1.2%) is not a
dependable input, so each fair comparison's own readings of the unchanged setup give an
estimate (``measure.between_load_spread``), saved here and reused. The constant stays
only as a fallback, and reports say when it was used.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

#: How many recent estimates are kept per key.
KEEP = 10
#: How many estimates are needed before they replace the fallback value.
MIN_ESTIMATES = 2


def family(gpu_layers: int | None) -> str:
    """Setups that behave alike for noise: all on the processor, or partly on the card."""
    return "card" if gpu_layers else "processor"


class NoiseStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _read(self) -> dict[str, list[float]]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): [float(v) for v in values if isinstance(v, (int, float))]
            for key, values in data.items()
            if isinstance(values, list)
        }

    def estimate(self, key: str) -> float | None:
        """Median of the saved estimates for ``key``; None until at least
        ``MIN_ESTIMATES`` comparisons gave one (one comparison is too noisy alone)."""
        values = self._read().get(key)
        return float(statistics.median(values)) if values and len(values) >= MIN_ESTIMATES else None

    def add(self, key: str, value: float) -> None:
        data = self._read()
        data[key] = ([*data.get(key, []), float(value)])[-KEEP:]
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            return


def noise_key(model_name: str, gpu_layers: int | None, workload_id: str, variant: str = "") -> str:
    """``variant`` names anything else that changes how the setup runs (explicit piece
    placement, conversation-memory formats), so spreads are never shared across them."""
    base = f"{model_name}|{family(gpu_layers)}|{workload_id}"
    return f"{base}|{variant}" if variant else base


__all__ = ["KEEP", "NoiseStore", "family", "noise_key"]
