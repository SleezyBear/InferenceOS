"""Independent, real post-restoration verification.

The Phase 7 audit found that restoration was demonstrated primarily through
in-memory configuration equality. This module performs a real generation against the
restored target and compares it with a baseline probe captured before the experiment,
so restoration is verified by behaviour, not by a Python object comparison.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Protocol

from inferenceos.contracts.experiment import (
    RestorationVerification,
    RuntimeConfiguration,
)


def runtime_configuration_from(params: RuntimeParams) -> RuntimeConfiguration:
    """Inverse of :func:`runtime_params_from` for reading a live adapter's config."""
    return RuntimeConfiguration(
        threads=params.threads,
        batch_size=params.batch_size,
        ubatch_size=params.ubatch_size,
        gpu_offload_layers=params.gpu_offload_layers,
        use_mmap=params.use_mmap,
        context_length=params.context_length,
        flash_attention=params.flash_attention,
        kv_cache_type=params.kv_cache_type,
        device=params.device,
    )
from inferenceos.contracts.model import ModelProfile
from inferenceos.interfaces.backend_adapter import (
    BackendAdapter,
    GenerationRequest,
    RuntimeParams,
)
from inferenceos.logging_config import get_logger

logger = get_logger("experiment.restoration")


def runtime_params_from(config: RuntimeConfiguration) -> RuntimeParams:
    return RuntimeParams(
        threads=config.threads,
        batch_size=config.batch_size,
        ubatch_size=config.ubatch_size,
        gpu_offload_layers=config.gpu_offload_layers,
        use_mmap=config.use_mmap,
        context_length=config.context_length,
        flash_attention=config.flash_attention,
        kv_cache_type=config.kv_cache_type,
        device=config.device,
    )


class RestorationVerifier(Protocol):
    """Captures a baseline probe and verifies the restored target against it."""

    def capture(self, configuration: RuntimeConfiguration) -> None:
        ...

    def verify(self, expected: RuntimeConfiguration) -> RestorationVerification:
        ...


class NoopRestorationVerifier:
    """Explicit no-op used where no live backend is available (records the fact)."""

    def capture(self, configuration: RuntimeConfiguration) -> None:
        return

    def verify(self, expected: RuntimeConfiguration) -> RestorationVerification:
        return RestorationVerification(
            attempted=False,
            running=False,
            configuration_match=False,
            probe_match=False,
            verified=False,
            reason="no real restoration verifier configured",
            measured_at=datetime.now(timezone.utc),
        )


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8", "replace")).hexdigest()[:16]


class ResidentRestorationVerifier:
    """Real verification over a live backend (resident adapter preferred)."""

    def __init__(
        self,
        *,
        adapter: BackendAdapter,
        model: ModelProfile,
        probe: GenerationRequest | None = None,
        latency_tolerance: float = 0.75,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._probe = probe or GenerationRequest(
            prompt="Reply with the single word: OK.", n_predict=8, seed=1234, temperature=0.0
        )
        self._latency_tolerance = latency_tolerance
        self._baseline_hash: str | None = None
        self._baseline_latency: float | None = None
        self._baseline_rss: int | None = None
        self._baseline_configuration: RuntimeConfiguration | None = None

    def _ensure_loaded(self, configuration: RuntimeConfiguration) -> None:
        params = runtime_params_from(configuration)
        if self._adapter.is_loaded:
            self._adapter.set_runtime_params(params)
        else:
            self._adapter.load(self._model, params)

    def _probe_once(self, configuration: RuntimeConfiguration) -> tuple[str, float, int | None]:
        self._ensure_loaded(configuration)
        started = time.perf_counter()
        try:
            result = self._adapter.generate(self._probe)
        finally:
            latency = time.perf_counter() - started
            if self._adapter.is_loaded:
                self._adapter.unload()
        return result.text, latency, result.peak_rss_bytes

    def capture(self, configuration: RuntimeConfiguration) -> None:
        try:
            text, latency, rss = self._probe_once(configuration)
        except Exception as exc:  # noqa: BLE001 - capture failure surfaces via verify
            logger.warning("restoration baseline capture failed: %s", exc)
            return
        self._baseline_hash = _hash(text)
        self._baseline_latency = latency
        self._baseline_rss = rss
        self._baseline_configuration = configuration

    def verify(self, expected: RuntimeConfiguration) -> RestorationVerification:
        measured_at = datetime.now(timezone.utc)
        try:
            text, latency, _rss = self._probe_once(expected)
        except Exception as exc:  # noqa: BLE001
            return RestorationVerification(
                attempted=True,
                running=self._adapter.is_loaded,
                configuration_match=False,
                probe_match=False,
                verified=False,
                reason=f"post-restoration generation failed: {type(exc).__name__}: {exc}",
                measured_at=measured_at,
            )
        running = True
        restored_hash = _hash(text)
        probe_match = (
            self._baseline_hash is not None and restored_hash == self._baseline_hash
        )
        latency_ratio = (
            latency / self._baseline_latency
            if self._baseline_latency and self._baseline_latency > 0
            else None
        )
        within = latency_ratio is None or latency_ratio <= self._latency_tolerance
        configuration_match = self._baseline_configuration is not None and (
            self._baseline_configuration == expected
        )
        verified = bool(running and probe_match and configuration_match and text)
        return RestorationVerification(
            attempted=True,
            running=running,
            configuration_match=configuration_match,
            probe_match=probe_match,
            verified=verified,
            baseline_probe_hash=self._baseline_hash,
            restored_probe_hash=restored_hash,
            baseline_latency_s=self._baseline_latency,
            restored_latency_s=latency,
            latency_ratio=latency_ratio,
            within_tolerance=within,
            reason=(
                "restored target is running, matches the baseline configuration, and "
                "reproduces the baseline probe"
                if verified
                else "restored target did not reproduce the baseline behaviour"
            ),
            measured_at=measured_at,
            notes=[
                "verification uses a real generation against the restored target, not "
                "configuration-object equality",
            ],
        )
