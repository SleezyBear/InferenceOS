"""Independent, real post-restoration verification.

The Phase 7 audit found that restoration was demonstrated primarily through
in-memory configuration equality. This module performs a real generation against the
restored target and compares it with a baseline probe captured before the experiment,
so restoration is verified by behaviour, not by a Python object comparison.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from inferenceos.backends.observed import ObservedConfigurationBaseline, observed_configuration

from inferenceos.contracts.experiment import (
    RestorationVerification,
    RuntimeConfiguration,
)


def runtime_configuration_from(params: RuntimeParams) -> RuntimeConfiguration:
    """Inverse of :func:`runtime_params_from` for reading a live adapter's config."""
    return RuntimeConfiguration(
        threads=params.threads,
        threads_batch=params.threads_batch,
        engine_switches=params.engine_switches,
        batch_size=params.batch_size,
        ubatch_size=params.ubatch_size,
        gpu_offload_layers=params.gpu_offload_layers,
        use_mmap=params.use_mmap,
        context_length=params.context_length,
        flash_attention=params.flash_attention,
        kv_cache_type=params.kv_cache_type,
        kv_cache_type_k=params.kv_cache_type_k,
        kv_cache_type_v=params.kv_cache_type_v,
        placement_overrides=params.placement_overrides,
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
        threads_batch=config.threads_batch,
        engine_switches=config.engine_switches,
        batch_size=config.batch_size,
        ubatch_size=config.ubatch_size,
        gpu_offload_layers=config.gpu_offload_layers,
        use_mmap=config.use_mmap,
        context_length=config.context_length,
        flash_attention=config.flash_attention,
        kv_cache_type=config.kv_cache_type,
        kv_cache_type_k=config.kv_cache_type_k,
        kv_cache_type_v=config.kv_cache_type_v,
        placement_overrides=config.placement_overrides,
        device=config.device,
    )


class RestorationVerifier(Protocol):
    """Captures a baseline probe and verifies the restored target against it."""

    def capture(self, configuration: RuntimeConfiguration) -> None:
        ...

    def verify(self, expected: RuntimeConfiguration) -> RestorationVerification:
        ...


def _hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8", "replace")).hexdigest()[:16]


@dataclass(frozen=True)
class _ProbeReading:
    text: str
    latency_s: float
    rss_bytes: int | None
    observed: RuntimeConfiguration
    contradictions: tuple[str, ...]
    unverified: tuple[str, ...]


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
        self._baseline_observed: RuntimeConfiguration | None = None
        self._baseline_problems: tuple[str, ...] = ()

    def _ensure_loaded(self, configuration: RuntimeConfiguration) -> None:
        params = runtime_params_from(configuration)
        if self._adapter.is_loaded:
            self._adapter.set_runtime_params(params)
        else:
            self._adapter.load(self._model, params)

    def _probe_once(self, configuration: RuntimeConfiguration) -> _ProbeReading:
        failure: BaseException | None = None
        try:
            # A failed load may already own resources; its adapter remains the
            # cleanup owner. ResidentMeasurementAdapter deliberately keeps unload
            # inert while its workload owns the live target.
            self._ensure_loaded(configuration)
            started = time.perf_counter()
            result = self._adapter.generate(self._probe)
            latency = time.perf_counter() - started
            observed, contradictions, unknown = observed_configuration(configuration, self._adapter.telemetry())
            return _ProbeReading(result.text, latency, result.peak_rss_bytes, observed,
                                 tuple(contradictions), tuple(unknown))
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                self._adapter.unload()
            except Exception as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"restoration probe unload also failed: {cleanup_error}")

    def capture(self, configuration: RuntimeConfiguration) -> None:
        # A verifier can be reused. Failed capture belongs to this experiment;
        # it must not inherit the successful probe of an earlier experiment.
        self._baseline_hash = None
        self._baseline_latency = None
        self._baseline_rss = None
        self._baseline_configuration = None
        self._baseline_observed = None
        self._baseline_problems = ()
        try:
            reading = self._probe_once(configuration)
        except Exception as exc:  # noqa: BLE001 - capture failure surfaces via verify
            logger.warning("restoration baseline capture failed: %s", exc)
            return
        self._baseline_hash = _hash(reading.text)
        self._baseline_latency = reading.latency_s
        self._baseline_rss = reading.rss_bytes
        self._baseline_configuration = configuration
        self._baseline_observed = reading.observed
        self._baseline_problems = tuple(
            [f"baseline contradiction: {item}" for item in reading.contradictions]
            + [f"baseline unverified setting: {item}" for item in reading.unverified]
        )

    def verify(self, expected: RuntimeConfiguration) -> RestorationVerification:
        measured_at = datetime.now(timezone.utc)
        try:
            reading = self._probe_once(expected)
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
        contradictions = list(reading.contradictions)
        unknown = list(reading.unverified)
        if self._baseline_observed is not None:
            # Defaults are still actual baseline settings when the engine showed
            # them. An unchanged request cannot excuse changed or lost evidence.
            contradictions.extend(ObservedConfigurationBaseline(
                self._baseline_observed).problems_after(reading.observed))
        running = True
        restored_hash = _hash(reading.text)
        probe_match = (
            self._baseline_hash is not None and restored_hash == self._baseline_hash
        )
        latency_ratio = (
            reading.latency_s / self._baseline_latency
            if self._baseline_latency and self._baseline_latency > 0
            else None
        )
        within = latency_ratio is None or latency_ratio <= self._latency_tolerance
        configuration_match = self._baseline_configuration is not None and (
            self._baseline_configuration == expected
        ) and not self._baseline_problems and not contradictions and not unknown
        verified = bool(running and probe_match and configuration_match and reading.text)
        return RestorationVerification(
            attempted=True,
            running=running,
            configuration_match=configuration_match,
            probe_match=probe_match,
            verified=verified,
            baseline_probe_hash=self._baseline_hash,
            restored_probe_hash=restored_hash,
            baseline_latency_s=self._baseline_latency,
            restored_latency_s=reading.latency_s,
            latency_ratio=latency_ratio,
            within_tolerance=within,
            reason=(
                "restored target answered with observed baseline settings and "
                "reproduced the baseline probe"
                if verified
                else "restored target did not reproduce the baseline behaviour"
            ),
            measured_at=measured_at,
            notes=[
                "verification uses a real generation against the restored target, not "
                "configuration-object equality",
                "running records successful operation during the probe; the adapter's unload policy applies afterwards",
                "latency tolerance is a diagnostic and does not gate restoration",
                *self._baseline_problems,
                *[f"contradiction: {item}" for item in contradictions],
                *[f"unverified setting: {item}" for item in unknown],
            ],
        )
