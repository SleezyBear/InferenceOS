"""Compare engine observations with requested runtime settings."""
from __future__ import annotations

from dataclasses import dataclass

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.interfaces.backend_adapter import BackendTelemetry


@dataclass(frozen=True)
class ObservedConfigurationBaseline:
    """Detached engine evidence, separate from the owner's automatic requests.

    Previously known defaults must remain equal and observable after restoration.
    An initially contradicted/unverified request cannot become proof retroactively.
    Unknown unrequested defaults remain unknown; no extra probe is performed.
    """

    observed: RuntimeConfiguration
    problems: tuple[str, ...] = ()

    @classmethod
    def capture(cls, requested: RuntimeConfiguration, telemetry: BackendTelemetry | None) -> ObservedConfigurationBaseline:
        shown, contradictions, unknown = observed_configuration(requested, telemetry)
        return cls(shown.model_copy(deep=True), tuple(
            [f"baseline contradiction: {item}" for item in contradictions]
            + [f"baseline unverified setting: {item}" for item in unknown]))

    def problems_after(self, restored: RuntimeConfiguration) -> list[str]:
        problems = list(self.problems)
        for name in RuntimeConfiguration.model_fields:
            before = getattr(self.observed, name)
            if before is None:
                continue
            after = getattr(restored, name)
            if after is None:
                problems.append(f"baseline {name}={before!r} is no longer observable")
            elif after != before:
                problems.append(f"baseline {name}: was {before!r}, restored engine showed {after!r}")
        return problems


def observed_configuration(
    requested: RuntimeConfiguration, telemetry: BackendTelemetry | None
) -> tuple[RuntimeConfiguration, list[str], list[str]]:
    """What the engine showed about ``requested``, from its own evidence only (review R3).

    Returns ``(observed, contradictions, unverified)``: ``observed`` carries only the
    values the engine showed (everything else None -- never the request echoed back);
    a contradiction names a requested setting the engine showed differently; an
    unverified entry names a requested setting the engine gave no evidence for.
    Settings checked: layers on the card (the engine caps a request at the model's
    layer count), threads, the weight-load mode, flash attention and the
    conversation-memory formats (each explicitly requested side on its own). Every
    other requested setting is unverified."""
    shown: dict[str, object] = {}
    contradictions: list[str] = []
    unverified: list[str] = []
    checked: set[str] = set()

    def compare(name: str, want: object, seen: object, expected: object | None = None) -> None:
        checked.add(name)
        if want is None:
            if seen is not None:
                shown[name] = seen
            return
        if seen is None:
            unverified.append(name)
            return
        shown[name] = seen
        target = want if expected is None else expected
        if seen != target:
            contradictions.append(f"{name}: requested {want!r}, engine showed {seen!r}")

    t = telemetry
    layers = t.gpu_layers_offloaded if t is not None else None
    total = t.gpu_layers_total if t is not None else None
    want_layers = requested.gpu_offload_layers
    # Match the resident backend's supported automatic requests in every consumer,
    # not only its startup check. Preserve the actual chosen values in `shown`.
    if want_layers is not None and want_layers < 0:
        want_layers = None
    expected_layers = (
        min(max(want_layers, 0), total) if want_layers is not None and total is not None else None
    )
    compare("gpu_offload_layers", want_layers, layers, expected_layers)
    want_threads = None if requested.threads == 0 else requested.threads
    compare("threads", want_threads, t.effective_threads if t is not None else None)
    compare("use_mmap", requested.use_mmap, t.mmap_effective if t is not None else None)
    observed_items = dict(t.observed) if t is not None else {}
    # The general telemetry context field is the requested value. Only the
    # server's own per-slot capacity establishes the effective request limit.
    context = observed_items.get("context_length")
    context = context if type(context) is int and context > 0 else None
    compare("context_length", requested.context_length,
            context, max(context, requested.context_length) if context is not None
            and requested.context_length is not None else None)
    flash = observed_items.get("flash_attention")
    compare("flash_attention", requested.flash_attention, flash if isinstance(flash, bool) else None)

    # Conversation-memory formats (review of the R3 fix): each side the request names
    # explicitly (per side, or through the shared setting) is compared on its own; a
    # side left to the engine's default is not required to be anything. A side the
    # engine showed is recorded whatever it is; a mismatch on one side is a
    # contradiction even when the other side was not shown.
    checked.update(("kv_cache_type", "kv_cache_type_k", "kv_cache_type_v"))
    keys, values = requested.effective_kv_types()
    for side, want, field_name in (("k", keys, "kv_cache_type_k"), ("v", values, "kv_cache_type_v")):
        seen = observed_items.get(f"cache_type_{side}")
        if isinstance(seen, str):
            shown[field_name] = seen
        if want is None:
            continue
        source = field_name if getattr(requested, field_name) is not None else "kv_cache_type"
        if seen is None:
            if source not in unverified:
                unverified.append(source)
        elif seen != want:
            contradictions.append(f"{source} ({'keys' if side == 'k' else 'values'}): "
                                  f"requested {want!r}, engine showed {seen!r}")

    for name in type(requested).model_fields:
        if name in checked or getattr(requested, name) is None:
            continue
        unverified.append(name)
    return RuntimeConfiguration.model_validate(shown), contradictions, unverified

