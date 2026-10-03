"""The ``start`` command: run a model with its saved setup and keep it running (step 5).

Before starting, a quick check compares the machine and the model file with how they
were when the setup was tuned. Anything that changed is reported, and the model is
started anyway (the user's decision) with a suggestion to run ``tune`` again. A
different model file is never started with another model's setup.

The model is reachable from this computer only, and a new password is made at every
start. The password is kept in an owner-only file that is deleted when the model
stops; only the file's location is printed.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from inferenceos.contracts.experiment import RuntimeConfiguration
from inferenceos.contracts.model import ModelProfile
from inferenceos.contracts.tune import MachineCheck, SavedSetup, StartCheck
from inferenceos.experiment.restoration import runtime_params_from
from inferenceos.interfaces.backend_adapter import RuntimeParams
from inferenceos.tune.machine import machine_changes
from inferenceos.tune.profiles import SetupStore, power_key, power_words, same_model_file

LOOPBACK = "127.0.0.1"


def quick_check(
    *,
    store: SetupStore,
    model_path: Path,
    context_length: int | None,
    machine: MachineCheck,
) -> StartCheck:
    """Pick the saved setup for now and say what changed since it was tuned."""
    power = power_key(machine.power_source)
    setups = store.for_model(model_path)
    if context_length is not None:
        setups = [item for item in setups if item.context_length == context_length]
    if not setups:
        where = f" at context {context_length}" if context_length is not None else ""
        return StartCheck(
            power_source=power,
            problems=[f"no saved setup for this model{where}; run `tune` first"],
        )
    contexts = sorted({item.context_length for item in setups})
    if context_length is None and len(contexts) > 1:
        return StartCheck(
            power_source=power,
            problems=[
                "this model has saved setups for several context sizes ("
                + ", ".join(str(value) for value in contexts)
                + "); choose one with --context"
            ],
        )
    notes: list[str] = []
    chosen = next((item for item in setups if power_key(item.power_source) is power), None)
    matched = chosen is not None
    if chosen is None:
        chosen = setups[0]
        notes.append(
            f"no saved setup for {power_words(power)} yet; using the one for "
            f"{power_words(chosen.power_source)}. Run `tune` {power_words(power)} to make one."
        )
    same, why = same_model_file(chosen.model, model_path)
    problems = [] if same else [why + "; run `tune` again for this model"]
    changes = machine_changes(chosen.machine, machine)
    if changes:
        notes.append("Something changed since this setup was tuned; consider running `tune` again.")
    return StartCheck(
        setup=chosen,
        power_source=power,
        matched_power_source=matched,
        model_matches=same,
        changes=changes,
        problems=problems,
        notes=[*notes, why] if same else notes,
    )


def port_is_free(port: int, host: str = LOOPBACK) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


class RunningBackend(Protocol):
    """The part of the resident llama-server backend ``start`` uses."""

    def start(self, model: ModelProfile, params: RuntimeParams) -> object: ...
    def stop(self) -> object: ...
    def is_running(self) -> bool: ...
    @property
    def base_url(self) -> str | None: ...
    @property
    def api_key_file(self) -> Path | None: ...


@dataclass
class ServeOutcome:
    started: bool
    stopped_cleanly: bool
    reason: str
    base_url: str | None = None
    password_file: str | None = None


def serve_until_stopped(
    *,
    setup: SavedSetup,
    model: ModelProfile,
    backend: RunningBackend,
    say: Callable[[str], None] = print,
    keep_running: Callable[[], bool] | None = None,
    poll_s: float = 1.0,
) -> ServeOutcome:
    """Start the model with ``setup`` and keep it running until Ctrl+C (or it stops)."""
    config: RuntimeConfiguration = setup.configuration
    params = runtime_params_from(config)
    if params.context_length != setup.context_length:
        return ServeOutcome(False, True, "saved setup's context size does not match; refused")
    try:
        backend.start(model, params)
    except Exception as exc:  # noqa: BLE001 - reported to the user
        return ServeOutcome(False, True, f"the model did not start: {type(exc).__name__}: {exc}")
    url = backend.base_url
    key_file = backend.api_key_file
    say(f"Running {setup.model_name} at {url} (this computer only).")
    say(
        "Password for this run is in: "
        f"{key_file} (send it as 'Authorization: Bearer <password>'; "
        "a new one is made at every start, and the file is deleted when you stop)."
    )
    say("Press Ctrl+C to stop.")
    reason = "stopped by you"
    try:
        while True:
            if keep_running is not None and not keep_running():
                break
            if not backend.is_running():
                reason = "the model stopped on its own"
                break
            time.sleep(poll_s)
    except KeyboardInterrupt:
        reason = "stopped by you"
    finally:
        clean = True
        try:
            if backend.is_running():
                backend.stop()
        except Exception:  # noqa: BLE001 - reported, never hidden
            clean = False
    return ServeOutcome(
        True,
        clean,
        reason,
        base_url=url,
        password_file=str(key_file) if key_file else None,
    )


def describe_setup(setup: SavedSetup) -> str:
    config = setup.configuration
    threads = config.threads if config.threads is not None else "program default"
    gain = (
        f", confirmed {setup.confirmed_gain:+.1%} over the starting setup"
        if setup.confirmed_gain is not None
        else ""
    )
    extra = ""
    if config.threads_batch is not None:
        extra += f", {config.threads_batch} reading threads"
    if config.flash_attention is not None:
        extra += ", flash attention " + ("on" if config.flash_attention else "off")
    if config.ubatch_size is not None:
        extra += f", prompt batch {config.ubatch_size}"
    return (
        f"{config.gpu_offload_layers} graphics card layers, {threads} threads{extra}, "
        f"context {config.context_length} (saved {setup.created_at:%Y-%m-%d} "
        f"{power_words(setup.power_source)}{gain})"
    )


__all__ = [
    "LOOPBACK",
    "RunningBackend",
    "ServeOutcome",
    "describe_setup",
    "port_is_free",
    "quick_check",
    "serve_until_stopped",
]

