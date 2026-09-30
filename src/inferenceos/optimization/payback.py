"""The payback check (step 4): is a change worth the time still to be spent on it?

This replaces the old fixed guess (``benefit_scale_s`` = 60 s), which charged time
already spent searching against a fixed 60-second scale and so rejected every real
layer gain on the laptop.

The rule: a change that makes answering ``gain`` faster saves ``gain / (1 + gain)`` of
the answering time the setup is expected to get. It pays back when that is more than
the time still to be spent (for example a restart pause while the model runs, or a
test that has not run yet). Time already spent is never counted again.

The expected answering time is a setting: about 10 hours for a setup found before
loading (it is reused every time after), about 1 hour for a change found while the
model runs (it only helps the rest of the session).
"""

from __future__ import annotations

from collections.abc import Mapping

from inferenceos.config import Settings
from inferenceos.contracts.enums import Situation
from inferenceos.contracts.goal import PaybackCheck


def answering_seconds(settings: Settings, situation: Situation) -> float:
    """The expected answering time (seconds) a change will benefit from."""
    hours = (
        settings.payback_answering_hours_before_loading
        if situation is Situation.BEFORE_LOADING
        else settings.payback_answering_hours_while_running
    )
    if float(hours) < 0.0:
        raise ValueError("payback answering hours must not be negative")
    return float(hours) * 3600.0


def time_saved_s(gain: float, answering_s: float) -> float:
    """Seconds saved by a change that makes answering ``gain`` faster."""
    if gain <= 0.0:
        return 0.0
    return answering_s * gain / (1.0 + gain)


def payback_check(
    *,
    gain: float | None,
    cost_terms: Mapping[str, float],
    situation: Situation,
    settings: Settings,
) -> PaybackCheck:
    """Compare the time a change saves with the time still to be spent on it.

    ``cost_terms`` must hold only costs still to come. ``pays_back`` is ``None`` when
    the gain is unknown (nothing is guessed).
    """
    answering = answering_seconds(settings, situation)
    costs = {name: max(0.0, float(value)) for name, value in cost_terms.items()}
    cost = sum(costs.values())
    if gain is None:
        return PaybackCheck(
            situation=situation,
            gain=None,
            answering_s=answering,
            cost_s=cost,
            cost_terms=costs,
            pays_back=None,
            rationale=["gain unknown: payback not judged (never guessed)"],
        )
    saved = time_saved_s(gain, answering)
    pays = saved > cost
    return PaybackCheck(
        situation=situation,
        gain=gain,
        answering_s=answering,
        time_saved_s=saved,
        cost_s=cost,
        cost_terms=costs,
        pays_back=pays,
        rationale=[
            f"{situation.value}: {gain:+.1%} over {answering / 3600.0:g} h of answering "
            f"saves {saved:.0f} s; still to spend {cost:.0f} s {costs}; "
            f"{'pays back' if pays else 'does not pay back'}"
        ],
    )


__all__ = ["answering_seconds", "payback_check", "time_saved_s"]
