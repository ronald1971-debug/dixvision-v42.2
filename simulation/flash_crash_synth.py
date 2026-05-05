"""SIM-09 flash_crash_synth — synthetic flash-crash step function.

Models a sudden, sharp price dislocation (5-30% in seconds) followed
by a partial mean-reversion recovery. Used by the meta-controller's
strategy arena to stress-test position management under tail events
that historical replay rarely surfaces.

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): caller's average entry price.
* ``position_size_usd`` (float, > 0): notional of caller's open
  position.
* ``side`` (str, one of ``"long"`` / ``"short"``): position side.
* ``max_drop_pct`` (float, in ``(0, 1]``): maximum trough drop as a
  fraction of ``entry_price`` for a long position. Reused symmetrically
  for shorts (max upward spike).
* ``recovery_pct`` (float, in ``[0, 1]``): fraction of the drop that
  is recovered by the time the position is closed. ``0`` = position
  is closed at the trough; ``1`` = full mean-reversion to entry.

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. Jitter ``max_drop_pct`` by ``drop_jitter`` to model variability
   between flash-crash realities (e.g. one reality reaches a 12%
   trough; another only 8%).
3. Jitter ``recovery_pct`` by ``recovery_jitter``.
4. Compute trough and terminal price for the side.
5. P&L = ``size * (terminal - entry) / entry`` for long;
   sign-flipped for short.
6. Terminal drawdown = ``size * (entry - trough) / entry`` for long;
   sign-flipped for short.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.2 in the canonical-rebuild walk).
* manifest.md §549 (simulation/ tree).
* full_feature_spec §624 (SIM-XX module list).
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class FlashCrashConfig:
    """Versioned configuration for SIM-09 flash_crash_synth.

    Attributes:
        drop_jitter: Symmetric jitter applied to ``max_drop_pct``.
            Default 0.2 (i.e. realised trough varies ±20% around the
            scenario-supplied max).
        recovery_jitter: Symmetric jitter applied to
            ``recovery_pct``. Default 0.15.
    """

    drop_jitter: float = 0.2
    recovery_jitter: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.drop_jitter <= 1.0:
            raise ValueError(
                "FlashCrashConfig.drop_jitter must be in [0, 1], "
                f"got {self.drop_jitter!r}"
            )
        if not 0.0 <= self.recovery_jitter <= 1.0:
            raise ValueError(
                "FlashCrashConfig.recovery_jitter must be in [0, 1], "
                f"got {self.recovery_jitter!r}"
            )


_LONG = "long"
_SHORT = "short"


def _require_positive(meta: dict[str, Any], key: str) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    if not v > 0.0:
        raise ValueError(f"meta[{key!r}] must be > 0, got {v!r}")
    return v


def _require_pct_open(meta: dict[str, Any], key: str) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    if not 0.0 < v <= 1.0:
        raise ValueError(f"meta[{key!r}] must be in (0, 1], got {v!r}")
    return v


def _require_pct_closed(meta: dict[str, Any], key: str) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"meta[{key!r}] must be in [0, 1], got {v!r}")
    return v


def _require_side(meta: dict[str, Any]) -> str:
    if "side" not in meta:
        raise ValueError("RealityScenario.meta missing required key 'side'")
    side = meta["side"]
    if side not in (_LONG, _SHORT):
        raise ValueError(
            f"meta['side'] must be 'long' or 'short', got {side!r}"
        )
    return side


class FlashCrashSynth:
    """SIM-09 deterministic flash-crash step function."""

    def __init__(self, config: FlashCrashConfig | None = None) -> None:
        self._config = config or FlashCrashConfig()

    @property
    def config(self) -> FlashCrashConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        entry = _require_positive(meta, "entry_price")
        size_usd = _require_positive(meta, "position_size_usd")
        side = _require_side(meta)
        max_drop = _require_pct_open(meta, "max_drop_pct")
        recovery = _require_pct_closed(meta, "recovery_pct")

        cfg = self._config
        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        drop_factor = 1.0 + (rng.random() - 0.5) * 2.0 * cfg.drop_jitter
        recovery_factor = 1.0 + (rng.random() - 0.5) * 2.0 * cfg.recovery_jitter

        realised_drop = max(0.0, min(1.0, max_drop * drop_factor))
        realised_recovery = max(0.0, min(1.0, recovery * recovery_factor))

        if side == _LONG:
            trough = entry * (1.0 - realised_drop)
            terminal = trough + realised_recovery * (entry - trough)
            pnl = size_usd * (terminal - entry) / entry
            drawdown = size_usd * (entry - trough) / entry
            rule_fired = "long_flash_crash"
        else:
            spike = entry * (1.0 + realised_drop)
            terminal = spike - realised_recovery * (spike - entry)
            pnl = size_usd * (entry - terminal) / entry
            drawdown = size_usd * (spike - entry) / entry
            rule_fired = "short_flash_spike"

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=drawdown,
            fills_count=2,
            rule_fired=rule_fired,
        )


__all__ = ["FlashCrashSynth", "FlashCrashConfig"]
