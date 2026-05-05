"""SIM-17 regime_switch_sim — two-regime Markov walk step function.

Models the failure mode where market regime flips during the
execution window: a strategy that was sized for the prevailing
regime gets caught when the market flips into the other one
mid-fill. The walk alternates between two regimes (A and B) with
a per-step switch probability; each regime has its own drift and
volatility, so the realised price path depends on the regime
trajectory the seed produces.

Distinct from SIM-15 ``slippage_walk`` (single-regime geometric
walk with constant drift/std) and SIM-12 ``crowd_density``
(squeeze-vs-no-squeeze threshold gate): this module is the only
SIM that explicitly tracks the *number of regime transitions*
the seed observed, and emits ``rule_fired`` accordingly so a
downstream auditor can distinguish "stable" from "switching"
trajectories.

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): price at step 0.
* ``order_size_usd`` (float, > 0): notional being held through
  the walk.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).
* ``num_steps`` (int, in [1, ``max_steps``]): number of steps
  to simulate.
* ``switch_probability`` (float, in [0, 1]): per-step probability
  of flipping the active regime.
* ``regime_a_drift`` (float, in [-0.05, 0.05]): per-step signed
  drift fraction in regime A.
* ``regime_a_std`` (float, in [0, 0.5]): per-step volatility
  fraction in regime A.
* ``regime_b_drift`` (float, in [-0.05, 0.05]): per-step signed
  drift fraction in regime B.
* ``regime_b_std`` (float, in [0, 0.5]): per-step volatility
  fraction in regime B.
* ``starting_regime`` (str, one of ``"A"`` / ``"B"``): regime
  at step 0.

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. Walk the price across ``num_steps`` steps:

   a. Roll a ``rng.random()``; if it is ``< switch_probability``
      flip the active regime.
   b. Compute drift = ``rng.gauss(regime_drift, regime_std)``.
   c. Update ``price *= 1 + drift``.
3. ``delta_usd = order_size_usd * (price - entry) / entry``.
4. Buy fill: ``pnl = delta_usd``; Sell fill: ``pnl = -delta_usd``.

   (Sign convention: a buyer who paid ``entry`` and now holds at
   ``price`` has a notional MTM of ``+delta_usd``.)
5. ``terminal_drawdown_usd = max(0, -pnl)``.
6. ``fills_count = num_steps`` (one MTM tick per step).
7. ``rule_fired`` records the trajectory:

   * ``"stable_a"`` if 0 transitions and started in A.
   * ``"stable_b"`` if 0 transitions and started in B.
   * ``"switching_few"`` if 1..floor(num_steps / 4) transitions.
   * ``"switching_many"`` if > floor(num_steps / 4) transitions.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.
NaN-safe with the +inf gate from PR #263 review applied to all
positive-float validators.

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.10 in the canonical-rebuild walk).
* manifest.md §549 (simulation/ tree).
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class RegimeSwitchSimConfig:
    """Versioned configuration for SIM-17 regime_switch_sim.

    Attributes:
        max_steps: Hard cap on ``num_steps`` to keep the loop
            bounded. Default 10_000.
    """

    max_steps: int = 10_000

    def __post_init__(self) -> None:
        if not 0 < self.max_steps <= 1_000_000:
            raise ValueError(
                "RegimeSwitchSimConfig.max_steps must be in "
                f"(0, 1_000_000], got {self.max_steps!r}"
            )


_BUY = "buy"
_SELL = "sell"
_REGIME_A = "A"
_REGIME_B = "B"


def _require_positive_float(meta: dict[str, Any], key: str) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    # NaN-safe (`not v > 0.0`) + finite (`math.isfinite`) — PR #263 review.
    if not v > 0.0 or not math.isfinite(v):
        raise ValueError(f"meta[{key!r}] must be > 0 and finite, got {v!r}")
    return v


def _require_positive_int(meta: dict[str, Any], key: str) -> int:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"meta[{key!r}] must be int, got {type(raw).__name__}")
    if raw <= 0:
        raise ValueError(f"meta[{key!r}] must be > 0, got {raw!r}")
    return raw


def _require_bounded_float(
    meta: dict[str, Any], key: str, low: float, high: float
) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    # `not low <= v <= high` rejects NaN under IEEE 754 and rejects
    # +/-inf when the bounds are finite.
    if not low <= v <= high:
        raise ValueError(
            f"meta[{key!r}] must be in [{low}, {high}], got {v!r}"
        )
    return v


def _require_side(meta: dict[str, Any]) -> str:
    if "side" not in meta:
        raise ValueError("RealityScenario.meta missing required key 'side'")
    side = meta["side"]
    if side not in (_BUY, _SELL):
        raise ValueError(
            f"meta['side'] must be 'buy' or 'sell', got {side!r}"
        )
    return side


def _require_regime(meta: dict[str, Any]) -> str:
    if "starting_regime" not in meta:
        raise ValueError(
            "RealityScenario.meta missing required key 'starting_regime'"
        )
    regime = meta["starting_regime"]
    if regime not in (_REGIME_A, _REGIME_B):
        raise ValueError(
            f"meta['starting_regime'] must be 'A' or 'B', got {regime!r}"
        )
    return regime


class RegimeSwitchSim:
    """SIM-17 deterministic two-regime Markov walk step fn."""

    def __init__(self, config: RegimeSwitchSimConfig | None = None) -> None:
        self._config = config or RegimeSwitchSimConfig()

    @property
    def config(self) -> RegimeSwitchSimConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        cfg = self._config

        entry = _require_positive_float(meta, "entry_price")
        size_usd = _require_positive_float(meta, "order_size_usd")
        num_steps = _require_positive_int(meta, "num_steps")
        if num_steps > cfg.max_steps:
            raise ValueError(
                f"meta['num_steps'] {num_steps} exceeds "
                f"max_steps {cfg.max_steps}"
            )
        switch_prob = _require_bounded_float(meta, "switch_probability", 0.0, 1.0)
        a_drift = _require_bounded_float(meta, "regime_a_drift", -0.05, 0.05)
        a_std = _require_bounded_float(meta, "regime_a_std", 0.0, 0.5)
        b_drift = _require_bounded_float(meta, "regime_b_drift", -0.05, 0.05)
        b_std = _require_bounded_float(meta, "regime_b_std", 0.0, 0.5)
        side = _require_side(meta)
        regime = _require_regime(meta)

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        price = entry
        transitions = 0
        for _ in range(num_steps):
            if rng.random() < switch_prob:
                regime = _REGIME_B if regime == _REGIME_A else _REGIME_A
                transitions += 1
            if regime == _REGIME_A:
                drift = rng.gauss(a_drift, a_std)
            else:
                drift = rng.gauss(b_drift, b_std)
            price = price * (1.0 + drift)

        delta_usd = size_usd * (price - entry) / entry
        if side == _BUY:
            pnl = delta_usd
        else:
            pnl = -delta_usd
        drawdown = max(0.0, -pnl)

        # rule_fired: encode regime trajectory shape.
        switching_threshold = num_steps // 4
        if transitions == 0:
            starting = meta["starting_regime"]
            rule_fired = f"stable_{starting.lower()}"
        elif transitions <= switching_threshold:
            rule_fired = "switching_few"
        else:
            rule_fired = "switching_many"

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=drawdown,
            fills_count=num_steps,
            rule_fired=rule_fired,
        )


__all__ = ["RegimeSwitchSim", "RegimeSwitchSimConfig"]
