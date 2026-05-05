"""SIM-19 drawdown_walk — peak-to-trough drawdown-tracking step function.

Models the failure mode where the *path* of the price during an
active execution window matters as much as the terminal value:
many strategies are stopped out by an interim drawdown even when
the terminal pnl is positive. This step function walks the price
deterministically and records the **maximum peak-to-trough
drawdown observed during the walk**, separately from the terminal
pnl.

Distinct from earlier SIM modules:

* **vs SIM-15 ``slippage_walk``** — slippage_walk reports
  ``terminal_drawdown_usd`` as ``max(0, -terminal_pnl)`` (it is
  only the *terminal* underwater value); this one reports the
  **maximum running drawdown** observed at any step, which can
  be larger than the terminal underwater value.
* **vs SIM-17 ``regime_switch_sim``** — regime_switch reports the
  same terminal-underwater field; drawdown_walk explicitly tracks
  the running peak and current price each step.
* **Unique to SIM-19:** ``rule_fired`` classifies the *depth* of
  the max running drawdown observed:
  ``shallow`` (< 1%) / ``moderate`` (1-5%) / ``deep`` (5-15%) /
  ``catastrophic`` (≥ 15%).

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): price at step 0.
* ``order_size_usd`` (float, > 0): notional held through the walk.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).
* ``num_steps`` (int, in [1, ``max_steps``]): walk length.
* ``per_step_drift`` (float, in [-0.01, 0.01]): per-step drift mean.
* ``per_step_std`` (float, in [0, 0.1]): per-step volatility.

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. Walk:

   For each step ``i`` in ``range(num_steps)``:

   a. ``price *= 1 + rng.gauss(drift, std)``.
   b. For a buy fill: ``running_peak = max(running_peak, price)``;
      ``running_dd = max(running_dd, running_peak - price)``.
   c. For a sell fill: ``running_trough = min(running_trough, price)``;
      ``running_dd_unit = max(running_dd_unit, price - running_trough)``.
3. ``delta_usd = order_size_usd * (price - entry) / entry``.
4. Buy fill: ``pnl = delta_usd``; Sell fill: ``pnl = -delta_usd``.
5. ``terminal_drawdown_usd = order_size_usd * running_dd / entry``.
6. ``rule_fired`` (depth thresholds on the max running drawdown
   relative to ``entry_price``):

   * ``"shallow"`` if ratio < 0.01.
   * ``"moderate"`` if 0.01 ≤ ratio < 0.05.
   * ``"deep"`` if 0.05 ≤ ratio < 0.15.
   * ``"catastrophic"`` if ratio ≥ 0.15.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.
NaN-safe + +inf-safe (PR #263 review pattern applied from start).

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.12 in the canonical-rebuild walk).
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class DrawdownWalkConfig:
    """Versioned configuration for SIM-19 drawdown_walk.

    Attributes:
        max_steps: Hard cap on ``num_steps`` to keep the loop
            bounded. Default 10_000.
    """

    max_steps: int = 10_000

    def __post_init__(self) -> None:
        if not 0 < self.max_steps <= 1_000_000:
            raise ValueError(
                "DrawdownWalkConfig.max_steps must be in "
                f"(0, 1_000_000], got {self.max_steps!r}"
            )


_BUY = "buy"
_SELL = "sell"


def _require_positive_float(meta: dict[str, Any], key: str) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
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


class DrawdownWalk:
    """SIM-19 deterministic peak-to-trough drawdown-tracking walk."""

    def __init__(self, config: DrawdownWalkConfig | None = None) -> None:
        self._config = config or DrawdownWalkConfig()

    @property
    def config(self) -> DrawdownWalkConfig:
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
        drift = _require_bounded_float(meta, "per_step_drift", -0.01, 0.01)
        std = _require_bounded_float(meta, "per_step_std", 0.0, 0.1)
        side = _require_side(meta)

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        price = entry
        # For a buy: we're long, so drawdown = peak - current (price
        # falling from peak hurts). For a sell: we're short, so the
        # equivalent "drawdown" is current - trough (price rising
        # from trough hurts). Track both with side-aware accounting.
        running_peak = entry
        running_trough = entry
        running_dd_unit = 0.0  # Max underwater in *price* units.

        for _ in range(num_steps):
            price = price * (1.0 + rng.gauss(drift, std))
            if side == _BUY:
                if price > running_peak:
                    running_peak = price
                underwater = running_peak - price
            else:
                if price < running_trough:
                    running_trough = price
                underwater = price - running_trough
            if underwater > running_dd_unit:
                running_dd_unit = underwater

        delta_usd = size_usd * (price - entry) / entry
        if side == _BUY:
            pnl = delta_usd
        else:
            pnl = -delta_usd

        # Convert max running drawdown from price-units to USD-units
        # via the same notional/entry scaling used for pnl.
        terminal_drawdown = size_usd * running_dd_unit / entry

        # rule_fired classifies the depth of the *max running
        # drawdown* relative to entry_price.
        ratio = running_dd_unit / entry
        if ratio < 0.01:
            rule_fired = "shallow"
        elif ratio < 0.05:
            rule_fired = "moderate"
        elif ratio < 0.15:
            rule_fired = "deep"
        else:
            rule_fired = "catastrophic"

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=terminal_drawdown,
            fills_count=num_steps,
            rule_fired=rule_fired,
        )


__all__ = ["DrawdownWalk", "DrawdownWalkConfig"]
