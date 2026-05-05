"""SIM-16 order_book_decay — multi-level book-sweep step function.

Models the failure mode where a market sweep arrives at a stale
order book: each price level still posted has been decaying since
the most recent quote, with **inside levels decaying faster than
outer levels** (high-frequency makers flee aggressively at the
slightest pressure; passive limit-order workers leave deeper rest
intact). The strategy then walks the depleted book level by level,
paying the offset of each level it consumes.

Distinct from SIM-11 ``liquidity_decay`` (TWAP slicing through a
single inside-touch depth that decays over slices): this module
keeps depth uniform across slices but stratifies across N price
levels with **per-level** decay coefficients.

Inputs (read from ``RealityScenario.meta``):

* ``reference_price`` (float, > 0): mid-market price at quote time.
* ``order_size_usd`` (float, > 0): total notional being swept.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).
* ``num_levels`` (int, in [1, ``max_levels``]): number of price
  levels in the book on the relevant side.
* ``level_spacing_bps`` (float, in [0.1, 100]): spacing between
  adjacent levels expressed in basis points of the mid; level
  ``i`` (0 = inside) sits at offset ``(i + 1) * spacing`` bps from
  the mid.
* ``level_depth_usd`` (float, > 0): notional posted at each level
  before decay (uniform).
* ``decay_rate`` (float, in [0, 10]): exponential decay rate per
  unit ``elapsed_seconds`` at the inside level. Outer levels
  decay at ``decay_rate * (1 - i / num_levels)`` so the
  inside-faster invariant is enforced by construction.
* ``elapsed_seconds`` (float, in [0, 3600]): time since the book
  was last fresh.

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)`` (used only for tiny
   per-level depth jitter so two runs at the same seed are
   bit-identical).
2. For each level ``i`` in ``[0, num_levels)``:

   a. ``offset_frac = (i + 1) * level_spacing_bps / 10_000``.
   b. ``rate_i = decay_rate * (1 - i / num_levels)``;
      inside ``i = 0`` decays at the full rate, the deepest
      level decays at near zero.
   c. ``depth_alive = level_depth_usd * exp(-rate_i * elapsed)
      * (1 + 0.05 * rng.uniform(-1, 1))``;
      jitter clamped so ``depth_alive >= 0``.
   d. ``eaten = min(remaining, depth_alive)``.
   e. ``cost += eaten * offset_frac`` (USD).
   f. ``remaining -= eaten``;
      break early once ``remaining <= residual_epsilon_usd``.
3. ``fills_count = number of levels touched``.
4. ``pnl = -cost`` (always non-positive — we are paying to take
   the book).
5. ``terminal_drawdown_usd = cost``.
6. ``rule_fired = "fully_swept" | "book_too_thin"`` depending on
   whether residual remained.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.
NaN-safe (validators use ``not low <= v <= high`` and
``not v > 0.0`` patterns; PR #234 / PR #261 lesson applied).

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.9 in the canonical-rebuild walk).
* manifest.md §549 (simulation/ tree).
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class OrderBookDecayConfig:
    """Versioned configuration for SIM-16 order_book_decay.

    Attributes:
        max_levels: Hard cap on ``num_levels`` so the loop stays
            bounded. Default 10_000.
        residual_epsilon_usd: Sub-microcent residual treated as
            "fully swept" so dust doesn't trip ``book_too_thin``.
    """

    max_levels: int = 10_000
    residual_epsilon_usd: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.max_levels <= 1_000_000:
            raise ValueError(
                "OrderBookDecayConfig.max_levels must be in "
                f"(0, 1_000_000], got {self.max_levels!r}"
            )
        if not 0.0 <= self.residual_epsilon_usd <= 1.0:
            raise ValueError(
                "OrderBookDecayConfig.residual_epsilon_usd must be "
                f"in [0, 1], got {self.residual_epsilon_usd!r}"
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
    # `not v > 0.0` rejects NaN (NaN > 0 is False) — see PR #234 / #261.
    if not v > 0.0:
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
    # `not low <= v <= high` rejects NaN — IEEE 754 makes every
    # NaN comparison False.
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


class OrderBookDecay:
    """SIM-16 deterministic multi-level book-sweep step fn."""

    def __init__(self, config: OrderBookDecayConfig | None = None) -> None:
        self._config = config or OrderBookDecayConfig()

    @property
    def config(self) -> OrderBookDecayConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        cfg = self._config

        _ = _require_positive_float(meta, "reference_price")  # validate only
        size_usd = _require_positive_float(meta, "order_size_usd")
        num_levels = _require_positive_int(meta, "num_levels")
        if num_levels > cfg.max_levels:
            raise ValueError(
                f"meta['num_levels'] {num_levels} exceeds "
                f"max_levels {cfg.max_levels}"
            )
        spacing_bps = _require_bounded_float(meta, "level_spacing_bps", 0.1, 100.0)
        depth_per_level = _require_positive_float(meta, "level_depth_usd")
        decay_rate = _require_bounded_float(meta, "decay_rate", 0.0, 10.0)
        elapsed = _require_bounded_float(meta, "elapsed_seconds", 0.0, 3600.0)
        side = _require_side(meta)

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        remaining = size_usd
        cost = 0.0
        levels_touched = 0
        for i in range(num_levels):
            offset_frac = (i + 1) * spacing_bps / 10_000.0
            rate_i = decay_rate * (1.0 - i / num_levels)
            decay_factor = math.exp(-rate_i * elapsed)
            jitter = 1.0 + 0.05 * rng.uniform(-1.0, 1.0)
            depth_alive = max(0.0, depth_per_level * decay_factor * jitter)
            eaten = min(remaining, depth_alive)
            if eaten > 0.0:
                cost += eaten * offset_frac
                remaining -= eaten
                levels_touched += 1
            if remaining <= cfg.residual_epsilon_usd:
                remaining = 0.0
                break

        rule_fired = (
            "fully_swept"
            if remaining <= cfg.residual_epsilon_usd
            else "book_too_thin"
        )

        # Sign convention: a market sweep always pays the offset (buy
        # pays the ask side; sell receives less than mid). Both yield
        # negative pnl. Side is recorded only on rule_fired.
        if side == _BUY:
            rule_fired = f"buy_{rule_fired}"
        else:
            rule_fired = f"sell_{rule_fired}"

        pnl = -cost
        drawdown = cost

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=drawdown,
            fills_count=levels_touched,
            rule_fired=rule_fired,
        )


__all__ = ["OrderBookDecay", "OrderBookDecayConfig"]
