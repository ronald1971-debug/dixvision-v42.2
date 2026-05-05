"""SIM-20 fill_starvation — limit-order queue-position starvation step fn.

Models the failure mode where a passive limit order cannot get
filled because queue priority does not advance fast enough: each
step the maker order has a small probability of getting a
partial fill, while the price walks away. The terminal state may
be partial-filled or fully starved; either way the unfilled
notional is exposed to opportunity cost.

Distinct from earlier SIM modules:

* **vs SIM-14 ``partial_fill_chaos``** — partial_fill_chaos is a
  taker walk where chasing prices generates partial fills; this
  one is a *maker queue* model where waiting may yield no fills
  at all.
* **vs SIM-13 ``latency_jitter``** — latency_jitter assumes a
  single fill at a delayed timestamp; this one tracks per-step
  fractional fills against a walking mid-price.
* **Unique to SIM-20:** emits ``rule_fired ∈ {full_fill,
  partial_fill, starved}`` so an auditor can distinguish a
  successful passive entry from a chronic queue-position failure
  mode.

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): mid-price at step 0 (also the
  resting limit-order price).
* ``order_size_usd`` (float, > 0): notional the order is trying
  to fill at the limit price.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).
* ``num_steps`` (int, in [1, ``max_steps``]): walk length.
* ``per_step_fill_probability`` (float, in [0, 1]): per-step
  probability of a partial fill arriving.
* ``per_step_fill_fraction`` (float, in [0, 1]): when a fill
  arrives, the fraction of *remaining* notional filled this step.
* ``per_step_drift`` (float, in [-0.005, 0.005]): per-step drift
  of the underlying mid.
* ``per_step_std`` (float, in [0, 0.05]): per-step volatility.

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. Walk:

   For each step ``i`` in ``range(num_steps)``:

   a. ``mid_price *= 1 + rng.gauss(drift, std)``.
   b. If ``rng.random() < fill_p`` and unfilled > 0:
      ``filled_this_step = unfilled * fill_fraction``;
      record fill at ``entry_price`` (limit-order price);
      ``filled += filled_this_step``;
      ``unfilled -= filled_this_step``;
      ``fills_count += 1``.
3. Filled-portion pnl: a buy-fill at ``entry_price`` becomes
   ``filled_usd * (mid - entry) / entry``; a sell-fill is the
   negative of that.
4. ``terminal_drawdown_usd`` = ``order_size_usd - filled_usd``
   (the unfilled notional is the *opportunity loss*).
5. ``rule_fired``:

   * ``"full_fill"`` if filled_fraction ≥ 0.99.
   * ``"partial_fill"`` if 0.01 ≤ filled_fraction < 0.99.
   * ``"starved"`` if filled_fraction < 0.01.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.
NaN-safe + +inf-safe (PR #263 review pattern applied from start).

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.13 in the canonical-rebuild walk).
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class FillStarvationConfig:
    """Versioned configuration for SIM-20 fill_starvation.

    Attributes:
        max_steps: Hard cap on ``num_steps`` to keep the loop
            bounded. Default 10_000.
    """

    max_steps: int = 10_000

    def __post_init__(self) -> None:
        if not 0 < self.max_steps <= 1_000_000:
            raise ValueError(
                "FillStarvationConfig.max_steps must be in "
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


class FillStarvation:
    """SIM-20 deterministic limit-order queue-starvation step fn."""

    def __init__(self, config: FillStarvationConfig | None = None) -> None:
        self._config = config or FillStarvationConfig()

    @property
    def config(self) -> FillStarvationConfig:
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
        fill_p = _require_bounded_float(
            meta, "per_step_fill_probability", 0.0, 1.0
        )
        fill_fraction = _require_bounded_float(
            meta, "per_step_fill_fraction", 0.0, 1.0
        )
        drift = _require_bounded_float(meta, "per_step_drift", -0.005, 0.005)
        std = _require_bounded_float(meta, "per_step_std", 0.0, 0.05)
        side = _require_side(meta)

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        mid = entry
        filled_usd = 0.0
        unfilled_usd = size_usd
        fills_count = 0

        for _ in range(num_steps):
            mid = mid * (1.0 + rng.gauss(drift, std))
            if rng.random() < fill_p and unfilled_usd > 0.0:
                filled_this_step = unfilled_usd * fill_fraction
                # Skip zero-notional fills (per_step_fill_fraction=0)
                # so fills_count stays consistent with filled_usd.
                if filled_this_step > 0.0:
                    filled_usd += filled_this_step
                    unfilled_usd -= filled_this_step
                    fills_count += 1

        # Reject overflow / non-finite mids — happens for valid
        # inputs only when (drift, std, num_steps) compound past
        # float64 range under a custom max_steps. Rather than
        # silently propagate NaN/inf into ``pnl_usd``, fail fast.
        if not math.isfinite(mid):
            raise ValueError(
                "fill_starvation walk overflowed to non-finite mid; "
                "tighten num_steps or per_step_drift / per_step_std"
            )

        # PnL: only the filled portion participates in the price
        # move; entry was at the limit price ``entry``, mark to
        # the terminal mid.
        delta_per_unit = (mid - entry) / entry
        if side == _BUY:
            pnl = filled_usd * delta_per_unit
        else:
            pnl = -filled_usd * delta_per_unit

        # Opportunity loss = unfilled notional. This is reported in
        # ``terminal_drawdown_usd`` because in the maker-queue
        # model "drawdown" means failing to deploy capital.
        terminal_drawdown = unfilled_usd

        filled_fraction = filled_usd / size_usd
        if filled_fraction >= 0.99:
            rule_fired = "full_fill"
        elif filled_fraction >= 0.01:
            rule_fired = "partial_fill"
        else:
            rule_fired = "starved"

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=terminal_drawdown,
            fills_count=fills_count,
            rule_fired=rule_fired,
        )


__all__ = ["FillStarvation", "FillStarvationConfig"]
