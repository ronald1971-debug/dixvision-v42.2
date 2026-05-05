"""SIM-21 fee_inversion — cumulative-fee drag step function.

Models the failure mode where a *gross-profitable* trade ends up
*net-unprofitable* because round-trip taker fees, exit slippage,
and per-step funding accrue to more than the price gain. This is
the canonical hidden-cost failure mode that backtests built only
on close-to-close price diffs miss.

Distinct from earlier SIM modules:

* **vs SIM-15 ``slippage_walk``** — slippage_walk models *price
  walk + slippage* on a single leg; this one folds in **funding**
  accrual and **two-sided** taker fees so the drag accumulates
  over the holding window.
* **vs SIM-20 ``fill_starvation``** — fill_starvation's
  ``terminal_drawdown_usd`` encodes *unfilled notional*
  (opportunity loss); here it encodes the *fee/funding burn*
  (the exact USD that gross gain had to clear before the trade
  was net positive).
* **Unique to SIM-21:** emits ``rule_fired ∈ {profitable,
  breakeven, inverted, straight_loss}`` so an auditor can
  distinguish the four economic outcomes:

  - ``profitable``:    gross > 0  *and*  net  > breakeven_band
  - ``breakeven``:     ``|net| <= breakeven_band``
  - ``inverted``:      gross > 0  *and*  net < -breakeven_band
                       (the unique fee-drag failure)
  - ``straight_loss``: gross <= 0 *and*  net < -breakeven_band

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): price at step 0.
* ``order_size_usd`` (float, > 0): notional held through the walk.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).
* ``num_steps`` (int, in [1, ``max_steps``]): walk length. One
  funding period per step.
* ``per_step_drift`` (float, in [-0.005, 0.005]): per-step drift.
* ``per_step_std`` (float, in [0, 0.1]): per-step std.
* ``taker_fee_bps`` (float, in [0, 200]): one-sided taker fee in
  basis points. Charged on both entry and exit notional.
* ``funding_rate_bps_per_step`` (float, in [-100, 100]): per-step
  funding charged to the long, paid to the short. Positive value
  ⇒ longs pay; negative value ⇒ shorts pay.
* ``exit_slippage_bps`` (float, in [0, 500]): one-sided exit
  slippage in basis points. Charged on exit notional only.
* ``breakeven_band_bps`` (float, in [0, 100]): symmetric tolerance
  band, in basis points of ``order_size_usd``, inside which the
  trade is classified as ``breakeven``.

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. Walk: for each step ``i`` in ``range(num_steps)``,
   ``price *= 1 + rng.gauss(per_step_drift, per_step_std)``.
3. Compute economics:

   * ``gross_pnl = ±size * (final - entry) / entry``
     (sign by side).
   * ``entry_fee = exit_fee = size * taker_fee_bps / 10_000``.
   * ``exit_slip = size * exit_slippage_bps / 10_000``.
   * ``funding = size * funding_rate_bps_per_step / 10_000 *
                 num_steps`` charged to longs, paid to shorts.
   * ``net_pnl = gross_pnl - entry_fee - exit_fee - exit_slip
                 - sign_funding * funding``  where
     ``sign_funding = +1`` for buy, ``-1`` for sell.

4. ``terminal_drawdown_usd = max(0, gross_pnl - net_pnl)`` —
   this is the **fee/funding burn** specifically, not a
   price-walk drawdown.
5. ``rule_fired`` per the four-way classification above.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.
NaN-safe + +inf-safe (PR #263 review pattern).

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.14 in the canonical-rebuild walk).
* manifest.md §549 (simulation/ tree).
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class FeeInversionConfig:
    """Versioned configuration for SIM-21 fee_inversion.

    Attributes:
        max_steps: Hard cap on ``num_steps`` to keep the loop
            bounded. Default 10_000.
    """

    max_steps: int = 10_000

    def __post_init__(self) -> None:
        if not 0 < self.max_steps <= 1_000_000:
            raise ValueError(
                "FeeInversionConfig.max_steps must be in "
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
    if not math.isfinite(v):
        raise ValueError(f"meta[{key!r}] must be finite, got {v!r}")
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


class FeeInversion:
    """SIM-21 deterministic cumulative-fee-drag step fn."""

    def __init__(self, config: FeeInversionConfig | None = None) -> None:
        self._config = config or FeeInversionConfig()

    @property
    def config(self) -> FeeInversionConfig:
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
        drift = _require_bounded_float(meta, "per_step_drift", -0.005, 0.005)
        std = _require_bounded_float(meta, "per_step_std", 0.0, 0.1)
        taker_fee_bps = _require_bounded_float(
            meta, "taker_fee_bps", 0.0, 200.0
        )
        funding_bps_per_step = _require_bounded_float(
            meta, "funding_rate_bps_per_step", -100.0, 100.0
        )
        exit_slip_bps = _require_bounded_float(
            meta, "exit_slippage_bps", 0.0, 500.0
        )
        breakeven_band_bps = _require_bounded_float(
            meta, "breakeven_band_bps", 0.0, 100.0
        )
        side = _require_side(meta)

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        price = entry
        for _ in range(num_steps):
            price = price * (1.0 + rng.gauss(drift, std))

        # Reject overflow / non-finite prices — happens for valid
        # inputs only when (drift, std, num_steps) compound past
        # float64 range under a custom max_steps. Rather than
        # silently propagate NaN/inf into ``pnl_usd``, fail fast.
        # (Same guard as SIM-20 fill_starvation per Devin Review.)
        if not math.isfinite(price):
            raise ValueError(
                "fee_inversion walk overflowed to non-finite price; "
                "tighten num_steps or per_step_drift / per_step_std"
            )

        delta_per_unit = (price - entry) / entry
        if side == _BUY:
            gross_pnl = size_usd * delta_per_unit
            funding_sign = 1.0
        else:
            gross_pnl = -size_usd * delta_per_unit
            funding_sign = -1.0

        entry_fee = size_usd * taker_fee_bps / 10_000.0
        exit_fee = size_usd * taker_fee_bps / 10_000.0
        exit_slip = size_usd * exit_slip_bps / 10_000.0
        funding_cost = (
            size_usd * funding_bps_per_step / 10_000.0 * num_steps
        )

        net_pnl = (
            gross_pnl
            - entry_fee
            - exit_fee
            - exit_slip
            - funding_sign * funding_cost
        )

        # Burn = the USD of cost drag the gross gain had to clear.
        # Always non-negative: the cost components are all
        # positive except funding, which can flip sign by side,
        # but ``funding_sign * funding_cost`` is always the
        # *charge to this side*. We bound by 0 to keep the field
        # consistent with the contract's >= 0 invariant.
        burn = max(0.0, gross_pnl - net_pnl)

        breakeven_band = size_usd * breakeven_band_bps / 10_000.0
        if abs(net_pnl) <= breakeven_band:
            rule_fired = "breakeven"
        elif net_pnl > breakeven_band:
            rule_fired = "profitable"
        elif gross_pnl > 0.0:
            rule_fired = "inverted"
        else:
            rule_fired = "straight_loss"

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=net_pnl,
            terminal_drawdown_usd=burn,
            fills_count=num_steps,
            rule_fired=rule_fired,
        )


__all__ = ["FeeInversion", "FeeInversionConfig"]
