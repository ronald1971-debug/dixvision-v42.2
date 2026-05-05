"""SIM-15 slippage_walk — multi-leg cumulative slippage step fn.

Models the failure mode where a sliced execution walks adversely
across legs: each leg's actual fill price perturbs the next leg's
reference, so the slippage compounds rather than averaging out.
This is distinct from SIM-13 latency_jitter (single fill, Gaussian
slippage) — slippage_walk is multi-leg with autocorrelated drift.

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): the price the strategy intends to
  pay (buy) / receive (sell) per unit.
* ``order_size_usd`` (float, > 0): total notional being filled,
  split equally across ``num_legs`` legs.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).
* ``num_legs`` (int, > 0): number of legs the order is sliced into.
* ``per_leg_drift_mean`` (float, in [-0.1, 0.1]): expected per-leg
  signed price drift as a fraction of the previous leg's price.
  Positive means the price walks up (adverse to buyers,
  favourable to sellers).
* ``per_leg_drift_std`` (float, in [0, 0.5]): standard deviation of
  per-leg drift as a fraction.

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. Walk the price across ``num_legs`` legs:
   ``price_i = price_{i-1} * (1 + rng.gauss(mean, std))``.
3. For each leg, accumulate
   ``delta_usd = leg_size_usd * (price_i - entry) / entry``.
4. Buy fill: ``pnl = -sum(delta_usd)``;
   Sell fill: ``pnl = sum(delta_usd)``.
5. ``terminal_drawdown_usd = max(0, -pnl)``.
6. ``fills_count = num_legs``.
7. ``rule_fired = "buy_walk" | "sell_walk"``.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.
NaN-safe (validators use `not low <= v <= high` and `not v > 0.0`
patterns; PR #234 / PR #261 lesson applied).

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.8 in the canonical-rebuild walk).
* manifest.md §549 (simulation/ tree).
* full_feature_spec §624 (SIM-XX module list).
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class SlippageWalkConfig:
    """Versioned configuration for SIM-15 slippage_walk.

    Attributes:
        max_legs: Hard cap on ``num_legs`` to keep the loop bounded.
            Default 10_000.
    """

    max_legs: int = 10_000

    def __post_init__(self) -> None:
        if not 0 < self.max_legs <= 1_000_000:
            raise ValueError(
                "SlippageWalkConfig.max_legs must be in (0, 1_000_000], "
                f"got {self.max_legs!r}"
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
    # `math.isfinite` additionally rejects +inf, which `> 0.0` lets through
    # and which silently produces NaN downstream (e.g. `inf - inf` in
    # geometric walks). Devin Review BUG_pr-review-job-318da2deb_0001.
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


class SlippageWalk:
    """SIM-15 deterministic multi-leg cumulative slippage step fn."""

    def __init__(self, config: SlippageWalkConfig | None = None) -> None:
        self._config = config or SlippageWalkConfig()

    @property
    def config(self) -> SlippageWalkConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        cfg = self._config

        entry = _require_positive_float(meta, "entry_price")
        size_usd = _require_positive_float(meta, "order_size_usd")
        num_legs = _require_positive_int(meta, "num_legs")
        if num_legs > cfg.max_legs:
            raise ValueError(
                f"meta['num_legs'] {num_legs} exceeds "
                f"max_legs {cfg.max_legs}"
            )
        drift_mean = _require_bounded_float(
            meta, "per_leg_drift_mean", -0.1, 0.1
        )
        drift_std = _require_bounded_float(
            meta, "per_leg_drift_std", 0.0, 0.5
        )
        side = _require_side(meta)

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        leg_size_usd = size_usd / num_legs
        price = entry
        delta_sum_usd = 0.0
        for _ in range(num_legs):
            drift = rng.gauss(drift_mean, drift_std)
            price = price * (1.0 + drift)
            # Per-leg notional displacement from the intended entry.
            delta_sum_usd += leg_size_usd * (price - entry) / entry

        # Sign convention:
        #   delta_sum > 0 means price walked up: bad for buy, good for sell.
        if side == _BUY:
            pnl = -delta_sum_usd
            rule_fired = "buy_walk"
        else:
            pnl = delta_sum_usd
            rule_fired = "sell_walk"

        drawdown = max(0.0, -pnl)

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=drawdown,
            fills_count=num_legs,
            rule_fired=rule_fired,
        )


__all__ = ["SlippageWalk", "SlippageWalkConfig"]
