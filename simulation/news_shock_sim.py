"""SIM-18 news_shock_sim — discrete news-event shock step function.

Models the failure mode where a discrete news event fires during
an active execution window: there is a small per-step probability
that the wire prints, and when it does the price jumps by a
signed shock magnitude (direction drawn from a per-scenario bias)
and then decays through aftershocks of exponentially-falling
volatility for the remainder of the walk.

Distinct from earlier SIM modules:

* **vs SIM-09 ``flash_crash_synth``** — flash_crash is a guaranteed
  one-shot crash with deterministic recovery; this one is a
  *probabilistic* shock with a stochastic firing time and may
  not fire at all.
* **vs SIM-17 ``regime_switch_sim``** — regime_switch flips
  drift/std parameters; this one injects a *discrete jump* on
  top of the per-step diffusion.
* **Unique to SIM-18:** records ``latency_to_shock_steps`` in the
  outcome via ``fills_count`` semantic overload, and emits
  ``rule_fired ∈ {no_shock, buy_shock, sell_shock}`` so an
  auditor can distinguish "shock fired and which direction"
  from "no event during the window".

Inputs (read from ``RealityScenario.meta``):

* ``entry_price`` (float, > 0): price at step 0.
* ``order_size_usd`` (float, > 0): notional held through the walk.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).
* ``num_steps`` (int, in [1, ``max_steps``]): walk length.
* ``shock_probability_per_step`` (float, in [0, 1]): per-step
  probability of the news wire printing (memoryless geometric).
* ``shock_magnitude_bps`` (float, in [0, 10_000]): magnitude of
  the price jump as a fraction of the previous price, in basis
  points. 100 bps = 1%.
* ``shock_bullish_probability`` (float, in [0, 1]): probability
  the shock direction is up vs down. 0.5 = unbiased.
* ``baseline_drift`` (float, in [-0.005, 0.005]): per-step drift
  before any shock.
* ``baseline_std`` (float, in [0, 0.1]): per-step std before any
  shock.
* ``aftershock_decay`` (float, in [0, 5.0]): exponential decay
  rate for post-shock volatility. After the shock, std at step
  ``s`` is ``shock_magnitude * exp(-decay * (s - shock_step))``.

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. Walk:

   For each step ``i`` in ``range(num_steps)``:

   a. If shock has not fired and
      ``rng.random() < shock_probability_per_step``: fire it.
      Direction = ``+1`` if ``rng.random() < bullish_probability``
      else ``-1``. Apply ``price *= 1 + direction * mag``.
   b. Compute std for this step: ``baseline_std`` if no shock or
      shock just fired this step; otherwise
      ``shock_mag * exp(-decay * (i - shock_step))``.
   c. ``price *= 1 + rng.gauss(baseline_drift, std)``.
3. ``delta_usd = order_size_usd * (price - entry) / entry``.
4. Buy fill: ``pnl = delta_usd``; Sell fill: ``pnl = -delta_usd``.
5. ``rule_fired``:

   * ``"no_shock"`` if the wire did not print during the window.
   * ``"buy_shock"`` if the printed shock direction was up.
   * ``"sell_shock"`` if down.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.
NaN-safe + +inf-safe (PR #263 review pattern applied from start).

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.11 in the canonical-rebuild walk).
* manifest.md §549 (simulation/ tree).
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class NewsShockSimConfig:
    """Versioned configuration for SIM-18 news_shock_sim.

    Attributes:
        max_steps: Hard cap on ``num_steps`` to keep the loop
            bounded. Default 10_000.
    """

    max_steps: int = 10_000

    def __post_init__(self) -> None:
        if not 0 < self.max_steps <= 1_000_000:
            raise ValueError(
                "NewsShockSimConfig.max_steps must be in "
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


class NewsShockSim:
    """SIM-18 deterministic discrete news-event shock step fn."""

    def __init__(self, config: NewsShockSimConfig | None = None) -> None:
        self._config = config or NewsShockSimConfig()

    @property
    def config(self) -> NewsShockSimConfig:
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
        shock_p = _require_bounded_float(
            meta, "shock_probability_per_step", 0.0, 1.0
        )
        shock_mag_bps = _require_bounded_float(
            meta, "shock_magnitude_bps", 0.0, 10_000.0
        )
        bullish_p = _require_bounded_float(
            meta, "shock_bullish_probability", 0.0, 1.0
        )
        baseline_drift = _require_bounded_float(
            meta, "baseline_drift", -0.005, 0.005
        )
        baseline_std = _require_bounded_float(
            meta, "baseline_std", 0.0, 0.1
        )
        aftershock_decay = _require_bounded_float(
            meta, "aftershock_decay", 0.0, 5.0
        )
        side = _require_side(meta)

        rng = random.Random(f"{seed}:{scenario.scenario_id}")
        price = entry
        shock_fired = False
        shock_step = -1
        shock_direction = 0
        shock_mag_frac = shock_mag_bps / 10_000.0

        for i in range(num_steps):
            if not shock_fired and rng.random() < shock_p:
                shock_fired = True
                shock_step = i
                shock_direction = 1 if rng.random() < bullish_p else -1
                price = price * (1.0 + shock_direction * shock_mag_frac)

            if shock_fired and i > shock_step:
                # Aftershock decay applied to volatility.
                std = shock_mag_frac * math.exp(
                    -aftershock_decay * (i - shock_step)
                )
            else:
                std = baseline_std
            price = price * (1.0 + rng.gauss(baseline_drift, std))

        delta_usd = size_usd * (price - entry) / entry
        if side == _BUY:
            pnl = delta_usd
        else:
            pnl = -delta_usd
        drawdown = max(0.0, -pnl)

        if not shock_fired:
            rule_fired = "no_shock"
        elif shock_direction > 0:
            rule_fired = "buy_shock"
        else:
            rule_fired = "sell_shock"

        # fills_count carries latency-to-shock semantic: the step
        # index at which the wire printed (0-based), or num_steps
        # if no shock fired during the window. This lets an auditor
        # answer "how long did the strategy live before the event"
        # from the outcome alone.
        latency_to_shock = shock_step if shock_fired else num_steps

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=drawdown,
            fills_count=latency_to_shock,
            rule_fired=rule_fired,
        )


__all__ = ["NewsShockSim", "NewsShockSimConfig"]
