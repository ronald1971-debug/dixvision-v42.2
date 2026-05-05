"""SIM-11 liquidity_decay — depth-evaporation step function.

Models a TWAP-style execution path where the order book's tradable
depth evaporates between slices — typical behaviour during news
shocks or hostile-market-maker withdrawal phases. The simulator
slices a notional ``order_size_usd`` into ``num_slices`` equal
chunks and walks them through a decaying depth profile, summing
the per-slice slippage cost.

Inputs (read from ``RealityScenario.meta``):

* ``reference_price`` (float, > 0): mid-market price at slice 0.
* ``order_size_usd`` (float, > 0): total notional being executed.
* ``initial_depth_usd`` (float, > 0): aggregate touch depth at
  slice 0.
* ``decay_rate`` (float, in [0, 1)): per-slice geometric decay of
  depth. Depth at slice ``t`` is ``initial_depth * (1 -
  decay_rate)^t`` before jitter.
* ``num_slices`` (int, in [1, max_slices]): number of TWAP slices.
* ``side`` (str, one of ``"buy"`` / ``"sell"``).

The deterministic step:

1. Derive a stable :class:`random.Random` from
   ``(seed, scenario.scenario_id)``.
2. ``slice_size = order_size_usd / num_slices``.
3. For each slice ``t`` in ``[0, num_slices)``:

   a. ``depth_t = initial_depth * (1 - decay_rate)^t``.
   b. Apply seeded jitter:
      ``depth_t *= (1 + jitter * depth_jitter)``;
      jitter clamped so depth never drops below ``min_depth_usd``.
   c. ``slippage_t = clamp(slice_size / depth_t, [0, 1])``.
   d. Accumulate ``cost_t = slice_size * slippage_t``.
4. ``cost_usd = sum(cost_t)``; ``pnl = -cost_usd``;
   ``terminal_drawdown_usd = cost_usd``.
5. ``fills_count = num_slices`` (one fill per slice).
6. ``rule_fired`` = ``"buy_decay"`` or ``"sell_decay"``.

Pure (INV-15 / INV-08): no clock, no PRNG outside the seeded
:class:`random.Random` instance, no IO, no engine cross-imports.

Refs:

* dixvision_executive_summary.md — "16 SIM modules"
  (this closes drift item H4.4 in the canonical-rebuild walk).
* execution_engine/strategic/almgren_chriss.py — the production
  scheduler whose worst-case cost path this simulator stresses.
* manifest.md §549 (simulation/ tree).
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

from core.contracts.simulation import RealityOutcome, RealityScenario


@dataclasses.dataclass(frozen=True, slots=True)
class LiquidityDecayConfig:
    """Versioned configuration for SIM-11 liquidity_decay.

    Attributes:
        depth_jitter: Symmetric jitter applied to per-slice depth to
            model intra-slice variance. Default 0.15.
        min_depth_usd: Hard floor on per-slice depth after decay +
            jitter. Prevents division by ~0 when ``decay_rate`` and
            jitter combine adversarially. Default $100.
        max_slices: Upper bound on ``num_slices`` to keep the
            simulation O(num_slices) bounded. Default 1000.
    """

    depth_jitter: float = 0.15
    min_depth_usd: float = 100.0
    max_slices: int = 1000

    def __post_init__(self) -> None:
        if not 0.0 <= self.depth_jitter <= 1.0:
            raise ValueError(
                "LiquidityDecayConfig.depth_jitter must be in [0, 1], "
                f"got {self.depth_jitter!r}"
            )
        if not self.min_depth_usd > 0.0:
            raise ValueError(
                "LiquidityDecayConfig.min_depth_usd must be > 0, "
                f"got {self.min_depth_usd!r}"
            )
        if not 1 <= self.max_slices <= 100_000:
            raise ValueError(
                "LiquidityDecayConfig.max_slices must be in [1, 100000], "
                f"got {self.max_slices!r}"
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
    if not v > 0.0:
        raise ValueError(f"meta[{key!r}] must be > 0, got {v!r}")
    return v


def _require_unit_interval(
    meta: dict[str, Any], key: str, *, exclusive_upper: bool
) -> float:
    if key not in meta:
        raise ValueError(f"RealityScenario.meta missing required key {key!r}")
    raw = meta[key]
    try:
        v = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta[{key!r}] must be numeric, got {raw!r}") from exc
    upper_ok = v < 1.0 if exclusive_upper else v <= 1.0
    if not (0.0 <= v and upper_ok):
        bound = "[0, 1)" if exclusive_upper else "[0, 1]"
        raise ValueError(f"meta[{key!r}] must be in {bound}, got {v!r}")
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


def _require_num_slices(meta: dict[str, Any], max_slices: int) -> int:
    if "num_slices" not in meta:
        raise ValueError(
            "RealityScenario.meta missing required key 'num_slices'"
        )
    raw = meta["num_slices"]
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(
            f"meta['num_slices'] must be an int, got {raw!r}"
        )
    if not 1 <= raw <= max_slices:
        raise ValueError(
            f"meta['num_slices'] must be in [1, {max_slices}], got {raw!r}"
        )
    return raw


class LiquidityDecay:
    """SIM-11 deterministic depth-evaporation step function."""

    def __init__(self, config: LiquidityDecayConfig | None = None) -> None:
        self._config = config or LiquidityDecayConfig()

    @property
    def config(self) -> LiquidityDecayConfig:
        return self._config

    def step(self, seed: int, scenario: RealityScenario) -> RealityOutcome:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed!r}")

        meta = dict(scenario.meta)
        cfg = self._config

        reference = _require_positive_float(meta, "reference_price")
        size_usd = _require_positive_float(meta, "order_size_usd")
        initial_depth = _require_positive_float(meta, "initial_depth_usd")
        decay_rate = _require_unit_interval(
            meta, "decay_rate", exclusive_upper=True
        )
        num_slices = _require_num_slices(meta, cfg.max_slices)
        side = _require_side(meta)

        # reference is exposed for downstream consumers; pin it via
        # an INV-29 invariant check.
        assert reference > 0.0  # noqa: S101

        rng = random.Random(f"{seed}:{scenario.scenario_id}")

        slice_size = size_usd / num_slices
        retention = 1.0 - decay_rate
        depth = initial_depth
        cost_usd = 0.0
        for _ in range(num_slices):
            jitter = (rng.random() - 0.5) * 2.0 * cfg.depth_jitter
            jittered_depth = max(cfg.min_depth_usd, depth * (1.0 + jitter))
            slippage = min(1.0, slice_size / jittered_depth)
            cost_usd += slice_size * slippage
            depth *= retention

        rule_fired = "buy_decay" if side == _BUY else "sell_decay"

        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=-cost_usd,
            terminal_drawdown_usd=cost_usd,
            fills_count=num_slices,
            rule_fired=rule_fired,
        )


__all__ = ["LiquidityDecay", "LiquidityDecayConfig"]
